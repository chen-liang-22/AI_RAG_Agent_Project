from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    """一次用户问题的检索前分析结果。

    这个对象对应设计文档里的 intent_analyze/query_rewrite/filter 输出。
    第一版先使用规则实现，不额外调用大模型：
    - 优点：速度快、成本低、不会影响流式首 token 等待太久。
    - 缺点：表达能力不如 LLM 意图识别，后续可以把 analyze() 替换成模型版本。
    """

    original_query: str  # 用户原始问题
    intents: list[str] = field(default_factory=list)  # 识别出的意图列表
    sub_queries: list[str] = field(default_factory=list)  # 用于多路召回的子查询
    filters: dict[str, list[str]] = field(default_factory=dict)  # Qdrant metadata filter
    keywords: list[str] = field(default_factory=list)  # rerank 阶段使用的命中词


class RuleBasedIntentAnalyzer:
    """规则版多意图识别器。

    当前项目的知识领域比较集中：扫地/扫拖机器人客服。
    因此第一阶段用关键词规则就能覆盖一批常见问题。

    后续升级方向：
    - 用 LLM 输出 JSON 格式的 intents/sub_queries/filters。
    - 把用户历史对话也放进意图判断。
    - 把天气、报告这类工具意图和 RAG 意图区分得更细。
    """

    # 每个意图对应一组触发关键词。
    # 只要用户问题里出现其中任意关键词，就认为命中该意图。
    intent_keywords: dict[str, list[str]] = {
        "purchase": [
            "选购",
            "推荐",
            "买",
            "适合",
            "类型",
            "哪种",
            "小户型",
            "大户型",
            "吸力",
            "导航",
            "地毯",
            "宠物",
        ],
        "troubleshooting": [
            "故障",
            "怎么办",
            "怎么处理",
            "解决",
            "修复",
            "检测",
            "迷路",
            "报错",
            "异常",
            "不工作",
            "找不到",
            "充电",
            "卡住",
            "排除",
        ],
        "maintenance": [
            "维护",
            "保养",
            "清理",
            "清洁",
            "耗材",
            "滤网",
            "主刷",
            "边刷",
            "拖布",
            "多久",
        ],
        "comparison": [
            "对比",
            "区别",
            "哪个更好",
            "比",
            "iRobot",
            "科沃斯",
            "石头",
            "云鲸",
            "追觅",
        ],
        "weather": [
            "天气",
            "下雨",
            "温度",
            "湿度",
            "空气质量",
            "AQI",
        ],
        "report": [
            "报告",
            "使用记录",
            "我的",
            "本月",
            "效率",
            "耗材",
        ],
    }

    # 意图到知识单元类型的映射。
    # 这些值会对应 Qdrant payload 里的 metadata.unit_type。
    intent_unit_types: dict[str, list[str]] = {
        "purchase": ["guide", "faq"],
        "comparison": ["guide", "faq"],
        "troubleshooting": ["troubleshooting", "faq"],
        "maintenance": ["maintenance", "faq"],
    }

    # 意图到业务分类的映射。
    # 这些值会对应 Qdrant payload 里的 metadata.category。
    intent_categories: dict[str, list[str]] = {
        "purchase": ["选购指南", "常见问答"],
        "comparison": ["选购指南", "常见问答"],
        "troubleshooting": ["故障排查", "常见问答"],
        "maintenance": ["维护保养", "常见问答"],
    }

    # 生成子查询时使用的人类可读描述。
    # 子查询不是给用户看的，而是帮助向量召回覆盖更多表达方式。
    intent_query_hints: dict[str, str] = {
        "purchase": "扫地机器人选购建议",
        "comparison": "扫地机器人品牌型号对比",
        "troubleshooting": "扫地机器人故障排查解决方法",
        "maintenance": "扫地机器人维护保养清理方法",
        "weather": "天气对扫地机器人使用的影响",
        "report": "扫地机器人用户使用报告",
    }

    def analyze(self, query: str) -> QueryAnalysis:
        """分析用户问题，输出意图、子查询、过滤条件和关键词。"""

        normalized_query = query.strip()
        matched_intents: list[str] = []
        matched_keywords: list[str] = []

        for intent, keywords in self.intent_keywords.items():
            hit_words = [keyword for keyword in keywords if keyword.lower() in normalized_query.lower()]
            if hit_words:
                matched_intents.append(intent)
                matched_keywords.extend(hit_words)

        if not matched_intents:
            matched_intents.append("general")

        sub_queries = self._build_sub_queries(normalized_query, matched_intents)
        filters = self._build_filters(matched_intents)
        keywords = self._build_keywords(normalized_query, matched_keywords, matched_intents)

        return QueryAnalysis(
            original_query=normalized_query,
            intents=matched_intents,
            sub_queries=sub_queries,
            filters=filters,
            keywords=keywords,
        )

    def _build_sub_queries(self, query: str, intents: list[str]) -> list[str]:
        """基于意图生成多路召回用的子查询。"""

        sub_queries = [query]

        for intent in intents:
            hint = self.intent_query_hints.get(intent)
            if not hint:
                continue

            sub_query = f"{hint}：{query}"
            if sub_query not in sub_queries:
                sub_queries.append(sub_query)

        return sub_queries

    def _build_filters(self, intents: list[str]) -> dict[str, list[str]]:
        """基于意图生成 Qdrant metadata filter。"""

        unit_types: list[str] = []
        categories: list[str] = []

        for intent in intents:
            unit_types.extend(self.intent_unit_types.get(intent, []))
            categories.extend(self.intent_categories.get(intent, []))

        result: dict[str, list[str]] = {}
        unique_unit_types = self._deduplicate(unit_types)
        unique_categories = self._deduplicate(categories)

        if unique_unit_types:
            result["unit_type"] = unique_unit_types

        if unique_categories:
            result["category"] = unique_categories

        return result

    def _build_keywords(self, query: str, matched_keywords: list[str], intents: list[str]) -> list[str]:
        """生成精排阶段使用的关键词。

        这里不做复杂中文分词，只保留：
        - 规则命中的关键词。
        - 意图描述中的核心词。
        - 用户原问题里常见的领域词。
        """

        domain_words = [
            "扫地机器人",
            "扫拖",
            "吸力",
            "导航",
            "地毯",
            "宠物",
            "拖布",
            "滤网",
            "故障",
            "保养",
            "选购",
        ]

        keywords = list(matched_keywords)
        keywords.extend(word for word in domain_words if word in query)

        for intent in intents:
            hint = self.intent_query_hints.get(intent, "")
            keywords.extend(part for part in hint.replace("：", " ").split() if part)

        return self._deduplicate(keywords)

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        """保持顺序去重。"""

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)

        return result
