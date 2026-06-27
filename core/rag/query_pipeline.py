from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    """一次用户问题的检索前分析结果。

    主 RAG 链路不再使用规则关键词硬匹配做意图识别。
    这个对象只保留为检索、精排和调试接口之间传递结构化信息的兼容数据结构。
    """

    original_query: str  # 用户原始问题
    intents: list[str] = field(default_factory=list)  # 识别出的意图列表
    sub_queries: list[str] = field(default_factory=list)  # 用于多路召回的子查询
    filters: dict[str, list[str]] = field(default_factory=dict)  # Qdrant metadata filter
    keywords: list[str] = field(default_factory=list)  # rerank 阶段使用的命中词
