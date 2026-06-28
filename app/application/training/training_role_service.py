"""销售训练角色生成纯逻辑服务。

这个模块承接“AI 客户角色生成”前后的本地逻辑：
- 组装向量检索 query；
- 渲染角色、场景润色、补充问答提示词；
- 规整 LLM 返回的补充问答题；
- 在 LLM 不可用时生成稳定兜底结果。

它不访问数据库、不调用向量库、不直接调用大模型，核心编排服务可以把它当成
角色生成子系统的本地策略服务来使用。
"""

from __future__ import annotations

import json
from typing import Any

from app.application.training_support.schemas import (
    RoleGenerateRequest,
    ScenarioPolishRequest,
    SupplementQuestion,
    SupplementQuestionOption,
)
from core.utils.prompt_manager import prompt_manager


class TrainingRoleService:
    """销售训练角色生成纯逻辑服务。"""

    @staticmethod
    def build_role_query(request: RoleGenerateRequest) -> str:
        """构造角色生成前的向量检索查询文本。"""

        return "\n".join(
            [
                request.profile_type,
                request.scenario_description,
                request.extra_details,
                " ".join(request.trainee.weakness_tags),
                json.dumps(request.selected_fields, ensure_ascii=False),
            ]
        )

    @staticmethod
    def role_prompt(request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> str:
        """构造 AI 客户角色生成提示词。"""

        return prompt_manager.render(
            "training.role_generation.user",
            trainee_json=request.trainee.model_dump_json(indent=2),
            selected_fields_json=json.dumps(request.selected_fields, ensure_ascii=False, indent=2),
            scenario_description=request.scenario_description,
            extra_details=request.extra_details,
            evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def scenario_polish_prompt(request: ScenarioPolishRequest) -> str:
        """构造场景描述润色提示词。"""

        return prompt_manager.render(
            "training.scenario_polish.user",
            profile_type=request.profile_type,
            selected_fields_json=json.dumps(request.selected_fields, ensure_ascii=False, indent=2),
            scenario_description=request.scenario_description,
            extra_details=request.extra_details,
        )

    @staticmethod
    def supplement_questions_prompt(request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> str:
        """构造补充问答生成提示词。"""

        return prompt_manager.render(
            "training.supplement_questions.user",
            trainee_json=request.trainee.model_dump_json(indent=2),
            selected_fields_json=json.dumps(request.selected_fields, ensure_ascii=False, indent=2),
            scenario_description=request.scenario_description,
            extra_details=request.extra_details,
            evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
        )

    @classmethod
    def normalize_supplement_questions(cls, raw_questions: Any, request: RoleGenerateRequest) -> list[SupplementQuestion]:
        """把 LLM 输出规整成前端稳定可渲染的 1-5 道题。"""

        questions: list[SupplementQuestion] = []
        source = raw_questions if isinstance(raw_questions, list) else []
        fallback = cls.fallback_supplement_questions(request)
        if not source:
            source = fallback

        for index, item in enumerate(source[:5], start=1):
            if not isinstance(item, dict):
                continue
            raw_options = item.get("options") if isinstance(item.get("options"), list) else []
            options: list[SupplementQuestionOption] = []
            for option_index, option in enumerate(raw_options[:4]):
                option_code = chr(ord("A") + option_index)
                if isinstance(option, dict):
                    option_text = str(option.get("option_text") or option.get("text") or "").strip()
                    option_code = str(option.get("option_code") or option_code).strip()[:1].upper() or option_code
                else:
                    option_text = str(option or "").strip()
                if option_text:
                    options.append(SupplementQuestionOption(option_code=option_code, option_text=option_text))

            if len(options) < 4:
                fallback_item = fallback[min(index - 1, len(fallback) - 1)]
                fallback_options = fallback_item["options"]
                for option in fallback_options[len(options):4]:
                    options.append(SupplementQuestionOption(**option))

            question_text = str(item.get("question") or "").strip()
            if not question_text:
                question_text = str(fallback[min(index - 1, len(fallback) - 1)]["question"])

            questions.append(
                SupplementQuestion(
                    question_id=str(item.get("question_id") or f"q{index}"),
                    question_no=int(item.get("question_no") or index),
                    question=question_text,
                    options=options[:4],
                    allow_other=bool(item.get("allow_other", True)),
                    dimension=str(item.get("dimension") or ""),
                )
            )

        if not questions:
            return [SupplementQuestion(**item) for item in fallback]
        return questions[:5]

    @staticmethod
    def fallback_polished_scenario(request: ScenarioPolishRequest) -> str:
        """模型润色失败时的本地兜底文案。

        兜底逻辑只做安全拼接，不虚构业务事实；这样即使 LLM 报错，
        前端也能得到一段可用的场景描述。
        """

        selected_fields = request.selected_fields or {}
        field_text = "；".join(
            f"{key}：{value}"
            for key, value in selected_fields.items()
            if str(value or "").strip()
        )
        parts = [
            f"当前客户画像为{request.profile_type}",
            f"画像信息包括{field_text}" if field_text else "",
            f"原始场景是：{request.scenario_description.strip()}",
            f"补充要求是：{request.extra_details.strip()}" if request.extra_details.strip() else "",
            "学员需要围绕客户真实顾虑展开提问，并用匹配的价值表达推动客户继续沟通。",
        ]
        return "。".join(part.strip("。") for part in parts if part).strip("。") + "。"

    @staticmethod
    def fallback_supplement_questions(request: RoleGenerateRequest) -> list[dict[str, Any]]:
        """补充问题兜底模板，保证模型不可用时流程仍可继续。"""

        fields = request.selected_fields or {}
        profile_name = str(fields.get("画像类型") or request.profile_type or "客户")
        scenario = request.scenario_description.strip() or "当前业务增长方案"
        return [
            {
                "question_id": "q1",
                "question_no": 1,
                "dimension": "决策顾虑",
                "question": f"如果“{profile_name}”确实能解决当前问题，在客户决策前，还有哪些因素最可能让客户犹豫？",
                "options": [
                    {"option_code": "A", "option_text": "先对比几家供应商，确认价格和方案是否更合理"},
                    {"option_code": "B", "option_text": "担心操作复杂，团队学习和迁移成本太高"},
                    {"option_code": "C", "option_text": "需要看到同行案例和真实效果证明"},
                    {"option_code": "D", "option_text": "希望先试用或小范围验证，再决定是否投入"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q2",
                "question_no": 2,
                "dimension": "价值判断",
                "question": "如果客户现在要换一套新工具，除了价格之外，最看重它做到什么程度才算值？",
                "options": [
                    {"option_code": "A", "option_text": "能明显减少重复工作，把时间省出来"},
                    {"option_code": "B", "option_text": "能降低错误率，避免报价、跟进或交付出问题"},
                    {"option_code": "C", "option_text": "能自动记住客户历史偏好，方便长期复购"},
                    {"option_code": "D", "option_text": "稳定、好上手、不添乱，价格合理就好"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q3",
                "question_no": 3,
                "dimension": "业务卡点",
                "question": "客户跟进业务时，哪个环节最容易让他觉得“明明可以更快但就是快不起来”？",
                "options": [
                    {"option_code": "A", "option_text": "每次都要翻聊天记录、邮件或历史报价"},
                    {"option_code": "B", "option_text": "报价格式、汇率、利润核算反复调整"},
                    {"option_code": "C", "option_text": "不同客户不同价格，担心记错或报错"},
                    {"option_code": "D", "option_text": "客户问一句答一句，碎片沟通占用太多时间"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q4",
                "question_no": 4,
                "dimension": "沟通性格",
                "question": "这位客户在沟通里更可能呈现哪种风格？",
                "options": [
                    {"option_code": "A", "option_text": "务实直接，先问价格和投入产出"},
                    {"option_code": "B", "option_text": "谨慎保守，需要反复确认风险"},
                    {"option_code": "C", "option_text": "结果导向，愿意听方案但只认效果"},
                    {"option_code": "D", "option_text": "容易质疑，喜欢追问细节和边界条件"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q5",
                "question_no": 5,
                "dimension": "训练挑战",
                "question": f"结合当前场景“{scenario[:40]}”，你希望 AI 客户重点挑战学员哪一类能力？",
                "options": [
                    {"option_code": "A", "option_text": "需求挖掘，逼学员问出真实痛点"},
                    {"option_code": "B", "option_text": "价格异议，持续追问为什么值得投入"},
                    {"option_code": "C", "option_text": "案例证明，要求学员用证据而不是口号说服"},
                    {"option_code": "D", "option_text": "推进下一步，考察学员能否争取试用或继续沟通"},
                ],
                "allow_other": True,
            },
        ]

    @staticmethod
    def fallback_role(request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> dict:
        """角色生成失败时的本地兜底结果。

        兜底只使用用户选择的画像字段和已召回证据，不凭空扩展业务事实。
        """

        selected_fields = request.selected_fields or {}

        def pick_field(*keys: str, default: str) -> str:
            """从前端选择字段中按多个候选名称取值。"""

            for key in keys:
                value = selected_fields.get(key)
                if value:
                    return str(value).strip()
            return default

        def compact_list(*items: str, min_size: int = 3) -> list[str]:
            """组装列表并去掉空字符串，避免前端展示空行。"""

            result = [str(item).strip() for item in items if str(item or "").strip()]
            while len(result) < min_size:
                result.append("需要在训练对话中进一步确认")
            return result

        industry = pick_field("行业", "客户行业", default="外贸企业")
        customer_type = pick_field("客户类型", "客户画像", default="谨慎型业务负责人")
        personality = pick_field("性格特征", "客户性格", default="务实谨慎，关注风险和投入产出")
        cooperation_stage = pick_field("合作阶段", default="初次接触")
        scenario = request.scenario_description.strip()
        extra_details = request.extra_details.strip()
        pain_source = extra_details or scenario or "客户正在评估新的业务增长方案，但担心成本投入、交付风险和团队执行压力。"
        trainee_weakness = "、".join(request.trainee.weakness_tags) or "需求挖掘和异议处理"
        knowledge_facts = [str(item.get("content") or "")[:160] for item in evidence if item.get("content")]
        if not knowledge_facts:
            knowledge_facts = [
                "训练知识库暂未召回明确事实，需要学员在对话中先确认客户背景。",
                "客户关注成本投入、交付风险和内部推进难度。",
                "学员需要用提问获取更多业务细节，再匹配方案价值。",
            ]

        return {
            "visible_profile": {
                "角色名称": f"{customer_type}客户",
                "性别": "男",
                "年龄": "33",
                "职位": "业务负责人",
                "身份": f"{industry}｜业务负责人",
                "性格特征": personality,
                "角色摘要": f"来自{industry}领域，处于{cooperation_stage}阶段，正在判断方案是否值得继续推进。客户关注实际收益、投入产出和交付风险。",
                "成本控制习惯": compact_list("日常运营严格审核支出", "优先选择高性价比方案", "对长期订阅费用递增较敏感"),
                "业务痛点": compact_list(pain_source[:120], "内部推动需要明确收益依据", "担心方案落地后额外增加团队负担"),
                "潜台词": compact_list("先证明你懂我的业务", "不要只讲功能，要讲对我有什么用", "风险说不清就不会继续推进"),
            },
            "hidden_profile": {
                "真实顾虑": compact_list("担心投入后没有效果", "担心需要新增人力或改变现有流程", "担心内部汇报时缺少可量化依据"),
                "成交触发器": compact_list("同类案例足够具体", "交付路径清晰", "能回答价格与效果的对应关系"),
                "追问策略": compact_list("先追问业务理解", "再追问效果证据", "最后追问落地成本和下一步安排"),
            },
            "role_profile": {
                "职位": "业务负责人",
                "角色简介": f"一位来自{industry}的{customer_type}，处于{cooperation_stage}阶段，正在判断方案是否值得继续推进。",
                "性格特征": personality,
                "成本控制习惯": compact_list("会把价格和效果绑定判断", "会追问是否增加团队执行成本", "倾向先小范围验证"),
                "业务痛点": compact_list(pain_source[:120], "缺少可靠案例支撑内部决策", "担心业务团队执行压力过大"),
                "潜台词": compact_list("你先证明你懂我的业务", "不要只讲功能，要讲对我有什么用", "如果风险说不清，我不会继续推进"),
                "挑战策略": compact_list(f"针对学员短板：{trainee_weakness}，持续追问证据", "对价格价值关系施压", "当回答空泛时要求举例"),
                "异议示例": compact_list("听起来不错，但我怎么判断不是概念包装", "预算不低，你们的价值怎么量化", "我们团队现在很忙，配合成本会不会很高"),
                "不能直接透露": compact_list("真实顾虑不要一次性说完", "不要主动告诉学员评分标准", "隐藏心理只能通过追问逐步体现"),
            },
            "role_confirm_card": {
                "角色名称": f"{customer_type}客户",
                "性别": "男",
                "年龄": "33",
                "身份": f"{industry}｜业务负责人",
                "性格特征": personality,
                "角色摘要": f"客户正在评估新的业务方案，关注{industry}场景下的实际效果、投入产出和交付风险。沟通中会先观察学员是否理解业务，再决定是否继续释放信息。",
                "成本控制习惯": compact_list("日常运营严格审核支出", "优先选择高性价比方案", "对长期订阅费用递增较敏感"),
                "业务痛点": compact_list(pain_source[:120], "内部推动需要明确收益依据", "担心落地成本超出预期"),
                "潜台词": compact_list("争取首年折扣或免费试用机会", "担心操作复杂且不能解决真实痛点", "对比效率提升与成本节省是否成正比"),
            },
        }
