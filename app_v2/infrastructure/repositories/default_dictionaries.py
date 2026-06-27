"""系统默认字典种子数据。"""

from core.rag.profile_dictionaries import PROFILE_DICTIONARY_ITEMS


DEFAULT_DICTIONARY_ITEMS = [
    {
        "dictionary_code": "document_structure",
        "dictionary_name": "文档结构类型",
        "items": [
            ("text", "普通文本型", None, 1, "没有稳定结构的普通文本", {"default": True}),
            ("qa", "问答型", None, 2, "按问题和答案组织的文档结构"),
            ("numbered", "编号条目型", None, 3, "按编号条目组织的文档结构"),
        ],
    },
    {
        "dictionary_code": "split_strategy",
        "dictionary_name": "切分策略",
        "items": [
            ("recursive", "递归通用切分", None, 1, "按分隔符和长度递归切分", {"default": True}),
            ("numbered_qa", "编号问答切分", None, 2, "把编号问答切成 QA 片段"),
            ("outline_qa", "目录问答切分", None, 3, "把 PDF 书签目录中的章节和问题切成 QA 片段"),
            ("numbered_segments", "编号条目切分", None, 4, "把编号条目切成普通片段"),
            ("llm_semantic", "LLM 语义切分", None, 5, "由 LLM 判断语义边界，后端按原文范围截取入库", {"recommendation": True}),
        ],
    },
    {
        "dictionary_code": "model_mode",
        "dictionary_name": "回答模型档位",
        "items": [
            ("high", "高", None, 1, "高质量模型档位", {"quality": "high", "default": True}),
            ("medium", "中", None, 2, "平衡模型档位", {"quality": "medium"}),
            ("low", "低", None, 3, "低延迟模型档位", {"quality": "low", "recommendation": True}),
        ],
    },
    {
        "dictionary_code": "output_mode",
        "dictionary_name": "输出模式",
        "items": [
            ("stream", "流式", None, 1, "边生成边返回", {"mode_kind": "stream", "default": True}),
            ("once", "一次性", None, 2, "生成完成后一次性返回", {"mode_kind": "once"}),
        ],
    },
    {
        "dictionary_code": "document_status",
        "dictionary_name": "知识库文件状态",
        "items": [
            ("uploaded", "已上传", None, 1, "文件已保存但未完成入库", {"tag_type": "info"}),
            ("indexing", "入库中", None, 2, "正在解析、切分和写入向量库", {"tag_type": "warning"}),
            ("indexed", "已索引", None, 3, "已完成向量索引", {"tag_type": "success"}),
            ("failed", "入库失败", None, 4, "入库过程失败", {"tag_type": "danger"}),
            ("deleted", "已删除", None, 5, "文件已标记删除", {"tag_type": "info"}),
        ],
    },
    {
        "dictionary_code": "conversation_status",
        "dictionary_name": "会话状态",
        "items": [
            ("active", "正常", None, 1, "可继续使用的会话"),
            ("deleted", "已删除", None, 2, "已删除的会话"),
        ],
    },
    {
        "dictionary_code": "message_role",
        "dictionary_name": "消息角色",
        "items": [
            ("user", "用户", None, 1, "用户消息"),
            ("assistant", "助手", None, 2, "助手消息"),
            ("system", "系统", None, 3, "系统消息"),
        ],
    },
    {
        "dictionary_code": "content_type",
        "dictionary_name": "内容类型",
        "items": [
            ("text", "文本", None, 1, "普通文本内容"),
            ("qa", "问答片段", None, 2, "问答型知识片段"),
            ("segment", "普通片段", None, 3, "普通知识片段"),
        ],
    },
    {
        "dictionary_code": "service_status",
        "dictionary_name": "服务状态",
        "items": [
            ("ok", "正常", None, 1, "服务可用"),
            ("degraded", "降级", None, 2, "部分依赖不可用"),
            ("unavailable", "不可用", None, 3, "服务或依赖不可用"),
        ],
    },
    {
        "dictionary_code": "knowledge_result_status",
        "dictionary_name": "知识库操作结果",
        "items": [
            ("indexed", "已索引", None, 1, "文件入库成功", {"result_kind": "indexed"}),
            ("duplicate", "重复", None, 2, "存在相同内容文件", {"result_kind": "duplicate"}),
            ("failed", "失败", None, 3, "操作失败", {"result_kind": "failed"}),
            ("ok", "成功", None, 4, "批量操作全部成功", {"result_kind": "ok"}),
            ("partial_failed", "部分失败", None, 5, "批量操作存在失败项", {"result_kind": "partial_failed"}),
        ],
    },
    {
        "dictionary_code": "preview_type",
        "dictionary_name": "预览类型",
        "items": [
            ("text", "TXT 文本", None, 1, "TXT 文件预览"),
            ("pdf_text", "PDF 文本", None, 2, "PDF 提取文本预览"),
            ("unsupported", "不支持", None, 3, "暂不支持预览"),
        ],
    },
    {
        "dictionary_code": "training_source_type",
        "dictionary_name": "销售训练资料来源类型",
        "items": [
            (
                "lms_case",
                "LMS 销售训练案例",
                None,
                1,
                "按客户案例、任务要求、标准答案、隐藏心理、评分标准做专门结构化拆分",
                {
                    "default": True,
                    "implemented": True,
                    "strategy": "LmsCaseIngestStrategy",
                    "collection_name": "sales_training_cases",
                },
            ),
        ],
    },
    {
        "dictionary_code": "training_case_part",
        "dictionary_name": "销售训练切片类型",
        "items": [
            ("case_profile", "客户背景", None, 1, "客户案例、公司背景、合作阶段等信息", {"tag_type": "info"}),
            ("task_requirement", "训练任务", None, 2, "本次训练要求、沟通目标和操作约束", {"tag_type": "primary"}),
            ("standard_answer", "参考话术", None, 3, "标准答案、优秀话术和建议表达", {"tag_type": "success"}),
            ("hidden_psychology", "客户顾虑", None, 4, "客户真实顾虑、隐性心理和潜在异议", {"tag_type": "warning"}),
            ("scoring_rubric", "评分依据", None, 5, "评分标准、命中点和扣分点", {"tag_type": "danger"}),
            ("product_fact", "产品事实", None, 6, "产品功能、参数、服务和交付事实", {"tag_type": "info"}),
            ("faq", "常见问答", None, 7, "常见客户问题和标准答复", {"tag_type": "info"}),
            ("competitor", "竞品信息", None, 8, "竞品对比、优势劣势和替代方案", {"tag_type": "warning"}),
            ("success_case", "成功案例", None, 9, "成交案例、客户证言和效果证明", {"tag_type": "success"}),
            ("glossary", "术语说明", None, 10, "行业术语、缩写和业务概念", {"tag_type": "info"}),
        ],
    },
    {
        "dictionary_code": "training_chunk_usage",
        "dictionary_name": "训练切片模型用途",
        "items": [
            ("visible", "通用知识", None, 1, "角色生成、对话训练和评分都可参考", {"tag_type": "info"}),
            ("hidden", "客户内部顾虑", None, 2, "主要用于 AI 客户扮演和追问，不作为权限展示", {"tag_type": "warning"}),
            ("scoring_only", "评分专用", None, 3, "主要用于训练结束评分，不作为权限展示", {"tag_type": "danger"}),
        ],
    },
    {
        "dictionary_code": "training_batch_status",
        "dictionary_name": "销售训练资料批次状态",
        "items": [
            ("parsing", "解析中", None, 1, "文件已保存，正在解析和切片", {"tag_type": "warning"}),
            ("pending_review", "待确认", None, 2, "切片和质量评估已完成，等待人工确认发布", {"tag_type": "warning"}),
            ("embedding", "发布中", None, 3, "正在从临时向量库发布到正式向量库", {"tag_type": "warning"}),
            ("published", "已发布", None, 4, "训练资料已完成向量入库，可以参与训练检索", {"tag_type": "success"}),
            ("archived", "历史版本", None, 5, "同一资料的新版本已发布，该版本不再参与训练检索，可按需回滚", {"tag_type": "info"}),
            ("parsing_failed", "解析失败", None, 6, "文件解析或临时向量库写入失败，需要查看错误信息", {"tag_type": "danger"}),
            ("publish_failed", "发布失败", None, 7, "从临时向量库发布到正式向量库失败，可重试发布", {"tag_type": "danger"}),
            ("deleted", "已删除", None, 8, "批次已从 MySQL、Qdrant 和 MinIO 全链路删除", {"tag_type": "info"}),
            ("duplicated", "重复复用", None, 9, "上传响应状态，表示文件 MD5 命中已发布批次并复用历史数据", {"tag_type": "info"}),
        ],
    },
] + PROFILE_DICTIONARY_ITEMS

DEPRECATED_DICTIONARY_CODES = {"collection_domain_keyword", "sales_customer_profile_template"}
