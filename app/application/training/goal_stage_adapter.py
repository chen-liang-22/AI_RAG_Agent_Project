"""训练目标阶段结构适配器。"""

from __future__ import annotations

from typing import Any


class GoalStageAdapter:
    """把 LLM 或数据库中的阶段 JSON 适配成接口稳定结构。

    这里使用适配器模式：LLM 返回的 JSON 字段类型可能不稳定，例如把条件列表返回成字符串；
    接口层需要稳定的 ``list[str]``，所以在进入 Pydantic DTO 前统一归一化。
    """

    DEFAULT_STAGE_NAME = "开放式需求挖掘"
    DEFAULT_CORE_GOAL = "围绕客户需求推进有效沟通"
    DEFAULT_SUCCESS_CONDITION = "客户愿意继续推进沟通"
    DEFAULT_FAILURE_CONDITION = "客户明确拒绝继续沟通"

    @classmethod
    def normalize_stages(
            cls,
            raw_stages: Any,
            *,
            fallback_stages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """归一化阶段列表，保证每个阶段都能被 GoalStage 接收。"""

        source_stages = raw_stages if isinstance(raw_stages, list) else []
        if not source_stages:
            source_stages = fallback_stages or []
        fallback_by_index = fallback_stages or []
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(source_stages):
            fallback = fallback_by_index[index] if index < len(fallback_by_index) else {}
            normalized.append(cls.normalize_stage(item, index=index, fallback=fallback))
        return normalized

    @classmethod
    def normalize_stage(cls, raw_stage: Any, *, index: int, fallback: dict[str, Any]) -> dict[str, Any]:
        """归一化单个阶段，缺失字段使用兜底阶段或默认文案补齐。"""

        stage = raw_stage if isinstance(raw_stage, dict) else {}
        return {
            "stage_no": cls.int_value(stage.get("stage_no"), default=index + 1),
            "stage_name": cls.text_value(stage.get("stage_name"), fallback.get("stage_name"), cls.DEFAULT_STAGE_NAME),
            "core_goal": cls.text_value(stage.get("core_goal"), fallback.get("core_goal"), cls.DEFAULT_CORE_GOAL),
            "success_conditions": cls.text_list(
                stage.get("success_conditions"),
                fallback.get("success_conditions"),
                cls.DEFAULT_SUCCESS_CONDITION,
            ),
            "failure_conditions": cls.text_list(
                stage.get("failure_conditions"),
                fallback.get("failure_conditions"),
                cls.DEFAULT_FAILURE_CONDITION,
            ),
        }

    @staticmethod
    def int_value(value: Any, *, default: int) -> int:
        """把阶段序号转换成正整数。"""

        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default
        return max(1, result)

    @staticmethod
    def text_value(*values: Any) -> str:
        """按顺序返回第一个非空文本。"""

        for value in values:
            text = str(value).strip() if value is not None else ""
            if text:
                return text
        return ""

    @classmethod
    def text_list(cls, value: Any, fallback: Any, default: str) -> list[str]:
        """把字符串或列表统一转换成非空字符串列表。"""

        result = cls.list_value(value)
        if result:
            return result
        result = cls.list_value(fallback)
        return result or [default]

    @staticmethod
    def list_value(value: Any) -> list[str]:
        """把 LLM 返回的条件字段转换成字符串列表。"""

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
