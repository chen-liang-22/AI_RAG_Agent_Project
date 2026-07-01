"""销售训练资料适用性门禁。

切片质量分只说明“结构是否健康”，不能说明“资料是不是销售训练资料”。
本模块单独评估资料是否包含客户案例、任务要求、销售话术、隐藏心理和评分标准等信号，
防止 LLM 把明显无关资料硬套成训练切片后拿到高分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.application.training_support.strategies.knowledge_ingest_strategy import TrainingChunk
from core.utils.config_handler import training_conf


@dataclass
class TrainingMaterialSuitabilityReport:
    """销售训练资料适用性报告。"""

    score: int  # 0-100 的适用性分数。
    passed: bool  # 是否通过适用性门禁。
    level: str  # good / review / poor。
    summary: str  # 给前端展示的一句话结论。
    signals: dict[str, bool] = field(default_factory=dict)  # 命中的资料信号。
    warnings: list[str] = field(default_factory=list)  # 不适合原因或风险提示。

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""

        return {
            "score": self.score,
            "passed": self.passed,
            "level": self.level,
            "summary": self.summary,
            "signals": self.signals,
            "warnings": self.warnings,
        }


class TrainingMaterialSuitabilityEvaluator:
    """销售训练资料适用性评估器。

    这里不用 LLM，先用可解释的本地信号做门禁。
    LLM 可以辅助切片，但不能单独决定一份资料是否适合进入销售训练库。
    """

    DEFAULT_CONFIG = {
        "pass_score": 60,
        "review_score": 40,
        "fail_score_cap": 40,
        "required_signals": ["customer", "task", "sales_answer"],
        "signal_keywords": {
            "customer": ["客户", "企业", "老板", "学员", "画像", "行业", "阶段"],
            "task": ["任务", "要求", "目标", "训练", "沟通", "谈单", "销售"],
            "sales_answer": ["话术", "答案", "参考", "回应", "异议", "成交", "方案"],
            "psychology": ["隐性", "心理", "顾虑", "痛点", "需求", "担心"],
            "rubric": ["评分", "命中", "扣分", "标准", "能力维度"],
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化适用性评估器，优先读取 config/training.yml。"""

        merged_config = dict(self.DEFAULT_CONFIG)
        merged_config.update(config or self._load_config())
        self.config = merged_config

    def evaluate(self, *, source_text: str = "", chunks: list[TrainingChunk] | None = None) -> TrainingMaterialSuitabilityReport:
        """评估原文和切片是否像销售训练资料。"""

        clean_source = source_text.strip()
        # 适用性门禁优先看原文，不能让 LLM 后补的“客户画像/任务要求”等结构词反向证明资料合格。
        # 如果没有原文，才退而求其次查看切片正文，兼容历史调用。
        signal_text = clean_source or self._chunks_text(chunks or [])
        signal_keywords = self._signal_keywords()
        signals = {
            signal_name: any(keyword in signal_text for keyword in keywords)
            for signal_name, keywords in signal_keywords.items()
        }
        required_signals = [str(item) for item in self.config.get("required_signals") or []]
        missing_required = [signal_name for signal_name in required_signals if not signals.get(signal_name)]

        score = self._score(signals, missing_required, bool(clean_source or chunks))
        pass_score = int(self.config.get("pass_score") or self.DEFAULT_CONFIG["pass_score"])
        review_score = int(self.config.get("review_score") or self.DEFAULT_CONFIG["review_score"])
        warnings: list[str] = []
        if missing_required:
            warnings.append(f"缺少销售训练核心信号：{', '.join(missing_required)}。")
        if not any(signals.values()):
            warnings.append("未识别到客户、任务、话术、评分等销售训练信号。")

        if score >= pass_score:
            level = "good"
            summary = "资料适合进入销售训练库。"
        elif score >= review_score:
            level = "review"
            summary = "资料和销售训练有一定相关性，建议人工确认。"
        else:
            level = "poor"
            summary = "资料不像销售训练资料，不建议使用 LLM 兜底结果。"

        return TrainingMaterialSuitabilityReport(
            score=score,
            passed=score >= pass_score,
            level=level,
            summary=summary,
            signals=signals,
            warnings=warnings,
        )

    def fail_score_cap(self) -> int:
        """返回适用性失败时的质量分封顶值。"""

        return int(self.config.get("fail_score_cap") or self.DEFAULT_CONFIG["fail_score_cap"])

    @staticmethod
    def apply_to_quality_report(
            quality_report: dict[str, Any],
            suitability_report: TrainingMaterialSuitabilityReport,
            *,
            fail_score_cap: int,
    ) -> dict[str, Any]:
        """把适用性报告合并到质量报告，并在不适合时封顶质量分。"""

        result = dict(quality_report)
        warnings = list(result.get("warnings") or [])
        result["suitability"] = suitability_report.to_dict()
        if not suitability_report.passed:
            original_score = int(result.get("score") or 0)
            capped_score = min(original_score, int(fail_score_cap))
            result["score"] = capped_score
            result["passed"] = False
            result["level"] = "poor"
            result["suitability_score_cap_applied"] = True
            result["original_score_before_suitability"] = original_score
            warnings.extend(suitability_report.warnings)
            warnings.append("资料适用性未通过，质量分已封顶，避免无关资料被当作高质量训练资料。")
        result["warnings"] = warnings
        return result

    @classmethod
    def _score(cls, signals: dict[str, bool], missing_required: list[str], has_content: bool) -> int:
        """按命中信号计算适用性分数。"""

        if not has_content:
            return 0
        weights = {
            "customer": 25,
            "task": 25,
            "sales_answer": 25,
            "psychology": 15,
            "rubric": 10,
        }
        score = sum(weight for signal_name, weight in weights.items() if signals.get(signal_name))
        if missing_required:
            score -= min(30, len(missing_required) * 10)
        return max(0, min(100, score))

    @staticmethod
    def _chunks_text(chunks: list[TrainingChunk]) -> str:
        """合并切片正文，作为没有原文时的兜底判断输入。"""

        chunk_text = "\n".join(chunk.text or "" for chunk in chunks)
        return chunk_text

    def _signal_keywords(self) -> dict[str, list[str]]:
        """读取并清洗信号关键词配置。"""

        raw_keywords = self.config.get("signal_keywords")
        if not isinstance(raw_keywords, dict):
            raw_keywords = self.DEFAULT_CONFIG["signal_keywords"]
        result: dict[str, list[str]] = {}
        for signal_name, keywords in raw_keywords.items():
            if isinstance(keywords, list):
                result[str(signal_name)] = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        return result

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """从训练配置读取适用性门禁参数。"""

        suitability_config = training_conf.get("suitability")
        return suitability_config if isinstance(suitability_config, dict) else {}
