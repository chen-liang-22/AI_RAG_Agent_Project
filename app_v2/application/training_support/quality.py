from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import yaml

from app_v2.application.training_support.strategies.knowledge_ingest_strategy import TrainingChunk
from core.utils.path_tool import get_abs_path


TRAINING_INGEST_CONFIG_PATH = get_abs_path("config/training_ingest.yml")


@dataclass
class TrainingIngestQualityReport:
    """训练资料切片质量报告。

    这个报告只评估切片结果是否健康，不关心具体业务关键词。
    """

    score: int  # 0-100 的质量分。
    level: str  # good / review / poor。
    passed: bool  # 是否达到自动通过阈值。
    summary: str  # 给前端展示的一句话结论。
    metrics: dict[str, Any] = field(default_factory=dict)  # 详细指标。
    warnings: list[str] = field(default_factory=list)  # 风险提示。

    def to_dict(self) -> dict[str, Any]:
        """转换成可 JSON 序列化的字典。"""

        return {
            "score": self.score,
            "level": self.level,
            "passed": self.passed,
            "summary": self.summary,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


class TrainingIngestQualityEvaluator:
    """训练资料切片质量评估器。

    第一阶段不用 LLM，只用通用健康指标评估规则切片是否可靠。
    """

    DEFAULT_CONFIG = {
        "pass_score": 80,
        "review_score": 60,
        "max_chunk_chars": 1800,
        "max_single_part_ratio": 0.75,
        "required_parts": ["case_profile", "task_requirement", "standard_answer"],
        "recommended_parts": ["scoring_rubric"],
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化质量评估器，优先使用配置文件中的阈值。"""

        merged_config = dict(self.DEFAULT_CONFIG)
        merged_config.update(config or self._load_quality_config())
        self.config = merged_config

    def evaluate(self, chunks: list[TrainingChunk]) -> TrainingIngestQualityReport:
        """对切片结果打分。"""

        warnings: list[str] = []
        metrics = self._build_metrics(chunks)
        score = 100

        if metrics["chunk_count"] <= 0:
            return TrainingIngestQualityReport(
                score=0,
                level="poor",
                passed=False,
                summary="没有生成有效切片，不能发布入库。",
                metrics=metrics,
                warnings=["没有切出任何有效内容。"],
            )

        if metrics["case_count"] <= 0:
            score -= 20
            warnings.append("没有识别到明确案例序号，可能是文档结构不符合 LMS 模板。")

        missing_required = metrics["missing_required_parts"]
        if missing_required:
            score -= min(30, len(missing_required) * 10)
            warnings.append(f"缺少核心片段：{', '.join(missing_required)}。")

        missing_recommended = metrics["missing_recommended_parts"]
        if missing_recommended:
            score -= min(10, len(missing_recommended) * 5)
            warnings.append(f"缺少推荐片段：{', '.join(missing_recommended)}。")

        if metrics["single_part_ratio"] > float(self.config["max_single_part_ratio"]):
            score -= 20
            warnings.append("内容过度集中在单一片段，说明规则切分可能没有识别出完整结构。")

        if metrics["oversize_chunk_count"] > 0:
            score -= min(15, metrics["oversize_chunk_count"] * 5)
            warnings.append("存在过长切片，可能影响向量检索精度。")

        if metrics["avg_chunk_chars"] < 80:
            score -= 10
            warnings.append("平均切片过短，可能缺少可用于训练的完整上下文。")

        score = max(0, min(100, score))
        pass_score = int(self.config["pass_score"])
        review_score = int(self.config["review_score"])
        if score >= pass_score:
            level = "good"
            summary = "切片结构较完整，可以确认发布。"
        elif score >= review_score:
            level = "review"
            summary = "切片质量一般，建议人工预览后再发布。"
        else:
            level = "poor"
            summary = "切片质量较低，建议调整资料格式后重新上传。"

        return TrainingIngestQualityReport(
            score=score,
            level=level,
            passed=score >= pass_score,
            summary=summary,
            metrics=metrics,
            warnings=warnings,
        )

    def _build_metrics(self, chunks: list[TrainingChunk]) -> dict[str, Any]:
        """统计切片健康指标。"""

        part_counts = Counter(chunk.case_part for chunk in chunks)
        total_chars = sum(len(chunk.text or "") for chunk in chunks)
        max_chunk_chars = int(self.config["max_chunk_chars"])
        case_indexes = {
            chunk.metadata.get("case_index")
            for chunk in chunks
            if chunk.metadata.get("case_index") is not None
        }
        required_parts = [str(item) for item in self.config.get("required_parts") or []]
        recommended_parts = [str(item) for item in self.config.get("recommended_parts") or []]
        present_parts = set(part_counts.keys())
        largest_part_chars = 0
        for case_part in present_parts:
            part_chars = sum(len(chunk.text or "") for chunk in chunks if chunk.case_part == case_part)
            largest_part_chars = max(largest_part_chars, part_chars)

        return {
            "chunk_count": len(chunks),
            "case_count": len(case_indexes) or (1 if chunks else 0),
            "part_counts": dict(part_counts),
            "present_parts": sorted(present_parts),
            "missing_required_parts": [part for part in required_parts if part not in present_parts],
            "missing_recommended_parts": [part for part in recommended_parts if part not in present_parts],
            "total_chars": total_chars,
            "avg_chunk_chars": int(total_chars / len(chunks)) if chunks else 0,
            "max_chunk_chars": max((len(chunk.text or "") for chunk in chunks), default=0),
            "oversize_chunk_count": sum(1 for chunk in chunks if len(chunk.text or "") > max_chunk_chars),
            "single_part_ratio": round(largest_part_chars / total_chars, 4) if total_chars else 0,
        }

    @staticmethod
    def _load_quality_config() -> dict[str, Any]:
        """从配置文件读取质量评估阈值。"""

        try:
            with open(TRAINING_INGEST_CONFIG_PATH, "r", encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError):
            return {}
        quality_config = data.get("quality") if isinstance(data, dict) else {}
        return quality_config if isinstance(quality_config, dict) else {}
