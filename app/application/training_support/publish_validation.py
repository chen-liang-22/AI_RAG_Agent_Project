"""销售训练资料发布后验证。

资料发布到正式 Qdrant collection 后，会随机抽样几个切片再检索一次。
验证目标不是评估内容质量，而是确认“按 batch_id 能不能从正式向量库找回来”。
如果这里失败，通常说明向量点复制、metadata 或过滤条件有问题。
"""

from typing import Any


from core.utils.logger_handler import logger
from core.utils.config_handler import training_conf


class TrainingPublishValidator:
    """训练资料发布后的抽样检索验证器。

    发布成功只代表向量点写入完成，不代表后续检索一定能按 batch_id 命中。
    这个验证器会抽取少量切片，用切片正文回查 Qdrant，并统计是否命中本批次。
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "sample_size": 3,
        "search_k": 3,
        "min_hit_ratio": 0.6,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化发布验证器，并读取验证阈值。"""

        merged_config = dict(self.DEFAULT_CONFIG)
        merged_config.update(config or self._load_config())
        self.config = merged_config

    def validate(self, *, vector_service: Any, batch_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """执行发布后抽样检索验证。"""

        if not bool(self.config.get("enabled")):
            return {"enabled": False, "passed": True, "summary": "发布后抽样检索验证未启用。"}
        if not hasattr(vector_service, "search_documents"):
            return {
                "enabled": True,
                "passed": False,
                "summary": "当前向量服务不支持抽样检索验证。",
                "sample_size": 0,
                "hit_count": 0,
                "hit_ratio": 0,
            }

        sample_size = max(1, int(self.config.get("sample_size") or self.DEFAULT_CONFIG["sample_size"]))
        search_k = max(1, int(self.config.get("search_k") or self.DEFAULT_CONFIG["search_k"]))
        min_hit_ratio = float(self.config.get("min_hit_ratio") or self.DEFAULT_CONFIG["min_hit_ratio"])
        samples = self._select_samples(chunks, sample_size)
        if not samples:
            return {
                "enabled": True,
                "passed": False,
                "summary": "没有可用于抽样验证的切片。",
                "sample_size": 0,
                "hit_count": 0,
                "hit_ratio": 0,
            }

        results: list[dict[str, Any]] = []
        hit_count = 0
        for row in samples:
            query = self._query_text(row)
            try:
                documents = vector_service.search_documents(
                    query,
                    k=search_k,
                    filters={"batch_id": [batch_id]},
                )
            except (RuntimeError, TimeoutError, OSError, ValueError, TypeError, AttributeError) as exc:
                logger.warning(
                    "[销售训练][发布验证] 抽样检索失败 批次编号=%s 切片编号=%s 错误=%s",
                    batch_id,
                    row.get("chunk_id"),
                    exc,
                )
                documents = []

            hit = any(document.metadata.get("batch_id") == batch_id for document in documents)
            if hit:
                hit_count += 1
            results.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "case_part": row.get("case_part"),
                    "hit": hit,
                    "top_scores": [
                        round(float(document.metadata.get("_vector_score") or 0), 6)
                        for document in documents[:search_k]
                    ],
                }
            )

        hit_ratio = round(hit_count / len(samples), 4)
        passed = hit_ratio >= min_hit_ratio
        summary = "发布后抽样检索验证通过。" if passed else "发布后抽样检索验证未达标，请检查向量库写入或过滤条件。"
        return {
            "enabled": True,
            "passed": passed,
            "summary": summary,
            "sample_size": len(samples),
            "hit_count": hit_count,
            "hit_ratio": hit_ratio,
            "min_hit_ratio": min_hit_ratio,
            "results": results,
        }

    @staticmethod
    def _select_samples(chunks: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
        """按列表顺序均匀抽取样本，避免完全随机导致验证不可复现。"""

        if len(chunks) <= sample_size:
            return chunks
        if sample_size == 1:
            return [chunks[0]]
        step = (len(chunks) - 1) / (sample_size - 1)
        indexes = [round(index * step) for index in range(sample_size)]
        return [chunks[index] for index in indexes]

    @staticmethod
    def _query_text(row: dict[str, Any]) -> str:
        """从切片中取一段稳定查询文本。"""

        text = str(row.get("chunk_text") or "").strip()
        return text[:400]

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """从统一训练配置中读取发布后验证参数。"""

        validation_config = training_conf.get("publish_validation")
        return validation_config if isinstance(validation_config, dict) else {}
