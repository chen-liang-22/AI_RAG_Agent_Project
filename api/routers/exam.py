import json
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient, models

from api.exam_schemas import (
    ExamAnswerRequest,
    ExamGenerateRequest,
    ExamGenerateResponse,
    ExamGradeRequest,
    ExamGradeResponse,
    ExamGradeResult,
    ExamQuestionResponse,
    ExamQuestionSource,
)
from model.factory import get_chat_model, get_chat_model_name_for_mode, normalize_chat_model_mode
from utils.logger_handler import logger
from utils.qdrant_options import get_qdrant_client_options, normalize_qdrant_collection_name

router = APIRouter()

MAX_EXAM_SCROLL_ROUNDS = 200


def _elapsed_ms(start_time: float) -> float:
    """计算接口耗时，单位毫秒。"""

    return (time.perf_counter() - start_time) * 1000


def _payload_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """从 Qdrant payload 中读取业务 metadata。"""

    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _payload_content(payload: dict[str, Any]) -> str:
    """从 Qdrant payload 中读取文本内容。"""

    content = payload.get("page_content") or payload.get("content") or ""
    return str(content).strip()


def _extract_reference_answer(content: str, question: str) -> str:
    """从结构化 QA 文本中抽取参考答案。"""

    answer_match = re.search(r"(?:^|\n)\s*答案[:：]\s*(.+)", content, flags=re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    legacy_answer_match = re.search(r"(?:^|\n)\s*(?:答|A)[:：]\s*(.+)", content, flags=re.IGNORECASE | re.DOTALL)
    if legacy_answer_match:
        return legacy_answer_match.group(1).strip()
    return content.replace(question, "", 1).strip() or content


def _match_section(metadata: dict[str, Any], section_path: str | None) -> bool:
    """判断题目是否属于指定章节。"""

    if not section_path:
        return True
    expected = section_path.strip()
    actual = str(metadata.get("section_path") or metadata.get("heading_path") or metadata.get("category") or "")
    return actual.startswith(expected)


def _build_exam_filter(request: ExamGenerateRequest) -> models.Filter:
    """根据组卷条件构造 Qdrant metadata 过滤器。"""

    conditions: list[models.FieldCondition] = [
        models.FieldCondition(key="metadata.content_type", match=models.MatchValue(value="qa")),
    ]
    if request.document_id:
        conditions.append(
            models.FieldCondition(key="metadata.document_id", match=models.MatchValue(value=request.document_id))
        )
    return models.Filter(must=conditions)


def _scroll_candidate_questions(request: ExamGenerateRequest) -> list[dict[str, Any]]:
    """从 Qdrant 中读取满足条件的结构化 QA 候选题。"""

    collection_name = normalize_qdrant_collection_name(request.collection_name)
    client = QdrantClient(**get_qdrant_client_options())
    candidates: list[dict[str, Any]] = []
    offset = None
    scroll_rounds = 0

    while True:
        scroll_rounds += 1
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=_build_exam_filter(request),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = _payload_metadata(payload)
            question = str(metadata.get("question") or "").strip()
            content = _payload_content(payload)
            if not question or not content or not _match_section(metadata, request.section_path):
                continue
            candidates.append({"payload": payload, "metadata": metadata, "content": content})
        if offset is None or scroll_rounds >= MAX_EXAM_SCROLL_ROUNDS:
            break

    logger.info(
        "[考试] 题源筛选完成 Collection=%s 文件编号=%s 章节=%s 扫描轮次=%s 候选题数=%s",
        collection_name,
        request.document_id,
        request.section_path,
        scroll_rounds,
        len(candidates),
    )
    return candidates


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按原始题目编号和题干去重，避免同一题多片段重复出现。"""

    seen: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for item in candidates:
        metadata = item["metadata"]
        key = str(metadata.get("question_id") or metadata.get("qa_id") or metadata.get("question") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def _build_exam_question(item: dict[str, Any], request: ExamGenerateRequest, paper_id: str, index: int) -> ExamQuestionResponse:
    """把 Qdrant 候选题转换成前端考试题目。"""

    metadata = item["metadata"]
    question = str(metadata.get("question") or "").strip()
    reference_answer = _extract_reference_answer(item["content"], question)
    source_question_id = str(metadata.get("question_id") or metadata.get("qa_id") or metadata.get("segment_id") or "")
    return ExamQuestionResponse(
        question_id=f"{paper_id}_q_{index:04d}",
        source_question_id=source_question_id or None,
        question_type="short_answer",
        difficulty=request.difficulty,
        question=question,
        reference_answer=reference_answer,
        score=request.score_per_question,
        source=ExamQuestionSource(
            document_id=_optional_text(metadata.get("document_id")),
            filename=_optional_text(metadata.get("source_file") or metadata.get("source")),
            section_path=_optional_text(metadata.get("section_path") or metadata.get("heading_path") or metadata.get("category")),
            source_page=_optional_int(metadata.get("source_page") or metadata.get("page_no") or metadata.get("page")),
        ),
    )


def _optional_text(value: object) -> str | None:
    """把可选字段转换成干净字符串，空值统一返回 None。"""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    """把页码等可选字段转换成整数，无法转换时返回 None。"""

    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@router.post("/exam/generate", response_model=ExamGenerateResponse)
def generate_exam(request: ExamGenerateRequest) -> ExamGenerateResponse:
    """从结构化题库生成简答试卷。"""

    start_time = time.perf_counter()
    if "short_answer" not in {item.strip() for item in request.question_types}:
        raise HTTPException(status_code=400, detail="第一阶段仅支持 short_answer 简答题")

    logger.info(
        "[考试] 开始生成试卷 模式=%s Collection=%s 文件编号=%s 章节=%s 题目数量=%s",
        request.mode,
        normalize_qdrant_collection_name(request.collection_name),
        request.document_id,
        request.section_path,
        request.question_count,
    )
    candidates = _deduplicate_candidates(_scroll_candidate_questions(request))
    if not candidates:
        raise HTTPException(status_code=404, detail="当前题源范围内没有可用于考试的结构化问答题")

    random_generator = random.Random(request.seed)
    random_generator.shuffle(candidates)
    selected_items = candidates[:request.question_count]
    paper_id = f"paper_{uuid.uuid4().hex}"
    questions = [
        _build_exam_question(item, request, paper_id, index)
        for index, item in enumerate(selected_items, start=1)
    ]
    logger.info(
        "[考试] 试卷生成完成 试卷编号=%s 题目数量=%s 耗时毫秒=%.2f",
        paper_id,
        len(questions),
        _elapsed_ms(start_time),
    )
    return ExamGenerateResponse(
        paper_id=paper_id,
        title="知识掌握度测评",
        total_score=sum(question.score for question in questions),
        questions=questions,
    )


def _parse_model_json(content: object) -> dict[str, Any]:
    """从模型返回文本中解析 JSON 对象。"""

    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0)
    return json.loads(text)


def _fallback_grade(answer: ExamAnswerRequest, message: str) -> ExamGradeResult:
    """模型评分失败时返回可展示的保底评分。"""

    return ExamGradeResult(
        question_id=answer.question_id,
        score=0,
        max_score=answer.max_score,
        wrong_points=["评分模型调用失败"],
        comment=message,
        review_suggestion="请稍后重新评分，或检查模型配置。",
    )


def _normalize_grade_result(answer: ExamAnswerRequest, raw_result: dict[str, Any]) -> ExamGradeResult:
    """规范化模型返回的评分 JSON。"""

    try:
        score = float(raw_result.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(score, float(answer.max_score)))

    def list_field(name: str) -> list[str]:
        value = raw_result.get(name) or []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    return ExamGradeResult(
        question_id=answer.question_id,
        score=score,
        max_score=answer.max_score,
        hit_points=list_field("hit_points"),
        missing_points=list_field("missing_points"),
        wrong_points=list_field("wrong_points"),
        comment=str(raw_result.get("comment") or "").strip(),
        review_suggestion=str(raw_result.get("review_suggestion") or "").strip(),
    )


def _grade_one_answer(answer: ExamAnswerRequest, model_mode: str | None) -> ExamGradeResult:
    """调用模型对单道简答题做结构化评分。"""

    if not answer.answer.strip():
        return ExamGradeResult(
            question_id=answer.question_id,
            score=0,
            max_score=answer.max_score,
            missing_points=["未作答"],
            comment="该题未作答。",
            review_suggestion="先补全答案后再提交评分。",
        )

    try:
        model = get_chat_model(model_mode)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是严格但公正的中文阅卷老师。"
                        "必须只根据题目和参考答案评分，不要引入外部知识。"
                        "必须只返回 JSON，不要返回 Markdown。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"题目：{answer.question}\n\n"
                        f"参考答案：{answer.reference_answer}\n\n"
                        f"学生答案：{answer.answer}\n\n"
                        f"满分：{answer.max_score}\n\n"
                        "返回 JSON 格式："
                        '{"score":0,"hit_points":[],"missing_points":[],"wrong_points":[],"comment":"","review_suggestion":""}'
                    )
                ),
            ]
        )
        return _normalize_grade_result(answer, _parse_model_json(response.content))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.error("[考试] 单题评分失败 题目编号=%s 错误=%s", answer.question_id, exc, exc_info=True)
        return _fallback_grade(answer, f"评分失败：{exc}")


@router.post("/exam/grade", response_model=ExamGradeResponse)
def grade_exam(request: ExamGradeRequest) -> ExamGradeResponse:
    """批量评分考试答案，返回结构化结果。"""

    start_time = time.perf_counter()
    if not request.answers:
        raise HTTPException(status_code=400, detail="评分答案不能为空")

    selected_model_mode = normalize_chat_model_mode(request.model_mode)
    selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
    logger.info(
        "[考试] 开始批量评分 试卷编号=%s 用户编号=%s 答案数量=%s 模型模式=%s 模型名称=%s",
        request.paper_id,
        request.user_id,
        len(request.answers),
        selected_model_mode,
        selected_model_name,
    )

    results: list[ExamGradeResult] = []
    max_workers = min(5, len(request.answers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_grade_one_answer, answer, selected_model_mode): answer.question_id
            for answer in request.answers
        }
        for future in as_completed(future_map):
            results.append(future.result())

    result_order = {answer.question_id: index for index, answer in enumerate(request.answers)}
    results.sort(key=lambda item: result_order.get(item.question_id, 0))
    total_score = sum(item.score for item in results)
    max_score = sum(answer.max_score for answer in request.answers)
    logger.info(
        "[考试] 批量评分完成 试卷编号=%s 用户编号=%s 总分=%.2f 满分=%s 耗时毫秒=%.2f",
        request.paper_id,
        request.user_id,
        total_score,
        max_score,
        _elapsed_ms(start_time),
    )
    return ExamGradeResponse(
        paper_id=request.paper_id,
        total_score=total_score,
        max_score=max_score,
        results=results,
    )
