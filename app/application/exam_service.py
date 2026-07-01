"""V2 对话式考试应用入口。

核心流程：
1. 从 Qdrant 中按向量库、文件和一级目录筛选结构化 QA 题源；
2. 开始考试时随机抽题，并优先调用 LLM 把原始 QA 润色成正式试题；
3. 用户逐轮提交答案，后端用 LLM 分析得分、正确答案、命中点和遗漏点；
4. 会话、题目、用户答案和分析结果全部写入业务数据库，供历史记录查看。

当前文件是 V2 化的第一步：先把旧考试真实实现搬到 app 路径，
避免 `/api/v2/exam/*` 继续直接挂旧路由。后续再继续拆成 service 和 repository。
"""

import json
import random
import re
import time
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient, models

from api.exam_schemas import (
    ExamAnswerAnalysis,
    ExamAnswerRequest,
    ExamAnswerResponse,
    ExamConversationQuestion,
    ExamHistoryListResponse,
    ExamQuestionRecord,
    ExamQuestionSource,
    ExamSectionsResponse,
    ExamSectionResponse,
    ExamSessionDeleteResponse,
    ExamSessionDetailResponse,
    ExamSessionSummary,
    ExamStartRequest,
    ExamStartResponse,
)
from app.infrastructure.repositories.exam_repository import ExamRepository, get_exam_repository
from core.model.factory import get_chat_model
from core.utils.database_connection import IntegrityErrorTypes
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager
from core.utils.qdrant_options import get_qdrant_client_options, normalize_qdrant_collection_name

router = APIRouter(prefix="/exam", tags=["V2 问答考试"])

# Qdrant scroll 的最大翻页次数，避免数据异常时接口长时间循环。
MAX_EXAM_SCROLL_ROUNDS = 200

# 当前考试支持的题型编码。前端传入空列表或非法值时，会回退到这里的完整集合。
ALL_QUESTION_TYPES = ["single_choice", "multiple_choice", "true_false", "short_answer", "fill_blank"]

# 入库时多级目录统一使用该分隔符；考试页只展示第一层目录。
SECTION_PATH_SEPARATOR = " / "

# 题型中文名主要用于命题和阅卷提示词，让模型更稳定地理解题型。
QUESTION_TYPE_LABELS = {
    "single_choice": "单选题",
    "multiple_choice": "多选题",
    "true_false": "判断题",
    "short_answer": "简答题",
    "fill_blank": "填空题",
}


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


def _format_response_time(value: object) -> str | None:
    """把数据库时间字段统一转换成接口响应字符串。"""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds", sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _metadata_section_path(metadata: dict[str, Any]) -> str:
    """从题目 metadata 中读取完整目录路径。"""

    return str(metadata.get("section_path") or metadata.get("heading_path") or metadata.get("category") or "").strip()


def _metadata_first_level_section(metadata: dict[str, Any]) -> str:
    """从题目 metadata 中读取一级目录，兼容旧数据没有 section_first_level 的情况。"""

    first_level = str(metadata.get("section_first_level") or metadata.get("section_title") or "").strip()
    if first_level:
        return first_level
    return _first_level_section_path(_metadata_section_path(metadata))


def _first_level_section_path(section_path: str | None) -> str:
    """把完整目录路径压缩为第一层目录，用于前端下拉展示。"""

    clean_path = (section_path or "").strip()
    if not clean_path:
        return ""
    return clean_path.split(SECTION_PATH_SEPARATOR, maxsplit=1)[0].strip()


def _section_sort_key(section_path: str) -> tuple[int, int, str]:
    """按目录中的数字自然排序，保证 1、2、3、10 这种顺序正确。"""

    number_match = re.search(r"\d+", section_path)
    if number_match:
        return 0, int(number_match.group(0)), section_path
    return 1, 0, section_path


def _match_section(metadata: dict[str, Any], section_path: str | None) -> bool:
    """判断题目是否属于指定目录路径。"""

    if not section_path:
        return True
    expected = section_path.strip()
    actual = _metadata_section_path(metadata)
    actual_first_level = _metadata_first_level_section(metadata)
    return (
        actual == expected
        or actual.startswith(f"{expected}{SECTION_PATH_SEPARATOR}")
        or actual_first_level == expected
    )


def _build_exam_filter(document_id: str | None) -> models.Filter:
    """根据题源条件构造 Qdrant metadata 过滤器。"""

    # 考试只从结构化 QA 片段中抽题，普通文本切片不直接进入题库。
    conditions: list[models.FieldCondition] = [
        models.FieldCondition(key="metadata.content_type", match=models.MatchValue(value="qa")),
    ]
    if document_id:
        conditions.append(models.FieldCondition(key="metadata.document_id", match=models.MatchValue(value=document_id)))
    return models.Filter(must=conditions)


def _scroll_candidate_questions(
        *,
        collection_name: str | None,
        document_id: str | None = None,
        section_path: str | None = None,
) -> list[dict[str, Any]]:
    """从 Qdrant 中读取满足条件的结构化 QA 候选题。"""

    final_collection_name = normalize_qdrant_collection_name(collection_name)
    client = QdrantClient(**get_qdrant_client_options())
    candidates: list[dict[str, Any]] = []
    offset = None
    scroll_rounds = 0

    if not client.collection_exists(final_collection_name):
        logger.warning("[考试] 题源向量库不存在，返回空题源 Collection=%s 文档编号=%s", final_collection_name, document_id)
        return []

    while True:
        scroll_rounds += 1
        points, offset = client.scroll(
            collection_name=final_collection_name,
            scroll_filter=_build_exam_filter(document_id),
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
            # 只保留有题干、有正文、且命中目录过滤条件的 QA 片段。
            if not question or not content or not _match_section(metadata, section_path):
                continue
            candidates.append({"payload": payload, "metadata": metadata, "content": content})
        if offset is None or scroll_rounds >= MAX_EXAM_SCROLL_ROUNDS:
            break

    logger.info(
        "[考试] 题源筛选完成 Collection=%s 文件编号=%s 目录=%s 扫描轮次=%s 候选题数=%s",
        final_collection_name,
        document_id,
        section_path,
        scroll_rounds,
        len(candidates),
    )
    return _deduplicate_candidates(candidates)


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


def _session_summary(row: dict[str, Any]) -> ExamSessionSummary:
    """把数据库考试会话记录转换成响应模型。"""

    return ExamSessionSummary(
        session_id=row["session_id"],
        user_id=row.get("user_id"),
        title=row.get("title") or "知识掌握度测评",
        collection_name=row["collection_name"],
        document_id=row.get("document_id"),
        filename=row.get("filename"),
        section_path=row.get("section_path"),
        round_count=int(row["round_count"]),
        answered_count=int(row["answered_count"]),
        total_score=round(float(row.get("total_score") or 0), 2),
        max_score=round(float(row.get("max_score") or 100), 2),
        status=row["status"],
        current_round=int(row.get("current_round") or 1),
        created_at=_format_response_time(row["created_at"]),
        updated_at=_format_response_time(row["updated_at"]),
        completed_at=_format_response_time(row.get("completed_at")),
    )


def _parse_json_field(value: str | None, default: Any) -> Any:
    """解析数据库 JSON 字段，失败时返回默认值。"""

    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _question_from_row(row: dict[str, Any]) -> ExamConversationQuestion:
    """把数据库考试题目记录转换成前端展示题目。"""

    return ExamConversationQuestion(
        exam_question_id=row["exam_question_id"],
        round_no=int(row["round_no"]),
        question_type=row["question_type"],
        prompt=row["prompt"],
        options=_parse_json_field(row.get("options_json"), []),
        max_score=round(float(row["max_score"]), 2),
        source=ExamQuestionSource(
            document_id=row.get("source_document_id"),
            filename=row.get("source_filename"),
            section_path=row.get("section_path"),
            source_page=_optional_int(row.get("source_page")),
        ),
    )


def _choice_label(index: int) -> str:
    """把选项下标转换成 A/B/C/D 这类展示编号。"""

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(labels):
        return labels[index]
    return str(index + 1)


def _display_choice_options(options: list[str]) -> list[str]:
    """给选择题选项加编号，避免页面只展示一组裸答案文本。"""

    result: list[str] = []
    for index, option in enumerate(options):
        text = str(option).strip()
        if not text:
            continue
        label = _choice_label(index)
        if re.match(r"^[A-Z][.、．]\s*", text):
            result.append(text)
        else:
            result.append(f"{label}. {text}")
    return result


def _strip_choice_label(option: object) -> str:
    """去掉模型可能额外生成的 A./A、/A． 这类选项前缀。"""

    text = str(option or "").strip()
    return re.sub(r"^[A-Z][.、．]\s*", "", text, count=1, flags=re.IGNORECASE).strip()


def _is_choice_label_text(value: object) -> bool:
    """判断文本是否只是 A/B/C 这类选项编号，避免把编号误当成选项内容。"""

    return bool(re.fullmatch(r"[A-Z]", str(value or "").strip(), flags=re.IGNORECASE))


def _clean_generated_choice_options(options: list[str]) -> list[str]:
    """清洗模型生成的选择题选项，统一去编号、去空值和去重。"""

    clean_options: list[str] = []
    for option in options:
        clean_option = _strip_choice_label(option)
        if not clean_option or _is_choice_label_text(clean_option):
            continue
        if clean_option not in clean_options:
            clean_options.append(clean_option)
    return clean_options


def _resolve_generated_choice_answer(answer: object, raw_options: list[str], clean_options: list[str]) -> str:
    """把模型返回的单选答案解析成真实选项文本，而不是直接保留 A/B/C。"""

    clean_answer = str(answer or "").strip()
    if not clean_answer:
        return ""
    stripped_answer = _strip_choice_label(clean_answer)
    if stripped_answer in clean_options:
        return stripped_answer
    label_match = re.match(r"^([A-Z])(?:[.、．]|\s|$)", clean_answer, flags=re.IGNORECASE)
    if label_match:
        option_index = ord(label_match.group(1).upper()) - ord("A")
        if 0 <= option_index < len(raw_options):
            mapped_option = _strip_choice_label(raw_options[option_index])
            if mapped_option and not _is_choice_label_text(mapped_option):
                return mapped_option
        if 0 <= option_index < len(clean_options):
            return clean_options[option_index]
    return stripped_answer


def _resolve_generated_multiple_choice_answers(
        answer: object,
        raw_options: list[str],
        clean_options: list[str],
) -> list[str]:
    """把模型返回的多选答案列表解析成真实选项文本列表。"""

    resolved_answers: list[str] = []
    for item in _normalize_string_list(answer, max_items=6):
        resolved_answer = _resolve_generated_choice_answer(item, raw_options, clean_options)
        if resolved_answer and resolved_answer not in resolved_answers:
            resolved_answers.append(resolved_answer)
    return resolved_answers


def _option_label_map(options: list[str]) -> dict[str, str]:
    """建立选项编号到完整选项文本的映射。"""

    return {
        _choice_label(index): option
        for index, option in enumerate(options)
    }


def _normalize_choice_answer(answer: str | list[str], options: list[str]) -> str | list[str]:
    """把模型返回的选择题正确答案转换成前端实际提交的选项编号。"""

    label_by_option = {option: _choice_label(index) for index, option in enumerate(options)}
    display_by_option = {
        display_option: _choice_label(index)
        for index, display_option in enumerate(_display_choice_options(options))
    }

    def normalize_one(value: object) -> str:
        """把单个模型答案文本转换成选项编号。"""

        clean_value = str(value or "").strip()
        if clean_value in label_by_option:
            return label_by_option[clean_value]
        if clean_value in display_by_option:
            return display_by_option[clean_value]
        label_match = re.match(r"^([A-Z])(?:[.、．]|\s|$)", clean_value, flags=re.IGNORECASE)
        if label_match:
            label = label_match.group(1).upper()
            if label in _option_label_map(options):
                return label
        return clean_value

    if isinstance(answer, list):
        return [normalize_one(item) for item in answer]
    return normalize_one(answer)


def _prepare_objective_question_for_display(question_type: str, options: list[str], correct_answer: Any) -> tuple[list[str], Any]:
    """把客观题选项转成编号展示，并同步转换标准答案。"""

    if question_type not in {"single_choice", "multiple_choice"} or not options:
        return options, correct_answer
    return _display_choice_options(options), _normalize_choice_answer(correct_answer, options)


def _is_all_select_multiple_choice(question_type: str, options: list[str], correct_answer: Any) -> bool:
    """判断多选题的正确答案是否覆盖了全部选项。"""

    if question_type != "multiple_choice" or not options:
        return False
    option_labels = {_choice_label(index) for index, _ in enumerate(options)}
    if isinstance(correct_answer, list):
        answer_labels = {str(item).strip().upper() for item in correct_answer if str(item).strip()}
    else:
        answer_labels = {
            item.strip().upper()
            for item in re.split(r"[、,，;；\n]+", str(correct_answer or ""))
            if item.strip()
        }
    return bool(option_labels) and option_labels.issubset(answer_labels)


def _previous_multiple_choice_questions_all_select(session_id: str, round_no: int) -> bool:
    """判断当前轮之前已经保存的多选题是否全部都是全选答案。"""

    previous_multiple_choice_questions = [
        question
        for question in _store().list_exam_questions(session_id)
        if question.get("question_type") == "multiple_choice" and int(question.get("round_no") or 0) < round_no
    ]
    if not previous_multiple_choice_questions:
        return False

    for question in previous_multiple_choice_questions:
        options = _parse_json_field(question.get("options_json"), [])
        correct_answer = _parse_json_field(question.get("correct_answer_json"), None)
        if not _is_all_select_multiple_choice("multiple_choice", options, correct_answer):
            return False
    return True


def _analysis_from_row(row: dict[str, Any]) -> ExamAnswerAnalysis | None:
    """把数据库题目分析字段转换成响应模型。"""

    if row.get("analysis_json") is None:
        return None
    analysis = _parse_json_field(row.get("analysis_json"), {})
    return ExamAnswerAnalysis(
        is_correct=bool(row.get("is_correct")),
        score=round(float(row.get("score") or 0), 2),
        max_score=round(float(row.get("max_score") or 0), 2),
        correct_answer=analysis.get("correct_answer"),
        reference_answer=row.get("reference_answer") or "",
        hit_points=[str(item) for item in analysis.get("hit_points", [])],
        missing_points=[str(item) for item in analysis.get("missing_points", [])],
        wrong_points=[str(item) for item in analysis.get("wrong_points", [])],
        comment=str(analysis.get("comment") or ""),
    )


def _normalize_answer_value(answer: str | list[str]) -> str:
    """把前端答案统一成可保存的字符串。"""

    if isinstance(answer, list):
        return "、".join(str(item).strip() for item in answer if str(item).strip())
    return str(answer or "").strip()


def _normalize_answer_value_for_question(question: dict[str, Any], answer: str | list[str]) -> str:
    """按题型把前端答案统一成后端阅卷使用的答案格式。"""

    question_type = question.get("question_type")
    if question_type not in {"single_choice", "multiple_choice"}:
        return _normalize_answer_value(answer)

    options = _parse_json_field(question.get("options_json"), [])
    display_to_label = {
        option: _choice_label(index)
        for index, option in enumerate(options)
    }

    def normalize_one(value: object) -> str:
        """把单个前端答案文本转换成选项编号。"""

        clean_value = str(value or "").strip()
        if clean_value in display_to_label:
            return display_to_label[clean_value]
        label_match = re.match(r"^([A-Z])(?:[.、．]|\s|$)", clean_value, flags=re.IGNORECASE)
        if label_match:
            return label_match.group(1).upper()
        return clean_value

    if isinstance(answer, list):
        return "、".join(normalize_one(item) for item in answer if str(item).strip())

    clean_answer = str(answer or "").strip()
    if question_type == "multiple_choice":
        parts = [part for part in re.split(r"[、,，;；\n]+", clean_answer) if part.strip()]
        return "、".join(normalize_one(part) for part in parts)
    return normalize_one(clean_answer)


def _split_answer_points(answer: str) -> list[str]:
    """从参考答案中拆出可用于选择题的短答案片段。"""

    clean_answer = re.sub(r"\s+", " ", answer).strip()
    parts = re.split(r"[；;。\n]|(?:\d+[.、])|(?:[（(]?[一二三四五六七八九十]+[）)]、?)", clean_answer)
    points = []
    for part in parts:
        text = part.strip(" ：:，,、-")
        if 4 <= len(text) <= 80:
            points.append(text)
    if not points and clean_answer:
        points.append(clean_answer[:80])
    return points[:6]


def _sample_distractors(candidates: list[dict[str, Any]], current_answer: str, random_generator: random.Random, count: int) -> list[str]:
    """从同一题源候选答案中抽取干扰项。"""

    current_norm = current_answer.strip().lower()
    pool: list[str] = []
    for item in candidates:
        metadata = item["metadata"]
        question = str(metadata.get("question") or "").strip()
        answer = _extract_reference_answer(item["content"], question)
        for point in _split_answer_points(answer):
            if point.strip().lower() != current_norm and point not in pool:
                pool.append(point)
    random_generator.shuffle(pool)
    return pool[:count]


def _make_single_choice(
        question: str,
        answer: str,
        candidates: list[dict[str, Any]],
        random_generator: random.Random,
) -> tuple[str, list[str], str] | None:
    """把原始问答改造成单选题。"""

    # 这里是模型不可用时的兜底题目，不是主链路；主链路会优先走 _generate_question_with_model。
    correct_answer = _split_answer_points(answer)[0] if _split_answer_points(answer) else answer[:80]
    distractors = _sample_distractors(candidates, correct_answer, random_generator, 3)
    if len(distractors) < 3:
        return None
    options = [correct_answer, *distractors]
    random_generator.shuffle(options)
    return f"{question}\n请选择最符合参考答案的一项。", options, correct_answer


def _make_multiple_choice(
        question: str,
        answer: str,
        candidates: list[dict[str, Any]],
        random_generator: random.Random,
) -> tuple[str, list[str], list[str]] | None:
    """把原始问答改造成多选题。"""

    # 多选题需要至少两个正确点和两个干扰项，否则回退成简答题更稳。
    correct_points = _split_answer_points(answer)[:3]
    if len(correct_points) < 2:
        return None
    distractors = _sample_distractors(candidates, " ".join(correct_points), random_generator, 2)
    if len(distractors) < 2:
        return None
    options = [*correct_points, *distractors]
    random_generator.shuffle(options)
    return f"{question}\n请选择所有正确选项。", options, correct_points


def _target_true_false_answer(random_generator: random.Random) -> str:
    """为判断题生成目标答案，让正确/错误分布更均衡。"""

    return "正确" if random_generator.random() >= 0.5 else "错误"


def _make_true_false(
        question: str,
        answer: str,
        random_generator: random.Random,
        *,
        target_answer: str | None = None,
) -> tuple[str, list[str], str]:
    """把原始问答改造成判断题。"""

    # 规则兜底时按目标答案生成正误说法，避免判断题长期偏向“正确”。
    final_answer = target_answer if target_answer in {"正确", "错误"} else _target_true_false_answer(random_generator)
    if final_answer == "正确":
        statement = f"{question}：{_split_answer_points(answer)[0] if _split_answer_points(answer) else answer[:80]}"
    else:
        statement = f"{question}：以上说法不完整或不准确"
    return f"判断正误：{statement}", ["正确", "错误"], final_answer


def _make_fill_blank(question: str, answer: str) -> tuple[str, list[str], str] | None:
    """把原始问答改造成填空题。"""

    # 填空题依赖参考答案中能抽出清晰短答案，抽不出来时回退成简答题。
    answer_points = _split_answer_points(answer)
    if not answer_points:
        return None
    correct_answer = answer_points[0]
    prompt = f"{question}\n请填写关键答案：____"
    return prompt, [], correct_answer


def _question_type_label(question_type: str) -> str:
    """把题型编码转换成中文题型名称。"""

    return QUESTION_TYPE_LABELS.get(question_type, question_type)


def _normalize_string_list(value: object, *, max_items: int = 8) -> list[str]:
    """把模型返回的列表字段清洗成去重字符串列表。"""

    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []

    clean_items: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text and text not in clean_items:
            clean_items.append(text)
    return clean_items[:max_items]


def _validate_generated_question(raw_result: dict[str, Any], requested_type: str) -> tuple[str, list[str], Any]:
    """校验模型生成的题目结构，返回题干、选项和标准答案。"""

    prompt = str(raw_result.get("prompt") or raw_result.get("question") or "").strip()
    if not prompt:
        raise ValueError("模型生成题干为空")

    raw_options = _normalize_string_list(raw_result.get("options"), max_items=6)
    options = _clean_generated_choice_options(raw_options)
    correct_answer = raw_result.get("correct_answer")

    if requested_type == "single_choice":
        correct_text = _resolve_generated_choice_answer(correct_answer, raw_options, options)
        if len(options) < 4 or not correct_text:
            raise ValueError("单选题选项或答案不完整")
        if correct_text not in options:
            options = [correct_text, *[item for item in options if item != correct_text]][:4]
        return prompt, options[:4], correct_text

    if requested_type == "multiple_choice":
        correct_items = _resolve_generated_multiple_choice_answers(correct_answer, raw_options, options)
        if len(options) < 4 or len(correct_items) < 2:
            raise ValueError("多选题选项或答案不完整")
        missing_answers = [item for item in correct_items if item not in options]
        options = [*missing_answers, *options]
        options = _normalize_string_list(options, max_items=6)
        return prompt, options, correct_items

    if requested_type == "true_false":
        correct_text = str(correct_answer or "").strip()
        if correct_text not in {"正确", "错误"}:
            raise ValueError("判断题答案必须是正确或错误")
        return prompt, ["正确", "错误"], correct_text

    if requested_type == "fill_blank":
        correct_text = str(correct_answer or "").strip()
        if not correct_text:
            raise ValueError("填空题答案为空")
        if "____" not in prompt and "（" not in prompt:
            prompt = f"{prompt}\n请填写空缺处：____"
        return prompt, [], correct_text

    correct_text = str(correct_answer or raw_result.get("reference_answer") or "").strip()
    if not correct_text:
        raise ValueError("简答题标准答案为空")
    return prompt, [], correct_text


def _generate_question_with_model(
        *,
        question: str,
        reference_answer: str,
        question_type: str,
        model_mode: str | None,
        target_answer: str | None = None,
) -> tuple[str, list[str], Any]:
    """调用模型把原始问答润色成正式考试题。"""

    # 所有题型都先经过 LLM 命题，使题干、选项和标准答案看起来像真正的考试题。
    model = get_chat_model(model_mode)
    extra_requirement = ""
    if question_type == "true_false" and target_answer in {"正确", "错误"}:
        extra_requirement = f"本题必须设计成答案为“{target_answer}”的判断题。"
    response = model.invoke(
        [
            SystemMessage(content=prompt_manager.get("exam.question_generation.system")),
            HumanMessage(
                content=prompt_manager.render(
                    "exam.question_generation.user",
                    question_type_label=_question_type_label(question_type),
                    question=question,
                    reference_answer=reference_answer,
                    extra_requirement=extra_requirement,
                )
            ),
        ]
    )
    raw_result = _parse_model_json(response.content)
    return _validate_generated_question(raw_result, question_type)


def _build_fallback_conversation_question(
        # 强制后续参数必须使用关键字传入，避免调用方按位置传参导致含义混乱。
        *,
        # 当前用于命题的原始 QA 数据，包含内容和 metadata 来源信息。
        item: dict[str, Any],
        # 当前题源范围内的完整候选题池，用于选择题生成干扰项。
        candidates: list[dict[str, Any]],
        # 期望生成的题型，例如单选、多选、判断、填空或简答。
        question_type: str,
        # 当前题目专用随机生成器，用于保证兜底生成结果可复现。
        random_generator: random.Random,
        # 当前题目的最高分。
        max_score: float,
) -> dict[str, Any]:
    """模型生成失败时，用规则兜底生成考试题。"""

    # 读取原始 QA 的 metadata，后续用于提取题干、来源和结构化字段。
    metadata = item["metadata"]
    # 从 metadata 中读取原始问题文本，并去掉首尾空白作为默认题干。
    question = str(metadata.get("question") or "").strip()
    # 从原始内容中提取参考答案，作为客观题答案依据或简答题参考答案。
    reference_answer = _extract_reference_answer(item["content"], question)
    # 如果当前期望题型是判断题，则提前确定目标答案，避免判断题答案过于偏向同一侧。
    target_true_false_answer = (
        # 根据随机生成器选择本次判断题期望答案。
        _target_true_false_answer(random_generator)
        # 只有判断题才需要目标真假答案。
        if question_type == "true_false"
        # 非判断题不需要目标真假答案。
        else None
    )
    # 初始化最终题型，默认沿用期望题型，生成失败时可能降级为简答题。
    final_type = question_type
    # 初始化题干，默认直接使用原始问题。
    prompt = question
    # 初始化选项列表，简答题和填空题可能为空。
    options: list[str] = []
    # 初始化正确答案，默认使用参考答案。
    correct_answer: Any = reference_answer

    # 如果期望生成单选题，则尝试用规则构造一个单选题。
    if question_type == "single_choice":
        # 根据原始问题、参考答案和候选题池生成单选题题干、选项和正确答案。
        generated = _make_single_choice(question, reference_answer, candidates, random_generator)
        # 如果单选题生成成功，则使用生成结果覆盖默认题干、选项和答案。
        if generated:
            # 解包规则生成的题干、选项和正确答案。
            prompt, options, correct_answer = generated
        # 如果单选题生成失败，说明干扰项不足或规则无法构造。
        else:
            # 降级为简答题，保证本轮考试仍然可以出题。
            final_type = "short_answer"
    # 如果期望生成多选题，则尝试用规则构造一个多选题。
    elif question_type == "multiple_choice":
        # 根据原始问题、参考答案和候选题池生成多选题题干、选项和正确答案。
        generated = _make_multiple_choice(question, reference_answer, candidates, random_generator)
        # 如果多选题生成成功，则使用生成结果覆盖默认题干、选项和答案。
        if generated:
            # 解包规则生成的题干、选项和正确答案。
            prompt, options, correct_answer = generated
        # 如果多选题生成失败，说明可用正确项或干扰项不足。
        else:
            # 降级为简答题，避免因为客观题构造失败导致整题丢失。
            final_type = "short_answer"
    # 如果期望生成判断题，则直接按规则生成判断题。
    elif question_type == "true_false":
        # 根据原始问题、参考答案和目标真假答案生成判断题题干、选项和正确答案。
        prompt, options, correct_answer = _make_true_false(
            # 传入原始问题作为判断题基础题干。
            question,
            # 传入参考答案，用于构造正确或错误的判断陈述。
            reference_answer,
            # 传入当前题目随机生成器，控制判断题陈述扰动。
            random_generator,
            # 传入目标真假答案，控制本题最终答案为对或错。
            target_answer=target_true_false_answer,
        )
    # 如果期望生成填空题，则尝试从参考答案中挖空生成。
    elif question_type == "fill_blank":
        # 根据原始问题和参考答案生成填空题题干、选项和正确答案。
        generated = _make_fill_blank(question, reference_answer)
        # 如果填空题生成成功，则使用生成结果覆盖默认题干、选项和答案。
        if generated:
            # 解包规则生成的题干、选项和正确答案。
            prompt, options, correct_answer = generated
        # 如果填空题生成失败，说明参考答案不适合挖空。
        else:
            # 降级为简答题，保持题目仍然可答。
            final_type = "short_answer"

    # 对客观题选项和答案做最终展示格式整理，例如答案字母、选项顺序等。
    options, correct_answer = _prepare_objective_question_for_display(final_type, options, correct_answer)
    # 返回可直接写入考试题目表的结构化题目数据。
    return {
        # 保存原始问题编号，兼容不同来源的 question_id、qa_id 或 segment_id。
        "source_question_id": _optional_text(metadata.get("question_id") or metadata.get("qa_id") or metadata.get("segment_id")),
        # 保存原始文档编号，方便追溯题目来自哪个文件。
        "source_document_id": _optional_text(metadata.get("document_id")),
        # 保存原始文件名，兼容 source_file 和 source 两种 metadata 字段。
        "source_filename": _optional_text(metadata.get("source_file") or metadata.get("source")),
        # 保存原始页码，兼容 source_page、page_no 和 page 三种 metadata 字段。
        "source_page": _optional_int(metadata.get("source_page") or metadata.get("page_no") or metadata.get("page")),
        # 保存原始目录路径，方便按章节定位题目来源。
        "section_path": _optional_text(_metadata_section_path(metadata)),
        # 保存最终题型，可能等于期望题型，也可能因生成失败降级为简答题。
        "question_type": final_type,
        # 保存最终题干。
        "prompt": prompt,
        # 保存最终选项列表；主观题通常为空列表。
        "options": options,
        # 保存最终正确答案，客观题可能是选项标识，主观题通常是文本答案。
        "correct_answer": correct_answer,
        # 保存参考答案，用于后续阅卷分析和展示解析。
        "reference_answer": reference_answer,
        # 保存当前题目的最高分。
        "max_score": max_score,
    }


def _build_conversation_question(
        *,
        item: dict[str, Any],
        candidates: list[dict[str, Any]],
        question_type: str,
        random_generator: random.Random,
        max_score: float,
        model_mode: str | None = None,
        prefer_model: bool = True,
) -> dict[str, Any]:
    """先用模型把原始 QA 润色成正式考试题，失败时回退到规则生成。"""

    # 先构造兜底题目，保证模型调用失败时也能保存一条完整考试题记录。
    fallback_question = _build_fallback_conversation_question(
        item=item,
        candidates=candidates,
        question_type=question_type,
        random_generator=random_generator,
        max_score=max_score,
    )
    if not prefer_model:
        # 首题优先快速返回，避免用户点击“开始测评”后被 LLM 命题耗时卡住。
        logger.info(
            "[考试] 已使用快速规则生成题目 题型=%s 来源题目编号=%s",
            fallback_question["question_type"],
            fallback_question["source_question_id"],
        )
        return fallback_question
    try:
        prompt, options, correct_answer = _generate_question_with_model(
            question=str(item["metadata"].get("question") or "").strip(),
            reference_answer=fallback_question["reference_answer"],
            question_type=question_type,
            model_mode=model_mode,
            target_answer=(
                str(fallback_question["correct_answer"])
                if question_type == "true_false"
                else None
            ),
        )
        options, correct_answer = _prepare_objective_question_for_display(question_type, options, correct_answer)
        fallback_question["question_type"] = question_type
        fallback_question["prompt"] = prompt
        fallback_question["options"] = options
        fallback_question["correct_answer"] = correct_answer
        logger.info(
            "[考试] 题目润色完成 题型=%s 来源题目编号=%s 标准答案=%s 选项数=%s",
            fallback_question["question_type"],
            fallback_question["source_question_id"],
            fallback_question["correct_answer"],
            len(fallback_question["options"]),
        )
        return fallback_question
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
        logger.error(
            "[考试] 题目润色失败，已使用规则兜底 题型=%s 来源题目编号=%s 错误=%s",
            fallback_question["question_type"],
            fallback_question["source_question_id"],
            exc,
            exc_info=True,
        )
        return fallback_question


def _rule_answer_analysis(question: dict[str, Any], user_answer: str) -> ExamAnswerAnalysis:
    """模型评分失败时，用规则兜底生成分析结果。"""

    # 兜底规则只做基础精确匹配；正常评分统一由 _model_answer_analysis 交给模型完成。
    question_type = question["question_type"]
    correct_answer = _parse_json_field(question.get("correct_answer_json"), None)
    max_score = float(question["max_score"])
    clean_user_answer = user_answer.strip()

    if question_type == "multiple_choice":
        expected_set = {str(item).strip() for item in correct_answer or [] if str(item).strip()}
        actual_set = {item.strip() for item in re.split(r"[、,，;；\n]+", clean_user_answer) if item.strip()}
        is_correct = bool(expected_set) and actual_set == expected_set
    else:
        is_correct = clean_user_answer.lower() == str(correct_answer or "").strip().lower()

    score = max_score if is_correct else 0.0
    missing_points = [] if is_correct else [f"正确答案：{correct_answer}"]
    wrong_points = [] if is_correct else ([f"你的答案：{clean_user_answer or '未作答'}"])
    comment = "回答正确。" if is_correct else "回答不正确，建议对照正确答案复习。"
    return ExamAnswerAnalysis(
        is_correct=is_correct,
        score=score,
        max_score=max_score,
        correct_answer=correct_answer,
        reference_answer=question.get("reference_answer") or "",
        hit_points=[clean_user_answer] if is_correct and clean_user_answer else [],
        missing_points=missing_points,
        wrong_points=wrong_points,
        comment=comment,
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


def _list_field(raw_result: dict[str, Any], name: str) -> list[str]:
    """从模型 JSON 中读取字符串列表字段。"""

    value = raw_result.get(name) or []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _model_answer_analysis(question: dict[str, Any], user_answer: str, model_mode: str | None) -> ExamAnswerAnalysis:
    """调用模型统一分析所有题型答案，并润色正确答案和阅卷点评。"""

    max_score = float(question["max_score"])
    if not user_answer.strip():
        correct_answer = _parse_json_field(question.get("correct_answer_json"), None)
        return ExamAnswerAnalysis(
            is_correct=False,
            score=0,
            max_score=max_score,
            correct_answer=correct_answer,
            reference_answer=question.get("reference_answer") or "",
            missing_points=["未作答"],
            comment="该题未作答。",
        )

    try:
        correct_answer = _parse_json_field(question.get("correct_answer_json"), None)
        model = get_chat_model(model_mode)
        # 简答、填空、选择、判断都统一交给模型分析，这样可以处理同义表达、漏点和表述不完整。
        response = model.invoke(
            [
                SystemMessage(content=prompt_manager.get("exam.answer_grading.system")),
                HumanMessage(
                    content=prompt_manager.render(
                        "exam.answer_grading.user",
                        question_type_label=_question_type_label(question["question_type"]),
                        prompt=question["prompt"],
                        correct_answer=correct_answer,
                        reference_answer=question.get("reference_answer") or "",
                        user_answer=user_answer,
                        max_score=max_score,
                    )
                ),
            ]
        )
        raw_result = _parse_model_json(response.content)
        score = max(0.0, min(float(raw_result.get("score") or 0), max_score))
        polished_answer = raw_result.get("correct_answer")
        if polished_answer in (None, "", []):
            polished_answer = correct_answer
        return ExamAnswerAnalysis(
            is_correct=bool(raw_result.get("is_correct")) or score >= max_score * 0.8,
            score=score,
            max_score=max_score,
            correct_answer=polished_answer,
            reference_answer=question.get("reference_answer") or "",
            hit_points=_list_field(raw_result, "hit_points"),
            missing_points=_list_field(raw_result, "missing_points"),
            wrong_points=_list_field(raw_result, "wrong_points"),
            comment=str(raw_result.get("comment") or "").strip(),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
        logger.error("[考试] 模型阅卷失败，已使用规则兜底 题目编号=%s 错误=%s", question["exam_question_id"], exc, exc_info=True)
        fallback = _rule_answer_analysis(question, user_answer)
        fallback.comment = f"{fallback.comment}（模型阅卷失败，已使用规则兜底。）"
        return fallback


def _analysis_to_metadata(analysis: ExamAnswerAnalysis) -> dict[str, Any]:
    """把分析模型转换成可保存 JSON。"""

    return {
        "correct_answer": analysis.correct_answer,
        "hit_points": analysis.hit_points,
        "missing_points": analysis.missing_points,
        "wrong_points": analysis.wrong_points,
        "comment": analysis.comment,
    }


def _build_exam_question_rows(
        # 强制后续参数只能用关键字传入，避免调用方按位置传参导致含义混乱。
        *,
        # 当前考试会话编号，所有生成出来的题目都会写到这个会话下。
        session_id: str,
        # 本场考试已经抽中的原始 QA 列表，每个元素会对应生成一轮考试题。
        selected_items: list[dict[str, Any]],
        # 当前题源范围内的完整候选题池，用于生成选项、干扰项和上下文参考。
        candidates: list[dict[str, Any]],
        # 本场考试允许使用的题型列表，例如单选、多选、判断等。
        question_types: list[str],
        # 命题时使用的模型模式；为空时由下游按默认模型处理。
        model_mode: str | None,
        # 本场考试随机种子，用于保证题型选择和题目生成过程可复现。
        seed: int | None,
        # 每一轮题目的最高分，由总分按轮数平均计算得到。
        max_score: float,
        # 起始生成轮次；小于该轮次的题目会跳过，常用于后台只补后续题。
        start_round: int = 1,
        # 是否优先使用模型生成题目；为 False 时优先走规则快速生成。
        prefer_model: bool = True,
) -> None:
    """按轮次生成并保存考试题目，已存在的轮次会自动跳过。"""

    # 基于考试种子创建题型随机生成器，保证同一场考试的题型序列可复现。
    question_type_random = random.Random(f"{seed}:question_type")
    # 遍历本场考试抽中的原始 QA，并按轮次逐题生成考试题。
    for round_no, item in enumerate(selected_items, start=1):
        # 从允许题型中随机选择当前轮题型。
        question_type = question_type_random.choice(question_types)
        # 如果当前轮次小于起始轮次，或数据库里已存在该轮题目，则直接跳过。
        if round_no < start_round or _store().get_exam_question(session_id=session_id, round_no=round_no):
            # 跳过已生成或不在本次生成范围内的轮次，避免重复写入。
            continue
        # 基于考试种子和轮次创建当前题目的随机生成器，保证单题生成结果可复现。
        question_random = random.Random(f"{seed}:question:{round_no}")
        # 根据原始 QA、候选题池和题型生成当前轮考试题数据。
        question_data = _build_conversation_question(
            # 传入当前轮对应的原始 QA。
            item=item,
            # 传入完整候选题池，用于生成干扰项或补充上下文。
            candidates=candidates,
            # 传入当前轮随机选中的题型。
            question_type=question_type,
            # 传入当前题目专用随机生成器。
            random_generator=question_random,
            # 传入当前题目的最高分。
            max_score=max_score,
            # 传入模型模式，供模型命题或润色使用。
            model_mode=model_mode,
            # 传入是否优先使用模型生成的开关。
            prefer_model=prefer_model,
        )
        # 检查当前生成的多选题是否属于“所有选项都正确”的情况，避免连续出现全选题。
        if (
                # 判断当前题是否是答案覆盖全部选项的多选题。
                _is_all_select_multiple_choice(
                    # 传入当前生成题目的题型。
                    question_data["question_type"],
                    # 传入当前生成题目的选项列表。
                    question_data["options"],
                    # 传入当前生成题目的正确答案。
                    question_data["correct_answer"],
                )
                # 判断本轮之前的多选题是否也已经出现全选情况。
                and _previous_multiple_choice_questions_all_select(session_id, round_no)
        ):
            # 记录全选多选题过于集中的告警，方便后续观察题目质量。
            logger.warning(
                # 日志模板记录会话编号和轮次，便于定位具体考试题。
                "[考试] 多选题全选过于集中，改用规则重新生成 会话编号=%s 轮次=%s",
                # 写入当前考试会话编号。
                session_id,
                # 写入当前生成轮次。
                round_no,
            )
            # 使用规则兜底方式重新生成一道多选题，降低连续全选题的概率。
            question_data = _build_fallback_conversation_question(
                # 传入当前轮对应的原始 QA。
                item=item,
                # 传入完整候选题池，用于兜底生成干扰项。
                candidates=candidates,
                # 固定按多选题重新生成，保持当前题型语义一致。
                question_type="multiple_choice",
                # 使用独立 fallback 随机种子，避免重新生成结果和首次生成完全一致。
                random_generator=random.Random(f"{seed}:question:{round_no}:fallback"),
                # 传入当前题目的最高分。
                max_score=max_score,
            )
        # 保存题目时捕获唯一键冲突，兼容接口线程和后台任务并发生成同一轮题的情况。
        try:
            # 将当前轮题目写入数据库，question_data 会展开为题干、选项、答案、解析等字段。
            _store().add_exam_question(session_id=session_id, round_no=round_no, **question_data)
        # 如果数据库提示唯一键冲突，说明该题已由其他任务先写入。
        except IntegrityErrorTypes:
            # 记录重复写入被跳过的日志，不再抛错影响整场考试生成流程。
            logger.info("[考试] 题目已由其他任务生成，跳过重复写入 会话编号=%s 轮次=%s", session_id, round_no)


def _rebuild_exam_context_from_session(
        # 考试会话记录，通常来自数据库实体转 dict 后的结果。
        session: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int | None, float]:
    """从会话记录重新构造可补题的上下文。"""

    # 从会话元数据 JSON 中解析出字典，主要用于恢复创建考试时保存的随机种子。
    metadata = _parse_json_field(session.get("metadata_json"), {})
    # 按会话记录中的题源范围重新从 Qdrant 读取候选题。
    candidates = _scroll_candidate_questions(
        # 使用会话保存的 Collection 名称，保证补题仍然来自原知识库。
        collection_name=session.get("collection_name"),
        # 使用会话保存的文档编号，保证补题仍然限定在原文件范围内。
        document_id=session.get("document_id"),
        # 使用会话保存的目录路径，保证补题仍然限定在原章节范围内。
        section_path=session.get("section_path"),
    )
    # 如果原题源已经没有候选题，说明无法继续补题。
    if not candidates:
        # 抛出运行时异常，让上层返回考试题源不可用的错误。
        raise RuntimeError("当前考试题源已无可用候选题")

    # 从会话元数据中读取创建考试时保存的随机种子。
    seed = metadata.get("seed")
    # 尝试把随机种子转换成整数，兼容 JSON 中可能保存成字符串的情况。
    try:
        # 有种子时转成 int；没有种子时保持 None。
        clean_seed = int(seed) if seed is not None else None
    # 如果种子类型异常或内容无法转成整数，则忽略该种子。
    except (TypeError, ValueError):
        # 使用 None 作为兜底种子，避免补题流程因脏数据中断。
        clean_seed = None

    # 使用恢复后的种子创建随机生成器，尽量复现原始抽题顺序。
    shuffle_random_generator = random.Random(clean_seed)
    # 按同一个随机种子打乱候选题，保证补题时选中的题目顺序与开考时一致。
    shuffle_random_generator.shuffle(candidates)
    # 按会话轮数截取本场考试实际使用的候选题列表。
    selected_items = candidates[:int(session["round_count"])]
    # 从会话记录中解析题型列表；解析失败时使用全部支持题型。
    question_types = _parse_json_field(session.get("question_types_json"), ALL_QUESTION_TYPES)
    # 过滤掉已经不被系统支持的题型；如果过滤后为空，则退回全部支持题型。
    question_types = [item for item in question_types if item in ALL_QUESTION_TYPES] or ALL_QUESTION_TYPES
    # 重新计算每轮最高分，总分 100 分按会话轮数平均分配并保留 4 位小数。
    max_score = round(100 / int(session["round_count"]), 4)
    # 返回补题所需的完整候选题、已选题、题型、随机种子和单题最高分。
    return candidates, selected_items, question_types, clean_seed, max_score


def _ensure_exam_question_ready(session: dict[str, Any], round_no: int) -> dict[str, Any]:
    """确保指定轮次题目已经生成，后台未完成时同步补题。"""

    question = _store().get_exam_question(session_id=session["session_id"], round_no=round_no)
    if question is not None:
        return question

    candidates, selected_items, question_types, seed, max_score = _rebuild_exam_context_from_session(session)
    if len(selected_items) < round_no:
        raise RuntimeError(f"当前题源只有 {len(selected_items)} 道题，不足第 {round_no} 轮")

    logger.warning("[考试] 当前轮题目未预生成，开始同步补题 会话编号=%s 轮次=%s", session["session_id"], round_no)
    _build_exam_question_rows(
        session_id=session["session_id"],
        selected_items=selected_items,
        candidates=candidates,
        question_types=question_types,
        model_mode=session.get("model_mode"),
        seed=seed,
        max_score=max_score,
        start_round=round_no,
    )
    question = _store().get_exam_question(session_id=session["session_id"], round_no=round_no)
    if question is None:
        raise RuntimeError(f"第 {round_no} 轮题目生成失败")
    return question


def _build_remaining_exam_questions_background(
        *,
        session_id: str,
        selected_items: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        question_types: list[str],
        model_mode: str | None,
        seed: int | None,
        max_score: float,
) -> None:
    """后台补齐第 2 轮之后的考试题，避免开始测评等待所有题目生成。"""

    start_time = time.perf_counter()
    try:
        _build_exam_question_rows(
            session_id=session_id,
            selected_items=selected_items,
            candidates=candidates,
            question_types=question_types,
            model_mode=model_mode,
            seed=seed,
            max_score=max_score,
            start_round=2,
        )
        logger.info(
            "[考试] 后台题目生成完成 会话编号=%s 总轮数=%s 耗时毫秒=%.2f",
            session_id,
            len(selected_items),
            _elapsed_ms(start_time),
        )
    except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
            OSError,
    ) as exc:
        logger.error("[考试] 后台题目生成失败 会话编号=%s 错误=%s", session_id, exc, exc_info=True)


def _store() -> ExamRepository:
    """获取考试仓储外观。"""

    return get_exam_repository()


@router.get("/sections", response_model=ExamSectionsResponse)
def list_exam_sections(
        collection_name: str | None = None,
        document_id: str | None = None,
) -> ExamSectionsResponse:
    """查询某个向量库/文件下可用于考试的目录路径。"""

    final_collection_name = normalize_qdrant_collection_name(collection_name)
    candidates = _scroll_candidate_questions(collection_name=final_collection_name, document_id=document_id)
    section_counts: dict[str, int] = {}
    for item in candidates:
        metadata = item["metadata"]
        # 前端目录只展示第一层，完整子目录仍保留在题目来源里。
        section = _metadata_first_level_section(metadata)
        if not section:
            continue
        section_counts[section] = section_counts.get(section, 0) + 1

    sections = [
        ExamSectionResponse(section_path=section_path, question_count=count)
        for section_path, count in sorted(section_counts.items(), key=lambda item: _section_sort_key(item[0]))
    ]
    return ExamSectionsResponse(collection_name=final_collection_name, document_id=document_id, sections=sections)


# 注册开始考试接口，前端调用 POST /sessions 时会进入这个方法。
@router.post("/sessions", response_model=ExamStartResponse)
def start_exam_session(
        # 开始考试请求体，包含用户、题源、轮数、题型、随机种子等参数。
        request: ExamStartRequest,
        # FastAPI 后台任务对象，用于把后续题目生成放到接口返回之后继续执行。
        background_tasks: BackgroundTasks,
) -> ExamStartResponse:
    """开始一场对话式随机考试。"""

    # 记录接口开始时间，用于统计创建考试会话的总耗时。
    start_time = time.perf_counter()
    # 规范化前端传入的 Qdrant Collection 名称，确保后续题源检索使用有效集合。
    final_collection_name = normalize_qdrant_collection_name(request.collection_name)
    # 过滤前端传入的题型，只保留系统支持的题型；如果过滤后为空，则使用全部题型。
    question_types = [item for item in request.question_types if item in ALL_QUESTION_TYPES] or ALL_QUESTION_TYPES
    # 确定本场考试随机种子；用户传了 seed 就复用，否则生成一个系统随机种子。
    exam_seed = request.seed if request.seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    # 从 Qdrant 中按 Collection、文件编号和目录范围读取可用于考试的结构化问答候选题。
    candidates = _scroll_candidate_questions(
        # 指定要读取的 Qdrant Collection。
        collection_name=final_collection_name,
        # 指定题源文件编号；为空时表示不按文件过滤。
        document_id=request.document_id,
        # 指定题源目录路径；为空时表示不按目录过滤。
        section_path=request.section_path,
    )
    # 如果题源范围内没有任何候选题，则直接返回 404 提示前端无法开始考试。
    if not candidates:
        # 抛出 HTTP 404，说明不是系统异常，而是当前题源确实没有可用结构化问答题。
        raise HTTPException(status_code=404, detail="当前题源范围内没有可用于考试的结构化问答题")

    # 使用本场考试种子创建随机生成器，保证同一个 seed 可以复现同样的抽题顺序。
    shuffle_random_generator = random.Random(exam_seed)
    # 对候选题进行原地打乱，后续按轮数从前面截取作为本场考试题目。
    shuffle_random_generator.shuffle(candidates)
    # 先随机抽出本场考试需要的原始 QA，再逐题命题并持久化。
    # 截取本场考试需要的候选题数量，数量由前端传入的 round_count 决定。
    selected_items = candidates[:request.round_count]
    # 如果候选题数量小于考试轮数，则不能完整生成本场考试。
    if len(selected_items) < request.round_count:
        # 抛出 HTTP 400，提示前端当前题源题量不足，需要减少轮数或扩大题源范围。
        raise HTTPException(status_code=400, detail=f"当前题源只有 {len(selected_items)} 道题，不足 {request.round_count} 轮")

    # 如果指定了文件编号，则读取文件信息，用于把文件名写入考试会话；否则不绑定文件名。
    document = _store().get_document(request.document_id) if request.document_id else None
    # 计算每轮题目的最高分，总分 100 分按考试轮数平均分配并保留 4 位小数。
    max_score = round(100 / request.round_count, 4)
    # 创建考试会话主记录，保存本场考试的用户、题源、轮数、题型和模型配置。
    session = _store().create_exam_session(
        # 保存考试所属用户编号。
        user_id=request.user_id,
        # 保存前端传入的考试标题。
        title=request.title,
        # 保存最终使用的 Qdrant Collection 名称。
        collection_name=final_collection_name,
        # 保存题源文件编号，方便后续按文件追溯题目来源。
        document_id=request.document_id,
        # 保存题源文件名；未指定文件时保持为空。
        filename=document.get("filename") if document else None,
        # 保存题源目录路径，方便后续按目录追溯题目来源。
        section_path=request.section_path,
        # 保存本场考试总轮数。
        round_count=request.round_count,
        # 保存本场考试允许生成的题型列表。
        question_types=question_types,
        # 保存命题和分析使用的模型模式。
        model_mode=request.model_mode,
        # 保存随机种子元数据，便于复现抽题顺序和区分用户传入种子。
        metadata={"seed": exam_seed, "user_seed": request.seed},
    )

    # 第一轮用规则快速生成，接口可以尽快把第一题返回给前端；后续题目仍交给后台模型润色。
    # 立即生成并持久化第一题，保证开始考试接口能直接返回当前题目。
    _build_exam_question_rows(
        # 指定题目要写入的考试会话编号。
        session_id=session["session_id"],
        # 只传入第一道候选题，用于快速生成第一轮题目。
        selected_items=selected_items[:1],
        # 传入完整候选题池，便于命题时构造干扰项或参考上下文。
        candidates=candidates,
        # 传入允许题型，控制第一题的题型选择范围。
        question_types=question_types,
        # 传入模型模式；虽然首题 prefer_model=False，但保持参数完整。
        model_mode=request.model_mode,
        # 传入考试随机种子，保证题型和选项生成可复现。
        seed=exam_seed,
        # 传入每轮最高分，用于写入题目分值。
        max_score=max_score,
        # 指定从第 1 轮开始生成。
        start_round=1,
        # 首题不优先调用模型，使用规则快速生成以降低接口等待时间。
        prefer_model=False,
    )
    # 第 2 轮之后放到后台继续生成，减少“开始测评”的等待时间。
    # 注册后台任务，在接口返回后继续生成剩余考试题目。
    background_tasks.add_task(
        # 后台任务函数，负责生成并保存第 2 轮及之后的题目。
        _build_remaining_exam_questions_background,
        # 传入考试会话编号，确保后台题目写入同一场考试。
        session_id=session["session_id"],
        # 传入本场考试已抽中的全部候选题，后台会跳过第一题继续生成。
        selected_items=selected_items,
        # 传入完整候选题池，便于后台命题时构造干扰项或参考上下文。
        candidates=candidates,
        # 传入允许题型，控制后续题目的题型选择范围。
        question_types=question_types,
        # 传入模型模式，控制后台模型润色或命题使用哪个模型配置。
        model_mode=request.model_mode,
        # 传入考试随机种子，保证后台生成过程和首题使用同一随机基础。
        seed=exam_seed,
        # 传入每轮最高分，用于后续题目分值写入。
        max_score=max_score,
    )

    # 重新读取考试会话，确认主记录已经成功落库并带上最新状态。
    refreshed_session = _store().get_exam_session(session["session_id"])
    # 读取第一轮题目，作为开始考试接口需要立即返回给前端的当前题。
    current_question = _store().get_exam_question(session_id=session["session_id"], round_no=1)
    # 如果会话或首题读取失败，说明创建流程出现异常，需要返回服务端错误。
    if refreshed_session is None or current_question is None:
        # 抛出 HTTP 500，提示考试会话创建后无法读取，属于服务端持久化异常。
        raise HTTPException(status_code=500, detail="考试会话创建后读取失败")

    # 记录考试开始成功日志，包含题源范围、轮数、首题返回状态和接口耗时。
    logger.info(
        # 日志模板保留 Collection 字段名，其余状态描述使用中文，便于排查考试创建链路。
        "[考试] 对话式考试开始 会话编号=%s Collection=%s 文件编号=%s 目录=%s 轮数=%s 首题已返回=true 耗时毫秒=%.2f",
        # 写入本场考试会话编号。
        session["session_id"],
        # 写入本场考试使用的 Qdrant Collection 名称。
        final_collection_name,
        # 写入题源文件编号。
        request.document_id,
        # 写入题源目录路径。
        request.section_path,
        # 写入本场考试轮数。
        request.round_count,
        # 写入从接口开始到准备完成的耗时毫秒数。
        _elapsed_ms(start_time),
    )
    # 返回考试会话摘要和第一轮题目，前端拿到后即可进入答题流程。
    return ExamStartResponse(session=_session_summary(refreshed_session), current_question=_question_from_row(current_question))


@router.post("/sessions/{session_id}/answer", response_model=ExamAnswerResponse)
def answer_exam_session(session_id: str, request: ExamAnswerRequest) -> ExamAnswerResponse:
    """提交当前轮答案，并返回分析和下一题。"""

    session = _store().get_exam_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="考试会话不存在")
    if session["status"] == "completed":
        raise HTTPException(status_code=400, detail="考试已完成，不能继续提交答案")

    current_round = int(session.get("answered_count") or 0) + 1
    try:
        question = _ensure_exam_question_ready(session, current_round)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if question["status"] == "answered":
        raise HTTPException(status_code=400, detail="当前轮题目已经作答")

    user_answer = _normalize_answer_value_for_question(question, request.answer)
    # 所有题型都走同一套模型阅卷逻辑，模型失败时内部自动回退到规则评分。
    analysis = _model_answer_analysis(question, user_answer, session.get("model_mode"))

    answered_question = _store().answer_exam_question(
        session_id=session_id,
        exam_question_id=question["exam_question_id"],
        user_answer=user_answer,
        is_correct=analysis.is_correct,
        score=analysis.score,
        analysis=_analysis_to_metadata(analysis),
    )
    refreshed_session = _store().get_exam_session(session_id)
    if refreshed_session is None:
        raise HTTPException(status_code=500, detail="考试会话更新后读取失败")

    next_question = None
    if refreshed_session["status"] != "completed":
        next_round = int(refreshed_session["answered_count"]) + 1
        try:
            next_row = _ensure_exam_question_ready(refreshed_session, next_round)
            next_question = _question_from_row(next_row)
        except RuntimeError as exc:
            logger.error("[考试] 下一轮题目生成失败 会话编号=%s 轮次=%s 错误=%s", session_id, next_round, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"下一轮题目生成失败：{exc}") from exc

    logger.info(
        "[考试] 单轮作答完成 会话编号=%s 轮次=%s 题型=%s 得分=%.2f 状态=%s",
        session_id,
        current_round,
        question["question_type"],
        analysis.score,
        refreshed_session["status"],
    )
    return ExamAnswerResponse(
        session=_session_summary(refreshed_session),
        answered_question=_question_from_row(answered_question),
        analysis=analysis,
        next_question=next_question,
    )


@router.get("/sessions", response_model=ExamHistoryListResponse)
def list_exam_sessions(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=10, ge=1, le=50),
        user_id: str | None = None,
        keyword: str | None = None,
) -> ExamHistoryListResponse:
    """分页查询考试历史记录。"""

    sessions, total = _store().list_exam_sessions(page=page, page_size=page_size, user_id=user_id, keyword=keyword)
    return ExamHistoryListResponse(
        items=[_session_summary(item) for item in sessions],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/sessions/{session_id}", response_model=ExamSessionDeleteResponse)
def delete_exam_session(session_id: str) -> ExamSessionDeleteResponse:
    """删除考试历史记录及其题目明细。"""

    deleted = _store().delete_exam_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="考试会话不存在")
    logger.info("[考试] 历史记录删除完成 会话编号=%s", session_id)
    return ExamSessionDeleteResponse(status="deleted", session_id=session_id)


@router.get("/sessions/{session_id}", response_model=ExamSessionDetailResponse)
def get_exam_session_detail(session_id: str) -> ExamSessionDetailResponse:
    """查询考试历史详情，包含题目、用户答案和分析。"""

    session = _store().get_exam_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="考试会话不存在")
    questions = [
        ExamQuestionRecord(
            question=_question_from_row(row),
            user_answer=row.get("user_answer"),
            analysis=_analysis_from_row(row),
            answered_at=_format_response_time(row.get("answered_at")),
        )
        for row in _store().list_exam_questions(session_id)
    ]
    return ExamSessionDetailResponse(session=_session_summary(session), questions=questions)
