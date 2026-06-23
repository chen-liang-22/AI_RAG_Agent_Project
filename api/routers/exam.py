"""对话式考试路由。

核心流程：
1. 从 Qdrant 中按向量库、文件和一级目录筛选结构化 QA 题源；
2. 开始考试时随机抽题，并优先调用 LLM 把原始 QA 润色成正式试题；
3. 用户逐轮提交答案，后端用 LLM 分析得分、正确答案、命中点和遗漏点；
4. 会话、题目、用户答案和分析结果全部写入业务数据库，供历史记录查看。
"""

import json
import random
import re
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient, models

from api.common_services import _get_knowledge_store
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
    ExamSessionDetailResponse,
    ExamSessionSummary,
    ExamStartRequest,
    ExamStartResponse,
)
from model.factory import get_chat_model
from rag.knowledge_store import KnowledgeStore
from utils.database_connection import IntegrityErrorTypes
from utils.logger_handler import logger
from utils.qdrant_options import get_qdrant_client_options, normalize_qdrant_collection_name

router = APIRouter()

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


def _metadata_section_path(metadata: dict[str, Any]) -> str:
    """从题目 metadata 中读取完整目录路径。"""

    return str(metadata.get("section_path") or metadata.get("heading_path") or metadata.get("category") or "").strip()


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
    return actual == expected or actual.startswith(f"{expected}{SECTION_PATH_SEPARATOR}")


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
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row.get("completed_at"),
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
            SystemMessage(
                content=(
                    "你是资深中文考试命题老师，负责把知识库问答改写成正式考试题。"
                    "必须只依据给定原始问题和参考答案命题，不要补充资料外的新事实。"
                    "题干要像正式试题，不能像客服问答或知识片段。"
                    "正确答案要改写成面向考生的标准答案，不能直接照搬原文。"
                    "只返回 JSON，不要返回 Markdown。"
                )
            ),
            HumanMessage(
                content=(
                    f"目标题型：{_question_type_label(question_type)}\n\n"
                    f"原始问题：{question}\n\n"
                    f"参考答案：{reference_answer}\n\n"
                    "请生成一题正式考试题，并严格返回 JSON：\n"
                    "{"
                    '"prompt":"正式题干",'
                    '"options":["选项A","选项B","选项C","选项D"],'
                    '"correct_answer":"标准答案或正确选项；多选题为字符串数组"'
                    "}\n"
                    "要求："
                    "单选题提供 4 个选项且只有 1 个正确答案；"
                    "多选题提供 4 到 6 个选项，正确答案至少 2 个；"
                    "判断题 options 固定为 [\"正确\",\"错误\"]，correct_answer 只能是 正确 或 错误；"
                    "填空题题干必须自然出现空缺，不要把原始问句原样当题干；"
                    "简答题不需要 options。"
                    f"{extra_requirement}"
                )
            ),
        ]
    )
    raw_result = _parse_model_json(response.content)
    return _validate_generated_question(raw_result, question_type)


def _build_fallback_conversation_question(
        *,
        item: dict[str, Any],
        candidates: list[dict[str, Any]],
        question_type: str,
        random_generator: random.Random,
        max_score: float,
) -> dict[str, Any]:
    """模型生成失败时，用规则兜底生成考试题。"""

    metadata = item["metadata"]
    question = str(metadata.get("question") or "").strip()
    reference_answer = _extract_reference_answer(item["content"], question)
    target_true_false_answer = (
        _target_true_false_answer(random_generator)
        if question_type == "true_false"
        else None
    )
    final_type = question_type
    prompt = question
    options: list[str] = []
    correct_answer: Any = reference_answer

    if question_type == "single_choice":
        generated = _make_single_choice(question, reference_answer, candidates, random_generator)
        if generated:
            prompt, options, correct_answer = generated
        else:
            final_type = "short_answer"
    elif question_type == "multiple_choice":
        generated = _make_multiple_choice(question, reference_answer, candidates, random_generator)
        if generated:
            prompt, options, correct_answer = generated
        else:
            final_type = "short_answer"
    elif question_type == "true_false":
        prompt, options, correct_answer = _make_true_false(
            question,
            reference_answer,
            random_generator,
            target_answer=target_true_false_answer,
        )
    elif question_type == "fill_blank":
        generated = _make_fill_blank(question, reference_answer)
        if generated:
            prompt, options, correct_answer = generated
        else:
            final_type = "short_answer"

    options, correct_answer = _prepare_objective_question_for_display(final_type, options, correct_answer)
    return {
        "source_question_id": _optional_text(metadata.get("question_id") or metadata.get("qa_id") or metadata.get("segment_id")),
        "source_document_id": _optional_text(metadata.get("document_id")),
        "source_filename": _optional_text(metadata.get("source_file") or metadata.get("source")),
        "source_page": _optional_int(metadata.get("source_page") or metadata.get("page_no") or metadata.get("page")),
        "section_path": _optional_text(_metadata_section_path(metadata)),
        "question_type": final_type,
        "prompt": prompt,
        "options": options,
        "correct_answer": correct_answer,
        "reference_answer": reference_answer,
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
                SystemMessage(
                    content=(
                        "你是严格但公正的中文考试阅卷老师。"
                        "必须根据题目、题型、标准答案和参考答案分析用户答案。"
                        "即使是选择题、判断题和填空题，也要给出自然、专业的阅卷点评。"
                        "correct_answer 必须润色成适合展示给考生的标准答案，不要机械照搬原始知识库片段。"
                        "评分要稳健：同义表达、合理缩写和顺序差异可以酌情判对；明显漏点或错选要扣分。"
                        "只返回 JSON，不要返回 Markdown。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"题型：{_question_type_label(question['question_type'])}\n\n"
                        f"题目：{question['prompt']}\n\n"
                        f"系统保存的标准答案：{correct_answer}\n\n"
                        f"参考答案：{question.get('reference_answer') or ''}\n\n"
                        f"用户答案：{user_answer}\n\n"
                        f"本题满分：{max_score}\n\n"
                        "请返回 JSON："
                        '{"score":0,"is_correct":false,"correct_answer":"润色后的标准答案",'
                        '"hit_points":[],"missing_points":[],"wrong_points":[],"comment":""}'
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
        *,
        session_id: str,
        selected_items: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        question_types: list[str],
        model_mode: str | None,
        seed: int | None,
        max_score: float,
        start_round: int = 1,
        prefer_model: bool = True,
) -> None:
    """按轮次生成并保存考试题目，已存在的轮次会自动跳过。"""

    question_type_random = random.Random(f"{seed}:question_type")
    for round_no, item in enumerate(selected_items, start=1):
        question_type = question_type_random.choice(question_types)
        if round_no < start_round or _store().get_exam_question(session_id=session_id, round_no=round_no):
            continue
        question_random = random.Random(f"{seed}:question:{round_no}")
        question_data = _build_conversation_question(
            item=item,
            candidates=candidates,
            question_type=question_type,
            random_generator=question_random,
            max_score=max_score,
            model_mode=model_mode,
            prefer_model=prefer_model,
        )
        if (
                _is_all_select_multiple_choice(
                    question_data["question_type"],
                    question_data["options"],
                    question_data["correct_answer"],
                )
                and _previous_multiple_choice_questions_all_select(session_id, round_no)
        ):
            logger.warning(
                "[考试] 多选题全选过于集中，改用规则重新生成 会话编号=%s 轮次=%s",
                session_id,
                round_no,
            )
            question_data = _build_fallback_conversation_question(
                item=item,
                candidates=candidates,
                question_type="multiple_choice",
                random_generator=random.Random(f"{seed}:question:{round_no}:fallback"),
                max_score=max_score,
            )
        try:
            _store().add_exam_question(session_id=session_id, round_no=round_no, **question_data)
        except IntegrityErrorTypes:
            logger.info("[考试] 题目已由其他任务生成，跳过重复写入 会话编号=%s 轮次=%s", session_id, round_no)


def _rebuild_exam_context_from_session(
        session: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int | None, float]:
    """从会话记录重新构造可补题的上下文。"""

    metadata = _parse_json_field(session.get("metadata_json"), {})
    candidates = _scroll_candidate_questions(
        collection_name=session.get("collection_name"),
        document_id=session.get("document_id"),
        section_path=session.get("section_path"),
    )
    if not candidates:
        raise RuntimeError("当前考试题源已无可用候选题")

    seed = metadata.get("seed")
    try:
        clean_seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        clean_seed = None

    shuffle_random_generator = random.Random(clean_seed)
    shuffle_random_generator.shuffle(candidates)
    selected_items = candidates[:int(session["round_count"])]
    question_types = _parse_json_field(session.get("question_types_json"), ALL_QUESTION_TYPES)
    question_types = [item for item in question_types if item in ALL_QUESTION_TYPES] or ALL_QUESTION_TYPES
    max_score = round(100 / int(session["round_count"]), 4)
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


def _store() -> KnowledgeStore:
    """获取知识库元数据仓库。"""

    return _get_knowledge_store()


@router.get("/exam/sections", response_model=ExamSectionsResponse)
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
        section = _first_level_section_path(_metadata_section_path(metadata))
        if not section:
            continue
        section_counts[section] = section_counts.get(section, 0) + 1

    sections = [
        ExamSectionResponse(section_path=section_path, question_count=count)
        for section_path, count in sorted(section_counts.items(), key=lambda item: _section_sort_key(item[0]))
    ]
    return ExamSectionsResponse(collection_name=final_collection_name, document_id=document_id, sections=sections)


@router.post("/exam/sessions", response_model=ExamStartResponse)
def start_exam_session(request: ExamStartRequest, background_tasks: BackgroundTasks) -> ExamStartResponse:
    """开始一场对话式随机考试。"""

    start_time = time.perf_counter()
    final_collection_name = normalize_qdrant_collection_name(request.collection_name)
    question_types = [item for item in request.question_types if item in ALL_QUESTION_TYPES] or ALL_QUESTION_TYPES
    exam_seed = request.seed if request.seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    candidates = _scroll_candidate_questions(
        collection_name=final_collection_name,
        document_id=request.document_id,
        section_path=request.section_path,
    )
    if not candidates:
        raise HTTPException(status_code=404, detail="当前题源范围内没有可用于考试的结构化问答题")

    shuffle_random_generator = random.Random(exam_seed)
    shuffle_random_generator.shuffle(candidates)
    # 先随机抽出本场考试需要的原始 QA，再逐题命题并持久化。
    selected_items = candidates[:request.round_count]
    if len(selected_items) < request.round_count:
        raise HTTPException(status_code=400, detail=f"当前题源只有 {len(selected_items)} 道题，不足 {request.round_count} 轮")

    document = _store().get_document(request.document_id) if request.document_id else None
    max_score = round(100 / request.round_count, 4)
    session = _store().create_exam_session(
        user_id=request.user_id,
        title=request.title,
        collection_name=final_collection_name,
        document_id=request.document_id,
        filename=document.get("filename") if document else None,
        section_path=request.section_path,
        round_count=request.round_count,
        question_types=question_types,
        model_mode=request.model_mode,
        metadata={"seed": exam_seed, "user_seed": request.seed},
    )

    # 第一轮用规则快速生成，接口可以尽快把第一题返回给前端；后续题目仍交给后台模型润色。
    _build_exam_question_rows(
        session_id=session["session_id"],
        selected_items=selected_items[:1],
        candidates=candidates,
        question_types=question_types,
        model_mode=request.model_mode,
        seed=exam_seed,
        max_score=max_score,
        start_round=1,
        prefer_model=False,
    )
    # 第 2 轮之后放到后台继续生成，减少“开始测评”的等待时间。
    background_tasks.add_task(
        _build_remaining_exam_questions_background,
        session_id=session["session_id"],
        selected_items=selected_items,
        candidates=candidates,
        question_types=question_types,
        model_mode=request.model_mode,
        seed=exam_seed,
        max_score=max_score,
    )

    refreshed_session = _store().get_exam_session(session["session_id"])
    current_question = _store().get_exam_question(session_id=session["session_id"], round_no=1)
    if refreshed_session is None or current_question is None:
        raise HTTPException(status_code=500, detail="考试会话创建后读取失败")

    logger.info(
        "[考试] 对话式考试开始 会话编号=%s Collection=%s 文件编号=%s 目录=%s 轮数=%s 首题已返回=true 耗时毫秒=%.2f",
        session["session_id"],
        final_collection_name,
        request.document_id,
        request.section_path,
        request.round_count,
        _elapsed_ms(start_time),
    )
    return ExamStartResponse(session=_session_summary(refreshed_session), current_question=_question_from_row(current_question))


@router.post("/exam/sessions/{session_id}/answer", response_model=ExamAnswerResponse)
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


@router.get("/exam/sessions", response_model=ExamHistoryListResponse)
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


@router.get("/exam/sessions/{session_id}", response_model=ExamSessionDetailResponse)
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
            answered_at=row.get("answered_at"),
        )
        for row in _store().list_exam_questions(session_id)
    ]
    return ExamSessionDetailResponse(session=_session_summary(session), questions=questions)
