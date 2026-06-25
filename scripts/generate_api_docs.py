"""生成当前项目接口文档。

脚本直接读取 FastAPI OpenAPI 元数据，并补充 schema 源码中的中文字段注释。
这样后续新增路由或 DTO 后，可以重新运行脚本刷新 docs/接口文档.md。
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.routing import APIRoute

from api.main import app


OUTPUT_PATH = PROJECT_ROOT / "docs" / "接口文档.md"
SCHEMA_SOURCE_FILES = [
    PROJECT_ROOT / "api" / "schemas.py",
    PROJECT_ROOT / "api" / "exam_schemas.py",
    PROJECT_ROOT / "training" / "schemas.py",
]

GROUP_TITLES = {
    "health": "健康检查",
    "chat": "聊天和会话",
    "knowledge": "知识库文件管理",
    "dictionaries": "系统字典",
    "exam": "知识掌握度测评",
    "training": "销售训练",
}

GROUP_ORDER = ["health", "chat", "knowledge", "dictionaries", "exam", "training"]

STREAM_EVENT_DESCRIPTIONS = {
    "/chat/stream": [
        ("meta", "流开始元信息，包含 conversation_id、模型档位、模型名称和 collection。"),
        ("metric", "首字耗时指标，当前包含 first_token_ms。"),
        ("chunk", "模型回答增量文本，data.content 是本次新增内容。"),
        ("done", "流式回答完成，包含 conversation_id、模型信息、first_token_ms、total_ms。"),
        ("error", "流式处理失败，data.error 是错误信息。"),
    ],
    "/training/sessions/{session_id}/turns": [
        ("retrieval_done", "本轮训练检索完成，包含 retrieved_chunk_ids 和 evidence。"),
        ("customer_delta", "AI 客户回复增量文本，data.content 是本次新增内容。"),
        ("stage_decision", "阶段和会话状态，包含 stage_status、session_status。"),
        ("turn_done", "本轮训练完成，data 是 TrainingTurnResponse 结构。"),
        ("error", "训练流式处理失败，data.error 是错误信息。"),
    ],
}


def main() -> None:
    """生成 Markdown 接口文档文件。"""

    schema = app.openapi()
    field_comments = collect_field_comments()
    route_modules = collect_route_modules()
    lines: list[str] = []

    lines.extend(render_header(schema))
    lines.extend(render_route_overview(schema, route_modules))
    lines.extend(render_api_details(schema, field_comments, route_modules))
    lines.extend(render_schema_appendix(schema, field_comments))

    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT_PATH}")


def collect_route_modules() -> dict[tuple[str, str], str]:
    """收集每个路由对应的 Python 模块，用于给接口分组。"""

    route_modules: dict[tuple[str, str], str] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        endpoint_module = inspect.getmodule(route.endpoint)
        module_name = endpoint_module.__name__ if endpoint_module else ""
        for method in route.methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            route_modules[(route.path, method.lower())] = module_name
    return route_modules


def collect_field_comments() -> dict[str, dict[str, str]]:
    """从 Pydantic schema 源码里提取字段行尾中文注释。"""

    comments: dict[str, dict[str, str]] = {}
    class_name: str | None = None
    pending_field: str | None = None
    class_pattern = re.compile(r"^class\s+(\w+)\(")
    field_pattern = re.compile(r"^\s{4}(\w+)\s*:\s*.*?(?:#\s*(.+))?$")

    for source_file in SCHEMA_SOURCE_FILES:
        for raw_line in source_file.read_text(encoding="utf-8").splitlines():
            class_match = class_pattern.match(raw_line)
            if class_match:
                class_name = class_match.group(1)
                comments.setdefault(class_name, {})
                pending_field = None
                continue

            if class_name is None:
                continue
            if raw_line and not raw_line.startswith(" "):
                pending_field = None
                continue

            field_match = field_pattern.match(raw_line)
            if field_match and not raw_line.lstrip().startswith("#"):
                pending_field = field_match.group(1)
                comment = clean_comment(field_match.group(2))
                if comment:
                    comments[class_name][pending_field] = comment
                continue

            if pending_field and "#" in raw_line:
                comment = clean_comment(raw_line.split("#", 1)[1])
                if comment and pending_field not in comments[class_name]:
                    comments[class_name][pending_field] = comment
                    pending_field = None
    return comments


def clean_comment(comment: str | None) -> str:
    """清理字段注释中的空白和句号。"""

    if not comment:
        return ""
    return " ".join(comment.strip().split())


def render_header(schema: dict[str, Any]) -> list[str]:
    """渲染文档头部和全局说明。"""

    info = schema.get("info", {})
    return [
        "# 接口文档",
        "",
        "本文档由 `scripts/generate_api_docs.py` 根据当前 FastAPI OpenAPI 元数据生成，",
        "并补充了流式 SSE 事件说明和 Pydantic 模型中的中文字段注释。",
        "",
        "## 基础说明",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| OpenAPI 标题 | {info.get('title', '')} |",
        f"| 版本 | {info.get('version', '')} |",
        f"| 描述 | {info.get('description', '')} |",
        "| 应用入口 | `api/main.py` |",
        "| 默认本地地址 | `http://127.0.0.1:8000` 或 `http://localhost:8000` |",
        "| 路由前缀 | 当前没有统一 `/api` 前缀，训练模块单独使用 `/training` 前缀 |",
        "| 普通响应 | JSON |",
        "| 流式响应 | SSE 文本流，`Content-Type: text/event-stream` |",
        "| 在线文档 | `/docs` Swagger UI，`/redoc` ReDoc，`/openapi.json` 原始 OpenAPI |",
        "",
        "## 通用约定",
        "",
        "- `model_mode` 当前使用字典项：`high`、`medium`、`low`。",
        "- `collection_name` 为空时，后端使用配置中的默认 Qdrant collection。",
        "- 普通接口校验失败会返回 FastAPI 标准 `422` 响应。",
        "- 本项目当前没有统一鉴权中间件，接口文档不包含登录态或 token 说明。",
        "- 知识正文、向量和训练切片正文以 Qdrant 为准，MySQL 只保存业务元数据和状态。",
        "",
    ]


def render_route_overview(schema: dict[str, Any], route_modules: dict[tuple[str, str], str]) -> list[str]:
    """渲染全部接口总览表。"""

    lines = ["## 接口总览", ""]
    grouped = group_operations(schema, route_modules)
    for group_key in GROUP_ORDER:
        operations = grouped.get(group_key, [])
        if not operations:
            continue
        lines.extend([f"### {GROUP_TITLES[group_key]}", "", "| 方法 | 路径 | 说明 |", "| --- | --- | --- |"])
        for operation in operations:
            lines.append(
                f"| `{operation['method'].upper()}` | `{operation['path']}` | "
                f"{operation['summary']} |"
            )
        lines.append("")
    return lines


def render_api_details(
        schema: dict[str, Any],
        field_comments: dict[str, dict[str, str]],
        route_modules: dict[tuple[str, str], str],
) -> list[str]:
    """渲染每个接口的详细参数和响应模型。"""

    lines = ["## 接口详情", ""]
    grouped = group_operations(schema, route_modules)
    components = schema.get("components", {}).get("schemas", {})

    for group_key in GROUP_ORDER:
        operations = grouped.get(group_key, [])
        if not operations:
            continue
        lines.extend([f"### {GROUP_TITLES[group_key]}", ""])
        for operation in operations:
            meta = operation["meta"]
            method = operation["method"].upper()
            path = operation["path"]
            lines.extend([
                f"#### {method} `{path}`",
                "",
                f"- 说明：{operation['summary']}",
            ])
            description = first_paragraph(meta.get("description"))
            if description:
                lines.append(f"- 详细说明：{description}")
            lines.append(f"- Operation ID：`{meta.get('operationId', '')}`")
            lines.append("")

            lines.extend(render_parameters(meta))
            lines.extend(render_request_body(meta, components, field_comments))
            lines.extend(render_response(meta, components, field_comments, path))
            lines.extend(render_stream_events(path))
    return lines


def render_parameters(meta: dict[str, Any]) -> list[str]:
    """渲染 path/query 参数表。"""

    parameters = meta.get("parameters") or []
    if not parameters:
        return ["请求参数：无 path/query 参数。", ""]

    lines = ["请求参数：", "", "| 参数 | 位置 | 必填 | 类型 | 默认值 | 说明 |", "| --- | --- | --- | --- | --- | --- |"]
    for parameter in parameters:
        schema = parameter.get("schema", {})
        lines.append(
            f"| `{parameter.get('name')}` | `{parameter.get('in')}` | {yes_no(parameter.get('required'))} | "
            f"`{compact_type(schema)}` | {format_default(schema)} | "
            f"{parameter.get('description') or schema.get('title') or ''} |"
        )
    lines.append("")
    return lines


def render_request_body(
        meta: dict[str, Any],
        components: dict[str, Any],
        field_comments: dict[str, dict[str, str]],
) -> list[str]:
    """渲染请求体说明。"""

    request_body = meta.get("requestBody")
    if not request_body:
        return ["请求体：无。", ""]

    content = request_body.get("content", {})
    content_type, body_schema = next(iter(content.items()))
    request_schema = body_schema.get("schema", {})
    schema_name = schema_display_name(request_schema)
    lines = [
        "请求体：",
        "",
        f"- Content-Type：`{content_type}`",
        f"- Schema：`{schema_name}`",
        "",
    ]
    lines.extend(render_schema_table(request_schema, components, field_comments))
    return lines


def render_response(
        meta: dict[str, Any],
        components: dict[str, Any],
        field_comments: dict[str, dict[str, str]],
        path: str,
) -> list[str]:
    """渲染 200 响应体说明。"""

    success_response = meta.get("responses", {}).get("200", {})
    content = success_response.get("content", {})
    if path in STREAM_EVENT_DESCRIPTIONS:
        return [
            "响应体：SSE 文本流，事件详见下方“流式事件”。",
            "",
        ]
    if not content:
        return ["响应体：普通 JSON，具体字段以运行时返回为准。", ""]

    content_type, response_info = next(iter(content.items()))
    response_schema = response_info.get("schema", {})
    schema_name = schema_display_name(response_schema)
    lines = [
        "响应体：",
        "",
        f"- Content-Type：`{content_type}`",
        f"- Schema：`{schema_name}`",
        "",
    ]
    lines.extend(render_schema_table(response_schema, components, field_comments))
    return lines


def render_stream_events(path: str) -> list[str]:
    """渲染流式接口事件表。"""

    events = STREAM_EVENT_DESCRIPTIONS.get(path)
    if not events:
        return []

    lines = ["流式事件：", "", "| 事件名 | 说明 |", "| --- | --- |"]
    for event_name, description in events:
        lines.append(f"| `{event_name}` | {description} |")
    lines.append("")
    return lines


def render_schema_appendix(schema: dict[str, Any], field_comments: dict[str, dict[str, str]]) -> list[str]:
    """渲染数据模型附录，方便前端按 schema 查询字段。"""

    components = schema.get("components", {}).get("schemas", {})
    lines = ["## 数据模型附录", ""]
    for schema_name in sorted(components):
        if schema_name in {"HTTPValidationError", "ValidationError"}:
            continue
        component_schema = components[schema_name]
        lines.extend([f"### `{schema_name}`", ""])
        description = component_schema.get("description")
        if description:
            lines.append(first_paragraph(description))
            lines.append("")
        lines.extend(render_object_fields(schema_name, component_schema, components, field_comments))
    return lines


def render_schema_table(
        schema: dict[str, Any],
        components: dict[str, Any],
        field_comments: dict[str, dict[str, str]],
) -> list[str]:
    """根据 schema 引用渲染字段表。"""

    ref_name = ref_schema_name(schema)
    if ref_name:
        return render_object_fields(ref_name, components.get(ref_name, {}), components, field_comments)

    if schema.get("type") == "array":
        item_schema = schema.get("items", {})
        item_ref = ref_schema_name(item_schema)
        if item_ref:
            lines = [f"数组元素：`{item_ref}`", ""]
            lines.extend(render_object_fields(item_ref, components.get(item_ref, {}), components, field_comments))
            return lines
        return [f"数组元素类型：`{compact_type(item_schema)}`", ""]

    if schema.get("properties"):
        return render_object_fields(schema.get("title", "InlineObject"), schema, components, field_comments)

    return [f"类型：`{compact_type(schema)}`。", ""]


def render_object_fields(
        schema_name: str,
        schema: dict[str, Any],
        components: dict[str, Any],
        field_comments: dict[str, dict[str, str]],
) -> list[str]:
    """渲染对象字段表。"""

    properties = schema.get("properties") or {}
    if not properties:
        return ["字段：无固定字段或由运行时动态返回。", ""]

    required_fields = set(schema.get("required") or [])
    comments = field_comments.get(schema_name, {})
    lines = ["| 字段 | 类型 | 必填 | 默认值 | 中文说明 |", "| --- | --- | --- | --- | --- |"]
    for field_name, field_schema in properties.items():
        field_ref_name = ref_schema_name(field_schema)
        description = comments.get(field_name) or field_schema.get("description") or field_schema.get("title") or ""
        if field_ref_name:
            description = f"{description}；对象模型：`{field_ref_name}`".strip("；")
        enum_values = enum_text(field_schema)
        if enum_values:
            description = f"{description}；可选值：{enum_values}".strip("；")
        lines.append(
            f"| `{field_name}` | `{compact_type(field_schema)}` | {yes_no(field_name in required_fields)} | "
            f"{format_default(field_schema)} | {description} |"
        )
    lines.append("")
    return lines


def group_operations(
        schema: dict[str, Any],
        route_modules: dict[tuple[str, str], str],
) -> dict[str, list[dict[str, Any]]]:
    """把 OpenAPI operation 按业务模块分组。"""

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in GROUP_ORDER}
    for path, methods in schema.get("paths", {}).items():
        for method, meta in methods.items():
            group_key = route_group(path, method, route_modules)
            grouped.setdefault(group_key, []).append(
                {
                    "path": path,
                    "method": method,
                    "meta": meta,
                    "summary": to_chinese_summary(path, method, meta),
                }
            )
    return grouped


def route_group(path: str, method: str, route_modules: dict[tuple[str, str], str]) -> str:
    """根据路径和路由模块判断接口分组。"""

    module_name = route_modules.get((path, method), "")
    if path.startswith("/training"):
        return "training"
    if "/exam" in path or module_name.endswith(".exam"):
        return "exam"
    if path.startswith("/knowledge"):
        return "knowledge"
    if path.startswith("/dictionaries"):
        return "dictionaries"
    if path.startswith("/health"):
        return "health"
    return "chat"


def to_chinese_summary(path: str, method: str, meta: dict[str, Any]) -> str:
    """把常见接口摘要转成更适合中文文档的说明。"""

    summary_map = {
        ("get", "/health"): "查询服务健康状态",
        ("get", "/conversations"): "分页查询聊天会话",
        ("get", "/conversations/{conversation_id}"): "查询聊天会话详情",
        ("delete", "/conversations/{conversation_id}"): "删除聊天会话",
        ("post", "/chat"): "一次性聊天回答",
        ("post", "/chat/stream"): "流式聊天回答",
        ("post", "/debug/retrieve"): "调试 RAG 检索结果",
        ("post", "/knowledge/upload/preview"): "上传知识库文件并预览识别结果",
        ("post", "/knowledge/upload/recommend"): "让模型推荐文档类型和切分策略",
        ("post", "/knowledge/upload/confirm"): "确认上传并正式入库",
        ("get", "/knowledge/files"): "查询知识库文件列表",
        ("get", "/knowledge/files/{document_id}"): "查询知识库文件详情",
        ("get", "/knowledge/files/{document_id}/preview"): "预览已入库文件原文",
        ("delete", "/knowledge/files/{document_id}"): "删除知识库文件",
        ("post", "/knowledge/files/reindex-all"): "全量重建知识库索引",
        ("post", "/knowledge/files/{document_id}/reindex"): "重建单个文件索引",
        ("post", "/knowledge/reload"): "扫描 data 目录并重载知识库",
        ("get", "/dictionaries"): "查询系统字典",
        ("post", "/dictionaries/items"): "新增或更新字典项",
        ("get", "/exam/sections"): "查询考试题源目录",
        ("post", "/exam/sessions"): "开始一场知识测评",
        ("get", "/exam/sessions"): "分页查询测评历史",
        ("post", "/exam/sessions/{session_id}/answer"): "提交当前轮答案",
        ("get", "/exam/sessions/{session_id}"): "查询测评详情",
        ("get", "/training/profile-dictionaries"): "查询销售训练画像字典",
        ("post", "/training/knowledge/upload"): "上传销售训练资料并生成待发布切片",
        ("get", "/training/knowledge/batches"): "分页查询训练资料批次",
        ("get", "/training/knowledge/batches/{batch_id}/preview"): "预览训练资料原文",
        ("delete", "/training/knowledge/batches/{batch_id}"): "删除训练资料批次",
        ("post", "/training/knowledge/batches/{batch_id}/publish"): "发布训练资料批次",
        ("post", "/training/knowledge/batches/{batch_id}/rollback"): "回滚训练资料版本",
        ("post", "/training/knowledge/batches/{batch_id}/reparse"): "重新切分训练资料批次",
        ("get", "/training/knowledge/batches/{batch_id}/versions"): "查询训练资料版本链",
        ("get", "/training/knowledge/batches/{batch_id}/chunks"): "查询训练资料切片预览",
        ("post", "/training/plans"): "创建销售训练方案",
        ("get", "/training/plans"): "分页查询销售训练方案",
        ("get", "/training/plans/{plan_id}"): "查询销售训练方案详情",
        ("put", "/training/plans/{plan_id}"): "更新销售训练方案",
        ("post", "/training/profiles/generate"): "生成 AI 客户角色",
        ("post", "/training/profiles/scenario/polish"): "润色训练场景描述",
        ("post", "/training/profiles/supplement-questions/generate"): "生成角色补充问答题",
        ("post", "/training/profiles/{profile_id}/goal-settings/generate"): "生成训练目标和评分设置",
        ("post", "/training/sessions"): "开始销售训练会话",
        ("get", "/training/sessions"): "分页查询销售训练会话",
        ("get", "/training/sessions/{session_id}"): "查询销售训练复盘详情",
        ("post", "/training/sessions/{session_id}/turns"): "提交学员回复",
        ("post", "/training/sessions/{session_id}/final-score"): "生成最终训练评分",
    }
    return summary_map.get((method, path), meta.get("summary") or meta.get("operationId") or "")


def compact_type(schema: dict[str, Any]) -> str:
    """把 OpenAPI schema 压缩成表格里可读的类型。"""

    ref_name = ref_schema_name(schema)
    if ref_name:
        return ref_name
    if "anyOf" in schema:
        return " | ".join(compact_type(item) for item in schema["anyOf"])
    if schema.get("type") == "array":
        return f"array[{compact_type(schema.get('items', {}))}]"
    if schema.get("additionalProperties") is not None and schema.get("type") == "object":
        return "object"
    if "enum" in schema:
        return "enum"
    return str(schema.get("type") or schema.get("title") or "object")


def schema_display_name(schema: dict[str, Any]) -> str:
    """返回请求体或响应体 schema 的展示名称。"""

    ref_name = ref_schema_name(schema)
    if ref_name:
        return ref_name
    if schema.get("type") == "array":
        return f"array[{compact_type(schema.get('items', {}))}]"
    return str(schema.get("title") or compact_type(schema))


def ref_schema_name(schema: dict[str, Any]) -> str | None:
    """读取 OpenAPI $ref 中的模型名称。"""

    ref = schema.get("$ref")
    if not ref:
        return None
    return ref.rsplit("/", 1)[-1]


def enum_text(schema: dict[str, Any]) -> str:
    """格式化枚举值说明。"""

    if "enum" in schema:
        return "、".join(f"`{item}`" for item in schema["enum"])
    if "anyOf" in schema:
        values = []
        for item in schema["anyOf"]:
            if "enum" in item:
                values.extend(item["enum"])
        return "、".join(f"`{item}`" for item in values)
    return ""


def format_default(schema: dict[str, Any]) -> str:
    """格式化默认值。"""

    if "default" not in schema:
        return ""
    value = schema["default"]
    if value is None:
        return "`null`"
    return f"`{value}`"


def first_paragraph(description: str | None) -> str:
    """提取 docstring 的第一段，避免接口详情过长。"""

    if not description:
        return ""
    paragraphs = [part.strip().replace("\n", " ") for part in description.split("\n\n") if part.strip()]
    return paragraphs[0] if paragraphs else ""


def yes_no(value: object) -> str:
    """把布尔值渲染成中文。"""

    return "是" if bool(value) else "否"


if __name__ == "__main__":
    main()
