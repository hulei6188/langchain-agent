from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage


def message_content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def message_reasoning_content(message: BaseMessage) -> str:
    parts: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts.extend(_reasoning_parts_from_blocks(content))
    parts.extend(_reasoning_parts_from_mapping(getattr(message, "additional_kwargs", {}) or {}))
    parts.extend(_reasoning_parts_from_mapping(getattr(message, "response_metadata", {}) or {}))
    return "".join(parts)


def _reasoning_parts_from_mapping(payload: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in (
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_text",
        "thinking_content",
        "reasoning_summary",
        "reasoning_details",
    ):
        if key in payload:
            parts.extend(_reasoning_parts_from_value(payload.get(key)))
    return parts


def _reasoning_parts_from_blocks(blocks: list[Any]) -> list[str]:
    parts: list[str] = []
    for item in blocks:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type") or "").lower()
        if "reasoning" in block_type or "thinking" in block_type:
            parts.extend(_reasoning_parts_from_value(item))
    return parts


def _reasoning_parts_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_reasoning_parts_from_value(item))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        for key in (
            "text",
            "content",
            "reasoning_content",
            "reasoning",
            "thinking",
            "summary",
            "thinking_content",
        ):
            if key in value:
                parts.extend(_reasoning_parts_from_value(value.get(key)))
        return parts
    return [str(value)]


def normalize_langchain_tool_calls(tool_calls: list[dict] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls or []):
        function = call.get("function") or {}
        raw_args = function.get("arguments") if function else call.get("args", {})
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            if not isinstance(args, dict):
                args = {"input": args}
        except (TypeError, ValueError, json.JSONDecodeError):
            args = {"input": str(raw_args)}
        converted.append(
            {
                "name": call.get("name") or function.get("name") or "",
                "args": args,
                "id": call.get("id") or f"call_{index}",
            }
        )
    return converted
