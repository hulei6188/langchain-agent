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
    return str(
        getattr(message, "additional_kwargs", {}).get("reasoning_content")
        or getattr(message, "additional_kwargs", {}).get("reasoning")
        or getattr(message, "additional_kwargs", {}).get("thinking")
        or ""
    )


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
