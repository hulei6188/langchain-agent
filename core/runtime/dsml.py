from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig


DSML_TOOL_MARKUP_ERROR = "工具调用格式异常，未能正确执行，请重试。"
DSML_TOOL_CALL_START_MARKER = "<||DSML||tool_calls>"
DSML_STREAM_GUARD_TAIL_CHARS = len(DSML_TOOL_CALL_START_MARKER) - 1

logger = logging.getLogger(__name__)

_DSML_TOOL_CALLS_BLOCK_RE = re.compile(
    r"<\|\|DSML\|\|tool_calls\s*>(?P<body>.*?)</\|\|DSML\|\|tool_calls\s*>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<\|\|DSML\|\|invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)</\|\|DSML\|\|invoke\s*>",
    re.DOTALL,
)
_DSML_PARAMETER_RE = re.compile(
    r"<\|\|DSML\|\|parameter\b(?P<attrs>[^>]*)>(?P<body>.*?)</\|\|DSML\|\|parameter\s*>",
    re.DOTALL,
)


def _normalize_dsml_markup(text: str) -> str:
    return text.replace("｜", "|")


def contains_dsml_tool_calls(text: str | None) -> bool:
    if not text:
        return False
    return DSML_TOOL_CALL_START_MARKER in _normalize_dsml_markup(text)


def _dsml_attr(attrs: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*([\"'])(.*?)\1", attrs or "", re.DOTALL)
    return match.group(2) if match else ""


def parse_dsml_tool_calls(text: str) -> list[dict]:
    if not text or not contains_dsml_tool_calls(text):
        return []

    normalized = _normalize_dsml_markup(text)
    tool_calls: list[dict] = []
    dsml_blocks_found = 0
    for block_match in _DSML_TOOL_CALLS_BLOCK_RE.finditer(normalized):
        dsml_blocks_found += 1
        body_start = block_match.start("body")
        body_end = block_match.end("body")
        normalized_body = normalized[body_start:body_end]
        original_body = text[body_start:body_end]
        for invoke_match in _DSML_INVOKE_RE.finditer(normalized_body):
            tool_name = _dsml_attr(invoke_match.group("attrs"), "name")
            if not tool_name:
                continue
            invoke_body_start = invoke_match.start("body")
            invoke_body_end = invoke_match.end("body")
            normalized_invoke_body = normalized_body[invoke_body_start:invoke_body_end]
            original_invoke_body = original_body[invoke_body_start:invoke_body_end]
            params: dict[str, str] = {}
            for param_match in _DSML_PARAMETER_RE.finditer(normalized_invoke_body):
                param_name = _dsml_attr(param_match.group("attrs"), "name")
                if not param_name:
                    continue
                params[param_name] = original_invoke_body[param_match.start("body") : param_match.end("body")]
            tool_calls.append(
                {
                    "id": f"call_dsml_{len(tool_calls)}",
                    "name": tool_name,
                    "args": params,
                }
            )
    if dsml_blocks_found > 0 and not tool_calls:
        logger.warning(
            "DSML tool call parsing failed: found %d DSML block(s) but extracted 0 valid tool calls. Raw content:\n%s",
            dsml_blocks_found,
            text,
        )
    return tool_calls


def strip_complete_dsml_tool_call_blocks(text: str) -> str:
    if not text:
        return ""
    normalized = _normalize_dsml_markup(text)
    spans = [match.span() for match in _DSML_TOOL_CALLS_BLOCK_RE.finditer(normalized)]
    if not spans:
        return text
    pieces: list[str] = []
    last = 0
    for start, end in spans:
        pieces.append(text[last:start])
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def contains_leaked_tool_markup(text: str | None) -> bool:
    if not text:
        return False
    normalized = _normalize_dsml_markup(text)
    return (
        contains_dsml_tool_calls(normalized)
        or "<||DSML||tool_calls" in normalized
        or "<||DSML||invoke" in normalized
        or bool(re.search(r"\binvoke\s+name\s*=\s*([\"'])", normalized))
    )


def strip_or_block_leaked_tool_markup(text: str) -> str:
    if not text:
        return ""
    if not contains_leaked_tool_markup(text):
        return text
    cleaned = strip_complete_dsml_tool_call_blocks(text).strip()
    if cleaned and not contains_leaked_tool_markup(cleaned):
        return cleaned
    return DSML_TOOL_MARKUP_ERROR


def dsml_preview(text: str | None) -> str:
    return (text or "")[:500]


def dsml_tool_names(tool_calls: list[dict]) -> list[str]:
    names = []
    for call in tool_calls:
        name = call.get("name") or (call.get("function") or {}).get("name")
        if name:
            names.append(str(name))
    return names


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


def coerce_dsml_tool_calls(response: AIMessage, *, stage: str) -> AIMessage:
    content = message_content_text(response)
    if not content or response.tool_calls or not contains_dsml_tool_calls(content):
        return response
    logger.warning("Detected DSML tool call markup in assistant content during %s", stage)
    dsml_calls = parse_dsml_tool_calls(content)
    if dsml_calls:
        logger.warning("Parsed DSML tool calls during %s: tools=%s", stage, dsml_tool_names(dsml_calls))
        return AIMessage(
            content="",
            additional_kwargs=response.additional_kwargs,
            tool_calls=dsml_calls,
        )
    logger.warning("Failed to parse DSML tool calls during %s; full content:\n%s", stage, content)
    return AIMessage(content=DSML_TOOL_MARKUP_ERROR, additional_kwargs=response.additional_kwargs)


def buffer_stream_content(
    pending: str,
    content: str,
    suppress_content_stream: bool,
) -> tuple[str, bool, list[str]]:
    if suppress_content_stream:
        return pending, True, []
    pending += content
    normalized = _normalize_dsml_markup(pending)
    marker_index = normalized.find(DSML_TOOL_CALL_START_MARKER)
    if marker_index >= 0:
        safe_prefix = pending[:marker_index]
        return "", True, [safe_prefix] if safe_prefix else []

    tail_length = 0
    max_tail = min(len(normalized), DSML_STREAM_GUARD_TAIL_CHARS)
    for length in range(max_tail, 0, -1):
        if DSML_TOOL_CALL_START_MARKER.startswith(normalized[-length:]):
            tail_length = length
            break
    if tail_length <= 0:
        return "", False, [pending] if pending else []
    safe_content = pending[:-tail_length]
    return pending[-tail_length:], False, [safe_content] if safe_content else []


class DSMLToolCallParser(Runnable[AIMessage, AIMessage]):
    """Convert DSML fallback markup into LangChain-native tool calls."""

    def invoke(self, input: AIMessage, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessage:
        metadata = (config or {}).get("metadata") or {}
        stage = str(kwargs.get("stage") or metadata.get("stage") or "model output")
        return coerce_dsml_tool_calls(input, stage=stage)


dsml_tool_call_parser = DSMLToolCallParser()

