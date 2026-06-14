from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from core.runtime.dsml import (
    DSML_TOOL_MARKUP_ERROR,
    buffer_stream_content,
    contains_dsml_tool_calls,
    contains_leaked_tool_markup,
    dsml_preview,
    dsml_tool_names,
    parse_dsml_tool_calls,
    strip_or_block_leaked_tool_markup,
)
from core.runtime.message_utils import message_content_text, message_reasoning_content
from core.runtime.tool_calls import finalize_stream_tool_calls, merge_stream_tool_call_chunks


logger = logging.getLogger(__name__)


def _noop() -> None:
    return None


def stream_response_to_writer(events: Iterable[dict], writer: Callable[[dict], None] | None) -> AIMessage:
    stream = iter(events)
    while True:
        try:
            event = next(stream)
        except StopIteration as stop:
            return stop.value or AIMessage(content="")
        if event and writer:
            writer(event)


def stream_chat_response(
    provider,
    *,
    agent,
    messages: list[BaseMessage],
    context: dict,
    tools: list[Any] | None = None,
    stream_content: bool = False,
    provisional_stream: bool = False,
    thinking_enabled: bool | None = None,
    cancel_event=None,
    raise_if_cancelled: Callable[[], None] | None = None,
):
    raise_if_cancelled = raise_if_cancelled or _noop
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_call_builders: dict[int, dict] = {}
    final_tool_calls: list[dict] = []
    saw_tool_call = False
    pending_live_content = ""
    suppress_content_stream = False
    emitted_live_content = False
    provisional_active = bool(provisional_stream and tools)
    provisional_sent = False
    provisional_cleared = False
    should_stream_content_live = stream_content and (not tools or provisional_active)
    for chunk in provider.stream(
        messages,
        model=agent.model,
        temperature=agent.temperature,
        runtime_config=agent.runtime_config,
        tools=tools,
        thinking_enabled=thinking_enabled,
        cancel_event=cancel_event,
    ):
        raise_if_cancelled()
        reasoning_chunk = message_reasoning_content(chunk)
        content_chunk = message_content_text(chunk)
        if reasoning_chunk:
            if context.get("thinking_enabled"):
                reasoning_chunks.append(reasoning_chunk)
                if stream_content and not saw_tool_call:
                    yield {"event": "reasoning_token", "content": reasoning_chunk}
        if content_chunk:
            content_chunks.append(content_chunk)
            if should_stream_content_live and not saw_tool_call:
                pending_live_content, suppress_content_stream, safe_chunks = buffer_stream_content(
                    pending_live_content,
                    content_chunk,
                    suppress_content_stream,
                )
                for safe_content in safe_chunks:
                    emitted_live_content = True
                    if provisional_active:
                        provisional_sent = True
                        yield {"event": "provisional_token", "content": safe_content}
                    else:
                        yield {"event": "token", "content": safe_content}
        tool_call_chunks = getattr(chunk, "tool_call_chunks", []) or []
        if tool_call_chunks:
            if not saw_tool_call and provisional_active and provisional_sent and not provisional_cleared:
                provisional_cleared = True
                yield {"event": "provisional_clear", "data": {"reason": "tool_call"}}
            saw_tool_call = True
            merge_stream_tool_call_chunks(tool_call_builders, tool_call_chunks)
        elif getattr(chunk, "tool_calls", None):
            if not saw_tool_call and provisional_active and provisional_sent and not provisional_cleared:
                provisional_cleared = True
                yield {"event": "provisional_clear", "data": {"reason": "tool_call"}}
            saw_tool_call = True
            final_tool_calls = list(chunk.tool_calls or [])
    raise_if_cancelled()
    if not final_tool_calls:
        final_tool_calls = finalize_stream_tool_calls(tool_call_builders)

    joined_content = "".join(content_chunks)
    content_for_response = joined_content
    if final_tool_calls:
        if joined_content.strip():
            logger.warning("Dropping assistant content emitted before tool calls during stream response; preview=%r", dsml_preview(joined_content))
        content_for_response = ""
    elif not final_tool_calls and contains_dsml_tool_calls(joined_content):
        logger.warning("Detected DSML tool call markup in streamed assistant content")
        if tools:
            dsml_calls = parse_dsml_tool_calls(joined_content)
            if dsml_calls:
                logger.warning("Parsed DSML tool calls from streamed content: tools=%s", dsml_tool_names(dsml_calls))
                final_tool_calls = dsml_calls
                content_for_response = ""
            else:
                logger.warning("Failed to parse DSML tool calls from streamed content; full content:\n%s", joined_content)
                content_for_response = DSML_TOOL_MARKUP_ERROR
        else:
            logger.warning("Blocked DSML tool call markup in streamed final content; preview=%r", dsml_preview(joined_content))
            content_for_response = strip_or_block_leaked_tool_markup(joined_content)
    elif not final_tool_calls and contains_leaked_tool_markup(joined_content):
        logger.warning("Blocked incomplete tool call markup in streamed content; full content:\n%s", joined_content)
        content_for_response = DSML_TOOL_MARKUP_ERROR

    if final_tool_calls and provisional_active and provisional_sent and not provisional_cleared:
        provisional_cleared = True
        yield {"event": "provisional_clear", "data": {"reason": "tool_call"}}

    if stream_content and not final_tool_calls and content_for_response:
        if should_stream_content_live:
            if suppress_content_stream:
                if provisional_active and provisional_sent and not provisional_cleared:
                    provisional_cleared = True
                    yield {"event": "provisional_clear", "data": {"reason": "blocked_tool_markup"}}
                if content_for_response == DSML_TOOL_MARKUP_ERROR or not emitted_live_content:
                    yield {"event": "token", "content": content_for_response}
            elif not contains_leaked_tool_markup(joined_content):
                if pending_live_content:
                    emitted_live_content = True
                    if provisional_active:
                        provisional_sent = True
                        yield {"event": "provisional_token", "content": pending_live_content}
                    else:
                        yield {"event": "token", "content": pending_live_content}
                if provisional_active and provisional_sent:
                    yield {"event": "provisional_commit", "data": {}}
    additional_kwargs = {}
    if reasoning_chunks:
        additional_kwargs["reasoning_content"] = "".join(reasoning_chunks)
    return AIMessage(
        content=content_for_response or "",
        additional_kwargs=additional_kwargs,
        tool_calls=final_tool_calls,
    )


def stream_chat_response_to_writer(
    provider,
    *,
    agent,
    messages: list[BaseMessage],
    context: dict,
    writer: Callable[[dict], None] | None,
    tools: list[Any] | None = None,
    stream_content: bool = True,
    provisional_stream: bool = False,
    thinking_enabled: bool | None = None,
    cancel_event=None,
    raise_if_cancelled: Callable[[], None] | None = None,
) -> AIMessage:
    return stream_response_to_writer(
        stream_chat_response(
            provider,
            agent=agent,
            messages=messages,
            context=context,
            tools=tools,
            stream_content=stream_content,
            provisional_stream=provisional_stream,
            thinking_enabled=thinking_enabled,
            cancel_event=cancel_event,
            raise_if_cancelled=raise_if_cancelled,
        ),
        writer,
    )


def stream_llm_response(
    provider,
    *,
    agent,
    messages: list[BaseMessage],
    context: dict,
    thinking_enabled: bool | None = None,
    cancel_event=None,
    raise_if_cancelled: Callable[[], None] | None = None,
):
    raise_if_cancelled = raise_if_cancelled or _noop
    chunks = []
    reasoning_chunks = []
    pending_live_content = ""
    suppress_content_stream = False
    emitted_live_content = False
    for chunk in provider.stream(
        messages,
        model=agent.model,
        temperature=agent.temperature,
        runtime_config=agent.runtime_config,
        thinking_enabled=thinking_enabled,
        cancel_event=cancel_event,
    ):
        raise_if_cancelled()
        reasoning_chunk = message_reasoning_content(chunk)
        content_chunk = message_content_text(chunk)
        if reasoning_chunk and context.get("thinking_enabled"):
            reasoning_chunks.append(reasoning_chunk)
            yield {"event": "reasoning_token", "content": reasoning_chunk}
        if content_chunk:
            chunks.append(content_chunk)
            pending_live_content, suppress_content_stream, safe_chunks = buffer_stream_content(
                pending_live_content,
                content_chunk,
                suppress_content_stream,
            )
            for safe_content in safe_chunks:
                emitted_live_content = True
                yield {"event": "token", "content": safe_content}
    raise_if_cancelled()
    raw_draft = "".join(chunks)
    draft = strip_or_block_leaked_tool_markup(raw_draft)
    if draft and draft != raw_draft:
        logger.warning("Blocked leaked tool call markup in streamed LLM node; preview=%r", dsml_preview(raw_draft))
    if draft:
        if suppress_content_stream:
            if draft == DSML_TOOL_MARKUP_ERROR or not emitted_live_content:
                yield {"event": "token", "content": draft}
        elif not contains_leaked_tool_markup(raw_draft) and pending_live_content:
            yield {"event": "token", "content": pending_live_content}
    additional_kwargs = {}
    if reasoning_chunks:
        additional_kwargs["reasoning_content"] = "".join(reasoning_chunks)
    return AIMessage(content=draft, additional_kwargs=additional_kwargs)
