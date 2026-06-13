from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from core.config import get_settings
from core.integrations.model_clients import (
    DASHSCOPE_COMPATIBLE_BASE,
    OPENAI_COMPATIBLE_DEFAULT_BASE,
    api_base,
    api_key,
)

logger = logging.getLogger(__name__)

class _CancelledError(Exception):
    """Raised when a chat request is cancelled mid-flight."""
    pass


class OpenAICompatibleProvider(BaseChatModel):
    """LangChain chat model for OpenAI-compatible gateways."""

    _active_http_client: httpx.Client | None = PrivateAttr(default=None)
    _last_chat_mock: bool = PrivateAttr(default=False)

    def __init__(self) -> None:
        super().__init__()

    @property
    def last_chat_mock(self) -> bool:
        return self._last_chat_mock

    @last_chat_mock.setter
    def last_chat_mock(self, value: bool) -> None:
        self._last_chat_mock = bool(value)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted_tools = self._openai_tools(tools)
        bind_kwargs: dict[str, Any] = {"tools": formatted_tools, **kwargs}
        if tool_choice:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._chat_message(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._chat_message_chunks(messages, stop=stop, **kwargs):
            yield ChatGenerationChunk(message=chunk)

    def cancel_active_request(self) -> None:
        """Close the active ChatOpenAI HTTP client to unblock streaming reads."""
        client = self._active_http_client
        if client is not None:
            self._active_http_client = None
            try:
                client.close()
            except Exception:
                pass

    def _chat_message(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        model: str | None = None,
        temperature: float = 0.4,
        runtime_config: dict | None = None,
        tools: Sequence[dict[str, Any] | Any] | None = None,
        thinking_enabled: bool | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        settings = get_settings()
        chat_api_key = api_key(settings, runtime_config, purpose="chat")
        formatted_tools = self._openai_tools(tools)
        if settings.mock_llm:
            self.last_chat_mock = True
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError()
            user_text = self._content_text(next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""))
            context_hint = " ".join(self._content_text(m.content)[:160] for m in messages if isinstance(m, SystemMessage))
            if formatted_tools:
                tool_names = [t.get("function", {}).get("name", "") for t in formatted_tools]
                call_id = f"mock_call_{hashlib.md5(user_text.encode()).hexdigest()[:8]}"
                return AIMessage(
                    content="",
                    tool_calls=[{"name": tool_names[0], "args": {"query": user_text[:120]}, "id": call_id}],
                )
            return AIMessage(content=f"Mock answer for: {user_text}\n\nContext summary: {context_hint[:220]}")
        if not chat_api_key:
            raise RuntimeError("Chat model API key is not configured")
        self.last_chat_mock = False

        chat_api_base = api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        llm, http_client = self._chat_openai(
            api_base=chat_api_base,
            api_key=chat_api_key,
            model=chat_model,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            streaming=False,
        )
        runnable = self._bind_chat_tools(llm, tools, tool_choice=kwargs.get("tool_choice"))
        self._active_http_client = http_client
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError()
            response = runnable.invoke(messages, stop=stop)
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError()
            return self._ensure_ai_message(response)
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError() from exc
            raise
        finally:
            if self._active_http_client is http_client:
                self._active_http_client = None
            http_client.close()

    def _chat_message_chunks(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        model: str | None = None,
        temperature: float = 0.4,
        runtime_config: dict | None = None,
        tools: Sequence[dict[str, Any] | Any] | None = None,
        thinking_enabled: bool | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> Iterable[AIMessageChunk]:
        settings = get_settings()
        chat_api_key = api_key(settings, runtime_config, purpose="chat")
        formatted_tools = self._openai_tools(tools)
        if settings.mock_llm:
            self.last_chat_mock = True
            message = self._chat_message(
                messages,
                model=model,
                temperature=temperature,
                runtime_config=runtime_config,
                tools=formatted_tools,
                thinking_enabled=thinking_enabled,
                cancel_event=cancel_event,
                **kwargs,
            )
            if message.tool_calls:
                yield AIMessageChunk(content="", tool_call_chunks=self._tool_call_chunks(message.tool_calls))
                return
            content = self._content_text(message.content)
            for index in range(0, len(content), 24):
                if cancel_event is not None and cancel_event.is_set():
                    return
                yield AIMessageChunk(content=content[index : index + 24])
            return
        if not chat_api_key:
            raise RuntimeError("Chat model API key is not configured")
        self.last_chat_mock = False

        chat_api_base = api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        llm, http_client = self._chat_openai(
            api_base=chat_api_base,
            api_key=chat_api_key,
            model=chat_model,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            streaming=True,
        )
        runnable = self._bind_chat_tools(llm, tools, tool_choice=kwargs.get("tool_choice"))
        self._active_http_client = http_client
        try:
            for chunk in runnable.stream(messages, stop=stop):
                if cancel_event is not None and cancel_event.is_set():
                    raise _CancelledError()
                yield self._ensure_ai_chunk(chunk)
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError() from exc
            raise
        finally:
            if self._active_http_client is http_client:
                self._active_http_client = None
            http_client.close()

    def requires_reasoning_replay(self, *, model: str | None = None, runtime_config: dict | None = None) -> bool:
        settings = get_settings()
        chat_api_base = api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        return self._is_deepseek(chat_api_base, chat_model)

    # ── private helpers ──────────────────────────────────────────

    def _chat_openai(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float,
        thinking_enabled: bool | None,
        streaming: bool,
    ) -> tuple[ChatOpenAI, httpx.Client]:
        timeout = httpx.Timeout(120.0 if streaming else 60.0)
        http_client = httpx.Client(timeout=timeout)
        model_kwargs = self._chat_model_kwargs(
            api_base=api_base,
            model=model,
            thinking_enabled=thinking_enabled,
        )
        return (
            ChatOpenAI(
                api_key=api_key,
                base_url=api_base.rstrip("/") or None,
                model=model,
                temperature=temperature,
                streaming=streaming,
                timeout=timeout,
                max_retries=0,
                http_client=http_client,
                **model_kwargs,
            ),
            http_client,
        )

    def _bind_chat_tools(
        self,
        llm: ChatOpenAI,
        tools: Sequence[dict[str, Any] | Any] | None,
        *,
        tool_choice: str | dict | bool | None = None,
    ) -> Runnable:
        if not tools:
            return llm
        return llm.bind_tools(list(tools), tool_choice=tool_choice or "auto")

    def _chat_model_kwargs(self, *, api_base: str, model: str, thinking_enabled: bool | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        if self._is_deepseek(api_base, model):
            enabled = bool(thinking_enabled)
            extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}
            if enabled:
                kwargs["reasoning_effort"] = "high"
        elif thinking_enabled is not None and self._is_dashscope_qwen(api_base, model):
            extra_body["enable_thinking"] = bool(thinking_enabled)
        elif thinking_enabled and self._is_openai_reasoning_model(api_base, model):
            kwargs["reasoning_effort"] = "high"
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _ensure_ai_message(self, message: BaseMessage) -> AIMessage:
        if isinstance(message, AIMessage):
            return message
        return AIMessage(
            content=self._content_text(message.content),
            additional_kwargs=dict(getattr(message, "additional_kwargs", {}) or {}),
            response_metadata=dict(getattr(message, "response_metadata", {}) or {}),
        )

    def _ensure_ai_chunk(self, chunk: BaseMessage) -> AIMessageChunk:
        if isinstance(chunk, AIMessageChunk):
            return chunk
        return AIMessageChunk(
            content=self._content_text(chunk.content),
            additional_kwargs=dict(getattr(chunk, "additional_kwargs", {}) or {}),
            response_metadata=dict(getattr(chunk, "response_metadata", {}) or {}),
        )

    def _openai_tools(self, tools: Sequence[dict[str, Any] | Any] | None) -> list[dict[str, Any]]:
        return [tool if isinstance(tool, dict) else convert_to_openai_tool(tool) for tool in tools or []]

    def _tool_call_chunks(self, tool_calls: list[dict]) -> list[dict[str, Any]]:
        chunks = []
        for index, call in enumerate(tool_calls):
            func = call.get("function") or {}
            raw_args = func.get("arguments", "") if func else call.get("args", "")
            if raw_args is None:
                raw_args = ""
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args or {}, ensure_ascii=False)
            chunks.append(
                {
                    "name": func.get("name") or call.get("name") or "",
                    "args": raw_args or "",
                    "id": call.get("id"),
                    "index": call.get("index", index),
                }
            )
        return chunks

    @staticmethod
    def _is_openai_reasoning_model(api_base: str, model: str) -> bool:
        normalized_model = (model or "").lower().strip()
        if not normalized_model:
            return False
        # OpenAI-compatible gateways usually keep the original OpenAI model
        # name even when api_base is not api.openai.com, so prefer model-name
        # detection here instead of only checking the base URL.
        openai_reasoning_prefixes = (
            "gpt-5",
            "o1",
            "o3",
            "o4",
        )
        return normalized_model.startswith(openai_reasoning_prefixes)

    @staticmethod
    def _is_dashscope_qwen(api_base: str, model: str) -> bool:
        normalized_base = (api_base or "").lower()
        normalized_model = (model or "").lower()
        return (
            ("dashscope.aliyuncs.com" in normalized_base or "dashscope-intl.aliyuncs.com" in normalized_base)
            and normalized_model.startswith("qwen")
        )

    @staticmethod
    def _is_deepseek(api_base: str, model: str) -> bool:
        normalized_base = (api_base or "").lower()
        return "api.deepseek.com" in normalized_base

    def _content_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    parts.append("[image]")
            return " ".join(parts)
        return str(content)
