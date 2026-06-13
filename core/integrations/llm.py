from __future__ import annotations

import hashlib
import http.client
import json
import logging
import socket
import ssl
import threading
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from core.config import get_settings

logger = logging.getLogger(__name__)

DASHSCOPE_COMPATIBLE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_COMPATIBLE_DEFAULT_BASE = DASHSCOPE_COMPATIBLE_BASE


class _CancelledError(Exception):
    """Raised when a chat request is cancelled mid-flight."""
    pass


class OpenAICompatibleProvider(BaseChatModel):
    """LangChain chat model for OpenAI-compatible gateways.

    Embedding and rerank helpers live on the same integration object because the
    product lets a single provider configuration cover chat, embedding and RAG
    reranking endpoints.
    """

    _active_response: http.client.HTTPResponse | None = PrivateAttr(default=None)
    _last_chat_mock: bool = PrivateAttr(default=False)
    _last_embed_mock: bool = PrivateAttr(default=False)

    def __init__(self) -> None:
        super().__init__()

    @property
    def last_chat_mock(self) -> bool:
        return self._last_chat_mock

    @last_chat_mock.setter
    def last_chat_mock(self, value: bool) -> None:
        self._last_chat_mock = bool(value)

    @property
    def last_embed_mock(self) -> bool:
        return self._last_embed_mock

    @last_embed_mock.setter
    def last_embed_mock(self, value: bool) -> None:
        self._last_embed_mock = bool(value)

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
        formatted_tools = [tool if isinstance(tool, dict) else convert_to_openai_tool(tool) for tool in tools]
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
        """Close the active streaming HTTP connection to unblock reads.

        Called from the cancellation registry when a run is cancelled.
        Safe to call from any thread.

        We close the underlying socket directly so the reading thread gets
        a clean OSError instead of an AttributeError (which would happen
        if we called response.close() which sets response.fp = None).
        """
        resp = self._active_response
        if resp is not None:
            self._active_response = None
            try:
                # response.fp is a socket.SocketIO; closing it shuts down
                # the TCP socket and causes the blocked readline() to raise
                # OSError in the streaming thread.
                fp = getattr(resp, 'fp', None)
                if fp is not None and hasattr(fp, 'close'):
                    fp.close()
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
        tools: list[dict] | None = None,
        thinking_enabled: bool | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        settings = get_settings()
        api_key = self._api_key(settings, runtime_config, purpose="chat")
        if settings.mock_llm:
            self.last_chat_mock = True
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError()
            user_text = self._content_text(next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""))
            context_hint = " ".join(self._content_text(m.content)[:160] for m in messages if isinstance(m, SystemMessage))
            if tools:
                tool_names = [t.get("function", {}).get("name", "") for t in tools]
                call_id = f"mock_call_{hashlib.md5(user_text.encode()).hexdigest()[:8]}"
                return AIMessage(
                    content="",
                    tool_calls=[{"name": tool_names[0], "args": {"query": user_text[:120]}, "id": call_id}],
                )
            return AIMessage(content=f"Mock answer for: {user_text}\n\nContext summary: {context_hint[:220]}")
        if not api_key:
            raise RuntimeError("Chat model API key is not configured")
        self.last_chat_mock = False

        api_base = self._api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        url = api_base.rstrip("/") + "/chat/completions"
        payload: dict = {
            "model": chat_model,
            "messages": self._openai_messages(messages),
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        self._apply_thinking_payload(payload, api_base=api_base, model=chat_model, thinking_enabled=thinking_enabled)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if kwargs.get("tool_choice"):
            payload["tool_choice"] = kwargs["tool_choice"]
        data = self._post_json(url, payload, api_key)
        return self._parse_ai_message(data)

    def _chat_message_chunks(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        model: str | None = None,
        temperature: float = 0.4,
        runtime_config: dict | None = None,
        tools: list[dict] | None = None,
        thinking_enabled: bool | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> Iterable[AIMessageChunk]:
        settings = get_settings()
        api_key = self._api_key(settings, runtime_config, purpose="chat")
        if settings.mock_llm:
            self.last_chat_mock = True
            message = self._chat_message(
                messages,
                model=model,
                temperature=temperature,
                runtime_config=runtime_config,
                tools=tools,
                thinking_enabled=thinking_enabled,
                cancel_event=cancel_event,
                **kwargs,
            )
            if message.tool_calls:
                yield AIMessageChunk(content="", tool_call_chunks=self._langchain_tool_call_chunks(message.tool_calls))
                return
            content = self._content_text(message.content)
            for index in range(0, len(content), 24):
                if cancel_event is not None and cancel_event.is_set():
                    return
                yield AIMessageChunk(content=content[index : index + 24])
            return
        if not api_key:
            raise RuntimeError("Chat model API key is not configured")
        self.last_chat_mock = False

        api_base = self._api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        url = api_base.rstrip("/") + "/chat/completions"
        payload: dict = {
            "model": chat_model,
            "messages": self._openai_messages(messages),
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        self._apply_thinking_payload(payload, api_base=api_base, model=chat_model, thinking_enabled=thinking_enabled)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if kwargs.get("tool_choice"):
            payload["tool_choice"] = kwargs["tool_choice"]
        yield from self._post_json_stream(url, payload, api_key, cancel_event=cancel_event)

    def requires_reasoning_replay(self, *, model: str | None = None, runtime_config: dict | None = None) -> bool:
        settings = get_settings()
        api_base = self._api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        return self._is_deepseek(api_base, chat_model)

    def embed(self, text: str, *, runtime_config: dict | None = None) -> list[float]:
        settings = get_settings()
        api_key = self._api_key(settings, runtime_config, purpose="embedding")
        if settings.mock_llm:
            self.last_embed_mock = True
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            return [((digest[i % len(digest)] / 255.0) * 2) - 1 for i in range(32)]
        if not api_key:
            raise RuntimeError("Embedding API key is not configured")
        self.last_embed_mock = False

        url = self._api_base(settings, runtime_config, purpose="embedding").rstrip("/") + "/embeddings"
        payload = {"model": settings.openai_embedding_model, "input": text}
        data = self._post_json(url, payload, api_key)
        return data["data"][0]["embedding"]

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None, model: str | None = None) -> list[dict]:
        settings = get_settings()
        api_key = self._api_key(settings, purpose="rerank")
        if not documents:
            return []
        if settings.mock_llm:
            query_terms = {term.lower() for term in query.split() if term.strip()}
            ranked = []
            for index, document in enumerate(documents):
                text = document.lower()
                score = sum(1 for term in query_terms if term in text) / max(len(query_terms), 1)
                ranked.append({"index": index, "relevance_score": float(score)})
            return sorted(ranked, key=lambda item: item["relevance_score"], reverse=True)[: top_n or len(documents)]
        if not api_key:
            raise RuntimeError("Rerank API key is not configured")

        url = self._api_base(settings, purpose="rerank").rstrip("/") + "/rerank"
        payload = {
            "model": model or settings.rag_rerank_model,
            "query": query,
            "documents": documents,
            **({"top_n": top_n} if top_n else {}),
        }
        data = self._post_json(url, payload, api_key)
        results = data.get("results") or data.get("data") or []
        normalized = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("document_index"))
            if index is None:
                document = item.get("document")
                if document in documents:
                    index = documents.index(document)
            if index is None:
                continue
            score = item.get("relevance_score", item.get("score", item.get("rank_score", 0)))
            normalized.append({"index": int(index), "relevance_score": float(score or 0)})
        return normalized[: top_n or len(normalized)]

    # ── private helpers ──────────────────────────────────────────

    def _openai_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        payload_messages = convert_to_openai_messages(messages)
        if not isinstance(payload_messages, list):
            payload_messages = [payload_messages]
        for source, payload in zip(messages, payload_messages):
            if isinstance(source, AIMessage):
                reasoning = (
                    source.additional_kwargs.get("reasoning_content")
                    or source.additional_kwargs.get("reasoning")
                    or source.additional_kwargs.get("thinking")
                )
                if reasoning:
                    payload["reasoning_content"] = str(reasoning)
        return payload_messages

    def _parse_ai_message(self, data: dict) -> AIMessage:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning_content = str(message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or "")
        tool_calls = self._openai_tool_calls_to_langchain(message.get("tool_calls") or [])
        additional_kwargs: dict[str, Any] = {}
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content
        return AIMessage(
            content=str(content) if content is not None else "",
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            response_metadata={"raw": data},
        )

    def _openai_tool_calls_to_langchain(self, raw_tool_calls: list[dict]) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for index, tc in enumerate(self._normalize_tool_calls(raw_tool_calls)):
            func = tc.get("function") or {}
            raw_args = func.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(args, dict):
                    args = {"input": args}
            except (TypeError, ValueError, json.JSONDecodeError):
                args = {"input": str(raw_args)}
            tool_calls.append(
                {
                    "name": func.get("name") or "",
                    "args": args,
                    "id": tc.get("id") or f"call_{index}",
                }
            )
        return tool_calls

    def _langchain_tool_call_chunks(self, tool_calls: list[dict]) -> list[dict[str, Any]]:
        chunks = []
        for index, call in enumerate(tool_calls):
            chunks.append(
                {
                    "name": call.get("name") or "",
                    "args": json.dumps(call.get("args") or {}, ensure_ascii=False),
                    "id": call.get("id"),
                    "index": index,
                }
            )
        return chunks

    def _openai_tool_call_chunks(self, raw_tool_calls: list[dict]) -> list[dict[str, Any]]:
        chunks = []
        for index, call in enumerate(raw_tool_calls):
            func = call.get("function") or {}
            args = func.get("arguments")
            if args is None and call.get("args") is not None:
                args = json.dumps(call.get("args") or {}, ensure_ascii=False)
            chunks.append(
                {
                    "name": func.get("name") or call.get("name"),
                    "args": args or "",
                    "id": call.get("id"),
                    "index": call.get("index", index),
                }
            )
        return chunks

    def _api_key(self, settings, runtime_config: dict | None = None, *, purpose: str = "chat") -> str | None:
        if runtime_config and purpose == "chat" and runtime_config.get("api_key"):
            return str(runtime_config["api_key"]).strip() or None
        if purpose == "embedding" and settings.embedding_api_key:
            return settings.embedding_api_key.strip() or None
        if purpose == "rerank" and settings.rerank_api_key:
            return settings.rerank_api_key.strip() or None
        if purpose == "chat":
            base = (settings.openai_api_base or "").rstrip("/")
            if settings.dashscope_api_key and base == DASHSCOPE_COMPATIBLE_BASE.rstrip("/"):
                return settings.dashscope_api_key.strip() or None
            if settings.deepseek_api_key and (base == settings.deepseek_api_base.rstrip("/") or settings.openai_model == settings.deepseek_model):
                return settings.deepseek_api_key.strip() or None
            if settings.openai_api_key:
                return settings.openai_api_key.strip() or None
            return (settings.dashscope_api_key or settings.deepseek_api_key or "").strip() or None
        return (settings.openai_api_key or settings.dashscope_api_key or "").strip() or None

    def _api_base(self, settings, runtime_config: dict | None = None, *, purpose: str = "chat") -> str:
        if runtime_config and purpose == "chat" and runtime_config.get("base_url"):
            return str(runtime_config["base_url"]).strip()
        if purpose == "embedding" and settings.embedding_api_base:
            return settings.embedding_api_base
        if purpose == "rerank" and settings.rerank_api_base:
            return settings.rerank_api_base
        if purpose == "chat" and settings.deepseek_api_key and (
            (settings.openai_api_base or "").rstrip("/") == settings.deepseek_api_base.rstrip("/")
            or settings.openai_model == settings.deepseek_model
        ):
            return settings.deepseek_api_base
        base = (settings.openai_api_base or "").strip()
        if settings.dashscope_api_key and (not base or base.rstrip("/") == OPENAI_COMPATIBLE_DEFAULT_BASE.rstrip("/")):
            return DASHSCOPE_COMPATIBLE_BASE
        return base or OPENAI_COMPATIBLE_DEFAULT_BASE

    def _post_json(self, url: str, payload: dict, api_key: str, *, timeout_seconds: int = 60) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(
                f"Model call failed: HTTP {exc.code}. Check OPENAI_API_BASE, API key and model name. {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
            raise RuntimeError(
                f"Model call failed: cannot connect to model gateway {url}. Check OPENAI_API_BASE, proxy, certs and API key. Raw error: {exc}"
            ) from exc

    def _apply_thinking_payload(self, payload: dict, *, api_base: str, model: str, thinking_enabled: bool | None) -> None:
        if self._is_deepseek(api_base, model):
            enabled = bool(thinking_enabled)
            payload["thinking"] = {"type": "enabled" if enabled else "disabled"}
            if enabled:
                payload["reasoning_effort"] = "high"
            return
        if thinking_enabled is None:
            return
        if self._is_dashscope_qwen(api_base, model):
            payload["enable_thinking"] = bool(thinking_enabled)
            return
        if self._is_openai_reasoning_model(api_base, model):
            if thinking_enabled:
                payload["reasoning_effort"] = "high"
            return

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

    def _post_json_stream(self, url: str, payload: dict, api_key: str, *, cancel_event: threading.Event | None = None) -> Iterable[AIMessageChunk]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                self._active_response = response
                try:
                    for raw_line in response:
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info("Streaming request cancelled mid-stream")
                            raise _CancelledError()
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload_text = line.removeprefix("data:").strip()
                        if payload_text == "[DONE]":
                            break
                        try:
                            data = json.loads(payload_text)
                        except json.JSONDecodeError:
                            continue
                        yield from self._stream_chunks(data)
                finally:
                    self._active_response = None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(
                f"Model call failed: HTTP {exc.code}. Check OPENAI_API_BASE, API key and model name. {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError, ValueError, AttributeError) as exc:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Streaming connection closed due to cancellation")
                raise _CancelledError()
            raise RuntimeError(
                f"Model call failed: cannot connect to model gateway {url}. Check OPENAI_API_BASE, proxy, certs and API key. Raw error: {exc}"
            ) from exc

    def _stream_chunks(self, data: dict) -> list[AIMessageChunk]:
        choices = data.get("choices") or []
        if not choices:
            return []
        first = choices[0] or {}
        delta = first.get("delta") or {}
        chunks: list[AIMessageChunk] = []
        if isinstance(delta, dict):
            chunks.extend(self._typed_value_chunks(delta.get("reasoning_content"), "reasoning"))
            chunks.extend(self._typed_value_chunks(delta.get("reasoning"), "reasoning"))
            chunks.extend(self._typed_value_chunks(delta.get("thinking"), "reasoning"))
            chunks.extend(self._typed_value_chunks(delta.get("content"), "content"))
            chunks.extend(self._typed_value_chunks(delta.get("text"), "content"))
            if delta.get("tool_calls"):
                chunks.append(AIMessageChunk(content="", tool_call_chunks=self._openai_tool_call_chunks(delta.get("tool_calls") or [])))
            if chunks:
                return chunks
        message = first.get("message") or {}
        if isinstance(message, dict):
            chunks.extend(self._typed_value_chunks(message.get("reasoning_content"), "reasoning"))
            chunks.extend(self._typed_value_chunks(message.get("reasoning"), "reasoning"))
            chunks.extend(self._typed_value_chunks(message.get("thinking"), "reasoning"))
            chunks.extend(self._typed_value_chunks(message.get("content"), "content"))
            if message.get("tool_calls"):
                chunks.append(AIMessageChunk(content="", tool_call_chunks=self._openai_tool_call_chunks(self._normalize_tool_calls(message.get("tool_calls") or []))))
            if chunks:
                return chunks
        text = first.get("text")
        return [AIMessageChunk(content=text)] if isinstance(text, str) else []

    def _normalize_tool_calls(self, raw_tool_calls: list[dict]) -> list[dict]:
        tool_calls = []
        for index, tc in enumerate(raw_tool_calls):
            func = tc.get("function") or {}
            tool_calls.append({
                "id": tc.get("id") or f"call_{index}",
                "type": tc.get("type") or "function",
                "function": {
                    "name": func.get("name") or "",
                    "arguments": func.get("arguments") or "{}",
                },
            })
        return tool_calls

    def _typed_value_chunks(self, value, default_type: str) -> list[AIMessageChunk]:
        if not value:
            return []
        if isinstance(value, str):
            if default_type == "reasoning":
                return [AIMessageChunk(content="", additional_kwargs={"reasoning_content": value})]
            return [AIMessageChunk(content=value)]
        if isinstance(value, list):
            chunks: list[AIMessageChunk] = []
            for item in value:
                if isinstance(item, str):
                    chunks.extend(self._typed_value_chunks(item, default_type))
                    continue
                if not isinstance(item, dict):
                    continue
                text = self._stream_item_text(item)
                if not text:
                    continue
                item_type = str(item.get("type") or default_type).lower()
                chunk_type = "reasoning" if ("reason" in item_type or "thinking" in item_type) else default_type
                chunks.extend(self._typed_value_chunks(text, chunk_type))
            return chunks
        return []

    def _stream_item_text(self, item: dict) -> str:
        for key in ("text", "content", "reasoning_content", "reasoning", "thinking"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        return ""

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
