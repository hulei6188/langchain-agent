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
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.config import get_settings

logger = logging.getLogger(__name__)

DASHSCOPE_COMPATIBLE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_COMPATIBLE_DEFAULT_BASE = DASHSCOPE_COMPATIBLE_BASE


class _CancelledError(Exception):
    """Raised when a chat request is cancelled mid-flight."""
    pass


@dataclass
class ChatResponse:
    content: str | None = None
    reasoning_content: str = ""
    tool_calls: list[dict] | None = None


@dataclass
class ChatStreamChunk:
    type: str
    content: str = ""
    tool_calls: list[dict] | None = None


class OpenAICompatibleProvider:
    """Small OpenAI-compatible client with function calling support."""

    def __init__(self) -> None:
        self.last_chat_mock = False
        self.last_embed_mock = False
        self._active_response: http.client.HTTPResponse | None = None

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

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        runtime_config: dict | None = None,
        tools: list[dict] | None = None,
        thinking_enabled: bool | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatResponse:
        if cancel_event is not None and cancel_event.is_set():
            raise _CancelledError()
        settings = get_settings()
        api_key = self._api_key(settings, runtime_config, purpose="chat")
        if settings.mock_llm:
            self.last_chat_mock = True
            user_text = self._content_text(next((m["content"] for m in reversed(messages) if m.get("role") == "user"), ""))
            context_hint = " ".join(self._content_text(m.get("content", ""))[:160] for m in messages if m.get("role") == "system")
            if tools:
                tool_names = [t.get("function", {}).get("name", "") for t in tools]
                return ChatResponse(
                    tool_calls=[{
                        "id": f"mock_call_{hashlib.md5(user_text.encode()).hexdigest()[:8]}",
                        "type": "function",
                        "function": {"name": tool_names[0], "arguments": json.dumps({"query": user_text[:120]}, ensure_ascii=False)},
                    }]
                )
            return ChatResponse(content=f"Mock answer for: {user_text}\n\nContext summary: {context_hint[:220]}")
        if not api_key:
            raise RuntimeError("Chat model API key is not configured")
        self.last_chat_mock = False

        api_base = self._api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        url = api_base.rstrip("/") + "/chat/completions"
        payload: dict = {
            "model": chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        self._apply_thinking_payload(payload, api_base=api_base, model=chat_model, thinking_enabled=thinking_enabled)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._post_json(url, payload, api_key)
        return self._parse_chat_response(data)

    def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        runtime_config: dict | None = None,
        tools: list[dict] | None = None,
        thinking_enabled: bool | None = None,
    ) -> Iterable[str]:
        for chunk in self.chat_stream_events(
            messages,
            model=model,
            temperature=temperature,
            runtime_config=runtime_config,
            tools=tools,
            thinking_enabled=thinking_enabled,
        ):
            if chunk.type == "content":
                yield chunk.content
            elif chunk.type == "tool_calls":
                yield json.dumps({"tool_calls": chunk.tool_calls or []}, ensure_ascii=False)

    def chat_stream_events(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        runtime_config: dict | None = None,
        tools: list[dict] | None = None,
        thinking_enabled: bool | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterable[ChatStreamChunk]:
        settings = get_settings()
        api_key = self._api_key(settings, runtime_config, purpose="chat")
        if settings.mock_llm:
            self.last_chat_mock = True
            text = self.chat(
                messages,
                model=model,
                temperature=temperature,
                runtime_config=runtime_config,
                tools=tools,
                thinking_enabled=thinking_enabled,
            )
            if text.tool_calls:
                yield ChatStreamChunk(type="tool_calls", tool_calls=text.tool_calls)
                return
            for index in range(0, len(text.content or ""), 24):
                if cancel_event is not None and cancel_event.is_set():
                    return
                yield ChatStreamChunk(type="content", content=(text.content or "")[index : index + 24])
            return
        if not api_key:
            raise RuntimeError("Chat model API key is not configured")
        self.last_chat_mock = False

        api_base = self._api_base(settings, runtime_config, purpose="chat")
        chat_model = model or (runtime_config or {}).get("chat_model") or settings.openai_model
        url = api_base.rstrip("/") + "/chat/completions"
        payload: dict = {
            "model": chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        self._apply_thinking_payload(payload, api_base=api_base, model=chat_model, thinking_enabled=thinking_enabled)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # When tools are present, stream normally — the caller (agent loop)
        # uses non-streaming chat() for tool-call decisions, so stream is
        # only for the final answer phase.
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

    def _parse_chat_response(self, data: dict) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        reasoning_content = str(message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or "")
        raw_tool_calls = message.get("tool_calls") or []
        if raw_tool_calls:
            tool_calls = self._normalize_tool_calls(raw_tool_calls)
            return ChatResponse(content=content or None, reasoning_content=reasoning_content, tool_calls=tool_calls)
        return ChatResponse(content=str(content) if content else "", reasoning_content=reasoning_content)

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

    def _post_json_stream(self, url: str, payload: dict, api_key: str, *, cancel_event: threading.Event | None = None) -> Iterable[ChatStreamChunk]:
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
                            break
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
                return
            raise RuntimeError(
                f"Model call failed: cannot connect to model gateway {url}. Check OPENAI_API_BASE, proxy, certs and API key. Raw error: {exc}"
            ) from exc

    def _stream_delta(self, data: dict) -> str:
        return "".join(chunk.content for chunk in self._stream_chunks(data) if chunk.type == "content")

    def _stream_chunks(self, data: dict) -> list[ChatStreamChunk]:
        choices = data.get("choices") or []
        if not choices:
            return []
        first = choices[0] or {}
        delta = first.get("delta") or {}
        chunks: list[ChatStreamChunk] = []
        if isinstance(delta, dict):
            chunks.extend(self._typed_value_chunks(delta.get("reasoning_content"), "reasoning"))
            chunks.extend(self._typed_value_chunks(delta.get("reasoning"), "reasoning"))
            chunks.extend(self._typed_value_chunks(delta.get("thinking"), "reasoning"))
            chunks.extend(self._typed_value_chunks(delta.get("content"), "content"))
            chunks.extend(self._typed_value_chunks(delta.get("text"), "content"))
            if delta.get("tool_calls"):
                chunks.append(ChatStreamChunk(type="tool_call_delta", tool_calls=delta.get("tool_calls") or []))
            if chunks:
                return chunks
        message = first.get("message") or {}
        if isinstance(message, dict):
            chunks.extend(self._typed_value_chunks(message.get("reasoning_content"), "reasoning"))
            chunks.extend(self._typed_value_chunks(message.get("reasoning"), "reasoning"))
            chunks.extend(self._typed_value_chunks(message.get("thinking"), "reasoning"))
            chunks.extend(self._typed_value_chunks(message.get("content"), "content"))
            if message.get("tool_calls"):
                chunks.append(ChatStreamChunk(type="tool_calls", tool_calls=self._normalize_tool_calls(message.get("tool_calls") or [])))
            if chunks:
                return chunks
        text = first.get("text")
        return [ChatStreamChunk(type="content", content=text)] if isinstance(text, str) else []

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

    def _typed_value_chunks(self, value, default_type: str) -> list[ChatStreamChunk]:
        if not value:
            return []
        if isinstance(value, str):
            return [ChatStreamChunk(type=default_type, content=value)]
        if isinstance(value, list):
            chunks: list[ChatStreamChunk] = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(ChatStreamChunk(type=default_type, content=item))
                    continue
                if not isinstance(item, dict):
                    continue
                text = self._stream_item_text(item)
                if not text:
                    continue
                item_type = str(item.get("type") or default_type).lower()
                chunk_type = "reasoning" if ("reason" in item_type or "thinking" in item_type) else default_type
                chunks.append(ChatStreamChunk(type=chunk_type, content=text))
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
