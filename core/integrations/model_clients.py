from __future__ import annotations

import hashlib
import json
import socket
import ssl
import urllib.error
import urllib.request

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from core.config import get_settings


DASHSCOPE_COMPATIBLE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_COMPATIBLE_DEFAULT_BASE = DASHSCOPE_COMPATIBLE_BASE


def api_key(settings, runtime_config: dict | None = None, *, purpose: str = "chat") -> str | None:
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


def api_base(settings, runtime_config: dict | None = None, *, purpose: str = "chat") -> str:
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


def post_json(url: str, payload: dict, key: str, *, timeout_seconds: int = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
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


class OpenAICompatibleEmbeddings(Embeddings):
    """LangChain embeddings adapter for OpenAI-compatible embedding endpoints."""

    def __init__(self) -> None:
        self.last_mock = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        key = api_key(settings, purpose="embedding")
        if settings.mock_llm:
            self.last_mock = True
            return [self._mock_embedding(text) for text in texts]
        if not key:
            raise RuntimeError("Embedding API key is not configured")
        self.last_mock = False
        return self._langchain_embeddings(settings, key=key).embed_documents(texts)

    def embed_query(self, text: str, *, runtime_config: dict | None = None) -> list[float]:
        settings = get_settings()
        key = api_key(settings, runtime_config, purpose="embedding")
        if settings.mock_llm:
            self.last_mock = True
            return self._mock_embedding(text)
        if not key:
            raise RuntimeError("Embedding API key is not configured")
        self.last_mock = False
        return self._langchain_embeddings(settings, key=key, runtime_config=runtime_config).embed_query(text)

    def _langchain_embeddings(
        self,
        settings,
        *,
        key: str,
        runtime_config: dict | None = None,
    ) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            api_key=key,
            base_url=api_base(settings, runtime_config, purpose="embedding").rstrip("/") or None,
            model=settings.openai_embedding_model,
            max_retries=0,
            timeout=60,
        )

    def _mock_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [((digest[i % len(digest)] / 255.0) * 2) - 1 for i in range(32)]


class OpenAICompatibleReranker:
    """Thin rerank client for OpenAI-compatible DashScope-style rerank endpoints."""

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None, model: str | None = None) -> list[dict]:
        settings = get_settings()
        key = api_key(settings, purpose="rerank")
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
        if not key:
            raise RuntimeError("Rerank API key is not configured")

        url = api_base(settings, purpose="rerank").rstrip("/") + "/rerank"
        payload = {
            "model": model or settings.rag_rerank_model,
            "query": query,
            "documents": documents,
            **({"top_n": top_n} if top_n else {}),
        }
        data = post_json(url, payload, key)
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
