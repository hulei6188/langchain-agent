from __future__ import annotations

import re
import time
from collections.abc import Callable

from fastapi import APIRouter
from sqlalchemy import text

from core.config import get_settings
from core.db.session import engine
from core.integrations.llm import DASHSCOPE_COMPATIBLE_BASE, OPENAI_COMPATIBLE_DEFAULT_BASE, OpenAICompatibleProvider
from core.integrations.vector_store import vector_store
from core.security.api_keys import secret_storage_ready
from core.services.rag_cache import redis_store
from core.services.web_search import web_search_status


_health_probe_cache: dict[str, tuple[float, dict]] = {}


def create_health_router(startup_error_getter: Callable[[], str | None]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health():
        settings = get_settings()
        provider = OpenAICompatibleProvider()
        chat_api_key = provider._api_key(settings, purpose="chat")
        embedding_api_key = provider._api_key(settings, purpose="embedding")
        model_mock = settings.mock_llm
        model_base = settings.openai_api_base
        if settings.deepseek_api_key and (
            (settings.openai_api_base or "").rstrip("/") == settings.deepseek_api_base.rstrip("/")
            or settings.openai_model == settings.deepseek_model
        ):
            model_base = settings.deepseek_api_base
        elif settings.dashscope_api_key and not settings.openai_api_key and model_base == OPENAI_COMPATIBLE_DEFAULT_BASE:
            model_base = DASHSCOPE_COMPATIBLE_BASE
        embedding_base = provider._api_base(settings, purpose="embedding")
        embedding_mock = settings.mock_llm
        embedding_model = (settings.openai_embedding_model or "").strip()
        issues = []
        database_status = {"configured": bool(settings.database_url), "available": False, "error": None}
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            database_status["available"] = True
        except Exception as exc:
            database_status["error"] = str(exc)[:240]
            issues.append("Database is configured but not reachable.")
        if not secret_storage_ready():
            issues.append("API_KEY_ENCRYPTION_KEY is required before storing user model keys or tool secrets.")
        startup_error = startup_error_getter()
        if startup_error:
            issues.append("Database initialization failed during startup.")
        redis_status = redis_store.status()
        vector_status = vector_store.status()
        model_probe = _model_probe("chat", enabled=bool(settings.health_model_probe_enabled and not model_mock))
        embedding_probe = _model_probe("embedding", enabled=bool(settings.health_model_probe_enabled and not embedding_mock and embedding_model))
        if model_mock:
            issues.append("Chat model is running in mock mode because AGENTBASE_MOCK_LLM is true.")
        elif not chat_api_key:
            issues.append("Chat model API key is not configured.")
        elif not model_probe["ok"]:
            issues.append("Chat model gateway probe failed.")
        if embedding_mock:
            issues.append("Embedding is running in mock mode because AGENTBASE_MOCK_LLM is true.")
        elif not embedding_model or not embedding_api_key:
            issues.append("Embedding is unavailable for real RAG because OPENAI_EMBEDDING_MODEL and a provider API key are required.")
        elif not embedding_probe["ok"]:
            issues.append("Embedding gateway probe failed.")
        if redis_status["required"] and not redis_status["available"]:
            issues.append("Redis is configured for RAG cache/job state but is not reachable.")
        if vector_status["fallback"]:
            issues.append("Milvus is configured but unavailable; vector operations are using the in-memory fallback.")
        return {
            "status": "degraded" if issues else "ok",
            "version": settings.app_version,
            "issues": issues,
            "dependencies": {
                "database": database_status,
                "startup": {"ok": startup_error is None, "error": startup_error},
                "cors": {"origins": settings.cors_origin_list},
                "redis": redis_status,
                "vector_store": vector_status,
                "model": {
                    "provider": "openai-compatible",
                    "model": settings.deepseek_model if model_base.rstrip("/") == settings.deepseek_api_base.rstrip("/") else settings.openai_model,
                    "base_url": model_base,
                    "mock": model_mock,
                    "configured": bool(chat_api_key),
                    "available": bool((not model_mock) and bool(chat_api_key)),
                    "probe": model_probe,
                },
                "embedding": {
                    "provider": "openai-compatible",
                    "model": embedding_model,
                    "base_url": embedding_base,
                    "mock": embedding_mock,
                    "configured": bool(embedding_model and embedding_api_key),
                    "available": bool(embedding_model and embedding_api_key and not embedding_mock),
                    "reason": None
                    if bool(embedding_model and embedding_api_key and not embedding_mock)
                    else _runtime_unavailable_reason(embedding_probe, vector_status),
                    "probe": embedding_probe,
                },
                "web_search": web_search_status(),
                "secret_storage": {"configured": secret_storage_ready()},
            },
        }

    return router


def _model_probe(purpose: str, *, enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "ok": False, "error": None, "cached": False}
    now = time.monotonic()
    cached = _health_probe_cache.get(purpose)
    if cached and now - cached[0] < 300:
        return {**cached[1], "cached": True}
    provider = OpenAICompatibleProvider()
    try:
        if purpose == "chat":
            settings_obj = get_settings()
            use_deepseek = bool(
                settings_obj.deepseek_api_key
                and (
                    (settings_obj.openai_api_base or "").rstrip("/") == settings_obj.deepseek_api_base.rstrip("/")
                    or settings_obj.openai_model == settings_obj.deepseek_model
                )
            )
            model = settings_obj.deepseek_model if use_deepseek else settings_obj.openai_model
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "health"}],
                "temperature": 0,
                "stream": False,
            }
            provider._post_json(
                provider._api_base(settings_obj, purpose="chat").rstrip("/") + "/chat/completions",
                payload,
                provider._api_key(settings_obj, purpose="chat") or "",
                timeout_seconds=8,
            )
        elif purpose == "embedding":
            settings_obj = get_settings()
            provider._post_json(
                provider._api_base(settings_obj, purpose="embedding").rstrip("/") + "/embeddings",
                {"model": settings_obj.openai_embedding_model, "input": "health"},
                provider._api_key(settings_obj, purpose="embedding") or "",
                timeout_seconds=8,
            )
        else:
            raise ValueError("Unsupported health probe")
        result = {"enabled": True, "ok": True, "error": None, "cached": False}
    except Exception as exc:
        result = {"enabled": True, "ok": False, "error": _sanitize_public_error(str(exc)), "cached": False}
    _health_probe_cache[purpose] = (now, result)
    return result


def _runtime_unavailable_reason(probe: dict, vector_status: dict) -> str:
    if not vector_status.get("available"):
        return "vector_store_unavailable"
    if probe.get("enabled") and not probe.get("ok"):
        return "provider_probe_failed"
    return "mock_or_vector_unavailable"


def _sanitize_public_error(message: str) -> str:
    cleaned = re.sub(r"(?i)(sk-[A-Za-z0-9_-]+|api[_-]?key\s*[:=]\s*\S+|secret\s*[:=]\s*\S+)", "[secret]", str(message))
    return cleaned.replace("\n", " ").replace("\r", " ").strip()[:500]
