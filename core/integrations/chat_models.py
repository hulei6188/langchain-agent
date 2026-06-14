from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as openai_chat_base


_REASONING_RESPONSE_KEYS = (
    "reasoning_content",
    "reasoning",
    "thinking",
    "reasoning_text",
    "thinking_content",
)


def _install_reasoning_content_patch() -> None:
    if getattr(openai_chat_base, "_agentbase_reasoning_content_patch", False):
        return

    original_delta_converter = openai_chat_base._convert_delta_to_message_chunk
    original_message_converter = openai_chat_base._convert_dict_to_message

    def convert_delta_with_reasoning(payload, default_class):
        message = original_delta_converter(payload, default_class)
        _copy_reasoning_content(message, payload)
        return message

    def convert_message_with_reasoning(payload):
        message = original_message_converter(payload)
        _copy_reasoning_content(message, payload)
        return message

    openai_chat_base._convert_delta_to_message_chunk = convert_delta_with_reasoning
    openai_chat_base._convert_dict_to_message = convert_message_with_reasoning
    openai_chat_base._agentbase_reasoning_content_patch = True


def _copy_reasoning_content(message, payload) -> None:
    reasoning = _first_reasoning_value(payload)
    if not reasoning:
        return
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        additional_kwargs.setdefault("reasoning_content", reasoning)


def _first_reasoning_value(payload):
    for key in _REASONING_RESPONSE_KEYS:
        if isinstance(payload, dict) and payload.get(key):
            return payload[key]
    return ""


_install_reasoning_content_patch()


def create_chat_openai(
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    thinking_enabled: bool | None,
    streaming: bool,
    runtime_config: dict | None = None,
) -> tuple[ChatOpenAI, httpx.Client]:
    timeout = httpx.Timeout(120.0 if streaming else 60.0)
    http_client = httpx.Client(timeout=timeout)
    model_kwargs = chat_model_kwargs(
        api_base=api_base,
        model=model,
        thinking_enabled=thinking_enabled,
        runtime_config=runtime_config,
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


def chat_model_kwargs(
    *,
    api_base: str,
    model: str,
    thinking_enabled: bool | None,
    runtime_config: dict | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    if is_deepseek(api_base, model):
        enabled = bool(thinking_enabled)
        extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if enabled:
            kwargs["reasoning_effort"] = "high"
    elif thinking_enabled and is_openai_reasoning_model(api_base, model):
        kwargs["reasoning_effort"] = "high"
    elif thinking_enabled is not None and is_dashscope_qwen(api_base, model):
        extra_body["enable_thinking"] = bool(thinking_enabled)
    elif thinking_enabled is not None and runtime_reasoning_type(runtime_config) == "native":
        extra_body["enable_thinking"] = bool(thinking_enabled)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def requires_reasoning_replay(*, api_base: str, model: str) -> bool:
    return is_deepseek(api_base, model)


def is_openai_reasoning_model(api_base: str, model: str) -> bool:
    normalized_model = (model or "").lower().strip()
    if not normalized_model:
        return False
    openai_reasoning_prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4",
    )
    return normalized_model.startswith(openai_reasoning_prefixes)


def is_dashscope_qwen(api_base: str, model: str) -> bool:
    normalized_base = (api_base or "").lower()
    normalized_model = (model or "").lower()
    return (
        ("dashscope.aliyuncs.com" in normalized_base or "dashscope-intl.aliyuncs.com" in normalized_base)
        and normalized_model.startswith("qwen")
    )


def is_deepseek(api_base: str, model: str) -> bool:
    normalized_base = (api_base or "").lower()
    if "api.deepseek.com" in normalized_base:
        return True
    normalized_model = (model or "").lower().replace("_", "-").strip()
    model_segments = [segment for segment in re.split(r"[/:\s]+", normalized_model) if segment]
    return any(segment == "deepseek" or segment.startswith("deepseek-") for segment in model_segments)


def runtime_reasoning_type(runtime_config: dict | None) -> str:
    reasoning_type = str((runtime_config or {}).get("reasoning_type") or "none").strip().lower()
    return reasoning_type if reasoning_type in {"native", "prompt", "none"} else "none"
