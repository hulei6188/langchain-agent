from __future__ import annotations

from core.services import web_search as web_search_service


def reasoning_label(reasoning_type: str) -> str:
    return {"native": "深度思考", "prompt": "提示词增强", "none": "不支持"}.get(reasoning_type, "不支持")


def thinking_status(model, requested: bool | None) -> dict:
    reasoning_type = str(getattr(model, "reasoning_type", "none") or "none")
    if reasoning_type not in {"native", "prompt", "none"}:
        reasoning_type = "none"
    supports_reasoning = bool(getattr(model, "supports_reasoning", False)) and reasoning_type != "none"
    label = str(getattr(model, "reasoning_label", "") or reasoning_label(reasoning_type))

    if not requested:
        return {
            "enabled": False,
            "requested": False,
            "type": reasoning_type,
            "label": label,
            "reason": "not_requested",
        }
    if not supports_reasoning:
        return {
            "enabled": False,
            "requested": True,
            "type": "none",
            "label": reasoning_label("none"),
            "reason": "model_not_supported",
        }
    return {
        "enabled": True,
        "requested": True,
        "type": reasoning_type,
        "label": label,
        "reason": "enabled",
    }


def thinking_messages(context: dict) -> list[dict]:
    status = context.get("thinking_status") or {}
    if not status.get("enabled"):
        return []
    if status.get("type") == "prompt":
        return [
            {
                "role": "system",
                "content": (
                    "本轮已开启深度思考模式，但当前模型使用提示词增强，不是原生推理。"
                    "请先进行更周全的分析，检查关键假设、约束、风险和反例，再给出清晰答案。"
                    "不要输出隐藏推理链，只输出必要的结论、依据和可执行步骤。"
                ),
            }
        ]
    return [
        {
            "role": "system",
            "content": "本轮已开启原生深度思考能力。请给出经过审慎推理后的答案，不要输出隐藏推理链。",
        }
    ]


def search_status(query: str, requested: bool | None) -> dict:
    runtime = web_search_service.web_search_status()
    provider = runtime.get("provider", "duckduckgo_html")
    if not requested:
        return {
            "enabled": False,
            "requested": False,
            "query": query,
            "provider": provider,
            "matched_results": 0,
            "sources_emitted": False,
            "items": [],
            "sources": [],
            "reason": "not_requested",
        }
    if not runtime.get("configured"):
        return {
            "enabled": False,
            "requested": True,
            "query": query,
            "provider": provider,
            "matched_results": 0,
            "sources_emitted": False,
            "items": [],
            "sources": [],
            "reason": "web_search_unavailable",
        }
    return {
        "enabled": True,
        "requested": True,
        "query": query,
        "provider": provider,
        "matched_results": 0,
        "sources_emitted": False,
        "items": [],
        "sources": [],
        "reason": "tool_available",
    }


def search_status_event(status: dict) -> dict:
    return {key: value for key, value in status.items() if key != "sources"}


def merge_web_search_tool_result(current_sources: list[dict], status: dict, result: dict) -> tuple[list[dict], dict]:
    result_json = result.get("result_json") if isinstance(result.get("result_json"), dict) else {}
    items = result_json.get("items") or []
    next_sources = [dict(item) for item in current_sources]
    new_sources = web_search_service.search_items_as_sources(items)
    offset = len(next_sources)
    for index, source in enumerate(new_sources, start=offset + 1):
        source["source_id"] = f"web-{index}"
        source["chunk_id"] = f"web-search-{index}"
        next_sources.append(source)
    next_status = {
        **status,
        "enabled": bool(next_sources),
        "requested": True,
        "query": result_json.get("query") or status.get("query") or "",
        "provider": result_json.get("provider") or status.get("provider") or "duckduckgo_html",
        "matched_results": len(next_sources),
        "sources_emitted": bool(next_sources),
        "items": [*(status.get("items") or []), *items],
        "sources": next_sources,
        "latency_ms": result.get("latency_ms", status.get("latency_ms", 0)),
        "reason": "tool_called" if next_sources else "no_results",
    }
    return next_sources, next_status


def web_source_text(sources: list[dict]) -> str:
    lines = []
    for index, item in enumerate(sources, start=1):
        title = item.get("title") or f"Result {index}"
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""
        lines.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)
