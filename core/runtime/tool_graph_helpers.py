from __future__ import annotations

import json
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from core.runtime.dsml import strip_or_block_leaked_tool_markup
from core.runtime.message_utils import message_content_text, message_reasoning_content


def tool_direct_output(state: dict, response: AIMessage, *, stream: bool) -> dict:
    response_content = message_content_text(response)
    response_reasoning = message_reasoning_content(response)
    output = {
        "draft": strip_or_block_leaked_tool_markup(response_content),
        "draft_reasoning": response_reasoning,
        "web_sources": state.get("web_sources", []),
        "search_status": state.get("search_status", {}),
        "tool_outputs": [],
        "tool_stats": {"total_calls": int(state.get("total_calls") or 0), "tools_used": list(state.get("tools_used") or [])},
        "events": list(state.get("events") or []),
    }
    if stream:
        output["draft_streamed"] = True
        output["draft_reasoning_streamed"] = bool(response_reasoning and state["context"].get("thinking_enabled"))
    return output


def tool_final_output(state: dict, response: AIMessage, *, stream: bool, max_rounds_reached: bool) -> dict:
    response_content = message_content_text(response)
    response_reasoning = message_reasoning_content(response)
    output = {
        "draft": strip_or_block_leaked_tool_markup(response_content),
        "draft_reasoning": response_reasoning,
        "web_sources": state.get("web_sources", []),
        "search_status": state.get("search_status", {}),
        "tool_outputs": [],
        "tool_stats": {
            "total_calls": int(state.get("total_calls") or 0),
            "tools_used": list(state.get("tools_used") or []),
            "max_rounds_reached": max_rounds_reached,
        },
        "events": list(state.get("events") or []),
    }
    if stream:
        output["draft_streamed"] = bool(response_content)
        output["draft_reasoning_streamed"] = bool(response_reasoning and state["context"].get("thinking_enabled"))
    return output


def tool_limits_reached(state: dict, *, max_tool_calls: int, max_tool_rounds: int) -> bool:
    if int(state.get("total_calls") or 0) >= int(state.get("max_tool_calls") or max_tool_calls):
        return True
    if int(state.get("round_index") or 0) >= max_tool_rounds:
        return True
    return (time.monotonic() - float(state.get("tool_loop_start") or time.monotonic())) > int(state.get("max_tool_wall_time") or 0)


def tool_message_fallback_result(job: dict, tool_message: ToolMessage | None) -> dict:
    content = tool_message.content if tool_message else f"Tool '{job['tool_name']}' not found"
    if tool_message and tool_message.status == "error":
        error_code = "tool_not_found" if not job.get("matching") and not job.get("internal") else "tool_error"
        return {"error": error_code, "content": content, "result_preview": content[:500], "latency_ms": 0}
    return {"content": content, "result_preview": str(content)[:500], "latency_ms": 0}


def tool_call_start_event(tool, *, tool_name: str, tool_call_id: str, input_preview: str = "") -> dict:
    return {
        "type": "tool_call_start",
        "tool_call_id": tool_call_id,
        "tool_id": tool.id if tool else None,
        "tool_name": tool.name if tool else tool_name,
        "tool_type": tool.type if tool else "unknown",
        "status": "running",
        "input_preview": input_preview[:500],
        "result_preview": "",
        "latency_ms": 0,
        "error_code": "",
    }


def tool_job_start_event(job: dict) -> dict:
    display_tool = job.get("matching")
    if not display_tool and job.get("tool_name") == "load_skill":
        display_tool = SimpleNamespace(id=None, name=job["tool_name"], type="internal")
    return tool_call_start_event(
        display_tool,
        tool_name=job["tool_name"],
        tool_call_id=job["tc"].get("id") or "",
        input_preview=json.dumps(job["tool_args"], ensure_ascii=False),
    )
