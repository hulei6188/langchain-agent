from __future__ import annotations

import json
import time
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from core.db.models import Tool
from core.runtime.persistence import persist_intermediate_message
from core.runtime.skill_runtime import skill_loader_tool
from core.runtime.status import merge_web_search_tool_result, search_status_event
from core.services.tools import build_langchain_tool, tool_call_event


class ToolGraphState(MessagesState, total=False):
    context: dict[str, Any]
    node: dict[str, Any]
    bound_tools: list[Tool]
    langchain_tools: list[Any]
    allowed_names: set[str]
    total_calls: int
    tools_used: list[str]
    events: list[dict[str, Any]]
    web_sources: list[dict[str, Any]]
    search_status: dict[str, Any]
    round_index: int
    tool_loop_start: float
    max_tool_calls: int
    max_tool_wall_time: int
    pending_calls: list[dict[str, Any]]
    last_tool_error_signature: str
    consecutive_tool_error_count: int
    force_final_answer: bool
    output: dict[str, Any]


def runtime_tools(db: Session, agent, context: dict | None = None) -> list[Tool]:
    tool_ids = getattr(agent, "tool_ids", []) or []
    tools: list[Tool] = []
    if tool_ids:
        tools = (
            db.query(Tool)
            .filter(Tool.id.in_(tool_ids), Tool.enabled.is_(True))
            .order_by(Tool.id.asc())
            .all()
        )
    if context and context.get("search_enabled"):
        from core.services.bootstrap import ensure_builtin_tools

        ensure_builtin_tools(db)
        existing_ids = {tool.id for tool in tools}
        search_tool = (
            db.query(Tool)
            .filter(Tool.name == "web_search", Tool.type == "builtin_search", Tool.enabled.is_(True))
            .first()
        )
        if search_tool and search_tool.id not in existing_ids:
            tools.append(search_tool)
    return tools


def tool_runtime_bindings(db: Session, agent, context: dict, allowed_names: set[str]) -> tuple[list[Tool], list[Any]]:
    bound_tools = runtime_tools(db, agent, context)
    if allowed_names:
        bound_tools = [tool for tool in bound_tools if tool.name in allowed_names or tool.type == "builtin_search"]
    langchain_tools = [
        build_langchain_tool(
            tool,
            session_key=str(context.get("session_id") or ""),
            agent_workdir=context.get("agent_workdir"),
        )
        for tool in bound_tools
    ]
    loader_tool = skill_loader_tool(db, agent, context)
    if loader_tool:
        langchain_tools.append(loader_tool)
    return bound_tools, langchain_tools


def invoke_toolnode(state: ToolGraphState) -> tuple[list[ToolMessage], dict[str, dict]]:
    messages = list(state.get("messages") or [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return [], {}
    captured_results: dict[str, dict] = {}

    def wrap_tool_call(request, handler):
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or "")
        started = time.monotonic()
        try:
            result = request.tool.invoke(tool_call.get("args") or {})
            if not isinstance(result, dict):
                result = {"tool": tool_name, "content": str(result), "result_preview": str(result)}
            result["latency_ms"] = result.get("latency_ms", int((time.monotonic() - started) * 1000))
        except ValueError as exc:
            result = {
                "tool_name": tool_name,
                "error": str(exc),
                "content": f"Error: {exc}",
                "result_preview": "",
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
        except Exception as exc:
            result = {
                "tool_name": tool_name,
                "error": str(exc),
                "content": f"Error: {exc}",
                "result_preview": "",
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
        captured_results[tool_call_id] = result
        status = "error" if result.get("error") or result.get("status") == "error" else "success"
        return ToolMessage(
            content=result.get("content") or result.get("result_preview") or "",
            name=tool_name,
            tool_call_id=tool_call_id,
            status=status,
        )

    tool_node = ToolNode(state.get("langchain_tools") or [], wrap_tool_call=wrap_tool_call)
    output = tool_node.invoke({"messages": [messages[-1]]})
    tool_messages = [message for message in output.get("messages", []) if isinstance(message, ToolMessage)]
    return tool_messages, captured_results


def tool_jobs(state: ToolGraphState) -> list[dict]:
    context = state["context"]
    jobs = []
    for tc in state.get("pending_calls") or []:
        tool_name = str(tc.get("name") or "")
        tool_args = tc.get("args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {"input": tool_args}
        is_skill_loader = tool_name == "load_skill"
        matching = next((tool for tool in state.get("bound_tools", []) if tool.name == tool_name), None)
        job = {
            "tc": tc,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "matching": matching,
            "_session_key": str(context.get("session_id") or ""),
            "_agent_workdir": context.get("agent_workdir"),
            "internal": is_skill_loader,
        }
        jobs.append(job)
    return jobs


def record_tool_job_result(
    db: Session,
    job: dict,
    result: dict,
    *,
    context: dict,
    node: dict,
    round_index: int,
    messages: list[BaseMessage],
    events: list[dict],
    web_sources: list[dict],
    search_status: dict,
    stream: bool,
    writer: Callable[[dict], None] | None,
) -> tuple[bool, list[dict], dict]:
    tc = job["tc"]
    tool_name = job["tool_name"]
    tool_args = job["tool_args"]
    matching = job.get("matching")
    loaded_skill = False

    if job.get("internal") and tool_name == "load_skill":
        display_tool = type("_", (), {"id": None, "name": tool_name, "type": "internal"})()
        status = "error" if result.get("status") == "error" else "success"
        event_data = tool_call_event(
            display_tool,
            result,
            status=status,
            input_preview=json.dumps(tool_args, ensure_ascii=False),
            error_code="skill_not_loadable" if status == "error" else None,
        )
        tool_content = result.get("content") or result.get("result_preview") or ""
        loaded_skill = status == "success"
    elif matching:
        if result.get("error"):
            event_data = tool_call_event(
                matching,
                result,
                status="error",
                input_preview=json.dumps(tool_args, ensure_ascii=False),
                error_code="tool_error",
            )
            tool_content = result.get("content") or f"Error: {result['error']}"
        else:
            event_data = tool_call_event(matching, result, input_preview=json.dumps(tool_args, ensure_ascii=False))
            tool_content = result.get("content") or result.get("result_preview") or ""
    else:
        display_tool = type("_", (), {"id": None, "name": tool_name, "type": "unknown"})()
        event_data = tool_call_event(
            display_tool,
            {"tool": tool_name, "content": "", "result_preview": "", "latency_ms": 0},
            status="error",
            input_preview="{}",
            error_code="tool_not_found",
        )
        tool_content = f"Tool '{tool_name}' not found"

    if stream:
        stream_event_data = {**event_data, "type": "tool_call_result", "tool_call_id": tc.get("id") or ""}
        if writer:
            writer({"event": "tool_call_result", "data": stream_event_data})
    else:
        stream_event_data = event_data
        events.append({"event": "tool_call", "data": event_data})

    if matching and not result.get("error") and matching.type == "builtin_search":
        web_sources, search_status = merge_web_search_tool_result(web_sources, search_status, result)
        search_event = search_status_event(search_status)
        if stream:
            search_event["tool_call_id"] = tc.get("id") or ""
            if writer:
                writer({"event": "search_status", "data": search_event})
        else:
            events.append({"event": "search_status", "data": search_event})

    messages.append(ToolMessage(content=tool_content, tool_call_id=tc["id"], name=tool_name))
    persist_intermediate_message(
        db,
        context,
        role="tool",
        content=tool_content,
        tool_call_id=tc["id"],
        tool_name=tool_name,
        meta={**stream_event_data, "node_id": node["id"], "round": round_index, "kind": "tool_result"},
    )
    return loaded_skill, web_sources, search_status
