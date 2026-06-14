from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from sqlalchemy.orm import Session

from core.runtime.dsml import dsml_tool_call_parser
from core.runtime.message_utils import message_content_text, message_reasoning_content
from core.runtime.persistence import persist_intermediate_message
from core.runtime.prompting import build_llm_messages
from core.runtime.streaming import stream_chat_response_to_writer
from core.runtime.tool_graph_helpers import (
    tool_final_output,
    tool_job_start_event,
    tool_limits_reached,
    tool_message_fallback_result,
)
from core.runtime.tool_runtime import (
    ToolGraphState,
    invoke_toolnode,
    record_tool_job_result,
    tool_jobs,
    tool_runtime_bindings,
)


MAX_TOOL_CALLS_PER_RUN = 200
MAX_TOOL_ROUNDS_PER_RUN = 50
MAX_TOOL_WALL_TIME_SECONDS = 1800


class ToolLoopRunner:
    """Run the LangGraph tool loop for one workflow Tool node."""

    def __init__(
        self,
        db: Session,
        provider,
        *,
        cancel_event,
        raise_if_cancelled: Callable[[], None],
    ) -> None:
        self.db = db
        self.provider = provider
        self.cancel_event = cancel_event
        self.raise_if_cancelled = raise_if_cancelled

    def run(
        self,
        agent,
        node: dict,
        context: dict,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None = None,
    ) -> dict:
        state = self._initial_state(agent, node, context)
        if state.get("output"):
            return state["output"]
        graph = self._build_graph(agent, stream=stream, writer=writer)
        final_state = graph.invoke(state)
        return final_state.get("output") or tool_final_output(
            final_state,
            AIMessage(content=""),
            stream=stream,
            max_rounds_reached=True,
        )

    def _initial_state(self, agent, node: dict, context: dict) -> ToolGraphState:
        tool_policy = agent.settings.get("tool_policy") or {}
        allowed_names = set(tool_policy.get("allowed_tool_names") or [])
        bound_tools, langchain_tools = tool_runtime_bindings(self.db, agent, context, allowed_names)
        if not langchain_tools:
            return {
                "context": context,
                "node": node,
                "output": {"tool_outputs": [], "tool_stats": {"total_calls": 0, "tools_used": []}},
            }
        return {
            "context": context,
            "node": node,
            "messages": build_llm_messages(agent, context),
            "bound_tools": bound_tools,
            "langchain_tools": langchain_tools,
            "allowed_names": allowed_names,
            "total_calls": 0,
            "tools_used": [],
            "events": [],
            "web_sources": list(context.get("web_sources", [])),
            "search_status": dict(context.get("search_status") or {}),
            "round_index": 0,
            "tool_loop_start": time.monotonic(),
            "max_tool_calls": MAX_TOOL_CALLS_PER_RUN,
            "max_tool_wall_time": MAX_TOOL_WALL_TIME_SECONDS,
            "pending_calls": [],
        }

    def _build_graph(
        self,
        agent,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ):
        graph_builder = StateGraph(ToolGraphState)
        graph_builder.add_node("call_model", lambda state: self._call_model(agent, state, stream=stream, writer=writer))
        graph_builder.add_node("tools", lambda state: self._execute_tools(agent, state, stream=stream, writer=writer))
        graph_builder.add_node("final_answer", lambda state: self._final_answer(agent, state, stream=stream, writer=writer))
        graph_builder.add_conditional_edges(
            START,
            lambda state: "final_answer"
            if tool_limits_reached(state, max_tool_calls=MAX_TOOL_CALLS_PER_RUN, max_tool_rounds=MAX_TOOL_ROUNDS_PER_RUN)
            else "call_model",
            {"call_model": "call_model", "final_answer": "final_answer"},
        )
        graph_builder.add_conditional_edges(
            "call_model",
            tools_condition,
            {"tools": "tools", "__end__": "final_answer"},
        )
        graph_builder.add_conditional_edges(
            "tools",
            lambda state: "final_answer"
            if tool_limits_reached(state, max_tool_calls=MAX_TOOL_CALLS_PER_RUN, max_tool_rounds=MAX_TOOL_ROUNDS_PER_RUN)
            else "call_model",
            {"call_model": "call_model", "final_answer": "final_answer"},
        )
        graph_builder.add_edge("final_answer", END)
        return graph_builder.compile(checkpointer=False)

    def _call_model(
        self,
        agent,
        state: ToolGraphState,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> ToolGraphState:
        self.raise_if_cancelled()
        state = dict(state)
        state.pop("output", None)
        context = state["context"]
        round_index = int(state.get("round_index") or 0)
        if stream:
            response = stream_chat_response_to_writer(
                self.provider,
                agent=agent,
                messages=state.get("messages", []),
                context=context,
                writer=writer,
                tools=state.get("langchain_tools") or [],
                provisional_stream=True,
                thinking_enabled=_thinking_request_value(context),
                cancel_event=self.cancel_event,
                raise_if_cancelled=self.raise_if_cancelled,
            )
            response = dsml_tool_call_parser.invoke(response, stage=f"tool node stream round {round_index}")
        else:
            response = self._invoke_chat_model(
                agent,
                state.get("messages", []),
                context,
                tools=state.get("langchain_tools") or [],
            )
            response = dsml_tool_call_parser.invoke(response, stage=f"tool node non-stream round {round_index}")

        state["pending_calls"] = []
        response_content = message_content_text(response)
        response_reasoning = message_reasoning_content(response)

        remaining_calls = max(
            0,
            int(state.get("max_tool_calls") or MAX_TOOL_CALLS_PER_RUN) - int(state.get("total_calls") or 0),
        )
        calls_this_round = (response.tool_calls or [])[:remaining_calls]
        events = list(state.get("events") or [])
        if response_reasoning and context.get("thinking_enabled") and not stream:
            reasoning_content = response_reasoning.strip()
            if reasoning_content:
                events.append({"event": "reasoning_token", "data": {"content": f"{reasoning_content}\n\n"}})
        messages = list(state.get("messages") or [])
        messages.append(
            AIMessage(
                content=response_content,
                additional_kwargs=response.additional_kwargs,
                tool_calls=calls_this_round,
            )
        )
        if calls_this_round:
            persist_intermediate_message(
                self.db,
                context,
                role="assistant",
                content=response_content,
                reasoning=response_reasoning,
                tool_calls=calls_this_round,
                meta={"node_id": state["node"]["id"], "round": round_index, "kind": "tool_calls"},
            )
        state["messages"] = messages
        state["events"] = events
        state["pending_calls"] = calls_this_round
        return state

    def _execute_tools(
        self,
        agent,
        state: ToolGraphState,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> ToolGraphState:
        self.raise_if_cancelled()
        state = dict(state)
        context = state["context"]
        node = state["node"]
        round_index = int(state.get("round_index") or 0)
        jobs = tool_jobs(state)
        if stream and writer:
            for job in jobs:
                writer({"event": "tool_call_start", "data": tool_job_start_event(job)})

        tool_messages, job_results = invoke_toolnode(state)
        tool_messages_by_id = {message.tool_call_id: message for message in tool_messages}

        messages = list(state.get("messages") or [])
        events = list(state.get("events") or [])
        tools_used = list(state.get("tools_used") or [])
        total_calls = int(state.get("total_calls") or 0)
        web_sources = list(state.get("web_sources") or [])
        search_status = dict(state.get("search_status") or {})
        loaded_skill_this_round = False

        for job in jobs:
            tool_call_id = job["tc"].get("id") or ""
            result = job_results.get(tool_call_id)
            if result is None:
                tool_message = tool_messages_by_id.get(tool_call_id)
                result = tool_message_fallback_result(job, tool_message)
            total_calls += 1
            tools_used.append(job["tool_name"])
            loaded, web_sources, search_status = record_tool_job_result(
                self.db,
                job,
                result,
                context=context,
                node=node,
                round_index=round_index,
                messages=messages,
                events=events,
                web_sources=web_sources,
                search_status=search_status,
                stream=stream,
                writer=writer,
            )
            loaded_skill_this_round = loaded_skill_this_round or loaded

        if loaded_skill_this_round:
            _refresh_system_message(messages, agent, context)
            bound_tools, langchain_tools = tool_runtime_bindings(self.db, agent, context, state.get("allowed_names") or set())
            state["bound_tools"] = bound_tools
            state["langchain_tools"] = langchain_tools

        state["messages"] = messages
        state["events"] = events
        state["tools_used"] = tools_used
        state["total_calls"] = total_calls
        state["web_sources"] = web_sources
        state["search_status"] = search_status
        state["round_index"] = round_index + 1
        state["pending_calls"] = []
        return state

    def _final_answer(
        self,
        agent,
        state: ToolGraphState,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> ToolGraphState:
        self.raise_if_cancelled()
        state = dict(state)
        latest = next(
            (message for message in reversed(state.get("messages", [])) if isinstance(message, AIMessage)),
            None,
        )
        if latest is not None and not latest.tool_calls:
            state["output"] = tool_final_output(
                state,
                latest,
                stream=stream,
                max_rounds_reached=tool_limits_reached(
                    state,
                    max_tool_calls=MAX_TOOL_CALLS_PER_RUN,
                    max_tool_rounds=MAX_TOOL_ROUNDS_PER_RUN,
                ),
            )
            return state
        if stream:
            final = stream_chat_response_to_writer(
                self.provider,
                agent=agent,
                messages=state.get("messages", []),
                context=state["context"],
                writer=writer,
                stream_content=True,
                thinking_enabled=_thinking_request_value(state["context"]),
                cancel_event=self.cancel_event,
                raise_if_cancelled=self.raise_if_cancelled,
            )
        else:
            final = self._invoke_chat_model(
                agent,
                state.get("messages", []),
                state["context"],
            )
        state["output"] = tool_final_output(state, final, stream=stream, max_rounds_reached=True)
        return state

    def _invoke_chat_model(
        self,
        agent,
        messages: list[BaseMessage],
        context: dict,
        *,
        tools: list[Any] | None = None,
    ) -> AIMessage:
        response = self.provider.invoke(
            messages,
            model=agent.model,
            temperature=agent.temperature,
            runtime_config=agent.runtime_config,
            tools=tools,
            thinking_enabled=_thinking_request_value(context),
            cancel_event=self.cancel_event,
        )
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=message_content_text(response), additional_kwargs=getattr(response, "additional_kwargs", {}))


def _thinking_request_value(context: dict) -> bool | None:
    status = context.get("thinking_status") or {}
    if status.get("type") not in {"native", "prompt"}:
        return None
    return bool(status.get("enabled"))


def _refresh_system_message(messages: list[BaseMessage], agent, context: dict) -> None:
    if not messages or not isinstance(messages[0], SystemMessage):
        return
    messages[0] = build_llm_messages(agent, context)[0]
