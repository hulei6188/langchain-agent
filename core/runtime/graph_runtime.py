from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from core.runtime.langgraph_persistence import get_graph_memory_store, get_workflow_checkpointer
from core.runtime.spec import workflow_graph_spec
from core.runtime.status import search_status_event


class WorkflowGraphState(MessagesState, total=False):
    context: dict[str, Any]
    steps: list[dict[str, Any]]


def initial_workflow_state(*, user_message: str, context: dict[str, Any]) -> WorkflowGraphState:
    return {
        "messages": [HumanMessage(content=user_message)],
        "context": context,
        "steps": [],
    }


def build_langgraph_workflow(
    *,
    runtime,
    stream: bool,
    execute_node: Callable[[Any, dict, dict], dict],
    stream_llm_node: Callable[[Any, dict, dict], Any],
    run_tool_node: Callable[..., dict],
    raise_if_cancelled: Callable[[], None],
):
    spec = workflow_graph_spec(getattr(runtime, "workflow", None))
    workflow = spec["nodes"]
    graph_builder = StateGraph(WorkflowGraphState)
    used_names: set[str] = set()
    graph_names: dict[str, str] = {}

    for index, node in enumerate(workflow):
        graph_node_name = langgraph_node_name(node, index, used_names)
        node_id = str(node.get("id") or f"node_{index}")
        graph_names[node_id] = graph_node_name
        graph_builder.add_node(
            graph_node_name,
            langgraph_node(
                runtime=runtime,
                node=node,
                stream=stream,
                execute_node=execute_node,
                stream_llm_node=stream_llm_node,
                run_tool_node=run_tool_node,
                raise_if_cancelled=raise_if_cancelled,
            ),
        )

    edge_sources = {edge["source"] for edge in spec["edges"]}
    edge_targets = {edge["target"] for edge in spec["edges"]}
    start_node_ids = [node_id for node_id in graph_names if node_id not in edge_targets] or [next(iter(graph_names))]
    for node_id in start_node_ids:
        graph_builder.add_edge(START, graph_names[node_id])
    for edge in spec["edges"]:
        source = graph_names.get(edge["source"])
        target = graph_names.get(edge["target"])
        if source and target:
            graph_builder.add_edge(source, target)
    terminal_node_ids = [node_id for node_id in graph_names if node_id not in edge_sources] or [next(reversed(graph_names))]
    for node_id in terminal_node_ids:
        graph_builder.add_edge(graph_names[node_id], END)
    return graph_builder.compile(checkpointer=get_workflow_checkpointer(), store=get_graph_memory_store())


def workflow_thread_config(context: dict) -> dict:
    thread_id = context.get("session_id") or context.get("run_id") or "workflow"
    return {"configurable": {"thread_id": f"session:{thread_id}", "checkpoint_ns": "workflow"}}


def langgraph_node(
    *,
    runtime,
    node: dict,
    stream: bool,
    execute_node: Callable[[Any, dict, dict], dict],
    stream_llm_node: Callable[[Any, dict, dict], Any],
    run_tool_node: Callable[..., dict],
    raise_if_cancelled: Callable[[], None],
) -> Callable[[WorkflowGraphState], dict[str, Any]]:
    def execute(state: WorkflowGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        context = state["context"]
        steps = list(state.get("steps") or [])

        if stream and node["type"] == "LLM":
            writer = get_stream_writer()
            output = consume_streaming_node(stream_llm_node(runtime, node, context), writer)
        elif stream and node["type"] == "Tool":
            writer = get_stream_writer()
            output = run_tool_node(runtime, node, context, stream=True, writer=writer)
        else:
            writer = None
            output = execute_node(runtime, node, context)

        raise_if_cancelled()
        output = dict(output or {})
        if not steps:
            output.setdefault("events", []).extend(initial_step_events(context))

        events = list(output.pop("events", []))
        context.update(output)
        step_payload = {
            "id": len(steps) + 1,
            "node_id": str(node["id"]),
            "node_type": str(node["type"]),
            "status": "succeeded",
            "output": output,
            "events": events,
        }
        next_steps = [*steps, step_payload]
        if writer is not None:
            writer({"event": "step", "step": step_payload})
        return {"context": context, "steps": next_steps}

    return execute


def consume_streaming_node(events, writer: Callable[[dict], None]) -> dict:
    while True:
        try:
            event = next(events)
        except StopIteration as stop:
            return dict(stop.value or {})
        if event:
            writer(event)


def initial_step_events(context: dict) -> list[dict]:
    return [
        {"event": "memory_used", "data": context.get("profile_memory_used", {})},
        {"event": "thinking_status", "data": context.get("thinking_status", {})},
        {"event": "search_status", "data": search_status_event(context.get("search_status", {}))},
        {"event": "skill_selection", "data": context.get("skill_selection", {})},
    ]


def langgraph_node_name(node: dict, index: int, used_names: set[str]) -> str:
    raw_name = str(node.get("id") or f"node_{index}").strip()
    base_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw_name).strip("_") or f"node_{index}"
    if base_name in {START, END}:
        base_name = f"workflow_{index}"
    name = base_name
    suffix = 2
    while name in used_names:
        name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name
