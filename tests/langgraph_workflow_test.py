from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

pytest.importorskip("langgraph")

from core.runtime.workflow import WorkflowRunner


def _runner_with_fake_persistence(monkeypatch):
    runner = WorkflowRunner.__new__(WorkflowRunner)
    runner._cancel_event = None
    persisted = []

    def persist_step(run, node, user_message, output):
        persisted.append({"node_id": node["id"], "output": dict(output)})
        return SimpleNamespace(
            id=len(persisted),
            node_id=node["id"],
            node_type=node["type"],
            status="succeeded",
        )

    monkeypatch.setattr(runner, "_persist_step", persist_step)
    return runner, persisted


def test_langgraph_workflow_invokes_nodes_in_order(monkeypatch):
    runner, persisted = _runner_with_fake_persistence(monkeypatch)

    def execute_node(runtime, node, context):
        return {
            "trace": [*context.get("trace", []), node["id"]],
            "events": [{"event": "node_event", "data": {"node": node["id"]}}],
        }

    monkeypatch.setattr(runner, "_execute_node", execute_node)
    runtime = SimpleNamespace(
        workflow=[
            {"id": "start", "type": "Start", "config": {}},
            {"id": "knowledge", "type": "Knowledge", "config": {}},
            {"id": "answer", "type": "Answer", "config": {}},
        ]
    )

    graph = runner._build_langgraph_workflow(
        runtime=runtime,
        run=SimpleNamespace(id=1),
        user_message="hello",
        stream=False,
    )
    final_state = graph.invoke({"context": _initial_context(), "steps": []})

    assert final_state["context"]["trace"] == ["start", "knowledge", "answer"]
    assert [step["node_id"] for step in final_state["steps"]] == ["start", "knowledge", "answer"]
    assert [item["node_id"] for item in persisted] == ["start", "knowledge", "answer"]
    assert final_state["steps"][0]["events"][0]["event"] == "node_event"
    assert final_state["steps"][0]["events"][1]["event"] == "memory_used"
    assert final_state["steps"][1]["events"] == [{"event": "node_event", "data": {"node": "knowledge"}}]


def test_langgraph_workflow_streams_custom_events(monkeypatch):
    runner, _persisted = _runner_with_fake_persistence(monkeypatch)

    def stream_llm_node(runtime, node, context):
        yield {"event": "token", "content": "hello"}
        return {"draft": "hello", "draft_streamed": True}

    monkeypatch.setattr(runner, "_stream_llm_node", stream_llm_node)
    runtime = SimpleNamespace(workflow=[{"id": "llm", "type": "LLM", "config": {}}])
    graph = runner._build_langgraph_workflow(
        runtime=runtime,
        run=SimpleNamespace(id=1),
        user_message="hello",
        stream=True,
    )

    custom_events = []
    final_state = None
    for part in graph.stream(
        {"context": _initial_context(), "steps": []},
        stream_mode=["custom", "values"],
        version="v2",
    ):
        if part["type"] == "custom":
            custom_events.append(part["data"])
        elif part["type"] == "values":
            final_state = part["data"]

    assert custom_events[0] == {"event": "token", "content": "hello"}
    assert custom_events[-1]["event"] == "step"
    assert final_state["context"]["draft"] == "hello"
    assert final_state["steps"][0]["output"]["draft_streamed"] is True


def test_tool_subgraph_loops_back_to_model_after_tool_call(monkeypatch):
    executed = []
    provider = _FakeProvider(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call(arguments={"query": "x"})]),
            AIMessage(content="final answer"),
        ]
    )
    runner = WorkflowRunner(db=SimpleNamespace())
    runner.provider = provider
    runner._runtime_tools = lambda agent, node, context: [_tool()]

    def fake_tool(query: str = ""):
        executed.append({"query": query})
        return {"content": "tool result", "result_preview": "tool result", "latency_ms": 1}

    monkeypatch.setattr(
        "core.runtime.workflow.build_langchain_tool",
        lambda *args, **kwargs: StructuredTool.from_function(fake_tool, name="test_tool", description="Test tool"),
    )

    output = runner._run_tool_node(
        _agent(),
        {"id": "tool", "type": "Tool", "config": {}},
        _tool_context(),
        stream=False,
    )

    assert output["draft"] == "final answer"
    assert output["tool_stats"]["total_calls"] == 1
    assert executed == [{"query": "x"}]
    assert len(provider.chat_calls) == 2
    assert any(
        isinstance(message, ToolMessage) and message.content == "tool result"
        for message in provider.chat_calls[1]["messages"]
    )


def _initial_context():
    return {
        "profile_memory_used": {},
        "thinking_status": {},
        "search_status": {},
        "skill_selection": {},
    }


class _FakeProvider:
    last_chat_mock = False

    def __init__(self, *, responses=None):
        self.responses = list(responses or [])
        self.chat_calls = []

    def invoke(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("No fake chat response left")
        return self.responses.pop(0)


def _agent():
    return SimpleNamespace(
        id=1,
        workspace_id=1,
        system_prompt="You are a test agent.",
        model="test-model",
        temperature=0.1,
        runtime_config={},
        settings={"tool_policy": {}},
    )


def _tool_context():
    return {
        "input": "use a tool",
        "sources": [],
        "tool_outputs": [],
        "history_messages": [],
        "variables": {},
        "memory_summary": "",
        "profile_memory": "",
        "web_sources": [],
        "search_status": {},
        "uploads": [],
        "thinking_status": {},
        "thinking_enabled": False,
        "reasoning_replay_required": False,
    }


def _tool():
    return SimpleNamespace(
        id=1,
        name="test_tool",
        label="test_tool",
        description="Test tool",
        type="http",
        enabled=True,
        query_schema={},
        body_schema={},
        method="GET",
    )


def _tool_call(arguments=None):
    return {
        "name": "test_tool",
        "args": arguments or {},
        "id": "call_test_0",
    }
