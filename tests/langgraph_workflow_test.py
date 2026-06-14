import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

pytest.importorskip("langgraph")

from core.runtime.tool_loop import ToolLoopRunner
from core.runtime.workflow import WorkflowRunner
from core.runtime.graph_runtime import initial_workflow_state, workflow_thread_config
from core.runtime.spec import workflow_graph_spec


def _runner_with_fake_persistence(monkeypatch):
    runner = WorkflowRunner.__new__(WorkflowRunner)
    runner._cancel_event = None
    return runner


def test_langgraph_workflow_invokes_nodes_in_order(monkeypatch):
    runner = _runner_with_fake_persistence(monkeypatch)

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
        stream=False,
    )
    initial_context = _initial_context()
    final_state = graph.invoke(
        initial_workflow_state(user_message="hello", context=initial_context),
        config=workflow_thread_config(initial_context),
    )

    assert final_state["context"]["trace"] == ["start", "knowledge", "answer"]
    assert [step["node_id"] for step in final_state["steps"]] == ["start", "knowledge", "answer"]
    assert [step["id"] for step in final_state["steps"]] == [1, 2, 3]
    assert final_state["steps"][0]["events"][0]["event"] == "node_event"
    assert final_state["steps"][0]["events"][1]["event"] == "memory_used"
    assert final_state["steps"][1]["events"] == [{"event": "node_event", "data": {"node": "knowledge"}}]


def test_langgraph_workflow_streams_custom_events(monkeypatch):
    runner = _runner_with_fake_persistence(monkeypatch)

    def stream_llm_node(runtime, node, context):
        yield {"event": "token", "content": "hello"}
        return {"draft": "hello", "draft_streamed": True}

    monkeypatch.setattr(runner, "_stream_llm_node", stream_llm_node)
    runtime = SimpleNamespace(workflow=[{"id": "llm", "type": "LLM", "config": {}}])
    graph = runner._build_langgraph_workflow(
        runtime=runtime,
        run=SimpleNamespace(id=1),
        stream=True,
    )

    custom_events = []
    final_state = None
    initial_context = _initial_context()
    for part in graph.stream(
        initial_workflow_state(user_message="hello", context=initial_context),
        config=workflow_thread_config(initial_context),
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


def test_workflow_runner_astart_stream_run_uses_async_persistence(monkeypatch):
    runner = _runner_with_fake_persistence(monkeypatch)
    runner.provider = SimpleNamespace(name="provider")
    runtime = SimpleNamespace(workflow=[{"id": "llm", "type": "LLM", "config": {}}])
    run = SimpleNamespace(id=123)
    context = {"session_id": 2, "input": "hello"}
    captured = {}
    registered = []
    unregistered = []

    async def async_checkpointer():
        return "async-checkpointer"

    async def async_store():
        return "async-store"

    def fake_build(**kwargs):
        captured.update(kwargs)
        return "graph"

    monkeypatch.setattr(runner, "_start_run", lambda **kwargs: (runtime, run, context))
    monkeypatch.setattr(runner, "_build_langgraph_workflow", fake_build)
    monkeypatch.setattr("core.runtime.workflow.get_async_workflow_checkpointer", async_checkpointer)
    monkeypatch.setattr("core.runtime.workflow.get_async_graph_memory_store", async_store)
    monkeypatch.setattr(
        "core.runtime.workflow.register_run",
        lambda run_id, provider: registered.append((run_id, provider)) or SimpleNamespace(is_set=lambda: False),
    )
    monkeypatch.setattr("core.runtime.workflow.unregister_run", lambda run_id: unregistered.append(run_id))

    stream_run = asyncio.run(
        runner.astart_stream_run(
            agent=SimpleNamespace(),
            chat_session=SimpleNamespace(),
            user_message="hello",
        )
    )

    assert stream_run.graph == "graph"
    assert captured["stream"] is True
    assert captured["checkpointer"] == "async-checkpointer"
    assert captured["store"] == "async-store"
    assert registered == [(123, runner.provider)]
    assert unregistered == []


def test_workflow_graph_spec_preserves_graph_fields():
    graph = workflow_graph_spec(
        {
            "nodes": [
                {"id": "start", "type": "Start"},
                {"id": "answer", "type": "Answer"},
            ],
            "edges": [{"source": "start", "target": "answer", "type": "linear"}],
            "conditional_edges": [{"source": "start", "path_map": {"ok": "answer", "bad": "missing"}}],
            "entrypoint": "start",
            "checkpointer_config": {"namespace": "agent"},
        }
    )

    assert graph["entrypoint"] == "start"
    assert graph["edges"] == [{"source": "start", "target": "answer", "type": "linear"}]
    assert graph["conditional_edges"] == [{"source": "start", "path_map": {"ok": "answer"}}]
    assert graph["checkpointer_config"] == {"namespace": "agent"}


def test_langgraph_workflow_uses_graph_spec_entrypoint(monkeypatch):
    runner = _runner_with_fake_persistence(monkeypatch)

    def execute_node(runtime, node, context):
        return {"trace": [*context.get("trace", []), node["id"]]}

    monkeypatch.setattr(runner, "_execute_node", execute_node)
    runtime = SimpleNamespace(
        workflow={
            "nodes": [
                {"id": "start", "type": "Start", "config": {}},
                {"id": "knowledge", "type": "Knowledge", "config": {}},
                {"id": "answer", "type": "Answer", "config": {}},
            ],
            "edges": [
                {"source": "start", "target": "knowledge"},
                {"source": "knowledge", "target": "answer"},
            ],
            "entrypoint": "knowledge",
        }
    )
    graph = runner._build_langgraph_workflow(
        runtime=runtime,
        run=SimpleNamespace(id=1),
        stream=False,
    )
    initial_context = _initial_context()

    final_state = graph.invoke(
        initial_workflow_state(user_message="hello", context=initial_context),
        config=workflow_thread_config(initial_context),
    )

    assert final_state["context"]["trace"] == ["knowledge", "answer"]


def test_langgraph_workflow_executes_conditional_edges(monkeypatch):
    runner = _runner_with_fake_persistence(monkeypatch)

    def execute_node(runtime, node, context):
        trace = [*context.get("trace", []), node["id"]]
        if node["id"] == "start":
            return {"trace": trace, "route": "skip"}
        return {"trace": trace}

    monkeypatch.setattr(runner, "_execute_node", execute_node)
    runtime = SimpleNamespace(
        workflow={
            "nodes": [
                {"id": "start", "type": "Start", "config": {}},
                {"id": "knowledge", "type": "Knowledge", "config": {}},
                {"id": "answer", "type": "Answer", "config": {}},
            ],
            "edges": [
                {"source": "start", "target": "knowledge"},
                {"source": "knowledge", "target": "answer"},
            ],
            "conditional_edges": [
                {
                    "source": "start",
                    "condition_key": "route",
                    "path_map": {"continue": "knowledge", "skip": "answer"},
                }
            ],
            "entrypoint": "start",
        }
    )
    graph = runner._build_langgraph_workflow(
        runtime=runtime,
        run=SimpleNamespace(id=1),
        stream=False,
    )
    initial_context = _initial_context()

    final_state = graph.invoke(
        initial_workflow_state(user_message="hello", context=initial_context),
        config=workflow_thread_config(initial_context),
    )

    assert final_state["context"]["trace"] == ["start", "answer"]
    assert [step["node_id"] for step in final_state["steps"]] == ["start", "answer"]


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
    monkeypatch.setattr("core.runtime.tool_runtime.runtime_tools", lambda db, agent, context: [_tool()])

    def fake_tool(query: str = ""):
        executed.append({"query": query})
        return {"content": "tool result", "result_preview": "tool result", "latency_ms": 1}

    monkeypatch.setattr(
        "core.runtime.tool_runtime.build_langchain_tool",
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


def test_tool_subgraph_stops_after_repeated_tool_errors(monkeypatch):
    executed = []
    provider = _FakeProvider(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call(arguments={"query": "x"})]),
            AIMessage(content="", tool_calls=[_tool_call(arguments={"query": "x"})]),
            AIMessage(content="", tool_calls=[_tool_call(arguments={"query": "x"})]),
            AIMessage(content="I could not use the tool with those arguments."),
        ]
    )
    runner = WorkflowRunner(db=SimpleNamespace())
    runner.provider = provider
    monkeypatch.setattr("core.runtime.tool_runtime.runtime_tools", lambda db, agent, context: [_tool()])

    def fake_tool(query: str = ""):
        executed.append({"query": query})
        raise ValueError("MCP tool request failed: Invalid input")

    monkeypatch.setattr(
        "core.runtime.tool_runtime.build_langchain_tool",
        lambda *args, **kwargs: StructuredTool.from_function(fake_tool, name="test_tool", description="Test tool"),
    )

    output = runner._run_tool_node(
        _agent(),
        {"id": "tool", "type": "Tool", "config": {}},
        _tool_context(),
        stream=False,
    )

    assert output["draft"] == "I could not use the tool with those arguments."
    assert output["tool_stats"]["total_calls"] == 3
    assert output["tool_stats"]["max_rounds_reached"] is True
    assert executed == [{"query": "x"}, {"query": "x"}, {"query": "x"}]
    assert len(provider.chat_calls) == 4
    assert provider.chat_calls[-1]["tools"] is None


def test_tool_subgraph_disables_inherited_checkpointing():
    runner = ToolLoopRunner(
        db=SimpleNamespace(),
        provider=SimpleNamespace(),
        cancel_event=None,
        raise_if_cancelled=lambda: None,
    )

    graph = runner._build_graph(_agent(), stream=False, writer=None)

    assert graph.checkpointer is False


def _initial_context():
    return {
        "session_id": 1,
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
