from types import SimpleNamespace

from datetime import timedelta
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from core.integrations.mcp_client import _adapter_connection, _adapter_tool_result, _stdio_adapter_connection
from core.runtime.agent_runtime import validate_model_capabilities
from core.runtime.langgraph_persistence import postgres_conn_string
from core.runtime.memory_runtime import save_runtime_memory_state
from core.services.memory import delete_session_memory_payload, get_session_memory_payload, update_session_memory
from core.runtime.persistence import persist_intermediate_message, trim_history_content
from core.runtime.prompting import build_llm_messages, history_messages_for_llm, merge_variables, user_content
from core.runtime.skill_runtime import apply_runtime_skills, handle_load_skill_call, skill_loader_schema
from core.runtime.streaming import stream_chat_response, stream_llm_response
from core.runtime.tool_graph_helpers import (
    tool_direct_output,
    tool_final_output,
    tool_job_start_event,
    tool_limits_reached,
    tool_message_fallback_result,
)


def test_tool_graph_output_helpers_preserve_stream_flags():
    state = {
        "context": {"thinking_enabled": True},
        "web_sources": [{"title": "web"}],
        "search_status": {"enabled": True},
        "total_calls": 2,
        "tools_used": ["search"],
        "events": [{"event": "tool_call"}],
    }
    response = AIMessage(content="answer", additional_kwargs={"reasoning_content": "why"})
    direct = tool_direct_output(state, response, stream=True)
    assert direct["draft"] == "answer"
    assert direct["draft_streamed"] is True
    assert direct["draft_reasoning_streamed"] is True

    final = tool_final_output(state, response, stream=True, max_rounds_reached=False)
    assert final["tool_stats"]["total_calls"] == 2
    assert final["tool_stats"]["max_rounds_reached"] is False


def test_postgres_conn_string_accepts_sqlalchemy_psycopg_url():
    assert (
        postgres_conn_string("postgresql+psycopg://user:pass@localhost:5432/app")
        == "postgresql://user:pass@localhost:5432/app"
    )
    assert (
        postgres_conn_string("postgresql+psycopg2://user:pass@localhost:5432/app")
        == "postgresql://user:pass@localhost:5432/app"
    )


def test_mcp_adapter_connection_shapes_http_and_sse_transports():
    http_connection = _adapter_connection(
        "http://localhost:8000/mcp",
        headers={"Authorization": "Bearer token"},
        transport="streamable_http",
        timeout_seconds=7,
    )
    assert http_connection["transport"] == "streamable_http"
    assert http_connection["timeout"] == timedelta(seconds=7)

    sse_connection = _adapter_connection(
        "http://localhost:8000/sse",
        headers={},
        transport="sse",
        timeout_seconds=5,
    )
    assert sse_connection["transport"] == "sse"
    assert sse_connection["timeout"] == 5.0

    stdio_connection = _stdio_adapter_connection(
        "npx",
        ["@modelcontextprotocol/server-filesystem"],
        env={"A": "B"},
        cwd="C:/tmp",
        timeout_seconds=9,
    )
    assert stdio_connection["transport"] == "stdio"
    assert stdio_connection["command"] == "npx"
    assert stdio_connection["args"] == ["@modelcontextprotocol/server-filesystem"]
    assert stdio_connection["session_kwargs"]["read_timeout_seconds"] == timedelta(seconds=9)


def test_mcp_adapter_tool_result_preserves_structured_payload():
    artifact = SimpleNamespace(structured_content={"ok": True})
    payload = _adapter_tool_result(([{"type": "text", "text": "done"}], artifact))

    assert payload["content"] == "done"
    assert payload["content_type"] == "application/json"
    assert payload["result_json"] == {"ok": True}


def test_tool_graph_limit_and_event_helpers():
    assert tool_limits_reached({"total_calls": 3}, max_tool_calls=3, max_tool_rounds=5)
    assert tool_limits_reached({"round_index": 5}, max_tool_calls=10, max_tool_rounds=5)
    assert not tool_limits_reached(
        {"total_calls": 1, "round_index": 1, "max_tool_wall_time": 9999},
        max_tool_calls=10,
        max_tool_rounds=5,
    )

    tool = SimpleNamespace(id=7, name="search", type="builtin_search")
    event = tool_job_start_event({"matching": tool, "tool_name": "search", "tc": {"id": "call_1"}, "tool_args": {"query": "x"}})
    assert event["tool_id"] == 7
    assert event["tool_call_id"] == "call_1"
    assert '"query": "x"' in event["input_preview"]

    fallback = tool_message_fallback_result({"tool_name": "missing", "matching": None, "internal": False}, None)
    assert fallback["content"] == "Tool 'missing' not found"

    error_message = ToolMessage(content="bad", tool_call_id="call_2", status="error")
    error_fallback = tool_message_fallback_result({"tool_name": "missing", "matching": None, "internal": False}, error_message)
    assert error_fallback["error"] == "tool_not_found"


class FakeStreamProvider:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = []

    def stream(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        yield from self.chunks


def drain_stream(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as exc:
            return events, exc.value


def test_stream_chat_response_replays_provisional_tokens_when_no_tool_call():
    provider = FakeStreamProvider([AIMessageChunk(content="hello"), AIMessageChunk(content=" world")])
    agent = SimpleNamespace(model="m", temperature=0.1, runtime_config={})
    events, response = drain_stream(
        stream_chat_response(
            provider,
            agent=agent,
            messages=[],
            context={"thinking_enabled": False},
            tools=[object()],
            stream_content=True,
            provisional_stream=True,
        )
    )
    assert [event["content"] for event in events if event["event"] == "token"] == ["hello", " world"]
    assert response.content == "hello world"
    assert provider.calls[0]["tools"]


def test_stream_llm_response_emits_reasoning_and_tokens():
    provider = FakeStreamProvider([AIMessageChunk(content="answer", additional_kwargs={"reasoning_content": "why"})])
    agent = SimpleNamespace(model="m", temperature=0.1, runtime_config={})
    events, response = drain_stream(
        stream_llm_response(
            provider,
            agent=agent,
            messages=[],
            context={"thinking_enabled": True},
            thinking_enabled=True,
        )
    )
    assert events == [
        {"event": "reasoning_token", "content": "why"},
        {"event": "token", "content": "answer"},
    ]
    assert response.content == "answer"
    assert response.additional_kwargs["reasoning_content"] == "why"


def test_prompting_helpers_filter_leaked_dsml_history_and_merge_variables():
    context = {
        "input": "current",
        "history_messages": [
            {"id": 1, "role": "user", "content": "before"},
            {"id": 2, "role": "assistant", "content": '<||DSML||tool_calls>\n<||DSML||invoke name="x">', "tool_calls": []},
            {"id": 3, "role": "user", "content": "after"},
        ],
        "uploads": [],
    }
    messages = history_messages_for_llm(context)
    assert [(type(message), message.content) for message in messages] == [
        (HumanMessage, "before"),
        (HumanMessage, "after"),
    ]
    assert merge_variables([{"key": "tone", "default_value": "formal"}], {"topic": "rag"}) == {
        "tone": "formal",
        "topic": "rag",
    }


def test_build_llm_messages_includes_image_upload_content():
    image_upload = SimpleNamespace(kind="image", filename="chart.png", data_url="data:image/png;base64,abc")
    context = {
        "input": "describe",
        "sources": [],
        "web_sources": [],
        "tool_outputs": [],
        "variables": {},
        "uploads": [image_upload],
        "skill_manifest": [],
        "loaded_skills": [],
        "memory_summary": "",
        "profile_memory": "",
        "history_messages": [],
    }
    agent = SimpleNamespace(system_prompt="System")
    messages = build_llm_messages(agent, context)
    assert messages[0].content.startswith("System")
    assert user_content("describe", [image_upload]) == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    assert messages[-1].content[1]["image_url"]["url"] == "data:image/png;base64,abc"


class FakeMessageDb:
    def __init__(self):
        self.added = []
        self.flushed = False
        self.committed = False

    def add(self, message):
        self.added.append(message)

    def flush(self):
        self.flushed = True

    def commit(self):
        self.committed = True


def test_persist_intermediate_message_keeps_visible_reasoning_and_meta():
    db = FakeMessageDb()
    persist_intermediate_message(
        db,
        {
            "session_id": 42,
            "run_id": 99,
            "thinking_enabled": True,
            "reasoning_replay_required": True,
        },
        role="assistant",
        content="answer",
        reasoning="why",
        tool_calls=[{"id": "call_1", "name": "search", "args": {"query": "x"}}],
        meta={"node": "model"},
    )

    assert db.flushed is True
    assert db.committed is True
    message = db.added[0]
    assert message.session_id == 42
    assert message.content == "answer"
    assert message.reasoning == "why"
    assert message.tool_calls[0]["name"] == "search"
    assert message.meta == {
        "is_intermediate": True,
        "run_id": 99,
        "thinking_enabled": True,
        "node": "model",
        "requires_reasoning_replay": True,
    }


def test_persist_intermediate_message_blocks_leaked_dsml_content():
    db = FakeMessageDb()
    persist_intermediate_message(
        db,
        {"session_id": 1, "thinking_enabled": False},
        role="assistant",
        content='<||DSML||tool_calls>\n<||DSML||invoke name="search">',
    )

    assert db.added[0].content == ""


def test_trim_history_content_strips_and_truncates():
    text = "  " + ("x" * 20) + "  "
    assert trim_history_content(text, limit=5) == "xxxxx\n[历史消息过长，已截断]"


def test_apply_runtime_skills_loads_always_manual_and_matching_auto_skills():
    runtime = SimpleNamespace(
        base_system_prompt="Base",
        system_prompt="Base",
        tool_ids=[1],
        knowledge_base_ids=[10],
        skill_bindings=[
            {
                "id": 1,
                "name": "常驻技能",
                "description": "",
                "category": "general",
                "tags": [],
                "activation_mode": "always",
                "priority": 0,
                "system_prompt": "Always prompt",
                "tool_ids": [2],
                "knowledge_base_ids": [20],
            },
            {
                "id": 2,
                "name": "手动技能",
                "description": "",
                "category": "general",
                "tags": [],
                "activation_mode": "manual",
                "priority": 0,
                "system_prompt": "Manual prompt",
                "tool_ids": [3],
                "knowledge_base_ids": [30],
            },
            {
                "id": 3,
                "name": "财报分析",
                "description": "财报 股票",
                "category": "finance",
                "tags": ["财报", "股票"],
                "activation_mode": "auto",
                "priority": 0,
                "system_prompt": "Auto prompt",
                "tool_ids": [4],
                "knowledge_base_ids": [40],
            },
        ],
    )
    context = {"input": "请用 @手动技能 分析这份财报和股票风险", "history_messages": [], "memory_summary": ""}

    apply_runtime_skills(runtime, context, SimpleNamespace(title="季度复盘"))

    assert [item["name"] for item in context["loaded_skills"]] == ["常驻技能", "手动技能", "财报分析"]
    assert runtime.tool_ids == [1, 2, 3, 4]
    assert runtime.knowledge_base_ids == [10, 20, 30, 40]
    assert "Always prompt" in runtime.system_prompt
    assert "Manual prompt" in runtime.system_prompt
    assert "Auto prompt" in runtime.system_prompt
    assert context["skill_selection"]["threshold"] == 0.25


def test_handle_load_skill_call_updates_agent_without_rag_when_disabled():
    agent = SimpleNamespace(
        workspace_id=1,
        system_prompt="Base",
        tool_ids=[],
        knowledge_base_ids=[],
        runtime_config={},
        skill_bindings=[
            {
                "id": 7,
                "name": "研究技能",
                "description": "research",
                "category": "general",
                "tags": [],
                "activation_mode": "manual",
                "priority": 0,
                "system_prompt": "Research prompt",
                "tool_ids": [11],
                "knowledge_base_ids": [22],
            }
        ],
    )
    context = {"input": "load", "loaded_skills": [], "skill_selection": {}, "rag_enabled": False, "sources": []}

    result = handle_load_skill_call(SimpleNamespace(), agent, context, {"skill_id": 7, "reason": "need it"})

    assert result["status"] == "success"
    assert "Research prompt" in agent.system_prompt
    assert agent.tool_ids == [11]
    assert agent.knowledge_base_ids == [22]
    assert context["loaded_skills"][0]["id"] == 7
    assert skill_loader_schema(agent, context) is None


def test_validate_model_capabilities_rejects_unsupported_upload_kinds():
    image_upload = SimpleNamespace(kind="image")
    document_upload = SimpleNamespace(kind="document")

    validate_model_capabilities(SimpleNamespace(supports_image=True, supports_document=True), [image_upload, document_upload])

    try:
        validate_model_capabilities(SimpleNamespace(supports_image=False, supports_document=True), [image_upload])
    except ValueError as exc:
        assert "image input" in str(exc)
    else:
        raise AssertionError("Expected image capability validation to fail")

    try:
        validate_model_capabilities(SimpleNamespace(supports_image=True, supports_document=False), [document_upload])
    except ValueError as exc:
        assert "document input" in str(exc)
    else:
        raise AssertionError("Expected document capability validation to fail")


def test_save_runtime_memory_state_only_updates_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "core.runtime.memory_runtime.update_session_memory",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    save_runtime_memory_state(
        SimpleNamespace(),
        session_id=1,
        user_message="u",
        answer="a",
        max_messages=12,
        enabled=False,
    )
    assert calls == []

    save_runtime_memory_state(
        SimpleNamespace(),
        session_id=1,
        user_message="u",
        answer="a",
        max_messages=12,
        enabled=True,
    )
    assert calls == [{"session_id": 1, "user_message": "u", "answer": "a", "max_messages": 12}]


def test_session_memory_uses_langgraph_store():
    session_id = 987654

    first = update_session_memory(
        session_id=session_id,
        user_message="u1",
        answer="a1",
        max_messages=2,
    )
    second = update_session_memory(
        session_id=session_id,
        user_message="u2",
        answer="a2",
        max_messages=2,
    )
    loaded = get_session_memory_payload(session_id=session_id)

    assert first["message_count"] == 2
    assert second["message_count"] == 4
    assert loaded["message_count"] == 4
    assert '"u1"' not in loaded["summary"]
    assert '"u2"' in loaded["summary"]

    delete_session_memory_payload(session_id=session_id)
    cleared = get_session_memory_payload(session_id=session_id)
    assert cleared["summary"] == ""
    assert cleared["message_count"] == 0
