import asyncio
from types import SimpleNamespace

from core.db.models import Agent, Session as ChatSession
from core.services.run_events import sse_event
from core.services.run_streams import execute_workflow_stream, safe_stream_error, sanitize_public_error, stream_workflow_sse


async def _collect(async_iterable):
    return [item async for item in async_iterable]


class FakeDb:
    def __init__(self):
        self.closed = False

    def get(self, model, row_id):
        return None

    def close(self):
        self.closed = True


def test_execute_workflow_stream_yields_error_for_missing_agent():
    events = asyncio.run(_collect(execute_workflow_stream(FakeDb(), {"agent_id": 1, "session_id": 2})))

    assert events == [sse_event("error", {"message": "Agent not found"})]


def test_stream_workflow_sse_closes_session(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr("core.services.run_streams.SessionLocal", lambda: db)

    events = asyncio.run(_collect(stream_workflow_sse({"agent_id": 1, "session_id": 2})))

    assert events == [sse_event("error", {"message": "Agent not found"})]
    assert db.closed is True


def test_execute_workflow_stream_consumes_graph_stream_parts(monkeypatch):
    agent = SimpleNamespace(id=1, user_model_config_id=None)
    chat_session = SimpleNamespace(id=2, title="Existing title")

    class FakeRunDb:
        def __init__(self):
            self.added = []
            self.commits = 0

        def get(self, model, row_id):
            if model is Agent:
                return agent
            if model is ChatSession:
                return chat_session
            return None

        def add(self, item):
            self.added.append(item)

        def commit(self):
            self.commits += 1

        def refresh(self, item):
            if getattr(item, "id", None) is None:
                item.id = 42

        def rollback(self):
            pass

    class FakeRunner:
        instances = []

        def __init__(self, db):
            self.closed = False
            self.streamed = False
            FakeRunner.instances.append(self)

        def start_stream_run(self, **kwargs):
            self.started = kwargs
            return SimpleNamespace(run=SimpleNamespace(id=99), context={"session_id": 2})

        async def astream_graph_events(self, workflow_stream):
            self.streamed = True
            yield {
                "event": "on_chain_stream",
                "name": "LangGraph",
                "data": {"chunk": ("custom", {"event": "token", "content": "hello"})},
            }
            yield {
                "event": "on_chain_stream",
                "name": "LangGraph",
                "data": {"chunk": ("values", {"context": {"answer": "hello"}, "steps": []})},
            }

        def complete_stream_run(self, **kwargs):
            self.completed = kwargs
            return "hello", [{"title": "source"}]

        def close_stream_run(self, run):
            self.closed = True

    persisted = []
    monkeypatch.setattr("core.services.run_streams.WorkflowRunner", FakeRunner)
    monkeypatch.setattr(
        "core.services.run_streams.append_run_event",
        lambda db, run_id, event, payload, sse: persisted.append((run_id, event, payload)),
    )

    events = asyncio.run(
        _collect(
            execute_workflow_stream(
                FakeRunDb(),
                {
                    "agent_id": 1,
                    "session_id": 2,
                    "user_message": "hi",
                    "mode": "draft",
                    "variables": {},
                    "rag_enabled": None,
                    "rag_options": {},
                    "thinking_enabled": None,
                    "search_enabled": None,
                    "attachments": [],
                    "user_message_id": 7,
                    "user_id": 3,
                },
            )
        )
    )

    assert FakeRunner.instances[0].streamed is True
    assert FakeRunner.instances[0].closed is True
    assert any(item.startswith("event: token") for item in events)
    assert any(item.startswith("event: sources") for item in events)
    assert any(item.startswith("event: done") for item in events)
    assert [event for _, event, _ in persisted][:2] == ["run_started", "token"]


def test_safe_stream_error_redacts_public_secret_errors():
    payload = safe_stream_error(RuntimeError("Chat model API key is not configured: sk-test-secret"))

    assert payload["error_code"] == "secret_config_error"
    assert "sk-test-secret" not in payload["message"]
    assert "[secret]" in payload["message"]


def test_sanitize_public_error_normalizes_multiline_text():
    assert sanitize_public_error("line1\nsecret=abc123\rline2") == "line1 [secret] line2"
