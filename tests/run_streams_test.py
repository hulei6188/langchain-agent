from core.services.run_events import sse_event
from core.services.run_streams import execute_workflow_stream, safe_stream_error, sanitize_public_error, stream_workflow_sse


class FakeDb:
    def __init__(self):
        self.closed = False

    def get(self, model, row_id):
        return None

    def close(self):
        self.closed = True


def test_execute_workflow_stream_yields_error_for_missing_agent():
    events = list(execute_workflow_stream(FakeDb(), {"agent_id": 1, "session_id": 2}))

    assert events == [sse_event("error", {"message": "Agent not found"})]


def test_stream_workflow_sse_closes_session(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr("core.services.run_streams.SessionLocal", lambda: db)

    events = list(stream_workflow_sse({"agent_id": 1, "session_id": 2}))

    assert events == [sse_event("error", {"message": "Agent not found"})]
    assert db.closed is True


def test_safe_stream_error_redacts_public_secret_errors():
    payload = safe_stream_error(RuntimeError("Chat model API key is not configured: sk-test-secret"))

    assert payload["error_code"] == "secret_config_error"
    assert "sk-test-secret" not in payload["message"]
    assert "[secret]" in payload["message"]


def test_sanitize_public_error_normalizes_multiline_text():
    assert sanitize_public_error("line1\nsecret=abc123\rline2") == "line1 [secret] line2"
