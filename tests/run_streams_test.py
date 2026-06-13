import queue

from core.services.run_streams import safe_stream_error, sanitize_public_error, workflow_sse_items


def test_workflow_sse_items_stops_on_sentinel():
    event_queue = queue.Queue()
    event_queue.put("event: token\ndata: {}\n\n")
    event_queue.put(None)
    event_queue.put("event: token\ndata: {\"content\":\"late\"}\n\n")

    assert list(workflow_sse_items(event_queue)) == ["event: token\ndata: {}\n\n"]


def test_safe_stream_error_redacts_public_secret_errors():
    payload = safe_stream_error(RuntimeError("Chat model API key is not configured: sk-test-secret"))

    assert payload["error_code"] == "secret_config_error"
    assert "sk-test-secret" not in payload["message"]
    assert "[secret]" in payload["message"]


def test_sanitize_public_error_normalizes_multiline_text():
    assert sanitize_public_error("line1\nsecret=abc123\rline2") == "line1 [secret] line2"
