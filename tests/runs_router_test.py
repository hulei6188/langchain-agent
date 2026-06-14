from types import SimpleNamespace

from api.routers import runs as runs_router


def test_mark_disconnected_run_cancelled_persists_partial_message(monkeypatch):
    run = SimpleNamespace(id=9, status="running", completed_at=None)
    appended = []

    class FakeDb:
        def __init__(self):
            self.added = []

        def add(self, item):
            self.added.append(item)

        def flush(self):
            for item in self.added:
                if getattr(item, "id", None) is None:
                    item.id = 77

    monkeypatch.setattr(
        runs_router,
        "run_stream_snapshot",
        lambda db, run_id: {
            "content": "partial answer",
            "reasoning": "partial reasoning",
            "sources": [{"title": "Source"}],
        },
    )
    monkeypatch.setattr(
        runs_router,
        "append_run_event",
        lambda db, run_id, event, payload, sse: appended.append((run_id, event, payload, sse)),
    )

    db = FakeDb()
    message_id = runs_router._mark_disconnected_run_cancelled(db, run=run, session_id=3)

    assert message_id == 77
    assert run.status == "cancelled"
    assistant = db.added[0]
    assert assistant.content == "partial answer"
    assert assistant.reasoning == "partial reasoning"
    assert assistant.meta == {"cancelled": True, "partial": True}
    assert appended[0][0] == 9
    assert appended[0][1] == "cancelled"
    assert appended[0][2]["message_id"] == 77
    assert appended[0][2]["content"] == "partial answer"
    assert appended[0][3].startswith("event: cancelled")
