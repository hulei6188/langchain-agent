from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.deps import get_current_membership
from core.db.models import Message, Run, Session as ChatSession, WorkspaceMember
from core.db.session import SessionLocal, get_db
from core.runtime.cancel import cancel_run
from core.security.permissions import can_manage
from core.services.run_events import append_run_event, list_run_events_since, run_stream_snapshot, sse_event, sse_event_name


router = APIRouter(prefix="/api/runs", tags=["runs"])
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def _require_session_access(session: ChatSession | None, membership: WorkspaceMember) -> None:
    if session is None:
        return
    if session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if can_manage(membership.role) or session.user_id == membership.user_id:
        return
    raise HTTPException(status_code=404, detail="Run not found")


def _require_run_access(db: Session, run_id: int, membership: WorkspaceMember) -> tuple[Run, int]:
    run = db.get(Run, run_id)
    if not run or run.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Run not found")
    session = db.get(ChatSession, run.session_id)
    _require_session_access(session, membership)
    return run, run.session_id


@router.get("/{run_id}")
def get_run(
    run_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    run, _session_id = _require_run_access(db, run_id, membership)
    return {"run": {"id": run.id, "status": run.status, "agent_id": run.agent_id, "session_id": run.session_id}}


@router.post("/{run_id}/cancel")
def cancel_run_endpoint(
    run_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    run, session_id = _require_run_access(db, run_id, membership)
    if cancel_run(run_id):
        return {"cancelled": True, "run_id": run_id}
    if run.status == "running":
        message_id = _mark_disconnected_run_cancelled(db, run=run, session_id=session_id)
        return {
            "cancelled": True,
            "run_id": run_id,
            "message_id": message_id,
            "message": "Run was no longer active and has been marked cancelled",
        }
    return {"cancelled": False, "run_id": run_id, "message": "Run not active or already completed"}


def _mark_disconnected_run_cancelled(db: Session, *, run: Run, session_id: int) -> int | None:
    snapshot = run_stream_snapshot(db, run_id=run.id)
    content = str(snapshot.get("content") or "")
    reasoning = str(snapshot.get("reasoning") or "")
    raw_sources = snapshot.get("sources")
    sources = raw_sources if isinstance(raw_sources, list) else []
    assistant_id = None
    if content or reasoning:
        assistant = Message(
            session_id=session_id,
            role="assistant",
            content=content,
            reasoning=reasoning,
            sources=sources,
            meta={"cancelled": True, "partial": True},
        )
        db.add(assistant)
        db.flush()
        assistant_id = assistant.id
    run.status = "cancelled"
    run.completed_at = datetime.now(timezone.utc)
    payload = {
        "session_id": session_id,
        "run_id": run.id,
        "message_id": assistant_id,
        "content": content,
        "reasoning_duration_ms": None,
    }
    append_run_event(db, run_id=run.id, event="cancelled", payload=payload, sse=sse_event("cancelled", payload))
    return assistant_id


@router.get("/{run_id}/events")
def stream_run_events(
    run_id: int,
    since: int = Query(0),
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    """Replay persisted run events and follow an active run until it finishes."""
    _run, session_id = _require_run_access(db, run_id, membership)

    def _event_stream() -> Iterable[str]:
        emitted = max(0, since)
        terminal_seen = False
        if emitted == 0:
            db_sess = SessionLocal()
            try:
                current = db_sess.get(Run, run_id)
                if current is not None and current.status == "running":
                    snapshot = run_stream_snapshot(db_sess, run_id=run_id)
                    emitted = int(snapshot.get("event_index") or 0)
                    yield sse_event("run_snapshot", snapshot)
            finally:
                db_sess.close()
        while True:
            db_sess = SessionLocal()
            try:
                events = list_run_events_since(db_sess, run_id=run_id, since=emitted)
            finally:
                db_sess.close()
            for event_row in events:
                sse_str = event_row.sse
                if sse_event_name(sse_str) in {"done", "cancelled", "error"}:
                    terminal_seen = True
                yield sse_str
                emitted = int(event_row.sequence) + 1
            db_sess = SessionLocal()
            try:
                current = db_sess.get(Run, run_id)
                if current is None or current.status != "running":
                    if not terminal_seen:
                        final_status = current.status if current else "unknown"
                        terminal_payload = {"run_id": run_id, "session_id": session_id, "status": final_status}
                        if final_status == "succeeded":
                            yield sse_event("done", terminal_payload)
                        elif final_status == "cancelled":
                            yield sse_event("cancelled", terminal_payload)
                        elif final_status == "failed":
                            yield sse_event("error", {**terminal_payload, "message": "Run failed"})
                        else:
                            yield sse_event("done", terminal_payload)
                    break
            finally:
                db_sess.close()
            time.sleep(0.3)

    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)
