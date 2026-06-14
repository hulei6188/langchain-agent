from __future__ import annotations

import json

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.db.models import RunEvent


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_event_name(sse_str: str) -> str:
    for line in str(sse_str or "").splitlines():
        if line.startswith("event:"):
            return line.partition(":")[2].strip()
    return ""


def append_run_event(db: Session, *, run_id: int, event: str, payload: dict, sse: str) -> RunEvent:
    latest = db.query(func.max(RunEvent.sequence)).filter(RunEvent.run_id == run_id).scalar()
    sequence = int(latest) + 1 if latest is not None else 0
    row = RunEvent(
        run_id=run_id,
        sequence=sequence,
        event=event,
        payload=payload or {},
        sse=sse,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_run_events_since(db: Session, *, run_id: int, since: int) -> list[RunEvent]:
    return (
        db.query(RunEvent)
        .filter(RunEvent.run_id == run_id, RunEvent.sequence >= max(0, int(since)))
        .order_by(RunEvent.sequence.asc())
        .all()
    )


def run_event_count(db: Session, *, run_id: int) -> int:
    return int(db.query(func.count(RunEvent.id)).filter(RunEvent.run_id == run_id).scalar() or 0)


def append_snapshot_timeline_event(timeline_events: list[dict], event_name: str, data: dict) -> None:
    if event_name == "reasoning_token":
        content = str(data.get("content") or "")
        if not content:
            return
        if timeline_events and timeline_events[-1].get("event") == "reasoning_token":
            previous_data = timeline_events[-1].setdefault("data", {})
            previous_data["content"] = f"{previous_data.get('content') or ''}{content}"
            return
        timeline_events.append({"event": "reasoning_token", "data": {"content": content}})
        return
    if event_name in {"tool_call", "tool_call_start", "tool_call_result", "search_status"}:
        timeline_events.append({"event": event_name, "data": data})


def run_stream_snapshot(db: Session, *, run_id: int) -> dict:
    rows = (
        db.query(RunEvent)
        .filter(RunEvent.run_id == run_id)
        .order_by(RunEvent.sequence.asc())
        .all()
    )
    content = ""
    provisional_content = ""
    reasoning = ""
    sources: list[dict] = []
    timeline_events: list[dict] = []
    for row in rows:
        event_name = row.event
        data = row.payload or {}
        if event_name == "token":
            content += str(data.get("content") or "")
        elif event_name == "provisional_token":
            chunk = str(data.get("content") or "")
            content += chunk
            provisional_content += chunk
        elif event_name == "provisional_clear":
            if provisional_content and content.endswith(provisional_content):
                content = content[: -len(provisional_content)]
            provisional_content = ""
        elif event_name == "provisional_commit":
            provisional_content = ""
        elif event_name == "reasoning_token":
            chunk = str(data.get("content") or "")
            reasoning += chunk
            append_snapshot_timeline_event(timeline_events, event_name, {"content": chunk})
        elif event_name == "sources":
            items = data.get("items")
            sources = items if isinstance(items, list) else []
        else:
            append_snapshot_timeline_event(timeline_events, event_name, data)

    return {
        "event_index": len(rows),
        "content": content,
        "reasoning": reasoning,
        "provisional_content": provisional_content,
        "sources": sources,
        "timeline_events": timeline_events[-200:],
    }
