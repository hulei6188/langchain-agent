from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from core.db.models import Message
from core.runtime.dsml import contains_leaked_tool_markup, dsml_preview


logger = logging.getLogger(__name__)


def persist_intermediate_message(
    db: Session,
    context: dict,
    *,
    role: str,
    content: str,
    reasoning: str = "",
    tool_calls: list[dict] | None = None,
    tool_call_id: str = "",
    tool_name: str = "",
    meta: dict | None = None,
) -> None:
    session_id = context.get("session_id")
    if not session_id:
        return
    if role == "assistant" and contains_leaked_tool_markup(content):
        logger.warning(
            "Blocked leaked tool call markup before persisting intermediate assistant message; preview=%r",
            dsml_preview(content),
        )
        content = ""
    visible_reasoning = reasoning if context.get("thinking_enabled") else ""
    payload_meta = {
        "is_intermediate": True,
        "run_id": context.get("run_id"),
        "thinking_enabled": bool(context.get("thinking_enabled")),
        **(meta or {}),
    }
    if role == "assistant" and visible_reasoning and context.get("reasoning_replay_required"):
        payload_meta["requires_reasoning_replay"] = True
    message = Message(
        session_id=session_id,
        role=role,
        content=content or "",
        reasoning=visible_reasoning or "",
        sources=[],
        tool_calls=tool_calls or [],
        tool_call_id=tool_call_id or "",
        tool_name=tool_name or "",
        meta=payload_meta,
    )
    db.add(message)
    db.flush()
    db.commit()


def session_history(db: Session, session_id: int, *, max_messages: int) -> list[dict]:
    rows = (
        db.query(Message)
        .filter(Message.session_id == session_id, Message.role.in_(["user", "assistant", "tool"]))
        .order_by(Message.id.desc())
        .limit(max(1, min(int(max_messages or 12), 100)))
        .all()
    )
    history = []
    for message in reversed(rows):
        content = trim_history_content(message.content or "")
        if not content and message.role != "assistant":
            continue
        history.append(
            {
                "id": message.id,
                "role": message.role,
                "content": content,
                "reasoning": trim_history_content(message.reasoning or ""),
                "tool_calls": message.tool_calls or [],
                "tool_call_id": message.tool_call_id or "",
                "tool_name": message.tool_name or "",
                "meta": message.meta or {},
            }
        )
    return history


def trim_history_content(content: str, limit: int = 6000) -> str:
    text = content.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[历史消息过长，已截断]"
