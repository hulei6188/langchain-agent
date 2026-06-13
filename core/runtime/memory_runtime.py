from __future__ import annotations

from sqlalchemy.orm import Session

from core.services.memory import load_graph_memory_context, update_session_memory


def load_runtime_memory_state(
    db: Session,
    *,
    workspace_id: int,
    user_id: int,
    agent_id: int,
    session_id: int,
    memory_config: dict,
) -> dict:
    memory_context = load_graph_memory_context(
        db,
        workspace_id=workspace_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
    )
    return {
        "memory_summary": memory_context.session_summary,
        "profile_memory": memory_context.profile_text,
        "profile_memory_used": memory_context.profile_event,
        "memory_enabled": memory_config.get("enabled", False),
    }


def save_runtime_memory_state(
    db: Session,
    *,
    session_id: int,
    user_message: str,
    answer: str,
    max_messages: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    update_session_memory(
        session_id=session_id,
        user_message=user_message,
        answer=answer,
        max_messages=max_messages,
    )
