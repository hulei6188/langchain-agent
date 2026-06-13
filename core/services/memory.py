from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.orm import Session

from core.db.models import AgentMemoryProfile, SessionMemory


MAX_MEMORY_SUMMARY_CHARS = 4000
MAX_MEMORY_FACTS = 50

graph_memory_checkpointer = InMemorySaver()
graph_memory_store = InMemoryStore()


@dataclass
class GraphMemoryContext:
    session_memory: SessionMemory | None
    profile_memory: AgentMemoryProfile | None
    session_summary: str
    profile_text: str
    profile_event: dict


def graph_memory_namespace(*, workspace_id: int, user_id: int, agent_id: int) -> tuple[str, ...]:
    return ("workspace", str(workspace_id), "user", str(user_id), "agent", str(agent_id), "memory")


def session_memory_namespace(*, session_id: int) -> tuple[str, ...]:
    return ("session", str(session_id), "memory")


def default_memory_profile_payload(agent_id: int) -> dict:
    return {
        "agent_id": agent_id,
        "enabled": False,
        "summary": "",
        "facts": [],
        "preferences": {},
        "updated_at": None,
    }


def memory_profile_payload(profile: AgentMemoryProfile | None, *, agent_id: int | None = None) -> dict:
    if not profile:
        if agent_id is None:
            raise ValueError("agent_id is required for a default memory profile payload")
        return default_memory_profile_payload(agent_id)
    return {
        "agent_id": profile.agent_id,
        "enabled": bool(profile.enabled),
        "summary": profile.summary or "",
        "facts": normalize_facts(profile.facts),
        "preferences": normalize_preferences(profile.preferences),
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


def get_memory_profile(
    db: Session,
    *,
    workspace_id: int,
    user_id: int,
    agent_id: int,
) -> AgentMemoryProfile | None:
    return (
        db.query(AgentMemoryProfile)
        .filter(
            AgentMemoryProfile.workspace_id == workspace_id,
            AgentMemoryProfile.user_id == user_id,
            AgentMemoryProfile.agent_id == agent_id,
        )
        .first()
    )


def get_session_memory(db: Session, *, session_id: int) -> SessionMemory | None:
    return db.query(SessionMemory).filter(SessionMemory.session_id == session_id).first()


def load_graph_memory_context(
    db: Session,
    *,
    workspace_id: int,
    user_id: int,
    agent_id: int,
    session_id: int,
) -> GraphMemoryContext:
    session_memory = get_session_memory(db, session_id=session_id)
    profile_memory = get_memory_profile(
        db,
        workspace_id=workspace_id,
        user_id=user_id,
        agent_id=agent_id,
    )
    profile_payload = memory_profile_payload(profile_memory, agent_id=agent_id)
    session_payload = {
        "session_id": session_id,
        "summary": session_memory.summary if session_memory else "",
        "message_count": session_memory.message_count if session_memory else 0,
    }
    graph_memory_store.put(
        graph_memory_namespace(workspace_id=workspace_id, user_id=user_id, agent_id=agent_id),
        "profile",
        profile_payload,
    )
    graph_memory_store.put(
        session_memory_namespace(session_id=session_id),
        "summary",
        session_payload,
    )
    return GraphMemoryContext(
        session_memory=session_memory,
        profile_memory=profile_memory,
        session_summary=session_payload["summary"],
        profile_text=format_profile_memory(profile_memory),
        profile_event=memory_used_event(profile_memory, session_summary_used=bool(session_payload["summary"])),
    )


def upsert_memory_profile(
    db: Session,
    *,
    workspace_id: int,
    user_id: int,
    agent_id: int,
    payload: dict,
) -> AgentMemoryProfile:
    profile = get_memory_profile(db, workspace_id=workspace_id, user_id=user_id, agent_id=agent_id)
    if not profile:
        profile = AgentMemoryProfile(
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        db.add(profile)

    if "enabled" in payload:
        profile.enabled = bool(payload["enabled"])
    if "summary" in payload:
        profile.summary = normalize_summary(payload["summary"])
    if "facts" in payload:
        profile.facts = normalize_facts(payload["facts"])
    if "preferences" in payload:
        profile.preferences = normalize_preferences(payload["preferences"])
    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile


def delete_memory_profile(
    db: Session,
    *,
    workspace_id: int,
    user_id: int,
    agent_id: int,
) -> bool:
    profile = get_memory_profile(db, workspace_id=workspace_id, user_id=user_id, agent_id=agent_id)
    if not profile:
        return False
    db.delete(profile)
    db.commit()
    return True


def memory_used_event(profile: AgentMemoryProfile | None, *, session_summary_used: bool) -> dict:
    return {
        "enabled": bool(profile.enabled) if profile else False,
        "profile_found": bool(profile),
        "summary_used": bool(profile and profile.enabled and (profile.summary or "").strip()),
        "facts_count": len(normalize_facts(profile.facts)) if profile and profile.enabled else 0,
        "preferences_keys": sorted(normalize_preferences(profile.preferences).keys()) if profile and profile.enabled else [],
        "session_summary_used": bool(session_summary_used),
    }


def format_profile_memory(profile: AgentMemoryProfile | None) -> str:
    if not profile or not profile.enabled:
        return ""
    parts = []
    summary = (profile.summary or "").strip()
    if summary:
        parts.append(f"Long-term memory summary:\n{summary}")
    facts = normalize_facts(profile.facts)
    if facts:
        parts.append("Long-term memory facts:\n" + "\n".join(f"- {item}" for item in facts))
    preferences = normalize_preferences(profile.preferences)
    if preferences:
        lines = [f"- {key}: {value}" for key, value in sorted(preferences.items())]
        parts.append("Long-term memory preferences:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def update_session_memory(
    db: Session,
    *,
    session_id: int,
    user_message: str,
    answer: str,
    max_messages: int,
) -> SessionMemory:
    memory = get_session_memory(db, session_id=session_id)
    if not memory:
        memory = SessionMemory(session_id=session_id, summary="", message_count=0)
        db.add(memory)
    memory.message_count += 2
    memory.summary = serialize_session_memory_turns(
        memory.summary,
        user_message=user_message,
        answer=answer,
        max_messages=max_messages,
    )
    graph_memory_store.put(
        session_memory_namespace(session_id=session_id),
        "summary",
        {
            "session_id": session_id,
            "summary": memory.summary,
            "message_count": memory.message_count,
        },
    )
    return memory


def serialize_session_memory_turns(
    existing_summary: str,
    *,
    user_message: str,
    answer: str,
    max_messages: int,
) -> str:
    try:
        dialogue_turns = json.loads(existing_summary) if existing_summary else []
        if not isinstance(dialogue_turns, list):
            dialogue_turns = []
    except Exception:
        dialogue_turns = []
        if str(existing_summary or "").strip():
            raw_turns = str(existing_summary).split("\n===\n")
            for turn_text in raw_turns:
                if "助手：" in turn_text:
                    parts = turn_text.split("助手：", 1)
                    user_part = parts[0].replace("用户：", "").strip()
                    assistant_part = parts[1].strip()
                    dialogue_turns.append({"user": user_part, "assistant": assistant_part})

    dialogue_turns.append({"user": user_message.strip(), "assistant": answer.strip()})
    max_turns = max(1, max_messages // 2)
    truncated_turns = dialogue_turns[-max_turns:]
    serialized = json.dumps(truncated_turns, ensure_ascii=False)
    if len(serialized) > 2000:
        for turn in truncated_turns:
            if len(turn["assistant"]) > 500:
                turn["assistant"] = turn["assistant"][:500] + "...(此回答过长已截断)..."
        serialized = json.dumps(truncated_turns, ensure_ascii=False)
    return serialized


def normalize_summary(value) -> str:
    return str(value or "").strip()[:MAX_MEMORY_SUMMARY_CHARS]


def normalize_facts(value) -> list[str]:
    if not isinstance(value, list):
        return []
    facts = []
    for item in value:
        text = str(item or "").strip()
        if text:
            facts.append(text)
    return facts[:MAX_MEMORY_FACTS]


def normalize_preferences(value) -> dict:
    if not isinstance(value, dict):
        return {}
    normalized = {}
    for key, raw_value in value.items():
        clean_key = str(key or "").strip()
        if not clean_key:
            continue
        clean_value = normalize_preference_value(raw_value)
        if clean_value is not None:
            normalized[clean_key] = clean_value
    return normalized


def normalize_preference_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                normalized.append(item)
        return normalized
    return None
