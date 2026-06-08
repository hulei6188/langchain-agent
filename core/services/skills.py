from __future__ import annotations

from sqlalchemy.orm import Session

from core.db.models import (
    AgentSkill,
    KnowledgeBase,
    Skill,
    SkillKnowledgeBase,
    SkillTool,
    Tool,
)
from core.services.tools import tool_payload


def skill_summary(skill: Skill) -> dict:
    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "icon": skill.icon,
        "category": skill.category,
        "tags": skill.tags or [],
        "enabled": skill.enabled,
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
    }


def get_skill_detail(db: Session, skill: Skill) -> dict:
    """Expand a Skill into its full detail including tools and knowledge bases."""
    skill_tool_rows = (
        db.query(SkillTool).filter(SkillTool.skill_id == skill.id).all()
    )
    tool_ids = [row.tool_id for row in skill_tool_rows]
    tools = db.query(Tool).filter(Tool.id.in_(tool_ids)).all() if tool_ids else []

    skill_kb_rows = (
        db.query(SkillKnowledgeBase)
        .filter(SkillKnowledgeBase.skill_id == skill.id)
        .all()
    )
    kb_ids = [row.knowledge_base_id for row in skill_kb_rows]
    kbs = (
        db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(kb_ids)).all()
        if kb_ids
        else []
    )

    return {
        **skill_summary(skill),
        "system_prompt": skill.system_prompt,
        "rag_config": skill.rag_config or {},
        "memory_config": skill.memory_config or {},
        "tools": [tool_payload(tool) for tool in tools],
        "knowledge_bases": [
            {"id": kb.id, "name": kb.name, "description": kb.description}
            for kb in kbs
        ],
    }


def list_workspace_skills(db: Session, workspace_id: int) -> list[dict]:
    skills = (
        db.query(Skill)
        .filter(Skill.workspace_id == workspace_id)
        .order_by(Skill.updated_at.desc())
        .all()
    )
    return [skill_summary(skill) for skill in skills]


def create_skill(
    db: Session, *, workspace_id: int, user_id: int, payload: dict
) -> Skill:
    skill = Skill(
        workspace_id=workspace_id,
        user_id=user_id,
        name=payload["name"],
        description=payload.get("description") or "",
        system_prompt=payload.get("system_prompt") or "",
        icon=payload.get("icon") or "SK",
        category=payload.get("category") or "general",
        tags=payload.get("tags") or [],
        rag_config=payload.get("rag_config") or {},
        memory_config=payload.get("memory_config") or {},
    )
    db.add(skill)
    db.flush()
    _replace_skill_tools(db, skill.id, payload.get("tool_ids") or [])
    _replace_skill_kbs(db, skill.id, payload.get("knowledge_base_ids") or [])
    db.commit()
    db.refresh(skill)
    return skill


def update_skill(db: Session, skill: Skill, payload: dict) -> Skill:
    for key in [
        "name",
        "description",
        "system_prompt",
        "icon",
        "category",
        "tags",
        "rag_config",
        "memory_config",
    ]:
        if key in payload and payload[key] is not None:
            setattr(skill, key, payload[key])
    if "enabled" in payload and payload["enabled"] is not None:
        skill.enabled = payload["enabled"]
    if "tool_ids" in payload and payload["tool_ids"] is not None:
        _replace_skill_tools(db, skill.id, payload["tool_ids"])
    if "knowledge_base_ids" in payload and payload["knowledge_base_ids"] is not None:
        _replace_skill_kbs(db, skill.id, payload["knowledge_base_ids"])
    db.commit()
    db.refresh(skill)
    return skill


def delete_skill(db: Session, skill: Skill) -> None:
    db.query(AgentSkill).filter(AgentSkill.skill_id == skill.id).delete(
        synchronize_session=False
    )
    db.query(SkillKnowledgeBase).filter(
        SkillKnowledgeBase.skill_id == skill.id
    ).delete(synchronize_session=False)
    db.query(SkillTool).filter(SkillTool.skill_id == skill.id).delete(
        synchronize_session=False
    )
    db.delete(skill)
    db.commit()


def _replace_skill_tools(db: Session, skill_id: int, tool_ids: list[int]) -> None:
    db.query(SkillTool).filter(SkillTool.skill_id == skill_id).delete()
    for tool_id in tool_ids:
        db.add(SkillTool(skill_id=skill_id, tool_id=tool_id))


def _replace_skill_kbs(db: Session, skill_id: int, kb_ids: list[int]) -> None:
    db.query(SkillKnowledgeBase).filter(
        SkillKnowledgeBase.skill_id == skill_id
    ).delete()
    for kb_id in kb_ids:
        db.add(SkillKnowledgeBase(skill_id=skill_id, knowledge_base_id=kb_id))
