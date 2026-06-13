from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from core.db.models import Agent, AgentSkill, Skill, SkillKnowledgeBase, SkillTool
from core.runtime.skill_selection import (
    dedupe_skill_bindings,
    score_runtime_skills,
    skill_explicitly_requested,
    skill_manifest,
    skill_selection_text,
)
from core.services.rag import run_rag_pipeline
from core.services.skills import normalize_activation_mode


SKILL_AUTO_TOP_K = 3
SKILL_AUTO_THRESHOLD = 0.25
SKILL_SELECTION_HISTORY_MESSAGES = 8


def runtime_skill_bindings(db: Session, agent: Agent) -> list[dict]:
    agent_skill_rows = (
        db.query(AgentSkill)
        .filter(
            AgentSkill.agent_id == agent.id,
            AgentSkill.enabled.is_(True),
        )
        .all()
    )
    if not agent_skill_rows:
        return []

    skill_ids = [row.skill_id for row in agent_skill_rows]
    skills = db.query(Skill).filter(Skill.id.in_(skill_ids), Skill.enabled.is_(True)).all()
    skills_by_id = {skill.id: skill for skill in skills}
    priority_map = {row.skill_id: row.priority for row in agent_skill_rows}

    tool_ids_by_skill: dict[int, list[int]] = {skill_id: [] for skill_id in skill_ids}
    for row in db.query(SkillTool).filter(SkillTool.skill_id.in_(skill_ids)).all():
        tool_ids_by_skill.setdefault(row.skill_id, []).append(row.tool_id)

    kb_ids_by_skill: dict[int, list[int]] = {skill_id: [] for skill_id in skill_ids}
    for row in db.query(SkillKnowledgeBase).filter(SkillKnowledgeBase.skill_id.in_(skill_ids)).all():
        kb_ids_by_skill.setdefault(row.skill_id, []).append(row.knowledge_base_id)

    bindings = []
    for skill_id in skill_ids:
        skill = skills_by_id.get(skill_id)
        if not skill:
            continue
        mode = normalize_activation_mode(skill.activation_mode)
        if mode == "disabled":
            continue
        bindings.append(
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description or "",
                "category": skill.category or "general",
                "tags": skill.tags or [],
                "activation_mode": mode,
                "priority": priority_map.get(skill.id, 0),
                "system_prompt": skill.system_prompt or "",
                "tool_ids": list(dict.fromkeys(tool_ids_by_skill.get(skill.id, []))),
                "knowledge_base_ids": list(dict.fromkeys(kb_ids_by_skill.get(skill.id, []))),
            }
        )
    return sorted(bindings, key=lambda item: item.get("priority", 0), reverse=True)


def apply_runtime_skills(runtime: Any, context: dict, chat_session: Any) -> None:
    bindings = list(getattr(runtime, "skill_bindings", []) or [])
    manifest = [skill_manifest(item) for item in bindings]
    context["skill_manifest"] = manifest
    if not bindings:
        context["skill_selection"] = {
            "loaded": [],
            "auto_candidates": [],
            "threshold": SKILL_AUTO_THRESHOLD,
            "top_k": SKILL_AUTO_TOP_K,
        }
        return

    always_skills = [item for item in bindings if item["activation_mode"] == "always"]
    manual_skills = [
        item
        for item in bindings
        if item["activation_mode"] == "manual" and skill_explicitly_requested(item, context.get("input") or "")
    ]

    selection_text = skill_selection_text(
        runtime,
        context,
        chat_session,
        [item["name"] for item in always_skills + manual_skills],
        history_limit=SKILL_SELECTION_HISTORY_MESSAGES,
    )
    auto_candidates = score_runtime_skills(
        selection_text,
        [item for item in bindings if item["activation_mode"] == "auto"],
    )
    auto_skills = [item for item, score in auto_candidates if score >= SKILL_AUTO_THRESHOLD][:SKILL_AUTO_TOP_K]

    loaded = dedupe_skill_bindings(always_skills + manual_skills + auto_skills)
    score_by_id = {item["id"]: score for item, score in auto_candidates}

    skill_blocks = []
    tool_ids = list(getattr(runtime, "tool_ids", []) or [])
    kb_ids = list(getattr(runtime, "knowledge_base_ids", []) or [])
    loaded_payload = []
    for item in loaded:
        if item["system_prompt"].strip():
            skill_blocks.append(f"## Skill: {item['name']}\n{item['system_prompt'].strip()}")
        tool_ids.extend(item.get("tool_ids") or [])
        kb_ids.extend(item.get("knowledge_base_ids") or [])
        loaded_payload.append(
            {
                **skill_manifest(item),
                "score": round(score_by_id.get(item["id"], 1.0 if item["activation_mode"] == "always" else 0.0), 4),
                "tool_ids": item.get("tool_ids") or [],
                "knowledge_base_ids": item.get("knowledge_base_ids") or [],
            }
        )

    base_prompt = getattr(runtime, "base_system_prompt", runtime.system_prompt) or ""
    runtime.system_prompt = "\n\n".join([base_prompt, *skill_blocks]).strip()
    runtime.tool_ids = list(dict.fromkeys(tool_ids))
    runtime.knowledge_base_ids = list(dict.fromkeys(kb_ids))
    context["loaded_skills"] = loaded_payload
    context["skill_selection"] = {
        "loaded": loaded_payload,
        "auto_candidates": [
            {**skill_manifest(item), "score": round(score, 4)}
            for item, score in auto_candidates[:10]
        ],
        "threshold": SKILL_AUTO_THRESHOLD,
        "top_k": SKILL_AUTO_TOP_K,
    }


def skill_loader_tool(db: Session, agent: Any, context: dict) -> StructuredTool | None:
    schema = skill_loader_schema(agent, context)
    if not schema:
        return None
    description = schema["function"]["description"]

    def load_skill(skill_id: int | None = None, skill_name: str = "", reason: str = "") -> dict:
        return handle_load_skill_call(
            db,
            agent,
            context,
            {"skill_id": skill_id, "skill_name": skill_name, "reason": reason},
        )

    return StructuredTool.from_function(
        load_skill,
        name="load_skill",
        description=description,
    )


def skill_loader_schema(agent: Any, context: dict) -> dict | None:
    loaded_ids = {item.get("id") for item in context.get("loaded_skills", [])}
    loadable = [
        item
        for item in getattr(agent, "skill_bindings", []) or []
        if item.get("activation_mode") in {"auto", "manual"} and item.get("id") not in loaded_ids
    ]
    if not loadable:
        return None
    names = "、".join(f"{item['name']}#{item['id']}" for item in loadable[:20])
    return {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load one available Skill's full instructions and its bound resources for this turn. "
                "Use this only when the user explicitly asks for a manual Skill or the manifest shows a relevant Skill that was not loaded. "
                f"Loadable skills: {names}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "integer", "description": "Skill id from the manifest."},
                    "skill_name": {"type": "string", "description": "Skill name from the manifest, used only if skill_id is absent."},
                    "reason": {"type": "string", "description": "Why this Skill is required for the current task."},
                },
                "required": [],
            },
        },
    }


def handle_load_skill_call(db: Session, agent: Any, context: dict, args: dict) -> dict:
    raw_skill_id = args.get("skill_id")
    skill_id = None
    try:
        skill_id = int(raw_skill_id) if raw_skill_id not in (None, "") else None
    except (TypeError, ValueError):
        skill_id = None
    skill_name = str(args.get("skill_name") or "").strip().lower()
    loaded_ids = {item.get("id") for item in context.get("loaded_skills", [])}
    binding = None
    for item in getattr(agent, "skill_bindings", []) or []:
        if item.get("id") in loaded_ids:
            continue
        if item.get("activation_mode") not in {"auto", "manual"}:
            continue
        if skill_id is not None and item.get("id") == skill_id:
            binding = item
            break
        if skill_id is None and skill_name and str(item.get("name") or "").strip().lower() == skill_name:
            binding = item
            break
    if not binding:
        return {
            "status": "error",
            "content": json.dumps({"error": "skill_not_loadable", "requested": args}, ensure_ascii=False),
            "result_preview": "Skill not found, already loaded, or not loadable.",
            "latency_ms": 0,
        }

    payload = merge_loaded_skill(agent, context, binding, score=1.0, reason="load_skill")
    retrieved = retrieve_skill_knowledge(db, agent, context, binding)
    content = {
        "loaded_skill": payload,
        "retrieved_sources": len(retrieved),
        "reason": args.get("reason") or "",
    }
    return {
        "status": "success",
        "content": json.dumps(content, ensure_ascii=False),
        "result_preview": f"Loaded Skill: {binding['name']}",
        "latency_ms": 0,
    }


def merge_loaded_skill(agent: Any, context: dict, item: dict, *, score: float, reason: str) -> dict:
    loaded = context.setdefault("loaded_skills", [])
    if any(existing.get("id") == item["id"] for existing in loaded):
        return next(existing for existing in loaded if existing.get("id") == item["id"])
    if item.get("system_prompt", "").strip():
        agent.system_prompt = "\n\n".join(
            part
            for part in [
                getattr(agent, "system_prompt", "") or "",
                f"## Skill: {item['name']}\n{item['system_prompt'].strip()}",
            ]
            if part.strip()
        )
    agent.tool_ids = list(dict.fromkeys((getattr(agent, "tool_ids", []) or []) + (item.get("tool_ids") or [])))
    agent.knowledge_base_ids = list(dict.fromkeys((getattr(agent, "knowledge_base_ids", []) or []) + (item.get("knowledge_base_ids") or [])))
    payload = {
        **skill_manifest(item),
        "score": round(score, 4),
        "reason": reason,
        "tool_ids": item.get("tool_ids") or [],
        "knowledge_base_ids": item.get("knowledge_base_ids") or [],
    }
    loaded.append(payload)
    selection = context.setdefault(
        "skill_selection",
        {"loaded": [], "auto_candidates": [], "threshold": SKILL_AUTO_THRESHOLD, "top_k": SKILL_AUTO_TOP_K},
    )
    selection["loaded"] = loaded
    return payload


def retrieve_skill_knowledge(db: Session, agent: Any, context: dict, item: dict) -> list[dict]:
    kb_ids = item.get("knowledge_base_ids") or []
    if not kb_ids or not context.get("rag_enabled", True):
        return []
    rag_result = run_rag_pipeline(
        db,
        workspace_id=agent.workspace_id,
        knowledge_base_ids=kb_ids,
        query=context["input"],
        config=context.get("rag_config") or {},
        runtime_config=getattr(agent, "runtime_config", None),
    )
    existing = {json.dumps(source, ensure_ascii=False, sort_keys=True) for source in context.get("sources", [])}
    added = []
    for source in rag_result.sources:
        key = json.dumps(source, ensure_ascii=False, sort_keys=True)
        if key in existing:
            continue
        existing.add(key)
        added.append(source)
    if added:
        context["sources"] = [*context.get("sources", []), *added]
    return added
