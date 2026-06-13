from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sqlalchemy.orm import Session

from core.db.models import Agent, AgentVersion
from core.runtime.skill_runtime import runtime_skill_bindings
from core.runtime.spec import default_workflow
from core.services.agents import get_agent_detail, normalize_memory, normalize_rag, normalize_tool_policy, normalize_workdir
from core.services.models import resolve_agent_model
from core.services.user_models import resolve_user_model_config, user_model_runtime_config


def build_runtime_agent(db: Session, agent: Agent, mode: str, user_id: int) -> SimpleNamespace:
    # Published mode falls back to draft if the agent has never been published.
    if mode == "published" and not agent.published_version_id:
        mode = "draft"

    if mode not in {"draft", "published"}:
        raise ValueError("mode must be draft or published")
    if mode == "published":
        source = published_agent_source(db, agent)
    else:
        source = draft_agent_source(db, agent)

    user_model_config = resolve_enabled_user_model_config(db, user_id, source["user_model_config_id"])
    runtime_config = user_model_runtime_config(user_model_config) if user_model_config else None
    model_config = resolve_agent_model(db, model_id=source["model_id"], model_name=source["model"])

    return SimpleNamespace(
        id=agent.id,
        workspace_id=agent.workspace_id,
        base_system_prompt=source["system_prompt"],
        system_prompt=source["system_prompt"],
        model_id=source["model_id"],
        user_model_config_id=source["user_model_config_id"],
        model=(runtime_config or {}).get("chat_model") or source["model"],
        temperature=source["temperature"],
        knowledge_base_ids=source["knowledge_base_ids"],
        tool_ids=source["tool_ids"],
        skill_bindings=runtime_skill_bindings(db, agent),
        workflow=source["workflow"],
        model_config=model_config,
        user_model_config=user_model_config,
        runtime_config=runtime_config,
        capability_config=user_model_config or model_config,
        settings={
            "variables": source["variables"],
            "memory": source["memory"],
            "rag": source["rag"],
            "tool_policy": source["tool_policy"],
            "workdir": source["workdir"],
        },
    )


def draft_agent_source(db: Session, agent: Agent) -> dict:
    detail = get_agent_detail(db, agent)
    return {
        "system_prompt": agent.system_prompt,
        "model_id": agent.model_id,
        "model": agent.model,
        "temperature": agent.temperature,
        "knowledge_base_ids": detail.get("knowledge_base_ids") or [],
        "tool_ids": [tool.get("id") for tool in detail.get("tools", []) if tool.get("id")],
        "workflow": detail.get("workflow") or default_workflow(),
        "variables": detail.get("variables") or [],
        "memory": normalize_memory(detail.get("memory")),
        "rag": normalize_rag(detail.get("rag")),
        "tool_policy": normalize_tool_policy(detail.get("tool_policy")),
        "workdir": normalize_workdir(detail.get("workdir")),
        "user_model_config_id": agent.user_model_config_id,
    }


def published_agent_source(db: Session, agent: Agent) -> dict:
    if not agent.published_version_id:
        raise ValueError("当前智能体还没有发布版本")
    version = db.get(AgentVersion, agent.published_version_id)
    if not version:
        raise ValueError("发布版本不存在")
    snapshot = version.snapshot or {}
    return {
        "system_prompt": snapshot.get("system_prompt", agent.system_prompt),
        "model_id": snapshot.get("model_id", agent.model_id),
        "model": snapshot.get("model", agent.model),
        "temperature": snapshot.get("temperature", agent.temperature),
        "knowledge_base_ids": snapshot.get("knowledge_base_ids") or [],
        "tool_ids": [tool.get("id") for tool in snapshot.get("tools", []) if tool.get("id")],
        "workflow": snapshot.get("workflow") or default_workflow(),
        "variables": snapshot.get("variables") or [],
        "memory": normalize_memory(snapshot.get("memory")),
        "rag": normalize_rag(snapshot.get("rag")),
        "tool_policy": normalize_tool_policy(snapshot.get("tool_policy")),
        "workdir": normalize_workdir(snapshot.get("workdir")),
        "user_model_config_id": snapshot.get("user_model_config_id", agent.user_model_config_id),
    }


def resolve_enabled_user_model_config(db: Session, user_id: int, config_id: int | None):
    if config_id is None:
        return None
    return resolve_user_model_config(db, user_id=user_id, config_id=config_id, enabled_only=True)


def validate_model_capabilities(model: Any, uploads: list[Any]) -> None:
    if not model:
        return
    has_image = any(upload.kind == "image" for upload in uploads)
    if has_image and not getattr(model, "supports_image", False):
        raise ValueError("Selected model does not support image input")
    has_document = any(upload.kind == "document" for upload in uploads)
    if has_document and not getattr(model, "supports_document", True):
        raise ValueError("Selected model does not support document input")
