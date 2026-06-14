from __future__ import annotations

from sqlalchemy.orm import Session

from core.config import get_settings
from core.db.models import (
    AgentTool,
    ModelConfig,
    Tool,
    User,
    Workspace,
    WorkspaceMember,
)
from core.security.auth import hash_password
from core.services.builtin_tools import BUILTIN_TOOLS
from core.runtime.spec import default_workflow


DEFAULT_WORKFLOW = default_workflow()


def ensure_builtin_tools(db: Session) -> None:
    # Clean up legacy builtin tools whose names are no longer in the registry.
    registry_names = set(BUILTIN_TOOLS.keys())
    legacy_ids = [
        row.id for row in db.query(Tool.id).filter(
            Tool.type == "builtin",
            Tool.name.notin_(registry_names),
        ).all()
    ]
    if legacy_ids:
        db.query(AgentTool).filter(AgentTool.tool_id.in_(legacy_ids)).delete(synchronize_session=False)
        db.query(Tool).filter(Tool.id.in_(legacy_ids)).delete(synchronize_session=False)

    # Upsert builtin tools from registry.
    for name, impl in BUILTIN_TOOLS.items():
        tool = db.query(Tool).filter(Tool.name == name, Tool.type == "builtin").first()
        if tool:
            tool.description = impl["description"]
            tool.label = name
            tool.enabled = True
            continue
        db.add(Tool(name=name, label=name, description=impl["description"], schema={}, type="builtin", enabled=True))

    # Upsert builtin_search adapter.
    search_tool = db.query(Tool).filter(Tool.name == "web_search").first()
    if search_tool:
        search_tool.type = "builtin_search"
        search_tool.label = "Web Search"
        search_tool.description = "Built-in web search adapter for Agent tools"
        search_tool.enabled = True
    else:
        db.add(Tool(name="web_search", label="Web Search", description="Built-in web search adapter for Agent tools", schema={}, type="builtin_search"))

    db.commit()


def ensure_default_models(db: Session) -> None:
    settings = get_settings()
    defaults = [
        {
            "model_name": settings.openai_model,
            "display_name": settings.openai_model,
            "provider": "openai-compatible",
            "supports_text": True,
            "supports_image": False,
            "supports_document": True,
            "supports_reasoning": True,
            "reasoning_type": "prompt",
            "reasoning_label": "提示词增强",
            "max_context": 131072,
            "default_temperature": 0.4,
        },
        {
            "model_name": "qwen-vl-plus",
            "display_name": "Qwen VL Plus",
            "provider": "openai-compatible",
            "supports_text": True,
            "supports_image": True,
            "supports_document": True,
            "supports_reasoning": True,
            "reasoning_type": "prompt",
            "reasoning_label": "提示词增强",
            "max_context": 32768,
            "default_temperature": 0.4,
        },
    ]
    changed = False
    for item in defaults:
        if db.query(ModelConfig).filter(ModelConfig.model_name == item["model_name"]).first():
            continue
        db.add(ModelConfig(enabled=True, **item))
        changed = True
    if changed:
        db.commit()


def create_first_user_workspace(db: Session, *, email: str, name: str, password: str) -> tuple[User, Workspace]:
    user = User(email=email.lower(), name=name, password_hash=hash_password(password))
    workspace = Workspace(name=f"{name} 的工作台", slug="default")
    db.add_all([user, workspace])
    db.flush()
    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="admin"))
    db.commit()
    ensure_default_models(db)
    ensure_builtin_tools(db)
    return user, workspace


def create_default_workspace_user(db: Session, *, email: str, name: str, password: str) -> tuple[User, Workspace]:
    workspace = db.query(Workspace).filter(Workspace.slug == "default").first()
    if not workspace:
        workspace = db.query(Workspace).order_by(Workspace.id.asc()).first()
    if not workspace:
        return create_first_user_workspace(db, email=email, name=name, password=password)
    user = User(email=email.lower(), name=name, password_hash=hash_password(password))
    db.add(user)
    db.flush()
    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="user"))
    db.commit()
    return user, workspace


def has_any_user(db: Session) -> bool:
    return db.query(User).first() is not None
