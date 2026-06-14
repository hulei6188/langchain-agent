from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.deps import get_current_membership, get_current_user, require_manager
from api.routers.auth import router as auth_router
from api.routers.health import create_health_router
from api.routers.knowledge import router as knowledge_router
from api.routers.memory import router as memory_router
from api.routers.models import router as models_router
from api.routers.prompt_templates import router as prompt_templates_router
from api.routers.runs import router as runs_router
from api.routers.sessions import router as sessions_router
from api.routers.skills import router as skills_router
from api.routers.tools import router as tools_router
from api.schemas import (
    AgentCreateRequest,
    AgentSkillsRequest,
    AgentUpdateRequest,
    ChatRequest,
    UploadCreateRequest,
    WorkflowUpdateRequest,
)
from core.config import get_settings
from core.db.models import (
    Agent,
    AgentSkill,
    AgentVersion,
    Message,
    ModelConfig,
    Run,
    Session as ChatSession,
    Skill,
    SkillKnowledgeBase,
    SkillTool,
    Tool,
    User,
    UserModelConfig,
    WorkflowDefinition,
    WorkspaceMember,
)
from core.db.session import SessionLocal, get_db, init_db
from core.runtime.spec import default_workflow, workflow_graph_spec
from core.runtime.langgraph_persistence import close_langgraph_persistence
from core.security.permissions import can_manage
from core.services.agents import (
    agent_summary,
    approve_agent,
    copy_agent_from_market,
    create_agent,
    delete_agent as delete_agent_service,
    ensure_template_agents_published,
    exclude_deprecated_template_agents,
    get_agent_detail,
    market_agent_summary,
    publish_agent,
    reject_agent,
    update_agent,
)
from core.services.run_streams import stream_workflow_sse
from core.services.tools import validate_tool_ids
from core.services.uploads import create_upload, upload_payload
from core.services.web_search import search_web, web_search_status


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(value or "").split("."):
        match = re.match(r"(\d+)", token)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts or [0])


def _validate_runtime_dependencies() -> None:
    import fastapi as fastapi_pkg
    import starlette as starlette_pkg

    starlette_version = getattr(starlette_pkg, "__version__", "0")
    if not ((0, 40, 0) <= _version_tuple(starlette_version) < (0, 42, 0)):
        raise RuntimeError(
            "Incompatible dependency set detected: "
            f"fastapi {getattr(fastapi_pkg, '__version__', 'unknown')} requires "
            f"starlette>=0.40.0,<0.42.0, but found starlette {starlette_version}. "
            "This usually happens when MCP-related dependencies upgrade Starlette transitively. "
            "Reinstall with the same interpreter used to start the server, for example: "
            "`python -m pip install -r requirements.txt` and then "
            "`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`."
        )


settings = get_settings()
_validate_runtime_dependencies()
app = FastAPI(title=settings.app_name, version=settings.app_version)
logger = logging.getLogger(__name__)
startup_error: str | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["http://127.0.0.1:5174", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_health_router(lambda: startup_error))
app.include_router(auth_router)
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(models_router)
app.include_router(prompt_templates_router)
app.include_router(runs_router)
app.include_router(sessions_router)
app.include_router(skills_router)
app.include_router(tools_router)


@app.on_event("startup")
def startup() -> None:
    global startup_error
    if settings.jwt_secret == "change-me-in-production":
        logger.warning(
            "SECURITY WARNING: JWT_SECRET is using the insecure default value. "
            "Set a strong random secret via the JWT_SECRET environment variable."
        )
    try:
        init_db()
        _cleanup_zombie_runs()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)[:500]
        logger.exception("Database initialization failed; API started in degraded mode")


@app.on_event("shutdown")
def shutdown() -> None:
    close_langgraph_persistence()


def _cleanup_zombie_runs() -> None:
    """Mark stale running runs as failed after server restart or crash."""
    try:
        db = SessionLocal()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        zombie_runs = (
            db.query(Run)
            .filter(Run.status == "running", Run.started_at < cutoff)
            .all()
        )
        if zombie_runs:
            for run in zombie_runs:
                run.status = "failed"
                run.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("Cleaned up %d zombie runs (stuck running >30min)", len(zombie_runs))
    except Exception:
        db.rollback()
        logger.exception("Failed to clean up zombie runs")
    finally:
        db.close()


@app.get("/api/search/test")
def test_web_search(q: str = Query(min_length=1, max_length=300), membership: WorkspaceMember = Depends(get_current_membership)):
    try:
        return {"ok": True, **search_web(q)}
    except ValueError as exc:
        return {"ok": False, "query": q, "provider": web_search_status().get("provider", settings.web_search_provider), "items": [], "error_code": str(exc)}


# ── Agent Skills ─────────────────────────────────────────────────────


@app.put("/api/agents/{agent_id}/skills")
def update_agent_skills(
    agent_id: int,
    request: AgentSkillsRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    # Remove existing bindings and recreate
    db.query(AgentSkill).filter(AgentSkill.agent_id == agent.id).delete()
    for skill_id in request.skill_ids:
        skill = db.query(Skill).filter(
            Skill.id == skill_id,
            Skill.workspace_id == membership.workspace_id,
        ).first()
        if not skill:
            raise HTTPException(status_code=400, detail=f"Skill {skill_id} not found or not accessible")
        db.add(AgentSkill(agent_id=agent.id, skill_id=skill_id))
    db.commit()
    return {"agent": get_agent_detail(db, agent)}


@app.post("/api/uploads")
def upload_file(request: UploadCreateRequest, membership: WorkspaceMember = Depends(get_current_membership), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        upload = create_upload(
            db,
            workspace_id=membership.workspace_id,
            user_id=current_user.id,
            filename=request.filename,
            content_type=request.content_type,
            content_base64=request.content_base64,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"upload": upload_payload(upload)}


@app.get("/api/agents")
def list_agents(membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    ensure_template_agents_published(db, membership.workspace_id)
    query = db.query(Agent).filter(Agent.workspace_id == membership.workspace_id)
    query = exclude_deprecated_template_agents(query)
    if not can_manage(membership.role):
        query = query.filter(Agent.created_by == membership.user_id)
    agents = query.order_by(Agent.updated_at.desc()).all()
    return {"items": [agent_summary(agent) for agent in agents]}


@app.post("/api/agents")
def create_agent_endpoint(request: AgentCreateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    try:
        validate_tool_ids(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_ids=request.tool_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = apply_model_selection(db, request.model_dump(), user_id=membership.user_id)
    agent = create_agent(db, workspace_id=membership.workspace_id, user_id=membership.user_id, payload=payload)
    return {"agent": get_agent_detail(db, agent)}


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    return {"agent": get_agent_detail(db, agent)}


@app.patch("/api/agents/{agent_id}")
def patch_agent(agent_id: int, request: AgentUpdateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    if request.tool_ids is not None:
        try:
            validate_tool_ids(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_ids=request.tool_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent = update_agent(db, agent, apply_model_selection(db, request.model_dump(exclude_unset=True), user_id=membership.user_id))
    return {"agent": get_agent_detail(db, agent)}


@app.delete("/api/agents/{agent_id}")
def delete_agent_endpoint(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    try:
        delete_agent_service(db, agent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True}


@app.post("/api/agents/{agent_id}/publish")
def publish_agent_endpoint(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    require_review = not can_manage(membership.role)
    version = publish_agent(db, agent, membership.user_id, require_review=require_review)
    return {
        "status": agent.status,
        "review_required": require_review,
        "version": {"id": version.id, "version": version.version, "snapshot": version.snapshot},
    }


@app.get("/api/admin/agent-reviews")
def list_agent_reviews(membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    agents = (
        db.query(Agent)
        .filter(Agent.workspace_id == membership.workspace_id, Agent.status == "pending_review")
        .order_by(Agent.updated_at.desc())
        .all()
    )
    return {"items": [review_payload(db, agent) for agent in agents]}


@app.post("/api/admin/agent-reviews/{agent_id}/approve")
def approve_agent_review(agent_id: int, membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    if agent.status != "pending_review":
        raise HTTPException(status_code=400, detail="Agent is not pending review")
    version = approve_agent(db, agent, membership.user_id)
    return {"agent": get_agent_detail(db, agent), "version": {"id": version.id, "version": version.version}}


@app.post("/api/admin/agent-reviews/{agent_id}/reject")
def reject_agent_review(agent_id: int, membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    if agent.status != "pending_review":
        raise HTTPException(status_code=400, detail="Agent is not pending review")
    reject_agent(db, agent)
    return {"agent": get_agent_detail(db, agent)}


@app.get("/api/market/agents")
def list_market_agents(membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agents = (
        exclude_deprecated_template_agents(db.query(Agent))
        .filter(
            Agent.workspace_id == membership.workspace_id,
            Agent.status == "published",
            Agent.published_version_id.isnot(None),
        )
        .order_by(Agent.updated_at.desc())
        .all()
    )
    items = []
    for agent in agents:
        version = db.get(AgentVersion, agent.published_version_id) if agent.published_version_id else None
        items.append(market_agent_summary(agent, version))
    return {"items": items}


@app.post("/api/market/agents/{agent_id}/copy")
def copy_market_agent(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    source = require_workspace_agent(db, membership.workspace_id, agent_id)
    if source.status != "published" or not source.published_version_id:
        raise HTTPException(status_code=404, detail="Market agent not found")
    try:
        copied = copy_agent_from_market(db, source=source, user_id=membership.user_id, workspace_id=membership.workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"agent": get_agent_detail(db, copied)}


@app.get("/api/agents/{agent_id}/versions")
def list_versions(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    versions = db.query(AgentVersion).filter_by(agent_id=agent.id).order_by(AgentVersion.version.desc()).all()
    return {"items": [{"id": item.id, "version": item.version, "created_at": item.created_at.isoformat()} for item in versions]}


@app.get("/api/agents/{agent_id}/draft")
def get_draft(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    return {"agent": get_agent_detail(db, agent)}


@app.get("/api/agents/{agent_id}/workflow")
def get_workflow(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    workflow = db.query(WorkflowDefinition).filter(WorkflowDefinition.agent_id == agent.id).first()
    graph = workflow_graph_spec(workflow.nodes if workflow else default_workflow())
    return {"graph": graph, "nodes": graph["nodes"]}


@app.patch("/api/agents/{agent_id}/workflow")
def update_workflow(agent_id: int, request: WorkflowUpdateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    graph = workflow_graph_spec(workflow_request_definition(request))
    validate_workflow_nodes(graph["nodes"])
    workflow = db.query(WorkflowDefinition).filter(WorkflowDefinition.agent_id == agent.id).first()
    if not workflow:
        workflow = WorkflowDefinition(agent_id=agent.id, nodes=graph)
        db.add(workflow)
    else:
        workflow.nodes = graph
    db.commit()
    return {"graph": graph, "nodes": graph["nodes"]}


@app.post("/api/agents/{agent_id}/chat/stream")
def chat_stream(agent_id: int, request: ChatRequest, membership: WorkspaceMember = Depends(get_current_membership), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    print(f"[DEBUG CHAT] agent_id: {agent_id} | mode: {request.mode} | session_id: {request.session_id}")
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)

    session = get_or_create_session(db, agent, current_user.id, request.session_id, request.message, is_debug=getattr(request, "is_debug", False))
    user_message = Message(session_id=session.id, role="user", content=request.message, sources=[])
    db.add(user_message)
    db.commit()

    bg_params = {
        "agent_id": agent.id,
        "session_id": session.id,
        "user_id": current_user.id,
        "user_message": request.message,
        "user_message_id": user_message.id,
        "mode": request.mode,
        "variables": request.variables,
        "rag_enabled": request.rag_enabled,
        "rag_options": request.rag_options.model_dump(exclude_none=True) if request.rag_options else None,
        "thinking_enabled": request.thinking_enabled,
        "search_enabled": request.search_enabled,
        "attachments": request.attachments,
    }
    return StreamingResponse(stream_workflow_sse(bg_params), media_type="text/event-stream")


def require_workspace_agent(db: Session, workspace_id: int, agent_id: int) -> Agent:
    agent = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def require_agent_read_access(agent: Agent, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if agent.created_by == membership.user_id:
        return
    raise HTTPException(status_code=403, detail="Agent access denied")


def require_agent_write_access(agent: Agent, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if agent.created_by == membership.user_id:
        return
    raise HTTPException(status_code=403, detail="Agent edit denied")


def review_payload(db: Session, agent: Agent) -> dict:
    version = db.query(AgentVersion).filter(AgentVersion.agent_id == agent.id).order_by(AgentVersion.version.desc()).first()
    return {
        **agent_summary(agent),
        "submitted_version": version.version if version else None,
        "submitted_at": version.created_at.isoformat() if version and version.created_at else None,
    }


def validate_workflow_nodes(nodes: list[dict]) -> None:
    allowed = {"Start", "LLM", "Knowledge", "Tool", "Answer"}
    seen = {node.get("type") for node in nodes}
    if not seen.issubset(allowed):
        raise HTTPException(status_code=400, detail="Unsupported workflow node type")
    if not {"Start", "Answer"}.issubset(seen):
        raise HTTPException(status_code=400, detail="Workflow requires Start and Answer nodes")


def workflow_request_definition(request: WorkflowUpdateRequest) -> dict:
    if request.graph is not None:
        return request.graph
    return request.model_dump(exclude_none=True, exclude={"graph"})


def apply_model_selection(db: Session, payload: dict, *, user_id: int) -> dict:
    if payload.get("user_model_config_id"):
        config = db.query(UserModelConfig).filter(
            UserModelConfig.id == payload["user_model_config_id"],
            UserModelConfig.user_id == user_id,
            UserModelConfig.enabled.is_(True),
        ).first()
        if not config:
            raise HTTPException(status_code=400, detail="User model config is not available")
        payload["model_id"] = None
        payload["model"] = config.chat_model
        if payload.get("temperature") is None:
            payload["temperature"] = config.default_temperature
        return payload
    if (
        not payload.get("model_id")
        and not payload.get("user_model_config_id")
        and "model_id" in payload
        and "user_model_config_id" in payload
    ):
        default_user_model = db.query(UserModelConfig).filter(
            UserModelConfig.user_id == user_id,
            UserModelConfig.enabled.is_(True),
            UserModelConfig.is_default.is_(True),
        ).order_by(UserModelConfig.id.asc()).first()
        if default_user_model:
            payload["user_model_config_id"] = default_user_model.id
            payload["model_id"] = None
            payload["model"] = default_user_model.chat_model
            if payload.get("temperature") is None:
                payload["temperature"] = default_user_model.default_temperature
            return payload
        payload["model"] = None
    if payload.get("model_id"):
        payload["user_model_config_id"] = None
    if not payload.get("model_id"):
        return payload
    model = db.get(ModelConfig, payload["model_id"])
    if not model or not model.enabled:
        raise HTTPException(status_code=400, detail="Model is not available")
    payload["model"] = model.model_name
    if payload.get("temperature") is None:
        payload["temperature"] = model.default_temperature
    return payload


def get_or_create_session(db: Session, agent: Agent, user_id: int, session_id: int | None, title_seed: str, is_debug: bool = False) -> ChatSession:
    if session_id:
        session = db.get(ChatSession, session_id)
        if session and session.agent_id == agent.id and session.user_id == user_id:
            return session
    session = ChatSession(
        workspace_id=agent.workspace_id,
        agent_id=agent.id,
        user_id=user_id,
        title="新会话",
        is_debug=is_debug,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session
