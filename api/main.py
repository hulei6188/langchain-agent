from __future__ import annotations

import json
import logging
import queue
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from collections.abc import Iterable

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from pydantic import BaseModel, Field
from api.deps import get_current_membership, get_current_user, require_manager
from api.schemas import (
    AgentCreateRequest,
    AgentSkillsRequest,
    AgentUpdateRequest,
    ChatRequest,
    FeedbackRequest,
    InviteAcceptRequest,
    InviteCreateRequest,
    KnowledgeBaseCreateRequest,
    KnowledgeDocumentBatchCreateRequest,
    KnowledgeDocumentCreateRequest,
    LoginRequest,
    MCPToolDiscoverRequest,
    MemoryProfileUpdateRequest,
    ModelConfigRequest,
    ModelConfigUpdateRequest,
    PromptTemplateCopyBuiltinRequest,
    PromptTemplateRequest,
    PromptTemplateUpdateRequest,
    RegisterRequest,
    SessionUpdateRequest,
    SkillCreateRequest,
    SkillItemIdsRequest,
    SkillUpdateRequest,
    ToolRequest,
    ToolTestRequest,
    ToolUpdateRequest,
    UploadCreateRequest,
    UserProfileUpdateRequest,
    UserModelCapabilityTestRequest,
    UserModelConfigRequest,
    UserModelConfigUpdateRequest,
    WorkflowUpdateRequest,
)
from core.config import get_settings
from core.db.models import (
    Agent,
    AgentSkill,
    AgentVersion,
    Feedback,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    Message,
    ModelConfig,
    Run,
    RunStep,
    Session as ChatSession,
    SessionMemory,
    Skill,
    SkillKnowledgeBase,
    SkillTool,
    Tool,
    User,
    UserModelConfig,
    WorkflowDefinition,
    WorkspaceInvite,
    WorkspaceMember,
)
from core.db.session import SessionLocal, engine, get_db, init_db
from core.integrations.llm import DASHSCOPE_COMPATIBLE_BASE, OPENAI_COMPATIBLE_DEFAULT_BASE, OpenAICompatibleProvider, _CancelledError
from core.integrations.vector_store import vector_store
from core.runtime.cancel import cancel_run
from core.runtime.workflow import WorkflowRunner, default_workflow
from core.security.auth import create_access_token, hash_password, verify_password
from core.security.api_keys import secret_storage_ready
from core.security.permissions import can_manage, normalize_role
from core.services.agents import (
    agent_summary,
    approve_agent,
    copy_agent_from_market,
    create_agent,
    delete_agent as delete_agent_service,
    ensure_template_agents_published,
    get_agent_detail,
    market_agent_summary,
    publish_agent,
    reject_agent,
    update_agent,
)
from core.services.bootstrap import (
    create_default_workspace_user,
    create_first_user_workspace,
    ensure_builtin_tools,
    ensure_default_models,
    has_any_user,
)
from core.services.knowledge import (
    KnowledgeDocumentError,
    add_document,
    clear_knowledge_base_documents,
    create_knowledge_base,
    delete_document,
    delete_knowledge_base,
    document_payload,
    knowledge_base_summary,
    list_document_chunks,
    reindex_knowledge_base,
    split_by_hierarchy,
    split_parent_child,
    index_document,
)
from core.services.rag_cache import redis_store
from core.services.memory import (
    delete_memory_profile,
    get_memory_profile,
    memory_profile_payload,
    upsert_memory_profile,
)
from core.services.models import create_model_config, delete_model_config, model_payload, update_model_config
from core.services.prompt_templates import (
    copy_builtin_prompt_template,
    create_prompt_template,
    delete_prompt_template,
    get_owned_prompt_template,
    list_prompt_templates,
    prompt_template_payload,
    update_prompt_template,
)
from core.services.tools import (
    create_tool,
    delete_tool,
    discover_mcp_tools,
    get_accessible_tool,
    list_available_tools,
    test_tool,
    tool_payload,
    update_tool,
    validate_tool_ids,
)
from core.services.skills import (
    create_skill,
    delete_skill,
    get_skill_detail,
    list_workspace_skills,
    update_skill,
)
from core.services.uploads import create_upload, upload_payload
from core.services.user_models import (
    create_user_model_config,
    delete_user_model_config,
    get_owned_user_model,
    list_user_model_configs,
    test_user_model_config,
    test_user_model_payload,
    update_user_model_config,
    user_model_payload,
)
from core.services.web_search import search_web, web_search_status


PUBLIC_CHAT_ERRORS = (
    "Selected model does not support document input",
    "Selected model does not support image input",
    "Upload not found or not accessible",
    "Stored API key is invalid",
    "Secure API key encryption is not configured",
    "当前智能体还没有发布版本",
    "发布版本不存在",
    "mode must be draft or published",
    "Model call failed",
    "Model returned an empty answer",
    "Chat model API key is not configured",
    "Embedding API key is not configured",
    "Rerank API key is not configured",
)


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
_health_probe_cache: dict[str, tuple[float, dict]] = {}

# Per-run event log for SSE reconnection after page refresh.
# Each entry is the raw SSE string (e.g. "event: token\ndata: {...}\n\n").
_run_event_logs: dict[int, list[str]] = {}
_run_event_lock = threading.Lock()


def _append_run_event(run_id: int, sse_str: str) -> None:
    with _run_event_lock:
        if run_id not in _run_event_logs:
            _run_event_logs[run_id] = []
        _run_event_logs[run_id].append(sse_str)


def _get_run_events_since(run_id: int, index: int) -> tuple[list[str], int]:
    with _run_event_lock:
        log = _run_event_logs.get(run_id, [])
        events = log[index:]
        return events, len(log)


def _cleanup_run_events(run_id: int) -> None:
    with _run_event_lock:
        _run_event_logs.pop(run_id, None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["http://127.0.0.1:5174", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _cleanup_zombie_runs() -> None:
    """Mark stale running runs as failed after server restart or crash."""
    from datetime import timedelta

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


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def safe_stream_error(exc: Exception) -> dict:
    message = str(exc)
    if any(public_error in message for public_error in PUBLIC_CHAT_ERRORS):
        return {"message": _sanitize_public_error(message), "error_code": _error_code(message)}
    return {
        "message": "智能体运行失败，请检查模型、知识库或附件配置后重试。",
        "error_code": _error_code(message),
    }


def _error_code(message: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")
    if "model_call_failed" in normalized or "gateway" in normalized:
        return "model_provider_error"
    if "model" in normalized and "image" in normalized:
        return "model_capability_error"
    if "model" in normalized and "document" in normalized:
        return "model_capability_error"
    if "upload" in normalized:
        return "attachment_error"
    if "publish" in normalized or "发布" in message:
        return "agent_version_error"
    if "api_key" in normalized or "secret" in normalized:
        return "secret_config_error"
    return "agent_runtime_error"


@app.get("/api/health")
def health():
    provider = OpenAICompatibleProvider()
    chat_api_key = provider._api_key(settings, purpose="chat")
    embedding_api_key = provider._api_key(settings, purpose="embedding")
    model_mock = settings.mock_llm
    model_base = settings.openai_api_base
    if settings.deepseek_api_key and ((settings.openai_api_base or "").rstrip("/") == settings.deepseek_api_base.rstrip("/") or settings.openai_model == settings.deepseek_model):
        model_base = settings.deepseek_api_base
    elif settings.dashscope_api_key and not settings.openai_api_key and model_base == OPENAI_COMPATIBLE_DEFAULT_BASE:
        model_base = DASHSCOPE_COMPATIBLE_BASE
    embedding_base = provider._api_base(settings, purpose="embedding")
    embedding_mock = settings.mock_llm
    embedding_model = (settings.openai_embedding_model or "").strip()
    issues = []
    database_status = {"configured": bool(settings.database_url), "available": False, "error": None}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_status["available"] = True
    except Exception as exc:
        database_status["error"] = str(exc)[:240]
        issues.append("Database is configured but not reachable.")
    if not secret_storage_ready():
        issues.append("API_KEY_ENCRYPTION_KEY is required before storing user model keys or tool secrets.")
    if startup_error:
        issues.append("Database initialization failed during startup.")
    redis_status = redis_store.status()
    vector_status = vector_store.status()
    model_probe = _model_probe("chat", enabled=bool(settings.health_model_probe_enabled and not model_mock))
    embedding_probe = _model_probe("embedding", enabled=bool(settings.health_model_probe_enabled and not embedding_mock and embedding_model))
    if model_mock:
        issues.append("Chat model is running in mock mode because LINGSHU_MOCK_LLM is true.")
    elif not chat_api_key:
        issues.append("Chat model API key is not configured.")
    elif not model_probe["ok"]:
        issues.append("Chat model gateway probe failed.")
    if embedding_mock:
        issues.append("Embedding is running in mock mode because LINGSHU_MOCK_LLM is true.")
    elif not embedding_model or not embedding_api_key:
        issues.append("Embedding is unavailable for real RAG because OPENAI_EMBEDDING_MODEL and a provider API key are required.")
    elif not embedding_probe["ok"]:
        issues.append("Embedding gateway probe failed.")
    if redis_status["required"] and not redis_status["available"]:
        issues.append("Redis is configured for RAG cache/job state but is not reachable.")
    if vector_status["fallback"]:
        issues.append("Milvus is configured but unavailable; vector operations are using the in-memory fallback.")
    return {
        "status": "degraded" if issues else "ok",
        "version": app.version,
        "issues": issues,
        "dependencies": {
            "database": database_status,
            "startup": {"ok": startup_error is None, "error": startup_error},
            "cors": {"origins": settings.cors_origin_list},
            "redis": redis_status,
            "vector_store": vector_status,
            "model": {
                "provider": "openai-compatible",
                "model": settings.deepseek_model if model_base.rstrip("/") == settings.deepseek_api_base.rstrip("/") else settings.openai_model,
                "base_url": model_base,
                "mock": model_mock,
                "configured": bool(chat_api_key),
                "available": bool((not model_mock) and bool(chat_api_key)),
                "probe": model_probe,
            },
            "embedding": {
                "provider": "openai-compatible",
                "model": embedding_model,
                "base_url": embedding_base,
                "mock": embedding_mock,
                "configured": bool(embedding_model and embedding_api_key),
                "available": bool(embedding_model and embedding_api_key and not embedding_mock),
                "reason": None if bool(embedding_model and embedding_api_key and not embedding_mock) else _runtime_unavailable_reason(embedding_probe, vector_status),
                "probe": embedding_probe,
            },
            "web_search": web_search_status(),
            "secret_storage": {
                "configured": secret_storage_ready(),
            },
        },
    }


def _model_probe(purpose: str, *, enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "ok": False, "error": None, "cached": False}
    now = time.monotonic()
    cached = _health_probe_cache.get(purpose)
    if cached and now - cached[0] < 300:
        return {**cached[1], "cached": True}
    provider = OpenAICompatibleProvider()
    try:
        if purpose == "chat":
            settings_obj = get_settings()
            use_deepseek = bool(
                settings_obj.deepseek_api_key
                and (
                    (settings_obj.openai_api_base or "").rstrip("/") == settings_obj.deepseek_api_base.rstrip("/")
                    or settings_obj.openai_model == settings_obj.deepseek_model
                )
            )
            model = settings_obj.deepseek_model if use_deepseek else settings_obj.openai_model
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "health"}],
                "temperature": 0,
                "stream": False,
            }
            provider._post_json(
                provider._api_base(settings_obj, purpose="chat").rstrip("/") + "/chat/completions",
                payload,
                provider._api_key(settings_obj, purpose="chat") or "",
                timeout_seconds=8,
            )
        elif purpose == "embedding":
            settings_obj = get_settings()
            provider._post_json(
                provider._api_base(settings_obj, purpose="embedding").rstrip("/") + "/embeddings",
                {"model": settings_obj.openai_embedding_model, "input": "health"},
                provider._api_key(settings_obj, purpose="embedding") or "",
                timeout_seconds=8,
            )
        else:
            raise ValueError("Unsupported health probe")
        result = {"enabled": True, "ok": True, "error": None, "cached": False}
    except Exception as exc:
        result = {"enabled": True, "ok": False, "error": _sanitize_public_error(str(exc)), "cached": False}
    _health_probe_cache[purpose] = (now, result)
    return result


def _runtime_unavailable_reason(probe: dict, vector_status: dict) -> str:
    if not vector_status.get("available"):
        return "vector_store_unavailable"
    if probe.get("enabled") and not probe.get("ok"):
        return "provider_probe_failed"
    return "mock_or_vector_unavailable"


def _sanitize_public_error(message: str) -> str:
    cleaned = re.sub(r"(?i)(sk-[A-Za-z0-9_-]+|api[_-]?key\s*[:=]\s*\S+|secret\s*[:=]\s*\S+)", "[secret]", str(message))
    return cleaned.replace("\n", " ").replace("\r", " ").strip()[:500]


def add_knowledge_document_from_request(
    db: Session,
    *,
    workspace_id: int,
    kb: KnowledgeBase,
    request: KnowledgeDocumentCreateRequest,
) -> tuple[KnowledgeDocument, dict]:
    document = add_document(
        db,
        workspace_id=workspace_id,
        kb=kb,
        filename=request.filename,
        title=request.title,
        text=request.text,
        content=request.content,
        content_type=request.content_type,
        content_base64=request.content_base64,
        source_type=request.source_type,
    )
    payload = document_payload(document, db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document.id).count())
    return document, payload


@app.get("/api/search/test")
def test_web_search(q: str = Query(min_length=1, max_length=300), membership: WorkspaceMember = Depends(get_current_membership)):
    try:
        return {"ok": True, **search_web(q)}
    except ValueError as exc:
        return {"ok": False, "query": q, "provider": web_search_status().get("provider", settings.web_search_provider), "items": [], "error_code": str(exc)}


@app.post("/api/auth/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == request.email.lower()).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    invite = None
    if request.invite_token:
        invite = (
            db.query(WorkspaceInvite)
            .filter(
                WorkspaceInvite.token == request.invite_token,
                WorkspaceInvite.accepted_at.is_(None),
                WorkspaceInvite.email == request.email.lower(),
            )
            .first()
        )
        if not invite:
            raise HTTPException(status_code=404, detail="Invite not found")
    if invite:
        user = User(email=request.email.lower(), name=request.name, password_hash=hash_password(request.password))
        db.add(user)
        db.flush()
        db.add(WorkspaceMember(workspace_id=invite.workspace_id, user_id=user.id, role=normalize_role(invite.role)))
        invite.accepted_at = datetime.now(timezone.utc)
        db.commit()
        workspace = invite_workspace(db, invite.workspace_id)
        role = normalize_role(invite.role)
    elif has_any_user(db):
        user, workspace = create_default_workspace_user(db, email=request.email, name=request.name, password=request.password)
        role = "user"
    else:
        user, workspace = create_first_user_workspace(db, email=request.email, name=request.name, password=request.password)
        role = "admin"
    token = create_access_token(user.id, workspace.id)
    return {"access_token": token, "token_type": "bearer", "user": user_payload(user), "workspace": workspace_payload(workspace, role)}


@app.post("/api/auth/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    membership = db.query(WorkspaceMember).filter(WorkspaceMember.user_id == user.id).first()
    token = create_access_token(user.id, membership.workspace_id if membership else None)
    return {"access_token": token, "token_type": "bearer", "user": user_payload(user)}


@app.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user), membership: WorkspaceMember = Depends(get_current_membership)):
    return {"user": user_payload(current_user), "membership": membership_payload(membership)}


@app.patch("/api/auth/me")
def update_me(request: UserProfileUpdateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patch = request.model_dump(exclude_unset=True)
    user = db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if "name" in patch and patch["name"] is not None:
        user.name = patch["name"].strip()
    if "avatar_url" in patch:
        avatar_url = patch["avatar_url"] or ""
        if avatar_url and not avatar_url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="avatar_url must be an image data URL")
        user.avatar_url = avatar_url
    db.commit()
    db.refresh(user)
    return {"user": user_payload(user)}


@app.get("/api/workspaces/current")
def current_workspace(membership: WorkspaceMember = Depends(get_current_membership)):
    return {"workspace": workspace_payload(membership.workspace, membership.role), "membership": membership_payload(membership)}


@app.get("/api/workspaces/invites")
def list_invites(membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    if not settings.invite_api_enabled:
        raise HTTPException(status_code=404, detail="Invite API is disabled")
    invites = db.query(WorkspaceInvite).filter(WorkspaceInvite.workspace_id == membership.workspace_id).all()
    return {"items": [invite_payload(invite, include_token=False) for invite in invites]}


@app.get("/api/workspaces/members")
def list_members(membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    rows = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == membership.workspace_id)
        .order_by(WorkspaceMember.id.asc())
        .all()
    )
    return {
        "items": [
            {
                "id": row.id,
                "role": normalize_role(row.role),
                "user": user_payload(row.user),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


@app.post("/api/workspaces/invites")
def create_invite(request: InviteCreateRequest, membership: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    if not settings.invite_api_enabled:
        raise HTTPException(status_code=404, detail="Invite API is disabled")
    if normalize_role(request.role) != "user":
        raise HTTPException(status_code=400, detail="Invite role must be user")
    invite = WorkspaceInvite(
        workspace_id=membership.workspace_id,
        email=request.email.lower(),
        role="user",
        token=secrets.token_urlsafe(24),
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return {"invite": invite_payload(invite, include_token=False)}


@app.post("/api/workspaces/invites/accept")
def accept_invite(request: InviteAcceptRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not settings.invite_api_enabled:
        raise HTTPException(status_code=404, detail="Invite API is disabled")
    invite = db.query(WorkspaceInvite).filter(WorkspaceInvite.token == request.token, WorkspaceInvite.accepted_at.is_(None)).first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    existing = db.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == invite.workspace_id, WorkspaceMember.user_id == current_user.id).first()
    if not existing:
        db.add(WorkspaceMember(workspace_id=invite.workspace_id, user_id=current_user.id, role=normalize_role(invite.role)))
    invite.accepted_at = datetime.now(timezone.utc)
    db.commit()
    return {"accepted": True}


@app.get("/api/tools")
def list_tools(membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    ensure_builtin_tools(db)
    tools = list_available_tools(db, workspace_id=membership.workspace_id, user_id=membership.user_id)
    return {"items": [tool_payload(tool) for tool in tools]}


@app.post("/api/tools")
def create_tool_endpoint(request: ToolRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    try:
        tool = create_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, payload=request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": tool_payload(tool)}


@app.post("/api/tools/mcp/discover")
def discover_mcp_tools_endpoint(
    request: MCPToolDiscoverRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    existing_tool = None
    if request.tool_id:
        existing_tool = get_accessible_tool(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            tool_id=request.tool_id,
        )
        if not existing_tool:
            raise HTTPException(status_code=404, detail="Tool not found")
    try:
        items = discover_mcp_tools(request.model_dump(), existing_tool=existing_tool)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": items}


@app.patch("/api/tools/{tool_id}")
def patch_tool_endpoint(tool_id: int, request: ToolUpdateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    tool = get_accessible_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    try:
        tool = update_tool(db, tool=tool, payload=request.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": tool_payload(tool)}


@app.delete("/api/tools/{tool_id}")
def delete_tool_endpoint(tool_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    tool = get_accessible_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    try:
        delete_tool(db, tool=tool)
    except ValueError as exc:
        status_code = 409 if "in use" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return {"deleted": True}


@app.post("/api/tools/{tool_id}/test")
def test_tool_endpoint(tool_id: int, request: ToolTestRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    tool = get_accessible_tool(db, workspace_id=membership.workspace_id, user_id=membership.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return test_tool(tool, input_data=request.input, body=request.body)


# ── Skills ───────────────────────────────────────────────────────────


@app.get("/api/skills")
def list_skills(
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    items = list_workspace_skills(db, workspace_id=membership.workspace_id)
    return {"items": items}


@app.post("/api/skills")
def create_skill_endpoint(
    request: SkillCreateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    if request.tool_ids:
        try:
            validate_tool_ids(
                db,
                workspace_id=membership.workspace_id,
                user_id=membership.user_id,
                tool_ids=request.tool_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    skill = create_skill(
        db,
        workspace_id=membership.workspace_id,
        user_id=membership.user_id,
        payload=request.model_dump(),
    )
    return {"skill": get_skill_detail(db, skill)}


@app.get("/api/skills/{skill_id}")
def get_skill(
    skill_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    return {"skill": get_skill_detail(db, skill)}


@app.patch("/api/skills/{skill_id}")
def patch_skill(
    skill_id: int,
    request: SkillUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    if request.tool_ids is not None:
        try:
            validate_tool_ids(
                db,
                workspace_id=membership.workspace_id,
                user_id=membership.user_id,
                tool_ids=request.tool_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    skill = update_skill(db, skill=skill, payload=request.model_dump(exclude_unset=True))
    return {"skill": get_skill_detail(db, skill)}


@app.delete("/api/skills/{skill_id}")
def delete_skill_endpoint(
    skill_id: int,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    delete_skill(db, skill=skill)
    return {"deleted": True}


@app.put("/api/skills/{skill_id}/tools")
def update_skill_tools(
    skill_id: int,
    request: SkillItemIdsRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    from core.services.skills import _replace_skill_tools

    if request.ids:
        try:
            validate_tool_ids(
                db,
                workspace_id=membership.workspace_id,
                user_id=membership.user_id,
                tool_ids=request.ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    _replace_skill_tools(db, skill.id, request.ids)
    db.commit()
    return {"skill": get_skill_detail(db, skill)}


@app.put("/api/skills/{skill_id}/knowledge-bases")
def update_skill_knowledge_bases(
    skill_id: int,
    request: SkillItemIdsRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    skill = require_workspace_skill(db, membership.workspace_id, skill_id)
    from core.services.skills import _replace_skill_kbs

    _replace_skill_kbs(db, skill.id, request.ids)
    db.commit()
    return {"skill": get_skill_detail(db, skill)}


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


@app.get("/api/prompt-templates")
def list_prompt_templates_endpoint(
    include_disabled: bool = Query(default=False),
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    return {
        "items": list_prompt_templates(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            include_disabled=include_disabled,
        )
    }


@app.post("/api/prompt-templates")
def create_prompt_template_endpoint(
    request: PromptTemplateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    try:
        template = create_prompt_template(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            payload=request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"template": prompt_template_payload(template)}


@app.patch("/api/prompt-templates/{template_id}")
def patch_prompt_template_endpoint(
    template_id: int,
    request: PromptTemplateUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    template = get_owned_prompt_template(db, workspace_id=membership.workspace_id, user_id=membership.user_id, template_id=template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    try:
        template = update_prompt_template(db, template=template, payload=request.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"template": prompt_template_payload(template)}


@app.delete("/api/prompt-templates/{template_id}")
def delete_prompt_template_endpoint(template_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    template = get_owned_prompt_template(db, workspace_id=membership.workspace_id, user_id=membership.user_id, template_id=template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    delete_prompt_template(db, template=template)
    return {"deleted": True}


@app.post("/api/prompt-templates/copy-builtin")
def copy_builtin_prompt_template_endpoint(
    request: PromptTemplateCopyBuiltinRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    try:
        template = copy_builtin_prompt_template(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            builtin_id=request.builtin_id,
            title=request.title,
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return {"template": prompt_template_payload(template)}


@app.get("/api/models")
def list_models(
    include_disabled: bool = Query(default=False),
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    ensure_default_models(db)
    query = db.query(ModelConfig)
    if include_disabled:
        if not can_manage(membership.role):
            raise HTTPException(status_code=403, detail="Admin role required")
    else:
        query = query.filter(ModelConfig.enabled.is_(True))
    models = query.order_by(ModelConfig.id.asc()).all()
    return {"items": [model_payload(model) for model in models]}


@app.post("/api/admin/models")
def create_model(request: ModelConfigRequest, _: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    if db.query(ModelConfig).filter(ModelConfig.model_name == request.model_name).first():
        raise HTTPException(status_code=409, detail="Model already exists")
    model = create_model_config(db, request.model_dump())
    return {"model": model_payload(model)}


@app.patch("/api/admin/models/{model_id}")
def patch_model(model_id: int, request: ModelConfigUpdateRequest, _: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    model = db.get(ModelConfig, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    payload = request.model_dump(exclude_unset=True)
    if "model_name" in payload and payload["model_name"] != model.model_name:
        existing = db.query(ModelConfig).filter(ModelConfig.model_name == payload["model_name"]).first()
        if existing:
            raise HTTPException(status_code=409, detail="Model already exists")
    model = update_model_config(db, model, payload)
    return {"model": model_payload(model)}


@app.delete("/api/admin/models/{model_id}")
def delete_model(model_id: int, _: WorkspaceMember = Depends(require_manager), db: Session = Depends(get_db)):
    model = db.get(ModelConfig, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        delete_model_config(db, model)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True}


@app.get("/api/user-models")
def list_user_models(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    configs = list_user_model_configs(db, user_id=current_user.id)
    return {"items": [user_model_payload(config) for config in configs]}


@app.post("/api/user-models")
def create_user_model(request: UserModelConfigRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        config = create_user_model_config(db, user_id=current_user.id, payload=request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"model_config": user_model_payload(config)}


@app.post("/api/user-models/test")
def test_user_model_draft(request: UserModelCapabilityTestRequest, current_user: User = Depends(get_current_user)):
    payload = request.model_dump()
    try:
        detect_image = bool(payload.pop("detect_image", False))
        return test_user_model_payload(payload, detect_image=detect_image)
    except ValueError as exc:
        logger.warning(
            "User model draft test rejected: %s; user_id=%s; fields=%s",
            str(exc),
            current_user.id,
            _user_model_draft_log_context(payload),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _user_model_draft_log_context(payload: dict) -> dict:
    return {
        "display_name_present": bool(str(payload.get("display_name") or "").strip()),
        "provider": payload.get("provider"),
        "base_url_present": bool(str(payload.get("base_url") or "").strip()),
        "api_key_present": bool(str(payload.get("api_key") or "").strip()),
        "chat_model_present": bool(str(payload.get("chat_model") or "").strip()),
        "reasoning_type": payload.get("reasoning_type"),
        "detect_image": bool(payload.get("detect_image")),
    }


@app.patch("/api/user-models/{config_id}")
def patch_user_model(
    config_id: int,
    request: UserModelConfigUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    config = get_owned_user_model(db, user_id=current_user.id, config_id=config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Model config not found")
    try:
        config = update_user_model_config(db, config=config, payload=request.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"model_config": user_model_payload(config)}


@app.delete("/api/user-models/{config_id}")
def delete_user_model(config_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    config = get_owned_user_model(db, user_id=current_user.id, config_id=config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Model config not found")
    try:
        delete_user_model_config(db, config=config)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True}


@app.post("/api/user-models/{config_id}/test")
def test_user_model(
    config_id: int,
    detect_image: bool = Query(default=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    config = get_owned_user_model(db, user_id=current_user.id, config_id=config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Model config not found")
    return test_user_model_config(config, detect_image=detect_image)


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


@app.get("/api/agents/{agent_id}/memory-profile")
def get_agent_memory_profile(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    profile = get_memory_profile(
        db,
        workspace_id=membership.workspace_id,
        user_id=membership.user_id,
        agent_id=agent.id,
    )
    return {"profile": memory_profile_payload(profile, agent_id=agent.id)}


@app.patch("/api/agents/{agent_id}/memory-profile")
def patch_agent_memory_profile(
    agent_id: int,
    request: MemoryProfileUpdateRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    try:
        profile = upsert_memory_profile(
            db,
            workspace_id=membership.workspace_id,
            user_id=membership.user_id,
            agent_id=agent.id,
            payload=request.model_dump(exclude_unset=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"profile": memory_profile_payload(profile)}


@app.delete("/api/agents/{agent_id}/memory-profile")
def delete_agent_memory_profile(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    delete_memory_profile(
        db,
        workspace_id=membership.workspace_id,
        user_id=membership.user_id,
        agent_id=agent.id,
    )
    return {"deleted": True}


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
        db.query(Agent)
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


@app.get("/api/knowledge-bases")
def list_knowledge_bases(membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == membership.workspace_id).order_by(KnowledgeBase.id.desc()).all()
    items = []
    for kb in kbs:
        count = db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id).count()
        items.append(knowledge_base_summary(kb, count))
    return {"items": items}


@app.post("/api/knowledge-bases")
def create_kb(request: KnowledgeBaseCreateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = create_knowledge_base(db, workspace_id=membership.workspace_id, user_id=membership.user_id, name=request.name, description=request.description)
    return {"knowledge_base": knowledge_base_summary(kb)}


@app.post("/api/knowledge-bases/{kb_id}/documents")
def upload_document(kb_id: int, request: KnowledgeDocumentCreateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    logger.info(
        "Knowledge document upload request: schema=%s kb_id=%s workspace_id=%s filename=%s source_type=%s",
        request.__class__.__name__,
        kb.id,
        membership.workspace_id,
        request.filename or request.title or "",
        request.source_type,
    )
    try:
        document, payload = add_knowledge_document_from_request(db, workspace_id=membership.workspace_id, kb=kb, request=request)
    except KnowledgeDocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        logger.exception("Knowledge document indexing failed")
        raise HTTPException(status_code=502, detail={"message": _sanitize_public_error(str(exc)), "error_code": "knowledge_index_failed"}) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("Knowledge document upload failed")
        raise HTTPException(status_code=500, detail={"message": "Knowledge document upload failed.", "error_code": "knowledge_upload_failed"}) from exc
    if document.status == "failed":
        raise HTTPException(status_code=422, detail={"message": document.error_message or "Document text extraction failed", "document": payload})
    return {"document": payload}


@app.post("/api/knowledge-bases/{kb_id}/documents/batch")
def upload_documents_batch(kb_id: int, request: KnowledgeDocumentBatchCreateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    logger.info(
        "Knowledge document upload request: schema=%s kb_id=%s workspace_id=%s count=%s filenames=%s",
        request.__class__.__name__,
        kb.id,
        membership.workspace_id,
        len(request.documents),
        [item.filename or item.title or f"document-{index + 1}" for index, item in enumerate(request.documents)],
    )
    documents = []
    errors = []

    for index, item in enumerate(request.documents):
        filename = item.filename or item.title or f"document-{index + 1}"
        try:
            document, payload = add_knowledge_document_from_request(db, workspace_id=membership.workspace_id, kb=kb, request=item)
            if document.status == "failed":
                errors.append({
                    "index": index,
                    "filename": filename,
                    "message": document.error_message or "Document text extraction failed",
                    "document": payload,
                })
            else:
                documents.append(payload)
        except KnowledgeDocumentError as exc:
            db.rollback()
            errors.append({"index": index, "filename": filename, "message": str(exc), "status_code": exc.status_code})
        except RuntimeError as exc:
            db.rollback()
            logger.exception("Knowledge document batch indexing failed")
            errors.append({
                "index": index,
                "filename": filename,
                "message": _sanitize_public_error(str(exc)),
                "error_code": "knowledge_index_failed",
            })
        except Exception as exc:
            db.rollback()
            logger.exception("Knowledge document batch upload failed")
            errors.append({
                "index": index,
                "filename": filename,
                "message": "Knowledge document upload failed.",
                "error_code": "knowledge_upload_failed",
            })

    payload = {
        "documents": documents,
        "errors": errors,
        "total": len(request.documents),
        "succeeded": len(documents),
        "failed": len(errors),
    }
    return payload


@app.get("/api/knowledge-bases/{kb_id}/documents")
def list_documents(kb_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    documents = db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id).order_by(KnowledgeDocument.id.desc()).all()
    return {
        "items": [
            document_payload(
                document,
                db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document.id).count(),
            )
            for document in documents
        ]
    }


@app.delete("/api/knowledge-bases/{kb_id}/documents")
def clear_documents(kb_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    try:
        summary = clear_knowledge_base_documents(db, workspace_id=membership.workspace_id, kb=kb)
    except Exception as exc:
        db.rollback()
        logger.exception(
            "Knowledge base documents clear failed: kb_id=%s workspace_id=%s",
            kb.id,
            membership.workspace_id,
        )
        raise HTTPException(status_code=500, detail={"message": "Knowledge base documents clear failed.", "error_code": "knowledge_documents_clear_failed"}) from exc
    logger.info(
        "Knowledge base documents cleared: kb_id=%s workspace_id=%s documents_deleted=%s chunks_deleted=%s vectors_delete_requested=%s",
        kb.id,
        membership.workspace_id,
        summary.get("documents_deleted", 0),
        summary.get("chunks_deleted", 0),
        summary.get("vectors_delete_requested", False),
    )
    return {"cleared": True, **summary}


@app.delete("/api/knowledge-bases/{kb_id}/documents/{document_id}")
def remove_document(kb_id: int, document_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    document = db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id, KnowledgeDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        summary = delete_document(db, workspace_id=membership.workspace_id, document=document)
    except Exception as exc:
        db.rollback()
        logger.exception(
            "Knowledge document delete failed: kb_id=%s document_id=%s workspace_id=%s",
            kb.id,
            document_id,
            membership.workspace_id,
        )
        raise HTTPException(status_code=500, detail={"message": "Knowledge document delete failed.", "error_code": "knowledge_document_delete_failed"}) from exc
    logger.info(
        "Knowledge document deleted: kb_id=%s document_id=%s workspace_id=%s chunks_deleted=%s vectors_delete_requested=%s",
        kb.id,
        document_id,
        membership.workspace_id,
        summary.get("chunks_deleted", 0),
        summary.get("vectors_delete_requested", False),
    )
    return {"deleted": True, **summary}


@app.post("/api/knowledge-bases/{kb_id}/index")
def index_kb(kb_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    job_id = f"kb-{kb.id}-sync"
    summary = reindex_knowledge_base(db, workspace_id=membership.workspace_id, kb=kb)
    status = "failed" if summary["documents_failed"] and not summary["documents_indexed"] else "succeeded"
    payload = {
        "job_id": job_id,
        "knowledge_base_id": kb.id,
        "status": status,
        "message": (
            f"Rebuilt {summary['chunks_indexed']} chunks for {summary['documents_indexed']} documents."
            if status == "succeeded"
            else "Knowledge base reindex failed for all documents."
        ),
        **summary,
    }
    redis_store.set_job(job_id, payload)
    return payload


@app.get("/api/knowledge/jobs/{job_id}")
def get_knowledge_job(job_id: str, _: WorkspaceMember = Depends(get_current_membership)):
    lookup = redis_store.get_job(job_id)
    if lookup.hit and lookup.value:
        return lookup.value
    return {"job_id": job_id, "status": "unknown", "message": "Job state is not available or Redis is not configured."}


@app.delete("/api/knowledge-bases/{kb_id}")
def delete_kb(kb_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    delete_knowledge_base(db, workspace_id=membership.workspace_id, kb=kb)
    return {"deleted": True}


@app.patch("/api/knowledge-bases/{kb_id}")
def update_kb(kb_id: int, request: KnowledgeBaseCreateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    kb.name = request.name
    kb.description = request.description
    db.commit()
    db.refresh(kb)
    return {"knowledge_base": knowledge_base_summary(kb)}


@app.get("/api/knowledge-bases/{kb_id}/documents/{document_id}/chunks")
def get_document_chunks(kb_id: int, document_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    document = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == kb.id,
        KnowledgeDocument.id == document_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = list_document_chunks(db, document_id=document.id)
    return {"document": document_payload(document, len(chunks)), "chunks": chunks}


# Duplicate ChatRequest definition removed; model defined in api.schemas.

class ResegmentRequest(BaseModel):
    parse_mode: str = "precise"
    segment_mode: str = "auto"
    delimiter: str | None = "##"
    max_chunk_len: int = 5000
    overlap_pct: int = 10
    hierarchy_level: int = 3
    keep_hierarchy_info: bool = True

@app.post("/api/knowledge-bases/{kb_id}/documents/{document_id}/preview")
def preview_document_chunks(
    kb_id: int,
    document_id: int,
    request: ResegmentRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db)
):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    document = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == kb.id,
        KnowledgeDocument.id == document_id
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    cfg = request.model_dump()
    seg_mode = cfg.get("segment_mode", "auto")
    
    # 内存直接切分并不入库
    if seg_mode == "hierarchy":
        chunks = split_by_hierarchy(
            document.text,
            kb_id=kb.id,
            document_id=document.id,
            max_level=cfg.get("hierarchy_level", 3),
            keep_hierarchy_info=cfg.get("keep_hierarchy_info", True)
        )
    elif seg_mode == "custom":
        chunks = split_parent_child(
            document.text,
            kb_id=kb.id,
            document_id=document.id,
            parent_size=cfg.get("max_chunk_len", 1600),
            child_size=int(cfg.get("max_chunk_len", 1600) * 0.35),
            overlap=int(cfg.get("max_chunk_len", 1600) * cfg.get("overlap_pct", 10) / 100)
        )
    else:
        chunks = split_parent_child(document.text, kb_id=kb.id, document_id=document.id)
        
    return {
        "chunks_count": len(chunks),
        "preview_items": [
            {
                "chunk_index": idx,
                "text": chunk.get("text", ""),
                "hierarchy_path": chunk.get("section", "")
            }
            for idx, chunk in enumerate(chunks)
        ]
    }

@app.post("/api/knowledge-bases/{kb_id}/documents/{document_id}/resegment")
def resegment_document_chunks(
    kb_id: int,
    document_id: int,
    request: ResegmentRequest,
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db)
):
    kb = require_workspace_kb(db, membership.workspace_id, kb_id)
    require_kb_write_access(kb, membership)
    document = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == kb.id,
        KnowledgeDocument.id == document_id
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 保存配置并同步启动重新索引
    document.segment_config = request.model_dump()
    db.commit()
    
    try:
        chunk_count = index_document(
            db,
            workspace_id=membership.workspace_id,
            kb=kb,
            document=document,
            clear_existing=True
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail={"message": f"Resegment index failed: {str(exc)}"})
        
    return {"document": document_payload(document, chunk_count)}


@app.get("/api/agents/{agent_id}/workflow")
def get_workflow(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    workflow = db.query(WorkflowDefinition).filter(WorkflowDefinition.agent_id == agent.id).first()
    return {"nodes": workflow.nodes if workflow else default_workflow()}


@app.patch("/api/agents/{agent_id}/workflow")
def update_workflow(agent_id: int, request: WorkflowUpdateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_write_access(agent, membership)
    validate_workflow_nodes(request.nodes)
    workflow = db.query(WorkflowDefinition).filter(WorkflowDefinition.agent_id == agent.id).first()
    if not workflow:
        workflow = WorkflowDefinition(agent_id=agent.id, nodes=request.nodes)
        db.add(workflow)
    else:
        workflow.nodes = request.nodes
    db.commit()
    return {"nodes": workflow.nodes}


@app.post("/api/agents/{agent_id}/chat/stream")
def chat_stream(agent_id: int, request: ChatRequest, membership: WorkspaceMember = Depends(get_current_membership), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    print(f"[DEBUG CHAT] agent_id: {agent_id} | mode: {request.mode} | session_id: {request.session_id}")
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)

    session = get_or_create_session(db, agent, current_user.id, request.session_id, request.message, is_debug=getattr(request, "is_debug", False))
    user_message = Message(session_id=session.id, role="user", content=request.message, sources=[])
    db.add(user_message)
    db.commit()

    event_queue: queue.Queue = queue.Queue()

    # Capture everything the background thread needs (don't pass ORM objects across threads)
    bg_params = {
        "agent_id": agent.id,
        "agent_workspace_id": agent.workspace_id,
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

    def _run_workflow_bg() -> None:
        bg_db = SessionLocal()
        try:
            _execute_workflow_thread(bg_db, bg_params, event_queue)
        except Exception:
            logger.exception("Background workflow thread crashed")
        finally:
            event_queue.put(None)  # sentinel
            bg_db.close()

    thread = threading.Thread(target=_run_workflow_bg, name=f"wf-{session.id}", daemon=True)
    thread.start()

    def _sse_generator() -> Iterable[str]:
        while True:
            item = event_queue.get()
            if item is None:  # sentinel — workflow finished
                break
            yield item

    return StreamingResponse(_sse_generator(), media_type="text/event-stream")


@app.get("/api/agents/{agent_id}/sessions")
def list_agent_sessions(agent_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    agent = require_workspace_agent(db, membership.workspace_id, agent_id)
    require_agent_read_access(agent, membership)
    query = db.query(ChatSession).filter(
        ChatSession.workspace_id == membership.workspace_id,
        ChatSession.agent_id == agent.id,
        ChatSession.is_debug == False,
    )
    if not can_manage(membership.role):
        query = query.filter(ChatSession.user_id == membership.user_id)
    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    return {"items": [session_payload(session, db) for session in sessions]}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session or session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Session not found")
    require_session_access(session, membership)
    messages = db.query(Message).filter(Message.session_id == session.id).order_by(Message.id.asc()).all()
    active_run = (
        db.query(Run)
        .filter(Run.session_id == session.id, Run.status == "running")
        .order_by(Run.started_at.desc())
        .first()
    )
    active_run_payload = None
    if active_run:
        active_run_payload = {
            "id": active_run.id,
            "status": active_run.status,
            "started_at": active_run.started_at.isoformat() if active_run.started_at else None,
        }
    return {
        "session": session_payload(session, db),
        "messages": chat_message_payloads(messages),
        "active_run": active_run_payload,
    }


@app.patch("/api/sessions/{session_id}")
def patch_session(session_id: int, request: SessionUpdateRequest, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session or session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Session not found")
    require_session_access(session, membership)
    session.title = request.title.strip()
    db.commit()
    db.refresh(session)
    return {"session": session_payload(session, db)}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session or session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Session not found")
    require_session_access(session, membership)

    message_ids = [
        row.id
        for row in db.query(Message.id).filter(Message.session_id == session.id).all()
    ]
    run_ids = [
        row.id
        for row in db.query(Run.id).filter(Run.session_id == session.id).all()
    ]
    if message_ids:
        db.query(Feedback).filter(Feedback.message_id.in_(message_ids)).delete(synchronize_session=False)
    if run_ids:
        db.query(RunStep).filter(RunStep.run_id.in_(run_ids)).delete(synchronize_session=False)
    db.query(SessionMemory).filter(SessionMemory.session_id == session.id).delete(synchronize_session=False)
    db.query(Message).filter(Message.session_id == session.id).delete(synchronize_session=False)
    db.query(Run).filter(Run.session_id == session.id).delete(synchronize_session=False)
    db.delete(session)
    db.commit()
    return {"deleted": True}


def _execute_workflow_thread(db: Session, params: dict, q: queue.Queue) -> None:
    """Execute the workflow in a background thread with its own DB session.

    Puts SSE event strings onto *q*.  Puts ``None`` as a sentinel when done.
    All DB writes happen here, so the run lifecycle is independent of the
    SSE connection.
    """
    agent = db.get(Agent, params["agent_id"])
    if not agent:
        q.put(sse_event("error", {"message": "Agent not found"}))
        return
    chat_session = db.get(ChatSession, params["session_id"])
    if not chat_session:
        q.put(sse_event("error", {"message": "Session not found"}))
        return

    runner = WorkflowRunner(db)
    _tracked_run_id: int | None = None

    def _emit(event_name: str, data: dict | None = None) -> None:
        """Push an SSE event to the queue AND the replay log."""
        sse_str = sse_event(event_name, data or {})
        q.put(sse_str)
        if _tracked_run_id is not None:
            _append_run_event(_tracked_run_id, sse_str)
    answer = ""
    sources: list[dict] = []
    reasoning = ""
    reasoning_started_at: float | None = None
    reasoning_duration_ms: int | None = None
    run: Run | None = None
    assistant_saved = False
    used_tools = False
    requires_reasoning_replay = False

    try:
        for event in runner.run_events(
            agent=agent,
            chat_session=chat_session,
            user_message=params["user_message"],
            mode=params["mode"],
            variables=params["variables"],
            rag_enabled=params["rag_enabled"],
            rag_options=params["rag_options"],
            thinking_enabled=params["thinking_enabled"],
            search_enabled=params["search_enabled"],
            attachments=params["attachments"],
            current_message_id=params["user_message_id"],
        ):
            if event["event"] == "run_started":
                _tracked_run_id = event["run_id"]
                _emit("run_started", {"run_id": _tracked_run_id})
            elif event["event"] == "token":
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                answer += event.get("content", "")
                _emit("token", {"content": event.get("content", "")})
            elif event["event"] == "reasoning_token":
                content = event.get("content", "")
                if reasoning_started_at is None:
                    reasoning_started_at = time.perf_counter()
                reasoning += content
                _emit("reasoning_token", {"content": content})
            elif event["event"] in {
                "tool_call_start",
                "tool_call_result",
                "tool_call",
                "search_status",
                "rag_status",
                "memory_used",
                "thinking_status",
            }:
                _emit(event["event"], event.get("data", {}) or {})
            elif event["event"] == "step":
                step = event["step"]
                for runtime_event in step.get("events", []):
                    runtime_event_name = runtime_event.get("event", "tool_call")
                    runtime_event_data = runtime_event.get("data", {}) or {}
                    if runtime_event_name == "reasoning_token":
                        content = runtime_event_data.get("content", "")
                        if content:
                            if reasoning_started_at is None:
                                reasoning_started_at = time.perf_counter()
                            reasoning += content
                        _emit("reasoning_token", {"content": content})
                        continue
                    _emit(runtime_event_name, runtime_event_data)
                _emit("run_step", step)
            elif event["event"] == "cancelled":
                run = db.get(Run, event["run_id"])
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                assistant = Message(
                    session_id=chat_session.id,
                    role="assistant",
                    content=answer,
                    reasoning=reasoning,
                    reasoning_duration_ms=reasoning_duration_ms,
                    sources=sources,
                    meta={"cancelled": True},
                )
                db.add(assistant)
                db.commit()
                db.refresh(assistant)
                assistant_saved = True
                _emit("cancelled", {
                    "session_id": chat_session.id,
                    "message_id": assistant.id,
                    "run_id": event["run_id"],
                    "content": answer,
                })
                return
            elif event["event"] == "complete":
                run = event["run"]
                answer = event["answer"]
                sources = event["sources"]
                used_tools = any(
                    (step.get("node_type") == "Tool")
                    and int(((step.get("output") or {}).get("tool_stats") or {}).get("total_calls") or 0) > 0
                    for step in event.get("steps", [])
                )
                requires_reasoning_replay = any(
                    bool((step.get("output") or {}).get("reasoning_replay_required"))
                    for step in event.get("steps", [])
                )

        if reasoning_started_at is not None and reasoning_duration_ms is None:
            reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
        if sources:
            _emit("sources", {"items": sources})
        assistant = Message(
            session_id=chat_session.id,
            role="assistant",
            content=answer,
            reasoning=reasoning,
            reasoning_duration_ms=reasoning_duration_ms,
            sources=sources,
            meta={
                **({"used_tools": True} if used_tools else {}),
                **({"reasoning_includes_intermediate": True} if used_tools and reasoning else {}),
                **({"requires_reasoning_replay": True} if used_tools and requires_reasoning_replay else {}),
            },
        )
        db.add(assistant)
        db.commit()
        db.refresh(assistant)
        assistant_saved = True
        _emit("done", {
            "session_id": chat_session.id,
            "message_id": assistant.id,
            "run_id": run.id,
            "content": answer,
            "reasoning_duration_ms": reasoning_duration_ms,
        })
    except _CancelledError:
        # Workflow was cancelled — run status already set by run_events()
        if not assistant_saved and answer:
            try:
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                assistant = Message(
                    session_id=chat_session.id,
                    role="assistant",
                    content=answer,
                    reasoning=reasoning,
                    reasoning_duration_ms=reasoning_duration_ms,
                    sources=sources,
                    meta={"cancelled": True, "partial": True},
                )
                db.add(assistant)
                db.commit()
            except Exception:
                db.rollback()
        _emit("cancelled", {
            "session_id": chat_session.id,
            "run_id": run.id if run else None,
            "content": answer,
        })
    except Exception as exc:
        if run is not None:
            try:
                run.status = "failed"
                run.completed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception:
                db.rollback()
        logger.exception("Agent chat stream failed")
        _emit("error", safe_stream_error(exc))
        # Safety net: persist partial answer
        if not assistant_saved and answer:
            try:
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                assistant = Message(
                    session_id=chat_session.id,
                    role="assistant",
                    content=answer,
                    reasoning=reasoning,
                    reasoning_duration_ms=reasoning_duration_ms,
                    sources=sources,
                    meta={"cancelled": True, "partial": True},
                )
                db.add(assistant)
                db.commit()
            except Exception:
                db.rollback()
    finally:
        if _tracked_run_id is not None:
            _cleanup_run_events(_tracked_run_id)


@app.get("/api/runs/{run_id}")
def get_run(run_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if not run or run.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Run not found")
    session = db.get(ChatSession, run.session_id)
    if session:
        require_session_access(session, membership)
    return {"run": {"id": run.id, "status": run.status, "agent_id": run.agent_id, "session_id": run.session_id}}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run_endpoint(run_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if not run or run.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Run not found")
    session = db.get(ChatSession, run.session_id)
    if session:
        require_session_access(session, membership)
    if cancel_run(run_id):
        return {"cancelled": True, "run_id": run_id}
    return {"cancelled": False, "run_id": run_id, "message": "Run not active or already completed"}


@app.get("/api/runs/{run_id}/events")
def stream_run_events(run_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    """Reconnect to an in-progress run's SSE event stream after page refresh.

    Replays buffered events then streams new events as they arrive.
    If the run is already finished, returns buffered events and closes.
    """
    run = db.get(Run, run_id)
    if not run or run.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Run not found")
    session = db.get(ChatSession, run.session_id)
    if session:
        require_session_access(session, membership)

    def _event_stream() -> Iterable[str]:
        emitted = 0
        # Phase 1: replay buffered events
        while True:
            events, total = _get_run_events_since(run_id, emitted)
            for sse_str in events:
                yield sse_str
                emitted += 1
            # Check if run is still active
            db_sess = SessionLocal()
            try:
                current = db_sess.get(Run, run_id)
                if current is None or current.status != "running":
                    break
            finally:
                db_sess.close()
            time.sleep(0.3)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/steps")
def get_run_steps(run_id: int, membership: WorkspaceMember = Depends(get_current_membership), db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if not run or run.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Run not found")
    session = db.get(ChatSession, run.session_id)
    if session:
        require_session_access(session, membership)
    steps = db.query(RunStep).filter(RunStep.run_id == run.id).order_by(RunStep.id.asc()).all()
    return {"items": [{"id": step.id, "node_id": step.node_id, "node_type": step.node_type, "status": step.status, "output": step.output} for step in steps]}


@app.post("/api/messages/{message_id}/feedback")
def create_feedback(
    message_id: int,
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    membership: WorkspaceMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
):
    message = db.get(Message, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    session = db.get(ChatSession, message.session_id)
    if not session or session.workspace_id != membership.workspace_id:
        raise HTTPException(status_code=404, detail="Message not found")
    require_session_access(session, membership)
    feedback = Feedback(message_id=message.id, user_id=current_user.id, rating=request.rating, comment=request.comment)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"feedback": {"id": feedback.id, "rating": feedback.rating, "comment": feedback.comment}}


def user_payload(user: User) -> dict:
    return {"id": user.id, "email": user.email, "name": user.name, "avatar_url": user.avatar_url or ""}


def workspace_payload(workspace, role: str) -> dict:
    return {"id": workspace.id, "name": workspace.name, "slug": workspace.slug, "role": normalize_role(role)}


def membership_payload(membership: WorkspaceMember) -> dict:
    return {"workspace_id": membership.workspace_id, "user_id": membership.user_id, "role": normalize_role(membership.role)}


def invite_payload(invite: WorkspaceInvite, *, include_token: bool = False) -> dict:
    payload = {"id": invite.id, "email": invite.email, "role": normalize_role(invite.role), "accepted_at": invite.accepted_at.isoformat() if invite.accepted_at else None}
    if include_token:
        payload["token"] = invite.token
    return payload


def session_payload(session: ChatSession, db: Session) -> dict:
    messages = db.query(Message).filter(Message.session_id == session.id).all()
    count = len([message for message in messages if visible_chat_message(message)])
    return {
        "id": session.id,
        "agent_id": session.agent_id,
        "title": session.title,
        "message_count": count,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


def chat_message_payloads(messages: list[Message]) -> list[dict]:
    payloads: list[dict] = []
    pending_reasoning: list[str] = []
    pending_timeline: list[dict] = []
    for message in messages:
        meta = message.meta or {}
        if message.role == "user":
            pending_reasoning = []
            pending_timeline = []
            if visible_chat_message(message):
                payloads.append(message_payload(message))
            continue
        if message.role == "assistant" and meta.get("is_intermediate"):
            if message.reasoning:
                pending_reasoning.append(message.reasoning)
                pending_timeline.append(
                    {
                        "id": f"stored-reasoning-{message.id}",
                        "type": "reasoning",
                        "content": message.reasoning,
                    }
                )
            continue
        if message.role == "tool":
            item = tool_message_timeline_item(message)
            if item:
                pending_timeline.append(item)
            continue
        if not visible_chat_message(message):
            continue
        payload = message_payload(message)
        if message.role == "assistant":
            payload_meta = payload.get("meta") or {}
            if pending_reasoning and not payload_meta.get("reasoning_includes_intermediate"):
                payload["reasoning"] = merge_reasoning_parts([*pending_reasoning, payload.get("reasoning") or ""])
            timeline = [*pending_timeline]
            final_reasoning = remaining_reasoning(payload.get("reasoning") or message.reasoning or "", pending_reasoning)
            if final_reasoning:
                timeline.append(
                    {
                        "id": f"stored-final-reasoning-{message.id}",
                        "type": "reasoning",
                        "content": final_reasoning,
                    }
                )
            if timeline:
                payload["reasoningTimeline"] = timeline
            pending_reasoning = []
            pending_timeline = []
        payloads.append(payload)
    return payloads


def merge_reasoning_parts(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def remaining_reasoning(full_reasoning: str, consumed_parts: list[str]) -> str:
    remaining = full_reasoning or ""
    for part in consumed_parts:
        candidates = [part or "", (part or "").strip()]
        for candidate in candidates:
            if not candidate:
                continue
            index = remaining.find(candidate)
            if index >= 0:
                remaining = remaining[index + len(candidate) :]
                break
    return remaining.strip()


def tool_message_timeline_item(message: Message) -> dict | None:
    meta = message.meta or {}
    if not meta.get("is_intermediate"):
        return None
    tool_name = str(meta.get("tool_name") or meta.get("tool") or message.tool_name or message.tool_call_id or "tool")
    tool_type = str(meta.get("tool_type") or "tool")
    status = str(meta.get("status") or ("error" if meta.get("error_code") else "success"))
    is_search = tool_type == "builtin_search" or tool_name == "web_search"
    raw_input = str(meta.get("input_preview") or "")
    raw_result = str(meta.get("error") or message.content or meta.get("result_preview") or "")
    input_summary = summarize_tool_input(raw_input)
    return {
        "id": f"stored-tool-{message.id}",
        "type": "search" if is_search else "tool",
        "status": status,
        "toolCallId": message.tool_call_id or str(meta.get("tool_call_id") or ""),
        "title": "调用联网搜索" if is_search else f"调用 {tool_name}",
        "meta": " · ".join(part for part in [tool_type, tool_status_label(status)] if part),
        "latency": timeline_latency(meta.get("latency_ms")),
        "inputLabel": input_summary["label"],
        "inputPreview": input_summary["text"],
        "summary": summarize_tool_result(raw_result, status=status),
        "rawInput": raw_input,
        "rawResult": raw_result,
    }


def tool_status_label(status: str) -> str:
    if status == "error":
        return "失败"
    if status == "running":
        return "运行中"
    return "完成"


def timeline_latency(value) -> str:
    try:
        ms = float(value or 0)
    except (TypeError, ValueError):
        return ""
    if ms <= 0:
        return ""
    if ms < 1000:
        return f"{round(ms)}ms"
    seconds = ms / 1000
    return f"{seconds:.1f}s" if ms < 10000 else f"{seconds:.0f}s"


def compact_timeline_text(value, limit: int = 220) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    compact = " ".join(str(text).split())
    return f"{compact[:limit]}..." if len(compact) > limit else compact


def summarize_tool_input(raw_input: str) -> dict:
    data = parse_json_preview(raw_input)
    if isinstance(data, dict):
        query = first_string_value(data, ["command", "query", "q", "keyword", "keywords", "search", "text", "input"])
        count = first_scalar_value(data, ["count", "limit", "top_k", "max_results", "num_results"])
        if query:
            is_command = "command" in data
            label = "命令" if is_command else "查询"
            suffix_parts = []
            if count:
                suffix_parts.append(f"数量 {count}")
            if is_command:
                cwd_val = str(data.get("cwd", "")).strip()
                if cwd_val:
                    suffix_parts.append(f"目录: {cwd_val}")
            suffix = " · " + " · ".join(suffix_parts) if suffix_parts else ""
            return {"label": label, "text": compact_timeline_text(f"{query}{suffix}", 120)}
        keys = [key for key, value in data.items() if value not in (None, "")]
        if keys:
            if len(keys) <= 3:
                preview = " · ".join(f"{key}: {compact_param_value(data.get(key))}" for key in keys)
            else:
                preview = f"已传入 {len(keys)} 个参数：" + "、".join(keys[:3])
            return {"label": "参数", "text": compact_timeline_text(preview, 140)}
    text = "已传入结构化参数。" if looks_like_json_text(raw_input) else compact_timeline_text(raw_input or "", 120)
    return {"label": "参数", "text": text}


def summarize_tool_result(raw_result: str, *, status: str = "success") -> str:
    if status == "running":
        return ""
    if status == "error":
        return f"调用失败：{compact_timeline_text(raw_result, 140)}" if raw_result else "调用失败。"
    data = parse_json_preview(raw_result)
    # PowerShell / command execution result
    if isinstance(data, dict) and "exit_code" in data and "command" in data:
        exit_code = data.get("exit_code", -1)
        is_timeout = bool(data.get("timeout"))
        stdout = str(data.get("stdout") or "")
        stderr = str(data.get("stderr") or "")
        duration_ms = data.get("duration_ms", 0)
        truncated = bool(data.get("truncated"))
        parts = []
        if is_timeout:
            parts.append("[timeout] 命令执行超时")
        elif exit_code == 0:
            parts.append("[OK] 命令执行成功")
        else:
            parts.append(f"[exit={exit_code}] 命令执行失败")
        duration_str = ""
        try:
            ms = int(duration_ms or 0)
            if ms >= 1000:
                duration_str = f"{ms / 1000:.1f}s"
            elif ms > 0:
                duration_str = f"{ms}ms"
        except (TypeError, ValueError):
            pass
        if duration_str:
            parts.append(f"耗时 {duration_str}")
        if truncated:
            parts.append("输出已截断")
        if stderr and not is_timeout:
            parts.append(f"stderr: {compact_timeline_text(stderr, 80)}")
        if stdout:
            preview = stdout.strip()[:120]
            parts.append(compact_timeline_text(preview, 120))
        return " · ".join(parts)
    items = collect_result_items(data)
    if items:
        return summarize_result_items(items, raw_result)
    fallback_count = count_json_field(raw_result, "snippet") or count_json_field(raw_result, "title")
    if fallback_count:
        dates = extract_date_signals(raw_result)
        parts = [f"搜索到约 {fallback_count} 条结果"]
        if dates:
            parts.append(f"结果中出现 {'、'.join(dates[:3])} 等日期")
        return compact_timeline_text("；".join(parts), 180)
    if isinstance(data, dict):
        error = first_string_value(data, ["error", "message", "detail"])
        if error:
            return compact_timeline_text(error, 160)
        keys = [key for key in data.keys() if key]
        if keys:
            return f"工具返回 {len(keys)} 个字段：" + "、".join(keys[:4])
    if isinstance(data, list):
        return f"工具返回 {len(data)} 条结构化结果。"
    if looks_like_json_text(raw_result):
        return "工具返回了结构化结果，原始内容可展开查看。"
    return compact_timeline_text(raw_result or "", 160)


def summarize_result_items(items: list, raw_result: str) -> str:
    names = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        name = item.get("title") or item.get("name") or item.get("hostname") or item.get("source") or item.get("url")
        if name:
            names.append(str(name))
    dates = extract_date_signals(raw_result or json.dumps(items, ensure_ascii=False))
    parts = [f"搜索到 {len(items)} 条结果"]
    if names:
        parts.append("包括 " + "、".join(names))
    if dates:
        parts.append(f"结果中出现 {'、'.join(dates[:3])} 等日期")
    return compact_timeline_text("；".join(parts), 180)


def parse_json_preview(value):
    if not value or not isinstance(value, str):
        return value or None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def looks_like_json_text(value: str) -> bool:
    text = str(value or "").strip()
    return text.startswith("{") or text.startswith("[")


def collect_result_items(data) -> list:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ["pages", "items", "results", "data", "documents"]:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def first_string_value(data: dict, keys: list[str]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def first_scalar_value(data: dict, keys: list[str]):
    for key in keys:
        value = data.get(key)
        if isinstance(value, (str, int, float)):
            return value
    return ""


def compact_param_value(value) -> str:
    if isinstance(value, (str, int, float, bool)):
        return compact_timeline_text(str(value), 50)
    if isinstance(value, list):
        return f"{len(value)} 项"
    if isinstance(value, dict):
        return "对象"
    return ""


def count_json_field(value: str, field_name: str) -> int:
    if not value:
        return 0
    return len(re.findall(rf'"{re.escape(field_name)}"\s*:', value))


def extract_date_signals(value: str) -> list[str]:
    if not value:
        return []
    matches = re.findall(r"20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?|20\d{2}年\d{1,2}月?", value)
    return list(dict.fromkeys(matches))[:5]


def message_payload(message: Message) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "reasoning": message.reasoning or "",
        "reasoningDurationMs": message.reasoning_duration_ms,
        "sources": message.sources or [],
        "toolCalls": message.tool_calls or [],
        "toolCallId": message.tool_call_id or "",
        "toolName": message.tool_name or "",
        "meta": message.meta or {},
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def visible_chat_message(message: Message) -> bool:
    if message.role == "tool":
        return False
    return not bool((message.meta or {}).get("is_intermediate"))


def invite_workspace(db: Session, workspace_id: int):
    from core.db.models import Workspace

    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


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


def require_session_access(session: ChatSession, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if session.user_id == membership.user_id:
        return
    raise HTTPException(status_code=404, detail="Session not found")


def review_payload(db: Session, agent: Agent) -> dict:
    version = db.query(AgentVersion).filter(AgentVersion.agent_id == agent.id).order_by(AgentVersion.version.desc()).first()
    return {
        **agent_summary(agent),
        "submitted_version": version.version if version else None,
        "submitted_at": version.created_at.isoformat() if version and version.created_at else None,
    }


def require_workspace_kb(db: Session, workspace_id: int, kb_id: int) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == workspace_id, KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def require_workspace_skill(db: Session, workspace_id: int, skill_id: int) -> Skill:
    skill = db.query(Skill).filter(Skill.workspace_id == workspace_id, Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


def require_kb_write_access(kb: KnowledgeBase, membership: WorkspaceMember) -> None:
    if can_manage(membership.role):
        return
    if kb.created_by == membership.user_id:
        return
    raise HTTPException(status_code=403, detail="Knowledge base edit denied")


def validate_workflow_nodes(nodes: list[dict]) -> None:
    allowed = {"Start", "LLM", "Knowledge", "Tool", "Answer"}
    seen = {node.get("type") for node in nodes}
    if not seen.issubset(allowed):
        raise HTTPException(status_code=400, detail="Unsupported workflow node type")
    if not {"Start", "Answer"}.issubset(seen):
        raise HTTPException(status_code=400, detail="Workflow requires Start and Answer nodes")


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
        title=title_seed[:60] or "新对话",
        is_debug=is_debug,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def chunk_text(text: str, size: int = 28) -> Iterable[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]
