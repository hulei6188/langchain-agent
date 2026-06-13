from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Iterable
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from core.db.models import Agent, Message, Run, Session as ChatSession
from core.db.session import SessionLocal
from core.integrations.llm import OpenAICompatibleProvider, _CancelledError
from core.runtime.workflow import WorkflowRunner
from core.services.run_events import append_run_event, sse_event
from core.services.user_models import resolve_user_model_config, user_model_runtime_config

logger = logging.getLogger(__name__)

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


def start_workflow_stream(params: dict) -> queue.Queue:
    """Start a workflow run on a background DB session and return its SSE queue."""
    event_queue: queue.Queue = queue.Queue()

    def _run_workflow_bg() -> None:
        bg_db = SessionLocal()
        try:
            execute_workflow_stream(bg_db, params, event_queue)
        except Exception:
            logger.exception("Background workflow thread crashed")
            _mark_latest_running_run_failed(bg_db, session_id=params["session_id"])
        finally:
            event_queue.put(None)
            bg_db.close()

    thread = threading.Thread(target=_run_workflow_bg, name=f"wf-{params['session_id']}", daemon=True)
    thread.start()
    return event_queue


def workflow_sse_items(event_queue: queue.Queue) -> Iterable[str]:
    while True:
        item = event_queue.get()
        if item is None:
            break
        yield item


def execute_workflow_stream(db: Session, params: dict, q: queue.Queue) -> None:
    """Execute a workflow run and write SSE event strings into *q*."""
    agent = db.get(Agent, params["agent_id"])
    if not agent:
        q.put(sse_event("error", {"message": "Agent not found"}))
        return
    chat_session = db.get(ChatSession, params["session_id"])
    if not chat_session:
        q.put(sse_event("error", {"message": "Session not found"}))
        return

    runner = WorkflowRunner(db)
    tracked_run_id: int | None = None

    def emit(event_name: str, data: dict | None = None) -> None:
        payload = data or {}
        sse_str = sse_event(event_name, payload)
        q.put(sse_str)
        if tracked_run_id is None:
            return
        try:
            append_run_event(db, run_id=tracked_run_id, event=event_name, payload=payload, sse=sse_str)
        except Exception:
            db.rollback()
            logger.exception("Failed to persist run event %s for run %s", event_name, tracked_run_id)

    answer = ""
    provisional_answer = ""
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
            event_name = event["event"]
            if event_name == "run_started":
                tracked_run_id = event["run_id"]
                run = db.get(Run, tracked_run_id)
                emit("run_started", {"run_id": tracked_run_id, "session_id": chat_session.id})
            elif event_name == "token":
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                answer += event.get("content", "")
                emit("token", {"content": event.get("content", "")})
            elif event_name == "provisional_token":
                content = event.get("content", "")
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                provisional_answer += content
                emit("provisional_token", {"content": content})
            elif event_name == "reasoning_token":
                content = event.get("content", "")
                if reasoning_started_at is None:
                    reasoning_started_at = time.perf_counter()
                reasoning += content
                emit("reasoning_token", {"content": content})
            elif event_name == "provisional_clear":
                provisional_answer = ""
                if not reasoning:
                    reasoning_started_at = None
                    reasoning_duration_ms = None
                emit("provisional_clear", event.get("data", {}) or {})
            elif event_name == "provisional_commit":
                if provisional_answer:
                    answer += provisional_answer
                provisional_answer = ""
                emit("provisional_commit", event.get("data", {}) or {})
            elif event_name in {
                "tool_call_start",
                "tool_call_result",
                "tool_call",
                "search_status",
                "rag_status",
                "memory_used",
                "thinking_status",
            }:
                emit(event_name, event.get("data", {}) or {})
            elif event_name == "step":
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
                        emit("reasoning_token", {"content": content})
                        continue
                    emit(runtime_event_name, runtime_event_data)
                emit("run_step", step)
            elif event_name == "cancelled":
                run = db.get(Run, event["run_id"])
                if reasoning_started_at is not None and reasoning_duration_ms is None:
                    reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
                cancel_answer = answer + provisional_answer
                cancel_reasoning = reasoning
                assistant = Message(
                    session_id=chat_session.id,
                    role="assistant",
                    content=cancel_answer,
                    reasoning=cancel_reasoning,
                    reasoning_duration_ms=reasoning_duration_ms,
                    sources=sources,
                    meta={"cancelled": True},
                )
                db.add(assistant)
                db.commit()
                db.refresh(assistant)
                assistant_saved = True
                emit(
                    "cancelled",
                    {
                        "session_id": chat_session.id,
                        "message_id": assistant.id,
                        "run_id": event["run_id"],
                        "content": cancel_answer,
                    },
                )
                return
            elif event_name == "complete":
                run = event["run"]
                answer = event["answer"] or answer + provisional_answer
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
            emit("sources", {"items": sources})
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
        title_runtime_config = _resolve_title_runtime_config(db, agent, params["user_id"])
        _auto_title_session(db, chat_session, params["user_message"], answer, runtime_config=title_runtime_config)
        emit(
            "done",
            {
                "session_id": chat_session.id,
                "message_id": assistant.id,
                "run_id": run.id if run else tracked_run_id,
                "content": answer,
                "reasoning_duration_ms": reasoning_duration_ms,
                "title": chat_session.title,
            },
        )
    except _CancelledError:
        partial_answer = answer + provisional_answer
        partial_reasoning = reasoning
        if not assistant_saved and partial_answer:
            _persist_partial_assistant(
                db,
                chat_session=chat_session,
                content=partial_answer,
                reasoning=partial_reasoning,
                reasoning_started_at=reasoning_started_at,
                reasoning_duration_ms=reasoning_duration_ms,
                sources=sources,
            )
        emit(
            "cancelled",
            {
                "session_id": chat_session.id,
                "run_id": run.id if run else (tracked_run_id or None),
                "content": partial_answer,
            },
        )
    except Exception as exc:
        resolved_run = run
        if resolved_run is None and tracked_run_id is not None:
            resolved_run = db.get(Run, tracked_run_id)
        if resolved_run is not None and resolved_run.status == "running":
            try:
                resolved_run.status = "failed"
                resolved_run.completed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("Failed to mark run %s as failed", getattr(resolved_run, "id", None))
        logger.exception("Agent chat stream failed")
        emit("error", safe_stream_error(exc))
        partial_answer = answer + provisional_answer
        partial_reasoning = reasoning
        if not assistant_saved and partial_answer:
            _persist_partial_assistant(
                db,
                chat_session=chat_session,
                content=partial_answer,
                reasoning=partial_reasoning,
                reasoning_started_at=reasoning_started_at,
                reasoning_duration_ms=reasoning_duration_ms,
                sources=sources,
            )


def safe_stream_error(exc: Exception) -> dict:
    message = str(exc)
    if any(public_error in message for public_error in PUBLIC_CHAT_ERRORS):
        return {"message": sanitize_public_error(message), "error_code": error_code(message)}
    return {
        "message": "智能体运行失败，请检查模型、知识库或附件配置后重试。",
        "error_code": error_code(message),
    }


def sanitize_public_error(message: str) -> str:
    cleaned = re.sub(r"(?i)(sk-[A-Za-z0-9_-]+|api[_-]?key\s*[:=]\s*\S+|secret\s*[:=]\s*\S+)", "[secret]", str(message))
    return cleaned.replace("\n", " ").replace("\r", " ").strip()[:500]


def error_code(message: str) -> str:
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


def _mark_latest_running_run_failed(db: Session, *, session_id: int) -> None:
    try:
        zombie = (
            db.query(Run)
            .filter(
                Run.session_id == session_id,
                Run.status == "running",
            )
            .order_by(Run.started_at.desc())
            .first()
        )
        if zombie is not None:
            zombie.status = "failed"
            zombie.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("Marked zombie run %s as failed after thread crash", zombie.id)
    except Exception:
        db.rollback()
        logger.exception("Failed to mark zombie run after thread crash")


def _persist_partial_assistant(
    db: Session,
    *,
    chat_session: ChatSession,
    content: str,
    reasoning: str,
    reasoning_started_at: float | None,
    reasoning_duration_ms: int | None,
    sources: list[dict],
) -> None:
    try:
        if reasoning_started_at is not None and reasoning_duration_ms is None:
            reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)
        assistant = Message(
            session_id=chat_session.id,
            role="assistant",
            content=content,
            reasoning=reasoning,
            reasoning_duration_ms=reasoning_duration_ms,
            sources=sources,
            meta={"cancelled": True, "partial": True},
        )
        db.add(assistant)
        db.commit()
    except Exception:
        db.rollback()


def _resolve_title_runtime_config(db: Session, agent: Agent, user_id: int) -> dict | None:
    try:
        user_model_config = resolve_user_model_config(db, user_id=user_id, config_id=agent.user_model_config_id)
        if user_model_config:
            return user_model_runtime_config(user_model_config)
    except Exception:
        logger.exception("Failed to resolve runtime config for auto-title, using defaults")
    return None


def _auto_title_session(
    db: Session,
    chat_session: ChatSession,
    user_message: str,
    answer: str,
    runtime_config: dict | None = None,
) -> None:
    if chat_session.title and chat_session.title not in {"新会话", "新对话"}:
        return
    try:
        provider = OpenAICompatibleProvider()
        title_prompt = (
            "你是一个会话标题生成助手。请根据用户和助手之间的对话内容，生成一个简洁的会话标题。\n"
            "要求：\n"
            "- 标题长度不超过20个字\n"
            "- 只返回标题本身，不要加引号或其他说明\n"
            "- 标题应反映用户问题的核心主题\n\n"
            f"用户：{user_message[:200]}\n"
            f"助手：{answer[:300]}"
        )
        response = provider.invoke(
            [HumanMessage(content=title_prompt)],
            temperature=0.3,
            runtime_config=runtime_config,
        )
        title = (response.content or "").strip()
        title = title.replace('"', "").replace("「", "").replace("」", "").replace("\n", " ").strip()[:60]
        if title:
            chat_session.title = title
            db.commit()
            logger.info("Auto-titled session %d to: %s", chat_session.id, title)
    except Exception:
        logger.exception("Failed to auto-title session %d", chat_session.id)
