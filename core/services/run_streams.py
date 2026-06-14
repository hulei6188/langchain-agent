from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
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


async def stream_workflow_sse(params: dict) -> AsyncIterator[str]:
    """Stream a workflow run directly from LangGraph runtime events as SSE."""
    db = SessionLocal()
    try:
        async for event in execute_workflow_stream(db, params):
            yield event
    except Exception as exc:
        logger.exception("Workflow stream crashed")
        _mark_latest_running_run_failed(db, session_id=params["session_id"])
        yield sse_event("error", safe_stream_error(exc))
    finally:
        db.close()


async def execute_workflow_stream(db: Session, params: dict) -> AsyncIterator[str]:
    """Execute a workflow run and yield SSE event strings."""
    agent = db.get(Agent, params["agent_id"])
    if not agent:
        yield sse_event("error", {"message": "Agent not found"})
        return
    chat_session = db.get(ChatSession, params["session_id"])
    if not chat_session:
        yield sse_event("error", {"message": "Session not found"})
        return

    runner = WorkflowRunner(db)
    tracked_run_id: int | None = None

    def emit(event_name: str, data: dict | None = None) -> str:
        payload = data or {}
        sse_str = sse_event(event_name, payload)
        if tracked_run_id is None:
            return sse_str
        try:
            _append_run_event_isolated(run_id=tracked_run_id, event=event_name, payload=payload, sse=sse_str)
        except Exception:
            logger.exception("Failed to persist run event %s for run %s", event_name, tracked_run_id)
        return sse_str

    answer_parts: list[str] = []
    provisional_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_started_at: float | None = None
    reasoning_duration_ms: int | None = None
    assistant_saved = False
    sources: list[dict] = []
    used_tools = False
    requires_reasoning_replay = False
    run: Run | None = None
    workflow_stream = None

    def partial_answer() -> str:
        return "".join(answer_parts) + "".join(provisional_parts)

    def finish_reasoning_duration() -> None:
        nonlocal reasoning_duration_ms
        if reasoning_started_at is not None and reasoning_duration_ms is None:
            reasoning_duration_ms = int((time.perf_counter() - reasoning_started_at) * 1000)

    def add_reasoning(content: str) -> None:
        nonlocal reasoning_started_at
        if content and reasoning_started_at is None:
            reasoning_started_at = time.perf_counter()
        reasoning_parts.append(content)

    try:
        stream_run_kwargs = {
            "agent": agent,
            "chat_session": chat_session,
            "user_message": params["user_message"],
            "mode": params["mode"],
            "variables": params["variables"],
            "rag_enabled": params["rag_enabled"],
            "rag_options": params["rag_options"],
            "thinking_enabled": params["thinking_enabled"],
            "search_enabled": params["search_enabled"],
            "attachments": params["attachments"],
            "current_message_id": params["user_message_id"],
        }
        if hasattr(runner, "astart_stream_run"):
            workflow_stream = await runner.astart_stream_run(**stream_run_kwargs)
        else:
            workflow_stream = runner.start_stream_run(**stream_run_kwargs)
        tracked_run_id = workflow_stream.run.id
        run = workflow_stream.run
        yield emit("run_started", {"run_id": tracked_run_id, "session_id": chat_session.id})

        final_state = None
        async for graph_event in runner.astream_graph_events(workflow_stream):
            graph_chunk = langgraph_stream_chunk(graph_event)
            if graph_chunk is None:
                continue
            stream_mode, stream_payload = graph_chunk
            if stream_mode == "values":
                final_state = stream_payload
                continue
            if stream_mode != "custom":
                continue
            event = stream_payload
            event_name = event["event"]
            if event_name == "token":
                content = event.get("content", "")
                finish_reasoning_duration()
                answer_parts.append(content)
                yield emit("token", {"content": content})
            elif event_name == "provisional_token":
                content = event.get("content", "")
                finish_reasoning_duration()
                provisional_parts.append(content)
                yield emit("provisional_token", {"content": content})
            elif event_name == "reasoning_token":
                content = event.get("content", "")
                add_reasoning(content)
                yield emit("reasoning_token", {"content": content})
            elif event_name == "provisional_clear":
                provisional_parts.clear()
                if not reasoning_parts:
                    reasoning_started_at = None
                    reasoning_duration_ms = None
                yield emit("provisional_clear", event.get("data", {}) or {})
            elif event_name == "provisional_commit":
                if provisional_parts:
                    answer_parts.extend(provisional_parts)
                    provisional_parts.clear()
                yield emit("provisional_commit", event.get("data", {}) or {})
            elif event_name in {
                "tool_call_start",
                "tool_call_result",
                "tool_call",
                "search_status",
                "rag_status",
                "memory_used",
                "thinking_status",
            }:
                yield emit(event_name, event.get("data", {}) or {})
            elif event_name == "step":
                step = event["step"]
                for runtime_event in step.get("events", []):
                    runtime_event_name = runtime_event.get("event", "tool_call")
                    runtime_event_data = runtime_event.get("data", {}) or {}
                    if runtime_event_name == "reasoning_token":
                        content = runtime_event_data.get("content", "")
                        add_reasoning(content)
                        yield emit("reasoning_token", {"content": content})
                        continue
                    yield emit(runtime_event_name, runtime_event_data)
                yield emit("run_step", step)

        context = final_state["context"] if final_state is not None else workflow_stream.context
        steps = final_state["steps"] if final_state is not None else []
        final_answer, sources = runner.complete_stream_run(
            stream_run=workflow_stream,
            chat_session=chat_session,
            user_message=params["user_message"],
            context=context,
        )
        answer = final_answer or partial_answer()
        answer_parts[:] = [answer]
        provisional_parts.clear()
        used_tools = any(
            (step.get("node_type") == "Tool")
            and int(((step.get("output") or {}).get("tool_stats") or {}).get("total_calls") or 0) > 0
            for step in steps
        )
        requires_reasoning_replay = any(
            bool((step.get("output") or {}).get("reasoning_replay_required"))
            for step in steps
        )

        finish_reasoning_duration()
        if sources:
            yield emit("sources", {"items": sources})
        assistant = Message(
            session_id=chat_session.id,
            role="assistant",
            content=answer,
            reasoning="".join(reasoning_parts),
            reasoning_duration_ms=reasoning_duration_ms,
            sources=sources,
            meta={
                **({"used_tools": True} if used_tools else {}),
                **({"reasoning_includes_intermediate": True} if used_tools and reasoning_parts else {}),
                **({"requires_reasoning_replay": True} if used_tools and requires_reasoning_replay else {}),
            },
        )
        db.add(assistant)
        db.commit()
        db.refresh(assistant)
        assistant_saved = True
        title_runtime_config = _resolve_title_runtime_config(db, agent, params["user_id"])
        _auto_title_session(db, chat_session, params["user_message"], answer, runtime_config=title_runtime_config)
        yield emit(
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
        if run is not None:
            runner.mark_stream_run_cancelled(run)
        partial = partial_answer()
        partial_reasoning = "".join(reasoning_parts)
        finish_reasoning_duration()
        partial_assistant: Message | None = None
        if not assistant_saved and (partial or partial_reasoning):
            partial_assistant = _persist_partial_assistant(
                db,
                chat_session=chat_session,
                content=partial,
                reasoning=partial_reasoning,
                reasoning_started_at=reasoning_started_at,
                reasoning_duration_ms=reasoning_duration_ms,
                sources=sources,
            )
        yield emit(
            "cancelled",
            {
                "session_id": chat_session.id,
                "run_id": run.id if run else (tracked_run_id or None),
                "message_id": partial_assistant.id if partial_assistant is not None else None,
                "content": partial,
                "reasoning_duration_ms": (
                    partial_assistant.reasoning_duration_ms
                    if partial_assistant is not None
                    else reasoning_duration_ms
                ),
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
        yield emit("error", safe_stream_error(exc))
        partial = partial_answer()
        partial_reasoning = "".join(reasoning_parts)
        if not assistant_saved and partial:
            _persist_partial_assistant(
                db,
                chat_session=chat_session,
                content=partial,
                reasoning=partial_reasoning,
                reasoning_started_at=reasoning_started_at,
                reasoning_duration_ms=reasoning_duration_ms,
                sources=sources,
            )
    finally:
        if run is not None:
            runner.close_stream_run(run)


def langgraph_stream_chunk(graph_event: dict):
    if graph_event.get("event") != "on_chain_stream" or graph_event.get("name") != "LangGraph":
        return None
    chunk = (graph_event.get("data") or {}).get("chunk")
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return chunk
    if isinstance(chunk, list) and len(chunk) == 2:
        return tuple(chunk)
    return None


def _append_run_event_isolated(*, run_id: int, event: str, payload: dict, sse: str) -> None:
    event_db = SessionLocal()
    try:
        append_run_event(event_db, run_id=run_id, event=event, payload=payload, sse=sse)
    except Exception:
        event_db.rollback()
        raise
    finally:
        event_db.close()


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
) -> Message | None:
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
        db.refresh(assistant)
        return assistant
    except Exception:
        db.rollback()
        return None


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
