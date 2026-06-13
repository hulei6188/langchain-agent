from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
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


@dataclass
class RunStreamState:
    answer: str = ""
    provisional_answer: str = ""
    sources: list[dict] = field(default_factory=list)
    reasoning: str = ""
    reasoning_started_at: float | None = None
    reasoning_duration_ms: int | None = None
    assistant_saved: bool = False
    used_tools: bool = False
    requires_reasoning_replay: bool = False

    @property
    def partial_answer(self) -> str:
        return self.answer + self.provisional_answer

    def finish_reasoning_duration(self) -> None:
        if self.reasoning_started_at is not None and self.reasoning_duration_ms is None:
            self.reasoning_duration_ms = int((time.perf_counter() - self.reasoning_started_at) * 1000)

    def add_token(self, content: str) -> None:
        self.finish_reasoning_duration()
        self.answer += content

    def add_provisional_token(self, content: str) -> None:
        self.finish_reasoning_duration()
        self.provisional_answer += content

    def add_reasoning_token(self, content: str) -> None:
        if content and self.reasoning_started_at is None:
            self.reasoning_started_at = time.perf_counter()
        self.reasoning += content

    def clear_provisional(self) -> None:
        self.provisional_answer = ""
        if not self.reasoning:
            self.reasoning_started_at = None
            self.reasoning_duration_ms = None

    def commit_provisional(self) -> None:
        if self.provisional_answer:
            self.answer += self.provisional_answer
        self.provisional_answer = ""

    def apply_complete(self, event: dict) -> None:
        self.answer = event["answer"] or self.partial_answer
        self.sources = event["sources"]
        steps = event.get("steps", [])
        self.used_tools = any(
            (step.get("node_type") == "Tool")
            and int(((step.get("output") or {}).get("tool_stats") or {}).get("total_calls") or 0) > 0
            for step in steps
        )
        self.requires_reasoning_replay = any(
            bool((step.get("output") or {}).get("reasoning_replay_required"))
            for step in steps
        )


def stream_workflow_sse(params: dict) -> Iterable[str]:
    """Stream a workflow run directly from LangGraph runtime events as SSE."""
    db = SessionLocal()
    try:
        yield from execute_workflow_stream(db, params)
    except Exception as exc:
        logger.exception("Workflow stream crashed")
        _mark_latest_running_run_failed(db, session_id=params["session_id"])
        yield sse_event("error", safe_stream_error(exc))
    finally:
        db.close()


def execute_workflow_stream(db: Session, params: dict) -> Iterable[str]:
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
            append_run_event(db, run_id=tracked_run_id, event=event_name, payload=payload, sse=sse_str)
        except Exception:
            db.rollback()
            logger.exception("Failed to persist run event %s for run %s", event_name, tracked_run_id)
        return sse_str

    stream_state = RunStreamState()
    run: Run | None = None
    workflow_stream = None

    try:
        workflow_stream = runner.start_stream_run(
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
        )
        tracked_run_id = workflow_stream.run.id
        run = workflow_stream.run
        yield emit("run_started", {"run_id": tracked_run_id, "session_id": chat_session.id})

        final_state = None
        for part in runner.stream_graph_parts(workflow_stream):
            if part["type"] == "values":
                final_state = part["data"]
                continue
            if part["type"] != "custom":
                continue
            event = part["data"]
            event_name = event["event"]
            if event_name == "token":
                content = event.get("content", "")
                stream_state.add_token(content)
                yield emit("token", {"content": content})
            elif event_name == "provisional_token":
                content = event.get("content", "")
                stream_state.add_provisional_token(content)
                yield emit("provisional_token", {"content": content})
            elif event_name == "reasoning_token":
                content = event.get("content", "")
                stream_state.add_reasoning_token(content)
                yield emit("reasoning_token", {"content": content})
            elif event_name == "provisional_clear":
                stream_state.clear_provisional()
                yield emit("provisional_clear", event.get("data", {}) or {})
            elif event_name == "provisional_commit":
                stream_state.commit_provisional()
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
                        stream_state.add_reasoning_token(content)
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
        stream_state.apply_complete(
            {
                "answer": final_answer,
                "sources": sources,
                "steps": steps,
            }
        )

        stream_state.finish_reasoning_duration()
        if stream_state.sources:
            yield emit("sources", {"items": stream_state.sources})
        assistant = Message(
            session_id=chat_session.id,
            role="assistant",
            content=stream_state.answer,
            reasoning=stream_state.reasoning,
            reasoning_duration_ms=stream_state.reasoning_duration_ms,
            sources=stream_state.sources,
            meta={
                **({"used_tools": True} if stream_state.used_tools else {}),
                **({"reasoning_includes_intermediate": True} if stream_state.used_tools and stream_state.reasoning else {}),
                **({"requires_reasoning_replay": True} if stream_state.used_tools and stream_state.requires_reasoning_replay else {}),
            },
        )
        db.add(assistant)
        db.commit()
        db.refresh(assistant)
        stream_state.assistant_saved = True
        title_runtime_config = _resolve_title_runtime_config(db, agent, params["user_id"])
        _auto_title_session(db, chat_session, params["user_message"], stream_state.answer, runtime_config=title_runtime_config)
        yield emit(
            "done",
            {
                "session_id": chat_session.id,
                "message_id": assistant.id,
                "run_id": run.id if run else tracked_run_id,
                "content": stream_state.answer,
                "reasoning_duration_ms": stream_state.reasoning_duration_ms,
                "title": chat_session.title,
            },
        )
    except _CancelledError:
        if run is not None:
            runner.mark_stream_run_cancelled(run)
        partial_answer = stream_state.partial_answer
        partial_reasoning = stream_state.reasoning
        if not stream_state.assistant_saved and partial_answer:
            _persist_partial_assistant(
                db,
                chat_session=chat_session,
                content=partial_answer,
                reasoning=partial_reasoning,
                reasoning_started_at=stream_state.reasoning_started_at,
                reasoning_duration_ms=stream_state.reasoning_duration_ms,
                sources=stream_state.sources,
            )
        yield emit(
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
        yield emit("error", safe_stream_error(exc))
        partial_answer = stream_state.partial_answer
        partial_reasoning = stream_state.reasoning
        if not stream_state.assistant_saved and partial_answer:
            _persist_partial_assistant(
                db,
                chat_session=chat_session,
                content=partial_answer,
                reasoning=partial_reasoning,
                reasoning_started_at=stream_state.reasoning_started_at,
                reasoning_duration_ms=stream_state.reasoning_duration_ms,
                sources=stream_state.sources,
            )
    finally:
        if run is not None:
            runner.close_stream_run(run)


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
