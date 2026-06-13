from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage
from sqlalchemy.orm import Session

from core.db.models import (
    Agent,
    AgentKnowledgeBase,
    Run,
    Session as ChatSession,
    WorkflowDefinition,
)
from core.integrations.llm import OpenAICompatibleProvider, _CancelledError
from core.runtime.agent_runtime import build_runtime_agent, validate_model_capabilities
from core.runtime.cancel import register_run, unregister_run
from core.runtime.dsml import (
    strip_or_block_leaked_tool_markup,
)
from core.runtime.graph_runtime import WorkflowGraphState, build_langgraph_workflow, workflow_thread_config
from core.runtime.memory_runtime import load_runtime_memory_state, save_runtime_memory_state
from core.runtime.message_utils import (
    message_content_text as _message_content_text,
    message_reasoning_content as _message_reasoning_content,
)
from core.runtime.persistence import session_history
from core.runtime.prompting import build_llm_messages, llm_output, merge_variables
from core.runtime.skill_runtime import apply_runtime_skills
from core.runtime.status import (
    search_status,
    thinking_status,
)
from core.runtime.streaming import stream_llm_response
from core.runtime.tool_loop import ToolLoopRunner
from core.services.agents import normalize_memory, normalize_rag
from core.services.rag import run_rag_pipeline
from core.services.uploads import get_workspace_uploads

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStreamRun:
    runtime: Any
    run: Run
    context: dict
    graph: Any


class WorkflowRunner:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.provider = OpenAICompatibleProvider()
        self._cancel_event: threading.Event | None = None

    def set_cancel_event(self, event: threading.Event) -> None:
        """Set the cancel event for this runner."""
        self._cancel_event = event

    def _raise_if_cancelled(self) -> None:
        """Raise _CancelledError if a cancel has been requested for this run."""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise _CancelledError()

    def run(
        self,
        *,
        agent: Agent,
        chat_session: ChatSession,
        user_message: str,
        mode: str = "draft",
        variables: dict | None = None,
        rag_enabled: bool | None = None,
        rag_options: dict | None = None,
        thinking_enabled: bool | None = None,
        search_enabled: bool | None = None,
        attachments: list[dict] | None = None,
    ) -> tuple[Run, str, list[dict], list[dict]]:
        runtime, run, context = self._start_run(
            agent=agent,
            chat_session=chat_session,
            user_message=user_message,
            mode=mode,
            variables=variables,
            rag_enabled=rag_enabled,
            rag_options=rag_options,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            attachments=attachments,
            current_message_id=None,
        )
        graph = self._build_langgraph_workflow(
            runtime=runtime,
            run=run,
            user_message=user_message,
            stream=False,
        )
        final_state = graph.invoke({"context": context, "steps": []}, config=workflow_thread_config(context))
        context = final_state["context"]
        steps = final_state["steps"]

        final_answer, sources = self._complete_run_success(
            run=run,
            chat_session=chat_session,
            runtime=runtime,
            user_message=user_message,
            context=context,
        )
        return run, final_answer, sources, steps

    def start_stream_run(
        self,
        *,
        agent: Agent,
        chat_session: ChatSession,
        user_message: str,
        mode: str = "draft",
        variables: dict | None = None,
        rag_enabled: bool | None = None,
        rag_options: dict | None = None,
        thinking_enabled: bool | None = None,
        search_enabled: bool | None = None,
        attachments: list[dict] | None = None,
        current_message_id: int | None = None,
    ) -> WorkflowStreamRun:
        runtime, run, context = self._start_run(
            agent=agent,
            chat_session=chat_session,
            user_message=user_message,
            mode=mode,
            variables=variables,
            rag_enabled=rag_enabled,
            rag_options=rag_options,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            attachments=attachments,
            current_message_id=current_message_id,
        )
        self._cancel_event = register_run(run.id, self.provider)
        graph = self._build_langgraph_workflow(
            runtime=runtime,
            run=run,
            user_message=user_message,
            stream=True,
        )
        return WorkflowStreamRun(runtime=runtime, run=run, context=context, graph=graph)

    def stream_graph_parts(self, stream_run: WorkflowStreamRun):
        for part in stream_run.graph.stream(
            {"context": stream_run.context, "steps": []},
            config=workflow_thread_config(stream_run.context),
            stream_mode=["custom", "values"],
            version="v2",
        ):
            self._raise_if_cancelled()
            yield part

    def complete_stream_run(
        self,
        *,
        stream_run: WorkflowStreamRun,
        chat_session: ChatSession,
        user_message: str,
        context: dict,
    ) -> tuple[str, list[dict]]:
        return self._complete_run_success(
            run=stream_run.run,
            chat_session=chat_session,
            runtime=stream_run.runtime,
            user_message=user_message,
            context=context,
        )

    def mark_stream_run_cancelled(self, run: Run) -> None:
        run.status = "cancelled"
        run.completed_at = datetime.utcnow()
        self.db.commit()

    def close_stream_run(self, run: Run) -> None:
        unregister_run(run.id)

    def _start_run(
        self,
        *,
        agent: Agent,
        chat_session: ChatSession,
        user_message: str,
        mode: str,
        variables: dict | None,
        rag_enabled: bool | None,
        rag_options: dict | None,
        thinking_enabled: bool | None,
        search_enabled: bool | None,
        attachments: list[dict] | None,
        current_message_id: int | None = None,
    ) -> tuple[object, Run, dict]:
        runtime = build_runtime_agent(self.db, agent, mode, chat_session.user_id)
        upload_ids = [str(item.get("id")) for item in attachments or [] if item.get("id")]
        uploads = get_workspace_uploads(self.db, workspace_id=agent.workspace_id, upload_ids=upload_ids)
        validate_model_capabilities(runtime.capability_config, uploads)
        thinking_status_value = thinking_status(runtime.capability_config, thinking_enabled)
        search_status_value = search_status(user_message, search_enabled)

        rag_config = normalize_rag({**dict(runtime.settings.get("rag") or {}), **dict(rag_options or {})})
        memory_config = normalize_memory(runtime.settings.get("memory"))
        effective_rag_enabled = rag_config["enabled_by_default"] if rag_enabled is None else bool(rag_enabled)

        run = Run(workspace_id=agent.workspace_id, agent_id=agent.id, session_id=chat_session.id, status="running")
        self.db.add(run)
        self.db.flush()
        self.db.commit()
        self.db.refresh(run)

        memory_state = load_runtime_memory_state(
            self.db,
            workspace_id=agent.workspace_id,
            user_id=chat_session.user_id,
            agent_id=agent.id,
            session_id=chat_session.id,
            memory_config=memory_config,
        )
        context: dict = {
            "session_id": chat_session.id,
            "run_id": run.id,
            "input": user_message,
            "sources": [],
            "tool_outputs": [],
            "draft": "",
            "history_messages": session_history(self.db, chat_session.id, max_messages=int(memory_config.get("max_messages") or 12)),
            "current_message_id": current_message_id,
            "variables": merge_variables(runtime.settings.get("variables", []), variables or {}),
            "agent_workdir": runtime.settings.get("workdir"),
            **memory_state,
            "rag_enabled": effective_rag_enabled,
            **({"rag_enabled_request": rag_enabled} if rag_enabled is not None else {}),
            "rag_top_k": rag_config["top_k"],
            "rag_config": rag_config,
            "thinking_enabled": thinking_status_value["enabled"],
            "thinking_status": thinking_status_value,
            "reasoning_replay_required": self.provider.requires_reasoning_replay(model=runtime.model, runtime_config=runtime.runtime_config),
            "search_enabled": search_status_value["enabled"],
            "search_status": search_status_value,
            "web_sources": search_status_value.get("sources", []),
            "uploads": uploads,
        }
        apply_runtime_skills(runtime, context, chat_session)
        return runtime, run, context

    def _complete_run_success(
        self,
        *,
        run: Run,
        chat_session: ChatSession,
        runtime,
        user_message: str,
        context: dict,
    ) -> tuple[str, list[dict]]:
        final_answer = strip_or_block_leaked_tool_markup(context.get("answer") or context.get("draft") or "当前智能体没有生成回答。")
        save_runtime_memory_state(
            self.db,
            session_id=chat_session.id,
            user_message=user_message,
            answer=final_answer,
            max_messages=int(runtime.settings.get("memory", {}).get("max_messages", 12)),
            enabled=bool(context.get("memory_enabled")),
        )
        run.status = "succeeded"
        run.completed_at = datetime.utcnow()
        self.db.commit()
        return final_answer, [*context.get("sources", []), *context.get("web_sources", [])]

    def _build_langgraph_workflow(
        self,
        *,
        runtime,
        run: Run,
        user_message: str,
        stream: bool,
    ):
        return build_langgraph_workflow(
            runtime=runtime,
            run_id=run.id,
            user_message=user_message,
            stream=stream,
            execute_node=self._execute_node,
            stream_llm_node=self._stream_llm_node,
            run_tool_node=self._run_tool_node,
            raise_if_cancelled=self._raise_if_cancelled,
        )

    def _execute_node(self, agent, node: dict, context: dict) -> dict:
        handlers = {
            "Start": self._execute_start_node,
            "Knowledge": self._execute_knowledge_node,
            "Tool": lambda runtime, current_node, current_context: self._run_tool_node(runtime, current_node, current_context, stream=False),
            "LLM": self._execute_llm_node,
            "Answer": self._execute_answer_node,
        }
        handler = handlers.get(node["type"])
        return handler(agent, node, context) if handler else {}

    def _execute_start_node(self, agent, node: dict, context: dict) -> dict:
        return {
            "started": True,
            "variables": context.get("variables", {}),
            "rag_enabled": context.get("rag_enabled", True),
            "search_enabled": context.get("search_enabled", False),
            "attachment_count": len(context.get("uploads", [])),
            "loaded_skills": context.get("loaded_skills", []),
        }

    def _execute_knowledge_node(self, agent, node: dict, context: dict) -> dict:
        effective_source = "request" if "rag_enabled_request" in context else "agent_default"
        if not context.get("rag_enabled", True):
            status = {
                "enabled": False,
                "effective_source": effective_source,
                "knowledge_base_ids": [],
                "query": context["input"],
                "top_k": int(context.get("rag_top_k") or node.get("config", {}).get("top_k", 4)),
                "matched_chunks": 0,
                "sources_emitted": False,
                "reason": "disabled",
                "dense": {"matched": 0},
                "bm25": {"matched": 0},
                "rrf": {"matched": 0},
                "rerank": {"enabled": False, "applied": False, "model": None, "error": None},
                "cache": {"enabled": False, "hit": False, "backend": "none"},
                "no_evidence": False,
            }
            return {"sources": [], "rag_enabled": False, "rag_status": status, "events": [{"event": "rag_status", "data": status}]}

        kb_ids = getattr(agent, "knowledge_base_ids", None)
        if kb_ids is None:
            kb_ids = [
                row.knowledge_base_id
                for row in self.db.query(AgentKnowledgeBase).filter(AgentKnowledgeBase.agent_id == agent.id).all()
            ]
        rag_result = run_rag_pipeline(
            self.db,
            workspace_id=agent.workspace_id,
            knowledge_base_ids=kb_ids,
            query=context["input"],
            config=context.get("rag_config") or {},
            runtime_config=getattr(agent, "runtime_config", None),
        )
        sources = rag_result.sources
        status = {**rag_result.status, "effective_source": effective_source}
        return {"sources": sources, "rag_enabled": True, "rag_status": status, "events": [{"event": "rag_status", "data": status}]}

    def _execute_llm_node(self, agent, node: dict, context: dict) -> dict:
        self._raise_if_cancelled()
        if context.get("draft"):
            return llm_output(agent, context, context["draft"], reasoning=context.get("draft_reasoning") or "", last_chat_mock=self.provider.last_chat_mock)
        messages = build_llm_messages(agent, context)
        response = self._invoke_chat_model(agent, messages, context)
        draft = _message_content_text(response)
        draft = strip_or_block_leaked_tool_markup(draft)
        return llm_output(agent, context, draft, last_chat_mock=self.provider.last_chat_mock)

    def _execute_answer_node(self, agent, node: dict, context: dict) -> dict:
        answer = strip_or_block_leaked_tool_markup((context.get("draft") or "").strip()).strip()
        if not answer:
            raise ValueError("Model returned an empty answer")
        return {"answer": answer, "citation_count": len([*context.get("sources", []), *context.get("web_sources", [])])}

    def _invoke_chat_model(
        self,
        agent,
        messages: list[BaseMessage],
        context: dict,
        *,
        tools: list[Any] | None = None,
    ) -> AIMessage:
        response = self.provider.invoke(
            messages,
            model=agent.model,
            temperature=agent.temperature,
            runtime_config=agent.runtime_config,
            tools=tools,
            thinking_enabled=self._thinking_request_value(context),
            cancel_event=self._cancel_event,
        )
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=_message_content_text(response), additional_kwargs=getattr(response, "additional_kwargs", {}))

    def _run_tool_node(
        self,
        agent,
        node: dict,
        context: dict,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None = None,
    ) -> dict:
        return ToolLoopRunner(
            self.db,
            self.provider,
            cancel_event=self._cancel_event,
            raise_if_cancelled=self._raise_if_cancelled,
        ).run(agent, node, context, stream=stream, writer=writer)

    def _stream_tool_node(self, agent, node: dict, context: dict):
        buffered_events: list[dict] = []
        output = self._run_tool_node(
            agent,
            node,
            context,
            stream=True,
            writer=buffered_events.append,
        )
        for event in buffered_events:
            yield event
        return output

    def _stream_llm_node(self, agent, node: dict, context: dict):
        draft = strip_or_block_leaked_tool_markup(context.get("draft", ""))
        if draft:
            draft_reasoning = context.get("draft_reasoning") or ""
            if draft_reasoning and context.get("thinking_enabled") and not context.get("draft_reasoning_streamed"):
                yield {"event": "reasoning_token", "content": draft_reasoning}
            if not context.get("draft_streamed"):
                # Draft already produced by the tool-calling loop — stream as chunked text
                for index in range(0, len(draft), 24):
                    self._raise_if_cancelled()
                    yield {"event": "token", "content": draft[index : index + 24]}
            return llm_output(agent, context, draft, reasoning=draft_reasoning, last_chat_mock=self.provider.last_chat_mock)
        messages = build_llm_messages(agent, context)
        response = yield from stream_llm_response(
            self.provider,
            agent=agent,
            messages=messages,
            context=context,
            thinking_enabled=self._thinking_request_value(context),
            cancel_event=self._cancel_event,
            raise_if_cancelled=self._raise_if_cancelled,
        )
        return llm_output(agent, context, _message_content_text(response), reasoning=_message_reasoning_content(response), last_chat_mock=self.provider.last_chat_mock)

    def _thinking_request_value(self, context: dict) -> bool | None:
        status = context.get("thinking_status") or {}
        if status.get("type") not in {"native", "prompt"}:
            return None
        return bool(status.get("enabled"))

