from __future__ import annotations

from datetime import datetime
import json
import logging
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from core.db.models import (
    Agent,
    AgentKnowledgeBase,
    AgentSkill,
    AgentTool,
    AgentVersion,
    Message,
    ModelConfig,
    Run,
    Session as ChatSession,
    Skill,
    SkillKnowledgeBase,
    SkillTool,
    Tool,
    UserModelConfig,
    Upload,
    WorkflowDefinition,
)
from core.integrations.llm import OpenAICompatibleProvider, _CancelledError
from core.runtime.cancel import register_run, unregister_run, is_cancelled
from core.runtime.dsml import (
    DSML_TOOL_MARKUP_ERROR,
    buffer_stream_content,
    contains_dsml_tool_calls,
    contains_leaked_tool_markup,
    dsml_preview,
    dsml_tool_call_parser,
    dsml_tool_names,
    parse_dsml_tool_calls,
    strip_complete_dsml_tool_call_blocks,
    strip_or_block_leaked_tool_markup,
)
from core.runtime.message_utils import (
    message_content_text as _message_content_text,
    message_reasoning_content as _message_reasoning_content,
    normalize_langchain_tool_calls as _normalize_langchain_tool_calls,
)
from core.runtime.spec import WorkflowGraphSpec, default_workflow, workflow_graph_spec
from core.runtime.skill_selection import (
    dedupe_skill_bindings,
    loaded_skill_text,
    score_runtime_skills,
    skill_explicitly_requested,
    skill_manifest,
    skill_manifest_text,
    skill_selection_text,
)
from core.runtime.status import (
    merge_web_search_tool_result,
    search_status,
    search_status_event,
    thinking_messages,
    thinking_status,
    web_source_text,
)
from core.runtime.tool_calls import finalize_stream_tool_calls, merge_stream_tool_call_chunks
from core.services.agents import get_agent_detail, normalize_memory, normalize_rag, normalize_tool_policy, normalize_workdir
from core.services.rag import run_rag_pipeline
from core.services.memory import load_graph_memory_context, update_session_memory
from core.services.models import resolve_agent_model
from core.services.skills import normalize_activation_mode
from core.services.tools import build_langchain_tool, tool_call_event
from core.services.uploads import get_workspace_uploads
from core.services.user_models import (
    resolve_user_model_config,
    user_model_runtime_config,
)


MAX_TOOL_CALLS_PER_RUN = 200
MAX_TOOL_ROUNDS_PER_RUN = 50
MAX_TOOL_WALL_TIME_SECONDS = 1800
SKILL_AUTO_TOP_K = 3
SKILL_AUTO_THRESHOLD = 0.25
SKILL_SELECTION_HISTORY_MESSAGES = 8

logger = logging.getLogger(__name__)


class WorkflowGraphState(TypedDict):
    context: dict[str, Any]
    steps: list[dict[str, Any]]


class ToolGraphState(TypedDict, total=False):
    context: dict[str, Any]
    node: dict[str, Any]
    messages: list[BaseMessage]
    bound_tools: list[Tool]
    langchain_tools: list[Any]
    allowed_names: set[str]
    total_calls: int
    tools_used: list[str]
    events: list[dict[str, Any]]
    web_sources: list[dict[str, Any]]
    search_status: dict[str, Any]
    round_index: int
    tool_loop_start: float
    max_tool_calls: int
    max_tool_wall_time: int
    pending_calls: list[dict[str, Any]]
    latest_response: AIMessage | None
    loaded_skill_this_round: bool
    output: dict[str, Any]


class ToolNodeInvokeState(TypedDict):
    messages: list[BaseMessage]


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
        runtime = self._runtime_agent(agent, mode, chat_session.user_id)
        upload_ids = [str(item.get("id")) for item in attachments or [] if item.get("id")]
        uploads = get_workspace_uploads(self.db, workspace_id=agent.workspace_id, upload_ids=upload_ids)
        self._validate_model_capabilities(runtime.capability_config, uploads)
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

        memory_context = load_graph_memory_context(
            self.db,
            workspace_id=agent.workspace_id,
            user_id=chat_session.user_id,
            agent_id=agent.id,
            session_id=chat_session.id,
        )
        context: dict = {
            "session_id": chat_session.id,
            "run_id": run.id,
            "input": user_message,
            "sources": [],
            "tool_outputs": [],
            "draft": "",
            "history_messages": self._session_history(chat_session.id, max_messages=int(memory_config.get("max_messages") or 12)),
            "current_message_id": None,
            "variables": self._merge_variables(runtime.settings.get("variables", []), variables or {}),
            "agent_workdir": runtime.settings.get("workdir"),
            "memory_summary": memory_context.session_summary,
            "profile_memory": memory_context.profile_text,
            "profile_memory_used": memory_context.profile_event,
            "memory_enabled": memory_config.get("enabled", False),
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
        self._apply_runtime_skills(runtime, context, chat_session)
        graph = self._build_langgraph_workflow(
            runtime=runtime,
            run=run,
            user_message=user_message,
            stream=False,
        )
        final_state = graph.invoke({"context": context, "steps": []})
        context = final_state["context"]
        steps = final_state["steps"]

        final_answer = strip_or_block_leaked_tool_markup(context.get("answer") or context.get("draft") or "当前智能体没有生成回答。")
        if context.get("memory_enabled"):
            update_session_memory(
                self.db,
                session_id=chat_session.id,
                user_message=user_message,
                answer=final_answer,
                max_messages=int(runtime.settings.get("memory", {}).get("max_messages", 12)),
            )
        run.status = "succeeded"
        run.completed_at = datetime.utcnow()
        self.db.commit()
        return run, final_answer, [*context.get("sources", []), *context.get("web_sources", [])], steps

    def run_events(
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
    ):
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
        # Emit run_id immediately so frontend can cancel
        yield {"event": "run_started", "run_id": run.id}
        # Register for cancellation
        self._cancel_event = register_run(run.id, self.provider)

        steps: list[dict] = []
        try:
            graph = self._build_langgraph_workflow(
                runtime=runtime,
                run=run,
                user_message=user_message,
                stream=True,
            )
            final_state: WorkflowGraphState | None = None
            for part in graph.stream(
                {"context": context, "steps": []},
                stream_mode=["custom", "values"],
                version="v2",
            ):
                self._raise_if_cancelled()
                if part["type"] == "custom":
                    yield part["data"]
                    continue
                if part["type"] == "values":
                    final_state = part["data"]

            if final_state is not None:
                context = final_state["context"]
                steps = final_state["steps"]

            final_answer = strip_or_block_leaked_tool_markup(context.get("answer") or context.get("draft") or "当前智能体没有生成回答。")
            if context.get("memory_enabled"):
                update_session_memory(
                    self.db,
                    session_id=chat_session.id,
                    user_message=user_message,
                    answer=final_answer,
                    max_messages=int(runtime.settings.get("memory", {}).get("max_messages", 12)),
                )
            run.status = "succeeded"
            run.completed_at = datetime.utcnow()
            self.db.commit()
            yield {
                "event": "complete",
                "run": run,
                "answer": final_answer,
                "sources": [*context.get("sources", []), *context.get("web_sources", [])],
                "steps": steps,
            }
        except _CancelledError:
            run.status = "cancelled"
            run.completed_at = datetime.utcnow()
            self.db.commit()
            yield {"event": "cancelled", "run_id": run.id}
        finally:
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
        runtime = self._runtime_agent(agent, mode, chat_session.user_id)
        upload_ids = [str(item.get("id")) for item in attachments or [] if item.get("id")]
        uploads = get_workspace_uploads(self.db, workspace_id=agent.workspace_id, upload_ids=upload_ids)
        self._validate_model_capabilities(runtime.capability_config, uploads)
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

        memory_context = load_graph_memory_context(
            self.db,
            workspace_id=agent.workspace_id,
            user_id=chat_session.user_id,
            agent_id=agent.id,
            session_id=chat_session.id,
        )
        context: dict = {
            "session_id": chat_session.id,
            "run_id": run.id,
            "input": user_message,
            "sources": [],
            "tool_outputs": [],
            "draft": "",
            "history_messages": self._session_history(chat_session.id, max_messages=int(memory_config.get("max_messages") or 12)),
            "current_message_id": current_message_id,
            "variables": self._merge_variables(runtime.settings.get("variables", []), variables or {}),
            "agent_workdir": runtime.settings.get("workdir"),
            "memory_summary": memory_context.session_summary,
            "profile_memory": memory_context.profile_text,
            "profile_memory_used": memory_context.profile_event,
            "memory_enabled": memory_config.get("enabled", False),
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
        self._apply_runtime_skills(runtime, context, chat_session)
        return runtime, run, context

    def _build_langgraph_workflow(
        self,
        *,
        runtime,
        run: Run,
        user_message: str,
        stream: bool,
    ):
        spec = workflow_graph_spec(getattr(runtime, "workflow", None))
        workflow = spec["nodes"]
        graph_builder = StateGraph(WorkflowGraphState)
        used_names: set[str] = set()
        graph_names: dict[str, str] = {}

        for index, node in enumerate(workflow):
            graph_node_name = self._langgraph_node_name(node, index, used_names)
            node_id = str(node.get("id") or f"node_{index}")
            graph_names[node_id] = graph_node_name
            graph_builder.add_node(
                graph_node_name,
                self._langgraph_node(runtime, run, user_message, node, stream=stream),
            )

        edge_sources = {edge["source"] for edge in spec["edges"]}
        edge_targets = {edge["target"] for edge in spec["edges"]}
        start_node_ids = [node_id for node_id in graph_names if node_id not in edge_targets] or [next(iter(graph_names))]
        for node_id in start_node_ids:
            graph_builder.add_edge(START, graph_names[node_id])
        for edge in spec["edges"]:
            source = graph_names.get(edge["source"])
            target = graph_names.get(edge["target"])
            if source and target:
                graph_builder.add_edge(source, target)
        terminal_node_ids = [node_id for node_id in graph_names if node_id not in edge_sources] or [next(reversed(graph_names))]
        for node_id in terminal_node_ids:
            graph_builder.add_edge(graph_names[node_id], END)
        return graph_builder.compile()

    def _langgraph_node(
        self,
        runtime,
        run: Run,
        user_message: str,
        node: dict,
        *,
        stream: bool,
    ) -> Callable[[WorkflowGraphState], dict[str, Any]]:
        def execute(state: WorkflowGraphState) -> dict[str, Any]:
            self._raise_if_cancelled()
            context = state["context"]
            steps = list(state.get("steps") or [])

            if stream and node["type"] == "LLM":
                writer = get_stream_writer()
                output = self._consume_streaming_node(
                    self._stream_llm_node(runtime, node, context),
                    writer,
                )
            elif stream and node["type"] == "Tool":
                writer = get_stream_writer()
                output = self._run_tool_node(
                    runtime,
                    node,
                    context,
                    stream=True,
                    writer=writer,
                )
            else:
                writer = None
                output = self._execute_node(runtime, node, context)

            self._raise_if_cancelled()
            output = dict(output or {})
            if not steps:
                output.setdefault("events", []).extend(self._initial_step_events(context))

            events = list(output.pop("events", []))
            context.update(output)
            step_payload = {
                "id": len(steps) + 1,
                "node_id": str(node["id"]),
                "node_type": str(node["type"]),
                "status": "succeeded",
                "output": output,
                "events": events,
            }
            next_steps = [*steps, step_payload]
            if writer is not None:
                writer({"event": "step", "step": step_payload})
            return {"context": context, "steps": next_steps}

        return execute

    def _consume_streaming_node(self, events, writer: Callable[[dict], None]) -> dict:
        while True:
            try:
                event = next(events)
            except StopIteration as stop:
                return dict(stop.value or {})
            if event:
                writer(event)

    def _initial_step_events(self, context: dict) -> list[dict]:
        return [
            {"event": "memory_used", "data": context.get("profile_memory_used", {})},
            {"event": "thinking_status", "data": context.get("thinking_status", {})},
            {"event": "search_status", "data": search_status_event(context.get("search_status", {}))},
            {"event": "skill_selection", "data": context.get("skill_selection", {})},
        ]

    @staticmethod
    def _langgraph_node_name(node: dict, index: int, used_names: set[str]) -> str:
        raw_name = str(node.get("id") or f"node_{index}").strip()
        base_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw_name).strip("_") or f"node_{index}"
        if base_name in {START, END}:
            base_name = f"workflow_{index}"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        return name

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
            return self._llm_output(agent, context, context["draft"], reasoning=context.get("draft_reasoning") or "")
        messages = self._llm_messages(agent, context)
        response = self._invoke_chat_model(agent, messages, context)
        draft = _message_content_text(response)
        draft = strip_or_block_leaked_tool_markup(draft)
        return self._llm_output(agent, context, draft)

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
        state = self._initial_tool_graph_state(agent, node, context)
        if state.get("output"):
            return state["output"]
        graph = self._build_tool_loop_graph(agent, stream=stream, writer=writer)
        final_state = graph.invoke(state)
        return final_state.get("output") or self._tool_final_output(final_state, AIMessage(content=""), stream=stream, max_rounds_reached=True)

    def _initial_tool_graph_state(self, agent, node: dict, context: dict) -> ToolGraphState:
        tool_policy = agent.settings.get("tool_policy") or {}
        allowed_names = set(tool_policy.get("allowed_tool_names") or [])
        bound_tools, langchain_tools = self._tool_runtime_bindings(agent, node, context, allowed_names)
        if not langchain_tools:
            return {
                "context": context,
                "node": node,
                "output": {"tool_outputs": [], "tool_stats": {"total_calls": 0, "tools_used": []}},
            }
        return {
            "context": context,
            "node": node,
            "messages": self._llm_messages(agent, context),
            "bound_tools": bound_tools,
            "langchain_tools": langchain_tools,
            "allowed_names": allowed_names,
            "total_calls": 0,
            "tools_used": [],
            "events": [],
            "web_sources": list(context.get("web_sources", [])),
            "search_status": dict(context.get("search_status") or {}),
            "round_index": 0,
            "tool_loop_start": time.monotonic(),
            "max_tool_calls": MAX_TOOL_CALLS_PER_RUN,
            "max_tool_wall_time": MAX_TOOL_WALL_TIME_SECONDS,
            "pending_calls": [],
            "latest_response": None,
            "loaded_skill_this_round": False,
        }

    def _tool_runtime_bindings(self, agent, node: dict, context: dict, allowed_names: set[str]) -> tuple[list[Tool], list[Any]]:
        bound_tools = self._runtime_tools(agent, node, context)
        if allowed_names:
            bound_tools = [tool for tool in bound_tools if tool.name in allowed_names or tool.type == "builtin_search"]
        langchain_tools = [
            build_langchain_tool(
                tool,
                session_key=str(context.get("session_id") or ""),
                agent_workdir=context.get("agent_workdir"),
            )
            for tool in bound_tools
        ]
        skill_loader_tool = self._skill_loader_tool(agent, context)
        if skill_loader_tool:
            langchain_tools.append(skill_loader_tool)
        return bound_tools, langchain_tools

    def _build_tool_loop_graph(
        self,
        agent,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ):
        graph_builder = StateGraph(ToolGraphState)
        graph_builder.add_node("check_limits", lambda state: state)
        graph_builder.add_node("call_model", lambda state: self._tool_graph_call_model(agent, state, stream=stream, writer=writer))
        graph_builder.add_node("execute_tools", lambda state: self._tool_graph_execute_tools(agent, state, stream=stream, writer=writer))
        graph_builder.add_node("final_answer", lambda state: self._tool_graph_final_answer(agent, state, stream=stream, writer=writer))
        graph_builder.add_edge(START, "check_limits")
        graph_builder.add_conditional_edges(
            "check_limits",
            lambda state: "final_answer" if self._tool_limits_reached(state) else "call_model",
            {"call_model": "call_model", "final_answer": "final_answer"},
        )
        graph_builder.add_conditional_edges(
            "call_model",
            self._route_after_tool_model,
            {"execute_tools": "execute_tools", "final_answer": "final_answer", "end": END},
        )
        graph_builder.add_conditional_edges(
            "execute_tools",
            lambda state: "final_answer" if self._tool_limits_reached(state) else "call_model",
            {"call_model": "call_model", "final_answer": "final_answer"},
        )
        graph_builder.add_edge("final_answer", END)
        return graph_builder.compile()

    def _tool_graph_call_model(
        self,
        agent,
        state: ToolGraphState,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> ToolGraphState:
        self._raise_if_cancelled()
        state = dict(state)
        state.pop("output", None)
        context = state["context"]
        round_index = int(state.get("round_index") or 0)
        if stream:
            response = self._stream_chat_response_to_writer(
                agent,
                state.get("messages", []),
                context,
                writer,
                tools=state.get("langchain_tools") or [],
                provisional_stream=True,
            )
            response = dsml_tool_call_parser.invoke(response, stage=f"tool node stream round {round_index}")
        else:
            response = self._invoke_chat_model(
                agent,
                state.get("messages", []),
                context,
                tools=state.get("langchain_tools") or [],
            )
            response = dsml_tool_call_parser.invoke(response, stage=f"tool node non-stream round {round_index}")

        state["latest_response"] = response
        state["pending_calls"] = []
        response_content = _message_content_text(response)
        response_reasoning = _message_reasoning_content(response)
        if response_content and not response.tool_calls:
            state["output"] = self._tool_direct_output(state, response, stream=stream)
            return state
        if not response.tool_calls:
            return state

        remaining_calls = max(0, int(state.get("max_tool_calls") or MAX_TOOL_CALLS_PER_RUN) - int(state.get("total_calls") or 0))
        calls_this_round = response.tool_calls[:remaining_calls]
        if not calls_this_round:
            return state
        events = list(state.get("events") or [])
        if response_reasoning and context.get("thinking_enabled") and not stream:
            reasoning_content = response_reasoning.strip()
            if reasoning_content:
                events.append({"event": "reasoning_token", "data": {"content": f"{reasoning_content}\n\n"}})
        messages = list(state.get("messages") or [])
        messages.append(
            AIMessage(
                content=response_content,
                additional_kwargs=response.additional_kwargs,
                tool_calls=calls_this_round,
            )
        )
        self._persist_intermediate_message(
            context,
            role="assistant",
            content=response_content,
            reasoning=response_reasoning,
            tool_calls=calls_this_round,
            meta={"node_id": state["node"]["id"], "round": round_index, "kind": "tool_calls"},
        )
        state["messages"] = messages
        state["events"] = events
        state["pending_calls"] = calls_this_round
        return state

    def _route_after_tool_model(self, state: ToolGraphState) -> str:
        if state.get("output"):
            return "end"
        if state.get("pending_calls"):
            return "execute_tools"
        return "final_answer"

    def _tool_graph_execute_tools(
        self,
        agent,
        state: ToolGraphState,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> ToolGraphState:
        self._raise_if_cancelled()
        state = dict(state)
        context = state["context"]
        node = state["node"]
        round_index = int(state.get("round_index") or 0)
        jobs = self._tool_jobs(agent, state)
        if stream and writer:
            for job in jobs:
                writer({"event": "tool_call_start", "data": self._tool_job_start_event(job)})

        tool_messages, job_results = self._invoke_toolnode(state)
        tool_messages_by_id = {message.tool_call_id: message for message in tool_messages}

        messages = list(state.get("messages") or [])
        events = list(state.get("events") or [])
        tools_used = list(state.get("tools_used") or [])
        total_calls = int(state.get("total_calls") or 0)
        web_sources = list(state.get("web_sources") or [])
        search_status = dict(state.get("search_status") or {})
        loaded_skill_this_round = False

        for job in jobs:
            tool_call_id = job["tc"].get("id") or ""
            result = job_results.get(tool_call_id)
            if result is None:
                tool_message = tool_messages_by_id.get(tool_call_id)
                result = self._tool_message_fallback_result(job, tool_message)
            total_calls += 1
            tools_used.append(job["tool_name"])
            loaded, web_sources, search_status = self._record_tool_job_result(
                job,
                result,
                context=context,
                node=node,
                round_index=round_index,
                messages=messages,
                events=events,
                web_sources=web_sources,
                search_status=search_status,
                stream=stream,
                writer=writer,
            )
            loaded_skill_this_round = loaded_skill_this_round or loaded

        if loaded_skill_this_round:
            self._refresh_system_message(messages, agent, context)
            bound_tools, langchain_tools = self._tool_runtime_bindings(
                agent,
                node,
                context,
                state.get("allowed_names") or set(),
            )
            state["bound_tools"] = bound_tools
            state["langchain_tools"] = langchain_tools

        state["messages"] = messages
        state["events"] = events
        state["tools_used"] = tools_used
        state["total_calls"] = total_calls
        state["web_sources"] = web_sources
        state["search_status"] = search_status
        state["round_index"] = round_index + 1
        state["pending_calls"] = []
        state["loaded_skill_this_round"] = loaded_skill_this_round
        return state

    def _tool_graph_final_answer(
        self,
        agent,
        state: ToolGraphState,
        *,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> ToolGraphState:
        self._raise_if_cancelled()
        state = dict(state)
        if stream:
            final = self._stream_chat_response_to_writer(
                agent,
                state.get("messages", []),
                state["context"],
                writer,
                stream_content=True,
            )
        else:
            final = self._invoke_chat_model(
                agent,
                state.get("messages", []),
                state["context"],
            )
        state["output"] = self._tool_final_output(state, final, stream=stream, max_rounds_reached=True)
        return state

    def _stream_chat_response_to_writer(
        self,
        agent,
        messages: list[BaseMessage],
        context: dict,
        writer: Callable[[dict], None] | None,
        *,
        tools: list[Any] | None = None,
        stream_content: bool = True,
        provisional_stream: bool = False,
    ) -> AIMessage:
        stream_events = self._stream_chat_response(
            agent,
            messages,
            context,
            tools=tools,
            stream_content=stream_content,
            provisional_stream=provisional_stream,
        )
        while True:
            try:
                event = next(stream_events)
            except StopIteration as stop:
                return stop.value or AIMessage(content="")
            if event and writer:
                writer(event)

    def _tool_direct_output(self, state: ToolGraphState, response: AIMessage, *, stream: bool) -> dict:
        response_content = _message_content_text(response)
        response_reasoning = _message_reasoning_content(response)
        output = {
            "draft": strip_or_block_leaked_tool_markup(response_content),
            "draft_reasoning": response_reasoning,
            "web_sources": state.get("web_sources", []),
            "search_status": state.get("search_status", {}),
            "tool_outputs": [],
            "tool_stats": {"total_calls": int(state.get("total_calls") or 0), "tools_used": list(state.get("tools_used") or [])},
            "events": list(state.get("events") or []),
        }
        if stream:
            output["draft_streamed"] = True
            output["draft_reasoning_streamed"] = bool(response_reasoning and state["context"].get("thinking_enabled"))
        return output

    def _tool_final_output(self, state: ToolGraphState, response: AIMessage, *, stream: bool, max_rounds_reached: bool) -> dict:
        response_content = _message_content_text(response)
        response_reasoning = _message_reasoning_content(response)
        output = {
            "draft": strip_or_block_leaked_tool_markup(response_content),
            "draft_reasoning": response_reasoning,
            "web_sources": state.get("web_sources", []),
            "search_status": state.get("search_status", {}),
            "tool_outputs": [],
            "tool_stats": {
                "total_calls": int(state.get("total_calls") or 0),
                "tools_used": list(state.get("tools_used") or []),
                "max_rounds_reached": max_rounds_reached,
            },
            "events": list(state.get("events") or []),
        }
        if stream:
            output["draft_streamed"] = bool(response_content)
            output["draft_reasoning_streamed"] = bool(response_reasoning and state["context"].get("thinking_enabled"))
        return output

    def _tool_limits_reached(self, state: ToolGraphState) -> bool:
        if int(state.get("total_calls") or 0) >= int(state.get("max_tool_calls") or MAX_TOOL_CALLS_PER_RUN):
            return True
        if int(state.get("round_index") or 0) >= MAX_TOOL_ROUNDS_PER_RUN:
            return True
        return (time.monotonic() - float(state.get("tool_loop_start") or time.monotonic())) > int(state.get("max_tool_wall_time") or MAX_TOOL_WALL_TIME_SECONDS)

    def _invoke_toolnode(self, state: ToolGraphState) -> tuple[list[ToolMessage], dict[str, dict]]:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return [], {}
        captured_results: dict[str, dict] = {}

        def wrap_tool_call(request, handler):
            tool_call = request.tool_call
            tool_name = str(tool_call.get("name") or "")
            tool_call_id = str(tool_call.get("id") or "")
            started = time.monotonic()
            try:
                result = request.tool.invoke(tool_call.get("args") or {})
                if not isinstance(result, dict):
                    result = {"tool": tool_name, "content": str(result), "result_preview": str(result)}
                result["latency_ms"] = result.get("latency_ms", int((time.monotonic() - started) * 1000))
            except ValueError as exc:
                result = {
                    "tool_name": tool_name,
                    "error": str(exc),
                    "content": f"Error: {exc}",
                    "result_preview": "",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                }
            except Exception as exc:
                result = {
                    "tool_name": tool_name,
                    "error": str(exc),
                    "content": f"Error: {exc}",
                    "result_preview": "",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                }
            captured_results[tool_call_id] = result
            status = "error" if result.get("error") or result.get("status") == "error" else "success"
            return ToolMessage(
                content=result.get("content") or result.get("result_preview") or "",
                name=tool_name,
                tool_call_id=tool_call_id,
                status=status,
            )

        graph_builder = StateGraph(ToolNodeInvokeState)
        graph_builder.add_node(
            "tools",
            ToolNode(state.get("langchain_tools") or [], wrap_tool_call=wrap_tool_call),
        )
        graph_builder.add_edge(START, "tools")
        graph_builder.add_edge("tools", END)
        output = graph_builder.compile().invoke({"messages": [messages[-1]]})
        tool_messages = [message for message in output.get("messages", []) if isinstance(message, ToolMessage)]
        return tool_messages, captured_results

    def _tool_message_fallback_result(self, job: dict, tool_message: ToolMessage | None) -> dict:
        content = tool_message.content if tool_message else f"Tool '{job['tool_name']}' not found"
        if tool_message and tool_message.status == "error":
            error_code = "tool_not_found" if not job.get("matching") and not job.get("internal") else "tool_error"
            return {"error": error_code, "content": content, "result_preview": content[:500], "latency_ms": 0}
        return {"content": content, "result_preview": str(content)[:500], "latency_ms": 0}

    def _tool_jobs(self, agent, state: ToolGraphState) -> list[dict]:
        context = state["context"]
        jobs = []
        for tc in state.get("pending_calls") or []:
            tool_name = str(tc.get("name") or "")
            tool_args = tc.get("args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"input": tool_args}
            is_skill_loader = tool_name == "load_skill"
            matching = next((tool for tool in state.get("bound_tools", []) if tool.name == tool_name), None)
            langchain_tool = next((tool for tool in state.get("langchain_tools", []) if tool.name == tool_name), None)
            job = {
                "tc": tc,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "matching": matching,
                "langchain_tool": langchain_tool,
                "_session_key": str(context.get("session_id") or ""),
                "_agent_workdir": context.get("agent_workdir"),
                "internal": is_skill_loader,
            }
            jobs.append(job)
        return jobs

    def _tool_job_start_event(self, job: dict) -> dict:
        display_tool = job.get("matching")
        if not display_tool and job.get("tool_name") == "load_skill":
            display_tool = type("_", (), {"id": None, "name": job["tool_name"], "type": "internal"})()
        return self._tool_call_start_event(
            display_tool,
            tool_name=job["tool_name"],
            tool_call_id=job["tc"].get("id") or "",
            input_preview=json.dumps(job["tool_args"], ensure_ascii=False),
        )

    def _record_tool_job_result(
        self,
        job: dict,
        result: dict,
        *,
        context: dict,
        node: dict,
        round_index: int,
        messages: list[BaseMessage],
        events: list[dict],
        web_sources: list[dict],
        search_status: dict,
        stream: bool,
        writer: Callable[[dict], None] | None,
    ) -> tuple[bool, list[dict], dict]:
        tc = job["tc"]
        tool_name = job["tool_name"]
        tool_args = job["tool_args"]
        matching = job.get("matching")
        loaded_skill = False

        if job.get("internal") and tool_name == "load_skill":
            display_tool = type("_", (), {"id": None, "name": tool_name, "type": "internal"})()
            status = "error" if result.get("status") == "error" else "success"
            event_data = tool_call_event(
                display_tool,
                result,
                status=status,
                input_preview=json.dumps(tool_args, ensure_ascii=False),
                error_code="skill_not_loadable" if status == "error" else None,
            )
            tool_content = result.get("content") or result.get("result_preview") or ""
            loaded_skill = status == "success"
        elif matching:
            if result.get("error"):
                event_data = tool_call_event(
                    matching,
                    result,
                    status="error",
                    input_preview=json.dumps(tool_args, ensure_ascii=False),
                    error_code="tool_error",
                )
                tool_content = result.get("content") or f"Error: {result['error']}"
            else:
                event_data = tool_call_event(matching, result, input_preview=json.dumps(tool_args, ensure_ascii=False))
                tool_content = result.get("content") or result.get("result_preview") or ""
        else:
            display_tool = type("_", (), {"id": None, "name": tool_name, "type": "unknown"})()
            event_data = tool_call_event(
                display_tool,
                {"tool": tool_name, "content": "", "result_preview": "", "latency_ms": 0},
                status="error",
                input_preview="{}",
                error_code="tool_not_found",
            )
            tool_content = f"Tool '{tool_name}' not found"

        if stream:
            stream_event_data = {**event_data, "type": "tool_call_result", "tool_call_id": tc.get("id") or ""}
            if writer:
                writer({"event": "tool_call_result", "data": stream_event_data})
        else:
            stream_event_data = event_data
            events.append({"event": "tool_call", "data": event_data})

        if matching and not result.get("error") and matching.type == "builtin_search":
            web_sources, search_status = merge_web_search_tool_result(web_sources, search_status, result)
            search_event = search_status_event(search_status)
            if stream:
                search_event["tool_call_id"] = tc.get("id") or ""
                if writer:
                    writer({"event": "search_status", "data": search_event})
            else:
                events.append({"event": "search_status", "data": search_event})

        messages.append(ToolMessage(content=tool_content, tool_call_id=tc["id"], name=tool_name))
        self._persist_intermediate_message(
            context,
            role="tool",
            content=tool_content,
            tool_call_id=tc["id"],
            tool_name=tool_name,
            meta={**stream_event_data, "node_id": node["id"], "round": round_index, "kind": "tool_result"},
        )
        return loaded_skill, web_sources, search_status

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

    def _stream_chat_response(
        self,
        agent,
        messages: list[BaseMessage],
        context: dict,
        *,
        tools: list[Any] | None = None,
        stream_content: bool = False,
        provisional_stream: bool = False,
    ):
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_call_builders: dict[int, dict] = {}
        final_tool_calls: list[dict] = []
        saw_tool_call = False
        pending_live_content = ""
        suppress_content_stream = False
        emitted_live_content = False
        provisional_active = bool(provisional_stream and tools)
        provisional_chunks: list[str] = []
        # With tools available, models may emit a short natural-language preface
        # before deciding to call a tool. Buffer it until we know no tool call is
        # coming so final direct answers still stream as normal token chunks.
        should_stream_content_live = stream_content and (not tools or provisional_active)
        for chunk in self.provider.stream(
            messages,
            model=agent.model,
            temperature=agent.temperature,
            runtime_config=agent.runtime_config,
            tools=tools,
            thinking_enabled=self._thinking_request_value(context),
            cancel_event=self._cancel_event,
        ):
            self._raise_if_cancelled()
            reasoning_chunk = _message_reasoning_content(chunk)
            content_chunk = _message_content_text(chunk)
            if reasoning_chunk:
                if context.get("thinking_enabled"):
                    reasoning_chunks.append(reasoning_chunk)
                    if stream_content and not saw_tool_call:
                        yield {"event": "reasoning_token", "content": reasoning_chunk}
            if content_chunk:
                content_chunks.append(content_chunk)
                if should_stream_content_live and not saw_tool_call:
                    pending_live_content, suppress_content_stream, safe_chunks = buffer_stream_content(
                        pending_live_content,
                        content_chunk,
                        suppress_content_stream,
                    )
                    for safe_content in safe_chunks:
                        emitted_live_content = True
                        if provisional_active:
                            provisional_chunks.append(safe_content)
                        else:
                            yield {"event": "token", "content": safe_content}
            tool_call_chunks = getattr(chunk, "tool_call_chunks", []) or []
            if tool_call_chunks:
                saw_tool_call = True
                provisional_chunks = []
                merge_stream_tool_call_chunks(tool_call_builders, tool_call_chunks)
            elif getattr(chunk, "tool_calls", None):
                saw_tool_call = True
                provisional_chunks = []
                final_tool_calls = list(chunk.tool_calls or [])
        self._raise_if_cancelled()
        if not final_tool_calls:
            final_tool_calls = finalize_stream_tool_calls(tool_call_builders)

        joined_content = "".join(content_chunks)
        content_for_response = joined_content
        if final_tool_calls:
            if joined_content.strip():
                logger.warning("Dropping assistant content emitted before tool calls during stream response; preview=%r", dsml_preview(joined_content))
            content_for_response = ""
        elif not final_tool_calls and contains_dsml_tool_calls(joined_content):
            logger.warning("Detected DSML tool call markup in streamed assistant content")
            if tools:
                dsml_calls = parse_dsml_tool_calls(joined_content)
                if dsml_calls:
                    logger.warning("Parsed DSML tool calls from streamed content: tools=%s", dsml_tool_names(dsml_calls))
                    final_tool_calls = dsml_calls
                    content_for_response = ""
                else:
                    logger.warning("Failed to parse DSML tool calls from streamed content; full content:\n%s", joined_content)
                    content_for_response = DSML_TOOL_MARKUP_ERROR
            else:
                logger.warning("Blocked DSML tool call markup in streamed final content; preview=%r", dsml_preview(joined_content))
                content_for_response = strip_or_block_leaked_tool_markup(joined_content)
        elif not final_tool_calls and contains_leaked_tool_markup(joined_content):
            logger.warning("Blocked incomplete tool call markup in streamed content; full content:\n%s", joined_content)
            content_for_response = DSML_TOOL_MARKUP_ERROR

        if stream_content and not final_tool_calls and content_for_response:
            if should_stream_content_live:
                if suppress_content_stream:
                    if content_for_response == DSML_TOOL_MARKUP_ERROR or not emitted_live_content:
                        yield {"event": "token", "content": content_for_response}
                elif not contains_leaked_tool_markup(joined_content):
                    if provisional_active:
                        for safe_content in provisional_chunks:
                            yield {"event": "token", "content": safe_content}
                        if pending_live_content:
                            yield {"event": "token", "content": pending_live_content}
                    elif pending_live_content:
                        yield {"event": "token", "content": pending_live_content}
        additional_kwargs = {}
        if reasoning_chunks:
            additional_kwargs["reasoning_content"] = "".join(reasoning_chunks)
        return AIMessage(
            content=content_for_response or "",
            additional_kwargs=additional_kwargs,
            tool_calls=final_tool_calls,
        )

    def _tool_call_start_event(self, tool: Tool | None, *, tool_name: str, tool_call_id: str, input_preview: str = "") -> dict:
        return {
            "type": "tool_call_start",
            "tool_call_id": tool_call_id,
            "tool_id": tool.id if tool else None,
            "tool_name": tool.name if tool else tool_name,
            "tool_type": tool.type if tool else "unknown",
            "status": "running",
            "input_preview": input_preview[:500],
            "result_preview": "",
            "latency_ms": 0,
            "error_code": "",
        }

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
            return self._llm_output(agent, context, draft, reasoning=draft_reasoning)
        messages = self._llm_messages(agent, context)
        chunks = []
        reasoning_chunks = []
        pending_live_content = ""
        suppress_content_stream = False
        emitted_live_content = False
        for chunk in self.provider.stream(
            messages,
            model=agent.model,
            temperature=agent.temperature,
            runtime_config=agent.runtime_config,
            thinking_enabled=self._thinking_request_value(context),
            cancel_event=self._cancel_event,
        ):
            self._raise_if_cancelled()
            reasoning_chunk = _message_reasoning_content(chunk)
            content_chunk = _message_content_text(chunk)
            if reasoning_chunk and context.get("thinking_enabled"):
                reasoning_chunks.append(reasoning_chunk)
                yield {"event": "reasoning_token", "content": reasoning_chunk}
            if content_chunk:
                chunks.append(content_chunk)
                pending_live_content, suppress_content_stream, safe_chunks = buffer_stream_content(
                    pending_live_content,
                    content_chunk,
                    suppress_content_stream,
                )
                for safe_content in safe_chunks:
                    emitted_live_content = True
                    yield {"event": "token", "content": safe_content}
        self._raise_if_cancelled()
        raw_draft = "".join(chunks)
        draft = strip_or_block_leaked_tool_markup(raw_draft)
        if draft and draft != raw_draft:
            logger.warning("Blocked leaked tool call markup in streamed LLM node; preview=%r", dsml_preview(raw_draft))
        if draft:
            if suppress_content_stream:
                if draft == DSML_TOOL_MARKUP_ERROR or not emitted_live_content:
                    yield {"event": "token", "content": draft}
            elif not contains_leaked_tool_markup(raw_draft) and pending_live_content:
                yield {"event": "token", "content": pending_live_content}
        return self._llm_output(agent, context, draft, reasoning="".join(reasoning_chunks))

    def _thinking_request_value(self, context: dict) -> bool | None:
        status = context.get("thinking_status") or {}
        if status.get("type") not in {"native", "prompt"}:
            return None
        return bool(status.get("enabled"))

    def _llm_messages(self, agent, context: dict) -> list[BaseMessage]:
        source_text = "\n".join(f"- {item['title']}: {item['snippet']}" for item in context.get("sources", []))
        web_sources_text = web_source_text(context.get("web_sources", []))
        tool_text = "\n".join(f"- {item['tool']}: {item['content']}" for item in context.get("tool_outputs", []))
        variable_text = "\n".join(f"- {key}: {value}" for key, value in context.get("variables", {}).items())
        attachment_text = self._attachment_text(context.get("uploads", []))
        skill_manifest_content = skill_manifest_text(context.get("skill_manifest") or [])
        loaded_skill_content = loaded_skill_text(context.get("loaded_skills") or [])
        thinking_blocks = []
        thinking_msgs = thinking_messages(context)
        if thinking_msgs:
            thinking_blocks = [msg["content"] for msg in thinking_msgs]
        search_instruction = ""
        if context.get("search_enabled"):
            search_instruction = (
                "本轮联网搜索工具可用，但不是必选。只有当问题需要最新信息、外部事实、网页资料、天气、价格、新闻或可变信息时才调用；"
                "简单算术、常识推理、翻译、代码解释、当前会话内容总结等不需要联网搜索的问题，请直接回答。"
            )
            
        raw_summary = context.get('memory_summary') or ''
        formatted_summary = "无"
        if raw_summary.strip():
            try:
                turns = json.loads(raw_summary)
                if isinstance(turns, list):
                    formatted_summary = "\n".join(f"用户：{t['user']}\n助手：{t['assistant']}" for t in turns)
                else:
                    formatted_summary = raw_summary
            except Exception:
                formatted_summary = raw_summary
                
        system_parts = [
            agent.system_prompt or "你是一个自定义智能体。",
            *thinking_blocks,
            search_instruction,
            skill_manifest_content,
            loaded_skill_content,
            f"Web search results for this turn:\n{web_sources_text or 'None'}",
            f"可用知识片段：\n{source_text or '无'}",
            f"工具输出：\n{tool_text or '无'}",
            f"用户变量：\n{variable_text or '无'}",
            f"会话记忆摘要：\n{formatted_summary}",
            f"本轮附件上下文：\n{attachment_text or '无'}",
            f"Long-term Agent memory:\n{context.get('profile_memory') or 'None'}",
        ]
        system_content = "\n\n".join(part for part in system_parts if part.strip())
        # Guard against token overflow: truncate overly long system prompts
        max_system_chars = 100_000  # ~50k tokens, safe for most model context windows
        if len(system_content) > max_system_chars:
            system_content = system_content[:max_system_chars] + "\n\n[上下文已截断以避免超出模型上下文窗口限制]"
        messages: list[BaseMessage] = [SystemMessage(content=system_content)]
        history_messages = self._history_messages_for_llm(context)
        messages.extend(history_messages)
        if not self._history_contains_current_message(context):
            messages.append(HumanMessage(content=self._user_content(context["input"], context.get("uploads", []))))
        return messages

    def _history_messages_for_llm(self, context: dict) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        current_message_id = context.get("current_message_id")
        history = context.get("history_messages") or []
        index = 0
        while index < len(history):
            item = history[index]
            role = item.get("role")
            if role not in {"user", "assistant", "tool"}:
                index += 1
                continue
            if role == "tool":
                index += 1
                continue
            content = item.get("content") or ""
            if current_message_id and item.get("id") == current_message_id and role == "user":
                content = self._user_content(context["input"], context.get("uploads", []))
            if role == "assistant" and contains_leaked_tool_markup(content):
                cleaned_content = strip_complete_dsml_tool_call_blocks(content).strip()
                if cleaned_content and not contains_leaked_tool_markup(cleaned_content):
                    logger.warning(
                        "Cleaned DSML tool call markup from historical assistant message id=%s",
                        item.get("id"),
                    )
                    content = cleaned_content
                else:
                    logger.warning(
                        "Skipping historical assistant message with leaked DSML tool call markup id=%s; preview=%r",
                        item.get("id"),
                        dsml_preview(content),
                    )
                    index += 1
                    continue
            tool_calls = item.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                tool_call_ids = {call.get("id") for call in tool_calls if call.get("id")}
                tool_messages = []
                next_index = index + 1
                while next_index < len(history) and history[next_index].get("role") == "tool":
                    tool_item = history[next_index]
                    if tool_item.get("tool_call_id") in tool_call_ids:
                        tool_messages.append(tool_item)
                    next_index += 1
                if tool_call_ids and tool_call_ids.issubset({tool.get("tool_call_id") for tool in tool_messages}):
                    additional_kwargs = {}
                    if item.get("reasoning") and (item.get("meta") or {}).get("requires_reasoning_replay"):
                        additional_kwargs["reasoning_content"] = item.get("reasoning")
                    messages.append(
                        AIMessage(
                            content=content or "",
                            additional_kwargs=additional_kwargs,
                            tool_calls=_normalize_langchain_tool_calls(tool_calls),
                        )
                    )
                    for tool_item in tool_messages:
                        messages.append(
                            ToolMessage(
                                content=tool_item.get("content") or "",
                                tool_call_id=tool_item.get("tool_call_id") or "",
                                name=tool_item.get("tool_name") or None,
                            )
                        )
                index = next_index
                continue
            if content:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    additional_kwargs = {}
                    if item.get("reasoning") and (item.get("meta") or {}).get("requires_reasoning_replay"):
                        additional_kwargs["reasoning_content"] = item.get("reasoning")
                    messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
            index += 1
        return messages

    def _history_contains_current_message(self, context: dict) -> bool:
        current_message_id = context.get("current_message_id")
        if not current_message_id:
            return False
        return any(item.get("id") == current_message_id for item in context.get("history_messages") or [])

    def _llm_output(self, agent, context: dict, draft: str, *, reasoning: str = "") -> dict:
        return {
            "draft": draft,
            "used_memory": bool(context.get("memory_summary")),
            "used_profile_memory": bool(context.get("profile_memory")),
            "attachment_count": len(context.get("uploads", [])),
            "history_message_count": len(context.get("history_messages") or []),
            "model": agent.model,
            "mock": self.provider.last_chat_mock,
            "thinking_enabled": bool(context.get("thinking_enabled")),
            "thinking_type": (context.get("thinking_status") or {}).get("type", "none"),
            "reasoning_replay_required": bool(context.get("reasoning_replay_required")),
            "reasoning_chars": len(reasoning or ""),
            "search_enabled": bool(context.get("search_enabled")),
            "search_result_count": len(context.get("web_sources", [])),
            "loaded_skills": [
                {"id": item.get("id"), "name": item.get("name"), "activation_mode": item.get("activation_mode"), "score": item.get("score")}
                for item in context.get("loaded_skills", [])
            ],
        }

    def _runtime_agent(self, agent: Agent, mode: str, user_id: int):
        # Auto-fallback to draft if published is requested but agent has never been published.
        if mode == "published" and not agent.published_version_id:
            mode = "draft"

        if mode not in {"draft", "published"}:
            raise ValueError("mode must be draft or published")
        if mode == "published":
            if not agent.published_version_id:
                raise ValueError("当前智能体还没有发布版本")
            version = self.db.get(AgentVersion, agent.published_version_id)
            if not version:
                raise ValueError("发布版本不存在")
            snapshot = version.snapshot or {}
            source = {
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
        else:
            detail = get_agent_detail(self.db, agent)
            source = {
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

        user_model_config = self._user_model_config(user_id, source["user_model_config_id"])
        runtime_config = user_model_runtime_config(user_model_config) if user_model_config else None

        skill_bindings = self._runtime_skill_bindings(agent)

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
            skill_bindings=skill_bindings,
            workflow=source["workflow"],
            model_config=self._model_config(source["model_id"], source["model"]),
            user_model_config=user_model_config,
            runtime_config=runtime_config,
            capability_config=user_model_config or self._model_config(source["model_id"], source["model"]),
            settings={
                "variables": source["variables"],
                "memory": source["memory"],
                "rag": source["rag"],
                "tool_policy": source["tool_policy"],
                "workdir": source["workdir"],
            },
        )

    def _runtime_skill_bindings(self, agent: Agent) -> list[dict]:
        agent_skill_rows = (
            self.db.query(AgentSkill)
            .filter(
                AgentSkill.agent_id == agent.id,
                AgentSkill.enabled.is_(True),
            )
            .all()
        )
        if not agent_skill_rows:
            return []

        skill_ids = [row.skill_id for row in agent_skill_rows]
        skills = (
            self.db.query(Skill)
            .filter(Skill.id.in_(skill_ids), Skill.enabled.is_(True))
            .all()
        )
        skills_by_id = {skill.id: skill for skill in skills}
        priority_map = {row.skill_id: row.priority for row in agent_skill_rows}

        tool_ids_by_skill: dict[int, list[int]] = {skill_id: [] for skill_id in skill_ids}
        for row in self.db.query(SkillTool).filter(SkillTool.skill_id.in_(skill_ids)).all():
            tool_ids_by_skill.setdefault(row.skill_id, []).append(row.tool_id)

        kb_ids_by_skill: dict[int, list[int]] = {skill_id: [] for skill_id in skill_ids}
        for row in self.db.query(SkillKnowledgeBase).filter(SkillKnowledgeBase.skill_id.in_(skill_ids)).all():
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

    def _apply_runtime_skills(self, runtime, context: dict, chat_session: ChatSession) -> None:
        bindings = list(getattr(runtime, "skill_bindings", []) or [])
        manifest = [skill_manifest(item) for item in bindings]
        context["skill_manifest"] = manifest
        if not bindings:
            context["skill_selection"] = {"loaded": [], "auto_candidates": [], "threshold": SKILL_AUTO_THRESHOLD, "top_k": SKILL_AUTO_TOP_K}
            return

        always_skills = [item for item in bindings if item["activation_mode"] == "always"]
        manual_skills = [
            item for item in bindings
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
        auto_skills = [
            item
            for item, score in auto_candidates
            if score >= SKILL_AUTO_THRESHOLD
        ][:SKILL_AUTO_TOP_K]

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

    def _model_config(self, model_id: int | None, model_name: str | None) -> ModelConfig | None:
        return resolve_agent_model(self.db, model_id=model_id, model_name=model_name)

    def _user_model_config(self, user_id: int, config_id: int | None) -> UserModelConfig | None:
        if config_id is None:
            return None
        return resolve_user_model_config(self.db, user_id=user_id, config_id=config_id, enabled_only=True)

    def _validate_model_capabilities(self, model: ModelConfig | UserModelConfig | None, uploads: list[Upload]) -> None:
        if not model:
            return
        has_image = any(upload.kind == "image" for upload in uploads)
        if has_image and not getattr(model, "supports_image", False):
            raise ValueError("Selected model does not support image input")
        has_document = any(upload.kind == "document" for upload in uploads)
        if has_document and not getattr(model, "supports_document", True):
            raise ValueError("Selected model does not support document input")

    def _skill_loader_tool(self, agent, context: dict) -> StructuredTool | None:
        schema = self._skill_loader_schema(agent, context)
        if not schema:
            return None
        description = schema["function"]["description"]

        def load_skill(skill_id: int | None = None, skill_name: str = "", reason: str = "") -> dict:
            return self._handle_load_skill_call(
                agent,
                context,
                {"skill_id": skill_id, "skill_name": skill_name, "reason": reason},
            )

        return StructuredTool.from_function(
            load_skill,
            name="load_skill",
            description=description,
        )

    def _skill_loader_schema(self, agent, context: dict) -> dict | None:
        loaded_ids = {item.get("id") for item in context.get("loaded_skills", [])}
        loadable = [
            item for item in getattr(agent, "skill_bindings", []) or []
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

    def _handle_load_skill_call(self, agent, context: dict, args: dict) -> dict:
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

        payload = self._merge_loaded_skill(agent, context, binding, score=1.0, reason="load_skill")
        retrieved = self._retrieve_skill_knowledge(agent, context, binding)
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

    def _merge_loaded_skill(self, agent, context: dict, item: dict, *, score: float, reason: str) -> dict:
        loaded = context.setdefault("loaded_skills", [])
        if any(existing.get("id") == item["id"] for existing in loaded):
            return next(existing for existing in loaded if existing.get("id") == item["id"])
        if item.get("system_prompt", "").strip():
            agent.system_prompt = "\n\n".join(
                part for part in [
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
        selection = context.setdefault("skill_selection", {"loaded": [], "auto_candidates": [], "threshold": SKILL_AUTO_THRESHOLD, "top_k": SKILL_AUTO_TOP_K})
        selection["loaded"] = loaded
        return payload

    def _retrieve_skill_knowledge(self, agent, context: dict, item: dict) -> list[dict]:
        kb_ids = item.get("knowledge_base_ids") or []
        if not kb_ids or not context.get("rag_enabled", True):
            return []
        rag_result = run_rag_pipeline(
            self.db,
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

    def _refresh_system_message(self, messages: list[BaseMessage], agent, context: dict) -> None:
        if not messages or not isinstance(messages[0], SystemMessage):
            return
        messages[0] = self._llm_messages(agent, context)[0]

    def _runtime_tools(self, agent, node: dict, context: dict | None = None) -> list[Tool]:
        tool_ids = getattr(agent, "tool_ids", []) or []
        tools: list[Tool] = []
        if tool_ids:
            tools = (
                self.db.query(Tool)
                .filter(Tool.id.in_(tool_ids), Tool.enabled.is_(True))
                .order_by(Tool.id.asc())
                .all()
            )
        if context and context.get("search_enabled"):
            from core.services.bootstrap import ensure_builtin_tools

            ensure_builtin_tools(self.db)
            existing_ids = {tool.id for tool in tools}
            search_tool = (
                self.db.query(Tool)
                .filter(Tool.name == "web_search", Tool.type == "builtin_search", Tool.enabled.is_(True))
                .first()
            )
            if search_tool and search_tool.id not in existing_ids:
                tools.append(search_tool)
        return tools

    def _attachment_text(self, uploads: list[Upload]) -> str:
        lines = []
        for upload in uploads:
            if upload.kind == "document":
                lines.append(f"[{upload.filename}]\n{upload.text[:6000]}")
            elif upload.kind == "image":
                lines.append(f"[Image: {upload.filename}]")
        return "\n\n".join(lines)

    def _user_content(self, text: str, uploads: list[Upload]):
        image_uploads = [upload for upload in uploads if upload.kind == "image"]
        if not image_uploads:
            return text
        content = [{"type": "text", "text": text}]
        for upload in image_uploads:
            content.append({"type": "image_url", "image_url": {"url": upload.data_url}})
        return content

    def _merge_variables(self, definitions: list[dict], provided: dict) -> dict:
        merged = {}
        for definition in definitions:
            key = definition.get("key")
            if key:
                merged[key] = provided.get(key, definition.get("default_value"))
        for key, value in provided.items():
            if key not in merged:
                merged[key] = value
        return merged

    def _persist_intermediate_message(
        self,
        context: dict,
        *,
        role: str,
        content: str,
        reasoning: str = "",
        tool_calls: list[dict] | None = None,
        tool_call_id: str = "",
        tool_name: str = "",
        meta: dict | None = None,
    ) -> None:
        session_id = context.get("session_id")
        if not session_id:
            return
        if role == "assistant" and contains_leaked_tool_markup(content):
            logger.warning(
                "Blocked leaked tool call markup before persisting intermediate assistant message; preview=%r",
                dsml_preview(content),
            )
            content = ""
        visible_reasoning = reasoning if context.get("thinking_enabled") else ""
        payload_meta = {
            "is_intermediate": True,
            "run_id": context.get("run_id"),
            "thinking_enabled": bool(context.get("thinking_enabled")),
            **(meta or {}),
        }
        if role == "assistant" and visible_reasoning and context.get("reasoning_replay_required"):
            payload_meta["requires_reasoning_replay"] = True
        message = Message(
            session_id=session_id,
            role=role,
            content=content or "",
            reasoning=visible_reasoning or "",
            sources=[],
            tool_calls=tool_calls or [],
            tool_call_id=tool_call_id or "",
            tool_name=tool_name or "",
            meta=payload_meta,
        )
        self.db.add(message)
        self.db.flush()
        self.db.commit()

    def _session_history(self, session_id: int, *, max_messages: int) -> list[dict]:
        rows = (
            self.db.query(Message)
            .filter(Message.session_id == session_id, Message.role.in_(["user", "assistant", "tool"]))
            .order_by(Message.id.desc())
            .limit(max(1, min(int(max_messages or 12), 100)))
            .all()
        )
        history = []
        for message in reversed(rows):
            content = self._trim_history_content(message.content or "")
            if not content and message.role != "assistant":
                continue
            history.append(
                {
                    "id": message.id,
                    "role": message.role,
                    "content": content,
                    "reasoning": self._trim_history_content(message.reasoning or ""),
                    "tool_calls": message.tool_calls or [],
                    "tool_call_id": message.tool_call_id or "",
                    "tool_name": message.tool_name or "",
                    "meta": message.meta or {},
                }
            )
        return history

    @staticmethod
    def _trim_history_content(content: str, limit: int = 6000) -> str:
        text = content.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "\n[历史消息过长，已截断]"

