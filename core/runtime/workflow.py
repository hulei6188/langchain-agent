from __future__ import annotations

from datetime import datetime
import json
import time
from types import SimpleNamespace

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
    RunStep,
    Session as ChatSession,
    SessionMemory,
    Skill,
    SkillKnowledgeBase,
    SkillTool,
    Tool,
    UserModelConfig,
    Upload,
    WorkflowDefinition,
)
from core.integrations.llm import ChatResponse, OpenAICompatibleProvider
from core.services.agents import get_agent_detail, normalize_memory, normalize_rag, normalize_tool_policy
from core.services.rag import retrieve
from core.services.memory import format_profile_memory, get_memory_profile, memory_used_event
from core.services.models import resolve_agent_model
from core.services.tools import execute_tool, tool_call_event, tool_schema_for_llm
from core.services.uploads import get_workspace_uploads
from core.services.user_models import (
    resolve_user_model_config,
    user_model_runtime_config,
)
from core.services import web_search as web_search_service


MAX_TOOL_CALLS_PER_RUN = 50
MAX_TOOL_ROUNDS_PER_RUN = 16
MAX_TOOL_WALL_TIME_SECONDS = 180


def default_workflow() -> list[dict]:
    return [
        {"id": "start", "type": "Start", "name": "接收用户输入", "config": {}},
        {"id": "knowledge", "type": "Knowledge", "name": "检索绑定知识库", "config": {"top_k": 4}},
        {"id": "tool", "type": "Tool", "name": "调用绑定工具", "config": {"tools": []}},
        {"id": "llm", "type": "LLM", "name": "生成候选回答", "config": {}},
        {"id": "answer", "type": "Answer", "name": "输出最终回答", "config": {}},
    ]


class WorkflowRunner:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.provider = OpenAICompatibleProvider()

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
        thinking_status = self._thinking_status(runtime.capability_config, thinking_enabled)
        search_status = self._search_status(user_message, search_enabled)

        rag_config = normalize_rag({**dict(runtime.settings.get("rag") or {}), **dict(rag_options or {})})
        memory_config = normalize_memory(runtime.settings.get("memory"))
        effective_rag_enabled = rag_config["enabled_by_default"] if rag_enabled is None else bool(rag_enabled)

        run = Run(workspace_id=agent.workspace_id, agent_id=agent.id, session_id=chat_session.id, status="running")
        self.db.add(run)
        self.db.flush()
        self.db.commit()
        self.db.refresh(run)

        memory = self._session_memory(chat_session.id)
        profile_memory = get_memory_profile(
            self.db,
            workspace_id=agent.workspace_id,
            user_id=chat_session.user_id,
            agent_id=agent.id,
        )
        profile_memory_text = format_profile_memory(profile_memory)
        profile_memory_event = memory_used_event(profile_memory, session_summary_used=bool(memory and memory.summary))
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
            "memory_summary": memory.summary if memory else "",
            "profile_memory": profile_memory_text,
            "profile_memory_used": profile_memory_event,
            "memory_enabled": memory_config.get("enabled", False),
            "rag_enabled": effective_rag_enabled,
            **({"rag_enabled_request": rag_enabled} if rag_enabled is not None else {}),
            "rag_top_k": rag_config["top_k"],
            "rag_config": rag_config,
            "thinking_enabled": thinking_status["enabled"],
            "thinking_status": thinking_status,
            "reasoning_replay_required": self.provider.requires_reasoning_replay(model=runtime.model, runtime_config=runtime.runtime_config),
            "search_enabled": search_status["enabled"],
            "search_status": search_status,
            "web_sources": search_status.get("sources", []),
            "uploads": uploads,
        }
        steps: list[dict] = []
        for node in runtime.workflow:
            output = self._execute_node(runtime, node, context)
            if not steps:
                output.setdefault("events", []).append({"event": "memory_used", "data": profile_memory_event})
                output.setdefault("events", []).append({"event": "thinking_status", "data": thinking_status})
                output.setdefault("events", []).append({"event": "search_status", "data": self._search_status_event(search_status)})
            events = output.pop("events", [])
            context.update(output)
            step = RunStep(
                run_id=run.id,
                node_id=node["id"],
                node_type=node["type"],
                status="succeeded",
                input={"input": user_message},
                output=output,
            )
            self.db.add(step)
            self.db.flush()
            self.db.commit()
            self.db.refresh(step)
            steps.append(
                {
                    "id": step.id,
                    "node_id": step.node_id,
                    "node_type": step.node_type,
                    "status": step.status,
                    "output": output,
                    "events": events,
                }
            )

        final_answer = context.get("answer") or context.get("draft") or "当前智能体没有生成回答。"
        if context.get("memory_enabled"):
            self._update_session_memory(
                chat_session.id,
                user_message,
                final_answer,
                int(runtime.settings.get("memory", {}).get("max_messages", 12)),
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
        steps: list[dict] = []
        for node in runtime.workflow:
            if node["type"] == "LLM":
                output = yield from self._stream_llm_node(runtime, node, context)
            elif node["type"] == "Tool":
                output = yield from self._stream_tool_node(runtime, node, context)
            else:
                output = self._execute_node(runtime, node, context)
            if not steps:
                output.setdefault("events", []).append({"event": "memory_used", "data": context.get("profile_memory_used", {})})
                output.setdefault("events", []).append({"event": "thinking_status", "data": context.get("thinking_status", {})})
                output.setdefault("events", []).append({"event": "search_status", "data": self._search_status_event(context.get("search_status", {}))})
            events = output.pop("events", [])
            context.update(output)
            step = self._persist_step(run, node, user_message, output)
            step_payload = {
                "id": step.id,
                "node_id": step.node_id,
                "node_type": step.node_type,
                "status": step.status,
                "output": output,
                "events": events,
            }
            steps.append(step_payload)
            yield {"event": "step", "step": step_payload}

        final_answer = context.get("answer") or context.get("draft") or "当前智能体没有生成回答。"
        if context.get("memory_enabled"):
            self._update_session_memory(
                chat_session.id,
                user_message,
                final_answer,
                int(runtime.settings.get("memory", {}).get("max_messages", 12)),
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
        thinking_status = self._thinking_status(runtime.capability_config, thinking_enabled)
        search_status = self._search_status(user_message, search_enabled)

        rag_config = normalize_rag({**dict(runtime.settings.get("rag") or {}), **dict(rag_options or {})})
        memory_config = normalize_memory(runtime.settings.get("memory"))
        effective_rag_enabled = rag_config["enabled_by_default"] if rag_enabled is None else bool(rag_enabled)

        run = Run(workspace_id=agent.workspace_id, agent_id=agent.id, session_id=chat_session.id, status="running")
        self.db.add(run)
        self.db.flush()
        self.db.commit()
        self.db.refresh(run)

        memory = self._session_memory(chat_session.id)
        profile_memory = get_memory_profile(
            self.db,
            workspace_id=agent.workspace_id,
            user_id=chat_session.user_id,
            agent_id=agent.id,
        )
        profile_memory_text = format_profile_memory(profile_memory)
        profile_memory_event = memory_used_event(profile_memory, session_summary_used=bool(memory and memory.summary))
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
            "memory_summary": memory.summary if memory else "",
            "profile_memory": profile_memory_text,
            "profile_memory_used": profile_memory_event,
            "memory_enabled": memory_config.get("enabled", False),
            "rag_enabled": effective_rag_enabled,
            **({"rag_enabled_request": rag_enabled} if rag_enabled is not None else {}),
            "rag_top_k": rag_config["top_k"],
            "rag_config": rag_config,
            "thinking_enabled": thinking_status["enabled"],
            "thinking_status": thinking_status,
            "reasoning_replay_required": self.provider.requires_reasoning_replay(model=runtime.model, runtime_config=runtime.runtime_config),
            "search_enabled": search_status["enabled"],
            "search_status": search_status,
            "web_sources": search_status.get("sources", []),
            "uploads": uploads,
        }
        return runtime, run, context

    def _persist_step(self, run: Run, node: dict, user_message: str, output: dict) -> RunStep:
        step = RunStep(
            run_id=run.id,
            node_id=node["id"],
            node_type=node["type"],
            status="succeeded",
            input={"input": user_message},
            output=output,
        )
        self.db.add(step)
        self.db.flush()
        self.db.commit()
        self.db.refresh(step)
        return step

    def _execute_node(self, agent, node: dict, context: dict) -> dict:
        node_type = node["type"]
        if node_type == "Start":
            return {
                "started": True,
                "variables": context.get("variables", {}),
                "rag_enabled": context.get("rag_enabled", True),
                "search_enabled": context.get("search_enabled", False),
                "attachment_count": len(context.get("uploads", [])),
            }
        if node_type == "Knowledge":
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
            rag_result = retrieve(
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
        if node_type == "Tool":
            bound_tools = self._runtime_tools(agent, node, context)
            tool_policy = (agent.settings.get("tool_policy") or {})
            allowed_names = set(tool_policy.get("allowed_tool_names") or [])
            if allowed_names:
                bound_tools = [t for t in bound_tools if t.name in allowed_names or t.type == "builtin_search"]
            if not bound_tools:
                return {"tool_outputs": [], "tool_stats": {"total_calls": 0, "tools_used": []}}

            tool_schemas = [tool_schema_for_llm(t) for t in bound_tools]
            messages = self._llm_messages(agent, context)
            total_calls = 0
            tools_used: list[str] = []
            events = []
            web_sources = list(context.get("web_sources", []))
            search_status = dict(context.get("search_status") or {})
            max_tool_calls = MAX_TOOL_CALLS_PER_RUN
            max_tool_wall_time = MAX_TOOL_WALL_TIME_SECONDS
            tool_loop_start = time.monotonic()

            for _round in range(MAX_TOOL_ROUNDS_PER_RUN):
                if total_calls >= max_tool_calls:
                    break
                if time.monotonic() - tool_loop_start > max_tool_wall_time:
                    break
                response = self.provider.chat(
                    messages,
                    model=agent.model,
                    temperature=agent.temperature,
                    runtime_config=agent.runtime_config,
                    tools=tool_schemas,
                    thinking_enabled=self._thinking_request_value(context),
                )
                if response.content and not response.tool_calls:
                    return {
                        "draft": response.content,
                        "draft_reasoning": response.reasoning_content or "",
                        "web_sources": web_sources,
                        "search_status": search_status,
                        "tool_outputs": [],
                        "tool_stats": {"total_calls": total_calls, "tools_used": tools_used},
                        "events": events,
                    }
                if response.tool_calls:
                    calls_this_round = response.tool_calls[:max_tool_calls - total_calls]
                    if not calls_this_round:
                        break
                    if response.reasoning_content and context.get("thinking_enabled"):
                        reasoning_content = response.reasoning_content.strip()
                        if reasoning_content:
                            events.append({"event": "reasoning_token", "data": {"content": f"{reasoning_content}\n\n"}})
                    assistant_msg = {"role": "assistant", "content": response.content, "tool_calls": calls_this_round}
                    if response.reasoning_content:
                        assistant_msg["reasoning_content"] = response.reasoning_content
                    messages.append(assistant_msg)
                    self._persist_intermediate_message(
                        context,
                        role="assistant",
                        content=response.content or "",
                        reasoning=response.reasoning_content or "",
                        tool_calls=calls_this_round,
                        meta={"node_id": node["id"], "round": _round, "kind": "tool_calls"},
                    )
                    # Limit tool_calls per round to remaining budget
                    for tc in calls_this_round:
                        if time.monotonic() - tool_loop_start > max_tool_wall_time:
                            break
                        func = tc["function"]
                        tool_name = func["name"]
                        try:
                            tool_args = json.loads(func.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            tool_args = {"input": func.get("arguments") or ""}
                        matching = next((t for t in bound_tools if t.name == tool_name), None)
                        started = time.monotonic()
                        if matching:
                            try:
                                result = execute_tool(matching, {"input": tool_args})
                                result["latency_ms"] = result.get("latency_ms", int((time.monotonic() - started) * 1000))
                                tool_content = result.get("content") or result.get("result_preview") or ""
                                event_data = tool_call_event(matching, result, input_preview=json.dumps(tool_args, ensure_ascii=False))
                                events.append({"event": "tool_call", "data": event_data})
                                if matching.type == "builtin_search":
                                    web_sources, search_status = self._merge_web_search_tool_result(web_sources, search_status, result)
                                    events.append({"event": "search_status", "data": self._search_status_event(search_status)})
                                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_content})
                                self._persist_intermediate_message(
                                    context,
                                    role="tool",
                                    content=tool_content,
                                    tool_call_id=tc["id"],
                                    tool_name=tool_name,
                                    meta={**event_data, "node_id": node["id"], "round": _round, "kind": "tool_result"},
                                )
                            except ValueError as exc:
                                error_result = {"tool": tool_name, "content": "", "result_preview": "", "latency_ms": int((time.monotonic() - started) * 1000), "error": str(exc)}
                                event_data = tool_call_event(matching, error_result, status="error", input_preview=json.dumps(tool_args, ensure_ascii=False), error_code="tool_error")
                                tool_content = f"Error: {exc}"
                                events.append({"event": "tool_call", "data": event_data})
                                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_content})
                                self._persist_intermediate_message(
                                    context,
                                    role="tool",
                                    content=tool_content,
                                    tool_call_id=tc["id"],
                                    tool_name=tool_name,
                                    meta={**event_data, "node_id": node["id"], "round": _round, "kind": "tool_result"},
                                )
                        else:
                            unknown_tool = type("_", (), {"id": None, "name": tool_name, "type": "unknown"})()
                            event_data = tool_call_event(unknown_tool, {"tool": tool_name, "content": "", "result_preview": "", "latency_ms": 0}, status="error", input_preview="{}", error_code="tool_not_found")
                            tool_content = f"Tool '{tool_name}' not found"
                            events.append({"event": "tool_call", "data": event_data})
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_content})
                            self._persist_intermediate_message(
                                context,
                                role="tool",
                                content=tool_content,
                                tool_call_id=tc["id"],
                                tool_name=tool_name,
                                meta={**event_data, "node_id": node["id"], "round": _round, "kind": "tool_result"},
                            )
                        total_calls += 1
                        tools_used.append(tool_name)

            # Max rounds reached — get final answer from accumulated context
            final = self.provider.chat(
                messages,
                model=agent.model,
                temperature=agent.temperature,
                runtime_config=agent.runtime_config,
                thinking_enabled=self._thinking_request_value(context),
            )
            return {
                "draft": final.content or "",
                "draft_reasoning": final.reasoning_content or "",
                "web_sources": web_sources,
                "search_status": search_status,
                "tool_outputs": [],
                "tool_stats": {"total_calls": total_calls, "tools_used": tools_used, "max_rounds_reached": True},
                "events": events,
            }
        if node_type == "LLM":
            if context.get("draft"):
                return self._llm_output(agent, context, context["draft"], reasoning=context.get("draft_reasoning") or "")
            messages = self._llm_messages(agent, context)
            draft = self.provider.chat(
                messages,
                model=agent.model,
                temperature=agent.temperature,
                runtime_config=agent.runtime_config,
                thinking_enabled=self._thinking_request_value(context),
            ).content or ""
            return self._llm_output(agent, context, draft)
        if node_type == "Answer":
            answer = (context.get("draft") or "").strip()
            if not answer:
                raise ValueError("Model returned an empty answer")
            return {"answer": answer, "citation_count": len([*context.get("sources", []), *context.get("web_sources", [])])}
        return {}

    def _stream_tool_node(self, agent, node: dict, context: dict):
        bound_tools = self._runtime_tools(agent, node, context)
        tool_policy = (agent.settings.get("tool_policy") or {})
        allowed_names = set(tool_policy.get("allowed_tool_names") or [])
        if allowed_names:
            bound_tools = [t for t in bound_tools if t.name in allowed_names or t.type == "builtin_search"]
        if not bound_tools:
            return {"tool_outputs": [], "tool_stats": {"total_calls": 0, "tools_used": []}}

        tool_schemas = [tool_schema_for_llm(t) for t in bound_tools]
        messages = self._llm_messages(agent, context)
        total_calls = 0
        tools_used: list[str] = []
        events = []
        web_sources = list(context.get("web_sources", []))
        search_status = dict(context.get("search_status") or {})
        max_tool_calls = MAX_TOOL_CALLS_PER_RUN
        max_tool_wall_time = MAX_TOOL_WALL_TIME_SECONDS
        tool_loop_start = time.monotonic()

        for _round in range(MAX_TOOL_ROUNDS_PER_RUN):
            if total_calls >= max_tool_calls:
                break
            if time.monotonic() - tool_loop_start > max_tool_wall_time:
                break
            response = yield from self._stream_chat_response(
                agent,
                messages,
                context,
                tools=tool_schemas,
                stream_content=True,
                live_content_with_tools=total_calls > 0,
            )
            reasoning_streamed = bool(response.reasoning_content and context.get("thinking_enabled"))
            if response.content and not response.tool_calls:
                return {
                    "draft": response.content,
                    "draft_streamed": True,
                    "draft_reasoning": response.reasoning_content or "",
                    "draft_reasoning_streamed": reasoning_streamed,
                    "web_sources": web_sources,
                    "search_status": search_status,
                    "tool_outputs": [],
                    "tool_stats": {"total_calls": total_calls, "tools_used": tools_used},
                    "events": events,
                }
            if response.tool_calls:
                calls_this_round = response.tool_calls[:max_tool_calls - total_calls]
                if not calls_this_round:
                    break
                assistant_msg = {"role": "assistant", "content": response.content, "tool_calls": calls_this_round}
                if response.reasoning_content:
                    assistant_msg["reasoning_content"] = response.reasoning_content
                messages.append(assistant_msg)
                self._persist_intermediate_message(
                    context,
                    role="assistant",
                    content=response.content or "",
                    reasoning=response.reasoning_content or "",
                    tool_calls=calls_this_round,
                    meta={"node_id": node["id"], "round": _round, "kind": "tool_calls"},
                )
                for tc in calls_this_round:
                    if time.monotonic() - tool_loop_start > max_tool_wall_time:
                        break
                    web_sources, search_status, tool_name = yield from self._execute_stream_tool_call(
                        tc,
                        bound_tools,
                        messages,
                        web_sources,
                        search_status,
                        context,
                        node,
                        _round,
                    )
                    total_calls += 1
                    tools_used.append(tool_name)
            else:
                break

        final = yield from self._stream_chat_response(agent, messages, context, stream_content=True)
        return {
            "draft": final.content or "",
            "draft_streamed": bool(final.content),
            "draft_reasoning": final.reasoning_content or "",
            "draft_reasoning_streamed": bool(final.reasoning_content and context.get("thinking_enabled")),
            "web_sources": web_sources,
            "search_status": search_status,
            "tool_outputs": [],
            "tool_stats": {"total_calls": total_calls, "tools_used": tools_used, "max_rounds_reached": True},
            "events": events,
        }

    def _stream_chat_response(
        self,
        agent,
        messages: list[dict],
        context: dict,
        *,
        tools: list[dict] | None = None,
        stream_content: bool = False,
        live_content_with_tools: bool = False,
    ):
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_call_builders: dict[int, dict] = {}
        final_tool_calls: list[dict] = []
        saw_tool_call = False
        should_stream_content_live = stream_content and (not tools or live_content_with_tools)
        for chunk in self.provider.chat_stream_events(
            messages,
            model=agent.model,
            temperature=agent.temperature,
            runtime_config=agent.runtime_config,
            tools=tools,
            thinking_enabled=self._thinking_request_value(context),
        ):
            if chunk.type == "reasoning":
                if not context.get("thinking_enabled"):
                    continue
                reasoning_chunks.append(chunk.content)
                yield {"event": "reasoning_token", "content": chunk.content}
            elif chunk.type == "content":
                content_chunks.append(chunk.content)
                if should_stream_content_live and not saw_tool_call:
                    yield {"event": "token", "content": chunk.content}
            elif chunk.type == "tool_call_delta":
                saw_tool_call = True
                self._merge_stream_tool_call_deltas(tool_call_builders, chunk.tool_calls or [])
            elif chunk.type == "tool_calls":
                saw_tool_call = True
                final_tool_calls = chunk.tool_calls or []
        if not final_tool_calls:
            final_tool_calls = self._finalize_stream_tool_calls(tool_call_builders)
        if stream_content and tools and not should_stream_content_live and not final_tool_calls and content_chunks:
            for content in content_chunks:
                yield {"event": "token", "content": content}
        return ChatResponse(
            content="".join(content_chunks) or None,
            reasoning_content="".join(reasoning_chunks),
            tool_calls=final_tool_calls or None,
        )

    def _merge_stream_tool_call_deltas(self, builders: dict[int, dict], deltas: list[dict]) -> None:
        for delta in deltas:
            if not isinstance(delta, dict):
                continue
            try:
                index = int(delta.get("index")) if delta.get("index") is not None else len(builders)
            except (TypeError, ValueError):
                index = len(builders)
            call = builders.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if delta.get("id"):
                call["id"] = str(delta.get("id"))
            if delta.get("type"):
                call["type"] = str(delta.get("type"))
            func = delta.get("function") or {}
            if not isinstance(func, dict):
                continue
            name_part = func.get("name")
            if name_part:
                name_part = str(name_part)
                current_name = call["function"].get("name") or ""
                if not current_name or name_part.startswith(current_name):
                    call["function"]["name"] = name_part
                elif not current_name.endswith(name_part):
                    call["function"]["name"] = current_name + name_part
            if func.get("arguments") is not None:
                call["function"]["arguments"] = (call["function"].get("arguments") or "") + str(func.get("arguments"))

    def _finalize_stream_tool_calls(self, builders: dict[int, dict]) -> list[dict]:
        calls = []
        for index in sorted(builders):
            call = builders[index]
            function = call.get("function") or {}
            name = function.get("name") or ""
            if not name:
                continue
            calls.append(
                {
                    "id": call.get("id") or f"call_{index}",
                    "type": call.get("type") or "function",
                    "function": {
                        "name": name,
                        "arguments": function.get("arguments") or "{}",
                    },
                }
            )
        return calls

    def _execute_stream_tool_call(
        self,
        tc: dict,
        bound_tools: list[Tool],
        messages: list[dict],
        web_sources: list[dict],
        search_status: dict,
        context: dict,
        node: dict,
        round_index: int,
    ) -> tuple[list[dict], dict, str]:
        func = tc["function"]
        tool_name = func["name"]
        try:
            tool_args = json.loads(func.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_args = {"input": func.get("arguments") or ""}
        matching = next((t for t in bound_tools if t.name == tool_name), None)
        start_data = self._tool_call_start_event(
            matching,
            tool_name=tool_name,
            tool_call_id=tc.get("id") or "",
            input_preview=json.dumps(tool_args, ensure_ascii=False),
        )
        yield {"event": "tool_call_start", "data": start_data}
        started = time.monotonic()
        if matching:
            try:
                result = execute_tool(matching, {"input": tool_args})
                result["latency_ms"] = result.get("latency_ms", int((time.monotonic() - started) * 1000))
                tool_content = result.get("content") or result.get("result_preview") or ""
                event_data = {
                    **tool_call_event(matching, result, input_preview=json.dumps(tool_args, ensure_ascii=False)),
                    "type": "tool_call_result",
                    "tool_call_id": tc.get("id") or "",
                }
                yield {"event": "tool_call_result", "data": event_data}
                if matching.type == "builtin_search":
                    web_sources, search_status = self._merge_web_search_tool_result(web_sources, search_status, result)
                    search_event = self._search_status_event(search_status)
                    search_event["tool_call_id"] = tc.get("id") or ""
                    yield {"event": "search_status", "data": search_event}
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_content})
                self._persist_intermediate_message(
                    context,
                    role="tool",
                    content=tool_content,
                    tool_call_id=tc["id"],
                    tool_name=tool_name,
                    meta={**event_data, "node_id": node["id"], "round": round_index, "kind": "tool_result"},
                )
            except ValueError as exc:
                error_result = {"tool": tool_name, "content": "", "result_preview": "", "latency_ms": int((time.monotonic() - started) * 1000), "error": str(exc)}
                event_data = {
                    **tool_call_event(matching, error_result, status="error", input_preview=json.dumps(tool_args, ensure_ascii=False), error_code="tool_error"),
                    "type": "tool_call_result",
                    "tool_call_id": tc.get("id") or "",
                }
                tool_content = f"Error: {exc}"
                yield {"event": "tool_call_result", "data": event_data}
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_content})
                self._persist_intermediate_message(
                    context,
                    role="tool",
                    content=tool_content,
                    tool_call_id=tc["id"],
                    tool_name=tool_name,
                    meta={**event_data, "node_id": node["id"], "round": round_index, "kind": "tool_result"},
                )
        else:
            unknown_tool = type("_", (), {"id": None, "name": tool_name, "type": "unknown"})()
            event_data = {
                **tool_call_event(unknown_tool, {"tool": tool_name, "content": "", "result_preview": "", "latency_ms": 0}, status="error", input_preview="{}", error_code="tool_not_found"),
                "type": "tool_call_result",
                "tool_call_id": tc.get("id") or "",
            }
            tool_content = f"Tool '{tool_name}' not found"
            yield {"event": "tool_call_result", "data": event_data}
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_content})
            self._persist_intermediate_message(
                context,
                role="tool",
                content=tool_content,
                tool_call_id=tc["id"],
                tool_name=tool_name,
                meta={**event_data, "node_id": node["id"], "round": round_index, "kind": "tool_result"},
            )
        return web_sources, search_status, tool_name

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
        draft = context.get("draft", "")
        if draft:
            draft_reasoning = context.get("draft_reasoning") or ""
            if draft_reasoning and context.get("thinking_enabled") and not context.get("draft_reasoning_streamed"):
                yield {"event": "reasoning_token", "content": draft_reasoning}
            if not context.get("draft_streamed"):
                # Draft already produced by the tool-calling loop — stream as chunked text
                for index in range(0, len(draft), 24):
                    yield {"event": "token", "content": draft[index : index + 24]}
            return self._llm_output(agent, context, draft, reasoning=draft_reasoning)
        messages = self._llm_messages(agent, context)
        chunks = []
        reasoning_chunks = []
        for chunk in self.provider.chat_stream_events(
            messages,
            model=agent.model,
            temperature=agent.temperature,
            runtime_config=agent.runtime_config,
            thinking_enabled=self._thinking_request_value(context),
        ):
            if chunk.type == "reasoning":
                if not context.get("thinking_enabled"):
                    continue
                reasoning_chunks.append(chunk.content)
                yield {"event": "reasoning_token", "content": chunk.content}
            elif chunk.type == "content":
                chunks.append(chunk.content)
                yield {"event": "token", "content": chunk.content}
        draft = "".join(chunks)
        return self._llm_output(agent, context, draft, reasoning="".join(reasoning_chunks))

    def _thinking_request_value(self, context: dict) -> bool | None:
        status = context.get("thinking_status") or {}
        if status.get("type") not in {"native", "prompt"}:
            return None
        return bool(status.get("enabled"))

    def _llm_messages(self, agent, context: dict) -> list[dict]:
        source_text = "\n".join(f"- {item['title']}: {item['snippet']}" for item in context.get("sources", []))
        web_source_text = self._web_source_text(context.get("web_sources", []))
        tool_text = "\n".join(f"- {item['tool']}: {item['content']}" for item in context.get("tool_outputs", []))
        variable_text = "\n".join(f"- {key}: {value}" for key, value in context.get("variables", {}).items())
        attachment_text = self._attachment_text(context.get("uploads", []))
        thinking_blocks = []
        thinking_msgs = self._thinking_messages(context)
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
            f"Web search results for this turn:\n{web_source_text or 'None'}",
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
        messages = [{"role": "system", "content": system_content}]
        history_messages = self._history_messages_for_llm(context)
        messages.extend(history_messages)
        if not self._history_contains_current_message(context):
            messages.append({"role": "user", "content": self._user_content(context["input"], context.get("uploads", []))})
        return messages

    def _history_messages_for_llm(self, context: dict) -> list[dict]:
        messages = []
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
                    assistant_message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                    if item.get("reasoning") and (item.get("meta") or {}).get("requires_reasoning_replay"):
                        assistant_message["reasoning_content"] = item.get("reasoning")
                    messages.append(assistant_message)
                    for tool_item in tool_messages:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_item.get("tool_call_id"),
                                "content": tool_item.get("content") or "",
                            }
                        )
                index = next_index
                continue
            if content:
                message = {"role": role, "content": content}
                if role == "assistant" and item.get("reasoning") and (item.get("meta") or {}).get("requires_reasoning_replay"):
                    message["reasoning_content"] = item.get("reasoning")
                messages.append(message)
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
                "user_model_config_id": agent.user_model_config_id,
            }

        user_model_config = self._user_model_config(user_id, source["user_model_config_id"])
        runtime_config = user_model_runtime_config(user_model_config) if user_model_config else None

        # Merge Agent-bound Skills
        agent_skill_rows = (
            self.db.query(AgentSkill)
            .filter(
                AgentSkill.agent_id == agent.id,
                AgentSkill.enabled.is_(True),
            )
            .all()
        )
        if agent_skill_rows:
            skill_ids = [row.skill_id for row in agent_skill_rows]
            skills = (
                self.db.query(Skill)
                .filter(Skill.id.in_(skill_ids), Skill.enabled.is_(True))
                .all()
            )
            priority_map = {row.skill_id: row.priority for row in agent_skill_rows}
            for skill in sorted(skills, key=lambda s: priority_map.get(s.id, 0), reverse=True):
                # Merge system_prompt
                source["system_prompt"] += (
                    f"\n\n## Skill: {skill.name}\n{skill.system_prompt}"
                )
                # Merge tools (deduplicate)
                skill_tool_ids = [
                    st.tool_id
                    for st in self.db.query(SkillTool)
                    .filter(SkillTool.skill_id == skill.id)
                    .all()
                ]
                source["tool_ids"] = list(
                    dict.fromkeys(source["tool_ids"] + skill_tool_ids)
                )
                # Merge knowledge bases (deduplicate)
                skill_kb_ids = [
                    skb.knowledge_base_id
                    for skb in self.db.query(SkillKnowledgeBase)
                    .filter(SkillKnowledgeBase.skill_id == skill.id)
                    .all()
                ]
                source["knowledge_base_ids"] = list(
                    dict.fromkeys(source["knowledge_base_ids"] + skill_kb_ids)
                )

        return SimpleNamespace(
            id=agent.id,
            workspace_id=agent.workspace_id,
            system_prompt=source["system_prompt"],
            model_id=source["model_id"],
            user_model_config_id=source["user_model_config_id"],
            model=(runtime_config or {}).get("chat_model") or source["model"],
            temperature=source["temperature"],
            knowledge_base_ids=source["knowledge_base_ids"],
            tool_ids=source["tool_ids"],
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
            },
        )

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

    def _thinking_status(self, model: ModelConfig | UserModelConfig | None, requested: bool | None) -> dict:
        reasoning_type = str(getattr(model, "reasoning_type", "none") or "none")
        if reasoning_type not in {"native", "prompt", "none"}:
            reasoning_type = "none"
        supports_reasoning = bool(getattr(model, "supports_reasoning", False)) and reasoning_type != "none"
        label = str(getattr(model, "reasoning_label", "") or self._reasoning_label(reasoning_type))

        if not requested:
            return {
                "enabled": False,
                "requested": False,
                "type": reasoning_type,
                "label": label,
                "reason": "not_requested",
            }
        if not supports_reasoning:
            return {
                "enabled": False,
                "requested": True,
                "type": "none",
                "label": self._reasoning_label("none"),
                "reason": "model_not_supported",
            }
        return {
            "enabled": True,
            "requested": True,
            "type": reasoning_type,
            "label": label,
            "reason": "enabled",
        }

    def _thinking_messages(self, context: dict) -> list[dict]:
        status = context.get("thinking_status") or {}
        if not status.get("enabled"):
            return []
        if status.get("type") == "prompt":
            return [
                {
                    "role": "system",
                    "content": (
                        "本轮已开启深度思考模式，但当前模型使用提示词增强，不是原生推理。"
                        "请先进行更周全的分析，检查关键假设、约束、风险和反例，再给出清晰答案。"
                        "不要输出隐藏推理链，只输出必要的结论、依据和可执行步骤。"
                    ),
                }
            ]
        return [
            {
                "role": "system",
                "content": "本轮已开启原生深度思考能力。请给出经过审慎推理后的答案，不要输出隐藏推理链。",
            }
        ]

    @staticmethod
    def _reasoning_label(reasoning_type: str) -> str:
        return {"native": "深度思考", "prompt": "提示词增强", "none": "不支持"}.get(reasoning_type, "不支持")

    def _search_status(self, query: str, requested: bool | None) -> dict:
        runtime = web_search_service.web_search_status()
        provider = runtime.get("provider", "duckduckgo_html")
        if not requested:
            return {
                "enabled": False,
                "requested": False,
                "query": query,
                "provider": provider,
                "matched_results": 0,
                "sources_emitted": False,
                "items": [],
                "sources": [],
                "reason": "not_requested",
            }
        if not runtime.get("configured"):
            return {
                "enabled": False,
                "requested": True,
                "query": query,
                "provider": provider,
                "matched_results": 0,
                "sources_emitted": False,
                "items": [],
                "sources": [],
                "reason": "web_search_unavailable",
            }
        return {
            "enabled": True,
            "requested": True,
            "query": query,
            "provider": provider,
            "matched_results": 0,
            "sources_emitted": False,
            "items": [],
            "sources": [],
            "reason": "tool_available",
        }

    def _search_status_event(self, status: dict) -> dict:
        return {key: value for key, value in status.items() if key != "sources"}

    def _merge_web_search_tool_result(self, current_sources: list[dict], status: dict, result: dict) -> tuple[list[dict], dict]:
        result_json = result.get("result_json") if isinstance(result.get("result_json"), dict) else {}
        items = result_json.get("items") or []
        next_sources = [dict(item) for item in current_sources]
        new_sources = web_search_service.search_items_as_sources(items)
        offset = len(next_sources)
        for index, source in enumerate(new_sources, start=offset + 1):
            source["source_id"] = f"web-{index}"
            source["chunk_id"] = f"web-search-{index}"
            next_sources.append(source)
        next_status = {
            **status,
            "enabled": bool(next_sources),
            "requested": True,
            "query": result_json.get("query") or status.get("query") or "",
            "provider": result_json.get("provider") or status.get("provider") or "duckduckgo_html",
            "matched_results": len(next_sources),
            "sources_emitted": bool(next_sources),
            "items": [*(status.get("items") or []), *items],
            "sources": next_sources,
            "latency_ms": result.get("latency_ms", status.get("latency_ms", 0)),
            "reason": "tool_called" if next_sources else "no_results",
        }
        return next_sources, next_status

    def _web_source_text(self, sources: list[dict]) -> str:
        lines = []
        for index, item in enumerate(sources, start=1):
            title = item.get("title") or f"Result {index}"
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            lines.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")
        return "\n\n".join(lines)

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

    def _session_memory(self, session_id: int) -> SessionMemory | None:
        return self.db.query(SessionMemory).filter(SessionMemory.session_id == session_id).first()

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

    def _update_session_memory(self, session_id: int, user_message: str, answer: str, max_messages: int) -> None:
        memory = self._session_memory(session_id)
        if not memory:
            memory = SessionMemory(session_id=session_id, summary="", message_count=0)
            self.db.add(memory)
        memory.message_count += 2
        
        # Try parsing structured dialogue turns
        try:
            dialogue_turns = json.loads(memory.summary) if memory.summary else []
            if not isinstance(dialogue_turns, list):
                dialogue_turns = []
        except Exception:
            # Fallback: parse legacy plain text formatted summary
            dialogue_turns = []
            if memory.summary.strip():
                raw_turns = memory.summary.split("\n===\n")
                for turn_text in raw_turns:
                    if "助手：" in turn_text:
                        parts = turn_text.split("助手：", 1)
                        u_part = parts[0].replace("用户：", "").strip()
                        a_part = parts[1].strip()
                        dialogue_turns.append({"user": u_part, "assistant": a_part})

        # Append current dialogue turn
        dialogue_turns.append({
            "user": user_message.strip(),
            "assistant": answer.strip()
        })
        
        # max_messages represents dialogue turns (one turn has user + assistant, so max_turns = max_messages // 2)
        max_turns = max(1, max_messages // 2)
        truncated_turns = dialogue_turns[-max_turns:]
        
        # Re-serialize to JSON summary
        serialized = json.dumps(truncated_turns, ensure_ascii=False)
        # Limit token usage for extra long assistant codes/texts
        if len(serialized) > 2000:
            for turn in truncated_turns:
                if len(turn["assistant"]) > 500:
                    turn["assistant"] = turn["assistant"][:500] + "...(此回答过长已截断)..."
            serialized = json.dumps(truncated_turns, ensure_ascii=False)
            
        memory.summary = serialized

