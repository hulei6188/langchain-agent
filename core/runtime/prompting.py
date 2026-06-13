from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core.runtime.dsml import (
    contains_leaked_tool_markup,
    dsml_preview,
    strip_complete_dsml_tool_call_blocks,
)
from core.runtime.message_utils import normalize_langchain_tool_calls
from core.runtime.skill_selection import loaded_skill_text, skill_manifest_text
from core.runtime.status import thinking_messages, web_source_text


logger = logging.getLogger(__name__)


def build_llm_messages(agent, context: dict) -> list[BaseMessage]:
    source_text = "\n".join(f"- {item['title']}: {item['snippet']}" for item in context.get("sources", []))
    web_sources_text = web_source_text(context.get("web_sources", []))
    tool_text = "\n".join(f"- {item['tool']}: {item['content']}" for item in context.get("tool_outputs", []))
    variable_text = "\n".join(f"- {key}: {value}" for key, value in context.get("variables", {}).items())
    attachment_content = attachment_text(context.get("uploads", []))
    skill_manifest_content = skill_manifest_text(context.get("skill_manifest") or [])
    loaded_skill_content = loaded_skill_text(context.get("loaded_skills") or [])
    thinking_blocks = [msg["content"] for msg in thinking_messages(context)]
    search_instruction = ""
    if context.get("search_enabled"):
        search_instruction = (
            "本轮联网搜索工具可用，但不是必选。只有当问题需要最新信息、外部事实、网页资料、天气、价格、新闻或可变信息时才调用；"
            "简单算术、常识推理、翻译、代码解释、当前会话内容总结等不需要联网搜索的问题，请直接回答。"
        )

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
        f"会话记忆摘要：\n{format_memory_summary(context.get('memory_summary') or '')}",
        f"本轮附件上下文：\n{attachment_content or '无'}",
        f"Long-term Agent memory:\n{context.get('profile_memory') or 'None'}",
    ]
    system_content = "\n\n".join(part for part in system_parts if part.strip())
    max_system_chars = 100_000
    if len(system_content) > max_system_chars:
        system_content = system_content[:max_system_chars] + "\n\n[上下文已截断以避免超出模型上下文窗口限制]"
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    messages.extend(history_messages_for_llm(context))
    if not history_contains_current_message(context):
        messages.append(HumanMessage(content=user_content(context["input"], context.get("uploads", []))))
    return messages


def format_memory_summary(raw_summary: str) -> str:
    if not raw_summary.strip():
        return "无"
    try:
        turns = json.loads(raw_summary)
        if isinstance(turns, list):
            return "\n".join(f"用户：{turn['user']}\n助手：{turn['assistant']}" for turn in turns)
        return raw_summary
    except Exception:
        return raw_summary


def history_messages_for_llm(context: dict) -> list[BaseMessage]:
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
            content = user_content(context["input"], context.get("uploads", []))
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
                        tool_calls=normalize_langchain_tool_calls(tool_calls),
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


def history_contains_current_message(context: dict) -> bool:
    current_message_id = context.get("current_message_id")
    if not current_message_id:
        return False
    return any(item.get("id") == current_message_id for item in context.get("history_messages") or [])


def llm_output(agent, context: dict, draft: str, *, reasoning: str = "", last_chat_mock: bool = False) -> dict:
    return {
        "draft": draft,
        "used_memory": bool(context.get("memory_summary")),
        "used_profile_memory": bool(context.get("profile_memory")),
        "attachment_count": len(context.get("uploads", [])),
        "history_message_count": len(context.get("history_messages") or []),
        "model": agent.model,
        "mock": last_chat_mock,
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


def attachment_text(uploads: list) -> str:
    lines = []
    for upload in uploads:
        if upload.kind == "document":
            lines.append(f"[{upload.filename}]\n{upload.text[:6000]}")
        elif upload.kind == "image":
            lines.append(f"[Image: {upload.filename}]")
    return "\n\n".join(lines)


def user_content(text: str, uploads: list):
    image_uploads = [upload for upload in uploads if upload.kind == "image"]
    if not image_uploads:
        return text
    content = [{"type": "text", "text": text}]
    for upload in image_uploads:
        content.append({"type": "image_url", "image_url": {"url": upload.data_url}})
    return content


def merge_variables(definitions: list[dict], provided: dict) -> dict:
    merged = {}
    for definition in definitions:
        key = definition.get("key")
        if key:
            merged[key] = provided.get(key, definition.get("default_value"))
    for key, value in provided.items():
        if key not in merged:
            merged[key] = value
    return merged
