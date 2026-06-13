from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from core.db.models import Feedback, Message, Run, RunEvent, Session as ChatSession, SessionMemory


def cleanup_stale_session_runs(db: Session, session_id: int) -> None:
    stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    stale_runs = (
        db.query(Run)
        .filter(
            Run.session_id == session_id,
            Run.status == "running",
            Run.started_at < stale_cutoff,
        )
        .all()
    )
    for stale in stale_runs:
        stale.status = "failed"
        stale.completed_at = datetime.now(timezone.utc)
    if stale_runs:
        db.commit()


def active_run_payload(db: Session, session_id: int) -> dict | None:
    cleanup_stale_session_runs(db, session_id)
    active_run = (
        db.query(Run)
        .filter(Run.session_id == session_id, Run.status == "running")
        .order_by(Run.started_at.desc())
        .first()
    )
    if not active_run:
        return None
    return {
        "id": active_run.id,
        "status": active_run.status,
        "started_at": active_run.started_at.isoformat() if active_run.started_at else None,
    }


def delete_chat_session(db: Session, session: ChatSession) -> None:
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
        db.query(RunEvent).filter(RunEvent.run_id.in_(run_ids)).delete(synchronize_session=False)
    db.query(SessionMemory).filter(SessionMemory.session_id == session.id).delete(synchronize_session=False)
    db.query(Message).filter(Message.session_id == session.id).delete(synchronize_session=False)
    db.query(Run).filter(Run.session_id == session.id).delete(synchronize_session=False)
    db.delete(session)
    db.commit()


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
