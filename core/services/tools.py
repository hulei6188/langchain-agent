from __future__ import annotations

import json
import re
import urllib.parse

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.db.models import Agent, AgentTool, Tool
from core.integrations.mcp_client import MCPClientError, discover_mcp_tools as discover_mcp_tools_client, discover_stdio_mcp_tools
from core.security.api_keys import decrypt_api_key, encrypt_api_key
from core.services import web_search as web_search_service
from core.services.builtin_tools import (
    BUILTIN_TOOLS,
    SafeEvalVisitor,
    _exec_arxiv_search,
    _exec_advice_slip,
    _exec_bored_activity,
    _exec_calculator,
    _exec_character_counter,
    _exec_current_time,
    _exec_currency_converter,
    _exec_diff_checker,
    _exec_horoscope,
    _exec_image_search,
    _exec_ip_lookup,
    _exec_joke_generator,
    _exec_news_search,
    _exec_password_generator,
    _exec_qr_generator,
    _exec_run_powershell,
    _exec_url_shortener,
    _exec_uuid_generator,
    _exec_weather_lookup,
    _exec_web_reader,
    _exec_wikipedia,
    clean_html,
)
from core.services.http_tools import execute_http_tool as _execute_http_tool
from core.services.http_tools import validate_safe_https_url as _validate_safe_https_url
from core.services.mcp_tools import (
    DEFAULT_MCP_TIMEOUT_SECONDS,
    MAX_MCP_TIMEOUT_SECONDS,
    execute_mcp_tool as _execute_mcp_tool,
    mcp_input_schema_value as _mcp_input_schema_value,
    mcp_request_target as _mcp_request_target,
    mcp_transport_value as _mcp_transport_value,
    suggest_mcp_tool_name as _suggest_mcp_tool_name,
    tool_mcp_config as _tool_mcp_config,
    validate_safe_mcp_url as _validate_safe_mcp_url,
)
from core.services.tool_utils import dict_value as _dict_value
from core.services.tool_utils import preview as _preview


TOOL_TYPES = {"builtin", "builtin_search", "http", "mcp"}
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
AUTH_TYPES = {"none", "bearer", "header", "query"}
DEFAULT_HTTP_TIMEOUT_SECONDS = 10
MAX_HTTP_TIMEOUT_SECONDS = 30


def tool_payload(tool: Tool) -> dict:
    mcp = _tool_mcp_config(tool)
    return {
        "id": tool.id,
        "type": tool.type,
        "name": tool.name,
        "label": tool.label,
        "description": tool.description,
        "server_label": tool.server_label or "",
        "enabled": tool.enabled,
        "method": tool.method,
        "url": tool.url,
        "headers_schema": tool.headers_schema or {},
        "query_schema": tool.query_schema or {},
        "body_schema": tool.body_schema or {},
        "auth": {
            "type": tool.auth_type,
            "header_name": tool.auth_header_name or None,
            "query_name": tool.auth_query_name or None,
            "has_secret": bool(tool.encrypted_secret),
        },
        "response_path": tool.response_path,
        "timeout_seconds": tool.timeout_seconds,
        "search_options": tool.search_options or {},
        "mcp": mcp,
        "created_by": tool.user_id,
        "created_at": tool.created_at.isoformat() if tool.created_at else None,
        "updated_at": tool.updated_at.isoformat() if tool.updated_at else None,
    }


def list_available_tools(db: Session, *, workspace_id: int, user_id: int) -> list[Tool]:
    return (
        db.query(Tool)
        .filter(
            or_(Tool.workspace_id.is_(None), Tool.workspace_id == workspace_id),
            or_(Tool.user_id.is_(None), Tool.user_id == user_id),
        )
        .order_by(Tool.id.asc())
        .all()
    )


def get_accessible_tool(db: Session, *, workspace_id: int, user_id: int, tool_id: int) -> Tool | None:
    return (
        db.query(Tool)
        .filter(
            Tool.id == tool_id,
            or_(Tool.workspace_id.is_(None), Tool.workspace_id == workspace_id),
            or_(Tool.user_id.is_(None), Tool.user_id == user_id),
        )
        .first()
    )


def create_tool(db: Session, *, workspace_id: int, user_id: int, payload: dict) -> Tool:
    data = _tool_fields(payload)
    if _tool_name_exists(db, workspace_id=workspace_id, user_id=user_id, name=data["name"]):
        raise ValueError("Tool name already exists")
    secret = data.pop("secret", None)
    tool = Tool(workspace_id=workspace_id, user_id=user_id, encrypted_secret=encrypt_api_key(secret) if secret else "", **data)
    db.add(tool)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise _tool_persistence_error(exc) from exc
    db.refresh(tool)
    return tool


def update_tool(db: Session, *, tool: Tool, payload: dict) -> Tool:
    if tool.user_id is None:
        raise ValueError("Built-in tools cannot be modified")
    data = _tool_fields(payload, partial=True, existing=tool)
    if "name" in data and data["name"] != tool.name and _tool_name_exists(db, workspace_id=tool.workspace_id, user_id=tool.user_id, name=data["name"]):
        raise ValueError("Tool name already exists")
    secret = data.pop("secret", None)
    clear_secret = bool(data.pop("clear_secret", False))
    for key, value in data.items():
        setattr(tool, key, value)
    if clear_secret:
        tool.encrypted_secret = ""
    elif secret is not None:
        tool.encrypted_secret = encrypt_api_key(secret)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise _tool_persistence_error(exc) from exc
    db.refresh(tool)
    return tool


def delete_tool(db: Session, *, tool: Tool) -> None:
    if tool.user_id is None:
        raise ValueError("Built-in tools cannot be deleted")
    if db.query(AgentTool.id).filter(AgentTool.tool_id == tool.id).first():
        raise ValueError("Tool is in use")
    db.delete(tool)
    db.commit()


def validate_tool_ids(db: Session, *, workspace_id: int, user_id: int, tool_ids: list[int]) -> None:
    for tool_id in tool_ids:
        tool = get_accessible_tool(db, workspace_id=workspace_id, user_id=user_id, tool_id=tool_id)
        if not tool or not tool.enabled:
            raise ValueError("Tool is not available")


def discover_mcp_tools(payload: dict, *, existing_tool: Tool | None = None) -> list[dict]:
    data = {key: value for key, value in payload.items() if value is not None}
    existing_mcp = _tool_mcp_config(existing_tool)
    mcp_config = _auth_value(data.get("mcp")) if "mcp" in data else {}
    transport = _mcp_transport_value(
        mcp_config.get("transport") if mcp_config else data.get("transport"),
        url=str(data.get("url", "")).strip(),
        existing=existing_mcp.get("transport"),
    )

    # ── stdio transport ──
    if transport == "stdio":
        command = str(mcp_config.get("command") or "").strip()
        if not command:
            raise ValueError("MCP stdio transport requires a command")
        stdio_args = list(mcp_config.get("args") or []) if isinstance(mcp_config.get("args"), list) else []
        stdio_env = dict(mcp_config.get("env")) if isinstance(mcp_config.get("env"), dict) else None
        stdio_cwd = str(mcp_config.get("cwd") or "").strip() or None
        timeout = _timeout_seconds(data.get("timeout_seconds"), default=DEFAULT_MCP_TIMEOUT_SECONDS, maximum=MAX_MCP_TIMEOUT_SECONDS, label="MCP timeout")
        try:
            remote_tools = discover_stdio_mcp_tools(
                command,
                args=stdio_args,
                env=stdio_env,
                cwd=stdio_cwd,
                timeout_seconds=timeout,
            )
        except MCPClientError as exc:
            raise ValueError(f"MCP tool discovery failed: {_preview(str(exc), 300)}") from exc
        server_label = str(data.get("server_label") or "").strip() or command
        items = []
        for remote in remote_tools:
            remote_name = str(remote.get("name") or "").strip()
            if not remote_name:
                continue
            items.append(
                {
                    "name": _suggest_mcp_tool_name(remote_name),
                    "label": remote_name,
                    "description": str(remote.get("description") or "").strip(),
                    "server_label": server_label,
                    "mcp": {
                        "transport": transport,
                        "command": command,
                        "args": stdio_args,
                        "env": stdio_env,
                        "cwd": stdio_cwd,
                        "tool_name": remote_name,
                        "input_schema": _mcp_input_schema_value(remote.get("input_schema")),
                    },
                }
            )
        return items

    # ── HTTP / SSE transport ──
    url = str(data.get("url", "")).strip()
    _validate_safe_mcp_url(url)
    auth = _auth_value(data.get("auth"))
    auth_type = str(auth.get("type", "none")).strip() or "none"
    if auth_type not in AUTH_TYPES:
        raise ValueError("Unsupported auth type")
    timeout = _timeout_seconds(data.get("timeout_seconds"), default=DEFAULT_MCP_TIMEOUT_SECONDS, maximum=MAX_MCP_TIMEOUT_SECONDS, label="MCP timeout")
    secret = str(auth.get("secret") or "").strip()
    if not secret and existing_tool and existing_tool.encrypted_secret and auth_type != "none":
        secret = decrypt_api_key(existing_tool.encrypted_secret)
    target_url, headers = _mcp_request_target(
        url,
        auth_type=auth_type,
        auth_header_name=str(auth.get("header_name") or "Authorization").strip() or "Authorization",
        auth_query_name=str(auth.get("query_name") or "").strip(),
        secret=secret,
    )
    try:
        remote_tools = discover_mcp_tools_client(target_url, headers=headers, transport=transport, timeout_seconds=timeout)
    except MCPClientError as exc:
        raise ValueError(f"MCP tool discovery failed: {_preview(str(exc), 300)}") from exc
    server_label = str(data.get("server_label") or "").strip() or urllib.parse.urlparse(url).netloc
    items = []
    for remote in remote_tools:
        remote_name = str(remote.get("name") or "").strip()
        if not remote_name:
            continue
        items.append(
            {
                "name": _suggest_mcp_tool_name(remote_name),
                "label": remote_name,
                "description": str(remote.get("description") or "").strip(),
                "server_label": server_label,
                "mcp": {
                    "transport": transport,
                    "tool_name": remote_name,
                    "input_schema": _mcp_input_schema_value(remote.get("input_schema")),
                },
            }
        )
    return items


def test_tool(tool: Tool, *, input_data: dict | None = None, body=None) -> dict:
    started = time.monotonic()
    try:
        output = build_langchain_tool(tool, body=body).invoke(input_data or {})
        return {
            "ok": True,
            "tool_id": tool.id,
            "tool_type": tool.type,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "status_code": output.get("status_code"),
            "content_type": output.get("content_type"),
            "result_preview": output.get("result_preview", ""),
            "result_json": output.get("result_json"),
        }
    except ValueError as exc:
        message = str(exc)
        error_code = _error_code(message)
        return {
            "ok": False,
            "tool_id": tool.id,
            "tool_type": tool.type,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error_code": error_code,
            "message": message,
            **({"hint": _tool_timeout_hint(tool)} if error_code in {"timeout", "mcp_timeout"} else {}),
        }


def build_langchain_tool(
    tool: Tool,
    *,
    session_key: str = "",
    agent_workdir: str | None = None,
    body=None,
) -> StructuredTool:
    if not tool.enabled:
        raise ValueError("Tool is disabled")

    def invoke_tool(**kwargs):
        return _invoke_tool_backend(
            tool,
            {
                "input": kwargs,
                "body": body,
                "_session_key": session_key,
                "_agent_workdir": agent_workdir,
            },
        )

    return StructuredTool.from_function(
        func=invoke_tool,
        name=tool.name,
        description=_tool_description(tool),
        args_schema=_tool_args_model(tool),
    )


def _invoke_tool_backend(tool: Tool, context: dict) -> dict:
    if not tool.enabled:
        raise ValueError("Tool is disabled")
    if tool.type == "builtin":
        return _execute_builtin_tool(tool, context)
    if tool.type == "builtin_search":
        return _execute_builtin_search(tool, context)
    if tool.type == "http":
        return _execute_http_tool(tool, context)
    if tool.type == "mcp":
        return _execute_mcp_tool(tool, context)
    raise ValueError("Unsupported tool type")


def _tool_description(tool: Tool) -> str:
    if tool.type == "builtin_search":
        return (
            "Search the public web for current, time-sensitive, or external factual information. "
            "Use this only when the answer depends on recent events, live data, URLs, news, prices, weather, "
            "or facts that may have changed. Do not use it for arithmetic, simple reasoning, translation, "
            "summarizing the current conversation, or stable common knowledge."
        )
    return tool.description or tool.label or tool.name


def _tool_args_model(tool: Tool):
    schema = _tool_parameters_schema(tool)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") if isinstance(schema.get("required"), list) else [])
    fields = {}
    for name, spec in properties.items():
        if not isinstance(name, str) or not name:
            continue
        spec = spec if isinstance(spec, dict) else {}
        field_type = _json_schema_python_type(spec)
        default = ... if name in required else None
        fields[name] = (
            field_type,
            Field(default, description=str(spec.get("description") or name)),
        )
    if not fields:
        fields["input"] = (str, Field(..., description="传递给工具的输入文本"))
    model_name = f"{re.sub(r'[^0-9A-Za-z_]+', '_', tool.name).strip('_') or 'Tool'}Input"
    return create_model(model_name, **fields)


def _json_schema_python_type(spec: dict):
    field_type = spec.get("type")
    if field_type == "integer":
        return int
    if field_type == "number":
        return float
    if field_type == "boolean":
        return bool
    if field_type == "array":
        return list
    if field_type == "object":
        return dict
    return str


def tool_call_event(tool: Tool, result: dict, *, status: str = "success", input_preview: str = "", error_code: str | None = None) -> dict:
    return {
        "tool_id": tool.id,
        "tool_name": tool.name,
        "tool_type": tool.type,
        "status": status,
        "latency_ms": result.get("latency_ms", 0),
        "input_preview": _preview(input_preview),
        "result_preview": _preview(result.get("result_preview") or result.get("content") or ""),
        "error_code": error_code,
    }


def _tool_fields(payload: dict, *, partial: bool = False, existing: Tool | None = None) -> dict:
    data = {key: value for key, value in payload.items() if value is not None}
    current_type = existing.type if existing else "http"
    tool_type = str(data.get("type", current_type)).strip() if ("type" in data or not partial) else current_type
    if tool_type not in TOOL_TYPES:
        raise ValueError("Unsupported tool type")
    if tool_type == "builtin":
        raise ValueError("Built-in tools can only be managed by the system")

    result: dict = {}
    if "type" in data or not partial:
        result["type"] = tool_type
    for key in ["name", "label", "description"]:
        if key in data or (not partial and key in {"name", "label"}):
            value = str(data.get(key, "")).strip()
            if key in {"name", "label"} and not value:
                raise ValueError("Invalid tool config")
            result[key] = value
    if "server_label" in data or (not partial and tool_type == "mcp"):
        result["server_label"] = str(data.get("server_label", existing.server_label if existing else "")).strip()
    for key in ["headers_schema", "query_schema", "body_schema", "search_options"]:
        if key in data:
            result[key] = _dict_value(data[key])
        elif not partial and key in {"headers_schema", "query_schema", "body_schema", "search_options"}:
            result[key] = {}
    if "enabled" in data or not partial:
        result["enabled"] = bool(data.get("enabled", True))

    if tool_type == "http":
        method = str(data.get("method", existing.method if existing else "GET")).strip().upper()
        if method not in HTTP_METHODS:
            raise ValueError("Unsupported HTTP method")
        result["method"] = method
        if "url" in data or not partial:
            url = str(data.get("url", existing.url if existing else "")).strip()
            _validate_safe_https_url(url)
            result["url"] = url
        auth = _auth_value(data.get("auth")) if "auth" in data else {}
        auth_type = str(auth.get("type", existing.auth_type if existing else "none")).strip() or "none"
        if auth_type not in AUTH_TYPES:
            raise ValueError("Unsupported auth type")
        result["auth_type"] = auth_type
        result["auth_header_name"] = str(auth.get("header_name", existing.auth_header_name if existing else "Authorization")).strip() or "Authorization"
        result["auth_query_name"] = str(auth.get("query_name", existing.auth_query_name if existing else "")).strip()
        if "secret" in auth:
            secret = str(auth.get("secret") or "").strip()
            if not secret:
                raise ValueError("Tool secret cannot be empty")
            result["secret"] = secret
        if auth.get("clear_secret"):
            result["clear_secret"] = True
        result["response_path"] = str(data.get("response_path", existing.response_path if existing else "$")).strip() or "$"
        timeout = _timeout_seconds(
            data.get("timeout_seconds", existing.timeout_seconds if existing else DEFAULT_HTTP_TIMEOUT_SECONDS),
            default=DEFAULT_HTTP_TIMEOUT_SECONDS,
            maximum=MAX_HTTP_TIMEOUT_SECONDS,
            label="HTTP timeout",
        )
        result["timeout_seconds"] = timeout
        result["schema"] = {}
    elif tool_type == "mcp":
        existing_mcp = _tool_mcp_config(existing)
        mcp = _auth_value(data.get("mcp")) if "mcp" in data else {}
        transport = _mcp_transport_value(
            mcp.get("transport"),
            url=str(data.get("url") or (existing.url if existing else "")),
            existing=existing_mcp.get("transport"),
        )

        if transport == "stdio":
            # stdio transport: URL is optional, command is required
            if "url" in data or (not partial and not existing):
                result["url"] = str(data.get("url", existing.url if existing else "")).strip()
            command = str(mcp.get("command") or "").strip()
            if not command:
                raise ValueError("MCP stdio transport requires a command in mcp schema")
            timeout = _timeout_seconds(
                data.get("timeout_seconds", existing.timeout_seconds if existing else DEFAULT_MCP_TIMEOUT_SECONDS),
                default=DEFAULT_MCP_TIMEOUT_SECONDS,
                maximum=MAX_MCP_TIMEOUT_SECONDS,
                label="MCP timeout",
            )
            result["timeout_seconds"] = timeout
            if "server_label" not in result:
                result["server_label"] = str(data.get("server_label", existing.server_label if existing else "")).strip() or command
            remote_tool_name = str(mcp.get("tool_name", existing_mcp.get("tool_name", data.get("name", "")))).strip()
            if not remote_tool_name:
                raise ValueError("MCP tool_name is required")
            stdio_args = list(mcp.get("args") or []) if isinstance(mcp.get("args"), list) else []
            stdio_env = dict(mcp.get("env")) if isinstance(mcp.get("env"), dict) else None
            stdio_cwd = str(mcp.get("cwd") or "").strip() or None
            result["schema"] = {
                "mcp": {
                    "transport": transport,
                    "command": command,
                    "args": stdio_args,
                    "env": stdio_env,
                    "cwd": stdio_cwd,
                    "tool_name": remote_tool_name,
                    "input_schema": _mcp_input_schema_value(mcp.get("input_schema", existing_mcp.get("input_schema"))),
                }
            }
            # auth fields default to none for stdio (no HTTP auth needed)
            result["auth_type"] = "none"
            result["auth_header_name"] = "Authorization"
            result["auth_query_name"] = ""
        else:
            # HTTP / SSE transport: URL is required
            if "url" in data or not partial:
                url = str(data.get("url", existing.url if existing else "")).strip()
                _validate_safe_mcp_url(url)
                result["url"] = url
            auth = _auth_value(data.get("auth")) if "auth" in data else {}
            auth_type = str(auth.get("type", existing.auth_type if existing else "none")).strip() or "none"
            if auth_type not in AUTH_TYPES:
                raise ValueError("Unsupported auth type")
            result["auth_type"] = auth_type
            result["auth_header_name"] = str(auth.get("header_name", existing.auth_header_name if existing else "Authorization")).strip() or "Authorization"
            result["auth_query_name"] = str(auth.get("query_name", existing.auth_query_name if existing else "")).strip()
            if "secret" in auth:
                secret = str(auth.get("secret") or "").strip()
                if not secret:
                    raise ValueError("Tool secret cannot be empty")
                result["secret"] = secret
            if auth.get("clear_secret"):
                result["clear_secret"] = True
            timeout = _timeout_seconds(
                data.get("timeout_seconds", existing.timeout_seconds if existing else DEFAULT_MCP_TIMEOUT_SECONDS),
                default=DEFAULT_MCP_TIMEOUT_SECONDS,
                maximum=MAX_MCP_TIMEOUT_SECONDS,
                label="MCP timeout",
            )
            result["timeout_seconds"] = timeout
            if "server_label" not in result:
                result["server_label"] = str(data.get("server_label", existing.server_label if existing else "")).strip()
            remote_tool_name = str(mcp.get("tool_name", existing_mcp.get("tool_name", data.get("name", "")))).strip()
            if not remote_tool_name:
                raise ValueError("MCP tool_name is required")
            result["schema"] = {
                "mcp": {
                    "transport": transport,
                    "tool_name": remote_tool_name,
                    "input_schema": _mcp_input_schema_value(mcp.get("input_schema", existing_mcp.get("input_schema"))),
                }
            }
        result["method"] = "POST"
        result["headers_schema"] = {}
        result["query_schema"] = {}
        result["body_schema"] = {}
        result["response_path"] = "$"
        result["search_options"] = {}
    else:
        result.setdefault("method", "GET")
        result.setdefault("url", "")
        result.setdefault("auth_type", "none")
        result.setdefault("auth_header_name", "Authorization")
        result.setdefault("auth_query_name", "")
        result.setdefault("response_path", "$")
        result.setdefault("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS)
        result.setdefault("schema", {})
    return result


def _execute_builtin_search(tool: Tool, context: dict) -> dict:
    query = _search_query(context)
    search_options = tool.search_options or {}
    top_k = int(search_options.get("top_k") or search_options.get("max_results") or 3)
    search_result = web_search_service.search_web(query, top_k=top_k, timeout_seconds=tool.timeout_seconds)
    items = search_result["items"]
    preview = json.dumps(items, ensure_ascii=False)
    return {
        "tool": tool.name,
        "tool_type": "builtin_search",
        "content": preview,
        "status_code": 200,
        "content_type": "application/json",
        "latency_ms": search_result.get("latency_ms", 0),
        "result_preview": _preview(preview),
        "result_json": {"query": search_result["query"], "provider": search_result["provider"], "items": items},
    }


def _execute_builtin_tool(tool: Tool, context: dict) -> dict:
    impl = BUILTIN_TOOLS.get(tool.name)
    if not impl:
        raise ValueError(f"Built-in tool '{tool.name}' is not available")
    input_data = context.get("input")
    if isinstance(input_data, dict):
        args = dict(input_data)
        if tool.name == "run_powershell" and context.get("_agent_workdir"):
            args["_agent_workdir"] = context.get("_agent_workdir")
        return impl["execute"](args) | {"tool": tool.name, "tool_type": "builtin", "status_code": 200, "content_type": "application/json"}
    return impl["execute"]({}) | {"tool": tool.name, "tool_type": "builtin", "status_code": 200, "content_type": "application/json"}


def _auth_value(value) -> dict:
    return value if isinstance(value, dict) else {}


def _timeout_seconds(value, *, default: int, maximum: int, label: str) -> int:
    try:
        timeout = int(value if value is not None else default)
    except (TypeError, ValueError):
        timeout = default
    if timeout < 1 or timeout > maximum:
        raise ValueError(f"{label} must be between 1 and {maximum} seconds")
    return timeout


def _tool_persistence_error(exc: IntegrityError) -> ValueError:
    detail = str(getattr(exc, "orig", exc) or "").lower()
    if "tools_type_check" in detail and "mcp" in detail:
        return ValueError("Database tools.type constraint is outdated. Restart the backend once to run migrations, then save the MCP tool again.")
    return ValueError("Tool could not be saved")


def _search_query(context: dict) -> str:
    input_data = context.get("input")
    if isinstance(input_data, dict):
        return str(input_data.get("query") or input_data.get("q") or input_data.get("message") or "").strip() or "search"
    return str(context.get("input") or "").strip() or "search"


def _tool_name_exists(db: Session, *, workspace_id: int | None, user_id: int | None, name: str) -> bool:
    return (
        db.query(Tool.id)
        .filter(
            Tool.name == name,
            or_(Tool.workspace_id.is_(None), Tool.workspace_id == workspace_id),
            or_(Tool.user_id.is_(None), Tool.user_id == user_id),
        )
        .first()
        is not None
    )


def _tool_parameters_schema(tool: Tool) -> dict:
    if tool.type == "builtin":
        impl = BUILTIN_TOOLS.get(tool.name)
        if impl:
            return impl["parameters"]
    if tool.type == "mcp":
        return _mcp_input_schema_value(_tool_mcp_config(tool).get("input_schema"))
    if tool.type == "builtin_search":
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题",
                }
            },
            "required": ["query"],
        }
    properties: dict = {}
    required: list[str] = []
    for key, spec in (tool.query_schema or {}).items():
        prop = {"type": "string", "description": key}
        if isinstance(spec, dict):
            prop["description"] = spec.get("description") or key
            if spec.get("required"):
                required.append(key)
        properties[key] = prop
    if tool.method in {"POST", "PUT", "PATCH"}:
        for key, spec in (tool.body_schema or {}).items():
            prop = {"type": "string", "description": key}
            if isinstance(spec, dict):
                prop["description"] = spec.get("description") or key
                if spec.get("required"):
                    required.append(key)
            properties[key] = prop
    if not properties:
        properties["input"] = {"type": "string", "description": "传递给工具的输入文本"}
        required = ["input"]
    return {
        "type": "object",
        "properties": properties,
        "required": required[:10],
    }


def _error_code(message: str) -> str:
    if "MCP" in message and ("timed out" in message or "timeout" in message or "connection closed before the tool returned" in message):
        return "mcp_timeout"
    if "HTTPS" in message:
        return "https_required"
    if "blocked" in message:
        return "target_blocked"
    if "Timeout" in message or "timeout" in message:
        return "timeout"
    return "tool_error"


def _tool_timeout_hint(tool: Tool) -> str:
    if tool.type == "mcp":
        current = int(tool.timeout_seconds or DEFAULT_MCP_TIMEOUT_SECONDS)
        suggested = min(MAX_MCP_TIMEOUT_SECONDS, max(DEFAULT_MCP_TIMEOUT_SECONDS, current * 2))
        return f"该 MCP 工具可能执行较慢。当前超时为 {current}s，建议将 timeout_seconds 调整到 {suggested}s 后重试；阿里云百炼 WebParser 等网页解析工具常见耗时超过 10s。"
    current = int(tool.timeout_seconds or DEFAULT_HTTP_TIMEOUT_SECONDS)
    return f"当前超时为 {current}s。请检查目标服务响应时间，或适当调高 timeout_seconds。"
