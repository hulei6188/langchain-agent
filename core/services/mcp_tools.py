from __future__ import annotations

import ipaddress
import re
import socket
import time
import urllib.parse

from core.db.models import Tool
from core.integrations.mcp_client import MCPClientError, call_mcp_tool, call_stdio_mcp_tool
from core.security.api_keys import decrypt_api_key
from core.services.tool_utils import dict_value, preview, url_with_query


MCP_TRANSPORTS = {"streamable_http", "sse", "stdio"}
DEFAULT_MCP_TIMEOUT_SECONDS = 30
MAX_MCP_TIMEOUT_SECONDS = 120
CLOUD_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal"}
MCP_HTTPS_HOST_ALLOWLIST = {"dashscope.aliyuncs.com"}


def execute_mcp_tool(tool: Tool, context: dict) -> dict:
    mcp = tool_mcp_config(tool)
    remote_tool_name = str(mcp.get("tool_name") or "").strip() or tool.name
    if not remote_tool_name:
        raise ValueError("MCP tool is missing remote tool_name")
    transport = str(mcp.get("transport") or "streamable_http")
    input_schema = mcp.get("input_schema")
    arguments = sanitize_mcp_arguments(input_schema, dict_value(context.get("input")))
    validate_mcp_input(input_schema, arguments)
    started = time.monotonic()
    try:
        if transport == "stdio":
            command = str(mcp.get("command") or "").strip()
            if not command:
                raise ValueError("MCP stdio tool requires a command")
            session_key = str(context.get("_session_key") or context.get("session_id") or "").strip() or None
            result = call_stdio_mcp_tool(
                command,
                args=mcp.get("args") or [],
                tool_name=remote_tool_name,
                arguments=arguments,
                env=mcp.get("env"),
                cwd=mcp.get("cwd"),
                timeout_seconds=tool.timeout_seconds,
                session_key=session_key,
            )
        else:
            server_url = str(tool.url or "").strip()
            validate_safe_mcp_url(server_url)
            secret = decrypt_api_key(tool.encrypted_secret) if tool.encrypted_secret else ""
            target_url, headers = mcp_request_target(
                server_url,
                auth_type=tool.auth_type,
                auth_header_name=tool.auth_header_name,
                auth_query_name=tool.auth_query_name,
                secret=secret,
            )
            result = call_mcp_tool(
                target_url,
                remote_tool_name,
                arguments=arguments,
                headers=headers,
                transport=transport,
                timeout_seconds=tool.timeout_seconds,
            )
    except MCPClientError as exc:
        detail = preview(str(exc), 300)
        if "timed out" in detail.lower() or "timeout" in detail.lower() or "connection closed before the tool returned" in detail.lower():
            raise ValueError(f"MCP tool request timed out after {tool.timeout_seconds}s: {detail}") from exc
        raise ValueError(f"MCP tool request failed: {detail}") from exc
    return {
        "tool": tool.name,
        "tool_type": "mcp",
        "status_code": result.get("status_code", 200),
        "content_type": result.get("content_type", "application/json"),
        "latency_ms": int((time.monotonic() - started) * 1000),
        "content": preview(result.get("content") or "", 4000),
        "result_preview": preview(result.get("result_preview") or result.get("content") or ""),
        "result_json": result.get("result_json"),
    }


def tool_mcp_config(tool: Tool | None) -> dict:
    if tool is None or not isinstance(getattr(tool, "schema", None), dict):
        return {}
    mcp = tool.schema.get("mcp")
    if not isinstance(mcp, dict):
        return {}
    config = {
        "transport": str(mcp.get("transport") or "streamable_http").strip().lower() or "streamable_http",
        "tool_name": str(mcp.get("tool_name") or "").strip(),
        "input_schema": mcp_input_schema_value(mcp.get("input_schema")),
    }
    if config["transport"] == "stdio":
        config["command"] = str(mcp.get("command") or "").strip()
        raw_args = mcp.get("args")
        config["args"] = list(raw_args) if isinstance(raw_args, list) else []
        env = mcp.get("env")
        config["env"] = dict(env) if isinstance(env, dict) else None
        config["cwd"] = str(mcp.get("cwd") or "").strip() or None
    return config


def mcp_transport_value(value, *, url: str = "", existing: str | None = None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"streamable-http", "streamablehttp"}:
        raw = "streamable_http"
    if not raw:
        existing_value = str(existing or "").strip().lower()
        if existing_value in MCP_TRANSPORTS:
            raw = existing_value
        else:
            path = urllib.parse.urlparse(str(url or "")).path.lower().rstrip("/")
            raw = "sse" if path.endswith("/sse") or path == "sse" else "streamable_http"
    if raw not in MCP_TRANSPORTS:
        raise ValueError("Unsupported MCP transport")
    return raw


def validate_mcp_input(schema, arguments: dict, *, path: str = "") -> None:
    normalized = mcp_input_schema_value(schema)
    properties = normalized.get("properties") if isinstance(normalized.get("properties"), dict) else {}
    required = normalized.get("required") if isinstance(normalized.get("required"), list) else []
    for key in required:
        field_path = f"{path}.{key}" if path else str(key)
        spec = properties.get(key) if isinstance(properties, dict) else {}
        value = arguments.get(key) if isinstance(arguments, dict) else None
        if _is_missing_mcp_value(value, spec):
            raise ValueError(f"MCP input '{field_path}' is required")

    if not isinstance(arguments, dict):
        return
    for key, spec in properties.items():
        if key not in arguments:
            continue
        field_path = f"{path}.{key}" if path else str(key)
        value = arguments.get(key)
        if isinstance(spec, dict):
            field_type = spec.get("type")
            if field_type == "object" and isinstance(value, dict):
                validate_mcp_input(spec, value, path=field_path)


def sanitize_mcp_arguments(schema, arguments: dict) -> dict:
    normalized = mcp_input_schema_value(schema)
    if not isinstance(arguments, dict):
        return {}
    return _sanitize_mcp_arguments(normalized, arguments)


def _sanitize_mcp_arguments(schema: dict, arguments: dict) -> dict:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") if isinstance(schema.get("required"), list) else [])
    cleaned = {}
    for key, value in arguments.items():
        spec = properties.get(key)
        if key not in required and isinstance(spec, dict) and value is None:
            continue
        if isinstance(spec, dict) and spec.get("type") == "object" and isinstance(value, dict):
            cleaned[key] = _sanitize_mcp_arguments(spec, value)
        else:
            cleaned[key] = value
    return cleaned


def _is_missing_mcp_value(value, spec) -> bool:
    if value is None:
        return True
    if isinstance(spec, dict):
        field_type = spec.get("type")
        if field_type == "string":
            return not str(value).strip()
        if field_type == "array":
            return not isinstance(value, list) or len(value) == 0
        if field_type == "object":
            return not isinstance(value, dict) or len(value) == 0
    return False


def mcp_input_schema_value(value) -> dict:
    schema = value if isinstance(value, dict) else {}
    if not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    if normalized.get("type") != "object":
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "传递给 MCP 工具的输入",
                }
            },
            "required": ["input"],
        }
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    if not isinstance(normalized.get("required"), list):
        normalized["required"] = []
    return normalized


def suggest_mcp_tool_name(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name or "").strip()).strip("_") or "mcp_tool"
    if normalized and normalized[0].isdigit():
        normalized = f"mcp_{normalized}"
    return normalized[:120]


def mcp_request_target(url: str, *, auth_type: str, auth_header_name: str, auth_query_name: str, secret: str) -> tuple[str, dict[str, str]]:
    headers: dict[str, str] = {}
    target_url = url
    if auth_type in {"bearer", "header"} and secret:
        header_name = auth_header_name or "Authorization"
        headers[header_name] = f"Bearer {secret}" if auth_type == "bearer" else secret
    if auth_type == "query" and secret:
        target_url = url_with_query(url, {auth_query_name or "api_key": secret})
    return target_url, headers


def validate_safe_mcp_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MCP tools require an http:// or https:// URL")
    host = parsed.hostname.strip().lower()
    if host in {"metadata", "metadata.google.internal"}:
        raise ValueError("MCP tool target is blocked")
    if host == "localhost" or host.endswith(".localhost"):
        return url
    if parsed.scheme == "https" and host in MCP_HTTPS_HOST_ALLOWLIST:
        return url
    addr_info = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    resolved_ips = [ipaddress.ip_address(info[4][0]) for info in addr_info]
    if parsed.scheme == "http" and not any(ip.is_loopback for ip in resolved_ips):
        raise ValueError("MCP tools over http only support localhost")
    for ip in resolved_ips:
        if str(ip) in CLOUD_METADATA_HOSTS:
            raise ValueError("MCP tool target is blocked")
        if ip.is_loopback:
            continue
        if ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError("MCP tool target is blocked")
    return url
