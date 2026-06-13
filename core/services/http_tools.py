from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from core.db.models import Tool
from core.security.api_keys import decrypt_api_key
from core.services.builtin_tools import dns_pinned
from core.services.tool_utils import dict_value, preview, safe_json, url_with_query


MAX_RESPONSE_BYTES = 1024 * 1024
CLOUD_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal"}


def execute_http_tool(tool: Tool, context: dict) -> dict:
    validated_ip = validate_safe_https_url(tool.url)
    parsed_host = urllib.parse.urlparse(tool.url).hostname
    input_data = dict_value(context.get("input"))
    body = context.get("body")
    query = _query_params(tool.query_schema or {}, input_data)
    if tool.auth_type == "query" and tool.encrypted_secret:
        query_name = tool.auth_query_name or "api_key"
        query[query_name] = decrypt_api_key(tool.encrypted_secret)
    url = url_with_query(tool.url, query)
    headers = _headers(tool, input_data)
    data = None
    if tool.method in {"POST", "PUT", "PATCH"}:
        data = json.dumps(body if body is not None else _body_from_schema(tool.body_schema or {}, input_data)).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=headers, method=tool.method)
    started = time.monotonic()
    try:
        with dns_pinned(parsed_host, validated_ip):
            with urllib.request.urlopen(request, timeout=tool.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("Tool response is too large")
                text = raw.decode("utf-8", errors="replace")
                result_json = safe_json(text)
                return {
                    "tool": tool.name,
                    "tool_type": tool.type,
                    "status_code": response.status,
                    "content_type": content_type,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "content": preview(text, 4000),
                    "result_preview": preview(text),
                    "result_json": result_json,
                }
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        raise ValueError(f"HTTP tool request failed with status {exc.code}: {preview(detail, 200)}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise ValueError("HTTP tool request failed") from exc


def validate_safe_https_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("HTTP tools require an HTTPS URL")
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "metadata", "metadata.google.internal"} or host.endswith(".localhost"):
        raise ValueError("HTTP tool target is blocked")
    try:
        ip = ipaddress.ip_address(host)
        _reject_ip(ip)
        return str(ip)
    except ValueError:
        pass
    addr_info = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    for info in addr_info:
        _reject_ip(ipaddress.ip_address(info[4][0]))
    return str(addr_info[0][4][0])


def _reject_ip(ip: ipaddress._BaseAddress) -> None:
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved or str(ip) in CLOUD_METADATA_HOSTS:
        raise ValueError("HTTP tool target is blocked")


def _headers(tool: Tool, input_data: dict) -> dict:
    headers = {key: str(input_data.get(key, "")) for key in (tool.headers_schema or {}) if input_data.get(key) is not None}
    if tool.auth_type in {"bearer", "header"} and tool.encrypted_secret:
        secret = decrypt_api_key(tool.encrypted_secret)
        header_name = tool.auth_header_name or "Authorization"
        headers[header_name] = f"Bearer {secret}" if tool.auth_type == "bearer" else secret
    return headers


def _query_params(schema: dict, input_data: dict) -> dict:
    params = {key: input_data.get(key) for key in schema if input_data.get(key) is not None}
    for key, spec in schema.items():
        if isinstance(spec, dict) and spec.get("required") and key not in params:
            raise ValueError(f"Missing required tool input: {key}")
    return params


def _body_from_schema(schema: dict, input_data: dict) -> dict:
    return {key: input_data.get(key) for key in schema if input_data.get(key) is not None}
