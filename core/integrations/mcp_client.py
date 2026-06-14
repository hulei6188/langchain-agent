from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import httpx
import atexit
import logging
import threading
import time

logger = logging.getLogger("mcp_client")

try:
    from langchain_mcp_adapters.client import create_session as create_adapter_session
    from langchain_mcp_adapters.client import load_mcp_tools as load_adapter_mcp_tools
except Exception:  # pragma: no cover - optional dependency at runtime
    create_adapter_session = None
    load_adapter_mcp_tools = None


class MCPClientError(ValueError):
    """Raised when an MCP client request cannot be completed."""


def discover_mcp_tools(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
    transport: str = "streamable_http",
    timeout_seconds: int = 15,
) -> list[dict]:
    return asyncio.run(_discover_mcp_tools(server_url, headers=headers or {}, transport=transport, timeout_seconds=timeout_seconds))


def call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    transport: str = "streamable_http",
    timeout_seconds: int = 15,
) -> dict:
    timeout_seconds = _normalize_timeout(timeout_seconds)
    return asyncio.run(
        _call_mcp_tool(
            server_url,
            tool_name,
            arguments=arguments or {},
            headers=headers or {},
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
    )


def discover_stdio_mcp_tools(
    command: str,
    args: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 15,
    session_key: str | None = None,
) -> list[dict]:
    """Discover tools from a stdio-based MCP server (e.g. Playwright MCP).

    If *session_key* is provided the session will be pooled per-key so that
    subsequent tool calls within the same chat session reuse the same MCP
    process and browser state.
    """
    _require_stdio_sdk()
    timeout_seconds = _normalize_timeout(timeout_seconds)
    effective_args = _stdio_args_for_session(command, args or [], env=env)
    if not session_key:
        return asyncio.run(
            _discover_stdio_mcp_tools(
                command,
                effective_args,
                env=env,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        )
    pool = _get_stdio_pool()
    return pool.discover_tools(
        session_key,
        command,
        effective_args,
        env=env,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )


def call_stdio_mcp_tool(
    command: str,
    args: list[str] | None,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 15,
    session_key: str | None = None,
) -> dict:
    """Call a tool on a stdio-based MCP server (e.g. Playwright MCP).

    If *session_key* is provided the session is pooled per-key — the MCP
    server process and browser state survive across consecutive calls within
    the same chat session (browser_navigate → snapshot → click).

    Different session_keys get completely isolated MCP processes with
    independent browsers, cookies, and tabs.
    """
    _require_stdio_sdk()
    timeout_seconds = _normalize_timeout(timeout_seconds)
    effective_args = _stdio_args_for_session(command, args or [], env=env)
    if not session_key:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command,
                effective_args,
                tool_name,
                arguments or {},
                env=env,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        )
    pool = _get_stdio_pool()
    return pool.call_tool(
        session_key,
        command,
        effective_args,
        tool_name,
        arguments or {},
        env=env,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )


async def _discover_stdio_mcp_tools(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int,
) -> list[dict]:
    _require_stdio_sdk()
    tools = await _map_mcp_errors(
        _load_stdio_adapter_tools(
            command,
            args,
            env=env,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
    )
    return _adapter_tools_metadata(tools)


async def _call_stdio_mcp_tool(
    command: str,
    args: list[str],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int,
) -> dict:
    _require_stdio_sdk()
    return await _map_mcp_errors(
        _call_stdio_adapter_tool(
            command,
            args,
            tool_name,
            arguments=arguments,
            env=env,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
    )


async def _discover_mcp_tools(server_url: str, *, headers: dict[str, str], transport: str, timeout_seconds: int) -> list[dict]:
    timeout_seconds = _normalize_timeout(timeout_seconds)
    tools = await _map_mcp_errors(
        _load_adapter_tools(
            server_url,
            headers=headers,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
    )
    return _adapter_tools_metadata(tools)


async def _load_adapter_tools(
    server_url: str,
    *,
    headers: dict[str, str],
    transport: str,
    timeout_seconds: int,
) -> list:
    if load_adapter_mcp_tools is None:
        raise MCPClientError("langchain-mcp-adapters is not installed. Add the 'langchain-mcp-adapters' package.")
    return await load_adapter_mcp_tools(
        None,
        connection=_adapter_connection(
            server_url,
            headers=headers,
            transport=transport,
            timeout_seconds=timeout_seconds,
        ),
    )


async def _load_stdio_adapter_tools(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int,
) -> list:
    if load_adapter_mcp_tools is None:
        raise MCPClientError("langchain-mcp-adapters is not installed. Add the 'langchain-mcp-adapters' package.")
    return await load_adapter_mcp_tools(
        None,
        connection=_stdio_adapter_connection(
            command,
            args,
            env=env,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        ),
    )


async def _map_mcp_errors(awaitable):
    try:
        return await awaitable
    except MCPClientError:
        raise
    except BaseException as exc:
        _raise_mcp_client_error(exc)


def _adapter_tools_metadata(tools) -> list[dict]:
    return [
        {
            "name": str(getattr(tool, "name", "") or ""),
            "description": str(getattr(tool, "description", "") or ""),
            "input_schema": _langchain_tool_schema(tool),
        }
        for tool in tools
    ]


def _adapter_connection(
    server_url: str,
    *,
    headers: dict[str, str],
    transport: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    normalized_transport = str(transport or "streamable_http").strip().lower() or "streamable_http"
    if normalized_transport == "sse":
        return {
            "transport": "sse",
            "url": server_url,
            "headers": headers or None,
            "timeout": float(timeout_seconds),
            "sse_read_timeout": float(timeout_seconds),
        }
    return {
        "transport": "streamable_http",
        "url": server_url,
        "headers": headers or None,
        "timeout": timedelta(seconds=timeout_seconds),
        "sse_read_timeout": timedelta(seconds=timeout_seconds),
    }


def _stdio_adapter_connection(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    connection = {
        "transport": "stdio",
        "command": command,
        "args": list(args or []),
        "env": env,
        "cwd": cwd,
    }
    if timeout_seconds is not None:
        connection["session_kwargs"] = {"read_timeout_seconds": timedelta(seconds=timeout_seconds)}
    return connection


def _langchain_tool_schema(tool) -> dict:
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return _jsonable(schema.model_json_schema()) or {}
    if hasattr(schema, "schema"):
        return _jsonable(schema.schema()) or {}
    return {}


async def _call_adapter_tool(
    server_url: str,
    tool_name: str,
    *,
    arguments: dict[str, Any],
    headers: dict[str, str],
    transport: str,
    timeout_seconds: int,
) -> dict:
    return await _invoke_adapter_tool(
        await _load_adapter_tools(
            server_url,
            headers=headers,
            transport=transport,
            timeout_seconds=timeout_seconds,
        ),
        tool_name,
        arguments,
    )


async def _call_stdio_adapter_tool(
    command: str,
    args: list[str],
    tool_name: str,
    *,
    arguments: dict[str, Any],
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int,
) -> dict:
    return await _invoke_adapter_tool(
        await _load_stdio_adapter_tools(
            command,
            args,
            env=env,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        ),
        tool_name,
        arguments,
    )


async def _invoke_adapter_tool(tools, tool_name: str, arguments: dict[str, Any]) -> dict:
    tool = _adapter_tool_by_name(tools, tool_name)
    result = await tool.ainvoke(arguments or {})
    return _adapter_tool_result(result)


def _adapter_tool_by_name(tools, tool_name: str):
    tool = next((item for item in tools if item.name == tool_name), None)
    if tool is None:
        raise MCPClientError(f"MCP tool '{tool_name}' not found")
    return tool


def _adapter_tool_result(result) -> dict:
    content = result
    artifact = None
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
    if hasattr(content, "content") and not isinstance(content, (str, bytes, bytearray, dict, list, tuple)):
        artifact = getattr(content, "artifact", artifact)
        content = getattr(content, "content", "")
    structured = getattr(artifact, "structured_content", None)
    content_payload = _jsonable(content)
    text_content = _content_value_text(content_payload)
    if structured is not None:
        result_json = _jsonable(structured)
    elif isinstance(content_payload, (dict, list)):
        result_json = content_payload
    else:
        result_json = None
    return {
        "status_code": 200,
        "content_type": "application/json" if result_json is not None else "text/plain",
        "content": text_content,
        "result_preview": text_content[:500],
        "result_json": result_json,
    }


def _content_value_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_content_text(item))
            else:
                parts.append(str(item or ""))
        return "\n\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        return _content_text(value) or json.dumps(value, ensure_ascii=False)
    return str(value or "")


async def _call_mcp_tool(
    server_url: str,
    tool_name: str,
    *,
    arguments: dict[str, Any],
    headers: dict[str, str],
    transport: str,
    timeout_seconds: int,
) -> dict:
    timeout_seconds = _normalize_timeout(timeout_seconds)
    return await _map_mcp_errors(
        _call_adapter_tool(
            server_url,
            tool_name,
            arguments=arguments,
            headers=headers,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
    )


def _require_stdio_sdk() -> None:
    if load_adapter_mcp_tools is None or create_adapter_session is None:
        raise MCPClientError("langchain-mcp-adapters stdio support is not available. Add the 'langchain-mcp-adapters' package.")


def _normalize_timeout(timeout_seconds: int | float | None) -> int:
    try:
        timeout = int(timeout_seconds or 15)
    except (TypeError, ValueError):
        timeout = 15
    return max(1, timeout)


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _content_text(block) -> str:
    if not isinstance(block, dict):
        return str(block or "")
    block_type = str(block.get("type") or "").lower()
    if block_type == "text":
        return str(block.get("text") or "")
    if block_type == "image":
        mime = block.get("mimeType") or block.get("mime_type") or "image"
        return f"[MCP image: {mime}]"
    if block_type == "audio":
        mime = block.get("mimeType") or block.get("mime_type") or "audio"
        return f"[MCP audio: {mime}]"
    if block_type in {"resource", "embedded_resource"}:
        resource = block.get("resource") or {}
        if isinstance(resource, dict):
            if isinstance(resource.get("text"), str) and resource.get("text"):
                return str(resource["text"])
            if isinstance(resource.get("uri"), str) and resource.get("uri"):
                return f"[MCP resource: {resource['uri']}]"
        return "[MCP embedded resource]"
    if block:
        return json.dumps(block, ensure_ascii=False)
    return ""


# ── Per-session persistent stdio MCP session pool ──
# Each (session_key, command, args, env, cwd) tuple gets its own long-lived MCP process
# so that browser state (Playwright pages, cookies, tabs) lives across
# consecutive tool calls within the same chat session.
#
# Different session_keys → different processes → complete isolation.

_STDIO_POOL_IDLE_TTL = 30 * 60  # 30 minutes


def _stdio_args_for_session(command: str, args: list[str], *, env: dict[str, str] | None = None) -> list[str]:
    """Return process args adjusted for safe per-session browser isolation."""
    normalized_args = [str(item) for item in (args or [])]
    if not _is_playwright_mcp_command(command, normalized_args):
        return normalized_args
    if _has_playwright_profile_override(normalized_args, env):
        return normalized_args
    return [*normalized_args, "--isolated"]


def _is_playwright_mcp_command(command: str, args: list[str]) -> bool:
    tokens = [str(command or ""), *(str(item or "") for item in args)]
    return any("@playwright/mcp" in token or "playwright-mcp" in token for token in tokens)


def _has_playwright_profile_override(args: list[str], env: dict[str, str] | None) -> bool:
    profile_flags = (
        "--isolated",
        "--persistent",
        "--extension",
        "--user-data-dir",
    )
    for arg in args:
        normalized = str(arg or "").strip()
        if any(normalized == flag or normalized.startswith(f"{flag}=") for flag in profile_flags):
            return True
    env_values = {str(key).upper(): str(value) for key, value in (env or {}).items()}
    return any(
        env_values.get(key)
        for key in (
            "PLAYWRIGHT_MCP_ISOLATED",
            "PLAYWRIGHT_MCP_PERSISTENT",
            "PLAYWRIGHT_MCP_EXTENSION",
            "PLAYWRIGHT_MCP_USER_DATA_DIR",
        )
    )


def _pool_key(session_key: str | None, command: str, args: list[str], env: dict | None = None, cwd: str | None = None) -> str:
    """Deterministic pool key for a session+command combination."""
    sk = session_key or "__default__"
    config = {
        "command": str(command or ""),
        "args": list(args or []),
        "env": {str(key): str(value) for key, value in sorted((env or {}).items())},
        "cwd": str(cwd or ""),
    }
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{sk}|{digest}"


class _PooledSession:
    """A single stdio MCP session that can survive across tool calls."""

    def __init__(self, key: str, command: str, args: list[str], env: dict | None, cwd: str | None):
        self.key = key
        self._command = command
        self._args = list(args or [])
        self._env = dict(env) if env else None
        self._cwd = str(cwd) if cwd else None
        self._session_cm = None
        self._session = None
        self._last_used = time.monotonic()
        self._dead = False
        self._lock = asyncio.Lock()

    def touch(self) -> None:
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    async def _start_locked(self) -> None:
        logger.info("MCP stdio session [%s] starting: %s %s", self.key, self._command, " ".join(self._args))
        self._session_cm = create_adapter_session(
            _stdio_adapter_connection(
                self._command,
                self._args,
                env=self._env,
                cwd=self._cwd,
                timeout_seconds=None,
            )
        )
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        self._dead = False
        self.touch()
        logger.info("MCP stdio session [%s] ready", self.key)

    async def _stop_locked(self) -> None:
        logger.info("MCP stdio session [%s] stopping (idle %.0fs)", self.key, self.idle_seconds())
        try:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
        finally:
            self._session_cm = None
            self._session = None
            self._dead = True
        logger.info("MCP stdio session [%s] stopped", self.key)

    async def _stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def call_tool(self, tool_name: str, arguments: dict, timeout_seconds: int) -> dict:
        async with self._lock:
            for attempt in range(2):
                if self._session is None or self._dead:
                    if self._session is not None or self._session_cm is not None:
                        await self._stop_locked()
                    await self._start_locked()
                try:
                    tool = await self._adapter_tool(tool_name)
                    result = await asyncio.wait_for(tool.ainvoke(arguments or {}), timeout=timeout_seconds)
                    self.touch()
                    return _adapter_tool_result(result)
                except MCPClientError:
                    raise
                except BaseException:
                    logger.warning("MCP stdio session [%s] call_tool failed", self.key, exc_info=True)
                    self._dead = True
                    if attempt == 0:
                        await self._stop_locked()
                        continue
                    raise

    async def list_tools(self, timeout_seconds: int) -> list[dict]:
        async with self._lock:
            for attempt in range(2):
                if self._session is None or self._dead:
                    if self._session is not None or self._session_cm is not None:
                        await self._stop_locked()
                    await self._start_locked()
                try:
                    tools = await asyncio.wait_for(
                        load_adapter_mcp_tools(self._session),
                        timeout=_normalize_timeout(timeout_seconds),
                    )
                    self.touch()
                    return _adapter_tools_metadata(tools)
                except BaseException:
                    logger.warning("MCP stdio session [%s] list_tools failed", self.key, exc_info=True)
                    self._dead = True
                    if attempt == 0:
                        await self._stop_locked()
                        continue
                    raise

    async def _adapter_tool(self, tool_name: str):
        return _adapter_tool_by_name(await load_adapter_mcp_tools(self._session), tool_name)


class _StdioSessionPool:
    """Per-session-key pool of persistent stdio MCP sessions.

    Single background event-loop thread; per-session asyncio locks serialize
    access to each adapter-backed session, and threading.Lock protects the sync map.
    """

    def __init__(self):
        self._sync_lock = threading.Lock()
        self._sessions: dict[str, _PooledSession] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-stdio-pool", daemon=True)
        self._cleanup_future = None
        self._closed = False
        self._thread.start()
        atexit.register(self.shutdown)
        # Schedule periodic idle cleanup (runs every 120s on the background loop)
        self._schedule_cleanup()
        logger.info("MCP stdio session pool started (TTL=%ds)", _STDIO_POOL_IDLE_TTL)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _schedule_cleanup(self) -> None:
        """Kick off the first cleanup cycle once the loop is running."""
        self._cleanup_future = asyncio.run_coroutine_threadsafe(self._cleanup_loop(), self._loop)

    async def _cleanup_loop(self) -> None:
        """Periodically reap idle sessions every 120 seconds."""
        while not self._closed:
            await asyncio.sleep(120)
            try:
                removed = await self._cleanup_idle_async()
                if removed:
                    logger.info("MCP stdio pool cleanup: reaped %d idle sessions (TTL=%ds)", removed, _STDIO_POOL_IDLE_TTL)
            except asyncio.CancelledError:
                break
            except BaseException:
                logger.warning("MCP stdio pool cleanup error", exc_info=True)

    # ── sync helpers ──

    def _run_coroutine(self, coro):
        if threading.current_thread() is self._thread:
            if inspect.iscoroutine(coro):
                coro.close()
            raise MCPClientError("Cannot synchronously wait for the MCP stdio pool loop from its own thread")
        if not self._loop.is_running():
            if inspect.iscoroutine(coro):
                coro.close()
            raise MCPClientError("MCP stdio session pool loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ── pool management (called from sync public API) ──

    def _acquire_session(
        self,
        session_key: str | None,
        command: str,
        args: list[str],
        env: dict | None,
        cwd: str | None,
    ) -> _PooledSession:
        key = _pool_key(session_key, command, args, env=env, cwd=cwd)
        with self._sync_lock:
            if self._closed:
                raise MCPClientError("MCP stdio session pool is shut down")
            existing = self._sessions.get(key)
            if existing is not None and not existing._dead:
                existing.touch()
                logger.debug("MCP stdio session [%s] reused (alive=%d total)", key, len(self._sessions))
                return existing
            if existing is not None and existing._dead:
                logger.info("MCP stdio session [%s] was dead, replacing", key)
                self._sessions.pop(key, None)
                self._run_coroutine(existing._stop())
            session = _PooledSession(key, command, args, env, cwd)
            self._sessions[key] = session
            logger.info("MCP stdio session [%s] created (alive=%d total)", key, len(self._sessions))
            return session

    def call_tool(
        self,
        session_key: str | None,
        command: str,
        args: list[str],
        tool_name: str,
        arguments: dict,
        *,
        env: dict | None = None,
        cwd: str | None = None,
        timeout_seconds: int = 15,
    ):
        session = self._acquire_session(session_key, command, args, env, cwd)
        try:
            return self._run_coroutine(
                self._call_tool_with_retry(session, tool_name, arguments, timeout_seconds)
            )
        except BaseException:
            # If retry also failed, remove from pool
            with self._sync_lock:
                removed = self._sessions.pop(session.key, None) if self._sessions.get(session.key) is session else None
            if removed is not None:
                logger.error("MCP stdio session [%s] removed after unrecoverable error", session.key)
                self._run_coroutine(removed._stop())
            raise

    def discover_tools(
        self,
        session_key: str | None,
        command: str,
        args: list[str],
        *,
        env: dict | None = None,
        cwd: str | None = None,
        timeout_seconds: int = 15,
    ):
        session = self._acquire_session(session_key, command, args, env, cwd)
        try:
            return self._run_coroutine(self._discover_with_retry(session, timeout_seconds))
        except BaseException:
            with self._sync_lock:
                removed = self._sessions.pop(session.key, None) if self._sessions.get(session.key) is session else None
            if removed is not None:
                logger.error("MCP stdio session [%s] removed after unrecoverable error", session.key)
                self._run_coroutine(removed._stop())
            raise

    # ── async internals (run on background loop) ──

    async def _call_tool_with_retry(self, session, tool_name, arguments, timeout_seconds):
        return await session.call_tool(tool_name, arguments, timeout_seconds)

    async def _discover_with_retry(self, session, timeout_seconds):
        return await session.list_tools(timeout_seconds)

    # ── lifecycle ──

    def cleanup_idle(self) -> int:
        """Reap sessions idle longer than TTL. Returns number removed."""
        return self._run_coroutine(self._cleanup_idle_async())

    async def _cleanup_idle_async(self) -> int:
        with self._sync_lock:
            stale = [k for k, s in self._sessions.items() if s.idle_seconds() > _STDIO_POOL_IDLE_TTL]
        removed = 0
        for key in stale:
            with self._sync_lock:
                session = self._sessions.get(key)
                if session is None or session.idle_seconds() <= _STDIO_POOL_IDLE_TTL:
                    session = None
                else:
                    self._sessions.pop(key, None)
            if session is not None:
                logger.info("MCP stdio session [%s] idle %.0fs > TTL, reaping", key, session.idle_seconds())
                await session._stop()
                removed += 1
        return removed

    def shutdown(self) -> None:
        with self._sync_lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
        if self._cleanup_future is not None:
            self._cleanup_future.cancel()
            self._cleanup_future = None
        if not self._loop.is_running():
            return
        if sessions:
            logger.info("MCP stdio pool shutting down %d sessions", len(sessions))
            for s in sessions:
                try:
                    self._run_coroutine(s._stop())
                except BaseException:
                    logger.warning("MCP stdio session [%s] error during shutdown", s.key, exc_info=True)
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except BaseException:
            pass
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        logger.info("MCP stdio session pool shut down")

    @property
    def active_count(self) -> int:
        with self._sync_lock:
            return len(self._sessions)


# Singleton pool
_stdio_pool: _StdioSessionPool | None = None


def _get_stdio_pool() -> _StdioSessionPool:
    global _stdio_pool
    if _stdio_pool is None:
        _stdio_pool = _StdioSessionPool()
    return _stdio_pool


def _raise_mcp_client_error(exc: BaseException) -> None:
    if isinstance(exc, MCPClientError):
        raise exc
    message = _describe_mcp_exception(exc)
    raise MCPClientError(message) from exc


def _describe_mcp_exception(exc: BaseException) -> str:
    messages = _collect_exception_messages(exc)
    lowered = [message.lower() for message in messages]
    walked = list(_walk_exception_graph(exc))
    if any("all connection attempts failed" in message for message in lowered) or any(
        isinstance(item, httpx.ConnectError) for item in walked
    ):
        return "Unable to connect to the MCP server"
    if any("timed out" in message or "timeout" in message for message in lowered) or any(
        isinstance(item, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)) for item in walked
    ):
        return "MCP server request timed out"
    if any("unexpected content type" in message for message in lowered):
        return "MCP server returned an unexpected response content type. Confirm the MCP transport matches the endpoint, for example sse for /sse or streamable_http for /mcp."
    if any("connection closed" in message or "brokenresourceerror" in message for message in lowered) or any(
        item.__class__.__name__ in {"BrokenResourceError", "ClosedResourceError", "EndOfStream"} for item in walked
    ):
        return "MCP server connection closed before the tool returned. This is often caused by the configured timeout being too short."
    if isinstance(exc, asyncio.CancelledError) or any("cancelled via cancel scope" in message for message in lowered):
        return "MCP server request timed out"
    for message in messages:
        if message and not message.startswith("unhandled errors in a taskgroup"):
            return message
    return exc.__class__.__name__


def _collect_exception_messages(exc: BaseException) -> list[str]:
    messages: list[str] = []
    for item in _walk_exception_graph(exc):
        text = str(item).strip()
        if text and text not in messages:
            messages.append(text)
    return messages


def _walk_exception_graph(exc: BaseException):
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        nested = getattr(current, "exceptions", None)
        if nested:
            stack.extend(item for item in nested if isinstance(item, BaseException))
        if getattr(current, "__cause__", None) is not None:
            stack.append(current.__cause__)
        if getattr(current, "__context__", None) is not None:
            stack.append(current.__context__)


