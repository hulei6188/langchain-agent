from __future__ import annotations

import asyncio
import inspect
import json
import urllib.parse
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
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    try:
        from mcp.client.streamable_http import streamable_http_client
    except Exception:
        from mcp.client.streamable_http import streamablehttp_client as streamable_http_client
    try:
        from mcp.client.stdio import stdio_client, StdioServerParameters
    except Exception:
        stdio_client = None
        StdioServerParameters = None
except Exception:  # pragma: no cover - optional dependency at runtime
    ClientSession = None
    sse_client = None
    streamable_http_client = None
    stdio_client = None
    StdioServerParameters = None


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
    pool = _get_stdio_pool()
    raw_result = pool.discover_tools(
        session_key,
        command,
        args or [],
        env=env,
        cwd=cwd,
    )
    tools = []
    for item in getattr(raw_result, "tools", []) or []:
        input_schema = _jsonable(getattr(item, "inputSchema", None) or getattr(item, "input_schema", None) or {}) or {}
        tools.append(
            {
                "name": str(getattr(item, "name", "") or ""),
                "description": str(getattr(item, "description", "") or ""),
                "input_schema": input_schema if isinstance(input_schema, dict) else {},
            }
        )
    return tools


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
    pool = _get_stdio_pool()
    raw_result = pool.call_tool(
        session_key,
        command,
        args or [],
        tool_name,
        arguments or {},
        env=env,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    structured = _jsonable(getattr(raw_result, "structuredContent", None) or getattr(raw_result, "structured_content", None))
    content_blocks = [_jsonable(item) for item in (getattr(raw_result, "content", None) or [])]
    text_parts = [part for part in (_content_text(block) for block in content_blocks) if part]
    if not text_parts and structured is not None:
        text_parts.append(json.dumps(structured, ensure_ascii=False))
    if not text_parts and content_blocks:
        payload = content_blocks[0] if len(content_blocks) == 1 else content_blocks
        text_parts.append(json.dumps(payload, ensure_ascii=False))
    text_content = "\n\n".join(text_parts).strip()
    is_error = bool(getattr(raw_result, "isError", False) or getattr(raw_result, "is_error", False))
    if is_error:
        detail = text_content or json.dumps(structured or content_blocks or {"error": "MCP tool returned error"}, ensure_ascii=False)
        raise MCPClientError(detail[:500])
    result_json = structured
    if result_json is None and content_blocks:
        result_json = content_blocks[0] if len(content_blocks) == 1 else {"content": content_blocks}
    return {
        "status_code": 200,
        "content_type": "application/json" if result_json is not None else "text/plain",
        "content": text_content,
        "result_preview": text_content[:500],
        "result_json": result_json,
    }


async def _discover_stdio_mcp_tools(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int,
) -> list[dict]:
    _require_stdio_sdk()
    try:
        async with _stdio_session(command, args, env=env, cwd=cwd, timeout_seconds=timeout_seconds) as session:
            result = await session.list_tools()
    except MCPClientError:
        raise
    except BaseException as exc:
        _raise_mcp_client_error(exc)
    tools = []
    for item in getattr(result, "tools", []) or []:
        input_schema = _jsonable(getattr(item, "inputSchema", None) or getattr(item, "input_schema", None) or {}) or {}
        tools.append(
            {
                "name": str(getattr(item, "name", "") or ""),
                "description": str(getattr(item, "description", "") or ""),
                "input_schema": input_schema if isinstance(input_schema, dict) else {},
            }
        )
    return tools


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
    try:
        async with _stdio_session(command, args, env=env, cwd=cwd, timeout_seconds=timeout_seconds) as session:
            result = await session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            )
    except MCPClientError:
        raise
    except BaseException as exc:
        _raise_mcp_client_error(exc)
    structured = _jsonable(getattr(result, "structuredContent", None) or getattr(result, "structured_content", None))
    content_blocks = [_jsonable(item) for item in (getattr(result, "content", None) or [])]
    text_parts = [part for part in (_content_text(block) for block in content_blocks) if part]
    if not text_parts and structured is not None:
        text_parts.append(json.dumps(structured, ensure_ascii=False))
    if not text_parts and content_blocks:
        payload = content_blocks[0] if len(content_blocks) == 1 else content_blocks
        text_parts.append(json.dumps(payload, ensure_ascii=False))
    text_content = "\n\n".join(text_parts).strip()
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    if is_error:
        detail = text_content or json.dumps(structured or content_blocks or {"error": "MCP tool returned error"}, ensure_ascii=False)
        raise MCPClientError(detail[:500])
    result_json = structured
    if result_json is None and content_blocks:
        result_json = content_blocks[0] if len(content_blocks) == 1 else {"content": content_blocks}
    return {
        "status_code": 200,
        "content_type": "application/json" if result_json is not None else "text/plain",
        "content": text_content,
        "result_preview": text_content[:500],
        "result_json": result_json,
    }


async def _discover_mcp_tools(server_url: str, *, headers: dict[str, str], transport: str, timeout_seconds: int) -> list[dict]:
    _require_sdk(transport)
    timeout_seconds = _normalize_timeout(timeout_seconds)
    await _ensure_server_reachable(server_url, timeout_seconds=timeout_seconds)
    try:
        async with _session(server_url, headers=headers, transport=transport, timeout_seconds=timeout_seconds) as session:
            result = await session.list_tools()
    except MCPClientError:
        raise
    except BaseException as exc:
        _raise_mcp_client_error(exc)
    tools = []
    for item in getattr(result, "tools", []) or []:
        input_schema = _jsonable(getattr(item, "inputSchema", None) or getattr(item, "input_schema", None) or {}) or {}
        tools.append(
            {
                "name": str(getattr(item, "name", "") or ""),
                "description": str(getattr(item, "description", "") or ""),
                "input_schema": input_schema if isinstance(input_schema, dict) else {},
            }
        )
    return tools


async def _call_mcp_tool(
    server_url: str,
    tool_name: str,
    *,
    arguments: dict[str, Any],
    headers: dict[str, str],
    transport: str,
    timeout_seconds: int,
) -> dict:
    _require_sdk(transport)
    timeout_seconds = _normalize_timeout(timeout_seconds)
    await _ensure_server_reachable(server_url, timeout_seconds=timeout_seconds)
    try:
        async with _session(server_url, headers=headers, transport=transport, timeout_seconds=timeout_seconds) as session:
            result = await session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            )
    except MCPClientError:
        raise
    except BaseException as exc:
        _raise_mcp_client_error(exc)
    structured = _jsonable(getattr(result, "structuredContent", None) or getattr(result, "structured_content", None))
    content_blocks = [_jsonable(item) for item in (getattr(result, "content", None) or [])]
    text_parts = [part for part in (_content_text(block) for block in content_blocks) if part]
    if not text_parts and structured is not None:
        text_parts.append(json.dumps(structured, ensure_ascii=False))
    if not text_parts and content_blocks:
        payload = content_blocks[0] if len(content_blocks) == 1 else content_blocks
        text_parts.append(json.dumps(payload, ensure_ascii=False))
    text_content = "\n\n".join(text_parts).strip()
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    if is_error:
        detail = text_content or json.dumps(structured or content_blocks or {"error": "MCP tool returned error"}, ensure_ascii=False)
        raise MCPClientError(detail[:500])
    result_json = structured
    if result_json is None and content_blocks:
        result_json = content_blocks[0] if len(content_blocks) == 1 else {"content": content_blocks}
    return {
        "status_code": 200,
        "content_type": "application/json" if result_json is not None else "text/plain",
        "content": text_content,
        "result_preview": text_content[:500],
        "result_json": result_json,
    }


def _require_sdk(transport: str = "streamable_http") -> None:
    if ClientSession is None:
        raise MCPClientError("MCP Python SDK is not installed. Add the 'mcp' package to the backend environment.")
    if transport == "sse":
        if sse_client is None:
            raise MCPClientError("MCP SSE client is not available. Update the 'mcp' package in the backend environment.")
        return
    if streamable_http_client is None:
        raise MCPClientError("MCP Streamable HTTP client is not available. Update the 'mcp' package in the backend environment.")


def _require_stdio_sdk() -> None:
    if ClientSession is None:
        raise MCPClientError("MCP Python SDK is not installed. Add the 'mcp' package to the backend environment.")
    if stdio_client is None or StdioServerParameters is None:
        raise MCPClientError("MCP stdio client is not available. Update the 'mcp' package in the backend environment.")


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


class _SessionContext:
    def __init__(self, server_url: str, *, headers: dict[str, str], transport: str, timeout_seconds: int) -> None:
        self.server_url = server_url
        self.headers = headers
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self._http_client: httpx.AsyncClient | None = None
        self._transport_cm = None
        self._session_cm = None

    async def __aenter__(self):
        try:
            if self.transport == "sse":
                self._transport_cm = sse_client(
                    self.server_url,
                    headers=self.headers or None,
                    timeout=self.timeout_seconds,
                    sse_read_timeout=self.timeout_seconds,
                )
            elif _transport_supports_http_client():
                self._http_client = httpx.AsyncClient(
                    headers=self.headers or None,
                    follow_redirects=True,
                    timeout=self.timeout_seconds,
                )
                self._transport_cm = streamable_http_client(self.server_url, http_client=self._http_client)
            else:
                self._transport_cm = streamable_http_client(
                    self.server_url,
                    headers=self.headers or None,
                    timeout=self.timeout_seconds,
                )
            streams = await self._transport_cm.__aenter__()
            read_stream, write_stream = streams[0], streams[1]
            self._session_cm = ClientSession(read_stream, write_stream)
            session = await self._session_cm.__aenter__()
            await session.initialize()
            return session
        except BaseException as exc:
            await self._close(exc_type=type(exc), exc=exc, tb=exc.__traceback__)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            _raise_mcp_client_error(exc)

    async def __aexit__(self, exc_type, exc, tb):
        await self._close(exc_type=exc_type, exc=exc, tb=tb)

    async def _close(self, *, exc_type=None, exc=None, tb=None):
        try:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(exc_type, exc, tb)
                except BaseException:
                    pass
        finally:
            try:
                if self._transport_cm is not None:
                    try:
                        await self._transport_cm.__aexit__(exc_type, exc, tb)
                    except BaseException:
                        pass
            finally:
                if self._http_client is not None:
                    try:
                        await self._http_client.aclose()
                    except BaseException:
                        pass
        self._session_cm = None
        self._transport_cm = None
        self._http_client = None


def _session(server_url: str, *, headers: dict[str, str], transport: str, timeout_seconds: int) -> _SessionContext:
    normalized_transport = str(transport or "streamable_http").strip().lower() or "streamable_http"
    return _SessionContext(server_url, headers=headers, transport=normalized_transport, timeout_seconds=timeout_seconds)


class _StdioSessionContext:
    def __init__(self, command: str, args: list[str], *, env: dict[str, str] | None, cwd: str | None, timeout_seconds: int) -> None:
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )
        self._timeout_seconds = timeout_seconds
        self._transport_cm = None
        self._session_cm = None

    async def __aenter__(self):
        try:
            self._transport_cm = stdio_client(self._server_params)
            read_stream, write_stream = await self._transport_cm.__aenter__()
            self._session_cm = ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(seconds=self._timeout_seconds))
            session = await self._session_cm.__aenter__()
            await session.initialize()
            return session
        except BaseException as exc:
            await self._close(exc_type=type(exc), exc=exc, tb=exc.__traceback__)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            _raise_mcp_client_error(exc)

    async def __aexit__(self, exc_type, exc, tb):
        await self._close(exc_type=exc_type, exc=exc, tb=tb)

    async def _close(self, *, exc_type=None, exc=None, tb=None):
        try:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(exc_type, exc, tb)
                except BaseException:
                    pass
        finally:
            try:
                if self._transport_cm is not None:
                    try:
                        await self._transport_cm.__aexit__(exc_type, exc, tb)
                    except BaseException:
                        pass
            finally:
                pass
        self._session_cm = None
        self._transport_cm = None


def _stdio_session(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
    timeout_seconds: int,
) -> _StdioSessionContext:
    return _StdioSessionContext(command, args, env=env, cwd=cwd, timeout_seconds=timeout_seconds)


# ── Per-session persistent stdio MCP session pool ──
# Each (session_key, command, args) tuple gets its own long-lived MCP process
# so that browser state (Playwright pages, cookies, tabs) lives across
# consecutive tool calls within the same chat session.
#
# Different session_keys → different processes → complete isolation.

_STDIO_POOL_IDLE_TTL = 30 * 60  # 30 minutes


def _pool_key(session_key: str | None, command: str, args: list[str]) -> str:
    """Deterministic pool key for a session+command combination."""
    sk = session_key or "__default__"
    return f"{sk}|{command}|{json.dumps(list(args or []), sort_keys=True)}"


class _PooledSession:
    """A single stdio MCP session that can survive across tool calls."""

    def __init__(self, key: str, command: str, args: list[str], env: dict | None, cwd: str | None):
        self.key = key
        self._command = command
        self._args = list(args or [])
        self._env = dict(env) if env else None
        self._cwd = str(cwd) if cwd else None
        self._transport_cm = None
        self._session_cm = None
        self._session = None
        self._last_used = time.monotonic()
        self._dead = False

    def touch(self) -> None:
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    async def _start(self) -> None:
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
            cwd=self._cwd,
        )
        logger.info("MCP stdio session [%s] starting: %s %s", self.key, self._command, " ".join(self._args))
        self._transport_cm = stdio_client(server_params)
        read_stream, write_stream = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        self._dead = False
        self.touch()
        logger.info("MCP stdio session [%s] ready", self.key)

    async def _stop(self) -> None:
        logger.info("MCP stdio session [%s] stopping (idle %.0fs)", self.key, self.idle_seconds())
        try:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
        finally:
            try:
                if self._transport_cm is not None:
                    try:
                        await self._transport_cm.__aexit__(None, None, None)
                    except BaseException:
                        pass
            finally:
                pass
        self._session_cm = None
        self._transport_cm = None
        self._session = None
        self._dead = True
        logger.info("MCP stdio session [%s] stopped", self.key)

    async def call_tool(self, tool_name: str, arguments: dict, timeout_seconds: int):
        if self._session is None:
            await self._start()
        try:
            result = await self._session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            )
            self.touch()
            return result
        except BaseException:
            logger.warning("MCP stdio session [%s] call_tool failed, marking dead", self.key, exc_info=True)
            self._dead = True
            raise

    async def list_tools(self):
        if self._session is None:
            await self._start()
        try:
            result = await self._session.list_tools()
            self.touch()
            return result
        except BaseException:
            logger.warning("MCP stdio session [%s] list_tools failed, marking dead", self.key, exc_info=True)
            self._dead = True
            raise


class _StdioSessionPool:
    """Per-session-key pool of persistent stdio MCP sessions.

    Single background event-loop thread; asyncio.Lock for internal
    concurrency; threading.Lock for the sync public API.
    """

    def __init__(self):
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._sessions: dict[str, _PooledSession] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-stdio-pool", daemon=True)
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
        asyncio.run_coroutine_threadsafe(self._cleanup_loop(), self._loop)

    async def _cleanup_loop(self) -> None:
        """Periodically reap idle sessions every 120 seconds."""
        while True:
            await asyncio.sleep(120)
            try:
                removed = self.cleanup_idle()
                if removed:
                    logger.info("MCP stdio pool cleanup: reaped %d idle sessions (TTL=%ds)", removed, _STDIO_POOL_IDLE_TTL)
            except BaseException:
                logger.warning("MCP stdio pool cleanup error", exc_info=True)

    # ── sync helpers ──

    def _run_coroutine(self, coro):
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
        key = _pool_key(session_key, command, args)
        with self._sync_lock:
            existing = self._sessions.get(key)
            if existing is not None and not existing._dead:
                existing.touch()
                logger.debug("MCP stdio session [%s] reused (alive=%d total)", key, len(self._sessions))
                return existing
            if existing is not None and existing._dead:
                logger.info("MCP stdio session [%s] was dead, replacing", key)
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
                removed = self._sessions.pop(session.key, None)
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
    ):
        session = self._acquire_session(session_key, command, args, env, cwd)
        try:
            return self._run_coroutine(self._discover_with_retry(session))
        except BaseException:
            with self._sync_lock:
                removed = self._sessions.pop(session.key, None)
            if removed is not None:
                logger.error("MCP stdio session [%s] removed after unrecoverable error", session.key)
                self._run_coroutine(removed._stop())
            raise

    # ── async internals (run on background loop) ──

    async def _call_tool_with_retry(self, session, tool_name, arguments, timeout_seconds):
        try:
            return await session.call_tool(tool_name, arguments, timeout_seconds)
        except BaseException:
            logger.info("MCP stdio session [%s] died during call_tool, recreating...", session.key)
            await session._stop()
            await session._start()
            logger.info("MCP stdio session [%s] recreated, retrying call_tool", session.key)
            return await session.call_tool(tool_name, arguments, timeout_seconds)

    async def _discover_with_retry(self, session):
        try:
            return await session.list_tools()
        except BaseException:
            logger.info("MCP stdio session [%s] died during list_tools, recreating...", session.key)
            await session._stop()
            await session._start()
            logger.info("MCP stdio session [%s] recreated, retrying list_tools", session.key)
            return await session.list_tools()

    # ── lifecycle ──

    def cleanup_idle(self) -> int:
        """Reap sessions idle longer than TTL. Returns number removed."""
        with self._sync_lock:
            stale = [k for k, s in self._sessions.items() if s.idle_seconds() > _STDIO_POOL_IDLE_TTL]
        removed = 0
        for key in stale:
            with self._sync_lock:
                session = self._sessions.pop(key, None)
            if session is not None:
                logger.info("MCP stdio session [%s] idle %.0fs > TTL, reaping", key, session.idle_seconds())
                self._run_coroutine(session._stop())
                removed += 1
        return removed

    def shutdown(self) -> None:
        with self._sync_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
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


def _transport_supports_http_client() -> bool:
    if streamable_http_client is None:
        return False
    try:
        return "http_client" in inspect.signature(streamable_http_client).parameters
    except (TypeError, ValueError):
        return False


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


async def _ensure_server_reachable(server_url: str, *, timeout_seconds: int) -> None:
    parsed = urllib.parse.urlparse(server_url)
    host = parsed.hostname
    if not host:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_seconds)
    except (asyncio.TimeoutError, TimeoutError):
        raise MCPClientError("MCP server request timed out")
    except OSError:
        raise MCPClientError(f"Unable to connect to the MCP server at {host}:{port}")
    try:
        if reader:
            try:
                await asyncio.wait_for(reader.read(0), timeout=0.05)
            except (asyncio.TimeoutError, TimeoutError):
                pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
