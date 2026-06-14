from __future__ import annotations

import logging
import asyncio
from contextlib import AsyncExitStack, ExitStack

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)


class LangGraphPersistence:
    """Own the process-wide LangGraph checkpointer and store."""

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._async_stack = AsyncExitStack()
        self._async_lock: asyncio.Lock | None = None
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.backend = "memory"
        self.database_url = ""
        self.async_checkpointer = None
        self.async_store = None
        self.async_backend = "memory"
        self.async_database_url = ""

    def configure_postgres(self, database_url: str) -> None:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore

        conn_string = postgres_conn_string(database_url)
        if self.backend == "postgres" and self.database_url == conn_string:
            return

        stack = ExitStack()
        try:
            checkpointer = stack.enter_context(PostgresSaver.from_conn_string(conn_string))
            store = stack.enter_context(PostgresStore.from_conn_string(conn_string))
            checkpointer.setup()
            store.setup()
        except Exception:
            stack.close()
            raise

        self.close()
        self._stack = stack
        self.checkpointer = checkpointer
        self.store = store
        self.backend = "postgres"
        self.database_url = conn_string
        logger.info("Configured LangGraph persistence backend: postgres")

    async def aget_checkpointer(self):
        await self._ensure_async_backend()
        return self.async_checkpointer or self.checkpointer

    async def aget_store(self):
        await self._ensure_async_backend()
        return self.async_store or self.store

    async def _ensure_async_backend(self) -> None:
        if self.backend != "postgres":
            return
        if (
            self.async_backend == "postgres"
            and self.async_database_url == self.database_url
            and self.async_checkpointer is not None
            and self.async_store is not None
        ):
            return
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            if (
                self.async_backend == "postgres"
                and self.async_database_url == self.database_url
                and self.async_checkpointer is not None
                and self.async_store is not None
            ):
                return
            await self._close_async()

            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from langgraph.store.postgres.aio import AsyncPostgresStore

            stack = AsyncExitStack()
            try:
                checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(self.database_url))
                store = await stack.enter_async_context(AsyncPostgresStore.from_conn_string(self.database_url))
                await checkpointer.setup()
                await store.setup()
            except Exception:
                await stack.aclose()
                raise

            self._async_stack = stack
            self.async_checkpointer = checkpointer
            self.async_store = store
            self.async_backend = "postgres"
            self.async_database_url = self.database_url
            logger.info("Configured async LangGraph persistence backend: postgres")

    def close(self) -> None:
        self._stack.close()
        self._stack = ExitStack()
        if self.backend != "memory":
            self.checkpointer = InMemorySaver()
            self.store = InMemoryStore()
            self.backend = "memory"
            self.database_url = ""

    async def aclose(self) -> None:
        await self._close_async()
        self.close()

    async def _close_async(self) -> None:
        await self._async_stack.aclose()
        self._async_stack = AsyncExitStack()
        self.async_checkpointer = None
        self.async_store = None
        self.async_backend = "memory"
        self.async_database_url = ""


langgraph_persistence = LangGraphPersistence()


def postgres_conn_string(database_url: str) -> str:
    value = str(database_url or "").strip()
    if value.startswith("postgresql+psycopg://"):
        return "postgresql://" + value[len("postgresql+psycopg://") :]
    if value.startswith("postgresql+psycopg2://"):
        return "postgresql://" + value[len("postgresql+psycopg2://") :]
    if value.startswith("postgres+psycopg://"):
        return "postgres://" + value[len("postgres+psycopg://") :]
    if value.startswith("postgres+psycopg2://"):
        return "postgres://" + value[len("postgres+psycopg2://") :]
    return value


def configure_langgraph_persistence(database_url: str) -> None:
    langgraph_persistence.configure_postgres(database_url)


def close_langgraph_persistence() -> None:
    langgraph_persistence.close()


async def aclose_langgraph_persistence() -> None:
    await langgraph_persistence.aclose()


def get_workflow_checkpointer():
    return langgraph_persistence.checkpointer


def get_graph_memory_store():
    return langgraph_persistence.store


async def get_async_workflow_checkpointer():
    return await langgraph_persistence.aget_checkpointer()


async def get_async_graph_memory_store():
    return await langgraph_persistence.aget_store()
