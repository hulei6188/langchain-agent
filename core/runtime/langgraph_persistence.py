from __future__ import annotations

import logging
from contextlib import ExitStack

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)


class LangGraphPersistence:
    """Own the process-wide LangGraph checkpointer and store."""

    def __init__(self) -> None:
        self._stack = ExitStack()
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.backend = "memory"
        self.database_url = ""

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

    def close(self) -> None:
        self._stack.close()
        self._stack = ExitStack()
        if self.backend != "memory":
            self.checkpointer = InMemorySaver()
            self.store = InMemoryStore()
            self.backend = "memory"
            self.database_url = ""


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


def get_workflow_checkpointer():
    return langgraph_persistence.checkpointer


def get_graph_memory_store():
    return langgraph_persistence.store
