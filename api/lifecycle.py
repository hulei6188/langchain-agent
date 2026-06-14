from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from core.db.models import Run
from core.db.session import SessionLocal, init_db
from core.runtime.langgraph_persistence import aclose_langgraph_persistence


class AppLifecycleState:
    startup_error: str | None = None


def create_lifespan(settings, logger) -> tuple[AppLifecycleState, Any]:
    state = AppLifecycleState()

    @asynccontextmanager
    async def lifespan(app):
        if settings.jwt_secret == "change-me-in-production":
            logger.warning(
                "SECURITY WARNING: JWT_SECRET is using the insecure default value. "
                "Set a strong random secret via the JWT_SECRET environment variable."
            )
        try:
            init_db()
            cleanup_zombie_runs(logger)
            state.startup_error = None
        except Exception as exc:
            state.startup_error = str(exc)[:500]
            logger.exception("Database initialization failed; API started in degraded mode")
        try:
            yield
        finally:
            await aclose_langgraph_persistence()

    return state, lifespan


def cleanup_zombie_runs(logger) -> None:
    db = None
    try:
        db = SessionLocal()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        zombie_runs = (
            db.query(Run)
            .filter(Run.status == "running", Run.started_at < cutoff)
            .all()
        )
        if zombie_runs:
            for run in zombie_runs:
                run.status = "failed"
                run.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("Cleaned up %d zombie runs (stuck running >30min)", len(zombie_runs))
    except Exception:
        if db is not None:
            db.rollback()
        logger.exception("Failed to clean up zombie runs")
    finally:
        if db is not None:
            db.close()
