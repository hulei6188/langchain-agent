"""Thread-safe cancellation registry for in-flight agent runs."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cancel_events: dict[int, threading.Event] = {}
_active_providers: dict[int, Any] = {}


def register_run(run_id: int, provider: Any | None = None) -> threading.Event:
    """Register a run for cancellation tracking.

    Returns the Event that will be set when cancellation is requested.
    """
    event = threading.Event()
    with _lock:
        _cancel_events[run_id] = event
        if provider is not None:
            _active_providers[run_id] = provider
    logger.debug("Registered run %s for cancellation tracking", run_id)
    return event


def cancel_run(run_id: int) -> bool:
    """Signal cancellation for a run.

    Sets the cancel event AND closes the provider's active HTTP connection
    (if any) to unblock streaming reads. Returns True if the run was found.
    """
    with _lock:
        event = _cancel_events.get(run_id)
        provider = _active_providers.pop(run_id, None)
    if event is None:
        logger.warning("Cancel requested for unknown run %s", run_id)
        return False
    event.set()
    if provider is not None:
        try:
            provider.cancel_active_request()
        except Exception:
            logger.exception("Error closing active HTTP connection for run %s", run_id)
    logger.info("Run %s cancelled", run_id)
    return True


def is_cancelled(run_id: int) -> bool:
    """Check whether a run has been cancelled."""
    with _lock:
        event = _cancel_events.get(run_id)
    if event is None:
        return False
    return event.is_set()


def unregister_run(run_id: int) -> None:
    """Remove a run from the cancellation registry."""
    with _lock:
        _cancel_events.pop(run_id, None)
        _active_providers.pop(run_id, None)
    logger.debug("Unregistered run %s from cancellation tracking", run_id)
