from __future__ import annotations

import threading


class RunEventReplayBuffer:
    def __init__(self, *, max_events: int = 2000) -> None:
        self.max_events = max_events
        self._logs: dict[int, list[str]] = {}
        self._lock = threading.Lock()

    def append(self, run_id: int, event: str) -> None:
        with self._lock:
            log = self._logs.setdefault(run_id, [])
            log.append(event)
            if len(log) > self.max_events:
                self._logs[run_id] = log[-self.max_events:]

    def since(self, run_id: int, index: int) -> tuple[list[str], int]:
        with self._lock:
            log = self._logs.get(run_id, [])
            return log[index:], len(log)

    def snapshot(self, run_id: int) -> list[str]:
        with self._lock:
            return list(self._logs.get(run_id, []))

    def cleanup(self, run_id: int) -> None:
        with self._lock:
            self._logs.pop(run_id, None)


run_event_buffer = RunEventReplayBuffer()
