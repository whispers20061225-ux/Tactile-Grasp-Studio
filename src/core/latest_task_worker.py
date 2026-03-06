from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar


TTask = TypeVar("TTask")
TResult = TypeVar("TResult")


@dataclass
class WorkerResult(Generic[TTask, TResult]):
    task: TTask
    value: Optional[TResult]
    started_at: float
    finished_at: float
    error: Optional[str] = None


class LatestTaskWorker(Generic[TTask, TResult]):
    """Run only the newest submitted task and drop stale pending work."""

    def __init__(self, name: str, task_fn: Callable[[TTask], TResult]):
        self._name = name
        self._task_fn = task_fn
        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._busy = False
        self._pending_task: Optional[TTask] = None
        self._latest_result: Optional[WorkerResult[TTask, TResult]] = None
        self._overwrite_count = 0

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        thread: Optional[threading.Thread]
        with self._condition:
            self._running = False
            self._pending_task = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def submit(self, task: TTask) -> None:
        with self._condition:
            if self._busy or self._pending_task is not None:
                self._overwrite_count += 1
            self._pending_task = task
            self._condition.notify()

    def take_result(self) -> Optional[WorkerResult[TTask, TResult]]:
        with self._condition:
            result = self._latest_result
            self._latest_result = None
            return result

    def take_overwrite_count(self) -> int:
        with self._condition:
            count = self._overwrite_count
            self._overwrite_count = 0
            return count

    def get_backlog_size(self) -> int:
        with self._condition:
            return int(self._busy) + int(self._pending_task is not None)

    def is_busy(self) -> bool:
        with self._condition:
            return self._busy

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending_task is None:
                    self._condition.wait()
                if not self._running:
                    self._busy = False
                    return
                task = self._pending_task
                self._pending_task = None
                self._busy = True

            started_at = time.time()
            value: Optional[TResult] = None
            error: Optional[str] = None
            try:
                value = self._task_fn(task)
            except Exception as exc:  # pragma: no cover - exercised in runtime
                error = str(exc)
            finished_at = time.time()

            with self._condition:
                self._latest_result = WorkerResult(
                    task=task,
                    value=value,
                    started_at=started_at,
                    finished_at=finished_at,
                    error=error,
                )
                self._busy = False
