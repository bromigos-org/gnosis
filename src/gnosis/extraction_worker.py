"""In-process background queue for ingest-time fact extraction.

With ``GNOSIS_FACT_EXTRACTION_MODE=background``, the write request returns
right after the verbatim facts land and the extraction LLM call plus the
extracted-fact writes run here, on an asyncio task queue. The queue mirrors
the buffered-write posture: a bounded pending set with bounded concurrency,
and on overflow the job is dropped with a structured warning - extraction is
an enhancement and must never backpressure message ingestion.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Final, Literal

from gnosis.models import ExtractionQueueStatus

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

type ExtractionJob = Callable[[], Awaitable[None]]


class BackgroundExtractionQueue:
    """Bounded asyncio queue that runs extraction jobs off the request path.

    Pending is bounded by ``GNOSIS_FACT_EXTRACTION_MAX_PENDING`` (drop with a
    structured warning on overflow, never backpressure) and execution by
    ``GNOSIS_FACT_EXTRACTION_MAX_CONCURRENCY``.

    Idempotency guard: an in-process set of pending ``source_memory_ids``
    keys, cleared when a job finishes. Source ids are freshly minted UUIDs
    per add request, so a duplicate key can only arise from a double enqueue
    inside this process; a graph existence check would spend one Cypher
    round trip per add guarding a case that cannot occur across restarts.
    """

    def __init__(self, *, max_concurrency: int, max_pending: int) -> None:
        self._max_concurrency: int = max_concurrency
        self._max_pending: int = max_pending
        self._concurrency: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._pending_sources: set[frozenset[str]] = set()
        self._processed: int = 0
        self._failed: int = 0
        self._dropped: int = 0

    def submit(self, job: ExtractionJob, *, source_memory_ids: Sequence[str]) -> bool:
        """Queue one extraction job; False when deduplicated or dropped."""
        key = frozenset(source_memory_ids)
        if key and key in self._pending_sources:
            _LOGGER.info(
                "background extraction already pending for sources; skipping",
                extra={"source_memory_ids": sorted(key)},
            )
            return False
        if len(self._tasks) >= self._max_pending:
            self._dropped += 1
            _LOGGER.warning(
                "background extraction queue full; dropping extraction",
                extra={
                    "dropped_total": self._dropped,
                    "max_pending": self._max_pending,
                    "source_memory_ids": sorted(key),
                },
            )
            return False
        self._pending_sources.add(key)
        task = asyncio.get_running_loop().create_task(self._run(job, key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run(self, job: ExtractionJob, key: frozenset[str]) -> None:
        try:
            async with self._concurrency:
                await job()
        except Exception:
            # Enhancement path: any failure is counted and logged, never
            # raised - the verbatim add already succeeded.
            self._failed += 1
            _LOGGER.exception(
                "background extraction task failed; verbatim facts kept",
                extra={"source_memory_ids": sorted(key)},
            )
        else:
            self._processed += 1
            _LOGGER.info(
                "background extraction task completed",
                extra={"source_memory_ids": sorted(key)},
            )
        finally:
            self._pending_sources.discard(key)

    async def drain(self, *, drain_window_seconds: float) -> None:
        """Give pending jobs a bounded window to finish, then cancel the rest."""
        tasks = {task for task in self._tasks if not task.done()}
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=drain_window_seconds)
        if not pending:
            return
        _LOGGER.warning(
            "background extraction drain timed out; cancelling remaining tasks",
            extra={
                "cancelled": len(pending),
                "drain_window_seconds": drain_window_seconds,
            },
        )
        for task in pending:
            _ = task.cancel()

    def status(
        self,
        *,
        mode: Literal["sync", "background"],
    ) -> ExtractionQueueStatus:
        """Snapshot the queue for /v1/diagnostics."""
        return ExtractionQueueStatus(
            mode=mode,
            max_concurrency=self._max_concurrency,
            max_pending=self._max_pending,
            pending=len(self._tasks),
            processed=self._processed,
            failed=self._failed,
            dropped=self._dropped,
        )
