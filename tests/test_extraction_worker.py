import asyncio
import logging

import pytest

from gnosis.extraction_worker import BackgroundExtractionQueue


@pytest.mark.anyio
async def test_queue_runs_jobs_and_counts_processed() -> None:
    # Given: a queue with room and two independent jobs.
    queue = BackgroundExtractionQueue(max_concurrency=2, max_pending=10)
    ran: list[str] = []

    async def first() -> None:
        ran.append("first")

    async def second() -> None:
        ran.append("second")

    # When: both jobs are submitted and the queue drains.
    assert queue.submit(first, source_memory_ids=["id-1"])
    assert queue.submit(second, source_memory_ids=["id-2"])
    await queue.drain(drain_window_seconds=1)

    # Then: both ran off the submit path and the processed counter advanced.
    assert sorted(ran) == ["first", "second"]
    status = queue.status(mode="background")
    assert status.processed == 2
    assert status.failed == 0
    assert status.dropped == 0
    assert status.pending == 0


@pytest.mark.anyio
async def test_queue_bounds_concurrency() -> None:
    # Given: four jobs that hold their slot until released, on a queue
    # bounded to two concurrent jobs.
    queue = BackgroundExtractionQueue(max_concurrency=2, max_pending=10)
    release = asyncio.Event()
    running = 0
    peak = 0

    async def job() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        _ = await release.wait()
        running -= 1

    for index in range(4):
        assert queue.submit(job, source_memory_ids=[f"id-{index}"])

    # When: the event loop is driven while the jobs hold their slots.
    for _ in range(10):
        await asyncio.sleep(0)

    # Then: no more than two jobs ever ran at once, and all finish once
    # released.
    assert peak == 2
    assert queue.status(mode="background").pending == 4
    release.set()
    await queue.drain(drain_window_seconds=1)
    assert queue.status(mode="background").processed == 4


@pytest.mark.anyio
async def test_queue_overflow_drops_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a full queue (one pending job, max_pending=1).
    queue = BackgroundExtractionQueue(max_concurrency=1, max_pending=1)
    release = asyncio.Event()

    async def job() -> None:
        _ = await release.wait()

    assert queue.submit(job, source_memory_ids=["a"])

    # When: another extraction arrives.
    with caplog.at_level(logging.WARNING, logger="gnosis.extraction_worker"):
        accepted = queue.submit(job, source_memory_ids=["b"])

    # Then: it is dropped with a structured warning, never queued or raised.
    assert accepted is False
    assert "queue full" in caplog.text
    status = queue.status(mode="background")
    assert status.dropped == 1
    assert status.pending == 1
    release.set()
    await queue.drain(drain_window_seconds=1)


@pytest.mark.anyio
async def test_queue_skips_duplicate_pending_sources() -> None:
    # Given: a job pending for a set of source memory ids.
    queue = BackgroundExtractionQueue(max_concurrency=1, max_pending=10)
    release = asyncio.Event()
    runs = 0

    async def job() -> None:
        nonlocal runs
        runs += 1
        _ = await release.wait()

    assert queue.submit(job, source_memory_ids=["dup-1", "dup-2"])

    # When: the same sources are enqueued again while the first is pending.
    accepted = queue.submit(job, source_memory_ids=["dup-2", "dup-1"])

    # Then: the duplicate is skipped without counting as a drop.
    assert accepted is False
    assert queue.status(mode="background").dropped == 0
    release.set()
    await queue.drain(drain_window_seconds=1)
    assert runs == 1

    # Then: once the first job finished, the sources may be extracted again.
    assert queue.submit(job, source_memory_ids=["dup-1", "dup-2"])
    await queue.drain(drain_window_seconds=1)
    assert runs == 2


@pytest.mark.anyio
async def test_queue_counts_failures_and_keeps_serving(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a job that fails at runtime.
    queue = BackgroundExtractionQueue(max_concurrency=1, max_pending=10)

    async def bad() -> None:
        message = "extraction exploded"
        raise RuntimeError(message)

    async def good() -> None:
        return None

    # When: the failing and a healthy job run.
    with caplog.at_level(logging.WARNING, logger="gnosis.extraction_worker"):
        assert queue.submit(bad, source_memory_ids=["bad"])
        assert queue.submit(good, source_memory_ids=["good"])
        await queue.drain(drain_window_seconds=1)

    # Then: the failure is counted and logged; the healthy job still ran.
    assert "background extraction task failed" in caplog.text
    status = queue.status(mode="background")
    assert status.failed == 1
    assert status.processed == 1


@pytest.mark.anyio
async def test_drain_cancels_jobs_past_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a job that never finishes.
    queue = BackgroundExtractionQueue(max_concurrency=1, max_pending=10)

    async def stuck() -> None:
        _ = await asyncio.Event().wait()

    assert queue.submit(stuck, source_memory_ids=["stuck"])

    # When: the drain window elapses.
    with caplog.at_level(logging.WARNING, logger="gnosis.extraction_worker"):
        await queue.drain(drain_window_seconds=0.01)

    # Then: the job is cancelled with a warning instead of stalling shutdown.
    assert "drain timed out" in caplog.text
    for _ in range(10):
        await asyncio.sleep(0)
    assert queue.status(mode="background").pending == 0
