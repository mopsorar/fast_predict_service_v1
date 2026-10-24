from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.metrics import HTTPMetrics


def test_initial_metrics() -> None:
    assert HTTPMetrics().snapshot() == {
        "total_requests": 0,
        "average_latency_ms": 0.0,
    }


def test_average_latency_in_milliseconds() -> None:
    metrics = HTTPMetrics()
    metrics.record(0.125)
    assert metrics.snapshot() == {"total_requests": 1, "average_latency_ms": 125.0}

    metrics.record(0.375)
    assert metrics.snapshot() == {"total_requests": 2, "average_latency_ms": 250.0}


def test_snapshot_does_not_expose_mutable_state() -> None:
    metrics = HTTPMetrics()
    snapshot = metrics.snapshot()
    snapshot["total_requests"] = 100
    assert metrics.snapshot()["total_requests"] == 0


def test_concurrent_updates_and_snapshots() -> None:
    metrics = HTTPMetrics()
    workers = 8
    updates_per_worker = 100
    barrier = Barrier(workers)

    def record_requests(_worker: int) -> None:
        barrier.wait()
        for _ in range(updates_per_worker):
            metrics.record(0.125)
            snapshot = metrics.snapshot()
            assert snapshot["total_requests"] > 0
            assert snapshot["average_latency_ms"] == pytest.approx(125.0)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(record_requests, range(workers)))

    assert metrics.snapshot() == {
        "total_requests": workers * updates_per_worker,
        "average_latency_ms": 125.0,
    }
