from threading import Lock


class HTTPMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total_requests = 0
        self._total_seconds = 0.0

    def record(self, duration_seconds: float) -> None:
        with self._lock:
            self._total_requests += 1
            self._total_seconds += duration_seconds

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            total_requests = self._total_requests
            total_seconds = self._total_seconds

        average_latency_ms = (
            total_seconds * 1000 / total_requests if total_requests else 0.0
        )
        return {
            "total_requests": total_requests,
            "average_latency_ms": average_latency_ms,
        }
