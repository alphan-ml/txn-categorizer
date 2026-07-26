"""In-memory metrics. Swap for Prometheus client in production deploy."""
import time
from collections import deque
from statistics import quantiles


class Metrics:
    def __init__(self, window: int = 5000):
        self.events = deque(maxlen=window)

    def record(self, path: str, latency_ms: float, confidence: float):
        self.events.append({"path": path, "latency_ms": latency_ms,
                            "confidence": confidence, "ts": time.time()})

    def summary(self) -> dict:
        if not self.events:
            return {"total": 0}
        lat = sorted(e["latency_ms"] for e in self.events)
        llm = sum(1 for e in self.events if e["path"] == "llm")
        n = len(self.events)
        pct = (lambda q: quantiles(lat, n=100)[q - 1]) if n >= 2 else (lambda q: lat[0])
        return {
            "total": n,
            "fallback_rate": round(llm / n, 4),
            "latency_p50_ms": round(pct(50), 2),
            "latency_p95_ms": round(pct(95), 2),
            "latency_p99_ms": round(pct(99), 2),
            "window_start_ts": self.events[0]["ts"],
        }


METRICS = Metrics()
