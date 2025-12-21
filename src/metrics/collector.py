"""
Metrics Collection for Observability.

Simple in-memory metrics collection that can be exported
to various backends (logs, files, Prometheus, etc).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from collections import deque
import json

import structlog


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"     # Monotonic increasing
    GAUGE = "gauge"         # Point-in-time value
    HISTOGRAM = "histogram" # Distribution


@dataclass
class Metric:
    """A single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and exports metrics.

    Features:
    - In-memory storage with configurable history
    - Export to JSON file
    - Structured logging export
    - Simple aggregations
    """

    def __init__(
        self,
        max_history: int = 1000,
        export_interval: int = 60,
    ):
        """
        Initialize metrics collector.

        Args:
            max_history: Max data points to keep in memory
            export_interval: Seconds between auto-exports
        """
        self.max_history = max_history
        self.export_interval = export_interval

        self.logger = structlog.get_logger(__name__)

        # Metrics storage
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._history: deque[Metric] = deque(maxlen=max_history)

        # Timing
        self._start_time = datetime.utcnow()
        self._last_export = datetime.utcnow()

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """
        Increment a counter.

        Args:
            name: Metric name
            value: Amount to increment
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

        metric = Metric(
            name=name,
            value=self._counters[key],
            metric_type=MetricType.COUNTER,
            labels=labels or {},
        )
        self._history.append(metric)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Set a gauge value.

        Args:
            name: Metric name
            value: Current value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        self._gauges[key] = value

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {},
        )
        self._history.append(metric)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create unique key for metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0)

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all current metric values."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
        }

    def export_to_log(self) -> None:
        """Export metrics to structured log."""
        metrics = self.get_all_metrics()

        self.logger.info(
            "metrics_export",
            counters=metrics["counters"],
            gauges=metrics["gauges"],
            uptime=metrics["uptime_seconds"],
        )

        self._last_export = datetime.utcnow()

    def export_to_file(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        metrics = self.get_all_metrics()
        metrics["exported_at"] = datetime.utcnow().isoformat()

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.debug("metrics_exported_to_file", filepath=filepath)

    def should_export(self) -> bool:
        """Check if export interval has passed."""
        elapsed = (datetime.utcnow() - self._last_export).total_seconds()
        return elapsed >= self.export_interval


# Global metrics instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
