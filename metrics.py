"""Prometheus text-format metrics registry. Stdlib only."""
import threading
from typing import Dict, List, Optional, Tuple

_lock = threading.Lock()

_counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_gauges: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_histograms: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[float]] = {}
_meta: Dict[str, Dict] = {}


def register_counter(name: str, help_: str) -> None:
    _meta[name] = {"help": help_, "type": "counter"}


def register_gauge(name: str, help_: str) -> None:
    _meta[name] = {"help": help_, "type": "gauge"}


def register_histogram(name: str, help_: str, buckets: List[float]) -> None:
    _meta[name] = {"help": help_, "type": "histogram", "buckets": list(buckets)}


def _labels_key(labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def inc(name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
    k = (name, _labels_key(labels))
    with _lock:
        _counters[k] = _counters.get(k, 0.0) + value


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    k = (name, _labels_key(labels))
    with _lock:
        _gauges[k] = float(value)


def observe(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    buckets = _meta[name]["buckets"]
    k = (name, _labels_key(labels))
    with _lock:
        entry = _histograms.get(k)
        if entry is None:
            entry = [0.0, 0.0] + [0.0] * len(buckets)
            _histograms[k] = entry
        entry[0] += 1
        entry[1] += value
        for i, b in enumerate(buckets):
            if value <= b:
                entry[2 + i] += 1


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels_str(
    lk: Tuple[Tuple[str, str], ...],
    extra: Optional[Tuple[str, str]] = None,
) -> str:
    items = list(lk)
    if extra:
        items = items + [extra]
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in items) + "}"


def render() -> str:
    """Return Prometheus text exposition format (0.0.4)."""
    out: List[str] = []
    seen = set()

    def _emit_header(name: str) -> None:
        if name in seen:
            return
        m = _meta.get(name, {"help": "", "type": "untyped"})
        out.append(f"# HELP {name} {m.get('help', '')}")
        out.append(f"# TYPE {name} {m['type']}")
        seen.add(name)

    with _lock:
        for (name, lk), v in sorted(_counters.items()):
            _emit_header(name)
            out.append(f"{name}{_labels_str(lk)} {v}")
        for (name, lk), v in sorted(_gauges.items()):
            _emit_header(name)
            out.append(f"{name}{_labels_str(lk)} {v}")
        for (name, lk), entry in sorted(_histograms.items()):
            _emit_header(name)
            buckets = _meta[name]["buckets"]
            count, total = entry[0], entry[1]
            for i, b in enumerate(buckets):
                out.append(
                    f'{name}_bucket{_labels_str(lk, ("le", format(b, "g")))} {entry[2 + i]}'
                )
            out.append(f'{name}_bucket{_labels_str(lk, ("le", "+Inf"))} {count}')
            out.append(f"{name}_sum{_labels_str(lk)} {total}")
            out.append(f"{name}_count{_labels_str(lk)} {count}")
    return "\n".join(out) + "\n"


# ── Metric definitions ────────────────────────────────────────────────
# Buckets chosen to span fast-fail (<1s), normal Z.AI response (1-30s),
# slow thinking (30-60s), and hang-at-ceiling (60-120s, >120s).

_ZAI_DURATION_BUCKETS = [0.5, 1, 2, 5, 10, 20, 30, 60, 90, 120]

register_counter(
    "smart_proxy_zai_calls_total",
    "Z.AI upstream calls by outcome (status, failure_kind, model)",
)
register_histogram(
    "smart_proxy_zai_call_duration_seconds",
    "Z.AI upstream call wall-clock duration (s). Top bucket reveals hang-at-timeout.",
    _ZAI_DURATION_BUCKETS,
)
register_counter(
    "smart_proxy_zai_retries_total",
    "Z.AI retries attempted after a transient failure",
)
register_counter(
    "smart_proxy_zai_frontier_fallbacks_total",
    "Fallbacks from primary Z.AI model to configured frontier_fallback",
)
register_counter(
    "smart_proxy_zai_breaker_trips_total",
    "Circuit breaker trips (model exceeded failure threshold in window)",
)
register_counter(
    "smart_proxy_zai_breaker_skips_total",
    "Primary Z.AI calls skipped because breaker was open; routed to fallback",
)
register_gauge(
    "smart_proxy_zai_inflight",
    "Z.AI calls currently in flight (all models)",
)
register_gauge(
    "smart_proxy_zai_breaker_unhealthy",
    "1 if the per-model circuit breaker is currently tripped, else 0",
)
register_gauge(
    "smart_proxy_up",
    "Always 1 if the /metrics endpoint answered; for scrape liveness",
)
register_counter(
    "smart_proxy_local_gpu_fallbacks_total",
    "Requests routed to local GPU after all Z.AI tiers failed",
)
