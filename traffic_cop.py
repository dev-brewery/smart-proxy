"""Traffic cop: routes z.ai requests by intelligence requirement and concurrency state.

Tool-result acks (57%+ of Openclaw traffic) need zero frontier intelligence — route
them to high-concurrency models (GLM-4.6V, cap 10) instead of burning GLM-5.2/5-Turbo
slots (cap ~1).
"""
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import quality_gate

log = logging.getLogger("traffic_cop")

_model_semaphores: Dict[str, Optional[threading.BoundedSemaphore]] = {}
_sem_init_lock = threading.Lock()


def _get_semaphore(config: Any, model_name: str) -> Optional[threading.BoundedSemaphore]:
    if model_name not in _model_semaphores:
        with _sem_init_lock:
            if model_name not in _model_semaphores:
                mc = config.models.get(model_name)
                if mc and mc.concurrency_limit:
                    _model_semaphores[model_name] = threading.BoundedSemaphore(mc.concurrency_limit)
                else:
                    _model_semaphores[model_name] = None
    return _model_semaphores.get(model_name)


def route(
    config: Any,
    messages: List[Dict],
    resolved_model: Any,
) -> Tuple[Optional[str], str]:
    """Decide whether to reroute this request to a different z.ai model.

    Returns (override_model_name | None, reason_string).
    None means passthrough to the originally requested model.
    """
    if not resolved_model or not resolved_model.quality_gate:
        return None, "passthrough"

    msg_type = quality_gate.classify_last_msg(messages)

    if msg_type in ("tool_result", "tool_result_role"):
        ack_target = resolved_model.ack_model
        if ack_target:
            return ack_target, "tool_result_ack"

    return None, "passthrough"


def acquire(config: Any, model_name: str) -> bool:
    """Non-blocking acquire of a per-model concurrency slot.

    Returns True if acquired (caller MUST release in finally).
    Returns False if at capacity or no limit configured (caller should NOT release).
    """
    sem = _get_semaphore(config, model_name)
    if sem is None:
        return False
    acquired = sem.acquire(blocking=False)
    if not acquired:
        log.info("Concurrency limit reached for %s", model_name)
    return acquired


def release(model_name: str) -> None:
    """Release a per-model concurrency slot."""
    sem = _model_semaphores.get(model_name)
    if sem is not None:
        try:
            sem.release()
        except ValueError:
            log.warning("Semaphore over-release for %s", model_name)
