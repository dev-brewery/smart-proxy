"""Quality gate for Z.AI responses — evaluates via the model router."""
import hashlib
import json
import logging
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import metrics

log = logging.getLogger("quality_gate")
reqlog_logger = logging.getLogger("reqlog")


# ── REQLOG instrumentation ────────────────────────────────────────────
#
# One-shot 24-hour observability pass so we can characterize Openclaw
# traffic shape and correlate z.ai transport failures with in-flight
# concurrency. Emits one structured JSONL line per significant event,
# tagged `REQLOG` for easy grep/jq. Never raises — logging must not
# break the request path.

_zai_inflight_lock = threading.Lock()
_zai_inflight_count = 0


def _zai_inflight_inc() -> int:
    global _zai_inflight_count
    with _zai_inflight_lock:
        _zai_inflight_count += 1
        n = _zai_inflight_count
    metrics.set_gauge("smart_proxy_zai_inflight", n)
    return n


def _zai_inflight_dec() -> int:
    global _zai_inflight_count
    with _zai_inflight_lock:
        _zai_inflight_count -= 1
        n = _zai_inflight_count
    metrics.set_gauge("smart_proxy_zai_inflight", n)
    return n


def reqlog(event: Dict[str, Any]) -> None:
    try:
        reqlog_logger.info("REQLOG %s", json.dumps(event, ensure_ascii=True, default=str))
    except Exception:
        pass


def classify_last_msg(messages: List[Dict]) -> str:
    if not messages:
        return "empty"
    last = messages[-1]
    role = last.get("role", "")
    content = last.get("content", "")
    if role == "tool":
        return "tool_result_role"
    if isinstance(content, list):
        has_tr = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        has_text = any(isinstance(b, dict) and b.get("type") == "text" for b in content)
        if has_tr and not has_text:
            return "tool_result"
        if has_tr and has_text:
            return "tool_result_plus_text"
        if has_text:
            return f"{role}_text_blocks"
        return "other_blocks"
    if isinstance(content, str):
        return f"{role}_text"
    return "other"


def detect_images(messages: List[Dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def hash_tool_names(tools: List[Dict]) -> str:
    if not tools:
        return ""
    names = sorted(t.get("name", "") for t in tools if isinstance(t, dict) and t.get("name"))
    if not names:
        return ""
    return hashlib.sha1(",".join(names).encode("utf-8")).hexdigest()[:8]


def _classify_failure_kind(exc: Exception) -> str:
    s = str(exc)
    if "Connection reset" in s or "ECONNRESET" in s:
        return "ECONNRESET"
    if "Broken pipe" in s or "EPIPE" in s:
        return "EPIPE"
    if "Remote end closed" in s:
        return "REMOTE_CLOSED"
    lower = s.lower()
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT"
    return "OTHER_TRANSPORT"


# ── Per-Model Circuit Breaker ─────────────────────────────────────────
#
# Tracks rolling failure count per Z.AI model. When failures cross the
# threshold within the window, the model is marked unhealthy for a short
# cooldown. call_zai() skips the primary and routes straight to
# frontier_fallback while unhealthy, so we stop hammering z.ai during
# its bad windows.

_breaker_lock = threading.Lock()
_breaker_state: Dict[str, Dict[str, Any]] = {}

_BREAKER_WINDOW_SECONDS = 60.0
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECONDS = 30.0


def _breaker_record(model: str, success: bool) -> None:
    if not model:
        return
    now = time.monotonic()
    with _breaker_lock:
        st = _breaker_state.setdefault(model, {"failures": [], "unhealthy_until": 0.0})
        cutoff = now - _BREAKER_WINDOW_SECONDS
        st["failures"] = [t for t in st["failures"] if t >= cutoff]
        if success:
            return
        st["failures"].append(now)
        if len(st["failures"]) >= _BREAKER_THRESHOLD and now >= st["unhealthy_until"]:
            st["unhealthy_until"] = now + _BREAKER_COOLDOWN_SECONDS
            reqlog({
                "event": "breaker_trip",
                "model": model,
                "failures_in_window": len(st["failures"]),
                "cooldown_s": _BREAKER_COOLDOWN_SECONDS,
            })
            metrics.inc("smart_proxy_zai_breaker_trips_total", {"model": model})
            metrics.set_gauge(
                "smart_proxy_zai_breaker_unhealthy", 1, {"model": model}
            )


def _breaker_is_unhealthy(model: str) -> bool:
    if not model:
        return False
    now = time.monotonic()
    with _breaker_lock:
        st = _breaker_state.get(model)
        if not st:
            return False
        return now < st["unhealthy_until"]


def breaker_snapshot() -> Dict[str, Dict[str, Any]]:
    """Read-only view of per-model breaker state for /v1/status and /metrics.

    Also refreshes the smart_proxy_zai_breaker_unhealthy gauge as a side
    effect so expired cooldowns show up as healed without a dedicated
    background thread.
    """
    now = time.monotonic()
    out: Dict[str, Dict[str, Any]] = {}
    with _breaker_lock:
        for model, st in _breaker_state.items():
            cutoff = now - _BREAKER_WINDOW_SECONDS
            recent = [t for t in st["failures"] if t >= cutoff]
            remaining = max(0.0, st["unhealthy_until"] - now)
            out[model] = {
                "failures_in_window": len(recent),
                "unhealthy": remaining > 0,
                "unhealthy_remaining_s": round(remaining, 1),
            }
    for model, info in out.items():
        metrics.set_gauge(
            "smart_proxy_zai_breaker_unhealthy",
            1 if info["unhealthy"] else 0,
            {"model": model},
        )
    return out

# ── Deterministic Tool Call Validation ────────────────────────────────

VALID_AGENT_IDS = {"assistant", "manager", "homelab", "church", "aquatics", "gardeners"}
SPECIALIST_AGENT_IDS = VALID_AGENT_IDS - {"manager"}
LIKELY_REQUIRED_PARAMS = {"agentId", "task", "sessionKey", "path", "command"}

_DELEGATION_RE = re.compile(
    r"(?:delegat|rout|send|dispatch|forward|assign)\w*\s+to\s+(\w+)",
    re.IGNORECASE,
)


def check_tool_calls(
    zai_response: Dict, request_messages: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Deterministic validation of tool_use blocks. Returns judge-shaped verdict."""
    content = zai_response.get("content", [])
    if not isinstance(content, list):
        return {"pass": True, "failures": [], "feedback": ""}

    tool_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    if not tool_blocks:
        return {"pass": True, "failures": [], "feedback": ""}

    # Extract assistant text for Rule 3 (text-tool consistency)
    response_text = " ".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )

    failures = []
    feedback_parts = []

    for block in tool_blocks:
        name = block.get("name", "")
        inputs = block.get("input", {}) or {}

        if name == "sessions_spawn":
            # Rule 1: agentId must be present and valid
            agent_id = inputs.get("agentId")
            if not agent_id or agent_id not in VALID_AGENT_IDS:
                failures.append("MISSING_AGENT_ID")
                feedback_parts.append(
                    "The sessions_spawn call is missing a valid agentId. "
                    "You said you were delegating to a specific agent — include "
                    "agentId in the tool call. Valid agents: "
                    + ", ".join(sorted(VALID_AGENT_IDS - {"manager"})) + "."
                )
                continue  # stop at first failure for this tool call

            # Rule 5: task must be present and non-empty
            task = inputs.get("task")
            if not task or (isinstance(task, str) and not task.strip()):
                failures.append("MISSING_TASK")
                feedback_parts.append(
                    "The sessions_spawn call has no task description. "
                    "Include a detailed task string so the specialist agent knows what to do."
                )
                continue

            # Rule 2: runtime must be "acp" for specialist agents
            if agent_id in SPECIALIST_AGENT_IDS:
                runtime = inputs.get("runtime")
                if runtime != "acp":
                    failures.append("WRONG_RUNTIME")
                    feedback_parts.append(
                        "When delegating to a specialist agent, you must use "
                        "runtime: 'acp' so the agent gets its own tool permissions. "
                        "Change runtime to 'acp'."
                    )
                    continue

            # Rule 3: text-tool consistency
            delegation_target = _extract_delegation_target(response_text)
            if delegation_target and delegation_target != agent_id:
                failures.append("TEXT_TOOL_MISMATCH")
                feedback_parts.append(
                    f"Your text says you're routing to {delegation_target} but "
                    f"the sessions_spawn call targets {agent_id}. Fix the agentId to match."
                )
                continue

        # Rule 4: general null required params (any tool_use)
        for param_name in LIKELY_REQUIRED_PARAMS:
            if param_name in inputs and inputs[param_name] is None:
                failures.append("NULL_REQUIRED_PARAM")
                feedback_parts.append(
                    f"Tool call to {name} has null value for required parameter "
                    f"{param_name}. Provide a valid value."
                )
                break  # one failure per tool call

    if failures:
        return {
            "pass": False,
            "failures": failures,
            "feedback": " ".join(feedback_parts),
        }
    return {"pass": True, "failures": [], "feedback": ""}


def _extract_delegation_target(text: str) -> Optional[str]:
    """Extract agent name from delegation language in response text."""
    match = _DELEGATION_RE.search(text)
    if match:
        candidate = match.group(1).lower()
        if candidate in VALID_AGENT_IDS:
            return candidate
    return None


# ── Tool-Nudge: Qwen pre-classification for action requests ─────────

_TOOL_MATCH_SYSTEM = (
    "Match the user's request to available tools. "
    "Reply with comma-separated tool names, or NONE if no tool is needed."
)


def _build_tool_summary(tools: List[Dict]) -> str:
    """Compact tool list — names only, minimal tokens."""
    names = [t.get("name", "") for t in tools if t.get("name")]
    return ", ".join(names) if names else ""


def _match_tools(
    messages: List[Dict],
    tools: List[Dict],
    helper_url: str = "http://localhost:8094",  # dedicated tool-match agent
    timeout_s: float = 8.0,  # 8s budget (was 5s, median on old agent was 4.8s)
) -> Optional[List[str]]:
    """Ask Qwen which tool(s) match the user's request.
    Returns list of tool names, or None on failure/NONE/timeout."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None
    last_text = _extract_text(user_msgs[-1])[:500]
    if not last_text.strip():
        return None

    tool_summary = _build_tool_summary(tools)
    if not tool_summary:
        return None

    valid_names = {t.get("name", "") for t in tools if t.get("name")}

    prompt = f"Tools: {tool_summary}\nRequest: {last_text}\nWhich tool?"
    data = json.dumps({
        "model": "any",
        "messages": [
            {"role": "system", "content": _TOOL_MATCH_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 30,
        "temperature": 0.0,
    }).encode("utf-8")

    endpoint = helper_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read())
            answer = body["choices"][0]["message"]["content"].strip()
            if answer.upper() == "NONE":
                log.info("Tool match: NONE (not an action request)")
                return None
            # Parse comma-separated tool names, validate against payload
            candidates = [t.strip() for t in answer.split(",")]
            matched = [c for c in candidates if c in valid_names]
            if matched:
                log.info("Tool match: %s", matched)
                return matched
            log.warning("Tool match: no valid names in '%s'", answer)
            return None
    except Exception as exc:
        log.warning("Tool match failed (%s), skipping nudge", exc)
        return None


def _inject_tool_nudge(payload: Dict, tool_names: List[str]) -> Dict:
    """Return a shallow-copied payload with a tool-nudge message inserted."""
    nudged = dict(payload)
    nudged_messages = list(payload.get("messages", []))

    tools_str = ", ".join(tool_names)
    nudge_msg = {
        "role": "user",
        "content": (
            f"[SYSTEM GUIDANCE] The user is requesting an action. You MUST respond "
            f"with a tool_use block. Use one of: {tools_str}. "
            f"Do NOT describe what you would do — call the tool."
        ),
    }
    # Insert before the last message so user's request stays final
    nudged_messages.insert(max(len(nudged_messages) - 1, 0), nudge_msg)
    nudged["messages"] = nudged_messages
    return nudged


def _build_retry_feedback(
    verdict: Dict, tools: List[Dict], matched_tools: Optional[List[str]]
) -> str:
    """Build structured retry feedback based on failure type."""
    failures = verdict.get("failures", [])

    if "MISSING_TOOLS" in failures and matched_tools:
        tools_str = ", ".join(matched_tools)
        return (
            f"Your previous response described an action instead of performing it. "
            f"You have these tools available: {tools_str}. "
            f"Respond with a tool_use block for the appropriate tool. "
            f"Do not narrate — call the tool now."
        )
    elif "MISSING_TOOLS" in failures and tools:
        tool_names = [t.get("name", "") for t in tools if t.get("name")][:10]
        return (
            f"Your previous response described an action instead of performing it. "
            f"Available tools: {', '.join(tool_names)}. "
            f"Respond with a tool_use block, not a text description."
        )
    else:
        return (
            f"Your previous response was inadequate: "
            f"{verdict.get('feedback', 'unknown issue')}. "
            f"Please provide a complete, well-formed response."
        )


JUDGE_SYSTEM = """You are a response quality evaluator. Given a conversation and an AI response, evaluate whether the response is complete and well-formed. Check for:

1. EMPTY — Response has no meaningful content.

2. MISSING_TOOLS — The response describes what it WOULD do instead of actually calling a tool.
   The agent being evaluated has access to tools listed in the AVAILABLE TOOLS section below (if provided).
   Flag MISSING_TOOLS ONLY when ALL of these are true:
     a) The user explicitly asked for a concrete ACTION (e.g. "check the logs", "delegate to homelab", "read that file", "run the backup", "search for X", "send a message")
     b) The response contains NO tool_use blocks
     c) The response instead narrates what it would/could/should do (e.g. "I would run...", "I can check...", "You could try...", "Here's how to...")
   Do NOT flag MISSING_TOOLS for: knowledge questions, opinions, explanations, summaries, analysis, greetings, math, comparisons, or anything answerable with text alone.

3. TRUNCATED — Response cuts off mid-sentence or mid-thought.

4. INCOHERENT — Response does not address what was asked.

Reply with JSON only: {"pass": true/false, "failures": [], "feedback": "brief fix instruction"}"""


# ── Request Classification ────────────────────────────────────────────

def classify(messages: List[Dict], headers: Optional[Dict] = None) -> bool:
    """Return True if the response should be evaluated, False to skip."""
    if headers and headers.get("x-skip-quality-gate", "").lower() == "true":
        return False

    if not messages:
        return False

    last = messages[-1]

    # Skip tool results — just data flowing back
    if last.get("role") == "tool" or _is_tool_result(last):
        return False

    # Skip heartbeats
    content_str = _extract_text(last)
    if "HEARTBEAT" in content_str or "HEARTBEAT_OK" in content_str:
        return False

    # Skip very short acks
    if last.get("role") == "user" and len(content_str.split()) < 5:
        lower = content_str.lower().strip()
        if lower in ("ok", "thanks", "got it", "yes", "no", "continue", "go ahead"):
            return False

    return True


def _is_tool_result(msg: Dict) -> bool:
    """Check if message contains tool_result content blocks (Anthropic format)."""
    content = msg.get("content")
    if isinstance(content, list):
        return any(b.get("type") == "tool_result" for b in content if isinstance(b, dict))
    return False


def _extract_text(msg: Dict) -> str:
    """Extract text content from a message (handles both string and block formats)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


# ── Z.AI Call ─────────────────────────────────────────────────────────

_429_RETRIABLE_CODES = {"1302", "1303", "1305"}
_RETRIABLE_HTTP_STATUSES = {500, 502, 503, 504}
_RETRY_BASE_DELAYS = [1.0, 3.0, 7.0]
_RETRY_JITTER = 0.35


def _jittered(base: float) -> float:
    return max(0.1, base + random.uniform(-_RETRY_JITTER, _RETRY_JITTER) * base)


def _is_retriable(status: int, err_code: Optional[str]) -> bool:
    if status == 0:
        return True  # transport failure (reset, timeout, remote closed)
    if status == 429 and err_code in _429_RETRIABLE_CODES:
        return True
    if status in _RETRIABLE_HTTP_STATUSES:
        return True
    return False


def _do_single_zai_call(
    endpoint: str,
    api_key: str,
    data: bytes,
    ctx: Any,
) -> Tuple[int, Dict, str, Optional[str]]:
    """Single HTTP call to z.ai. Returns (status, body, zai_request_id, err_code)."""
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", api_key)
    req.add_header("anthropic-version", "2023-06-01")

    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            raw = resp.read()
            zai_rid = resp.headers.get("x-request-id", "") or ""
            body = json.loads(raw)
            return resp.status, body, zai_rid, None
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        try:
            zai_rid = exc.headers.get("x-request-id", "") if exc.headers else ""
        except Exception:
            zai_rid = ""
        try:
            body = json.loads(body_bytes)
        except Exception:
            body = {"error": {"message": body_bytes.decode("utf-8", errors="replace")[:500]}}
        err_code = None
        if isinstance(body, dict):
            err = body.get("error") or {}
            if isinstance(err, dict) and err.get("code") is not None:
                err_code = str(err["code"])
        log.error("Z.AI returned %d: %s", exc.code, str(body)[:200])
        return exc.code, body, zai_rid, err_code
    except Exception as exc:
        kind = _classify_failure_kind(exc)
        log.error("Z.AI call failed (%s): %s", kind, type(exc).__name__)
        return 0, {"error": {"kind": kind, "message": f"upstream transport: {kind}"}}, "", None


def _call_zai_attempt(
    endpoint: str,
    api_key: str,
    send_payload: Dict,
    ctx: Any,
    *,
    req_id: Optional[str],
    attempt: int,
    retry_num: Any,
    tool_count: int,
    model_name: str,
) -> Tuple[int, Dict, Optional[str], Optional[str]]:
    """One z.ai call with full reqlog + breaker bookkeeping.

    Returns (status, body, err_code, failure_kind).
    """
    data = json.dumps(send_payload).encode("utf-8")
    inflight_at_entry = _zai_inflight_inc()
    t0 = time.monotonic()
    status, body, zai_rid, err_code = _do_single_zai_call(endpoint, api_key, data, ctx)
    duration_ms = int((time.monotonic() - t0) * 1000)
    _zai_inflight_dec()

    failure_kind = None
    stop_reason = None
    input_tokens = None
    output_tokens = None
    if status == 200 and isinstance(body, dict):
        stop_reason = body.get("stop_reason")
        usage = body.get("usage") or {}
        if isinstance(usage, dict):
            output_tokens = usage.get("output_tokens")
            input_tokens = usage.get("input_tokens")
    elif status == 0:
        err = body.get("error") if isinstance(body, dict) else {}
        failure_kind = (err or {}).get("kind") or _classify_failure_kind(
            Exception(str((err or {}).get("message", "")))
        )
    elif status != 200:
        failure_kind = f"HTTP_{status}"

    reqlog({
        "event": "call_zai",
        "req_id": req_id,
        "attempt": attempt,
        "retry_num": retry_num,
        "model": send_payload.get("model", model_name),
        "payload_bytes": len(data),
        "tool_count": tool_count,
        "inflight_at_entry": inflight_at_entry,
        "duration_ms": duration_ms,
        "status": status,
        "failure_kind": failure_kind,
        "zai_err_code": err_code,
        "zai_request_id": zai_rid,
        "stop_reason": stop_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })

    metric_model = send_payload.get("model", model_name) or "unknown"
    metrics.inc(
        "smart_proxy_zai_calls_total",
        {
            "model": metric_model,
            "status": str(status),
            "failure_kind": failure_kind or "none",
        },
    )
    metrics.observe(
        "smart_proxy_zai_call_duration_seconds",
        duration_ms / 1000.0,
        {"model": metric_model, "status": str(status)},
    )

    _breaker_record(send_payload.get("model", model_name), success=(status == 200))
    return status, body, err_code, failure_kind


def call_zai(
    url: str,
    api_key: str,
    payload: Dict,
    req_id: Optional[str] = None,
    attempt: int = 1,
    frontier_fallback: Optional[str] = None,
) -> Tuple[int, Dict]:
    """Forward an Anthropic Messages API request to Z.AI.

    Retries transient transport/HTTP errors (429-retriable, 5xx, socket
    failures) with jittered backoff. A per-model circuit breaker short-
    circuits to frontier_fallback while the primary is unhealthy.
    """
    endpoint = url.rstrip("/") + "/v1/messages"
    ctx = ssl.create_default_context()

    send_payload = dict(payload)
    send_payload["stream"] = False

    tool_count = len(payload.get("tools", []) or [])
    model_name = payload.get("model", "")

    # Circuit breaker: if primary is unhealthy and we have a fallback,
    # skip the primary entirely and go straight to fallback. Avoids
    # burning 3 retries every request during a bad z.ai window.
    if (
        frontier_fallback
        and model_name
        and model_name != frontier_fallback
        and _breaker_is_unhealthy(model_name)
    ):
        reqlog({
            "event": "breaker_skip_primary",
            "req_id": req_id,
            "from_model": model_name,
            "to_model": frontier_fallback,
        })
        metrics.inc(
            "smart_proxy_zai_breaker_skips_total",
            {"from_model": model_name, "to_model": frontier_fallback},
        )
        send_payload["model"] = frontier_fallback
        status, body, _, _ = _call_zai_attempt(
            endpoint, api_key, send_payload, ctx,
            req_id=req_id, attempt=attempt, retry_num="breaker_fallback",
            tool_count=tool_count, model_name=frontier_fallback,
        )
        if status == 200:
            payload["model"] = frontier_fallback
        return status, body

    last_status = 0
    last_body: Dict = {}

    max_retries = len(_RETRY_BASE_DELAYS)
    for retry_num in range(1 + max_retries):
        status, body, err_code, _ = _call_zai_attempt(
            endpoint, api_key, send_payload, ctx,
            req_id=req_id, attempt=attempt, retry_num=retry_num,
            tool_count=tool_count, model_name=model_name,
        )

        if status == 200:
            return status, body

        last_status, last_body = status, body

        if not _is_retriable(status, err_code):
            break

        if retry_num < max_retries:
            delay = _jittered(_RETRY_BASE_DELAYS[retry_num])
            reqlog({
                "event": "zai_retry",
                "req_id": req_id,
                "model": send_payload.get("model", model_name),
                "retry_num": retry_num + 1,
                "delay_s": round(delay, 2),
                "status": status,
                "zai_err_code": err_code,
            })
            metrics.inc(
                "smart_proxy_zai_retries_total",
                {
                    "model": send_payload.get("model", model_name) or "unknown",
                    "status": str(status),
                },
            )
            time.sleep(delay)

    # Frontier fallback — one attempt to the secondary model.
    if frontier_fallback and send_payload.get("model") != frontier_fallback:
        reqlog({
            "event": "frontier_fallback",
            "req_id": req_id,
            "from_model": send_payload.get("model", model_name),
            "to_model": frontier_fallback,
        })
        metrics.inc(
            "smart_proxy_zai_frontier_fallbacks_total",
            {
                "from_model": send_payload.get("model", model_name) or "unknown",
                "to_model": frontier_fallback,
            },
        )
        send_payload["model"] = frontier_fallback
        status, body, _, _ = _call_zai_attempt(
            endpoint, api_key, send_payload, ctx,
            req_id=req_id, attempt=attempt, retry_num="fallback",
            tool_count=tool_count, model_name=frontier_fallback,
        )
        if status == 200:
            payload["model"] = frontier_fallback
        return status, body

    return last_status, last_body


# ── Judge Evaluation ──────────────────────────────────────────────────

HELPER_URL = "http://localhost:8094"  # dedicated tool-match agent


def _call_judge(endpoint: str, payload: Dict, label: str) -> Optional[Dict]:
    """Send judge prompt to an endpoint. Returns verdict dict or None on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            content = body["choices"][0]["message"]["content"]
            verdict = json.loads(content)
            log.info(
                "Judge verdict (%s): pass=%s failures=%s",
                label,
                verdict.get("pass"),
                verdict.get("failures", []),
            )
            return verdict
    except Exception as exc:
        log.warning("Judge evaluation failed (%s): %s", label, exc)
        return None


def _override_false_truncated(verdict: Dict, zai_response: Dict) -> Dict:
    """Override false TRUNCATED verdicts when stop_reason proves completion."""
    stop_reason = zai_response.get("stop_reason", "")
    if stop_reason == "end_turn" and "TRUNCATED" in verdict.get("failures", []):
        verdict["failures"] = [f for f in verdict["failures"] if f != "TRUNCATED"]
        if not verdict["failures"]:
            verdict["pass"] = True
            verdict["feedback"] = ""
        log.info("Overrode false TRUNCATED (stop_reason=end_turn)")
    return verdict


def evaluate(
    request_messages: List[Dict],
    zai_response: Dict,
    router_url: str,
    tools: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Evaluate a Z.AI response using the active model via the router.
    Falls back to helper agent if GPU judge is unavailable.
    Returns {"pass": bool, "failures": [], "feedback": str}."""

    # Deterministic tool call checks first — fast, reliable
    tool_verdict = check_tool_calls(zai_response, request_messages)
    if not tool_verdict["pass"]:
        log.info("Deterministic check failed: %s", tool_verdict["failures"])
        return tool_verdict

    # Log stop_reason for debugging truncation reports
    stop_reason = zai_response.get("stop_reason", "")
    if stop_reason:
        log.info("Z.AI stop_reason: %s", stop_reason)

    # Build a summary of the conversation for the judge
    conv_summary = _summarize_conversation(request_messages)
    response_summary = json.dumps(zai_response.get("content", []), ensure_ascii=False)[:2000]

    tool_context = ""
    if tools:
        names = [t.get("name", "") for t in tools if t.get("name")]
        if names:
            tool_context = f"\nAVAILABLE TOOLS: {', '.join(names)}\n"

    judge_prompt = (
        f"CONVERSATION:\n{conv_summary}\n\n"
        f"{tool_context}"
        f"AI RESPONSE:\n{response_summary}\n\n"
        f"Evaluate this response."
    )

    payload = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_prompt},
        ],
        "max_tokens": 150,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Try GPU judge first (via router)
    gpu_endpoint = router_url.rstrip("/") + "/v1/chat/completions"
    verdict = _call_judge(gpu_endpoint, payload, "gpu")
    if verdict is not None:
        verdict = _override_false_truncated(verdict, zai_response)
        return verdict

    # GPU unavailable (mid-swap, etc.) — fall back to helper agent
    helper_endpoint = HELPER_URL.rstrip("/") + "/v1/chat/completions"
    verdict = _call_judge(helper_endpoint, payload, "helper-fallback")
    if verdict is not None:
        verdict = _override_false_truncated(verdict, zai_response)
        return verdict

    # Both judges unavailable — default PASS
    log.warning("All judges unavailable, defaulting to PASS")
    return {"pass": True, "failures": [], "feedback": "all_judges_unavailable"}


def _summarize_conversation(messages: List[Dict], max_messages: int = 3) -> str:
    """Summarize last N messages for the judge prompt."""
    recent = messages[-max_messages:]
    lines = []
    for msg in recent:
        role = msg.get("role", "unknown")
        text = _extract_text(msg)[:500]
        if text:
            lines.append(f"{role.title()}: {text}")
    return "\n".join(lines) if lines else "(empty conversation)"


def _try_local_gpu_fallback(
    payload: Dict,
    req_id: Optional[str],
    local_url: str,
    local_model: Optional[str],
) -> Tuple[int, Dict]:
    """Attempt to serve the request from the active local GPU model."""
    import translate

    original_model = payload.get("model", "unknown")
    reqlog({
        "event": "local_gpu_fallback",
        "req_id": req_id,
        "from_model": original_model,
        "to_local_model": local_model or "active",
    })
    metrics.inc("smart_proxy_local_gpu_fallbacks_total", {"from_model": original_model})

    try:
        oai_payload = translate.anthropic_to_openai_request(payload)
        oai_payload["model"] = local_model or "local"
        data = json.dumps(oai_payload).encode("utf-8")

        req = urllib.request.Request(
            local_url.rstrip("/") + "/v1/chat/completions",
            data=data, method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=300) as resp:
            oai_resp = json.loads(resp.read())

        anthropic_resp = translate.openai_to_anthropic_response(oai_resp, original_model)
        log.info("Local GPU fallback succeeded (model=%s)", local_model)
        return 200, anthropic_resp
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        log.error("Local GPU fallback failed: HTTP %d: %s", exc.code, detail)
        return 502, {
            "error": {
                "message": f"Local GPU fallback failed: HTTP {exc.code}: {detail}",
                "type": "local_fallback_error",
            }
        }
    except Exception as exc:
        log.error("Local GPU fallback failed: %s", exc)
        return 502, {
            "error": {
                "message": f"Local GPU fallback failed: {type(exc).__name__}: {exc}",
                "type": "local_fallback_error",
            }
        }


# ── Gate Orchestration ────────────────────────────────────────────────

def gate(
    zai_url: str,
    api_key: str,
    payload: Dict,
    router_url: str,
    max_retries: int = 2,
    req_id: Optional[str] = None,
    frontier_fallback: Optional[str] = None,
    local_fallback_url: Optional[str] = None,
    local_fallback_model: Optional[str] = None,
    swap_in_progress: bool = False,
) -> Tuple[int, Dict]:
    """Full quality gate pipeline. Returns (http_status, anthropic_response)."""

    messages = payload.get("messages", [])
    tools_in_payload = payload.get("tools", [])

    # Pre-classify: ask Qwen which tool(s) match, inject nudge if action request
    matched_tools = None
    send_payload = payload
    if tools_in_payload and classify(messages):
        matched_tools = _match_tools(messages, tools_in_payload)
        if matched_tools:
            send_payload = _inject_tool_nudge(payload, matched_tools)
            log.info("Tool nudge: injected for %s", matched_tools)

    # Attempt 1 (with nudge if applicable)
    status, response = call_zai(
        zai_url, api_key, send_payload, req_id=req_id, attempt=1,
        frontier_fallback=frontier_fallback,
    )
    if status != 200:
        # Local GPU fallback — last resort when all Z.AI tiers failed
        if local_fallback_url and not swap_in_progress:
            return _try_local_gpu_fallback(
                payload, req_id, local_fallback_url, local_fallback_model,
            )
        return status, response

    # Skip classification
    if not classify(messages):
        log.info("Quality gate: skipped (classified as no-eval)")
        return status, response

    # Evaluate
    best_response = response
    best_score = 0

    for attempt in range(1 + max_retries):
        if attempt > 0:
            # Retry with structured feedback (uses original payload, not nudged)
            retry_payload = dict(payload)
            retry_messages = list(messages)
            feedback = _build_retry_feedback(verdict, tools_in_payload, matched_tools)
            retry_messages.append({
                "role": "user",
                "content": feedback,
            })
            retry_payload["messages"] = retry_messages
            status, response = call_zai(
                zai_url, api_key, retry_payload, req_id=req_id, attempt=attempt + 1
            )
            if status != 200:
                log.warning("Z.AI retry %d failed: %d", attempt, status)
                continue

        verdict = evaluate(messages, response, router_url, tools_in_payload)

        # Track best attempt
        score = 5 if verdict.get("pass") else len(verdict.get("failures", []))
        if verdict.get("pass") or score < best_score or best_score == 0:
            best_response = response
            best_score = score if verdict.get("pass") else score

        if verdict.get("pass"):
            if attempt > 0:
                log.info("Quality gate: passed on retry %d", attempt)
            else:
                log.info("Quality gate: passed on first attempt")
            return 200, response

        log.info(
            "Quality gate: failed attempt %d/%d — %s",
            attempt + 1,
            1 + max_retries,
            verdict.get("failures", []),
        )

        # Only retry on deterministic tool-call failures (bad agentId, wrong
        # runtime, null params). Subjective judge verdicts don't improve on retry.
        deterministic_failures = {"MISSING_AGENT_ID", "MISSING_TASK", "WRONG_RUNTIME",
                                   "NULL_REQUIRED_PARAM", "TEXT_TOOL_MISMATCH"}
        if not (set(verdict.get("failures", [])) & deterministic_failures):
            log.info("Quality gate: judge-only failure, skipping retries")
            return 200, response

    log.warning("Quality gate: all attempts failed, returning best response")
    return 200, best_response
