#!/usr/bin/env python3
"""Smart routing proxy for multi-model LLM inference via Portainer stacks.

Listens on port 4000, routes OpenAI-compatible requests to the appropriate
backend based on model alias. Auto-swaps GPU model stacks via Portainer API
with cooldown-based anti-flap safeguards.
"""
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import anthropic_sse
import metrics
import quality_gate
import traffic_cop
from config import UPSTREAM_ROUTES, Config
from swap import SwapManager

metrics.set_gauge("smart_proxy_up", 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("proxy")

CONFIG_PATH = os.environ.get(
    "SMART_PROXY_CONFIG", "/opt/smart-proxy/config.yaml"
)
# Fallback for running outside Docker
if not Path(CONFIG_PATH).exists():
    CONFIG_PATH = str(Path(__file__).resolve().parent / "config.yaml")

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a request classifier. Given a user message, output exactly one label:\n\n"
    "SIMPLE \u2014 factual question, status check, short answer\n"
    "RAG \u2014 needs to search documents or a knowledge base\n"
    "CODE \u2014 code generation, debugging, or programming task\n"
    "REASON \u2014 multi-step analysis, comparison, planning\n\n"
    "Reply with the single label only, nothing else."
)
_CLASSIFY_TIMEOUT_S = 2.0  # 2s budget (cold prompt eval ~600ms, warm ~250ms)
_VALID_LABELS = {"SIMPLE", "RAG", "CODE", "REASON"}


# ── Helpers ─────────────────────────────────────────────────────────────

def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    # Clamp to a valid HTTP status: failure paths (e.g. quality_gate transport
    # errors) can produce 0, which send_response() would emit as an invalid
    # "HTTP/1.1 0" status line and break the client connection.
    if not isinstance(status, int) or not 100 <= status <= 599:
        status = 502
    data = _json_bytes(payload)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_json_with_retry_after(
    handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any], retry_after: int
) -> None:
    data = _json_bytes(payload)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Retry-After", str(retry_after))
    handler.end_headers()
    handler.wfile.write(data)


def _proxy_request(
    handler: BaseHTTPRequestHandler,
    backend_url: str,
    path: str,
    body: bytes,
    stream: bool = False,
    timeout_s: int = 600,
) -> None:
    """Forward a request to a backend and relay the response."""
    url = backend_url.rstrip("/") + path
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            ctype = resp.headers.get("Content-Type", "application/json")

            if stream:
                handler.send_response(resp.status)
                handler.send_header("Content-Type", ctype)
                handler.send_header("Cache-Control", "no-cache")
                handler.send_header("Transfer-Encoding", "chunked")
                handler.end_headers()
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    handler.wfile.write(f"{len(line):x}\r\n".encode())
                    handler.wfile.write(line)
                    handler.wfile.write(b"\r\n")
                    handler.wfile.flush()
                    if line.strip() == b"data: [DONE]":
                        break
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
            else:
                data = resp.read()
                handler.send_response(resp.status)
                handler.send_header("Content-Type", ctype)
                handler.send_header("Content-Length", str(len(data)))
                handler.end_headers()
                handler.wfile.write(data)

    except urllib.error.HTTPError as exc:
        body_err = exc.read().decode("utf-8", errors="replace")
        _send_json(handler, HTTPStatus.BAD_GATEWAY, {
            "error": {"message": f"Backend error: {exc.code}", "type": "backend_error", "detail": body_err[:500]},
        })
    except Exception as exc:
        _send_json(handler, HTTPStatus.BAD_GATEWAY, {
            "error": {"message": f"Backend unreachable: {exc}", "type": "backend_error"},
        })


def _proxy_get(
    handler: BaseHTTPRequestHandler, backend_url: str, path: str, timeout_s: int = 10
) -> None:
    """Forward a GET request."""
    url = backend_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "application/json")
            handler.send_response(resp.status)
            handler.send_header("Content-Type", ctype)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
    except Exception as exc:
        _send_json(handler, HTTPStatus.BAD_GATEWAY, {
            "error": {"message": f"Backend unreachable: {exc}", "type": "backend_error"},
        })


# ── Request Handler ────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "SmartProxy/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def _cfg(self) -> Config:
        return self.server.cfg  # type: ignore[attr-defined]

    @property
    def _swap(self) -> SwapManager:
        return self.server.swap_mgr  # type: ignore[attr-defined]

    def _queued_proxy(self, backend_url: str, path: str, body: bytes, stream: bool = False) -> None:
        """Serialize inference requests to backend. Rejects if 3+ already waiting."""
        srv = self.server
        with srv.inference_waiting_lock:  # type: ignore[attr-defined]
            if srv.inference_waiting >= 3:  # type: ignore[attr-defined]
                _send_json(self, HTTPStatus.SERVICE_UNAVAILABLE, {
                    "error": {
                        "message": "Too many inference requests queued. Try again shortly.",
                        "type": "server_busy",
                    },
                })
                return
            srv.inference_waiting += 1  # type: ignore[attr-defined]
        try:
            with srv.inference_lock:  # type: ignore[attr-defined]
                _proxy_request(self, backend_url, path, body, stream=stream)
        finally:
            with srv.inference_waiting_lock:  # type: ignore[attr-defined]
                srv.inference_waiting -= 1  # type: ignore[attr-defined]

    def _apply_thinking(self, payload: Dict[str, Any], model: Optional[Any]) -> bytes:
        """Inject chat_template_kwargs.enable_thinking based on model config.
        Respects client override if already set."""
        if model and not model.thinking:
            kwargs = payload.get("chat_template_kwargs", {})
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
                payload["chat_template_kwargs"] = kwargs
        return json.dumps(payload).encode("utf-8")

    def _classify_request(self, messages: list) -> Optional[str]:
        """Call the classifier agent to categorize a request. Returns label or None on failure."""
        classifier_url = self._cfg.get_backend_url("classifier")
        if not classifier_url:
            return None
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return None
        last_user_msg = user_msgs[-1].get("content", "")
        classify_payload = json.dumps({
            "model": "any",
            "messages": [
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": last_user_msg},
            ],
            "max_tokens": 5,
            "temperature": 0.0,
        }).encode("utf-8")
        url = classifier_url.rstrip("/") + "/v1/chat/completions"
        req = urllib.request.Request(url, data=classify_payload, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=_CLASSIFY_TIMEOUT_S) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                label = result["choices"][0]["message"]["content"].strip().upper()
                if label in _VALID_LABELS:
                    log.info("auto-route: classified as %s", label)
                    return label
                log.warning("auto-route: unexpected label '%s', falling through", label)
                return None
        except Exception as exc:
            log.warning("auto-route: classifier failed (%s), falling through to GPU", exc)
            return None

    def _read_body(self) -> Optional[bytes]:
        cl = self.headers.get("Content-Length")
        if not cl:
            return None
        return self.rfile.read(int(cl))

    def _parse_json(self, body: bytes) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None

    # ── GET routes ──────────────────────────────────────────────────

    def do_GET(self) -> None:
        if self.path == "/health":
            _send_json(self, HTTPStatus.OK, {
                "status": "ok",
                "service": "smart-proxy",
                "active_model": self._swap.active_model,
            })
            return

        if self.path == "/v1/status":
            status = self._swap.status()
            status["service"] = "smart-proxy"
            status["zai_breaker"] = quality_gate.breaker_snapshot()
            _send_json(self, HTTPStatus.OK, status)
            return

        if self.path == "/metrics":
            # Refresh breaker gauges (clears stale "unhealthy" after cooldown).
            quality_gate.breaker_snapshot()
            body = metrics.render().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/models":
            self._handle_models()
            return

        _send_json(self, HTTPStatus.NOT_FOUND, {
            "error": {"message": "Not found", "type": "invalid_request_error"},
        })

    def _handle_models(self) -> None:
        models = []
        now = int(time.time())
        for name, mc in self._cfg.models.items():
            available = mc.always_available or (mc.stack_name == self._swap.active_stack)
            loaded = mc.stack_name == self._swap.active_stack if mc.gpu_model else mc.always_available
            models.append({
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "local",
                "aliases": mc.aliases,
                "description": mc.description,
                "gpu_model": mc.gpu_model,
                "available": available,
                "loaded": loaded,
            })
        _send_json(self, HTTPStatus.OK, {"object": "list", "data": models})

    # ── POST routes ─────────────────────────────────────────────────

    def do_POST(self) -> None:
        if self.path == "/v1/messages":
            self._handle_anthropic_messages()
            return

        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
            return

        if self.path == "/v1/completions":
            self._handle_completions()
            return

        if self.path == "/v1/embeddings":
            self._handle_embeddings()
            return

        _send_json(self, HTTPStatus.NOT_FOUND, {
            "error": {"message": "Not found", "type": "invalid_request_error"},
        })

    def _handle_anthropic_messages(self) -> None:
        req_id = uuid.uuid4().hex[:12]
        t_start = time.monotonic()
        body = self._read_body()
        if not body:
            quality_gate.reqlog({
                "event": "msg_end", "req_id": req_id, "status": 400,
                "duration_ms": int((time.monotonic() - t_start) * 1000),
                "reason": "missing_body",
            })
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Missing request body", "type": "invalid_request_error"},
            })
            return

        payload = self._parse_json(body)
        if payload is None:
            quality_gate.reqlog({
                "event": "msg_end", "req_id": req_id, "status": 400,
                "duration_ms": int((time.monotonic() - t_start) * 1000),
                "reason": "invalid_json", "payload_bytes": len(body),
            })
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Invalid JSON", "type": "invalid_request_error"},
            })
            return

        model_name = payload.get("model", "")
        resolved = self._cfg.resolve_alias(model_name)
        # Normalize client alias -> canonical model name. Z.AI rejects compact
        # aliases (e.g. "glm5") with 1211 Unknown Model; only canonical names
        # like "glm-5.2" are valid, so forward the resolved name upstream.
        if resolved:
            payload["model"] = resolved.name
            model_name = resolved.name

        messages = payload.get("messages", []) or []
        tools = payload.get("tools", []) or []
        quality_gate.reqlog({
            "event": "msg_start",
            "req_id": req_id,
            "path": "/v1/messages",
            "client_ip": self.client_address[0],
            "user_agent": self.headers.get("User-Agent", ""),
            "model_requested": model_name,
            "model_resolved": resolved.name if resolved else None,
            "backend": resolved.backend if resolved else None,
            "payload_bytes": len(body),
            "message_count": len(messages),
            "tool_count": len(tools),
            "tools_hash": quality_gate.hash_tool_names(tools),
            "last_msg_type": quality_gate.classify_last_msg(messages),
            "has_images": quality_gate.detect_images(messages),
            "max_tokens": payload.get("max_tokens"),
            "wants_stream": bool(payload.get("stream", False)),
        })

        if resolved is None or not resolved.quality_gate:
            quality_gate.reqlog({
                "event": "msg_end", "req_id": req_id, "status": 400,
                "duration_ms": int((time.monotonic() - t_start) * 1000),
                "reason": "not_quality_gate_model",
            })
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {
                    "message": f"Model '{model_name}' is not available via /v1/messages. Use /v1/chat/completions for local models.",
                    "type": "invalid_request_error",
                },
            })
            return

        # Upstream routing map (binding, enforced in code — config.UPSTREAM_ROUTES):
        # legacy Z.AI ids are served by a different upstream model. Remap
        # before caps/fallbacks/metrics key on the model, so accounting
        # matches the pool Z.AI actually draws from. Clients keep their model
        # id. After this map every z.ai request is accounted on the pool that
        # serves it (glm-5.3 or glm-5.3-flash).
        upstream_name = UPSTREAM_ROUTES.get(resolved.name)
        if upstream_name:
            upstream = self._cfg.resolve_alias(upstream_name)
            if upstream and upstream.name != resolved.name:
                quality_gate.reqlog({
                    "event": "upstream_reroute",
                    "req_id": req_id,
                    "from_model": resolved.name,
                    "to_model": upstream.name,
                })
                payload = dict(payload)
                payload["model"] = upstream.name
                resolved = upstream
            else:
                log.error(
                    "Upstream route %s -> %s does not resolve; passing through as %s",
                    resolved.name, upstream_name, resolved.name,
                )

        # Traffic cop: reroute tool-result acks to high-concurrency model
        routed_model = None
        route_reason = "passthrough"
        try:
            routed_model, route_reason = traffic_cop.route(self._cfg, messages, resolved)
            if routed_model:
                payload = dict(payload)
                payload["model"] = routed_model
        except Exception:
            pass
        quality_gate.reqlog({
            "event": "traffic_cop",
            "req_id": req_id,
            "original_model": model_name,
            "routed_model": routed_model,
            "reason": route_reason,
        })

        zai_url = self._cfg.get_backend_url(resolved.backend)
        api_key = self._cfg.get_backend_api_key(resolved.backend)
        if not zai_url or not api_key:
            quality_gate.reqlog({
                "event": "msg_end", "req_id": req_id, "status": 502,
                "duration_ms": int((time.monotonic() - t_start) * 1000),
                "reason": "backend_not_configured",
            })
            _send_json(self, HTTPStatus.BAD_GATEWAY, {
                "error": {"message": "Z.AI backend not configured or missing API key", "type": "server_error"},
            })
            return

        router_url = f"http://localhost:{self._cfg.listen_port}"
        wants_stream = payload.get("stream", False)

        target_model = routed_model or (resolved.name if resolved else model_name)
        fb = resolved.frontier_fallback if resolved and not routed_model else None
        acquired = traffic_cop.acquire(self._cfg, target_model)
        if not acquired:
            mc = self._cfg.models.get(target_model)
            if mc and mc.concurrency_limit is not None:
                retry_after = 30
                quality_gate.reqlog({
                    "event": "msg_end", "req_id": req_id, "status": 503,
                    "duration_ms": int((time.monotonic() - t_start) * 1000),
                    "reason": "concurrency_limit", "model": target_model,
                })
                _send_json_with_retry_after(self, HTTPStatus.SERVICE_UNAVAILABLE, {
                    "error": {
                        "message": f"Concurrency limit reached for {target_model}",
                        "type": "concurrency_limit",
                    },
                }, retry_after=retry_after)
                return
        try:
            local_fb_url = None
            local_fb_model = None
            if resolved and resolved.local_fallback:
                # Pinned fallback model (binding decision 2026-09-03): NEVER
                # derived from the active stack. The inference port serves
                # whatever stack is active, so the fallback is only valid when
                # the pinned model's stack IS the active one — otherwise skip
                # (logged) rather than serve a different local model.
                pinned = self._cfg.local_fallback_model
                if pinned and pinned == self._swap.active_model:
                    local_fb_url = self._cfg.get_backend_url("inference")
                    local_fb_model = pinned
                else:
                    log.info(
                        "Local GPU fallback skipped: pinned model %s is not the active stack (active=%s)",
                        pinned, self._swap.active_model,
                    )
            status, response = quality_gate.gate(
                zai_url, api_key, payload, router_url,
                req_id=req_id, frontier_fallback=fb,
                local_fallback_url=local_fb_url,
                local_fallback_model=local_fb_model,
                swap_in_progress=self._swap.swap_in_progress,
            )
        finally:
            if acquired:
                traffic_cop.release(target_model)

        actual_model = payload.get("model", model_name)
        was_rerouted = actual_model != model_name

        quality_gate.reqlog({
            "event": "msg_end",
            "req_id": req_id,
            "status": status,
            "duration_ms": int((time.monotonic() - t_start) * 1000),
            "routed_model": actual_model if was_rerouted else None,
        })

        if status != 200:
            _send_json(self, status, response)
            return

        if wants_stream:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            if was_rerouted:
                self.send_header("X-Routed-Model", actual_model)
            self.end_headers()
            for chunk in anthropic_sse.synthesize(response):
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            if was_rerouted:
                self.send_response(HTTPStatus.OK)
                data = json.dumps(response, ensure_ascii=True).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Routed-Model", actual_model)
                self.end_headers()
                self.wfile.write(data)
            else:
                _send_json(self, HTTPStatus.OK, response)

    def _handle_chat_completions(self) -> None:
        body = self._read_body()
        if not body:
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Missing request body", "type": "invalid_request_error"},
            })
            return

        payload = self._parse_json(body)
        if payload is None:
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Invalid JSON", "type": "invalid_request_error"},
            })
            return

        model_name = payload.get("model", "")
        is_stream = bool(payload.get("stream", False))

        resolved = self._cfg.resolve_alias(model_name)

        # Auto-route: classify then dispatch to the right tier
        if resolved and resolved.name == "auto-route":
            label = self._classify_request(payload.get("messages", []))
            if label in ("SIMPLE", "RAG"):
                target = "helper"
            elif label == "CODE":
                # Use active GPU model for code — don't trigger a swap
                target = None
            else:
                # REASON or classifier failed — use active GPU model
                target = None
            if target:
                rerouted = self._cfg.resolve_alias(target)
                if rerouted:
                    log.info("auto-route: %s -> %s", label, rerouted.name)
                    resolved = rerouted
                else:
                    log.warning("auto-route: target '%s' not found, falling through to GPU", target)
                    resolved = None
            else:
                log.info("auto-route: %s -> active GPU model", label or "FALLBACK")
                resolved = None

        if resolved is None:
            # Unknown model or auto-route fallback — forward to active GPU backend
            if self._swap.active_model:
                inference_url = self._cfg.get_backend_url("inference")
                if inference_url:
                    active_mc = self._cfg.models.get(self._swap.active_model)
                    body = self._apply_thinking(payload, active_mc)
                    self._queued_proxy(inference_url, "/v1/chat/completions", body, stream=is_stream)
                    return
            _send_json(self, HTTPStatus.NOT_FOUND, {
                "error": {
                    "message": f"Unknown model: {model_name}. Use GET /v1/models for available models.",
                    "type": "invalid_request_error",
                },
            })
            return

        # Always-available backends (embedding, remote coder, classifier, helper)
        if resolved.always_available:
            if resolved.backend == "embedding":
                _send_json(self, HTTPStatus.BAD_REQUEST, {
                    "error": {
                        "message": f"Model '{model_name}' is an embedding model. Use /v1/embeddings instead.",
                        "type": "invalid_request_error",
                    },
                })
                return
            backend_url = self._cfg.get_backend_url(resolved.backend)
            if backend_url:
                _proxy_request(self, backend_url, "/v1/chat/completions", body, stream=is_stream)
            else:
                _send_json(self, HTTPStatus.BAD_GATEWAY, {
                    "error": {"message": f"Backend {resolved.backend} not configured", "type": "server_error"},
                })
            return

        # GPU model — check if it's the active one
        if resolved.stack_name == self._swap.active_stack:
            inference_url = self._cfg.get_backend_url("inference")
            if inference_url:
                body = self._apply_thinking(payload, resolved)
                self._queued_proxy(inference_url, "/v1/chat/completions", body, stream=is_stream)
            return

        # Need to swap to a different model
        if self._swap.swap_in_progress:
            est = self._cfg.health_timeout_seconds
            _send_json_with_retry_after(self, HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": {
                    "message": f"Model swap in progress to {self._swap.swap_target}. Retry in ~{est}s.",
                    "type": "server_busy",
                },
                "active_model": self._swap.active_model,
                "swap_target": self._swap.swap_target,
            }, retry_after=est)
            return

        if self._swap.cooldown_active:
            remaining = int(self._swap.cooldown_remaining)
            _send_json_with_retry_after(self, HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": {
                    "message": f"Swap cooldown active ({remaining}s remaining). Active model: {self._swap.active_model}.",
                    "type": "server_busy",
                },
                "active_model": self._swap.active_model,
                "requested_model": resolved.name,
                "cooldown_remaining": remaining,
            }, retry_after=remaining)
            return

        # Initiate swap
        result = self._swap.swap_to(resolved.name)
        if result.get("status") == "already_active":
            inference_url = self._cfg.get_backend_url("inference")
            if inference_url:
                body = self._apply_thinking(payload, resolved)
                self._queued_proxy(inference_url, "/v1/chat/completions", body, stream=is_stream)
            return

        est = result.get("estimated_seconds", 60)
        _send_json_with_retry_after(self, HTTPStatus.SERVICE_UNAVAILABLE, {
            "error": {
                "message": f"Swapping to {resolved.name} ({resolved.description}). Retry in ~{est}s.",
                "type": "model_swap_initiated",
            },
            "swap_status": result,
        }, retry_after=est)

    def _handle_completions(self) -> None:
        # Same logic as chat completions, different endpoint
        body = self._read_body()
        if not body:
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Missing request body", "type": "invalid_request_error"},
            })
            return

        payload = self._parse_json(body)
        if payload is None:
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Invalid JSON", "type": "invalid_request_error"},
            })
            return

        # For plain completions, just forward to active inference backend
        inference_url = self._cfg.get_backend_url("inference")
        if inference_url and self._swap.active_model:
            is_stream = bool(payload.get("stream", False))
            self._queued_proxy(inference_url, "/v1/completions", body, stream=is_stream)
        else:
            _send_json(self, HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": {"message": "No model currently loaded", "type": "server_error"},
            })

    def _handle_embeddings(self) -> None:
        body = self._read_body()
        if not body:
            _send_json(self, HTTPStatus.BAD_REQUEST, {
                "error": {"message": "Missing request body", "type": "invalid_request_error"},
            })
            return

        embedding_url = self._cfg.get_backend_url("embedding")
        if embedding_url:
            _proxy_request(self, embedding_url, "/v1/embeddings", body)
        else:
            _send_json(self, HTTPStatus.BAD_GATEWAY, {
                "error": {"message": "Embedding backend not configured", "type": "server_error"},
            })

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress default access log; we log important events explicitly
        return


# ── Main ───────────────────────────────────────────────────────────────

def main() -> int:
    log.info("Loading config from %s", CONFIG_PATH)
    cfg = Config(CONFIG_PATH)

    swap_mgr = SwapManager(cfg)
    swap_mgr.detect_active_model()

    httpd = ThreadingHTTPServer(("0.0.0.0", cfg.listen_port), ProxyHandler)
    httpd.cfg = cfg  # type: ignore[attr-defined]
    httpd.swap_mgr = swap_mgr  # type: ignore[attr-defined]
    httpd.inference_lock = threading.Lock()  # type: ignore[attr-defined] — serializes to backend
    httpd.inference_waiting = 0  # type: ignore[attr-defined]
    httpd.inference_waiting_lock = threading.Lock()  # type: ignore[attr-defined]

    log.info(
        "Smart proxy listening on 0.0.0.0:%d | active_model=%s | active_stack=%s",
        cfg.listen_port,
        swap_mgr.active_model,
        swap_mgr.active_stack,
    )
    log.info(
        "Models: %s",
        ", ".join(f"{n} ({'GPU' if m.gpu_model else m.backend})" for n, m in cfg.models.items()),
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
