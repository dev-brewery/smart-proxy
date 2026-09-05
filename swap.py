"""Portainer-based model stack swap manager with anti-flap safeguards.

Uses API key authentication (Portainer CE access tokens).
"""
import json
import logging
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import Config

log = logging.getLogger("swap")


class SwapManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._swap_in_progress = False
        self._swap_target: Optional[str] = None
        self._cooldown_until: float = 0.0
        self._active_stack: Optional[str] = None
        self._active_model: Optional[str] = None
        self._stack_ids: Dict[str, int] = {}
        self._ssl_ctx = ssl.create_default_context()
        if not cfg.portainer_verify_ssl:
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE

        self._load_state()
        self._discover_stacks()

    # ── Portainer API helpers ───────────────────────────────────────────

    def _portainer_request(
        self, method: str, path: str, data: Optional[bytes] = None
    ) -> Tuple[int, bytes]:
        url = self.cfg.portainer_url.rstrip("/") + path
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("X-API-Key", self.cfg.portainer_api_key)
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30, context=self._ssl_ctx) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            log.error("Portainer API %s %s -> %d: %s", method, path, exc.code, body[:200])
            return exc.code, body

    def _discover_stacks(self) -> None:
        status, body = self._portainer_request("GET", "/api/stacks")
        if status != 200:
            log.error("Failed to discover Portainer stacks: %d", status)
            return
        stacks = json.loads(body)
        for s in stacks:
            name = s.get("Name", "")
            sid = s.get("Id")
            if sid is not None:
                self._stack_ids[name] = sid
                log.info("Discovered stack: %s -> id=%d", name, sid)

    # ── State persistence ───────────────────────────────────────────────

    def _load_state(self) -> None:
        p = Path(self.cfg.state_file)
        if p.exists():
            try:
                state = json.loads(p.read_text())
                self._active_stack = state.get("active_stack")
                self._active_model = state.get("active_model")
                log.info("Loaded state: model=%s stack=%s", self._active_model, self._active_stack)
            except Exception as exc:
                log.warning("Failed to load state: %s", exc)

    def _save_state(self) -> None:
        p = Path(self.cfg.state_file)
        p.write_text(
            json.dumps(
                {
                    "active_stack": self._active_stack,
                    "active_model": self._active_model,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            )
        )

    # ── Health polling ──────────────────────────────────────────────────

    def _poll_health(self, timeout_s: int) -> bool:
        url = self.cfg.backends["inference"]["url"].rstrip("/") + "/health"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model

    @property
    def active_stack(self) -> Optional[str]:
        return self._active_stack

    @property
    def swap_in_progress(self) -> bool:
        return self._swap_in_progress

    @property
    def swap_target(self) -> Optional[str]:
        return self._swap_target

    @property
    def cooldown_remaining(self) -> float:
        r = self._cooldown_until - time.monotonic()
        return max(0.0, r)

    @property
    def cooldown_active(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def detect_active_model(self) -> None:
        """Try to detect which model is currently running by checking health."""
        url = self.cfg.backends["inference"]["url"].rstrip("/") + "/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    if self._active_model:
                        log.info("Backend healthy, active model: %s", self._active_model)
                    else:
                        log.info("Backend healthy but no model tracked in state")
                    return
        except Exception:
            pass
        log.info("No backend healthy on startup")

    def set_active_from_stack(self, stack_name: str) -> None:
        """Set the active model based on a stack name (for initialization)."""
        for name, mc in self.cfg.models.items():
            if mc.stack_name == stack_name:
                self._active_model = name
                self._active_stack = stack_name
                self._save_state()
                log.info("Set active model=%s stack=%s", name, stack_name)
                return

    def swap_to(self, target_model: str) -> Dict[str, Any]:
        """Initiate a swap to a different GPU model. Returns immediately with status."""
        mc = self.cfg.models.get(target_model)
        if not mc or not mc.gpu_model:
            return {"ok": False, "error": "unknown_gpu_model", "model": target_model}

        if mc.stack_name == self._active_stack:
            return {"ok": True, "status": "already_active"}

        if not self._lock.acquire(blocking=False):
            return {"ok": False, "error": "swap_already_in_progress", "target": self._swap_target}

        try:
            if self._swap_in_progress:
                return {"ok": False, "error": "swap_already_in_progress", "target": self._swap_target}

            if self.cooldown_active:
                remaining = int(self.cooldown_remaining)
                return {
                    "ok": False,
                    "error": "swap_cooldown",
                    "active_model": self._active_model,
                    "cooldown_remaining": remaining,
                }

            self._swap_in_progress = True
            self._swap_target = target_model
        finally:
            self._lock.release()

        # Run swap in background thread
        t = threading.Thread(target=self._do_swap, args=(target_model, mc.stack_name), daemon=True)
        t.start()

        return {
            "ok": True,
            "status": "swap_initiated",
            "target": target_model,
            "target_stack": mc.stack_name,
            "estimated_seconds": self.cfg.health_timeout_seconds,
        }

    def _do_swap(self, target_model: str, target_stack: str) -> None:
        prev_stack = self._active_stack
        prev_model = self._active_model
        eid = self.cfg.portainer_endpoint_id

        try:
            # Step 1: Stop current stack
            if prev_stack and prev_stack in self._stack_ids:
                sid = self._stack_ids[prev_stack]
                log.info("Stopping stack %s (id=%d)", prev_stack, sid)
                status, _ = self._portainer_request(
                    "POST", f"/api/stacks/{sid}/stop?endpointId={eid}"
                )
                if status not in (200, 400):  # 400 = already stopped
                    log.warning("Stop returned %d, continuing anyway", status)
                time.sleep(2)  # brief settle

            # Step 2: Start target stack
            if target_stack not in self._stack_ids:
                self._discover_stacks()  # refresh in case stack was added

            if target_stack not in self._stack_ids:
                log.error("Target stack %s not found in Portainer", target_stack)
                self._rollback(prev_stack, prev_model)
                return

            sid = self._stack_ids[target_stack]
            log.info("Starting stack %s (id=%d)", target_stack, sid)
            status, _ = self._portainer_request(
                "POST", f"/api/stacks/{sid}/start?endpointId={eid}"
            )
            if status not in (200, 409):  # 409 = already running
                log.error("Start stack %s failed: %d", target_stack, status)
                self._rollback(prev_stack, prev_model)
                return
            if status == 409:
                log.info("Stack %s already running", target_stack)

            # Step 3: Poll health
            log.info("Polling health for %s (timeout=%ds)", target_stack, self.cfg.health_timeout_seconds)
            if not self._poll_health(self.cfg.health_timeout_seconds):
                log.error("Stack %s failed health check", target_stack)
                self._rollback(prev_stack, prev_model)
                return

            # Success
            self._active_stack = target_stack
            self._active_model = target_model
            self._cooldown_until = time.monotonic() + self.cfg.cooldown_seconds
            self._save_state()
            log.info("Swap complete: %s -> %s", prev_model, target_model)

        except Exception as exc:
            log.exception("Swap failed: %s", exc)
            self._rollback(prev_stack, prev_model)
        finally:
            self._swap_in_progress = False
            self._swap_target = None

    def _rollback(self, prev_stack: Optional[str], prev_model: Optional[str]) -> None:
        log.warning("Rolling back to %s (%s)", prev_model, prev_stack)
        if prev_stack and prev_stack in self._stack_ids:
            sid = self._stack_ids[prev_stack]
            eid = self.cfg.portainer_endpoint_id
            self._portainer_request("POST", f"/api/stacks/{sid}/start?endpointId={eid}")
            if self._poll_health(self.cfg.health_timeout_seconds):
                self._active_stack = prev_stack
                self._active_model = prev_model
                self._save_state()
                log.info("Rollback successful")
            else:
                log.error("Rollback also failed — no healthy backend")

    def status(self) -> Dict[str, Any]:
        return {
            "active_model": self._active_model,
            "active_stack": self._active_stack,
            "swap_in_progress": self._swap_in_progress,
            "swap_target": self._swap_target,
            "cooldown_active": self.cooldown_active,
            "cooldown_remaining": int(self.cooldown_remaining),
        }
