"""Configuration loader for smart-proxy."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


# Binding upstream routing map (decided 2026-09-02/03; glm-5-turbo added
# 2026-09-04 — Z.AI began answering it with glm-5.3-flash responses, so it is
# no longer a distinct concurrency pool; a request must be accounted on the
# pool that serves it. memory zai-upstream-routing-decisions.md). ENFORCED IN
# CODE — proxy.py remaps these model ids to the upstream model that actually
# serves them, before concurrency caps / fallbacks / metrics key on the model.
# config.yaml `upstream_model` keys are documentational only;
# _validate_routing_refs warns on drift.
UPSTREAM_ROUTES = {
    "glm-5.2": "glm-5.3",
    "glm-4.7": "glm-5.3-flash",
    "glm-4.7-flash": "glm-5.3-flash",
    "glm-4.6v": "glm-5.3-flash",
    "glm-5-turbo": "glm-5.3-flash",
}


class ModelConfig:
    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self.stack_name: Optional[str] = data.get("stack_name")
        self.backend: Optional[str] = data.get("backend")
        self.aliases: List[str] = data.get("aliases", [])
        self.description: str = data.get("description", "")
        self.gpu_model: bool = data.get("gpu_model", False)
        self.always_available: bool = data.get("always_available", False)
        self.thinking: bool = data.get("thinking", True)
        self.quality_gate: bool = data.get("quality_gate", False)
        self.concurrency_limit: Optional[int] = data.get("concurrency_limit")
        self.ack_model: Optional[str] = data.get("ack_model")
        self.frontier_fallback: Optional[str] = data.get("frontier_fallback")
        # Documentational only — routing is enforced by UPSTREAM_ROUTES in
        # code. _validate_routing_refs warns if this key disagrees with it.
        self.upstream_model: Optional[str] = data.get("upstream_model")
        self.local_fallback: bool = data.get("local_fallback", False)


class Config:
    def __init__(self, path: str):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        self.listen_port: int = raw.get("listen_port", 4000)

        # Portainer
        pt = raw.get("portainer", {})
        self.portainer_url: str = pt.get("url", "https://localhost:9443")
        self.portainer_api_key: str = os.environ.get(
            pt.get("api_key_env", "PORTAINER_API_KEY"), ""
        )
        self.portainer_endpoint_id: int = pt.get("endpoint_id", 1)
        self.portainer_verify_ssl: bool = pt.get("verify_ssl", False)

        # Backends
        self.backends: Dict[str, Dict[str, str]] = raw.get("backends", {})

        # Models
        self.models: Dict[str, ModelConfig] = {}
        self._alias_map: Dict[str, str] = {}
        for name, data in raw.get("models", {}).items():
            mc = ModelConfig(name, data)
            self.models[name] = mc
            self._alias_map[name] = name
            for alias in mc.aliases:
                self._alias_map[alias.lower()] = name

        # Swap settings
        sw = raw.get("swap", {})
        self.cooldown_seconds: int = sw.get("cooldown_seconds", 600)
        self.health_timeout_seconds: int = sw.get("health_timeout_seconds", 120)
        self.state_file: str = sw.get(
            "state_file", "~/llm-hosts/smart-proxy/state.json"
        )

        self.default_model: str = raw.get("default_model", "qwen35-q8")

        # Pinned local GPU fallback model (binding decision 2026-09-03):
        # quality_gate local fallback always targets this model, never the
        # active-stack model. Served only when this model's stack is active.
        self.local_fallback_model: Optional[str] = raw.get("local_fallback_model")

        self._validate_routing_refs()

    def _validate_routing_refs(self) -> None:
        import logging
        log = logging.getLogger("config")
        for name, mc in self.models.items():
            for field in ("ack_model", "frontier_fallback"):
                ref = getattr(mc, field, None)
                if ref and ref not in self.models and not self._alias_map.get(ref.lower()):
                    log.warning("Model %s has %s=%s which does not resolve to a known model", name, field, ref)
            # Upstream routing is enforced by UPSTREAM_ROUTES (code). Config
            # upstream_model keys only document the map — alarm on any drift
            # between the two so they can never diverge silently.
            if mc.upstream_model is not None and UPSTREAM_ROUTES.get(name) != mc.upstream_model:
                log.warning(
                    "Model %s config upstream_model=%s DISAGREES with enforced UPSTREAM_ROUTES=%s (code map wins)",
                    name, mc.upstream_model, UPSTREAM_ROUTES.get(name))
            elif name in UPSTREAM_ROUTES and mc.upstream_model is None:
                log.warning(
                    "Model %s is rerouted by UPSTREAM_ROUTES to %s but config.yaml documents no upstream_model key",
                    name, UPSTREAM_ROUTES[name])

    def resolve_alias(self, name: str) -> Optional[ModelConfig]:
        key = self._alias_map.get(name.lower())
        if key:
            return self.models.get(key)
        return None

    def get_backend_url(self, backend_name: str) -> Optional[str]:
        b = self.backends.get(backend_name)
        return b.get("url") if b else None

    def get_backend_health(self, backend_name: str) -> Optional[str]:
        b = self.backends.get(backend_name)
        return b.get("health") if b else None

    def get_backend_api_key(self, backend_name: str) -> str:
        b = self.backends.get(backend_name)
        if not b:
            return ""
        env_var = b.get("api_key_env", "")
        return os.environ.get(env_var, "") if env_var else ""
