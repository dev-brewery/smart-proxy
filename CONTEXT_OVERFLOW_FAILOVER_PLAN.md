# Context-Overflow Failover: Auto-Reroute to Cloud

**Status:** Shelved (2026-03-31) — issue not recurring, avoid adding latency unnecessarily.

## Context

GPU models have hard context limits set in their llama.cpp configs (16k, 49k, 98k tokens). When a request's context exceeds this, the model truncates or produces degraded output. OpenClaw's `max_tokens` formula (`(model.maxTokens / 3) | 0`) compounds this — a model with 8192 maxTokens gets only 2730 output tokens.

**Problem:** The proxy is a pure pass-through with zero token inspection. Oversized requests hit the GPU model and get silently truncated. The idea: detect this and automatically reroute to a cloud model (Z.AI GLM family, 131k+ context).

**Key constraint:** Chat completions (OpenAI format) to Z.AI (Anthropic format) requires format conversion.

---

## Research Findings

### Proxy architecture (relevant to this change)
- `proxy.py` `_handle_chat_completions()` resolves model alias -> ModelConfig, forwards to backend
- Pure pass-through — no token counting, no context inspection
- `ModelConfig` (config.py) has NO contextWindow or maxTokens fields
- Two GPU forwarding paths: explicit model match (line ~441) and fallback-to-active-GPU (line ~404)

### OpenClaw token limits
- `anthropic.js` line 462: `max_tokens: options?.maxTokens || (model.maxTokens / 3) | 0`
- quality-gate models: maxTokens=131072 -> max_tokens=43690
- local-llm/qwen3-80b: maxTokens=8192 -> max_tokens=2730
- Fallback chain: quality-gate/glm-5-turbo -> glm-5 -> zai/glm-5 -> zai/glm-4.7 -> local-llm/qwen3-80b

### Format conversion needed
- OpenAI system messages -> Anthropic top-level `system` field
- Tool format: OpenAI `function.parameters` -> Anthropic `input_schema`
- Tool calls in assistant messages: OpenAI `tool_calls[].function` -> Anthropic `content[].tool_use`
- Tool results: OpenAI `role: "tool"` -> Anthropic `role: "user"` with `tool_result` content blocks
- Response: Anthropic `stop_reason` -> OpenAI `finish_reason` mapping

---

## Design (ready to implement if needed)

### Changes required (3 files)

**config.py** — Add `context_limit: Optional[int]` and `fallback_model: Optional[str]` to ModelConfig

**config.yaml** — Add to GPU models:
```yaml
qwen35-q8:   { context_limit: 16000,  fallback_model: "glm-5" }
qwen35-q6:   { context_limit: 49000,  fallback_model: "glm-5" }
qwen3-80b:   { context_limit: 98000,  fallback_model: "glm-5-turbo" }
qwen3-coder: { context_limit: 98000,  fallback_model: "glm-5-turbo" }
```

**proxy.py** — Add:
1. `_estimate_tokens(payload)` — chars/4 heuristic, includes messages + tool schemas + max_tokens reserve
2. `_openai_to_anthropic(payload, model_name)` — format conversion
3. `_anthropic_to_openai(response, model_name)` — reverse conversion
4. `_reroute_to_cloud()` method — calls quality_gate.gate() with converted payload
5. `_send_openai_sse()` — synthesize OpenAI SSE from complete response
6. Overflow check at two GPU forwarding points (threshold: 85% of context_limit)

### Design principles
- Fail-open: if estimation fails or fallback not configured, pass through as before
- 85% threshold on chars/4 provides ~15% safety margin
- First attempt uses quality gate (tool nudge included automatically)
- ~120 lines new code, zero changes to existing function signatures

### Failure modes
All fail-open — worst case is the request hits GPU as it does today.
