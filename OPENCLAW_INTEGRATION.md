# OpenClaw Integration with LLM Server Quality Gate

## Provider Configuration

Add this provider to `openclaw.json`. The quality gate proxy runs on the LLM server and handles all Z.AI authentication server-side — no API key needed from OpenClaw.

```json
"quality-gate": {
  "baseUrl": "http://agent-host.tailnet:4000",
  "api": "anthropic-messages",
  "apiKey": "not-required",
  "models": [
    {
      "id": "glm-5-turbo",
      "name": "GLM-5 Turbo (Quality Gate)",
      "reasoning": true,
      "input": ["text"],
      "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
      "contextWindow": 204800,
      "maxTokens": 131072
    },
    {
      "id": "glm-5",
      "name": "GLM-5 (Quality Gate)",
      "reasoning": true,
      "input": ["text"],
      "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
      "contextWindow": 204800,
      "maxTokens": 131072
    }
  ]
}
```

The endpoint is `POST /v1/messages` — standard Anthropic Messages API. Streaming and non-streaming both work.

## What the Quality Gate Does (so the agent doesn't have to)

The proxy automatically:
1. Forwards requests to Z.AI's Anthropic endpoint
2. Evaluates every response using the active local LLM (~10 t/s, 1-3s overhead)
3. Catches: empty payloads, missing tool calls, truncated responses, incoherent output
4. On failure: retries Z.AI with judge feedback appended (up to 2 retries, 3 total attempts)
5. Returns the best response if all attempts fail

**Automatically skipped** (no evaluation overhead):
- `tool_result` messages
- HEARTBEAT / HEARTBEAT_OK messages
- Simple acks (ok, thanks, continue, etc.)

## Agent Instruction Changes

**Remove from agent instructions:**
- Any retry logic for empty GLM responses
- Any fallback handling for missing tool calls
- Any "if response is empty, try again" patterns

**Add to agent instructions:**
- When using `quality-gate:glm-5-turbo`, responses are pre-validated. If you receive a response, it has been verified for completeness and proper tool use.
- To bypass quality evaluation on a specific request, add header `X-Skip-Quality-Gate: true`
- Expect ~1-3 seconds additional latency per request (judge evaluation time)

## Model Assignments

Update agent model assignments from `zai:glm-5-turbo` to `quality-gate:glm-5-turbo`. The original `zai` provider can stay as a fallback.

## Health Check

```
GET http://agent-host.tailnet:4000/health
GET http://agent-host.tailnet:4000/v1/status  (shows active model, gate stats)
GET http://agent-host.tailnet:4000/v1/models  (all available models)
```

## Network

Tailscale route from NUC to LLM server on port 4000 is already open. No firewall changes needed.
