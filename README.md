# smart-proxy

A dependency-free Python routing proxy that gives a fleet of local and cloud LLMs one OpenAI-compatible endpoint, with the reliability layer that agent workloads actually need.

Built for and battle-tested on a self-hosted inference server (dual Tesla P40, llama.cpp stacks swapped via Portainer) serving multi-agent traffic around the clock. One measured 40-hour window: 8,129 requests, 95.7% first-attempt pass rate through the quality gate, 0.17% failure rate, zero manual interventions.

## What it does

- **Model aliasing and tiered routing**: clients ask for `daily-driver`, `coder`, or `auto`; the proxy maps aliases to CPU, GPU, or cloud tiers. An `auto-route` path runs a local 4B classifier first and dispatches simple requests to the free CPU tier.
- **GPU stack swapping**: one large model resident at a time, swapped on demand through the Portainer API with an anti-flap cooldown and rollback on failed swap.
- **Quality gate for cloud models**: deterministic validation (tool-call structure, parameters) plus an LLM judge with an explicit rubric, and retry with structured feedback. A local model pre-classifies requests and injects tool nudges, cutting action-request latency from 36-45s to 15-20s.
- **Per-pool concurrency accounting**: caps are keyed to the pool that actually serves the request, not the model id the client sent, because vendors silently merge pools. Every reroute is logged and surfaced in an `X-Routed-Model` header.
- **Protocol translation**: Anthropic-format clients talk to OpenAI-format backends, including SSE streaming and the null-content edge cases that silently kill fallbacks.
- **Structured request logging (REQLOG)**: per-request event stream that made production issues (like a vendor pool merge) visible within minutes.
- **Fallbacks that name their target**: a cloud failure falls back to one pinned local model or fails loudly with the real backend error attached. Never "whatever happens to be loaded."

## Design rules it embodies

1. Routing is never the client's job; clients keep their configured model ids.
2. The routing map lives in code; config keys that document it trigger a drift warning if they disagree.
3. Validate at the protocol edge; exception paths produce exactly the status codes you didn't plan for.
4. A fallback chain is only as good as its error propagation.

## Running

```
cp .env.example .env   # add your keys
docker compose up -d
```

Config in `config.yaml` (model aliases, tiers, caps). See `docs/` in the companion [inference-fleet](../inference-fleet) repo for the measurement history behind the design decisions, and the blog series at michaelbrewer.me for the stories.
