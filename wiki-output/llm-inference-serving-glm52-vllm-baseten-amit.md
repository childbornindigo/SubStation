---
title: "LLM Inference Serving: GLM-5.2 Production Stack & How vLLM Works"
type: wiki-page
domain: ai-agents
status: active
created: 2026-06-25
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/ai-agent
  - island/knowledge
  - type/explainer
  - ai
  - llm-inference
  - serving
  - vllm
  - kv-cache
  - quantization
  - gpu
  - homelab
  - baseten
  - nvidia-dynamo
  - prefill-decode-disaggregation
  - speculative-decoding
  - glm-5-2
parent: "[[AI Agent Island]]"
---

> **TLDR:** Production LLM speed is won through KV-cache routing and prefill-decode disaggregation, not raw GPU FLOPs.

## Summary

Baseten's GLM-5.2 deployment achieves >280 tok/s by stacking five optimizations — custom attention, 4-bit quantization, cache-aware routing, prefill/decode disaggregation, and speculative decoding — all on NVIDIA Blackwell hardware. The fundamentals underneath these gains are explained by vLLM's architecture: GPU memory is the real bottleneck, and every optimization is a strategy to use KV-cache space more efficiently. GLM-5.2 (744B MoE, 40B active params, MIT license) is a frontier-quality open model at 70–80% lower cost, making it the right target for self-hosted agent infrastructure. For agent workloads specifically, cache-aware routing matters most because every agent step resends the same large system prompt — prefix cache hits eliminate redundant compute on every turn.

---

## GLM-5.2 at a Glance

| Property | Value |
|---|---|
| Architecture | 744B-param MoE, 40B active |
| Modes | Thinking + non-thinking |
| Context | Up to 1M tokens |
| License | MIT |
| Serving benchmark | >280 tok/s (Artificial Analysis, Baseten) |
| Cost vs frontier | ~70–80% cheaper |
| Quality reference | ~GPT-5.5 / Opus-4.8 tier |

---

## The Five Production Levers (Baseten)

### 1. Custom Engine with Shared DSA Attention
Baseten updated their runtime to implement GLM-5.2's **shared DSA attention weights** natively. No off-the-shelf runtime handled it at launch — this was required before any other optimization could apply.

### 2. NVFP4 Quantization
- Source weights: FP8 originals
- Tool: NVIDIA ModelOpt
- Output: NVFP4 (4-bit float with dual scale factors to preserve dynamic range)
- Hardware requirement: **NVIDIA Blackwell GPUs** (unlocks faster tensor cores)
- Quality check: validated on **BFCL function-calling benchmark** vs FP8 baseline — within margin of error
- Effect: cuts VRAM bandwidth burden → improves both TTFT and TPS
- **Key rule:** always validate quantization on **agentic benchmarks (BFCL)**, not just perplexity — quality-preserving for chat ≠ quality-preserving for tool calling

### 3. KV-Aware (Cache-Aware) Routing via NVIDIA Dynamo
Route new requests to the worker that already holds the matching KV prefix in cache, skipping expensive prefill re-computation.

- **Reasoning models:** TTFAT (time-to-first-*answer*-token) matters more than raw TTFT. In Baseten's example, 7.1s of a 7.9s budget was reasoning; only 0.8s was prefill — but cutting TTFT to ~800ms still matters for system throughput.
- **Agent workloads:** system prompts are resent every turn; prefix cache hits eliminate redundant compute on every step. This is the **single biggest lever for agent traffic**.

### 4. Prefill–Decode (PD) Disaggregation
The single biggest throughput lever: **2× higher TPS**.

| Phase | Nature | Bottleneck | Sets |
|---|---|---|---|
| Prefill | Builds KV cache from prompt | Compute-bound | TTFT |
| Decode | Generates tokens one at a time | Memory-bandwidth-bound | TPS |

Disaggregation splits these onto separate engine pools, stopping them from competing for the same GPU resources, allowing unequal provisioning, and enabling independent tuning per bottleneck type. Implementation via **NVIDIA Dynamo**: prefill queue, conditional-disaggregation thresholds, NIXL-based KV transfer with a transpose kernel for differing tensor-parallel layouts between pools.

### 5. Multi-Token Prediction (MTP) Speculation
GLM-5.2's improved MTP layer generates draft tokens cheaply with high acceptance rates. Speculation is **lossless** — a verification step guarantees output identity — making this a free TPS gain on compliant models. Tune draft sequence length to balance generation length vs acceptance rate.

**Additional production dials beyond benchmark setup:**
- Task-specific speculators trained on real traffic distributions
- Single-tenant traffic → more consistent prefix cache hits
- PD ratio tuning to match traffic profile
- Parallelism / batch-size tuning for latency-vs-throughput target

---

## vLLM Fundamentals

### Why Serving Is Hard
The constraint is **GPU memory**, not math speed. KV cache is the culprit:

- **Prefill** processes the entire prompt at once, writing each token's key/value "notes" to KV cache. Compute-bound; determines TTFT.
- **Decode** writes one new token per step, appending to cache. Memory-bandwidth-bound; determines TPS.
- KV cache grows proportionally to answer length and lives entirely in GPU VRAM → more concurrent users = more live KV caches = VRAM exhaustion.

### Why Naïve Serving Fails
Reserve one contiguous block per request sized for the worst-case answer:
- Short answers waste reserved space (**over-reservation**)
- Free memory scatters into gaps too small to reuse (**fragmentation**)
- GPU *looks* full while substantial VRAM sits idle → few concurrent users despite available memory

### PagedAttention (vLLM's Core Fix)
vLLM borrows the **virtual memory / paging** concept from OS design:
- KV cache divided into fixed-size **blocks** (pages, e.g. 16 tokens each)
- Blocks allocated **on demand** as the answer grows — no upfront reservation
- A **block table** maps logical token positions to physical block addresses per request
- Non-contiguous physical blocks are fine — the attention kernel walks the page table
- Result: near-zero internal fragmentation, near-zero over-reservation

**Block sharing bonus:** Two requests sharing an identical prefix (same system prompt) point their block tables at the *same* physical blocks — stored once, referenced many times. Beam search beams sharing a common prefix also reuse blocks. This is the same mechanism Dynamo's cache-aware routing exploits at fleet scale.

### Continuous Batching
Traditional static batching waits for the whole batch to finish, leaving finished short requests idle until the longest completes. vLLM uses **continuous (iteration-level) batching**: finished sequences are evicted mid-flight and new requests inserted immediately, every decode step. PagedAttention instantly frees the evicted request's blocks; continuous batching reuses the slot + memory in the same step. The two mechanisms compose.

### Prefix Caching
When multiple requests share a common prefix (system prompt, few-shot examples), vLLM hashes the prefix and reuses its KV blocks across requests — avoiding redundant prefill. This is what Dynamo exploits for cache-aware routing at the fleet level.

### OpenAI-Compatible API
vLLM exposes an OpenAI-compatible API server — existing tooling repoints by changing only the base URL. No client code changes required.

---

## Optimization Hierarchy (How the Levers Stack)

The techniques are **independent and composable**:

```
(memory)      PagedAttention
    ↓
(scheduling)  Continuous batching
    ↓
(precision)   NVFP4 / quantization
    ↓
(routing)     Cache-aware prefix routing
    ↓
(parallelism) Prefill-decode disaggregation
    ↓
(speculation) MTP / speculative decoding
```

Each layer is orthogonal — applying one does not preclude any other.

---

## Implications for Agent & Homelab Workloads

| Scenario | Key Lever |
|---|---|
| Agent loop (same system prompt every turn) | Prefix caching / cache-aware routing |
| Mixed short + long answers | Continuous batching + PagedAttention |
| Cost-sensitive self-hosting on 1–2 GPUs | NVFP4 / GGUF quantization to fit model in VRAM |
| Maximising throughput on single node | PD disaggregation (separate processes even on one machine) |
| Reasoning model (thinking mode) | Optimise TTFAT, not TTFT; cache the reasoning prefix |
| Many parallel agent workers | Shared prefix blocks eliminate per-worker KV redundancy |

GLM-5.2's MIT license makes it viable for production agent infrastructure without vendor lock-in. At 40B active parameters with MoE, it can run on a single-node multi-GPU setup with appropriate quantization. vLLM/SGLang are CUDA-only frameworks — this reinforces Linux/NVIDIA as the right homelab target, not Mac.

**Key takeaways:**
1. Inference is a systems-engineering discipline, not a knob. Start with memory (PagedAttention) before reaching for more GPUs.
2. Prefix/KV-cache reuse is the single biggest lever for agent workloads — keep system prompts stable so the cache hits every turn.
3. Always validate on agentic benchmarks (BFCL) — optimizations that preserve chat quality can degrade tool-calling accuracy.
4. Open frontier models (GLM-5.2) make self-hosting genuinely competitive for non-Anthropic/non-OpenAI bulk inference lanes.

---

## Counter-Arguments

- **PD disaggregation adds network overhead** for KV transfer between prefill and decode nodes; on a single machine the benefit is real but smaller than claimed for multi-node deployments.
- **NVFP4 requires Blackwell hardware** — unavailable in most homelabs today (H100/A100 owners get FP8 at best; consumer GPUs use GGUF/AWQ instead).
- **Prefix caching helps only when prefixes are truly shared** — diverse prompt patterns across users see limited benefit; gains are front-loaded on agent/RAG use cases.
- **Speculation (MTP) acceptance rate is workload-dependent** — high acceptance on typical chat, lower on code/structured output diverging from draft distribution.
- **The Baseten stack is Blackwell-specific** — reproducing the same benchmark on older NVIDIA hardware requires different quantization schemes and will see lower absolute TPS numbers.

---

## Sources

- [[Knowledge/AI/llm-inference-serving-glm52-vllm-baseten-amit.md|llm-inference-serving-glm52-vllm-baseten-amit]]
- [[Wiki/Domains/_shared/llm-inference-serving-glm-5-2-production-stack-how-vllm-work.md|llm-inference-serving-glm-5-2-production-stack-how-vllm-work]]
- [[llm-inference-serving-glm52-vllm-baseten-amit|llm-inference-serving-glm52-vllm-baseten-amit]]
- [[llm-inference-serving-glm-5-2-production-stack-how-vllm-work|llm-inference-serving-glm-5-2-production-stack-how-vllm-work]]

---

## Related

- [[Wiki/Domains/_shared/llm-inference-serving-glm-5-2-production-stack-how-vllm-work|Llm Inference Serving Glm 5 2 Production Stack How Vllm Work]]
- [[linux-offload-syspolicyd-thesis]]
- [[local-llm-bible-stack-theahmadosman]]
- [[homelab-gpu-rig-intel]]
- [[ds4-antirez-s-deepseek-v4-flash-inference-engine]]
