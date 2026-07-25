---
title: TurboQuant — Google's KV-Cache Compression (6x Memory, 8x Speed)
type: wiki-page
domain:
  - ai-agents
  - knowledge-mgmt
status: active
created: 2026-06-06
updated: 2026-07-24
confidence: medium
retention: durable
tags:
  - island/agency
  - island/ai-agent
  - island/knowledge
  - type/wiki-page
parent: "[[Wiki/Domains/_shared/turboquant-google-s-kv-cache-compression-6x-memory-8x-speed|TurboQuant — Google's KV-Cache Compression]]"
---

> **TLDR:** TurboQuant shrinks LLM KV-cache memory ~6x with up to 8x speedup, making long-context local inference practical.

## Summary

TurboQuant is a Google Research technique (open-sourced March 2026) that compresses LLM key-value cache memory while preserving model accuracy. The primary reported outcomes are ~6x lower KV-cache memory usage and up to 8x faster inference with claimed zero accuracy loss. For local deployments — especially on Apple Silicon with limited RAM — this meaningfully extends practical context windows and reduces pressure to offload long-session work to cloud models. Practical benefit depends on runtime integration, with MLX support identified as the key adoption gate.

## What It Is

TurboQuant targets the memory and throughput cost of storing and reading attention keys and values across large token windows — one of the main bottlenecks in long-context inference.

### Reported Outcomes
- ~6x KV-cache memory compression
- Up to 8x inference speedup
- Zero accuracy loss (per source summary)
- Public open-source release, no paywall

## Why It Matters

### For Local Inference
KV-cache growth limits long conversations and document-heavy workflows. If TurboQuant delivers on its claims, the same hardware gains:
- Longer viable context windows
- More comfortable RAM headroom
- Better throughput on extended sessions
- Reduced dependency on cloud model fallback

### For This Stack
TurboQuant is directly relevant to a local MLX setup running Qwen 3.5 35B-A3B on an M4 Mac Mini with 16 GB RAM:
- 100K+ token local conversations become more feasible
- The ~2.7 GB RAM headroom becomes less fragile
- Note-aware research agents can hold more context in a single pass
- The gap between local inference and premium cloud subscriptions narrows

## Impact on Architecture

### Agent Routing
[[agent-routing-rules]] tilts more toward local execution for research and synthesis tasks when context memory cost drops significantly.

### Memory-Heavy Note Workflows
Longer effective context improves per-note reasoning and session continuity, especially where many notes must remain active in the prompt simultaneously.

### Cost Optimization
Supports the direction described in [[cost-optimization-dual-max-sub]] by making fully local or mostly local inference increasingly realistic.

## Operational Status

Research is publicly available. Practical integration depends on runtime support.

| Stage | Status |
|---|---|
| Research published | ✅ Done (March 2026) |
| Open-source release | ✅ Confirmed |
| MLX integration | ⏳ Pending |
| Production use in this stack | ⏳ Blocked on MLX |

Track the MLX project for adoption; Apple local-inference workflows are the highest-leverage target once integration lands.

## Constraints and Unknowns

- Source material is a single note rather than independent corroborating benchmarks.
- Performance claims are summarized without benchmark detail or model-class breakdowns.
- "Zero accuracy loss" may be sensitive to model type, sequence length, and implementation quality.
- "Up to 8x" speedup likely represents best-case, not typical production throughput.
- Benefits to this stack remain contingent on MLX support maturity.

## Counter-Arguments

- KV-cache compression may not be the dominant bottleneck for every workload; model size, memory bandwidth, or generation quality can still limit usefulness independently.
- Longer context windows do not automatically solve retrieval, reasoning, or memory-architecture problems — they reduce one constraint, not all of them.
- Consumer Apple Silicon gains depend on mature runtime integration, not just promising research results.
- Best-case speedup figures may not reflect typical mixed-workload performance.

## Sources

- [[Wiki/Domains/_shared/turboquant-google-s-kv-cache-compression-6x-memory-8x-speed.md|turboquant-google-s-kv-cache-compression-6x-memory-8x-speed]]
- [[turboquant-google-s-kv-cache-compression-6x-memory-8x-speed|turboquant-google-s-kv-cache-compression-6x-memory-8x-speed]]

## Related

- [[agent-routing-rules]]
- [[cost-optimization-dual-max-sub]]
- [[2026-04-10-brain-gardener-retired-replaced-by-hermes|Brain Gardener]]
- [[Knowledge/Reference/turboquant-google-kv-cache-compression|turboquant-google-kv-cache-compression]]
