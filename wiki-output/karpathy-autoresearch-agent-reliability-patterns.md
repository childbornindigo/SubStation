---
title: Karpathy Autoresearch Agent Reliability Patterns
type: wiki-page
domain: ai-agents
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/ai-agent
  - island/creator
  - island/security
  - island/agency
  - island/knowledge
  - island/infrastructure
  - island/web-builder
  - island/sales
  - type/wiki-page
  - karpathy
  - autoresearch
  - agent-reliability
  - sub-agents
  - compaction
  - model-failover
  - auth-fallback
  - content-strategy
  - obsidian-pipeline
parent: "[[Wiki/Domains/sales-compass/synthesis-autoresearch-karpathy-agents|Synthesis Autoresearch Karpathy Agents]]"
---

> **TLDR:** Reliable autoresearch requires artifact verification, checkpointed execution, and correct auth/model failover ordering at load time.

## Summary

This page synthesizes operating patterns behind Karpathy-inspired LLM-wiki and autoresearch systems, covering both the technical reliability requirements and the content dynamics that drove adoption. Autonomous research systems fail silently without artifact verification, checkpointed execution, and correctly ordered auth/model failover. The strategic corollary is that implementation-first content compounds trust and reach better than idea-only content — making reliability and distribution part of the same system design.

## Karpathy Content Funnel Pattern

Karpathy's idea-file / LLM-wiki concept became both a content template and an architectural template for systems like [[ObsidianBrain]].

| Layer | Account | Impressions | Angle |
|---|---|---:|---|
| Idea | Karpathy | ~17M | Concept |
| Failure modes | shannholmberg | ~205K | Seven failure modes |
| Implementation | Allie K. Miller ("Claudeopedia") | ~149K | Running demo |
| Tooling | meta_alchemist | ~22K | How-to details |

### What the funnel shows

- **Implementation beats abstraction.** Working systems outperform summaries of ideas.
- **Bookmarks beat likes for utility content.** Claudeopedia's reported **2:1 bookmark-to-like ratio** signals return intent.
- **Specificity compounds value.** Each downstream layer becomes more operational and more reusable.

### Replicable remix pattern

1. Quote the original concept.
2. Add a concrete implementation.
3. Combine existing tools rather than inventing from scratch.
4. Show the system running, not just described.

A practical pattern combines idea-file workflow with `/last30days` and `/wiki` capabilities, then publishes the live result.

## Core Reliability Principle

> **Dispatch confirmation is not evidence of completion.**

Agent work should be treated as complete only when expected artifacts exist and are readable. Reliability must be grounded in outputs, not orchestration optimism.

## Sub-Agent Reliability Rules

Context compaction is the main silent failure mode for spawned agents, based on 18 logged failure events from 2026-04-14/15.

| Rule | Operational meaning |
|---|---|
| **Dispatch ≠ completion** | Never claim batch success until outputs are verified on disk |
| **Compaction kills long flights** | Long-running spawned tasks often die mid-flight without surfaced errors |
| **Verify before claiming** | Read output artifacts after execution rather than trusting spawn status |
| **Honesty protocol** | Report only work that actually landed |
| **Checkpoint-and-resume** | Persist intermediate state so work can survive context loss |

### Practical implications

- Run **atomic bulk operations** in the main session when correctness matters more than concurrency.
- Use sub-agents only for **genuinely independent tasks** with clear output contracts.
- Require every task to emit a **verifiable artifact**: file, log, state snapshot, or structured result.
- Prefer **checkpointed workflows** over long uninterrupted spawned runs.

### Tradeoff

Running in-session reduces compaction risk but consumes orchestrator context and can stall the main thread. Parallelism remains useful, but only when partial and silent failure are assumed upfront.

## Model Failover and Auth Hardening

SubStation-style failover only works when model routing and auth order are configured correctly at load time.

**Failover chain:** `opus → sonnet → blockrun → codex`

```yaml
agents.defaults.model.primary:   indigo/sonnet-4-6
agents.defaults.model.fallbacks: [opus, blockrun/premium, codex]
auth.order:                      oauth → manual → default
```

### Key principle

**Auth order is load-time configuration, not runtime resilience.**  
If the first auth path is dead and retries indefinitely, the system stalls without an explicit error. In that state, model failover never activates because the request never cleanly fails — the failure is invisible.

### Common failure modes

| Symptom | Root cause | Fix |
|---|---|---|
| `"Unknown model"` on `indigo/sonnet-4-6` | Config injection missing at plugin load | Patch config injection and restart gateway |
| Silent stall with no progress | First auth path retrying indefinitely | Set explicit timeout + fallback on auth step |
| Model failover never activates | Auth stall prevents clean failure signal | Ensure auth paths fail fast before model routing |

## Counter-Arguments

- **In-session execution is not always safer.** For very long tasks, orchestrator context exhaustion is also a failure mode — spawning remains necessary, just with failure assumptions baked in.
- **Artifact verification adds latency.** In high-throughput pipelines, read-after-write checks on every task can become a bottleneck; batching verification is a valid tradeoff.
- **Implementation-first content has a ceiling.** Tutorial content compounds early but can plateau once a niche is saturated; concept-layer content has longer tail distribution.

## Sources

- [[Wiki/Domains/_shared/karpathy-autoresearch-agent-reliability-patterns.md|karpathy-autoresearch-agent-reliability-patterns]]
- [[karpathy-autoresearch-agent-reliability-patterns|karpathy-autoresearch-agent-reliability-patterns]]

## Related

- [[Wiki/Domains/sales-compass/autoresearch-architecture-obsidian-refactoring-agent-reliabi|Autoresearch Architecture Obsidian Refactoring Agent Reliabi]]
- [[Wiki/Domains/sales-compass/synthesis-autoresearch-karpathy-agents|Synthesis Autoresearch Karpathy Agents]]
- [[Wiki/Domains/_shared/synthesis-agent-reliability-right-hand-2026-05-15|Synthesis Agent Reliability Right Hand 2026 05 15]]
- [[Wiki/Domains/_shared/autoresearch-self-improving-agent-system|Autoresearch Self Improving Agent System]]
