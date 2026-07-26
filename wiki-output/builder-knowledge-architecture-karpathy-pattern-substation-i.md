---
title: 'Builder Knowledge Architecture: Karpathy Pattern, SubStation Infra & Agent Hierarchy'
type: wiki-page
domain:
  - ai-agents
  - knowledge-mgmt
status: active
created: 2026-05-15
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/ai-agent
  - island/creator
  - island/security
  - island/trading
  - island/agency
  - island/knowledge
  - island/infrastructure
  - island/web-builder
  - island/sales
  - type/wiki-page
parent: '[[Knowledge Island]]'
---

> **TLDR:** Index-first, vectorless wiki architecture is externally validated and operationalized through SubStation runtime and agent coordination patterns.

## Summary

This page describes a knowledge architecture that turns raw captures into durable wiki pages through an index-first, exploration-driven compiler pipeline. It replaces heavier RAG infrastructure with a simpler full-text and structured-index model that is accurate at current vault scale and compounds through use. Independent alignment with Karpathy's published knowledge-base pattern strengthens confidence in the approach. SubStation runtime practices make the system operational by handling token rotation, identity injection, cron reliability, and parallel audits.

## Architecture Pattern

### Three-Layer Knowledge Stack

| Layer | Purpose |
|---|---|
| `Sources/` | Raw captures, logs, and notes; intake zone only |
| `Knowledge/Synthesis/` | Clustered, machine-generated intermediate artifacts |
| `Wiki/` | Human-readable, evergreen, cross-linked pages following schema standards |

The stack enforces a clean promotion path from unstructured input to reusable knowledge. Source intake must not bypass this pipeline.

### Core Principles

- **`index.md` first** — agents read a structured entry point before search, reducing latency and retrieval complexity
- **Query → wiki feedback loop** — exploration generates new durable pages, so the graph improves through use
- **Idea file pattern** — the transferable artifact is the concept and architecture, not necessarily the code
- **Wiki as coordination substrate** — multiple agents can use shared pages as common evidence for planning, debate, and execution

## External Validation

Karpathy's "LLM Knowledge Bases" post independently matched the in-house architecture almost one-for-one. The strongest overlap is the `index.md` first-read pattern, the exploration-driven wiki growth loop, and the idea-file approach to sharing systems. This suggests convergent design rather than internal overfitting. The post's 17M impressions indicate the pattern resonates broadly beyond this implementation.

## Retrieval Model

### Vectorless RAG

The current retrieval model favors structured indexes and full-text search over embeddings and chunking.

| Claim | Detail |
|---|---|
| Accuracy | `pageindex-vectorless-rag` reached **98.7%** at current vault scale |
| Vector DB | Not required |
| Embedding pipeline | Not required |
| Chunking overhead | Not present at current scale |

This advantage is explicitly scale-dependent. As vault size grows, the tradeoff with classical vector RAG should be re-evaluated.

## Knowledge Compiler

The compiler transforms synthesis clusters into stable wiki pages using an index-first workflow and an exploration feedback loop. Documented on 2026-04-10 and validated by the May 2026 wiki batch, which produced **17+ durable pages**, confirming the pipeline is active and generating reusable knowledge assets.

## SubStation Runtime

### Token Rotation

SubStation rotates among three Anthropic tokens on rate limit:

| Token | Role |
|---|---|
| `backup:anthropic:oauth` | Primary workhorse (~638 observed requests) |
| `ap:anthropic:oauth` | Secondary |
| `ap:anthropic:manual` | Tertiary |

### Runtime Gap

The system lacks automated cross-provider fallback. If all Anthropic tokens exhaust, SubStation fails rather than rerouting to another provider automatically. A manual emergency override exists at `/override?model=<provider>` but is not part of standard failover.

### Inline Identity Pattern

`CLAUDE.md` should be injected inline rather than read as a file at runtime. This preserves zero-latency identity loading and mirrors native Claude Code system prompting behavior.

## Agent Hierarchy

### Vegapunk Mapping

The hierarchy was corrected and locked on **2026-04-10 14:25**.

| One Piece Role | Indigo OS Mapping |
|---|---|
| Vegapunk (creator) | Dee |
| Stella (wisdom satellite, closest proxy) | Indigo |

All other satellites are subordinate worker agents. The operating assumption is intentionally asymmetric authority and trust rather than peer-flat delegation.

## Operational Patterns

### Cron Hardening

A 2026-04-10 SOP patched **10 failing `indigo-codex` cron jobs**. Reliable scheduling is a prerequisite for the pipeline running unattended.

### Parallel Audits

SubStation supports parallel agent execution, enabling concurrent audits across knowledge domains without serialization bottlenecks.

## Counter-Arguments

- **Vectorless RAG is scale-limited** — full-text and index-based search will degrade as vault size grows into the hundreds of thousands of nodes; the 98.7% figure is vault-size-specific and should not be treated as a permanent ceiling.
- **Manual failover is a reliability risk** — the `/override` escape hatch requires human intervention; an automated cross-provider fallback would be more resilient.
- **Asymmetric agent hierarchy creates single points of failure** — authority concentration in Dee → Indigo means workflow stalls if either is unavailable.

## Sources

- [[Wiki/Domains/_shared/builder-knowledge-architecture-karpathy-pattern-substation-i.md|builder-knowledge-architecture-karpathy-pattern-substation-i]]
- [[builder-knowledge-architecture-karpathy-pattern-substation-i|builder-knowledge-architecture-karpathy-pattern-substation-i]]

## Related

- [[Wiki/Domains/sales-compass/pipeline-and-website-builder-architecture-summary|Pipeline And Website Builder Architecture Summary]]
- [[Wiki/Domains/sales-compass/synthesis-substation-api-tokens-architecture|Synthesis Substation Api Tokens Architecture]]
- [[Wiki/Domains/sales-compass/site-intel-builder-architecture|Site Intel Builder Architecture]]
- [[Wiki/Domains/sales-compass/builder-site-architecture-intel|Builder Site Architecture Intel]]
- [[Wiki/Domains/sales-compass/site-intel-architecture-builder|Site Intel Architecture Builder]]
