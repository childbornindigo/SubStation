---
title: Knowledge Management And Memory Systems Architecture
type: wiki-page
domain: ai-agents, knowledge-mgmt
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/ai-agent
  - island/knowledge
  - island/agency
  - type/wiki-page
  - knowledge-management
  - memory-systems
  - architecture
  - retrieval
  - obsidian
  - wiki
  - archivist-oss
parent: "[[Wiki/Domains/_shared/llm-architecture-karpathy-knowledge|Llm Architecture Karpathy Knowledge]]"
---

> **TLDR:** Separate sources from compiled wiki memory, then isolate domains to improve provenance, retrieval quality, and agent decisions.

## Summary

This architecture defines a durable memory system for agents by separating immutable source material from LLM-compiled wiki pages governed by explicit schema rules. The wiki serves as agent-readable working memory, while sources remain the evidence layer with preserved provenance. This matters because it improves retrieval quality, supports contradiction checking and iterative synthesis, and scales from a single vault to multi-domain or multi-business agent systems.

## Core Architecture

### Three-layer compiler model

The canonical pipeline has three layers:

1. **Sources/** — raw, read-only inputs
2. **Wiki/** — synthesized markdown output for agent use
3. **Schema/config** — rules for compilation, linking, and page structure

This is a compiler pattern rather than simple folder hygiene. Sources hold evidence; wiki pages hold conclusions shaped for retrieval and reasoning.

### Why the split matters

- Prevents evidence from being overwritten by summaries
- Preserves traceability from claim back to source
- Enables repeated synthesis without mutating originals
- Makes the wiki a true memory layer instead of a note archive

A related graph rule is to link compiled pages upward to abstractions or islands rather than collapsing directly into raw private notes.

## Retrieval and Knowledge Compounding

### Query → wiki feedback loop

Knowledge quality should improve through use, not only ingestion. The recurring loop is:

1. Ingest source material
2. Compile or update wiki pages
3. Query the wiki
4. Detect gaps, contradictions, or weak synthesis
5. Update wiki pages again

The result is compounding memory quality over time.

### `index.md` as lightweight retrieval infrastructure

At small-to-medium scale, a structured `index.md` that the model reads first can outperform more complex retrieval setups on cost and simplicity.

Benefits:
- Lower system complexity
- Predictable navigation entry point
- Reduced search cost
- Better domain orientation before broader lookup

At very large scale, a static index can become a bottleneck and vector retrieval may become more effective. The transition point is context-dependent.

## Operational Patterns

### High fan-out from each source

One source may update roughly **10–15 wiki pages**. This indicates a graph-centric system where new material enriches many conceptual nodes rather than producing one isolated note.

### Contradiction checking and linting

A Karpathy-style linting pattern can be applied to knowledge systems: the model checks compiled output against existing wiki memory for contradictions. This makes the wiki a consistency layer, not just a storage layer.

### Idea files over code-first sharing

Architecture can be transmitted as structured concepts and operating patterns, not only executable software. This is useful when the main leverage is information design, synthesis rules, and memory structure.

## Domain Isolation and Scaling

### Per-domain boundaries

Before higher-level orchestration, each domain or business should have its own isolated memory environment:

- Separate **Sources/**
- Separate **Wiki/**
- Domain-specific synthesis and retrieval rules

### Why isolation comes first

Without isolation:
- Retrieval becomes noisy
- Decision criteria blur across contexts
- Relevance declines
- Supervisory or council-style systems lack clean boundaries

With isolation, a higher-level layer can compare domains without contaminating local memory.

## Memory-System Learnings

### Archivist-Oss pattern convergence

Analysis of Archivist-Oss suggests memory architecture value comes mostly from reusable patterns rather than storage mechanics alone.

Key findings:
- **10 novel patterns** were identified
- **9 were already adopted**
- A small set of additional patterns still appeared worthwhile
- Weaknesses were documented alongside strengths

### Implication

The architecture appears to be converging on a mature pattern set. Future gains likely come less from inventing new primitives and more from improving retrieval fidelity, contradiction detection, and domain-specific memory quality.

## Design Implications

### What this architecture optimizes for

- Durable provenance
- Agent-readable memory
- Incremental synthesis without source mutation
- Domain isolation before cross-domain orchestration

### What it trades off

- Higher upfront structure cost
- Requires discipline to keep sources immutable
- Index-based retrieval has a scale ceiling

## Counter-Arguments

- **Vector retrieval may supersede index.md earlier than expected** — embedding costs have dropped; semantic search at medium scale may now be cost-competitive with structured index navigation.
- **Domain isolation adds operational overhead** — separate Sources/Wiki per domain multiplies maintenance surface; teams with limited bandwidth may prefer a single vault with strong tagging.
- **Compiler pattern requires schema discipline** — benefits depend on consistent schema enforcement; without it, the wiki layer degrades into another note dump.

## Sources

- [[Wiki/Domains/_shared/knowledge-management-and-memory-systems-architecture.md|knowledge-management-and-memory-systems-architecture]]
- [[knowledge-management-and-memory-systems-architecture|knowledge-management-and-memory-systems-architecture]]

## Related

- [[Wiki/Domains/_shared/llm-architecture-karpathy-knowledge|Llm Architecture Karpathy Knowledge]]
- [[Wiki/Domains/_shared/knowledge-llm-architecture-karpathy|Knowledge Llm Architecture Karpathy]]
- [[Wiki/Domains/sales-compass/synthesis-obsidian-vault-hygiene-knowledge-management|Synthesis Obsidian Vault Hygiene Knowledge Management]]
- [[Wiki/Domains/sales-compass/architecture-knowledge-karpathy-llm|Architecture Knowledge Karpathy Llm]]
