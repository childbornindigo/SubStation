---
title: Six AI Engineer Judgment Calls (2026)
type: wiki-page
domain: ai-engineering
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - type/concept
  - ai-engineering
  - retrieval
  - agent-memory
  - evaluation
  - hybrid-search
parent: "[[Agency Island]]"
---

> **TLDR:** Six practitioner taste-calls separating real AI engineers from tutorial-followers — retrieval, memory, and eval attribution over model-blaming.

## Summary

A rapid-fire Instagram reel frames six "when/why" judgment calls that distinguish experienced AI engineers: knowing that hallucinations are retrieval failures, that memory is a liability not just an asset, and that eval failures indict the system scaffold before the model weights. The value is the six framings as a review rubric, not implementation recipes. For systems like ObsidianBrain that carry heavy persistent memory and GraphRAG retrieval, calls #1, #2, #3, and #6 function as a direct checklist against common anti-patterns.

## The Six Judgment Calls

| # | Call | Core Insight |
|---|------|--------------|
| 1 | **Hallucinations are retrieval failures** | The model isn't lying — it was handed wrong or empty context. Fix retrieval before blaming the LLM. |
| 2 | **Semantic vs hybrid search** | Pure vector similarity misses exact tokens (IDs, names, codes). Hybrid (vector + keyword/BM25) covers both; know which the query demands. |
| 3 | **When agent memory becomes a liability** | Stale, wrong, or over-broad memory poisons future turns. Sometimes the right move is *no* persistent memory. |
| 4 | **Structured outputs vs reasoning flexibility** | Forcing JSON/schema constrains the model and can suppress reasoning. Worth it for machine-consumed steps; harmful where open reasoning is needed. |
| 5 | **Latency vs accuracy trade-off** | Some paths need a fast 90%-right answer over a slow 99%-right one. Pick per use-case, not globally. |
| 6 | **Eval failure attribution** | A failing eval usually indicts retrieval/orchestration/prompt scaffolding, not the model weights. Localize the fault before swapping models. |

## Application to GraphRAG / Agent Memory

These calls map almost point-for-point onto the ObsidianBrain memory upgrade (graph + vector hybrid, ~9.5k notes / 175k edges, scoped 2026-06-22):

- **#1** is the entire thesis of doing GraphRAG properly — bad brain-layer answers are a retrieval problem to fix in the index, not a model problem to fix by upgrading the LLM.
- **#2** validates the hybrid search choice: the vault is full of exact tokens (`service_role`, `sbp_`, SKUs, project names) that pure embeddings miss — keyword/BM25 alongside vectors is mandatory.
- **#3** is the sharpest caution: Hermes agent infra leans heavily on persistent memory (auto-MEMORY.md, vault context). More memory ≠ better — stale entries actively mislead. Argues for recency decay + relevance gating in retrieval, not "shove the whole 175k-edge graph into context."
- **#6** reinforces the standing rule: measure retrieval quality before trusting answers; when the brain returns junk, suspect the retrieval/orchestration scaffolding first.

## Steal

- Adopt the six as a literal review rubric for any GraphRAG retrieval layer, especially hybrid (vector + keyword) and memory decay/relevance gating.
- Treat agent memory as requiring *eviction*, not just accumulation — counter to the default "remember everything" pattern.
- Note: reel is framing-only, no implementation. It validates direction already chosen; it doesn't add new technique. Treat as a confidence check, not a recipe.

## Counter-Arguments

- All six calls are taste framings, not empirically verified rules. "Hallucinations are always retrieval failures" is a useful heuristic but oversimplifies — some failure modes genuinely live in the model (instruction-following, reasoning, factual knowledge gaps post-cutoff).
- The memory-as-liability framing can overcorrect: stateless agents fail on multi-turn tasks where context accumulation is the whole point. Eviction policy design is non-trivial.
- Source is an Instagram reel with gated full guide (CTA-locked). Confidence is limited to the six headlines; no depth behind each call is provided.

## Sources

- [[Knowledge/AI/six-ai-engineer-judgment-calls-2026-intake.md|six-ai-engineer-judgment-calls-2026-intake]]
- [[six-ai-engineer-judgment-calls-2026-intake|six-ai-engineer-judgment-calls-2026-intake]]

## Related

- [[Wiki/Domains/_shared/6-things-every-ai-engineer-should-know-judgment-calls-2026|6 Things Every Ai Engineer Should Know Judgment Calls 2026]]
- [[karpathy-llm-wiki-field-notes]]
- [[2026-06-26-gbrain-architecture-mem0-article]]
- [[wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi]]
