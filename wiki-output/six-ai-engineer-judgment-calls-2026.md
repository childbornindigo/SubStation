---
title: Six AI Engineer Judgment Calls (2026)
type: wiki-page
domain: ai-agents
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/knowledge
  - island/ai-agent
  - island/agency
  - type/concept
  - ai-engineering
  - retrieval
  - agent-memory
  - evaluation
  - hybrid-search
parent: "[[Agency Island]]"
---

> **TLDR:** Six practitioner taste-calls that separate real AI engineers from tutorial-followers — retrieval, memory, and eval attribution over model-blaming.

## Summary

A practitioner framing of six "when/why" judgment calls that distinguish experienced AI engineers from beginners: hallucinations are retrieval failures, memory is a liability as much as an asset, and eval failures indict the system scaffold before the model weights. The six calls function as a review rubric for any RAG or agent system. For memory-heavy architectures (persistent context, GraphRAG, large vault indexes), calls #1, #2, #3, and #6 are a direct anti-pattern checklist with immediate actionability.

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

These calls map directly onto any large persistent-memory agent system (e.g., GraphRAG vault with ~9.5k notes / 175k edges):

- **#1** is the core thesis of doing GraphRAG properly — bad answers are a retrieval problem to fix in the index, not a model problem to fix by upgrading the LLM.
- **#2** validates hybrid search: vaults are full of exact tokens (service keys, SKUs, project names) that pure embeddings miss. Keyword/BM25 alongside vectors is mandatory, not optional.
- **#3** is the sharpest caution: heavy persistent memory (auto-MEMORY.md, vault context injection) is the default posture, but stale entries actively mislead. More memory ≠ better. Argues for recency decay + relevance gating in retrieval, not "shove the whole graph into context."
- **#6** reinforces a standing rule: measure retrieval quality before trusting agent answers. When the brain returns junk, suspect retrieval/orchestration scaffolding first.

## Steal

- Use the six as a literal review rubric for any RAG or agentic retrieval layer, especially when debugging wrong or hallucinated outputs.
- Treat agent memory as requiring **eviction**, not just accumulation — counter to the default "remember everything" pattern.
- Apply hybrid (vector + keyword) search wherever the query space contains exact tokens alongside semantic concepts.
- Note: the source is framing-only, no implementation depth. It validates direction; it does not add new technique. Treat as a confidence check, not a recipe.

## Counter-Arguments

- All six calls are taste framings, not empirically verified rules. "Hallucinations are always retrieval failures" is a useful heuristic but oversimplifies — some failures genuinely live in the model (instruction-following gaps, factual knowledge post-cutoff, reasoning errors).
- The memory-as-liability framing can overcorrect: stateless agents fail on multi-turn tasks where context accumulation is the whole point. Eviction policy design is non-trivial.
- Source is an Instagram reel with a gated full guide (CTA-locked). Confidence is limited to the six headlines; no implementation depth behind each call is available.

## Sources

- [[Wiki/Domains/_shared/six-ai-engineer-judgment-calls-2026-intake.md|Six AI Engineer Judgment Calls (2026)]]

## Related

- [[Wiki/Domains/_shared/6-things-every-ai-engineer-should-know-judgment-calls-2026|6 Things Every Ai Engineer Should Know Judgment Calls 2026]]
- [[Wiki/Domains/_shared/six-ai-engineer-judgment-calls-2026-intake|Six Ai Engineer Judgment Calls 2026 Intake]]
- [[karpathy-llm-wiki-field-notes]]
- [[2026-06-26-gbrain-architecture-mem0-article]]
- [[wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi]]
