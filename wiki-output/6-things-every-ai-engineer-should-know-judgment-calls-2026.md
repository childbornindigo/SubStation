---
title: 6 Things Every AI Engineer Should Know — Judgment Calls 2026
type: wiki-page
domain: ai-agents
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - type/wiki-page
  - island/ai-agent
  - island/agency
  - island/knowledge
  - ai-agents
  - knowledge-mgmt
  - retrieval
  - rag
  - hybrid-search
  - memory
  - evals
  - latency
  - structured-output
parent: "[[Agency Island]]"
---

> **TLDR:** Most AI engineering failures are system design mistakes, not model quality problems.

## Summary

Six recurring judgment calls shape production AI quality more than model selection or prompt tweaks. Failures commonly attributed to the model typically originate in retrieval, memory management, orchestration, output constraints, or latency tradeoffs. The operational mandate is to debug and optimize the surrounding system first, then reassess model choice. This framework applies directly to RAG, agent, and knowledge-heavy workflows.

## Core Claim

Good AI engineering depends less on selecting the "best" model and more on sound system-level decisions. In production, answer quality is shaped by what context is retrieved, how memory is filtered, how tools are orchestrated, and what performance constraints the product imposes.

## The Six Judgment Calls

### 1. Hallucinations Are Often Retrieval Failures

In retrieval-augmented systems, the model reasons only over the context it receives. Missing, poorly ranked, stale, or irrelevant context produces answers that look like model hallucinations but are retrieval defects.

- Inspect indexing, chunking, ranking, and context assembly first
- Benchmark retrieval separately from generation
- Treat hallucination diagnosis as a retrieval debugging task before a model selection task

### 2. Semantic vs Hybrid Search Is Query-Dependent

Semantic search works well for conceptual similarity and paraphrased queries but underperforms on exact identifiers, names, codes, SKUs, and literal strings. Hybrid retrieval combines vector and keyword/BM25 to support both modes.

| Use semantic when | Use hybrid when |
|---|---|
| Queries are conceptual | Queries include identifiers or code tokens |
| Wording is uncertain | Exact recall matters |
| Similarity matters more than literal match | Corpus contains terminology embeddings may miss |

### 3. Memory Helps Only When Bounded and Relevant

Persistent memory improves continuity, but unbounded memory introduces stale assumptions, irrelevant carryover, and context pollution. More memory is not inherently better.

- Use eviction and decay
- Gate memory by relevance and recency
- Retrieve memory selectively rather than injecting everything

### 4. Structured Outputs Trade Flexibility for Reliability

Schemas and JSON improve machine-readability and downstream automation safety, but constrain reasoning and reduce useful flexibility in exploratory tasks.

| Use structured outputs for | Avoid over-constraining when |
|---|---|
| APIs and deterministic handoffs | The task requires exploration |
| Workflows with exact field requirements | Reasoning quality benefits from open-ended synthesis |

### 5. Latency May Matter More Than Maximum Accuracy

A slower answer with marginally higher quality is not always the right product choice. Many systems benefit more from fast, good-enough responses than from peak accuracy with high delay.

Decision factors: user tolerance for delay, cost of incorrect answers, interactive vs offline usage, availability of fallback or escalation paths.

### 6. Eval Failures Are Often System Failures

Poor evaluation outcomes may result from retrieval defects, prompt formatting, tool orchestration errors, or weak context construction — not the underlying model. Swapping models without fault localization hides the real bottleneck.

**Recommended debug order:**
1. Check retrieval quality
2. Check prompt and context formatting
3. Check tool and orchestration behavior
4. Reassess model choice last

## Design Implications for Agent and Knowledge Systems

These judgment calls are especially important in RAG, GraphRAG, and agent-memory architectures. Knowledge systems should optimize retrieval fidelity, exact-match support, memory hygiene, and fault isolation before expanding context windows or upgrading models.

**Operational heuristics:**
- Treat bad answers as retrieval problems first
- Support hybrid retrieval in mixed conceptual and exact-match corpora
- Prevent memory pollution through recency and relevance controls
- Separate system evaluation from model evaluation

## Counter-Arguments

**Against "retrieval before model":** Some hallucinations genuinely originate in model weight biases or training data gaps, not retrieval. Retrieval-first diagnosis can delay identifying model-level capability ceilings.

**Against hybrid search always:** Hybrid retrieval adds operational complexity and latency. For narrow, well-defined domains with consistent phrasing, pure semantic search may be sufficient and cheaper to maintain.

**Against latency-first framing:** For high-stakes domains (medical, legal, financial), accuracy costs can far outweigh latency costs, making maximum accuracy the correct product tradeoff even with slower responses.

## Sources

- [[Wiki/Domains/_shared/6-things-every-ai-engineer-should-know-judgment-calls-2026.md|6-things-every-ai-engineer-should-know-judgment-calls-2026]]
- [[6-things-every-ai-engineer-should-know-judgment-calls-2026|6-things-every-ai-engineer-should-know-judgment-calls-2026]]

## Related

- [[Wiki/Domains/_shared/six-ai-engineer-judgment-calls-2026|Six AI Engineer Judgment Calls 2026]]
- [[Wiki/Domains/_shared/six-ai-engineer-judgment-calls-2026-intake|Six AI Engineer Judgment Calls 2026 Intake]]
