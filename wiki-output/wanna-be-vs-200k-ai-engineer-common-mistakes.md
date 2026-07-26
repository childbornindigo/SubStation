---
title: Wanna-Be vs 200K AI Engineer Common Mistakes
type: wiki-page
domain: ai-agents
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/ai-agent
  - island/knowledge
  - type/wiki-page
  - ai-engineering
  - rag
  - agents
  - retrieval
  - evals
  - fine-tuning
  - context-engineering
parent: "[[AI Agent Island]]"
---

> **TLDR:** Strong AI engineering fixes retrieval, orchestration, and context first — prompts and fine-tuning come last.

## Summary

Four recurring mistakes separate demo-style AI app building from production-grade AI engineering. Most failures stem from weak system design, not weak model behavior. Reliable AI systems prioritize retrieval quality, bounded orchestration, context assembly, and evaluation coverage — in that order — before touching prompt polish or model weights.

## Core Thesis

Beginners frame performance as a prompting or model problem. Production engineers treat it as a systems problem: retrieval, control flow, context construction, and measurement. That shift determines where teams invest effort and where they find the biggest gains.

## Common Mistakes

### 1. Treating Naive RAG as Production RAG

**Beginner pattern**
- Chunk documents, embed them, store in a vector DB, return nearest matches

**Production pattern**
- Hybrid search (not vector-only)
- Re-ranking after first-pass retrieval
- Query rewriting or transformation before search
- Direct measurement of retrieval quality

Naive RAG is sufficient for demos; it is rarely reliable enough for production.

### 2. Building Agents as Unscoped LLM Loops

**Beginner pattern**
- LLM in a loop, tools available, free to decide

**Production pattern**
- Tightly scoped tool access
- Deterministic steps kept deterministic
- Explicit state management
- Graph or state-machine orchestration where helpful

Unbounded loops increase token cost, unpredictability, and operational risk. Good agent engineering is mostly constraint design and control flow.

### 3. Overvaluing Prompt Engineering

**Beginner pattern**
- Treat prompting as the primary engineering skill

**Production pattern**
- Prompts are one layer among many
- Focus on whether the model has the right context
- Invest in context engineering over prompt polish

The right context with an average prompt often outperforms a polished prompt with weak context.

### 4. Reaching for Fine-Tuning Too Early

**Beginner pattern**
- Train or fine-tune to fix application issues

**Production pattern**
- Improve retrieval → improve context design → improve evals → fine-tune only after other levers stop producing gains

Fine-tuning is a late-stage optimization, not the default first move.

## Engineering Priority Order

1. **Retrieval quality**
2. **Context assembly**
3. **Evaluation coverage**
4. **Prompting**
5. **Fine-tuning** (if still needed)

## Implications for System Design

### Retrieval Systems
Treat retrieval as a first-class subsystem: hybrid retrieval, query transformation, re-ranking, and retrieval-specific evaluation metrics.

### Agent Systems
Favor bounded tool access, explicit state transitions, deterministic handling where possible, and orchestration over autonomy-by-default.

### Knowledge and Memory Architectures
For large note systems and AI knowledge bases, graph-plus-vector memory aligns with this framework. Hybrid retrieval across semantic similarity and link structure is often the difference between a convincing demo and a dependable production layer.

## Counter-Arguments

- Naive RAG may be sufficient for narrow, low-risk, or internal tasks.
- Prompt engineering can produce fast wins before deeper infrastructure exists.
- Fine-tuning may be justified earlier for highly specialized, latency-sensitive, or strict-format tasks.
- Autonomous agents can be acceptable in exploratory or research settings where unpredictability is tolerable.

## Sources

- [[Wiki/Domains/_shared/wanna-be-vs-200k-ai-engineer-common-mistakes.md|wanna-be-vs-200k-ai-engineer-common-mistakes]]
- [[wanna-be-vs-200k-ai-engineer-common-mistakes|wanna-be-vs-200k-ai-engineer-common-mistakes]]

## Related

- [[Wiki/Domains/_shared/wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi|Wannabe Vs 200K Ai Engineer Mistakes Bashifurikashi]]
- [[Wiki/Domains/_shared/six-ai-engineer-judgment-calls-2026-intake|Six Ai Engineer Judgment Calls 2026 Intake]]
- [[Wiki/Domains/_shared/6-things-every-ai-engineer-should-know-judgment-calls-2026|6 Things Every Ai Engineer Should Know Judgment Calls 2026]]
- [[AI Agent Island]]
