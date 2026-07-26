---
title: "Wannabe vs $200K AI Engineer — Mistakes (Bashifurikashi)"
type: wiki-page
domain: ai-engineering
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/ai-agent
  - type/contrast
  - ai-engineering
  - rag
  - agents
  - prompting
  - fine-tuning
  - context-engineering
parent: "[[AI Agent Island]]"
---

> **TLDR:** Naive RAG, unscoped agents, and prompt obsession are junior traps — context engineering and evals beat fine-tuning 90% of the time.

## Summary

Bashiri Smith (@bashi_fuirkashi) distills four axes where junior AI engineers get it wrong versus what a $200K practitioner actually does. The core shift is from demo-grade simplicity (chunk-embed-retrieve, LLM-in-a-loop, prompt mastery, train-from-scratch) toward production discipline: hybrid retrieval with re-ranking, scoped deterministic orchestration, context engineering, and retrieval + evals before touching model weights. Directly validates the graph + vector hybrid memory architecture chosen for ObsidianBrain's ~9.5k-note memory layer.

## The Four Contrasts

### 1. RAG

| Level | Belief |
|---|---|
| ❌ Wannabe | "Chunk docs → embed → vector DB → return best match. That's RAG." |
| ✅ Senior | Naive RAG is a demo, not a product. |

**Production checklist:**
- Hybrid search + re-ranking
- Query rewriting before the index is hit
- Measure retrieval quality before trusting any answer

### 2. Agents

| Level | Belief |
|---|---|
| ❌ Wannabe | "I made an agent — LLM in a loop with tools, just let it run." |
| ✅ Senior | An unscoped LLM-in-a-loop burns tokens and ships chaos. |

**Production checklist:**
- Scope the tool surface explicitly
- Keep deterministic steps deterministic
- Orchestrate state (e.g., LangGraph)

### 3. Prompting

| Level | Belief |
|---|---|
| ❌ Wannabe | "I need to master prompt engineering." |
| ✅ Senior | Prompts are ~10% of the job. |

**Key reframe:** Context engineering > prompt engineering. The real question is whether the model has the *right context*, not whether the prompt is clever.

### 4. Fine-Tuning / Training

| Level | Belief |
|---|---|
| ❌ Wannabe | "I should learn to train models from scratch first." |
| ✅ Senior | Fine-tuning is LAST, not first. |

**Rule of thumb:** Retrieval + context + evals solve 90% of any problem. Touch weights only when nothing else moves the number.

## Counter-Arguments

- The "prompts are 10%" framing undersells prompt structure in constrained tasks (tool-calling, structured output) where format discipline is a hard correctness requirement, not style.
- LangGraph is one orchestration choice; the determinism principle applies regardless of framework.
- "90% solved by retrieval + evals" assumes the problem is knowledge-retrieval-shaped; fine-tuning remains necessary for style, format, or latency constraints naive RAG cannot address.

## Sources

- [[Knowledge/AI/wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi.md|wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi]]
- [[wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi|wannabe-vs-200k-ai-engineer-mistakes-bashifurikashi]]

## Related

- [[Wiki/Domains/_shared/wanna-be-vs-200k-ai-engineer-common-mistakes|Wanna Be Vs 200K Ai Engineer Common Mistakes]]
- [[ObsidianBrain]]
- [[agent-native-memory-systems]]
- [[2026-04-14-article-reference-why-rag-and-semantic-memory-alwa]]
- [[agent-as-a-router-acrouter]]
- [[temp-reasoning-data-curation]]
