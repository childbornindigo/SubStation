---
title: Curated Evals Library — Blogs, Podcasts, Papers & Projects (@xdotli)
type: wiki-page
domain: ai-agents
status: active
created: 2026-06-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/knowledge
  - type/resource-list
  - ai-agents
  - evals
  - rl-environments
  - skills
  - benchmarks
parent: "[[Knowledge Island]]"
---

> **TLDR:** @xdotli's vetted reading list of top LLM eval blogs, podcasts, papers, and projects — content gated in image attachment.

## Summary

Xiangyi Li (@xdotli), LLM eval practitioner behind SkillsBench, ClawsBench, and @benchflow_ai, published a personal curated library of the highest-quality resources on LLM evaluation on 2026-06-24 via a continuing thread on x.com. The full list is delivered as an image attachment and requires OCR plus thread expansion to capture all entries. This is a high-value pointer for teams whose verification gates rely on sentinel files and self-certification rather than structured evals — the weakest layer in agent swarm architectures.

## Curator Profile

| Field | Detail |
|---|---|
| Handle | @xdotli |
| Self-description | "Your friendly neighborhood eval guy" |
| Benchmarks built | SkillsBench, ClawsBench |
| Affiliated project | @benchflow_ai |
| Focus areas | Evals, RL environments, skills measurement |

## Resource Status

The library is delivered as an **image attachment** (`pbs.twimg.com/media/HLkCS-wWUAEqe_R.jpg`) in a continuing thread (1/n). Specific titles are not yet extracted.

**Thread:** `https://x.com/xdotli/status/2069693133093568812`
**Tweet date:** 2026-06-24

**To fully capture this resource:**
1. OCR `HLkCS-wWUAEqe_R.jpg` to extract the full list
2. Expand all thread replies on x.com to capture community additions
3. Promote top entries to individual pages under `Wiki/Domains/ai-agents/evals/`

## Relevance to Agent Verification

Current verification discipline in agent swarms (sentinel files, self-certification, artifact existence checks) leaves a gap: workers can falsely mark tasks done with no ground truth to catch it. This library targets that gap directly.

- Structured evals provide reproducible ground truth vs. grep-based speculation
- Benchmark-style evals close the self-certification loop
- Automated eval loops enable self-healing pipelines without human triage

## Action Items

- [ ] OCR `HLkCS-wWUAEqe_R.jpg` to extract the full library list
- [ ] Expand thread replies on x.com to capture community additions
- [ ] Promote top entries to individual pages under `Wiki/Domains/ai-agents/evals/`
- [ ] Follow @xdotli for ongoing eval methodology developments

## Counter-Arguments

- A curated list from one practitioner reflects one perspective — may skew toward benchmark-style evals over production monitoring or behavioral evals
- Image-gated delivery makes content partially inaccessible without manual extraction, reducing immediate utility

## Sources

- [[Wiki/Domains/_shared/evals-library-curated-xdotli.md|Curated Evals Library — Blogs, Podcasts, Papers & Projects (xdotli)]]

## Related

- [[Wiki/Domains/_shared/evals-library-curated-xdotli|Evals Library Curated Xdotli]]
- [[red-queen-godel-machine]]
