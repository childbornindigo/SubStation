---
title: Blockrun To Codex Cron Job Migration
type: wiki-page
domain: shared
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/knowledge
  - island/infrastructure
  - island/trading
  - type/wiki-page
parent: "[[Builder Island]]"
---

> **TLDR:** Hybrid routing wins: 13/22 cron jobs can move to Codex; 7 timeout-prone jobs stay on Blockrun/free.

## Summary

An April 2026 audit of 22 cron jobs found 13 had zero Blockrun-specific tool dependencies and were safe to migrate to Codex-based execution. A blanket migration was blocked by `indigo-codex` (`gpt-5.1-codex`) imposing a hard ~61-second provider timeout, causing 7 longer-running jobs to fail. The practical architecture is hybrid routing: Codex for short plain-LLM tasks, Blockrun/free for anything timeout-sensitive or tool-dependent. Routing decisions must consider full pipeline runtime, not just per-step cost.

## Migration Assessment

### Movable Job Inventory

The audit identified **13 of 22 cron jobs** as migration candidates — they used **zero Blockrun-specific tools**.

### Migration Rationale

- Reduce unnecessary Blockrun dependence
- Simplify routing for plain LLM workloads
- Improve cost and infrastructure efficiency

### Limitation

A blanket migration was unsafe: some jobs exceeded the Codex provider timeout ceiling and became unreliable.

## Reliability Findings

### Timeout Root Cause

`indigo-codex` (`gpt-5.1-codex`) carries a **hard provider timeout of ~61 seconds**. Jobs exceeding that runtime failed.

### Operational Fix

All 7 failing jobs were switched to `blockrun/free`, which:

- Tolerated longer runtimes
- Added no extra reported cost
- Restored cron reliability immediately

### Design Implication

Cron routing must be based on **runtime profile and tool dependency**, not model capability alone.

## Routing Decision Framework

### Use Codex When

- Job uses no Blockrun-specific tools
- Task is mostly text generation or reasoning
- Expected runtime is safely below ~60 seconds

### Use Blockrun/free When

- Job may exceed ~60 seconds
- Previous Codex timeout failures exist
- Reliability matters more than provider consolidation
- Workflow needs a more tolerant execution window

## Trading Workflow Example

A related trading-research workflow illustrates why environment selection matters even for inexpensive tasks.

### Example Workflow

A swarm-dispatch process queried Predexon and [[polymarket|Polymarket]] to:

- Fetch the top 5 [[polymarket|Polymarket]] wallets by weekly PnL
- Retrieve detailed wallet profiles
- Classify traders as momentum, event-driven, fade, or copy
- Produce a markdown research table

### Cost Profile

Total reported cost: **~$0.026**

| Step | Cost |
|---|---|
| Leaderboard query | ~$0.001 |
| 5 wallet profiles | ~$0.005 each |

### Why It Matters

Even low-cost multi-step research jobs can exceed brittle timeout limits. Routing decisions must account for full pipeline behavior, not only per-step model cost.

## Practical Conclusions

- Migration opportunity was meaningful but partial: **13/22 jobs** movable
- Reliability constraints blocked a blanket Codex migration
- **7 timeout-prone jobs** are better kept on `blockrun/free`
- Recommended architecture: **hybrid routing by runtime, job type, and Blockrun-tool dependency**

## Counter-Arguments

- The 61-second ceiling may be a provider configuration issue rather than a permanent Codex limitation
- Mixed Blockrun/Codex routing adds operational complexity compared with a single-provider setup
- Future provider timeout improvements could make more jobs safely migratable, reducing the need for Blockrun retention

## Sources

- [[Wiki/Domains/_shared/blockrun-to-codex-cron-job-migration.md|blockrun-to-codex-cron-job-migration]]
- [[blockrun-to-codex-cron-job-migration|blockrun-to-codex-cron-job-migration]]

## Related

- [[Wiki/Domains/_shared/synthesis-blockrun-cron-cost-optimization-infrastructure-sch|Synthesis Blockrun Cron Cost Optimization Infrastructure Sch]]
- [[Wiki/Domains/_shared/synthesis-cron-2026-04-14-blockrun-to-codex-cron-job-migrati|Synthesis Cron 2026 04 14 Blockrun To Codex Cron Job Migrati]]
