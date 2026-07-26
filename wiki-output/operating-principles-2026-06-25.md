---
title: Operating Principles — Infra Takeaways We Tackle Now (No Hardware Needed)
type: wiki-page
domain: _shared
status: active
created: 2026-06-25
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/knowledge
  - island/security
  - type/principles
  - infra
  - operating-principles
  - orchestration
  - agents
  - cost
  - metrics
  - planning
parent: "[[Knowledge Island]]"
---

> **TLDR:** Four pure-policy infra wins (no hardware needed) decided 2026-06-25: tame fan-out, kill anti-metrics, route by completed-task cost, spec-first self-config.

## Summary

From a 2026-06-25 planning session with Dee, drawing on the Nerd Snipe podcast (Theo/t3.gg + Ben), four actionable operating principles were extracted that require zero new hardware — only discipline. The Linux/`syspolicyd` offload is parked pending hardware; these four apply immediately on the minis. Each principle addresses a demonstrated failure mode: worker-swarm stalls, gate spam, cheap-model re-dispatch tax, and manual config hunting.

## Principles

### 1. Tame the Fan-Out Multiplier

Total concurrency = `threads × per-thread cap`. At 10 threads × 20 agents = 200 concurrent, the kernel `syspolicyd` tax dominates and workers park rather than accelerate.

- **Rule:** prefer fewer, fatter workers over a swarm; be deliberate about the multiplier before uncapping.
- **Evidence:** 2026-06-25 — higher fan-out correlated with more parked workers needing re-dispatch, not faster throughput.

### 2. Kill Anti-Metric Gates (the G9 Lesson)

A metric that prevents improving the thing it measures is an anti-metric.

- **Theo's case:** a 95%-coverage gate blocked deleting 200k lines of dead code — deleting well-tested junk dropped coverage below the watermark.
- **Our G9:** a vitest gate that auto-re-ran after every job and re-flagged the same items all day — cost more cycles than it caught. It is NOT a security verdict (a real audit found 2 P0s G9 never would).
- **Rule:** when a gate costs more than it catches, mute it or scope it down. G9 = on-demand side regression check only; never auto-spam after every job; never the security source of truth.

### 3. Route by Price-Per-Completed-Task, Not Price-Per-Token

A 2× sticker model that one-shots beats a cheap model that needs three tries.

- **Evidence:** 2026-06-25 — every parked/faked worker needing re-dispatch was the cheap-model tax, paid in time and re-runs.
- **Rule:** route quality-sensitive work to the model that finishes. Stop optimizing the sticker price.

### 4. Spec-First Self-Config

Point the agent at the spec/source + a timestamp and let it configure itself — don't hunt config docs manually.

- Already partially implemented via the durable WO-file pattern.
- **Upgrade:** apply the same pattern to harness/tool config, not only task specs.

## Industry Context (Planning Horizon)

These reads inform architectural bets but require no immediate action:

| Signal | Implication |
|--------|-------------|
| Compute subsidies end by floor collapsing, not gouging | Build for parallelism now; it gets affordable, not pricier |
| Small team + parallel experiments > headcount | Validates the orchestrator-swarm bet (50-person OpenAI leapfrogged 2,000-person MSFT team) |
| Local/open-weight: good for single-task, bad for parallel loops | Closed APIs buy parallelism = our edge; local is a fallback only |
| ~⅓ of real prompts have images | Open-weight models with no image input are disqualified from our pipeline |
| Prefer API-drivable models | Walled models (Cursor Composer, IDE-only) are dead to our pipeline regardless of quality |
| Enterprise runs ~2 years behind | The gap between what we run daily and what a bank runs is the sellable delta — the LuxuryLane/MyPeptide done-for-you-agent-infra thesis |

## Counter-Arguments

- **Against fan-out caps:** some burst workloads are genuinely parallelizable and capping adds latency. Counter: measure completed-task throughput, not token velocity — the cap is a starting discipline, not a hard ceiling.
- **Against killing G9-style gates:** coverage gates catch regressions that audits miss between cycles. Counter: scope them to new code only; a gate that re-flags frozen dead code is pure noise.
- **Against price-per-completed-task routing:** it requires knowing which tasks are quality-sensitive upfront. Counter: the default should be the model that finishes; downgrade only after proving a cheaper model reliably one-shots for a specific task class.

## Sources

- [[Knowledge/Reference/operating-principles-2026-06-25.md|operating-principles-2026-06-25]]
- [[operating-principles-2026-06-25|operating-principles-2026-06-25]]

## Related

- [[Wiki/Domains/_shared/operating-principles-infra-takeaways-we-tackle-now-no-hardwa|Operating Principles Infra Takeaways We Tackle Now No Hardwa]]
