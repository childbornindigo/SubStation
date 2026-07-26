---
title: AI Agency Operations — Hormozi Pricing, Intel Build Pipeline, OpenClaw
type: wiki-page
domain: _shared
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/ai-agent
  - island/agency
  - island/knowledge
  - island/infrastructure
  - island/web-builder
  - island/sales
  - type/wiki-page
  - hormozi
  - build-pipeline
  - openclaw
  - automation
  - business-model
  - site-generation
  - devops
parent: '[[sales-compass]]'
---

> **TLDR:** $2K offer at $5 marginal cost via one system linking sales intel, automated site delivery, and tuned agent infrastructure.

## Summary

This page describes an AI agency operating model built from three tightly coupled parts: Hormozi-style pricing, an intel-driven website generation pipeline, and supporting OpenClaw plus Obsidian knowledge infrastructure. Sales automation and fulfillment automation are not separate systems — they run on the same inputs, workflows, and deployment stack. This collapses outreach, proposal generation, and fulfillment into one pipeline, making the economics credible and creating a clear path from website sales into retainers.

## Business Model

### Core Economics

| Metric | Value |
|---|---|
| Offer price | ~$2,000 |
| Marginal delivery cost | ~$5 |
| Task decomposition | 4–10 discrete tasks per engagement |

Margin only works if delivery is decomposed into repeatable, mostly automated steps using lightweight API usage and low-cost or free-tier deployment.

### Operating Principle

- Break each engagement into **4–10 discrete tasks**
- Automate each task independently — no monolithic agent flows
- Use the **website as the entry product**, not the final product; it is the foot-in-the-door for retainer upsells

### Sales Motion — Three-Design Close

Generate three site variants from three distinct workflows, then ask: **"Which one do you like best?"**

This shifts the conversation from *whether* to buy into *choosing among* concrete outputs. The website becomes a hook for ongoing retainer and upsell workflows.

### Identified Gaps

- Hormozi pricing logic and **five upsell workflows** are identified but not yet fully codified internally

## Intel-Driven Site Builder Pipeline

### Architecture

`build-site.sh` is the operational bridge between lead intelligence and delivered output.

- **~345 lines** of shell script
- **4–5 intel sources** run automatically at build time

### Intel Inputs (in order)

1. **GMB auto-extraction** via `gmb-extract.sh`
2. **Vault search** for industry-specific knowledge
3. [[sales-compass|Sales Compass]] lead data
4. Review data
5. Workflow references

If no data JSON is supplied, GMB extraction output is ingested automatically.

### Output Characteristics

| Artifact | Size / Value |
|---|---|
| React components generated | 10, populated with real business data |
| JS bundle | ~250 KB |
| CSS bundle | ~34 KB |
| Deploy target | Vercel, named `[client]-[industry]` |

Confirmed example deployment: `johnson-auto-collision`

### Operational Significance

The same intelligence used for prospecting and personalization drives the delivered website. This eliminates the handoff gap between sales and fulfillment.

### Known Gaps

- GMB-to-build-site bridge was previously disconnected; confirmed fixed **2026-04-15**
- [[2026-04-15-gmb-auto-scrape-location-matching-bug-2|GMB auto-scrape location matching bug]] still requires post-extraction validation

## OpenClaw Agent Infrastructure

### Timeout and Cron Tuning

| Setting | Old | New |
|---|---|---|
| `agents.defaults.timeoutSeconds` | 60–120s | 300s |
| Heartbeat frequency | 30 min | 60 min |
| Broken cron jobs | active (token burn) | disabled |

**Exec allowlist additions:** `launchctl`, `sqlite3`, `xurl`, `web_fetch`

**Heartbeat optimization:** Use a `HEARTBEAT.md` fast-path to skip heavy heartbeat steps on routine pings.

### CLI Responsiveness

- Set default model to `indigo/sonnet-4-6`
- Reduce heartbeat from 30 → 60 minutes

### Gateway Restart SOP

After any OpenClaw upgrade:

1. Manually run `openclaw gateway restart`
2. Reason: old PID may continue running with stale state
3. Symptom: `send` null bugs

Auto-upgrade behavior does **not** remove the need for manual restart.

### Identity Persistence Through Compaction

**Problem:** Compaction can reload the Claude Code system prompt instead of the intended OpenClaw startup sequence.

**Fix options (in order of preference):**
1. Add `CLAUDE.md` to the workspace mirroring startup instructions
2. Invoke with `--system-prompt`
3. Route through the OpenClaw frontend so `AGENTS.md` / `SOUL.md` are injected as project context

## Obsidian Knowledge Infrastructure

### Theme Strategy

- Use a community theme (**Minimal** or **AnuPpuccin**) as the structural base
- Layer aesthetic customization through **CSS snippets** — never replace structural theme behavior
- Keep glow, border, code block, and focus-style snippets modular and swappable

### Vault as Operational Memory

The vault functions as both long-term knowledge store and live input to the build pipeline. Intel sourced at prospecting time is the same intel used to generate delivered websites — no manual re-entry between systems.

## Counter-Arguments

- **Automation fragility**: 4–10 discrete automated tasks create 4–10 failure points; a single GMB extraction failure can break the entire build
- **Commoditization risk**: If competitors replicate the three-design close and similar pipelines, the pricing leverage erodes quickly
- **Retainer conversion is unproven**: The website-as-foot-in-the-door model assumes clients will upgrade; conversion rate is not yet documented

## Sources

- [[Wiki/Domains/sales-compass/ai-agency-operations-hormozi-pricing-intel-build-pipeline-op.md|ai-agency-operations-hormozi-pricing-intel-build-pipeline-op]]
- [[ai-agency-operations-hormozi-pricing-intel-build-pipeline-op|ai-agency-operations-hormozi-pricing-intel-build-pipeline-op]]

## Related

- [[Wiki/Domains/sales-compass/ai-agency-operations-hormozi-pricing-intel-build-pipeline-op|Ai Agency Operations Hormozi Pricing Intel Build Pipeline Op]]
- [[sales-compass]]
- [[openclaw]]
- [[sales-intel]]
- [[2026-04-15-gmb-auto-scrape-location-matching-bug-2|GMB auto-scrape location matching bug]]
