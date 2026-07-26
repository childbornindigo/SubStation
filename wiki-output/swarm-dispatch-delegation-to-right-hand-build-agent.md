---
title: Swarm Dispatch Delegation To Right Hand Build Agent
type: wiki-page
domain: _shared
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/agency
  - island/knowledge
  - type/wiki-page
  - swarm-dispatch
  - delegation
  - right-hand
  - build-agent
  - sop
parent: "[[Knowledge Island]]"
---

> **TLDR:** Swarm dispatch to a build agent works when the handoff is a fully scoped implementation contract, not just a goal.

## Summary

This page documents the delegation pattern for routing concrete implementation work through swarm dispatch to a right-hand or build agent. Handoffs must include exact artifacts, file paths, integrations, runtime cadence, trigger logic, and communication channels. Multi-agent execution becomes more reliable when orchestration and implementation are cleanly separated by a precise contract. The practical outcome is reduced ambiguity, fewer clarification cycles, and better reuse of shared infrastructure.

## Delegation Pattern

### Core Workflow

1. Frame the request as a delegated implementation task.
2. Route through swarm dispatch to the right-hand or build agent.
3. Provide exact technical and operational requirements before execution starts.
4. Reuse existing infrastructure wherever possible.
5. Treat the handoff as an SOP-aligned execution contract.

### Required Handoff Elements

A complete dispatch packet must specify:

| Element | Description |
|---|---|
| Target agent | Which agent or execution role receives the work |
| Artifact | Exactly what must be produced |
| File path | Where the output lives |
| Integrations | Required APIs or external systems |
| Cadence | Runtime polling interval or schedule |
| Trigger logic | Conditions, thresholds, or decision rules |
| Alerting | Notification channel and format |
| Reuse | Existing systems or infrastructure to leverage |

## Canonical Example

The reference case is a monitoring-script delegation with the following packet:

- **File path:** exact destination specified upfront
- **Integration:** Binance API
- **Cadence:** 5-minute polling loop
- **Trigger:** 3% price drop threshold
- **Alert channel:** Telegram
- **Reuse:** existing house-router

The lesson: swarm dispatch is most effective when the build agent receives a fully scoped operating context, not just a task description.

## Why This Pattern Works

### Reliability Benefits

- Narrows execution scope to reduce error surface
- Eliminates ambiguity at the handoff boundary
- Minimizes clarification cycles between orchestrator and builder
- Increases repeatability across similar task types

### Architectural Fit

This pattern is designed for multi-agent operating models where:

- **Swarm dispatch** handles routing and orchestration
- **Right-hand agent** handles implementation
- **SOPs** preserve handoff quality across runs
- **Shared infrastructure** acts as a reusable primitive

See [[Wiki/Domains/sales-compass/agent-operations-layer-swarm-intelligence-for-indigo|Agent Operations Layer Swarm Intelligence For Indigo]] for the broader orchestration context, and [[Wiki/Domains/_shared/handoff-session-right-hand-claude-code|Handoff Session Right Hand Claude Code]] for handoff mechanics.

## SOP Implications

### Delegation as Contract

The SOP is not a routing instruction — it is an implementation contract describing both *what* must be built and the *environment* in which it runs. Both halves are required.

### Success Condition

A delegation is sufficient when the receiving agent can execute with no follow-up questions. If major clarification is still needed after receipt, the packet was underspecified.

### Quality Checklist

A delegation is ready to dispatch when the build agent can answer all six:

1. What exactly am I building?
2. Where does it live?
3. What systems must it talk to?
4. How often does it run?
5. What event triggers action?
6. How is output communicated?

## When to Use

Apply this pattern when:

- The task is implementation-heavy and well-understood
- The orchestration role is explicitly separate from the builder role
- Existing systems should be reused rather than reinvented
- Precision delivers more value than exploratory autonomy

## Counter-Arguments

- High-specification delegation can reduce agent autonomy and constrain creative problem-solving in ambiguous tasks.
- A single recurring example (monitoring script) does not prove universal maturity of the pattern.
- Over-specification becomes brittle when file paths, integrations, or infrastructure change frequently — packets may need versioning.

## Sources

- [[Wiki/Domains/_shared/swarm-dispatch-delegation-to-right-hand-build-agent.md|swarm-dispatch-delegation-to-right-hand-build-agent]]
- [[swarm-dispatch-delegation-to-right-hand-build-agent|swarm-dispatch-delegation-to-right-hand-build-agent]]

## Related

- [[Wiki/Domains/sales-compass/agent-operations-layer-swarm-intelligence-for-indigo|Agent Operations Layer Swarm Intelligence For Indigo]]
- [[Wiki/Domains/_shared/right-hand-claude-code-integration-11-pm|Right Hand Claude Code Integration 11 Pm]]
- [[Wiki/Domains/_shared/synthesis-agent-reliability-right-hand-2026-05-15|Synthesis Agent Reliability Right Hand 2026 05 15]]
- [[Wiki/Domains/_shared/private-agent-network-test-build-plan|Private Agent Network Test Build Plan]]
- [[Wiki/Domains/_shared/handoff-session-right-hand-claude-code|Handoff Session Right Hand Claude Code]]
