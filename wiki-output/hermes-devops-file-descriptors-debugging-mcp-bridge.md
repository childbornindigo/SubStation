---
title: Hermes Devops File Descriptors Debugging Mcp Bridge
type: wiki-page
domain: ai-agents
status: active
created: 2026-06-24
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/infrastructure
  - island/ai-agent
  - type/wiki-page
  - hermes
  - devops
  - debugging
  - file-descriptors
  - mcp-bridge
  - sqlite
  - resource-management
  - timeouts
  - self-healing
parent: [[Wiki/Domains/sales-compass/hermes-devops-infrastructure-map-of-content|Hermes Devops Infrastructure Map Of Content]]
---

> **TLDR:** Hermes failures came from file descriptor leaks, duplicate bridge tasks, and upstream provider timeouts firing first.

## Summary

This page documents a Hermes debugging pattern where the process remained running but stopped completing useful work. The primary causes were file descriptor exhaustion from unsafe long-lived I/O, duplicate MCP bridge task writes compounding the noise, and provider timeouts that fired before Hermes-level timeout controls engaged. The operational lesson: agent reliability requires strict resource cleanup, queue deduplication, layered observability, and cleanup-aware self-healing — not just retry logic.

## Failure Pattern

Hermes entered an "alive but broken" state with these signals:

- Process liveness with no successful work completion
- Repeated cycle failures with accumulating pending MCP bridge tasks
- Recurring symptoms that obscured root cause isolation
- Key runtime error: `[Errno 24] Too many open files`

The `Too many open files` error is the canonical indicator of file descriptor exhaustion in a long-running process — usually from unclosed database connections, file handles, or compressed stream readers.

## Root Causes

### 1. File Descriptor Leaks

Primary failure source — unsafe resource handling in runtime code:

- `sqlite3.connect()` not consistently wrapped in context managers
- `file` and `gzip` handles opened without exception-safe closure
- Exception paths could bypass cleanup blocks
- Small leaks accumulated across many cycles until the OS descriptor limit was hit

Classic long-running service failure mode: localized cleanup mistakes become system-wide outages over time.

### 2. Missing MCP Bridge Deduplication

Compounding issue — 43 duplicate pending bridge tasks were observed:

- Equivalent work was written multiple times without dedup guards
- Queue growth increased noise during diagnosis and recovery
- Did not directly cause descriptor exhaustion, but masked the primary fault and slowed isolation

### 3. Provider Timeout Before Agent Timeout

Some failures originated at the upstream model provider:

- Prompts were too heavy for provider startup budgets
- Provider timeouts fired before Hermes agent-level timeout logic engaged
- Failures could be misread as controller instability rather than prompt weight

**The tightest timeout in the stack determines practical behavior.**

## Fixes Applied

### Resource Hygiene

Use context-managed I/O everywhere in long-lived agents:

```python
# Database
with sqlite3.connect(path) as conn:
    ...

# Compressed files
with gzip.open(file_path, "rt") as f:
    ...
```

Benefits: resources close on success and on exceptions; cleanup correctness becomes local and explicit; cumulative leak risk drops substantially.

### Bridge Cleanup and Dedup Guards

- Cleared 43 stale duplicate bridge tasks (one-time operational cleanup)
- Added dedup checks before bridge writes (systemic prevention)
- Prevents repeated enqueue of equivalent pending work

### Timeout Tuning

- Reduced prompt size and startup burden
- Added log distinction between provider timeout events and Hermes timeout events
- Eliminated assumption that agent-level timeout is always the first bottleneck

## Operational Guidance

### "Alive but Broken" Is a Distinct Health State

Process liveness ≠ service health. Health checks should verify:

- Successful cycle completion rate
- File descriptor count trends
- Pending queue size growth
- Repeated identical error signatures
- Provider timeout frequency

### Exception-Safe I/O Is a Reliability Requirement

In agent systems, Python resource management is not cosmetic style. Any repeated execution loop must assume exceptions will happen and enforce automatic cleanup via context managers.

### Preserve Layer Boundaries in Observability

Hermes failures span multiple layers — logs, alerts, and dashboards should keep them distinct:

| Layer | Signals |
|---|---|
| Local process / resource | `[Errno 24]`, fd count, memory |
| Bridge / queue | Pending task count, duplicate writes |
| Provider / model | Timeout type, prompt size, startup latency |

### Self-Healing Must Include Cleanup

Recovery logic should go beyond retries:

- Stale task purge before re-enqueue
- Descriptor count check before cycle restart
- Queue dedup validation as part of startup health check

## Counter-Arguments

- **"Context managers are verbose overhead"** — In short scripts, yes. In long-lived agent loops with thousands of cycles, unclosed handles compound into outages; the overhead is justified.
- **"Provider timeouts are outside our control"** — Prompt weight and startup complexity are controllable inputs that directly affect whether the provider timeout fires first.

## Sources

- [[Wiki/Domains/_shared/hermes-devops-file-descriptors-debugging-mcp-bridge.md|hermes-devops-file-descriptors-debugging-mcp-bridge]]
- [[hermes-devops-file-descriptors-debugging-mcp-bridge|hermes-devops-file-descriptors-debugging-mcp-bridge]]

## Related

- [[Wiki/Domains/sales-compass/hermes-devops-infrastructure-map-of-content|Hermes Devops Infrastructure Map Of Content]]
- [[Wiki/Domains/_shared/synthesis-hermes-devops-file-descriptors-debugging-mcp-bridg|Synthesis Hermes Devops File Descriptors Debugging Mcp Bridge]]
