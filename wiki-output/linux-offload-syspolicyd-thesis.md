---
title: Linux Offload — the syspolicyd Tax on Multi-Agent Mac mini Orchestration
type: wiki-page
domain: infrastructure
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/infrastructure
  - island/ai-agent
  - type/reference
  - infrastructure
  - homelab
  - orchestration
parent: "[[AI Agent Island]]"
---

> **TLDR:** macOS `syspolicyd` taxes every spawned child process; offload heavy agent fan-out to Linux to eliminate it.

## Summary

macOS enforces code-signing and Gatekeeper checks on every spawned process via `syspolicyd`, and multi-agent orchestration is its worst-case workload — each worker fanning out into sub-agents triggers cascading OS inspection that can consume >50% of available CPU. The fix is to SSH-dispatch heavy agent loops to a Linux box (where no equivalent daemon exists), demoting the Mac mini to thin orchestrator only. This is a **parked future line item** gated on owning a Linux homelab machine. Before purchasing hardware, the cheap first step is to measure the actual `syspolicyd` CPU% during a heavy fan-out day.

## The Core Problem

macOS inspects **every** spawned process via `syspolicyd` — the security policy daemon running code-signing, notarization, and Gatekeeper checks on each child. Multi-agent workloads are the worst case:

- Persistent workers (`lux`, `mypeptide`, `collective`, `flex`) running all day on one Mac mini.
- Each worker fans out into sub-agents → every child process triggers a cascade of `syspolicyd` checks.
- Theo (t3.gg) reported **>50% of CPU** lost to the OS "checking if the other half of the machine is behaving" under heavy agent fan-out.
- On Linux there is **no equivalent tax** — his 32 threads sat near-idle doing the same work.

A version of this was observed on 2026-06-25: a heavy fan-out day where workers kept parking and needing re-dispatch. Some stall is plausibly machine tax from process-policing (unproven — see Measure First below).

## The Thesis / Fix

**FUTURE — gated on owning a Linux homelab. Do not attempt before then.**

- SSH-dispatch heavy agent loops to a Linux box. The loops run where `syspolicyd` doesn't exist.
- The Mac mini becomes a **thin orchestrator** — dispatch and routing only, with no heavy local fan-out.
- A Mac mini cannot serve as the Linux host; this genuinely waits on new hardware.

## Measure First (Cheap Step Before Buying)

Before any hardware spend: on the next heavy fan-out day, open `btop` or Activity Monitor mid-fan-out and **record `syspolicyd` CPU%**. Goal is not to fix it — it's to get a number. "Here is the exact CPU bled to process-policing" is the metric that justifies the homelab purchase.

## Related Context

- **Local LLM stack selection** ([[local-llm-bible-stack-theahmadosman]]) maps hardware tiers to inference engines (llama.cpp / MLX-LM / vLLM / SGLang / TensorRT-LLM / Dynamo). vLLM and SGLang are CUDA-only, reinforcing the conclusion that the target box must be Linux/NVIDIA, not another Mac.
- **Sustained fan-out workload example** ([[codex-goal-loop-automation-tomosman]]) — hundreds of user stories processed in one agent loop; exactly the workload class this thesis wants to offload off the Mac mini.
- Everything else from the source video (fan-out discipline, killing anti-metric gates, price-per-task routing, spec-first self-config) is actionable today on the minis and being tackled separately.

## Sources

- [[Knowledge/Infrastructure/linux-offload-syspolicyd-thesis.md|linux-offload-syspolicyd-thesis]]
- [[linux-offload-syspolicyd-thesis|linux-offload-syspolicyd-thesis]]
- [[nerd-snipe-fable-banned-ai-regulation-theo-ben]]

## Related

- [[Wiki/Domains/_shared/linux-offload-the-syspolicyd-tax-on-multi-agent-mac-mini-orc|Linux Offload The Syspolicyd Tax On Multi Agent Mac Mini Orc]]
- [[local-llm-bible-stack-theahmadosman]]
- [[codex-goal-loop-automation-tomosman]]
