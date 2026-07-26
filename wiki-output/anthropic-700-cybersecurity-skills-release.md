---
title: Anthropic 700+ Cybersecurity Skills Release
type: wiki-page
domain: AI/security
status: active
created: 2026-06-26
updated: 2026-07-25
confidence: low
retention: durable
tags:
  - type/reference
  - island/security
  - anthropic
  - cybersecurity
  - skills
  - threat-hunting
  - cloud-security
parent: "[[Security Island]]"
---

> **TLDR:** Anthropic reportedly shipped an official 700+ skill cybersecurity pack spanning cloud, threat-intel, and web-app domains — unverified, needs direct sourcing.

## Summary

An Instagram reel from a cybersecurity content creator (2026-06-22) claims Anthropic released an official cybersecurity skill set containing 700+ skills across cloud security, threat hunting, threat intelligence, web application security, and additional domains. The creator emphasizes the pack is grounded in established security frameworks rather than LLM-improvised procedure. The official Anthropic source has not been confirmed — the reel gates the link behind a DM funnel and should not be treated as authoritative until verified directly from Anthropic.

## Domains Named

- Cloud security
- Threat hunting
- Threat intelligence
- Web application security
- Additional domains (unspecified in reel)

## Relevance to Active Infrastructure

We run a live `righthand-security` worker with home-grown skills (`security-review`, `ai-stack-security-audit`). An official Anthropic pack with framework-backed playbooks for threat-intel, cloud, and web-app security is a potential direct upgrade — coverage we currently hand-roll.

Directly applicable stacks: LuxuryLane (Supabase + Vercel), MyPeptide (Cloudflare Pages + Supabase), Sales Compass (Supabase + Vercel/CF). These have documented exposure — see known leaked LuxuryLane `service_role` key (rotation deferred, queued in memory).

## Action Items

- [ ] Find the **official Anthropic source** for this skill pack (skill repo / docs page). Do not use the reel DM funnel. Confirm existence, license, and format.
- [ ] If real: evaluate folding web-app + cloud security domains into the `security` worker's skill set.
- [ ] Cross-check against 2026-06-20 ai-built-site security deep-dives to identify which existing gaps the official pack covers with framework-backed playbooks.
- [ ] Use threat-intel/cloud-sec skills as toolkit to drive the deferred LuxuryLane `service_role` key rotation to closure.

## Confidence Notes

**Low confidence** — single source is an Instagram reel from a third-party creator gating the link behind engagement bait. The claim is plausible given Anthropic's documented interest in security tooling, but the artifact is unverified. Do not act on the specific skill count or domain list until confirmed from Anthropic directly.

## Sources

- [[Knowledge/AI/anthropic-700-cybersecurity-skills-release.md|anthropic-700-cybersecurity-skills-release]]
- [[anthropic-700-cybersecurity-skills-release|anthropic-700-cybersecurity-skills-release]]
- [[2026-06-20-rip-coders-ai-built-sites-need-real-security-review]]
- [[2026-06-20-vibe-coded-api-cost-abuse-rate-limiting]]

## Related

- [[Wiki/Domains/_shared/anthropic-released-a-700-cybersecurity-skill-set|Anthropic Released A 700 Cybersecurity Skill Set]]
- [[Wiki/Domains/_shared/anthropic-cybersecurity-skills-library-700-skills-baroobi-pr|Anthropic Cybersecurity Skills Library 700 Skills Baroobi Pr]]
- [[Security Island]]
