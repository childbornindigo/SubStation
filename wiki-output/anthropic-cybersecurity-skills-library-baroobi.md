---
title: Anthropic Cybersecurity Skills Library (Baroobi Promo)
type: wiki-page
domain: ai-engineering/security
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - type/intelligence
  - island/security
  - ai-agents
  - cybersecurity
  - skills-library
  - mitre-attack
  - nist-csf
  - claude-code
parent: "[[Security Island]]"
---

> **TLDR:** Third-party ~700-skill security pack for AI agents — open-source claimed, but provenance unconfirmed; verify before installing.

## Summary

A Baroobi (@baroobi.inc) Instagram reel promotes what it calls the largest open-source cybersecurity skills library built for AI agents, citing ~700–754 structured skills mapped to MITRE ATT&CK, NIST CSF 2.0, MITRE ATLAS, D3FEND, and NIST AI RMF across 26 domains. The pitch is that AI coding agents write security code with zero security knowledge — this pack would close that gap. The actual repo link is gated behind a comment-SECURITY DM lead-magnet, so provenance is unverified and may be paywalled despite the "open-source" framing.

## Claims

| Attribute | Detail |
|---|---|
| Skill count | ~700–754 structured skills |
| Frameworks mapped | MITRE ATT&CK, NIST CSF 2.0, MITRE ATLAS, D3FEND, NIST AI RMF |
| Domains | 26 — cloud security, threat hunting, threat intel, web-app vulns, IR, red team, DevSecOps, and more |
| Compatible platforms | Claude Code, Cursor, Copilot, and 20+ others |
| Source | Instagram reel — no direct repo link in post |

## Relevance to Active Projects

The Hermes harness already uses Claude Code's native Skill system for the `security` worker lane. A framework-mapped skills pack slots into that same mechanism and could harden the pre-deploy gate used on LuxuryLane, MyPeptide, and [[sales-compass|Sales Compass]]. Specifically relevant to RLS/service-key checks and LLM pentest tooling already in use.

## Steal / Action Items

- **If real + open-source:** evaluate importing a subset — web-app vulns, cloud/Supabase checks, secret-leak detection — as Claude Code skills for the security lane.
- **Before any install:** find via GitHub search (`Claude Code cybersecurity skills MITRE ATT&CK NIST CSF D3FEND`) rather than engaging the post. Treat DM lead-magnet links with suspicion.
- **Meta-learning:** framework mapping (ATT&CK/NIST/D3FEND) is the credibility signal for any security skill pack — if packaging internal security checks as skills, map them to known catalogs.

## ⚠️ Unconfirmed

- Audio in reel says "Anthropic finally did it" but this is a **third-party** skills pack, not an official Anthropic release.
- "Open-source" claim is unverified — the repo is gated behind the DM funnel. May be paywalled or affiliate-linked.
- Do not install or trust agent output from these skills without confirming provenance and reviewing the actual skill definitions.

## Counter-Arguments

- The ISC2 4.8M unfilled roles stat is a common marketing hook; does not validate the library's quality.
- Framework mapping to ATT&CK/NIST is a credibility signal but can be superficial — individual skill accuracy still requires manual review.
- Lead-magnet gating on an "open-source" library is a red flag; legitimate open-source projects link directly to the repo.

## Sources

- [[Knowledge/AI/anthropic-cybersecurity-skills-library-baroobi.md|anthropic-cybersecurity-skills-library-baroobi]]
- [[anthropic-cybersecurity-skills-library-baroobi|anthropic-cybersecurity-skills-library-baroobi]]

## Related

- [[Wiki/Domains/_shared/anthropic-released-a-700-cybersecurity-skill-set|Anthropic Released A 700 Cybersecurity Skill Set]]
- [[Wiki/Domains/_shared/anthropic-cybersecurity-skills-library-700-skills-baroobi-pr|Anthropic Cybersecurity Skills Library 700 Skills Baroobi Pr]]
- [[Wiki/Domains/_shared/anthropic-700-cybersecurity-skills-release|Anthropic 700 Cybersecurity Skills Release]]
