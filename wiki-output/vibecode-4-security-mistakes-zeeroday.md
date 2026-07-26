---
title: 4 Security Mistakes Vibe-Coded Startups Make (Zeeroday)
type: wiki-page
domain: security
status: active
created: 2026-06-26
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - type/security
  - island/security
  - vibe-coding
  - supabase
  - rls
  - authentication
  - rate-limiting
  - server-side-validation
  - ai-codegen
parent: "[[Security Island]]"
---

> **TLDR:** Four recurring security holes in AI-built startups: auth, rate limiting, RLS, and server-side validation.

## Summary

Zeeroday audited 20 startups using their pentest service over two weeks and found the same four vulnerabilities every time. AI codegen makes shipping trivial, meaning a single misconfig can leak an entire database. This is independent confirmation that the four-pillar checklist from the 2026-06-20 AI-built-site security deep-dive covers the right surface area — no re-architecture needed, just real coverage verification.

## The Four Pillars

### 1. Authentication
Verify users are who they claim to be. Email+password as baseline; Google/Apple SSO preferred for trust and fewer credential attack surfaces.

### 2. Rate Limiting
Cap requests per user. Blocks bots, scrapers, and brute-force attacks from hammering endpoints.

### 3. Row-Level Security (RLS)
Users read/write only their own rows — never another user's. Supabase RLS is the implementation target. Misconfiguration is the leaked-data nightmare scenario.

### 4. Server-Side Validation
Validate every field server-side against expected shape and type. Never trust client input.

## Relevance to Our Stack

All four map directly to LuxuryLane and MyPeptide exposure:

| Pillar | LuxuryLane | MyPeptide |
|---|---|---|
| Auth | Username+password CRM (no SSO — low urgency) | Phone-only gate |
| Rate limiting | In `ai-stack-security-audit` scope | In `ai-stack-security-audit` scope |
| RLS | Anon key + RLS — misconfig = full leak | Phone-gate enforces isolation |
| Server-side validation | In `ai-stack-security-audit` scope | In `ai-stack-security-audit` scope |

SSO upgrade is a maybe-later, not urgent. RLS and validation coverage are the live exposure.

## Counter-Arguments

- The four pillars are entry-level; mature stacks need additional layers (CSRF, secrets management, dependency scanning). This is a floor, not a ceiling.
- "20 startups" is a small, self-selected sample of Zeeroday's own clients — not a statistically representative audit.

## Sources

- [[Knowledge/AI/vibecode-4-security-mistakes-zeeroday.md|vibecode-4-security-mistakes-zeeroday]]
- [[vibecode-4-security-mistakes-zeeroday|vibecode-4-security-mistakes-zeeroday]]

## Related

- [[Wiki/Domains/sales-compass/4-security-mistakes-vibe-coded-startups-make|4 Security Mistakes Vibe Coded Startups Make]]
- [[Wiki/Domains/_shared/4-security-mistakes-vibe-coded-startups-make|4 Security Mistakes Vibe Coded Startups Make]]
- [[llm-pentest-tool-two-stage-eddy-carra]]
- [[six-layer-app-defense-patrick-minardi]]
