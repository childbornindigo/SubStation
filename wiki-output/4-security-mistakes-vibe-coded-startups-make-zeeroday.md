---
title: 4 Security Mistakes Vibe-Coded Startups Make (Zeeroday)
type: wiki-page
domain: business
status: active
created: 2026-06-26
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/security
  - island/agency
  - type/security
  - vibe-coding
  - supabase
  - rls
  - authentication
  - rate-limiting
  - server-side-validation
  - ai-codegen
parent: "[[Security Island]]"
---

> **TLDR:** Four security holes repeat across every AI-built startup: auth gaps, no rate limiting, misconfigured RLS, and missing server-side validation.

## Summary

Zeeroday's pentest service audited 20 startups and found the same four vulnerabilities in every codebase. AI codegen compresses shipping time to near-zero, which means a single misconfig can expose an entire database before the founder notices. This is independent field confirmation that the four-pillar checklist from the 2026-06-20 AI-built-site security deep-dive covers the right surface area — no architectural overhaul required, only real coverage verification on existing code.

## The Four Pillars

### 1. Authentication
Verify users are who they claim to be. Email+password is the minimum baseline; Google/Apple SSO is preferred for trust and fewer credential attack surfaces.

### 2. Rate Limiting
Cap requests per user per time window. Blocks bots, scrapers, and brute-force attacks from hammering endpoints at zero marginal cost to the attacker.

### 3. Row-Level Security (RLS)
Users must read and write only their own rows — never another user's. Supabase RLS is the primary implementation target. Misconfiguration is the direct path to full-database exposure via the anon key.

### 4. Server-Side Validation
Every field must be validated server-side against expected shape and type. Client-side validation is a UX nicety, not a security control — never trust client input.

## Relevance to Our Stack

All four map directly to LuxuryLane and MyPeptide exposure:

| Pillar | LuxuryLane | MyPeptide |
|---|---|---|
| Auth | Username+password CRM (no SSO — low urgency) | Phone-only gate |
| Rate limiting | In `ai-stack-security-audit` scope | In `ai-stack-security-audit` scope |
| RLS | Anon key + RLS — misconfig = full leak | Phone-gate enforces isolation |
| Server-side validation | In `ai-stack-security-audit` scope | In `ai-stack-security-audit` scope |

SSO upgrade is a maybe-later, not urgent. RLS policy correctness and server-side validation coverage are the live exposure to close first.

## Counter-Arguments

- The four pillars are entry-level; mature stacks require additional layers — CSRF protection, secrets management, dependency scanning, egress filtering. This checklist is a floor, not a ceiling.
- "20 startups" is a small, self-selected sample of Zeeroday's own pentest clients, not a statistically representative survey of AI-built products at large.

## Sources

- [[Wiki/Domains/_shared/vibecode-4-security-mistakes-zeeroday.md|4 Security Mistakes Vibe-Coded Startups Make (Zeeroday)]]

## Related

- [[Wiki/Domains/sales-compass/4-security-mistakes-vibe-coded-startups-make|4 Security Mistakes Vibe Coded Startups Make]]
- [[Wiki/Domains/_shared/vibecode-4-security-mistakes-zeeroday|Vibecode 4 Security Mistakes Zeeroday]]
- [[Wiki/Domains/_shared/4-security-mistakes-vibe-coded-startups-make|4 Security Mistakes Vibe Coded Startups Make]]
- [[Wiki/Domains/_shared/six-layer-app-defense-stack-for-vibe-coded-apps|Six Layer App Defense Stack For Vibe Coded Apps]]
- [[Wiki/Domains/_shared/vibe-coder-app-security-pt2-deserialization-access-auth|Vibe Coder App Security Pt2 Deserialization Access Auth]]
