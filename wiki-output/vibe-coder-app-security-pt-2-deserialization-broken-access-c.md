---
title: "Vibe Coder App Security Pt. 2 — Deserialization, Broken Access Control & Broken Auth"
type: wiki-page
domain: business
status: active
created: 2026-07-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - security
  - owasp
  - vibe-coding
  - web-security
  - supabase
  - auth
  - idor
  - deserialization
  - island/security
  - island/agency
  - type/reference
parent: "[[Security Island]]"
---

> **TLDR:** OWASP vulns 3–5 — deserialization, IDOR, broken auth — directly threaten AI-built Supabase stacks; fix RLS, rate-limit auth, never deserialize untrusted data.

## Summary

Part 2 of CT Web Solutions' OWASP-for-vibe-coders series covers insecure deserialization, broken access control (IDOR), and broken authentication. All three map directly onto fast AI-built stacks using Supabase anon keys. The leaked LuxuryLane `service_role` JWT (2026-05-15 incident) is a textbook broken access control failure — it bypasses RLS entirely regardless of policy correctness. These are not theoretical risks; IDOR bots scan sequential IDs at scale and credential stuffing runs continuously against any unprotected auth endpoint.

---

## The Three Vulnerabilities

### 1. Insecure Deserialization

Attackers poison serialized blobs so the object returned on read differs from what was stored — in ~9/10 cases the returned object is completely different. Severity: can silently install backdoors.

**Mitigations:**
- Never deserialize untrusted input.
- Sign and validate serialized blobs before reading.
- Prefer safe formats: **JSON over pickle / native object serialization**.
- Validate schema on every deserialization.

---

### 2. Broken Access Control (IDOR)

Users reach app resources without server-side authorization checks. Classic vector: increment a URL parameter (`/order/123` → `/order/124`) to access another user's data.

**Mitigations:**
- Enforce authz on **every** request server-side — never trust client-supplied IDs.
- Supabase: RLS policies must scope rows to the **authenticated user**, not `anon` + a client-side filter the attacker can bypass.
- A compromised `service_role` key bypasses RLS entirely — rotate immediately if leaked.

---

### 3. Broken Authentication

No password rate-limiting, allows common passwords, no session ID rotation on login, no MFA, no account lockout → credential stuffing and brute-force succeed via infinite retries.

**Mitigations:**
- Rate-limit auth endpoints.
- Block common/breached passwords.
- Rotate session token on every successful auth.
- Enforce MFA.
- Lock accounts after N failed attempts.

---

## Stack-Specific Exposure

| Surface | Vulnerability | Status |
|---|---|---|
| LuxuryLane / MyPeptide / GC Reno Supabase | Broken Access Control — RLS scoping unverified | Needs audit |
| LuxuryLane `service_role` JWT (leaked 2026-05-15) | Broken Access Control — RLS bypass via leaked key | Rotation queued |
| LuxuryLane CRM (`LuxLane2026!` shared login, no lockout/MFA) | Broken Authentication — textbook exposure | Unmitigated |
| Python pipelines (potential pickle usage) | Insecure Deserialization | Needs grep audit |

---

## Open Action Items

- [ ] Audit Supabase RLS on LuxuryLane + MyPeptide: confirm row scoping is server-enforced, not client-filtered.
- [ ] Confirm LuxuryLane `service_role` JWT rotation is complete.
- [ ] Add rate-limiting + lockout to LuxuryLane CRM shared login, or migrate off shared credentials.
- [ ] Grep all Python pipeline code for `pickle.loads` / `pickle.load` on untrusted input.

---

## Counter-Arguments

- **"Supabase handles auth so we're fine"** — Supabase auth covers identity, not authorization. RLS must be explicitly written and tested per-table; misconfiguration (e.g. permissive anon policies) exposes rows.
- **"Our apps are small targets"** — IDOR is automated via bots scanning sequential IDs at scale. Small apps are equally vulnerable; obscurity is not access control.
- **"We don't use pickle anywhere"** — Verify with a grep; AI-generated Python code often reaches for `pickle` for caching, and third-party libraries may also do object serialization.

---

## Sources

- [[Wiki/Domains/_shared/vibe-coder-app-security-pt2-deserialization-access-auth.md|Vibe Coder App Security Pt.2 — Deserialization, Broken Access Control & Broken Auth]]

---

## Related

- [[Wiki/Domains/sales-compass/4-security-mistakes-vibe-coded-startups-make|4 Security Mistakes Vibe Coded Startups Make]]
- [[Wiki/Domains/_shared/4-security-mistakes-vibe-coded-startups-make|4 Security Mistakes Vibe Coded Startups Make]]
- [[Wiki/Domains/_shared/cyber-expert-reacts-to-the-4-security-checks-for-vibe-coded|Cyber Expert Reacts To The 4 Security Checks For Vibe Coded]]
- [[Wiki/Domains/_shared/vibe-coder-app-security-pt2-deserialization-access-auth|Vibe Coder App Security Pt2 Deserialization Access Auth]]
