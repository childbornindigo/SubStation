---
title: Safeharbor Dark CTA Lead Capture Card
type: wiki-page
domain: _shared
status: active
created: 2026-06-13
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/knowledge
  - type/reference
  - ui/component
  - lead-capture
  - sales-compass
parent: "[[Wiki/Domains/_shared/safeharbor-lead-capture|Safeharbor Lead Capture]]"
---

> **TLDR:** A dark-themed in-page card component for lead capture that converts without interrupting trust-building flow.

## Summary

The Safeharbor Dark CTA Lead Capture Card is a reusable UI component — a visually prominent, dark-background card with a high-contrast call-to-action button — designed to capture prospect information at the right moment in a sales or landing flow. It sits in-page (never modal), placed after trust-establishing content so the user has already bought in before being asked to act. Fields are kept minimal and purposeful; the CTA language is direct and benefit-forward.

## Field Rationale

| Field | Rationale |
|-------|-----------|
| First Name | Personalizes follow-up; low friction — one word |
| Email | Primary contact channel; required for automation handoff |
| Phone (optional) | Enables SMS/iMessage cadence; optional keeps drop-off low |

**What to Avoid:**
- `<textarea>` / open message fields — high friction, kills conversion
- Asking for company/title on first touch — premature qualification
- More than 3 fields total at the top-of-funnel stage

## CTA Language

| Label | Verdict | Notes |
|-------|---------|-------|
| "Submit" | Avoid | Generic, no value signal |
| "Get Started" | Acceptable | Neutral but overused |
| "Send Me the Info" | Preferred | Prospect-POV, low-commitment framing |
| "Book My Spot" | Preferred | Urgency + ownership, use when scarcity is real |
| "Claim Your Free X" | Preferred | Benefit-first; strong for lead magnets |

## Placement Rules

1. **After trust content** — Place the card only after social proof, credentials, or a value explanation. Asking before trust = high abandonment.
2. **In-page, not modal** — Modals interrupt flow and trigger close-reflex. The card must live inline so the prospect controls pacing.
3. **No textarea** — Open-ended input fields signal high commitment and are the single largest conversion killer at this stage.

## Counter-Arguments

**"Dark cards feel aggressive / hard-sell"**
Contrast is a UX tool, not a personality statement. Dark backgrounds on CTAs are a proven pattern (Stripe, Linear, Notion all use them) because they focus attention without changing tone. The copy and field count do the tone work, not the color.

**"In-page forms get ignored — modals get seen"**
Modals have higher *impression* rates but lower *completion* rates due to dismiss-reflex. In-page cards seen by a user who chose to scroll are self-qualified; completion rates consistently outperform forced modal views in B2B flows.

**"More fields = more qualified leads"**
Gate qualification to the second touchpoint (call/email). Top-of-funnel over-qualification shrinks the pool before you've built relationship; let the setter conversation do the sorting.

## Sources

- [[Wiki/Domains/sales-compass/safeharbor-dark-cta-lead-capture-card.md|safeharbor-dark-cta-lead-capture-card]]
- [[safeharbor-dark-cta-lead-capture-card|safeharbor-dark-cta-lead-capture-card]]

## Related

- [[Wiki/Domains/_shared/safeharbor-lead-capture|Safeharbor Lead Capture]]
- [[Wiki/Domains/_shared/safeharbor-dark-cta-lead-capture-patterns-lead-capture-card|Safeharbor Dark Cta Lead Capture Patterns Lead Capture Card]]
- [[Wiki/Domains/sales-compass/safeharbor-dark-cta-lead-capture-card|Safeharbor Dark Cta Lead Capture Card (sales-compass)]]
