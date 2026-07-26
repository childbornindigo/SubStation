---
title: How ChatGPT Actually Picks Sources (Read from Raw Network Traffic)
type: wiki-page
domain: knowledge-mgmt
status: active
created: 2026-06-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - island/knowledge
  - island/agency
  - type/technique
  - ai
  - geo
  - ai-seo
  - chatgpt
  - source-selection
  - content-strategy
  - web-fetching
  - reverse-engineering
parent: "[[Knowledge Island]]"
---

> **TLDR:** ChatGPT routes fetches through four labelled pipelines — server-render your facts or a competitor gets cited instead.

## Summary

Suganthan Mohanadasan (20+ yr SEO, co-founder Keyword Insights & Snippet Digital) reverse-engineered ChatGPT's raw browser network traffic across ~1,240 source records, surfacing internal field values never exposed to readers. The findings reveal four distinct fetch pipelines, a query-classification bucket that bypasses search entirely, a large gap between fetched and cited sources, and a JS-wall fallback that hands attribution to third parties. Structural findings are firm; percentage figures are directional only given a small, tech-skewed sample.

---

## The Four Fetch Pipelines (`result_source`)

Every result ChatGPT pulls carries a hidden `result_source` stamp:

| Value | What it is | Notes |
|-------|------------|-------|
| `serp` | Open-web baseline | Mostly news (Yahoo, StreetInsider) |
| `labrador` | Allowlisted established publishers | Reuters, Guardian, WSJ, FT, Wikipedia, arXiv. Snippets ~1,080 chars. Likely a **licensed tier** |
| `bright` | Bright Data (commercial scraper) | **Dominant pipeline** — shopping, finance, weather, local |
| `oxylabs` | Oxylabs (rival commercial scraper) | Regional/local press, some open web |

A single weather query demonstrated split routing: Bright Data fetched global met sites, Oxylabs fetched local Gulf press simultaneously.

---

## `turn_use_case` — Some Queries Never Touch the Web

ChatGPT classifies each question before searching. Six observed buckets: `instant search`, `shopping`, `text`, `local`, `thinking`, `image gen`.

**Critical bucket: `text`** — no web search; answers from training corpus only.

- How-tos, code, translations → `text` (expected)
- "Latest treatment guidelines for type 2 diabetes" → also `text` (alarming — answered from training, no recency check)
- 3 of 10 deliberately time-sensitive questions received **no search at all**

**Wording controls the bucket, not the topic:**

| Phrasing | Bucket |
|----------|--------|
| "best 4K TVs to buy" | `shopping` |
| "best 4K TVs with reviews" | normal search |
| "best coffee near me" | `local` |
| maths / logic question | `thinking` → reasoning model |

---

## Fan-Out (Thinking Model)

- Fast model: ~1 reworded query per question
- Thinking model on a comparison task: **15–40 sub-queries**
- Fires `site:vendor.com/pricing` probes, guesses a price then searches to confirm it
- Page-reading is literal: uses browsing tool's `find` command to scan for `$`, `€`, `99`, "Agency" — server-side, not visible on screen

---

## Fetched ≠ Cited ≠ Mentioned

Three distinct outcomes with very different reader visibility:

| Outcome | Meaning | Reader-visible? |
|---------|---------|----------------|
| **Fetched** | Pulled into context via `result_source` | No |
| **Cited** | Clickable footnote attached to a specific sentence | Yes |
| **Mentioned** | Brand name appears as chip/link but not the claim source | Yes (weakly) |

**Empirical gap from the dataset:**
- Reddit: fetched 278×, cited 11×
- YouTube: fetched 201×, **cited 0×** — YT URL fetches return metadata not transcript; Reddit threads are full-text

Citations bind to a *specific sentence*, not topical relevance. Results also **dedupe by domain** — 20 thin pages from one domain collapse to a single result.

---

## The JS-Wall Fallback (Highest-Leverage GEO Finding)

When official pages are JavaScript-rendered and won't parse, ChatGPT explicitly reasoned:

> "I can quote third-party sources since the official page is hard to parse"

**Result: cited G2 instead of the vendor's own pricing page.**

Your facts, your competitor's attribution. The fix is unambiguous: **server-render or statically export any page containing facts you want attributed to you.**

---

## What Remains Opaque

- Domain authority / trust weights → no visibility; "anyone selling 'ChatGPT ranking factors' is selling snake-oil"
- Personalization: `personal_sources: ["convo_search","gmail","files"]` pulled into ~1/3 of answers
- Local result injection mechanics (field observed but internal weighting unknown)

---

## Actionable Implications

1. **Server-render fact pages** — pricing, specs, any claim you want cited must be in raw HTML, not JS-rendered
2. **Phrase time-sensitive queries with recency signals** — the `text` bucket silently returns stale training data with no warning
3. **Reddit presence ≠ citation** — fetched heavily but cited rarely; optimize for direct-source clarity instead
4. **`labrador` tier is not accessible by application** — focus effort on `serp`/`bright`/`oxylabs` optimization
5. **Third-party review sites (G2, etc.) are your citation competitors** for JS-heavy product pages

---

## Counter-Arguments

- Sample (~1,240 records) is small and tech-skewed; pipeline weights may differ in other verticals
- OpenAI can change internal routing at any time without notice — these findings are a snapshot, not a stable spec
- Percentage figures are directional only; replication with a larger corpus would be needed to confirm ratios

---

## Sources

- [[Wiki/Domains/_shared/how-chatgpt-actually-picks-sources-read-from-raw-network-tra.md|how-chatgpt-actually-picks-sources-read-from-raw-network-tra]]
- [[how-chatgpt-actually-picks-sources-read-from-raw-network-tra|how-chatgpt-actually-picks-sources-read-from-raw-network-tra]]

---

## Related

- [[Wiki/Domains/_shared/chatgpt-source-selection-network-traffic-suganthan|Chatgpt Source Selection Network Traffic Suganthan]]
