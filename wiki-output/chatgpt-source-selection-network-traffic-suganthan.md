---
title: How ChatGPT Actually Picks Sources (Network Traffic Analysis — Suganthan)
type: wiki-page
domain: AI
status: active
created: 2026-06-25
updated: 2026-07-25
confidence: medium
retention: durable
tags:
  - type/reference
  - type/research-finding
  - island/ai
  - geo
  - chatgpt
  - llm-search
  - seo
  - network-traffic
parent: "[[Knowledge Island]]"
---

> **TLDR:** ChatGPT routes every web fetch through one of 4 labelled pipelines — and some queries never search at all.

## Summary

Suganthan Mohanadasan (20+ yr SEO, co-founder Keyword Insights & Snippet Digital) inspected ChatGPT's raw browser network traffic via DevTools across ~1,240 source records and lifted out internal field labels never shown to users. The load-bearing find is a `result_source` field that stamps every fetched page with its pipeline: a licensed publisher tier (`labrador`), two commercial scrapers (`bright`, `oxylabs`), and an open-web baseline (`serp`). Separately, a `turn_use_case` field decides whether ChatGPT searches at all — queries bucketed as `text` are answered from training with no web fetch, no matter how current the topic sounds. Structural field names are firm (one observation is enough); percentage figures are directional only from a small, SaaS/tech-skewed sample.

## The `result_source` Field — 4 Fetch Pipelines

| Value | What It Is |
|-------|-----------|
| `serp` | Open-web baseline; mostly news (Yahoo, StreetInsider) |
| `labrador` | Allowlist of established publishers — Reuters, Guardian, WSJ, FT, Wikipedia, arXiv. Snippets ~1,080 chars (near full-article extracts). Looks like a licensed tier; several sources have OpenAI content deals. Closed unless you own a national newspaper. |
| `bright` | Bright Data (commercial scraper). Dominant for shopping, finance, weather, local. |
| `oxylabs` | Oxylabs (rival scraper). Regional/local press, some open web. |

Most fetching ran through `bright`. One weather query split cleanly: Bright Data fetched global sites (Met Office, AccuWeather), Oxylabs fetched local Gulf press.

## `turn_use_case` — Queries That Never Touch the Web

ChatGPT buckets each query before searching. The 6 observed values: `instant search`, `shopping`, `text`, `local`, `thinking`, `image gen`.

- **`text` = no search.** Answers from training corpus only.
- How-tos, code, translations → `text` (expected).
- **Alarming:** "latest treatment guidelines for type 2 diabetes" → `text`. 3 of 10 deliberately current questions got no search at all.
- **Wording decides the bucket, not the topic.** "best 4K TVs to buy" → `shopping`; "best 4K TVs with reviews" → normal search. "best coffee near me" → `local`. A maths question → `thinking` (reasoning model).

## Fan-Out (Thinking Model)

- Fast model: ~1 reworded sub-query.
- Thinking model on a compare task: **15–40 sub-queries**. Fires `site:vendor.com/pricing` probes, guesses a price then searches to confirm, and keeps widening scope.
- Page-reading is literal: model runs `find` for `$`, `€`, `99`, "Agency" using server-side browsing tool open/click commands — not a visible agent on the user's screen.

## Fetched ≠ Cited ≠ Mentioned

Three separate outcomes exist for every source:

| Outcome | Definition |
|---------|-----------|
| **Fetched** | Pulled into context via `result_source`. Never shown to reader. |
| **Cited** | Clickable footnote behind a specific sentence. |
| **Mentioned** | Brand name appears as chip/link, but is NOT the source of the claim. |

- Reddit: fetched 278×, cited 11×.
- YouTube: fetched 201×, **cited 0×** — fetching a YT search result returns metadata, not transcript. A Reddit thread is fully readable on the page.
- Ahrefs corroborates (1.4M prompts): Reddit cited 1.93% vs YouTube 0.51%.
- Citations bind to a **specific sentence**, not a topic. Topical relevance alone is not enough — you must be the best support for a precise claim.
- **Domain dedup**: 20 thin pages from the same domain collapse to 1 result.

## The JavaScript Trap (Killer Trace)

ChatGPT goes to the official page first for facts/pricing and narrates this reasoning. But on Profound and Peec, it concluded pricing "isn't showing up... possibly loaded with JavaScript," gave up, and stated: *"I can quote third-party sources since the official page is hard to parse"* → **cited G2 instead.** Your own facts, attributed to a competitor's review page, because your pricing was JS-rendered.

## What He Couldn't See

- No visible ranking/trust logic (domain authority weights are server-side — "anyone selling you 'ChatGPT's ranking factors' is selling snake-oil").
- Personalization is real: `personal_sources: ["convo_search","gmail","files"]` pulled into ~1/3 of answers.
- `local_results_limit: 2` — if you're not in the top 2 local results, you don't appear.
- Also observed: a bot-wall blocking scripting, a hidden shopping engine, and **573 live experiments** running on the logged-in account.

## GEO Actionables

1. **Put all facts and numbers in plain crawlable HTML.** Never behind JavaScript, a PDF, or an image. ChatGPT greps for `$`/`€` and gives up on JS pricing tables — handing your numbers to a third-party review site.
2. **You compete in the scraped tier** (`bright`/`oxylabs`). Be cleanly scrapable. The licensed `labrador` tier is closed unless you're a national newspaper.
3. **You cannot self-cite for recommendations** — earn third-party coverage (review sites, Reddit) for opinion claims; own your page only for raw facts.
4. **Survive a `site:yourdomain.com/pricing` probe** and write for the reworded sub-query the model actually fires.
5. **One authoritative page per claim** beats a pile of thin ones (domain dedup collapses them).
6. **Check whether the query even searches** — how-to and definitional queries are answered from training where no page can compete.

## Recon Method (Reproducible)

Open ChatGPT → Cmd+Opt+I → Network → tick Preserve log → run query → Cmd+Opt+F search responses for `result_source`. For fan-out/citations/reasoning: Console → `allow pasting` → fetch `/api/auth/session` for `accessToken`, then `/backend-api/conversation/<id>` with Bearer auth, walk JSON for `result_source` objects → `console.table`. Reads your own session only.

## Caveat

Snapshot of a system that changes weekly. Structure and field names hold; frequency numbers are directional only. Single logged-in Pro account, SaaS/tech-skewed query batch.

## Counter-Arguments

- Sample size (~1,240 records, tech/SaaS-skewed) limits generalizability to e-commerce or health queries.
- OpenAI could change internal field names or routing logic at any time, making this a perishable snapshot.
- The `labrador` tier interpretation as "licensed" is inferred, not confirmed by OpenAI.
- Percentages (Reddit 278× fetched, etc.) should not be treated as statistically representative.

## Sources

- [[Knowledge/AI/chatgpt-source-selection-network-traffic-suganthan.md|chatgpt-source-selection-network-traffic-suganthan]]
- [[chatgpt-source-selection-network-traffic-suganthan|chatgpt-source-selection-network-traffic-suganthan]]

## Related

- [[Wiki/Domains/_shared/how-chatgpt-actually-picks-sources-read-from-raw-network-tra|How ChatGPT Actually Picks Sources Read From Raw Network Tra]]
