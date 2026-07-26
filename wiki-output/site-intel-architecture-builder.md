---
title: Site Intel Architecture Builder
type: wiki-page
domain: sales-compass
status: active
created: 2026-04-14
updated: 2026-07-25
confidence: high
retention: durable
tags:
  - island/knowledge
  - island/sales
  - island/ai-agent
  - type/architecture
  - site-builder
  - build-pipeline
  - gmb
  - vault-search
  - lead-intel
  - automation
parent: "[[Wiki/Domains/sales-compass/pipeline-and-website-builder-architecture-summary|Pipeline And Website Builder Architecture Summary]]"
---

> **TLDR:** `build-site.sh` gained 5 auto-loading intel layers that inject real business context into every site generation prompt.

## Summary

The original `build-site.sh` pipeline only knew business-type design specs, reference sites, and universal patterns — it had no access to real business intelligence. Four missing bridges were identified: GMB data, sales pipeline lead intel, [[ObsidianBrain]] vault knowledge, and Beech template structures. All four were wired in (plus a fifth: voice-of-customer from Google reviews), making the builder context-aware rather than generic. Sites now generate from real customer language, live GMB data, and accumulated industry knowledge instead of placeholder logic.

## The Gap (Before)

Original `build-site.sh` inputs:
- Business-type design spec (e.g. `mechanic.md`)
- Universal design patterns
- Reference site URLs (e.g. `fixngo.ca`)

**What was missing:**

| Gap | Source | Why It Matters |
|-----|--------|----------------|
| GMB data | `gmb-extract.sh` output JSON | Real reviews, hours, photos, rating, categories went unused |
| Lead intel | Sales pipeline intake (calls, card scans, pain points) | Builder generated generic copy instead of personalized differentiators |
| Vault knowledge | [[ObsidianBrain]] industry notes, competitor analyses | Industry-specific conversion patterns never applied |
| Beech templates | Actual downloadable template structures | Only inspo URLs referenced, not structural patterns |

## Target Prompt Architecture

```
business data (GMB + lead card)
  + business-type design spec (mechanic.md)
  + reference sites (fixngo.ca patterns)
  + universal design patterns
  + vault intel (industry notes, competitor analysis)
  + animation guide
  → Claude agent builds site
```

## Implemented: 5 Intel Layers

### 1. GMB Auto-Extraction
Auto-runs `gmb-extract.sh` if no `data.json` is passed or the file doesn't exist. Pulls: business name, hours, photos, star rating, categories, place ID. Eliminates the manual GMB pre-step entirely.

### 2. Vault Search
Queries [[ObsidianBrain]] for notes matching the business type. Pulls up to 5 most relevant results: industry intel, conversion insights, competitor analyses. Injected as a context block into the build prompt.

### 3. Lead / Pipeline Data
Accepts `--lead-json path` with sales intake notes containing pain points, differentiators, competitor info, budget tier, and what the owner said on the call. Builder uses this for personalized copy instead of generic text.

### 4. Voice of Customer
Auto-fetches real Google reviews via the place ID from GMB data. Builder uses actual customer language in testimonials and weaves those phrases into service descriptions for authentic tone.

### 5. Workflow References
Loads stealable patterns from classified workflows in the taxonomy. As more methods are classified, the builder passively absorbs their best techniques.

## Status Banner

Displayed at build time to confirm which layers fired:

```
Intel: vault=yes lead=yes workflows=no reviews=yes
```

## Usage

```bash
# Full auto — extracts GMB + reviews + vault + workflows automatically
./build-site.sh "Johnson Auto Collision" mechanic

# With pre-supplied data JSON + lead intel
./build-site.sh "Johnson Auto Collision" mechanic data.json --lead-json lead.json
```

First live test target: [[Johnson Auto Collision]]

## Bridge Status

| Bridge | Status | Notes |
|--------|--------|-------|
| GMB → build-site | ✅ Wired | Auto-triggers `gmb-extract.sh` |
| Pipeline lead data | ✅ Wired | Via `--lead-json` flag |
| Vault search at build time | ✅ Wired | Queries [[ObsidianBrain]], top-5 results |
| Google reviews / VoC | ✅ Wired | Uses Place ID from GMB data |
| Workflow references | ✅ Wired | Pulls from classified taxonomy |
| Beech's actual templates | ⚠️ Pending | Structure ingestion not yet wired |

## Counter-Arguments

- **Over-engineering risk:** Assembling 5 data sources per build adds latency and failure surface area. If GMB API or vault search is unavailable, the build degrades or fails.
- **Vault quality dependency:** Vault search is only as good as what's been ingested — early builds on new industries yield little value from this layer.
- **Lead JSON is optional:** Without `--lead-json`, the personalization layer is silent; generic copy risk remains for leads with thin intake data.

## Sources

- [[Wiki/Domains/sales-compass/site-intel-architecture-builder.md|site-intel-architecture-builder]]
- [[Wiki/Domains/sales-compass/builder-site-architecture-intel.md|builder-site-architecture-intel]]
- [[site-intel-architecture-builder|site-intel-architecture-builder]]
- [[builder-site-architecture-intel|builder-site-architecture-intel]]

## Related

- [[Wiki/Domains/sales-compass/pipeline-and-website-builder-architecture-summary|Pipeline And Website Builder Architecture Summary]]
- [[Wiki/Domains/sales-compass/site-intel-builder-architecture|Site Intel Builder Architecture]]
- [[Wiki/Domains/sales-compass/site-builder-intel-layer-architecture|Site Builder Intel Layer Architecture]]
