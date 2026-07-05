# Graph Report - substation  (2026-07-04)

## Corpus Check
- 83 files · ~807,222 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 728 nodes · 984 edges · 47 communities (46 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b2ae5b01`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_index.js|index.js]]
- [[_COMMUNITY_MODEL-BY-MODEL DETAILED AUDIT|MODEL-BY-MODEL DETAILED AUDIT]]
- [[_COMMUNITY_analyze_tires.py|analyze_tires.py]]
- [[_COMMUNITY_scrape_utils.py|scrape_utils.py]]
- [[_COMMUNITY_LuxuryLane — Dee's 4 storefront issues + mismatch check (2026-05-31)|LuxuryLane — Dee's 4 storefront issues + mismatch check (2026-05-31)]]
- [[_COMMUNITY_LuxuryLane — Jun's New Feature Batch (May 29)|LuxuryLane — Jun's New Feature Batch (May 29)]]
- [[_COMMUNITY_right-hand-sdk.mjs|right-hand-sdk.mjs]]
- [[_COMMUNITY_scrape-all-local-2026-05-19.py|scrape-all-local-2026-05-19.py]]
- [[_COMMUNITY_scrape-all-local-shops.py|scrape-all-local-shops.py]]
- [[_COMMUNITY_build_excel.py|build_excel.py]]
- [[_COMMUNITY_Peptide Site Rebuild — Medvi Layout|Peptide Site Rebuild — Medvi Layout]]
- [[_COMMUNITY_package.json|package.json]]
- [[_COMMUNITY_compile-market-pricing.py|compile-market-pricing.py]]
- [[_COMMUNITY_build_final_excel.py|build_final_excel.py]]
- [[_COMMUNITY_scrape-all-local-2026-05-18.py|scrape-all-local-2026-05-18.py]]
- [[_COMMUNITY_scrape-tdot-krave.py|scrape-tdot-krave.py]]
- [[_COMMUNITY_scrape-local-market-scrapling.py|scrape-local-market-scrapling.py]]
- [[_COMMUNITY_tire_scraper.py|tire_scraper.py]]
- [[_COMMUNITY_Skill Design|Skill Design]]
- [[_COMMUNITY_README|README.md]]
- [[_COMMUNITY_8 Items — Execution Order|8 Items — Execution Order]]
- [[_COMMUNITY_Facebook Marketplace Listings — LuxuryLane Tires|Facebook Marketplace Listings — LuxuryLane Tires]]
- [[_COMMUNITY_scrape-ct-v2.py|scrape-ct-v2.py]]
- [[_COMMUNITY_scrape-ct-v3.py|scrape-ct-v3.py]]
- [[_COMMUNITY_LuxuryLane — Metadata  Grouping  $0  MSRP fix|LuxuryLane — Metadata / Grouping / $0 / MSRP fix]]
- [[_COMMUNITY_Canadian Tire Market Research — GTAVaughan Focus|Canadian Tire Market Research — GTA/Vaughan Focus]]
- [[_COMMUNITY_LuxuryLane Tires — Jun's Tire Business|LuxuryLane Tires — Jun's Tire Business]]
- [[_COMMUNITY_scrape-michelin-local.py|scrape-michelin-local.py]]
- [[_COMMUNITY_LuxuryLane — Parked Reintro + Quality Pass|LuxuryLane — Parked Reintro + Quality Pass]]
- [[_COMMUNITY_Market-Driven Pricing — Active Thread|Market-Driven Pricing — Active Thread]]
- [[_COMMUNITY_compile-michelin-pricing.py|compile-michelin-pricing.py]]
- [[_COMMUNITY_generate_listings.py|generate_listings.py]]
- [[_COMMUNITY_LuxuryLane CRM Backend Prep|LuxuryLane CRM Backend Prep]]
- [[_COMMUNITY_audit-tdg-all-brands.py|audit-tdg-all-brands.py]]
- [[_COMMUNITY_scrape-zracing-fresh.py|scrape-zracing-fresh.py]]
- [[_COMMUNITY_scrape-ct-final.py|scrape-ct-final.py]]
- [[_COMMUNITY_LuxuryLane — CT local-market price scrape (above-MSRP comps)|LuxuryLane — CT local-market price scrape (above-MSRP comps)]]
- [[_COMMUNITY_scrape-noble-tire.py|scrape-noble-tire.py]]
- [[_COMMUNITY_Infrastructure Reality Audit|Infrastructure Reality Audit]]
- [[_COMMUNITY_scrape-canadian-tire.py|scrape-canadian-tire.py]]
- [[_COMMUNITY_parse_page|parse_page]]
- [[_COMMUNITY_scrape_pmctire.py|scrape_pmctire.py]]
- [[_COMMUNITY_normalize_size|normalize_size]]

## God Nodes (most connected - your core abstractions)
1. `MODEL-BY-MODEL DETAILED AUDIT` - 22 edges
2. `LuxuryLane — Jun's New Feature Batch (May 29)` - 19 edges
3. `log()` - 15 edges
4. `LuxuryLane — Dee's 4 storefront issues + mismatch check (2026-05-31)` - 14 edges
5. `invokeClaudeSDK()` - 13 edges
6. `invokeCodex()` - 12 edges
7. `Facebook Marketplace Listings — LuxuryLane Tires` - 12 edges
8. `main()` - 11 edges
9. `scrape_shop()` - 11 edges
10. `LuxuryLane — Metadata / Grouping / $0 / MSRP fix` - 11 edges

## Surprising Connections (you probably didn't know these)
- `scrape_model()` --calls--> `_try_scrapling()`  [INFERRED]
  scripts/scrape-tdot-gap-fill.py → scripts/scrape_utils.py

## Import Cycles
- None detected.

## Communities (47 total, 1 thin omitted)

### Community 0 - "index.js"
Cohesion: 0.06
Nodes (55): ALLOWED_IMAGE_HOSTS, ANTHROPIC_MODELS, buildAnthropicBody(), buildCodexBody(), buildProviderModels(), CODEX_CHATGPT_REMAP, convertForAnthropic(), convertForCodex() (+47 more)

### Community 1 - "MODEL-BY-MODEL DETAILED AUDIT"
Cohesion: 0.05
Nodes (37): 10. Bridgestone Turanza Everdrive (70 sizes), 11. Pirelli P Zero AS Plus 3 (148 sizes), 12. Hankook Ventus S1 noble2 RunFlat (5 sizes), 13. Pirelli Scorpion Zero All Season (39 sizes), 14. Bridgestone Alenza Sport AS (20 sizes), 15. Toyo Open Country A50 (1 size), 16. Firestone Weathergrip (31 sizes), 17. Bridgestone WeatherPeak (31 sizes) (+29 more)

### Community 2 - "analyze_tires.py"
Cohesion: 0.08
Nodes (37): assess_quality(), build_output(), _clean_brand(), compute_category_leaders(), compute_composite(), compute_coverage(), compute_popularity(), compute_size_opportunities() (+29 more)

### Community 3 - "scrape_utils.py"
Cohesion: 0.09
Nodes (28): dedupe(), main(), parse_entries(), Parse tire entries. We scan through blocks anchored by #### headings.     Within, fetch_model_page(), main(), parse_sizes_and_prices(), Try each slug candidate until we get a successful scrape.     Returns (url_used, (+20 more)

### Community 4 - "LuxuryLane — Dee's 4 storefront issues + mismatch check (2026-05-31)"
Cohesion: 0.08
Nodes (25): #1 TITLES — FIXED + DEPLOYED. commit 78d659d (display_name ?? tdg_card_model ?? model)., #1 TITLES — ROOT CAUSE FOUND, code fix STAGED (needs deploy), #2 / #3 OUT-OF-STOCK — two compounding causes, #2/#3 ROOT CAUSE WAS WRONG — it's NOT staleness. Dee was right., #4 IMAGES — 1,793 active .jpg rows repointed to existing transparent .png twins., #4 IMAGES white bg — confirmed, *** CORRECTION + RESOLUTION (2026-05-31 PM) ***, Files (+17 more)

### Community 5 - "LuxuryLane — Jun's New Feature Batch (May 29)"
Cohesion: 0.08
Nodes (23): 2026-05-29 23:xx — Web-fallback pricing run + ZOMBIE PROCESS INCIDENT, 2026-05-30 ~15:25 — PROD DEPLOY SHIPPED ✅, BLOCKERS, Build sequence (LOCKED, in order):, Canvassing target spec (from Jun), DATA INTEGRITY AUDIT — May 29 16:14 (Dee: "real data only, no fabrication"), Deploy state, FULL RE-SCRAPE COMPLETE — May 29 15:38 (the real data foundation) (+15 more)

### Community 6 - "right-hand-sdk.mjs"
Cohesion: 0.13
Nodes (21): askClaude(), askSubStation(), AUTH_FAIL_FILE, authFailedTokens, benchToken(), cappedTokens, isAuthError(), isCapError() (+13 more)

### Community 7 - "scrape-all-local-2026-05-19.py"
Cohesion: 0.16
Nodes (22): clean_results(), deduplicate(), extract_ct_variant_price(), load_existing_data(), log(), main(), normalize_size(), parse_ct_pdp() (+14 more)

### Community 8 - "scrape-all-local-shops.py"
Cohesion: 0.13
Nodes (21): extract_tires_from_text(), normalize_size(), Scrape a shop for all brands/models. Returns list of {brand, model, size, price}, Canadian Tire — JS SPA, search by brand+model., TDot Performance — standard search., Active Green + Ross — search tires., Point S Canada — tire search., KRAVE Automotive — search. (+13 more)

### Community 9 - "build_excel.py"
Cohesion: 0.29
Nodes (18): build_brand_rankings(), build_margin_analysis(), build_master_catalogue(), build_price_intelligence(), build_seasonal_calendar(), build_size_opportunities(), build_top_models(), features_str() (+10 more)

### Community 10 - "Peptide Site Rebuild — Medvi Layout"
Cohesion: 0.11
Nodes (18): Brand Asset Audit, Brand Assets (3 logos Dee provided), Copy Changes, Dee's Correction (2026-05-11), E-Commerce / Checkout (confirmed 2026-05-11), Goal, Image Generation (Dee generating — 2026-05-11), Landing Page Redesign (+10 more)

### Community 11 - "package.json"
Cohesion: 0.11
Nodes (18): dependencies, @anthropic-ai/claude-agent-sdk, @supabase/supabase-js, description, keywords, license, main, name (+10 more)

### Community 12 - "compile-market-pricing.py"
Cohesion: 0.17
Nodes (18): build_local_index(), compute_recommendation(), find_local_matches(), generate_summary(), load_all_local_prices(), load_inventory(), load_shop_data(), main() (+10 more)

### Community 13 - "build_final_excel.py"
Cohesion: 0.12
Nodes (12): classify_category(), clean_model_name(), fuzzy_model_match(), parse_size(), performance_tier(), Parse '225/65R17' -> (225, 65, 17). Returns (None, None, None) on fail., Remove Bing Shopping junk from model strings., Classify into: Winter, All-Season, Performance, All-Terrain, Summer. (+4 more)

### Community 14 - "scrape-all-local-2026-05-18.py"
Cohesion: 0.16
Nodes (17): build_direct_url(), build_search_url(), extract_price(), extract_size(), load_progress(), main(), parse_tire_listings(), Build search/product URL for a given shop and tire model. (+9 more)

### Community 15 - "scrape-tdot-krave.py"
Cohesion: 0.17
Nodes (16): get_tdot_brand_page_products(), parse_krave_product_page(), parse_tdot_product_title(), Scrape a TDot product page and return the per-unit price., Scrape one page of a TDot brand listing.     Returns list of {'title': ..., 'url, Scrape TDot Performance brand pages.     Pages 1-3 per brand to keep it tractabl, Parse KRAVE product page. Size/price table appears as:       XX" - 285/70R17 - $, Use Scrapling directly since Firecrawl credits are exhausted. (+8 more)

### Community 16 - "scrape-local-market-scrapling.py"
Cohesion: 0.24
Nodes (15): main(), parse_prices_from_text(), Search Google Shopping for a specific tire and extract prices., Scrape Canadian Tire using Scrapling's dynamic fetch for JS-rendered SPA., Scrape OK Tire using stealthy fetch., Scrape Kal Tire using stealthy fetch., Scrape Costco Tire Centre., Fetch a URL using Scrapling CLI. Returns markdown content or None. (+7 more)

### Community 17 - "tire_scraper.py"
Cohesion: 0.24
Nodes (15): _detect_tire_type(), _extract_google_shopping_listings(), _extract_listing_count(), _extract_trends_score(), _log(), main(), _normalize_brand(), Phase 1: Google Shopping for tire prices across sizes. (+7 more)

### Community 18 - "Skill Design"
Cohesion: 0.13
Nodes (14): Analysis Output (what Jun gets), Architecture, Credit budget, Data source pivot, Goal, Implementation, Key change: Firecrawl key NOT configured, Scraping Skill — Plan (+6 more)

### Community 19 - "README.md"
Cohesion: 0.13
Nodes (14): 1. Install, 2. Add your Claude Max tokens, 3. Add ChatGPT tokens (optional), 4. Configure OpenClaw, 5-minute setup, 5. Restart OpenClaw, 8 models, 2 providers, 0 API cost, Endpoints (+6 more)

### Community 20 - "8 Items — Execution Order"
Cohesion: 0.14
Nodes (13): 1. Image Sizing (CSS fix), 2. Image Background Removal, 3. Restore Review Counts, 4. Tire Finder — Show All Fitting Tires, 5. Browse Page — Show All 33 Model Variants, 6. Merge Primacy Tour A/S Duplicate, 7. Local Market Pricing — ALL Brands Including Michelin, 8. Deploy to Jun's Vercel (luxurylanetires.ca) (+5 more)

### Community 21 - "Facebook Marketplace Listings — LuxuryLane Tires"
Cohesion: 0.15
Nodes (12): Facebook Marketplace Listings — LuxuryLane Tires, LISTING 10 — General: All Brands All Sizes, LISTING 11 — General: Tire Warehouse, LISTING 1 — Pirelli P Zero All Season 235/45R18, LISTING 2 — Hankook WeatherFlex GT 225/50R17, LISTING 3 — Hankook Kinergy 4S2 225/40R18, LISTING 4 — Hankook WeatherFlex GT 235/40R19, LISTING 5 — Hankook WeatherFlex GT 255/45R19 (+4 more)

### Community 22 - "scrape-ct-v2.py"
Cohesion: 0.29
Nodes (12): extract_prices_from_markdown(), find_ct_product_urls(), firecrawl_scrape(), firecrawl_search(), log(), main(), Search for CT product pages for a tire model., Scrape a CT product page and extract all sizes + prices. (+4 more)

### Community 23 - "scrape-ct-v3.py"
Cohesion: 0.28
Nodes (12): extract_size_variant_urls(), fc_scrape(), fc_search(), find_main_product_url(), log(), main(), Extract per-size variant URLs from main product page markdown., Scrape a per-size CT product page. Returns dict or None. (+4 more)

### Community 24 - "LuxuryLane — Metadata / Grouping / $0 / MSRP fix"
Cohesion: 0.17
Nodes (11): 2026-06-01 ROUND 2 (Dee follow-up), 2026-06-01 ROUND 3 — "prices not aligned with local data" (DWS06 $405 vs $373.99), Audit numbers (7,211 active tires), Creds: .env.local.prod (SUPABASE_SERVICE_ROLE_KEY). Repo: /Users/indigochild/luxurylanetires.ca, Deploy: vercel --prod --scope luxury-lane-tires-ca-s-projects (token .env*.vercel). Show-before-ship., LuxuryLane — Metadata / Grouping / $0 / MSRP fix, OPEN QUESTIONS FOR DEE, PLAN (+3 more)

### Community 25 - "Canadian Tire Market Research — GTA/Vaughan Focus"
Cohesion: 0.17
Nodes (11): Canadian Tire Market Research — GTA/Vaughan Focus, HIGH-MARGIN OPPORTUNITIES, Performance (GTA luxury market — Vaughan/Woodbridge = high exotic density), RECOMMENDED INITIAL INVENTORY, SEASONAL, Summer/All-Season, TOP 20 TIRE SIZES BY POPULARITY, TOP BRANDS (+3 more)

### Community 26 - "LuxuryLane Tires — Jun's Tire Business"
Cohesion: 0.17
Nodes (11): Business Model, Deliverables, Deployment, Future, Goal, Key Details, LuxuryLane Tires — Jun's Tire Business, Strategy Notes (+3 more)

### Community 27 - "scrape-michelin-local.py"
Cohesion: 0.24
Nodes (11): dedupe_by_size(), extract_prices(), extract_tire_size(), main(), parse_tire_listings(), Extract tire size patterns like 225/45R17, 265/70R17, etc., Extract prices from text. Handles $xxx.xx, xxx.xx$, $x,xxx.xx formats., Generic parser: tries multiple strategies to extract size+price pairs     from s (+3 more)

### Community 28 - "LuxuryLane — Parked Reintro + Quality Pass"
Cohesion: 0.18
Nodes (10): 2026-05-31 — getAllTires 1000-row cap fix, ACTIVATION GATE (per-model, nothing live until all 3 pass), DEE'S DIRECTIVE (do not re-litigate), DONE (persisted in DB, survived reboot), IN FLIGHT (background jobs, detached — survive session, NOT reboot-safe for /tmp), KEY FACTS, LuxuryLane — Parked Reintro + Quality Pass, NEXT (+2 more)

### Community 29 - "Market-Driven Pricing — Active Thread"
Cohesion: 0.18
Nodes (10): CRITICAL LESSON (May 18 — DO NOT REPEAT), Current State, Explicitly REMOVED (not local), Goal, Local Shops (confirmed by Dee), Market-Driven Pricing — Active Thread, Methodology, Next Actions (+2 more)

### Community 30 - "compile-michelin-pricing.py"
Cohesion: 0.38
Nodes (10): build_local_index(), compute_recommendation(), find_local_matches(), generate_summary(), load_all_local_prices(), load_inventory(), load_shop_data(), main() (+2 more)

### Community 31 - "generate_listings.py"
Cohesion: 0.31
Nodes (9): create_general_warehouse(), create_tire_listing(), download_image(), draw_rounded_rect(), get_font(), Download image from URL, return PIL Image., Draw a rounded rectangle., Create a professional FB Marketplace listing image. (+1 more)

### Community 32 - "LuxuryLane CRM Backend Prep"
Cohesion: 0.20
Nodes (9): Current State (2026-05-31), Decisions Made, DONE (2026-05-31, verified end-to-end on live DB, NOT deployed), Goal, LuxuryLane CRM Backend Prep, Open Loops / Forks for Dee, OPEN — needs Dee's call before next step, Why It Matters (+1 more)

### Community 33 - "audit-tdg-all-brands.py"
Cohesion: 0.29
Nodes (9): build_entry(), match_model(), normalize_size(), parse_brand_page(), Parse TDG page text for a specific brand's tires.     Returns list of {model, si, Build a tire entry from parsed components., Normalize TDG size to match our DB format: e.g., 'P225/45R17 XL' -> '225/45R17, Try to match a TDG model name to one of our model names.     Returns the matched (+1 more)

### Community 35 - "scrape-zracing-fresh.py"
Cohesion: 0.27
Nodes (9): is_404_or_empty(), main(), parse_price(), parse_tire_table(), Extract float price from string like '$270.00' or '270'., Parse markdown content from a ZRacing tire page.     Returns list of {brand, mod, Detect if page returned 404 or no product found., Try all slugs for a model, return list of price records or NOT FOUND entry. (+1 more)

### Community 36 - "scrape-ct-final.py"
Cohesion: 0.39
Nodes (8): get_size_variant_urls(), log(), main(), Extract per-size variant relative URLs from main product page content., Scrape a per-size CT product page. Returns result dict or None., Fetch a URL with scrapling stealthy mode. Returns text content or None., scrape_size_variant(), scrapling_fetch()

### Community 37 - "LuxuryLane — CT local-market price scrape (above-MSRP comps)"
Cohesion: 0.25
Nodes (7): COMPLETE (2026-06-01 ~13:15), Goal, KNOWN ISSUE to fix after run, LuxuryLane — CT local-market price scrape (above-MSRP comps), Michelin manual override (DONE), Next after scrape completes, State (2026-06-01)

### Community 38 - "scrape-noble-tire.py"
Cohesion: 0.36
Nodes (7): main(), parse_sizes(), parse_tires(), Extract unique width/profile/rim combos from inventory., POST to Noble Tire and return HTML response text., Parse Noble Tire HTML response, return list of matching tire dicts., search_noble_tire()

### Community 39 - "Infrastructure Reality Audit"
Cohesion: 0.29
Nodes (6): Audit Categories, Context, Goal, Infrastructure Reality Audit, Started, Status

### Community 40 - "scrape-canadian-tire.py"
Cohesion: 0.47
Nodes (5): extract_tire_prices(), main(), Extract tire sizes and prices from scraped markdown., Search Canadian Tire for a tire model., scrape_ct_search()

### Community 41 - "parse_page"
Cohesion: 0.47
Nodes (5): build_entry(), parse_page(), Parse the page text into Michelin tire entries.      The text follows this patte, Build a single tire entry from parsed components., scrape_michelin()

### Community 42 - "scrape_pmctire.py"
Cohesion: 0.70
Nodes (4): extract_product(), fetch_page(), main(), scrape_size()

## Knowledge Gaps
- **236 isolated node(s):** `name`, `version`, `description`, `type`, `main` (+231 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Download image from URL, return PIL Image.`, `Draw a rounded rectangle.`, `Create a professional FB Marketplace listing image.` to the rest of the system?**
  _363 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `index.js` be split into smaller, more focused modules?**
  _Cohesion score 0.061507936507936505 - nodes in this community are weakly interconnected._
- **Should `MODEL-BY-MODEL DETAILED AUDIT` be split into smaller, more focused modules?**
  _Cohesion score 0.05263157894736842 - nodes in this community are weakly interconnected._
- **Should `analyze_tires.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08392603129445235 - nodes in this community are weakly interconnected._
- **Should `scrape_utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09475806451612903 - nodes in this community are weakly interconnected._
- **Should `LuxuryLane — Dee's 4 storefront issues + mismatch check (2026-05-31)` be split into smaller, more focused modules?**
  _Cohesion score 0.07692307692307693 - nodes in this community are weakly interconnected._
- **Should `LuxuryLane — Jun's New Feature Batch (May 29)` be split into smaller, more focused modules?**
  _Cohesion score 0.08333333333333333 - nodes in this community are weakly interconnected._