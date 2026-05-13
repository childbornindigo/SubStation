---
name: Scraping Skill Plan
status: active
created: 2026-05-12
---

# Scraping Skill — Plan

## Goal
Build a reusable scraping pattern: **Firecrawl primary → ScrapLing fallback**. Then run the expanded tire scrape.

## Skill Design

### Architecture
```
scrape(url, extract_schema, prompt) →
  1. Try Firecrawl (API, structured extraction, JS rendering)
  2. If Firecrawl fails (CAPTCHA, WAF, blocked, timeout) → 
     ScrapLing CLI (stealth fetch, real Chrome, Cloudflare solve)
  3. Return unified result (markdown + extracted data)
```

### When to use which
- **Firecrawl**: default for everything. API-based, returns structured data via LLM extraction. 791/1000 credits left.
- **ScrapLing**: fallback for anti-bot-protected sites. CLI tool, uses real Chrome + stealth features. No credit limit.
- **Google Places**: separate use case — business/location data, not web scraping. Not part of this skill.

### ScrapLing integration
ScrapLing CLI at `/opt/homebrew/bin/scrapling`:
- `scrapling extract get <url>` — basic static fetch
- `scrapling extract fetch <url>` — JS-rendered (Playwright)
- `scrapling extract stealthy-fetch <url>` — anti-bot bypass mode
  - `--solve-cloudflare` — beats Turnstile challenges
  - `--real-chrome` — uses actual Chrome, not headless
  - `--hide-canvas` — defeats fingerprinting
  - `--block-webrtc` — hides real IP

### Implementation
Create a unified Python module `scrape_utils.py` that both existing scrapers can import:
- `smart_scrape(url, schema=None, prompt=None)` → returns {markdown, extracted, source}
- Tries Firecrawl first
- On failure → ScrapLing stealthy-fetch → parse markdown → LLM extract if schema provided
- Logs which engine succeeded

### Tire Scrape Run Plan
1. Update `scrape-firecrawl.py` to import `scrape_utils.py` and use smart fallback
2. Update `scrape-marketplaces.py` same way
3. Run retail scraper (65 sizes × 5 sources, ScrapLing for Canadian Tire/Walmart)
4. Run marketplace scraper (Kijiji + eBay, ScrapLing fallback)
5. Output to `market-intelligence/` dir

### Credit budget
- 791 Firecrawl credits remaining
- ~65 sizes × 3 primary sources = ~195 calls minimum
- With fallbacks and retries: ~250-300 credits estimated
- Leaves ~490 credits after run

### Analysis Output (what Jun gets)
After all scraping completes, run brand-level analysis:
- **Top brands by demand** — Google Trends interest + marketplace listing counts
- **Top brands by availability** — how many retailers carry them, how many sizes
- **Top brands by value** — price positioning vs. demand
- **Top 5 per category** — all-season, winter, performance
- **Overall top 10 brands ranked** — composite score (demand × availability × value)

Output: `market-intelligence/tire-brand-rankings.json` + human-readable summary

### Key change: Firecrawl key NOT configured
ScrapLing is primary engine. Firecrawl becomes automatic upgrade when `FIRECRAWL_API_KEY` env var is set.

### What actually happened
- Individual retailer sites (1010tires, TireRack, SimpleTire) are Cloudflare/Akamai blocked
- PMCtire works but doesn't filter by size server-side (JS-side filtering)
- Kijiji/eBay return empty shells (too JS-heavy)
- **Google Shopping is the winner**: returns 50-80 listings per size with brands, models, prices from multiple Canadian retailers

### Data source pivot
- Primary: Google Shopping (`google.ca/search?q=buy+{size}+tires+canada&tbm=shop`)
- Marketplace: Google site-search for Kijiji/eBay counts
- Brand trends: Google Trends for top 15 brand names
- Size trends: Google Trends for top 20 sizes

## Status
- [x] Build scrape_utils.py (ScrapLing primary, Firecrawl optional)
- [x] Build tire_scraper.py (Google Shopping + Trends + marketplace)
- [x] Build analyze_tires.py (brand ranking for Jun)
- [~] Run full pipeline (running now, ~20 min ETA)
- [ ] Verify output data quality
- [ ] Run analyze_tires.py and deliver results to Dee
