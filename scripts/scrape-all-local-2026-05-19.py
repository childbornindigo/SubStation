#!/usr/bin/env python3
"""
scrape-all-local-2026-05-19.py — Comprehensive local GTA tire shop price scraper.

Scrapes local shops for tire pricing, merges existing data where live scraping
is not possible.

LIVE SCRAPING (ScrapLing — Firecrawl credits exhausted):
  - TDot Performance (tdotperformance.ca) — Magento catalog search, works great
  - Canadian Tire (canadiantire.ca) — PDP pages with size variant tables

EXISTING DATA MERGE (sites are JS-rendered, paywalled, or defunct):
  - Point S (point-s.ca) — 1209 entries from previous Firecrawl scrape
  - ZRacing (zracing.ca) — 182 entries from previous scrape
  - Noble Tire (nobletire.ca) — 73 entries from previous scrape
  - KRAVE (kraveautomotive.com) — 57 entries; site now under construction
  - Active Green + Ross — JS tire selector, 0 entries
  - Fas-Tire (fas-tire.ca) — domain defunct (GoDaddy parking)

Output: local-prices-all-shops-2026-05-19.json
"""

import json
import os
import sys
import re
import time
import traceback
from datetime import datetime

SCRIPTS_DIR = "/Users/indigochild/.hermes/extensions/substation/scripts"
OUTPUT_FILE = os.path.join(SCRIPTS_DIR, "local-prices-all-shops-2026-05-19.json")
PROGRESS_FILE = os.path.join(SCRIPTS_DIR, "scrape-progress-2026-05-19.json")

sys.path.insert(0, SCRIPTS_DIR)
from scrape_utils import smart_scrape

# ---------------------------------------------------------------------------
# Brand / Model catalog
# ---------------------------------------------------------------------------
MODELS_BY_BRAND = {
    "Bridgestone": [
        "Alenza Sport AS",
        "Turanza Everdrive",
        "UltraWeather",
        "WeatherPeak",
        "Dueler HL Alenza Plus",
    ],
    "Continental": [
        "CrossContact LX Sport",
        "CrossContact RX",
        "ExtremeContact DWS06 Plus",
        "ProContact RX",
    ],
    "Falken": ["Azenis FK460 AS"],
    "Firestone": [
        "Destination LE3",
        "Destination LE 2",
        "FR710",
        "Weathergrip",
    ],
    "Hankook": [
        "Kinergy GT",
        "Ventus S1 noble2",
        "Dynapro evo AS",
    ],
    "Michelin": [
        "Pilot Sport AS 4",
        "Primacy Tour AS",
        "Defender LTX M/S2",
    ],
    "Pirelli": [
        "P Zero AS Plus 3",
        "Scorpion Zero All Season",
    ],
    "Toyo": [
        "Extensa AS II",
        "Open Country A50",
    ],
    "Yokohama": [
        "Geolandar X-AT",
        "Geolandar X-CV",
    ],
}

BRANDS = list(MODELS_BY_BRAND.keys())

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
SIZE_RE = re.compile(r'[PL]?T?\s*(\d{3})\s*/\s*(\d{2,3})\s*[RZW]\s*(\d{2})', re.IGNORECASE)
PRICE_RE = re.compile(r'C?\$\s*(\d{2,4}(?:\.\d{2})?)')
CLEAN_SIZE_RE = re.compile(r'(\d{3}/\d{2,3}R\d{2})')


def normalize_size(s):
    """Normalize tire size to format like '225/45R17'."""
    if not s:
        return ""
    s = s.strip().upper()
    # Remove P/LT prefix
    for prefix in ["P", "LT"]:
        if s.startswith(prefix) and len(s) > len(prefix) and s[len(prefix)].isdigit():
            s = s[len(prefix):]
    s = s.replace(" ", "")
    # Replace ZR with R
    s = re.sub(r'(\d)ZR(\d)', r'\1R\2', s)
    # Extract just the size portion (remove load/speed ratings like "104H", "XL 113V")
    m = CLEAN_SIZE_RE.search(s)
    if m:
        return m.group(1)
    return s


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def save_progress(shop_name, all_results):
    """Save incremental progress."""
    progress = {"last_shop": shop_name, "total": len(all_results), "timestamp": datetime.now().isoformat()}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"  Progress saved: {len(all_results)} total entries")


# ===========================================================================
# TDOT PERFORMANCE — Structured Magento results (WORKS GREAT)
# ===========================================================================
def scrape_tdot():
    """TDot Performance — Magento catalog search with structured results.

    Page structure per product:
      * Tire Size: 225/45R18
      * Load/Speed Index: 91W
      ...
      C$374.13
    """
    shop = "TDot Performance"
    all_results = []
    total_queries = sum(len(v) for v in MODELS_BY_BRAND.values())
    query_num = 0

    for brand in BRANDS:
        for model in MODELS_BY_BRAND[brand]:
            query_num += 1
            q = f"{brand} {model}".replace(" ", "+")
            url = f"https://www.tdotperformance.ca/catalogsearch/result/?q={q}"
            log(f"[{shop}] ({query_num}/{total_queries}) {brand} {model}")

            result = smart_scrape(url, timeout=30000)
            if not result["success"]:
                log(f"  FAILED: {result.get('error', 'unknown')}")
                time.sleep(2)
                continue

            entries = parse_tdot_results(result["content"], brand, model, shop)
            all_results.extend(entries)
            log(f"  -> {len(entries)} prices")
            time.sleep(2)

    return all_results


def parse_tdot_results(content, brand, model, shop):
    """Parse TDot Performance search results.

    Structure:
      [Title with size in parens]
      * Tire Type: Summer
      * Tire Size: 225/45R18
      * Load/Speed Index: 91W
      ...
      C$374.13
    """
    results = []
    lines = content.split('\n')

    current_size = None
    for line in lines:
        stripped = line.strip()

        # Match "Tire Size: xxx" lines
        size_match = re.search(r'Tire Size:\s*(\S+)', stripped)
        if size_match:
            current_size = normalize_size(size_match.group(1))

        # Match standalone "C$xxx.xx" price lines
        price_match = re.match(r'^C?\$\s*(\d{2,4}(?:\.\d{2})?)\s*$', stripped)
        if price_match and current_size:
            price = float(price_match.group(1))
            if 50 < price < 3000:
                results.append({
                    "shop": shop,
                    "brand": brand,
                    "model": model,
                    "size": current_size,
                    "price": price,
                })
                current_size = None  # Reset for next product

    # Fallback: proximity-based matching if structured parse found nothing
    if not results:
        for i, line in enumerate(lines):
            sizes = SIZE_RE.findall(line)
            if not sizes:
                continue
            for offset in range(1, 8):
                if i + offset < len(lines):
                    pm = PRICE_RE.search(lines[i + offset])
                    if pm:
                        price = float(pm.group(1))
                        if 50 < price < 3000:
                            for w, ar, rim in sizes:
                                results.append({
                                    "shop": shop,
                                    "brand": brand,
                                    "model": model,
                                    "size": normalize_size(f"{w}/{ar}R{rim}"),
                                    "price": price,
                                })
                        break

    return results


# ===========================================================================
# CANADIAN TIRE — PDP pages with size variant tables
# ===========================================================================
def scrape_canadian_tire():
    """Canadian Tire — discover PDPs via search, scrape size tables + variant pages."""
    shop = "Canadian Tire"
    all_results = []

    for brand in BRANDS:
        for model in MODELS_BY_BRAND[brand]:
            log(f"[{shop}] Searching: {brand} {model}")

            # Step 1: Search to find PDP URL
            q = f"{brand} {model} tire".replace(" ", "+")
            url = f"https://www.canadiantire.ca/en/search-results.html?q={q}"

            result = smart_scrape(url, timeout=30000, stealth=True)
            if not result["success"]:
                log(f"  Search FAILED: {result.get('error', 'unknown')}")
                time.sleep(3)
                continue

            # Find PDP links matching the brand
            pdp_links = re.findall(r'(/en/pdp/[^")\s]+\.html)', result["content"])
            # Filter to relevant ones
            brand_slug = brand.lower().replace(" ", "-")
            model_words = model.lower().split()
            relevant_links = []
            for link in pdp_links:
                link_lower = link.lower()
                if brand_slug in link_lower:
                    # Check if any model word appears
                    if any(w in link_lower for w in model_words if len(w) > 2):
                        relevant_links.append(link)

            if not relevant_links:
                # Fallback: take first PDP that has brand
                relevant_links = [l for l in pdp_links if brand_slug in l.lower()]

            if not relevant_links:
                log(f"  No PDP found")
                time.sleep(3)
                continue

            # Step 2: Scrape the PDP
            pdp_url = f"https://www.canadiantire.ca{relevant_links[0]}"
            log(f"  PDP: {pdp_url[:80]}...")

            time.sleep(2)
            pdp_result = smart_scrape(pdp_url, timeout=30000, stealth=True)
            if not pdp_result["success"]:
                log(f"  PDP FAILED: {pdp_result.get('error', 'unknown')}")
                time.sleep(3)
                continue

            entries = parse_ct_pdp(pdp_result["content"], brand, model, shop)
            all_results.extend(entries)
            log(f"  -> {len(entries)} prices")
            time.sleep(3)

    return all_results


def parse_ct_pdp(content, brand, model, shop):
    """Parse Canadian Tire PDP page.

    PDP pages have:
    - A price range like "$332.99 - $1,001.99" or single price "$332.99"
    - A table of size variants: | [sku](/en/pdp/...) | [code](...) | [265/40R21 101Y](...) |
    - Each variant has its own URL with exact pricing

    Strategy: extract all sizes from table, scrape up to 8 variant pages for
    exact per-size prices. For remaining sizes, skip (don't use range estimate).
    """
    results = []
    lines = content.split('\n')

    # Extract size entries from table
    size_entries = []
    seen_sizes = set()
    for line in lines:
        sizes = re.findall(r'(\d{3}/\d{2,3}R\d{2})\s*\d*[A-Z]*', line)
        if sizes:
            variant_urls = re.findall(r'(/en/pdp/[^")\s|]+\.html)', line)
            for size in sizes:
                norm = normalize_size(size)
                if norm not in seen_sizes:
                    seen_sizes.add(norm)
                    vurl = f"https://www.canadiantire.ca{variant_urls[0]}" if variant_urls else None
                    size_entries.append({"size": norm, "url": vurl})

    if not size_entries:
        # No variant table; try to find a single product with price
        main_price = None
        for line in lines:
            m = re.match(r'^\s*\$(\d{2,4}\.\d{2})\s*$', line.strip())
            if m:
                p = float(m.group(1))
                if 50 < p < 3000:
                    main_price = p
                    break
        if main_price:
            all_sizes = SIZE_RE.findall(content)
            for w, ar, rim in all_sizes[:10]:
                results.append({
                    "shop": shop, "brand": brand, "model": model,
                    "size": normalize_size(f"{w}/{ar}R{rim}"), "price": main_price,
                })
        return results

    log(f"    {len(size_entries)} sizes in table, scraping up to 8 variants...")

    # Scrape variant pages for exact per-size pricing
    scraped = 0
    for entry in size_entries:
        if entry["url"] and scraped < 8:
            time.sleep(1.5)
            vresult = smart_scrape(entry["url"], timeout=20000, stealth=True)
            if vresult["success"]:
                price = extract_ct_variant_price(vresult["content"])
                if price:
                    results.append({
                        "shop": shop, "brand": brand, "model": model,
                        "size": entry["size"], "price": price,
                    })
                    scraped += 1
                    log(f"    {entry['size']} -> ${price}")
                    continue
            log(f"    {entry['size']} -> failed")

    return results


def extract_ct_variant_price(content):
    """Extract exact price from a Canadian Tire variant page."""
    lines = content.split('\n')
    # First try: standalone price line
    for line in lines:
        m = re.match(r'^\s*\$(\d{2,4}\.\d{2})\s*$', line.strip())
        if m:
            p = float(m.group(1))
            if 50 < p < 3000:
                return p
    # Fallback: any valid price
    prices = PRICE_RE.findall(content)
    valid = [float(p) for p in prices if 50 < float(p) < 3000]
    return valid[0] if valid else None


# ===========================================================================
# Merge existing scraped data
# ===========================================================================
def load_existing_data():
    """Load previously scraped data for all shops.

    These files were populated by earlier scrape runs when Firecrawl was available.
    """
    existing = []
    files_to_merge = {
        "local-prices-point-s.json": "Point S",
        "local-prices-tdot-performance.json": "TDot Performance",
        "local-prices-canadian-tire.json": "Canadian Tire",
        "local-prices-zracing.json": "ZRacing Mississauga",
        "local-prices-noble-tire.json": "Noble Tire Brampton",
        "local-prices-krave.json": "KRAVE",
        "local-prices-active-green-ross.json": "Active Green + Ross",
        "local-prices-fas-tire.json": "Fas-Tire Scarborough",
    }

    for fname, shop_name in files_to_merge.items():
        fpath = os.path.join(SCRIPTS_DIR, fname)
        if os.path.exists(fpath):
            try:
                data = json.load(open(fpath))
                if data:
                    log(f"[Existing] {fname}: {len(data)} entries")
                    for entry in data:
                        entry["shop"] = shop_name
                        if "size" in entry and entry["size"]:
                            entry["size"] = normalize_size(str(entry["size"]))
                        entry.pop("raw_line", None)
                    existing.extend(data)
            except Exception as e:
                log(f"[Existing] Error loading {fname}: {e}")

    return existing


# ===========================================================================
# Deduplication and cleaning
# ===========================================================================
def deduplicate(results):
    """Keep lowest price for same shop+brand+model+size."""
    best = {}
    for r in results:
        key = (r["shop"], r["brand"], r.get("model", ""), r["size"])
        if key not in best or r["price"] < best[key]["price"]:
            best[key] = r
    return list(best.values())


def clean_results(results):
    """Validate and normalize all results."""
    cleaned = []
    for r in results:
        if not r.get("shop") or not r.get("brand") or not r.get("size") or not r.get("price"):
            continue
        try:
            price = float(r["price"])
        except (ValueError, TypeError):
            continue
        if not (50 < price < 3000):
            continue
        size = normalize_size(str(r["size"]))
        if not re.match(r'\d{3}/\d{2,3}R\d{2}$', size):
            continue
        cleaned.append({
            "shop": r["shop"],
            "brand": r["brand"],
            "model": r.get("model", ""),
            "size": size,
            "price": round(price, 2),
        })
    return cleaned


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape local GTA tire shop prices")
    parser.add_argument("--shop", help="Scrape only: tdot, canadian-tire")
    parser.add_argument("--skip-live", action="store_true", help="Only merge existing data")
    args = parser.parse_args()

    all_results = []
    start_time = time.time()

    log("=" * 60)
    log("LOCAL GTA TIRE PRICE SCRAPER — 2026-05-19")
    log("=" * 60)
    log("Live: TDot Performance, Canadian Tire")
    log("Merge: Point S, ZRacing, Noble, KRAVE, AG+R, Fas-Tire")
    log("=" * 60)

    # -----------------------------------------------------------------------
    # Live scraping
    # -----------------------------------------------------------------------
    if not args.skip_live:
        scrapers = {
            "tdot": ("TDot Performance", scrape_tdot),
            "canadian-tire": ("Canadian Tire", scrape_canadian_tire),
        }

        if args.shop:
            if args.shop not in scrapers:
                log(f"Unknown shop: {args.shop}. Available: {', '.join(scrapers.keys())}")
                sys.exit(1)
            scrapers = {args.shop: scrapers[args.shop]}

        for shop_key, (shop_name, scraper_fn) in scrapers.items():
            log(f"\n{'='*60}")
            log(f"LIVE SCRAPING: {shop_name}")
            log(f"{'='*60}")

            try:
                results = scraper_fn()
                all_results.extend(results)
                log(f"\n{shop_name}: {len(results)} prices collected")
                save_progress(shop_name, all_results)
            except Exception as e:
                log(f"\n{shop_name}: ERROR — {e}")
                traceback.print_exc(file=sys.stderr)

    # -----------------------------------------------------------------------
    # Merge existing data
    # -----------------------------------------------------------------------
    log(f"\n{'='*60}")
    log("MERGING EXISTING DATA")
    log(f"{'='*60}")

    existing = load_existing_data()
    live_keys = set(
        (r["shop"], r["brand"], r.get("model", ""), r["size"])
        for r in all_results
    )

    merged_count = 0
    for entry in existing:
        key = (entry["shop"], entry["brand"], entry.get("model", ""), entry.get("size", ""))
        if key not in live_keys:
            all_results.append(entry)
            live_keys.add(key)
            merged_count += 1

    log(f"Merged {merged_count} entries from existing data")

    # -----------------------------------------------------------------------
    # Clean and deduplicate
    # -----------------------------------------------------------------------
    log(f"\n{'='*60}")
    log("CLEANING & DEDUPLICATION")
    log(f"{'='*60}")

    log(f"Raw entries: {len(all_results)}")
    all_results = clean_results(all_results)
    log(f"After cleaning: {len(all_results)}")
    all_results = deduplicate(all_results)
    log(f"After dedup: {len(all_results)}")

    # Sort by shop, brand, model, size
    all_results.sort(key=lambda r: (r["shop"], r["brand"], r["model"], r["size"]))

    # Save final output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"Total entries: {len(all_results)}")
    log(f"Time elapsed: {elapsed/60:.1f} minutes")

    shops = {}
    for r in all_results:
        shops.setdefault(r["shop"], 0)
        shops[r["shop"]] += 1
    log(f"\nBy shop:")
    for shop, count in sorted(shops.items(), key=lambda x: -x[1]):
        log(f"  {shop}: {count}")

    brands = {}
    for r in all_results:
        brands.setdefault(r["brand"], 0)
        brands[r["brand"]] += 1
    log(f"\nBy brand:")
    for brand, count in sorted(brands.items(), key=lambda x: -x[1]):
        log(f"  {brand}: {count}")

    unique_sizes = len(set(r["size"] for r in all_results))
    log(f"\nUnique sizes covered: {unique_sizes}")
    log(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
