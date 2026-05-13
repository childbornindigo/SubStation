#!/usr/bin/env python3
"""
tire_scraper.py — Tire market intelligence scraper for Canadian tire business.

Uses Google Shopping as primary data source (aggregates across retailers),
with Google Trends for brand popularity and marketplace demand signals.

Usage:
    python tire_scraper.py              # full run
    python tire_scraper.py --test       # test run (5 sizes)
    python tire_scraper.py --sizes 10   # partial run
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from scrape_utils import smart_scrape

OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "market-intelligence")

SIZES = [
    "195/65R15", "205/55R16", "215/60R16", "225/65R17", "235/65R18",
    "245/75R16", "265/70R17", "275/55R20", "215/55R17", "225/60R18",
    "205/65R16", "215/65R17", "225/55R17", "235/55R19", "245/45R18",
    "255/70R18", "265/65R18", "275/60R20", "285/45R22", "195/60R15",
    "205/60R16", "215/45R17", "225/45R17", "235/45R18", "245/40R18",
    "255/55R18", "265/60R18", "275/45R20", "285/65R18", "195/55R16",
    "205/50R17", "215/50R17", "225/40R18", "235/40R19", "245/35R20",
    "255/45R19", "265/50R20", "275/40R20", "285/70R17", "195/50R16",
    "205/45R17", "215/70R16", "225/50R18", "235/70R16", "245/65R17",
    "255/65R18", "265/70R16", "275/65R18", "285/75R16", "175/65R15",
    "185/65R15", "185/60R15", "225/45R18", "235/55R18", "245/60R18",
    "255/50R19", "265/75R16", "275/55R19", "225/60R17", "235/60R18",
    "245/55R19", "255/35R19", "265/60R20", "275/70R18", "225/50R17",
]

KNOWN_BRANDS = [
    "Michelin", "Bridgestone", "Goodyear", "Continental", "Pirelli",
    "Nokian", "Toyo", "Yokohama", "Hankook", "Kumho",
    "Falken", "General", "Cooper", "BFGoodrich", "Firestone",
    "Dunlop", "Nexen", "Sailun", "GT Radial", "Gislaved",
    "Uniroyal", "Maxxis", "Nitto", "Federal", "Achilles",
    "Kenda", "Ironman", "Westlake", "Triangle", "Zeetex",
]

_BRAND_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in KNOWN_BRANDS) + r")\b",
    re.IGNORECASE,
)

_TYPE_KEYWORDS = {
    "winter": ["winter", "ice", "snow", "blizzak", "wintercontact", "x-ice", "hakkapeliitta", "w-speed"],
    "all-season": ["all-season", "all season", "a/s", "4-season", "four season", "defender", "assurance", "all weather"],
    "performance": ["performance", "sport", "pilot sport", "potenza", "p zero", "ultra high performance", "uhp"],
}

TOP_BRANDS_FOR_TRENDS = [
    "Michelin", "Bridgestone", "Goodyear", "Continental", "Pirelli",
    "Nokian", "Toyo", "Yokohama", "Hankook", "Cooper",
    "BFGoodrich", "Firestone", "Dunlop", "Falken", "Kumho",
]

REQUEST_DELAY = 6.0  # Bing is lenient but stay respectful


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _detect_tire_type(text: str) -> str:
    lower = text.lower()
    for tire_type, keywords in _TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return tire_type
    return "unknown"


def _normalize_brand(brand_text: str) -> str:
    for known in KNOWN_BRANDS:
        if known.lower() == brand_text.lower():
            return known
    return brand_text


def _extract_google_shopping_listings(content: str, target_size: str) -> list[dict]:
    """Extract tire listings from Google Shopping results."""
    listings = []
    if not content:
        return listings

    # Google Shopping format: product name line, then $price, then retailer
    # Split content into product blocks — each product is near a price
    price_pattern = re.compile(r'\$(\d{2,4}\.\d{2})')
    seen = set()

    for m in price_pattern.finditer(content):
        price = float(m.group(1))
        if price < 50 or price > 800:
            continue

        # Get context: 500 chars before price, 200 after
        start = max(0, m.start() - 500)
        end = min(len(content), m.end() + 200)
        block = content[start:end]

        # Find brand in this block
        brand_match = _BRAND_PATTERN.search(block)
        if not brand_match:
            continue

        brand = _normalize_brand(brand_match.group(1))

        # Extract product name — look for full product title line near brand
        # It's usually: "Brand Model Details Size"
        lines = block.split('\n')
        product_line = ""
        for line in lines:
            stripped = line.strip()
            if brand.lower() in stripped.lower() and len(stripped) > 15 and len(stripped) < 200:
                if '$' not in stripped and 'http' not in stripped.lower():
                    product_line = stripped
                    break

        # Extract model from product line
        model = "Unknown Model"
        if product_line:
            # Remove brand name to get model
            model_part = re.sub(re.escape(brand), '', product_line, flags=re.IGNORECASE).strip()
            # Clean up
            model_part = re.sub(r'\(.*?\)', '', model_part)  # remove parenthetical
            model_part = re.sub(r'\d{3}/\d{2,3}\s*R?\d{2}', '', model_part)  # remove size
            model_part = re.sub(r'\d{2,3}[HTVWYZ]\b', '', model_part)  # remove speed rating
            model_part = re.sub(r'\bXL\b|\bBSW\b|\bOWL\b', '', model_part)
            model_part = re.sub(r'\s+', ' ', model_part).strip(' -–—/\\,.()')
            if model_part and len(model_part) > 2:
                model = model_part[:60]

        # Extract retailer — usually appears on line after price
        retailer = "unknown"
        price_line_found = False
        for line in lines:
            if f"${m.group(1)}" in line:
                price_line_found = True
                continue
            if price_line_found:
                stripped = line.strip()
                # Skip review ratings: "4.5(373)", "4.5", "4.0 · 91"
                if re.match(r'^\d\.\d', stripped):
                    continue
                # Skip pure numbers and short noise
                if re.match(r'^[\d·\s]+$', stripped) or len(stripped) < 3:
                    continue
                if stripped and len(stripped) < 50 and '$' not in stripped:
                    retailer = re.sub(r'\s*[&\xa0]+\s*more$', '', stripped).strip()
                    break

        tire_type = _detect_tire_type(block)

        # Size from product text or target
        size_match = re.search(r'(\d{3}/\d{2,3}\s*R?\s*\d{2})', block)
        listing_size = size_match.group(1).replace(' ', '') if size_match else target_size

        key = f"{brand}|{model}|{price}"
        if key in seen:
            continue
        seen.add(key)

        listings.append({
            "brand": brand,
            "model": model,
            "price": price,
            "type": tire_type,
            "size": listing_size,
            "retailer": retailer,
            "source": "bing_shopping",
        })

    return listings


def _extract_listing_count(markdown: str) -> int | None:
    if not markdown:
        return None
    patterns = [
        r"([\d,]+)\s*(?:results?|listings?|ads?|items?)",
        r"of\s+([\d,]+)",
        r"([\d,]+)\s*(?:annonces?|résultats?)",
    ]
    for pat in patterns:
        m = re.search(pat, markdown, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _extract_trends_score(markdown: str) -> int | None:
    if not markdown:
        return None
    m = re.search(r"interest over time.*?(\d{1,3})", markdown, re.IGNORECASE | re.DOTALL)
    if m:
        score = int(m.group(1))
        if 0 <= score <= 100:
            return score
    return None


def scrape_retail_prices(sizes: list[str], scrape_log: list) -> dict:
    """Phase 1: Google Shopping for tire prices across sizes."""
    _log(f"\n{'='*60}")
    _log(f"PHASE 1: Google Shopping — Retail Prices")
    _log(f"  Sizes: {len(sizes)}")
    _log(f"{'='*60}")

    results = {}
    captcha_backoff = 0
    for idx, size in enumerate(sizes, 1):
        url = f"https://www.bing.com/shop?q={size}+tires+canada"
        _log(f"  [{idx}/{len(sizes)}] Shopping: {size}...")

        if captcha_backoff > 0:
            _log(f"    (CAPTCHA cooldown: waiting {captcha_backoff}s)")
            time.sleep(captcha_backoff)

        try:
            result = smart_scrape(url, stealth=True)
        except Exception as e:
            _log(f"    ERROR: {e}")
            scrape_log.append({"url": url, "source": "bing_shopping", "size": size, "phase": "retail", "success": False, "engine": "unknown", "error": str(e)})
            time.sleep(REQUEST_DELAY + random.uniform(0, 5))
            continue

        # CAPTCHA detection
        is_captcha = False
        if result["success"] and result["content"]:
            lower_content = result["content"].lower()
            if 'captcha' in lower_content or 'unusual traffic' in lower_content or ('sorry' in lower_content[:300] and len(result["content"]) < 3000):
                is_captcha = True
                captcha_backoff = min(captcha_backoff + 60, 180)
                _log(f"    -> CAPTCHA detected! Backing off {captcha_backoff}s")
                scrape_log.append({"url": url, "source": "bing_shopping", "size": size, "phase": "retail", "success": False, "engine": result["engine"], "error": "CAPTCHA"})
                results[size] = {"listings": []}
                time.sleep(REQUEST_DELAY + random.uniform(0, 5))
                continue

        if not is_captcha and captcha_backoff > 0:
            captcha_backoff = max(0, captcha_backoff - 30)

        scrape_log.append({"url": url, "source": "bing_shopping", "size": size, "phase": "retail", "success": result["success"], "engine": result["engine"], "error": result.get("error")})

        listings = []
        if result["success"]:
            listings = _extract_google_shopping_listings(result["content"], size)
            _log(f"    -> {len(listings)} listings extracted")
        else:
            _log(f"    -> FAILED: {result.get('error', 'unknown')}")

        results[size] = {"listings": listings}
        time.sleep(REQUEST_DELAY + random.uniform(0, 5))

    total = sum(len(v["listings"]) for v in results.values())
    sizes_with_data = sum(1 for v in results.values() if v["listings"])
    _log(f"\n  Retail complete: {total} listings across {sizes_with_data}/{len(sizes)} sizes")
    return results


def scrape_marketplace_demand(sizes: list[str], scrape_log: list) -> dict:
    """Phase 2: Kijiji + eBay demand signals."""
    _log(f"\n{'='*60}")
    _log(f"PHASE 2: Marketplace Demand (Kijiji + eBay)")
    _log(f"  Sizes: {len(sizes)}")
    _log(f"{'='*60}")

    # Use Google search to find listing counts (more reliable than scraping Kijiji directly)
    results = {}
    for idx, size in enumerate(sizes, 1):
        results[size] = {}
        for mp in ["kijiji", "ebay"]:
            if mp == "kijiji":
                url = f"https://www.bing.com/search?q=site:kijiji.ca+{size}+tires"
            else:
                url = f"https://www.bing.com/search?q=site:ebay.ca+{size}+tires"

            _log(f"  [{idx}/{len(sizes)}] {mp}: {size}...")

            try:
                result = smart_scrape(url, stealth=True)
            except Exception as e:
                _log(f"    ERROR: {e}")
                scrape_log.append({"url": url, "source": mp, "size": size, "phase": "marketplace", "success": False, "engine": "unknown", "error": str(e)})
                time.sleep(REQUEST_DELAY + random.uniform(0, 5))
                continue

            scrape_log.append({"url": url, "source": mp, "size": size, "phase": "marketplace", "success": result["success"], "engine": result["engine"], "error": result.get("error")})

            count = None
            if result["success"]:
                count = _extract_listing_count(result["content"])
                _log(f"    -> count: {count if count is not None else 'could not extract'}")
            else:
                _log(f"    -> FAILED")

            results[size][mp] = {"listing_count": count, "scraped": result["success"]}
            time.sleep(REQUEST_DELAY + random.uniform(0, 5))

    return results


def scrape_google_trends(sizes: list[str], scrape_log: list) -> dict:
    """Phase 3: Google Trends for size + brand popularity."""
    _log(f"\n{'='*60}")
    _log(f"PHASE 3: Google Trends")
    _log(f"  Sizes: {len(sizes)} | Brands: {len(TOP_BRANDS_FOR_TRENDS)}")
    _log(f"{'='*60}")

    results = {"sizes": {}, "brands": {}}

    for idx, size in enumerate(sizes, 1):
        url = f"https://trends.google.com/trends/explore?q={size}+tires&geo=CA"
        _log(f"  [{idx}/{len(sizes)}] Size trend: {size}...")
        try:
            result = smart_scrape(url, stealth=True)
        except Exception as e:
            scrape_log.append({"url": url, "source": "google_trends", "query": size, "phase": "trends", "success": False, "engine": "unknown", "error": str(e)})
            time.sleep(REQUEST_DELAY + random.uniform(0, 5))
            continue
        scrape_log.append({"url": url, "source": "google_trends", "query": size, "phase": "trends", "success": result["success"], "engine": result["engine"], "error": result.get("error")})
        score = _extract_trends_score(result["content"]) if result["success"] else None
        _log(f"    -> {score if score is not None else 'no score'}")
        results["sizes"][size] = {"interest_score": score, "scraped": result["success"]}
        time.sleep(REQUEST_DELAY + random.uniform(0, 5))

    for idx, brand in enumerate(TOP_BRANDS_FOR_TRENDS, 1):
        url = f"https://trends.google.com/trends/explore?q={brand}+tires+canada&geo=CA"
        _log(f"  [{idx}/{len(TOP_BRANDS_FOR_TRENDS)}] Brand trend: {brand}...")
        try:
            result = smart_scrape(url, stealth=True)
        except Exception as e:
            scrape_log.append({"url": url, "source": "google_trends", "query": brand, "phase": "trends_brands", "success": False, "engine": "unknown", "error": str(e)})
            time.sleep(REQUEST_DELAY + random.uniform(0, 5))
            continue
        scrape_log.append({"url": url, "source": "google_trends", "query": brand, "phase": "trends_brands", "success": result["success"], "engine": result["engine"], "error": result.get("error")})
        score = _extract_trends_score(result["content"]) if result["success"] else None
        _log(f"    -> {score if score is not None else 'no score'}")
        results["brands"][brand] = {"interest_score": score, "scraped": result["success"]}
        time.sleep(REQUEST_DELAY + random.uniform(0, 5))

    return results


def _save_json(data: dict, filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _log(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Tire market intelligence scraper")
    parser.add_argument("--test", action="store_true", help="Test run: 5 sizes only")
    parser.add_argument("--sizes", type=int, default=None, help="Limit number of sizes")
    parser.add_argument("--skip-marketplace", action="store_true", help="Skip marketplace demand")
    parser.add_argument("--skip-trends", action="store_true", help="Skip Google Trends")
    args = parser.parse_args()

    if args.test:
        sizes = SIZES[:5]
        _log("MODE: Test run (5 sizes)")
    else:
        limit = args.sizes if args.sizes else len(SIZES)
        sizes = SIZES[:limit]
        _log(f"MODE: {'Partial' if args.sizes else 'Full'} run ({len(sizes)} sizes)")

    demand_sizes = sizes[:20]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scrape_log = []
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _log(f"\nTire Market Intelligence Scraper")
    _log(f"Started: {timestamp}")
    _log(f"Output:  {OUTPUT_DIR}")

    retail_data = scrape_retail_prices(sizes, scrape_log)
    _save_json({"scraped_at": timestamp, "sizes": retail_data}, "retail-data.json")

    if not args.skip_marketplace:
        mp_data = scrape_marketplace_demand(demand_sizes, scrape_log)
        _save_json({"scraped_at": timestamp, "sizes": mp_data}, "marketplace-demand.json")
    else:
        _log("\n  Skipping marketplace demand")

    if not args.skip_trends:
        trends_data = scrape_google_trends(demand_sizes, scrape_log)
        _save_json({"scraped_at": timestamp, **trends_data}, "trends-data.json")
    else:
        _log("\n  Skipping Google Trends")

    log_summary = {
        "scraped_at": timestamp,
        "total_requests": len(scrape_log),
        "successful": sum(1 for e in scrape_log if e["success"]),
        "failed": sum(1 for e in scrape_log if not e["success"]),
        "entries": scrape_log,
    }
    _save_json(log_summary, "scrape-log.json")

    _log(f"\n{'='*60}")
    _log(f"SCRAPE COMPLETE")
    _log(f"  Total requests:  {log_summary['total_requests']}")
    _log(f"  Successful:      {log_summary['successful']}")
    _log(f"  Failed:          {log_summary['failed']}")
    _log(f"  Output dir:      {OUTPUT_DIR}")
    _log(f"{'='*60}\n")


if __name__ == "__main__":
    main()
