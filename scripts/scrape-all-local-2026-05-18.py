#!/usr/bin/env python3
"""
Scrape local GTA tire shop prices for our 25 tire models.
Focus on shops with gaps in existing data.
Uses Firecrawl first, ScrapLing as fallback.
"""

import os
import sys
import json
import re
import time
import urllib.parse

# Set up the env
os.environ["FIRECRAWL_API_KEY"] = "fc-2d1cf8b9d6e84fc797d360879036f290"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrape_utils import smart_scrape

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local-prices-all-shops-2026-05-18.json")
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrape-progress-2026-05-18.json")

# Our 25 target models
TARGET_MODELS = [
    ("Bridgestone", "Alenza Sport AS"),
    ("Bridgestone", "Turanza Everdrive"),
    ("Bridgestone", "UltraWeather"),
    ("Bridgestone", "WeatherPeak"),
    ("Continental", "CrossContact LX Sport"),
    ("Continental", "CrossContact RX"),
    ("Continental", "ExtremeContact DWS06 PLUS"),
    ("Continental", "ProContact RX"),
    ("Falken", "Azenis FK460 AS"),
    ("Firestone", "Destination LE 2"),
    ("Firestone", "Destination LE3"),
    ("Firestone", "FR710"),
    ("Firestone", "Weathergrip"),
    ("Hankook", "Kinergy GT"),
    ("Hankook", "Ventus S1 noble2"),
    ("Hankook", "Dynapro evo AS"),
    ("Michelin", "Defender LTX M/S2"),
    ("Michelin", "Pilot Sport AS 4"),
    ("Michelin", "Primacy Tour AS"),
    ("Pirelli", "P Zero AS Plus 3"),
    ("Pirelli", "Scorpion Zero All Season"),
    ("Toyo", "Extensa AS II"),
    ("Toyo", "OPA50"),
    ("Yokohama", "Geolandar X-AT G016"),
    ("Yokohama", "Geolandar X-CV"),
]

# Size pattern: matches things like 205/45R17, 265/70R18, 315/40R21, P245/65R17, LT275/65R18
SIZE_PATTERN = re.compile(r'(?:P|LT)?(\d{3}/\d{2,3}[RZV]?\d{2})', re.IGNORECASE)
# Also match sizes with load/speed like "205/45R17 88W" or "205/45R17 XL 92W"
SIZE_FULL_PATTERN = re.compile(r'(?:P|LT)?(\d{3}/\d{2,3}R\d{2})\s*(?:XL\s*)?(?:\d{2,3}[A-Z])?', re.IGNORECASE)
# Price pattern: $249.44 or 249.44
PRICE_PATTERN = re.compile(r'\$\s?(\d{2,4}(?:\.\d{2})?)')


def extract_size(text):
    """Extract standardized tire size from text."""
    m = SIZE_PATTERN.search(text)
    if m:
        size = m.group(1).upper()
        # Normalize: ensure R is present
        if 'R' not in size and 'Z' not in size and 'V' not in size:
            return None
        return size
    return None


def extract_price(text):
    """Extract price from text. Returns float or None."""
    m = PRICE_PATTERN.search(text)
    if m:
        price = float(m.group(1))
        # Sanity check: tire prices should be $50-$1500
        if 50 <= price <= 1500:
            return price
    return None


def parse_tire_listings(markdown, brand, model, shop_name):
    """Parse markdown from a tire listing page into structured entries."""
    results = []
    if not markdown:
        return results

    lines = markdown.split('\n')

    # Strategy 1: Look for lines that have BOTH a size and a price
    for i, line in enumerate(lines):
        size = extract_size(line)
        if size:
            # Look for price in this line and surrounding lines
            price = extract_price(line)
            if not price and i + 1 < len(lines):
                price = extract_price(lines[i + 1])
            if not price and i + 2 < len(lines):
                price = extract_price(lines[i + 2])
            if not price and i - 1 >= 0:
                price = extract_price(lines[i - 1])

            if price:
                results.append({
                    "shop": shop_name,
                    "brand": brand,
                    "model": model,
                    "size": size,
                    "price": price
                })

    # Strategy 2: Look for table-formatted data
    if not results:
        # Look for markdown table rows with size and price
        for line in lines:
            if '|' in line:
                cells = [c.strip() for c in line.split('|')]
                size_found = None
                price_found = None
                for cell in cells:
                    if not size_found:
                        size_found = extract_size(cell)
                    if not price_found:
                        price_found = extract_price(cell)
                if size_found and price_found:
                    results.append({
                        "shop": shop_name,
                        "brand": brand,
                        "model": model,
                        "size": size_found,
                        "price": price_found
                    })

    # Strategy 3: Block parsing - look for blocks separated by blank lines or headers
    if not results:
        blocks = re.split(r'\n{2,}|(?=^#{1,3}\s)', markdown, flags=re.MULTILINE)
        for block in blocks:
            size = extract_size(block)
            price = extract_price(block)
            if size and price:
                results.append({
                    "shop": shop_name,
                    "brand": brand,
                    "model": model,
                    "size": size,
                    "price": price
                })

    # Deduplicate
    seen = set()
    deduped = []
    for r in results:
        key = (r["size"], r["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def build_search_url(shop_key, brand, model):
    """Build search/product URL for a given shop and tire model."""
    query = f"{brand} {model}"
    encoded = urllib.parse.quote_plus(query)

    urls = {
        "canadian_tire": f"https://www.canadiantire.ca/en/search?q={encoded}&categoryId=CT10510",
        "tdot": f"https://www.tdotperformance.ca/catalogsearch/result/?q={encoded}",
        "agr": f"https://www.activegreenross.com/tires?search={encoded}",
        "point_s": f"https://www.point-s.ca/en/tires?search={encoded}",
        "krave": f"https://kraveauto.ca/?s={encoded}&post_type=product",
        "noble": f"https://nobletire.ca/?s={encoded}&post_type=product",
        "fas_tire": f"https://fas-tire.ca/?s={encoded}&post_type=product",
        "zracing": f"https://zracing.ca/search?q={encoded}&type=product",
    }
    return urls.get(shop_key)


def build_direct_url(shop_key, brand, model):
    """Build direct product page URLs for specific known URL patterns."""
    # Slug-ify the model name
    slug = f"{brand}-{model}".lower().replace(' ', '-').replace('/', '-').replace('+', 'plus')

    urls = {
        "canadian_tire": None,  # CT uses search, no predictable slugs
        "tdot": None,  # Complex product URLs
        "agr": None,
        "point_s": None,
        "krave": None,
        "noble": None,
        "fas_tire": None,
        "zracing": None,
    }
    return urls.get(shop_key)


# Shop configs: which shops need scraping and for which models
SHOP_CONFIGS = {
    "agr": {
        "name": "Active Green + Ross",
        "priority": 1,  # Only 2/25 models - highest priority
        "models": "all",
    },
    "fas_tire": {
        "name": "Fas-Tire Scarborough",
        "priority": 1,  # 0/25 models
        "models": "all",
    },
    "krave": {
        "name": "KRAVE Automotive",
        "priority": 2,  # 2/25 models
        "models": "all",
    },
    "noble": {
        "name": "Noble Tire Brampton",
        "priority": 2,  # 8/25 models
        "models": "all",
    },
    "tdot": {
        "name": "TDot Performance",
        "priority": 3,  # 12/25 - fill gaps
        "missing_models": [
            ("Bridgestone", "UltraWeather"),
            ("Bridgestone", "WeatherPeak"),
            ("Continental", "CrossContact LX Sport"),
            ("Continental", "CrossContact RX"),
            ("Continental", "ExtremeContact DWS06 PLUS"),
            ("Continental", "ProContact RX"),
            ("Falken", "Azenis FK460 AS"),
            ("Firestone", "FR710"),
            ("Hankook", "Dynapro evo AS"),
            ("Pirelli", "P Zero AS Plus 3"),
            ("Pirelli", "Scorpion Zero All Season"),
            ("Toyo", "OPA50"),
            ("Yokohama", "Geolandar X-AT G016"),
        ],
    },
    "canadian_tire": {
        "name": "Canadian Tire",
        "priority": 3,  # 17/25 - fill gaps
        "missing_models": [
            ("Bridgestone", "Turanza Everdrive"),
            ("Bridgestone", "UltraWeather"),
            ("Continental", "ProContact RX"),
            ("Falken", "Azenis FK460 AS"),
            ("Firestone", "FR710"),
            ("Hankook", "Dynapro evo AS"),
            ("Toyo", "Extensa AS II"),
            ("Toyo", "OPA50"),
        ],
    },
}


def load_progress():
    """Load progress from disk."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_progress(progress):
    """Save progress to disk."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def scrape_shop(shop_key, config, progress):
    """Scrape a single shop for all target models."""
    shop_name = config["name"]

    # Determine which models to scrape
    if config.get("models") == "all":
        models_to_scrape = TARGET_MODELS
    else:
        models_to_scrape = config.get("missing_models", TARGET_MODELS)

    all_results = []

    for brand, model in models_to_scrape:
        task_key = f"{shop_key}:{brand}:{model}"

        if task_key in progress["completed"]:
            print(f"  [skip] Already done: {brand} {model}")
            continue

        url = build_search_url(shop_key, brand, model)
        if not url:
            print(f"  [skip] No URL for {shop_key}: {brand} {model}")
            progress["completed"].append(task_key)
            save_progress(progress)
            continue

        print(f"  [scrape] {brand} {model} -> {url}")

        result = smart_scrape(url, timeout=45000)

        if result["success"] and result["content"]:
            entries = parse_tire_listings(result["content"], brand, model, shop_name)
            if entries:
                print(f"    Found {len(entries)} size/price entries via {result['engine']}")
                all_results.extend(entries)
                progress["results"].extend(entries)
            else:
                print(f"    Got content ({len(result['content'])} chars) but no parseable prices")
                # Save a snippet for debugging
                snippet = result["content"][:500].replace('\n', ' ')
                print(f"    Snippet: {snippet[:200]}...")
        else:
            print(f"    Failed: {result.get('error', 'unknown')}")

        progress["completed"].append(task_key)
        save_progress(progress)

        # Rate limiting
        time.sleep(3)

    return all_results


def main():
    print("=" * 60)
    print("GTA Local Tire Shop Price Scraper")
    print(f"Date: 2026-05-18")
    print(f"Target: 25 models across {len(SHOP_CONFIGS)} shops")
    print("=" * 60)

    progress = load_progress()
    print(f"Loaded progress: {len(progress['completed'])} tasks done, {len(progress['results'])} results so far")

    all_results = list(progress["results"])

    # Sort shops by priority (lowest number = highest priority)
    sorted_shops = sorted(SHOP_CONFIGS.items(), key=lambda x: x[1]["priority"])

    for shop_key, config in sorted_shops:
        shop_name = config["name"]
        models = TARGET_MODELS if config.get("models") == "all" else config.get("missing_models", [])

        # Check how many are already done
        done_count = sum(1 for b, m in models if f"{shop_key}:{b}:{m}" in progress["completed"])
        remaining = len(models) - done_count

        if remaining == 0:
            print(f"\n[{shop_name}] All {len(models)} models already scraped, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{shop_name}] Scraping {remaining} remaining models (priority {config['priority']})")
        print(f"{'=' * 60}")

        new_results = scrape_shop(shop_key, config, progress)
        all_results.extend(new_results)

        print(f"  => {len(new_results)} new entries from {shop_name}")

    # Deduplicate all results
    seen = set()
    deduped = []
    for r in all_results:
        key = (r["shop"], r["brand"], r["model"], r["size"], r["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Save final output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(deduped, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE: {len(deduped)} total entries saved to {OUTPUT_FILE}")

    # Summary by shop
    by_shop = {}
    for r in deduped:
        by_shop.setdefault(r["shop"], []).append(r)
    for shop, entries in sorted(by_shop.items()):
        models = set(f"{e['brand']} {e['model']}" for e in entries)
        print(f"  {shop}: {len(entries)} entries, {len(models)} models")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
