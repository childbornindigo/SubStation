#!/usr/bin/env python3
"""
Scrape TDot Performance for ALL 25 target tire models.
TDot returns clean structured data via ScrapLing.
"""

import os
import sys
import json
import re
import time
import urllib.parse

os.environ['FIRECRAWL_API_KEY'] = ''  # Skip firecrawl (credits exhausted)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrape_utils import _try_scrapling

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPTS_DIR, "local-prices-tdot-2026-05-18.json")
PROGRESS_FILE = os.path.join(SCRIPTS_DIR, "tdot-scrape-progress.json")

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

SIZE_PATTERN = re.compile(r'(?:P|LT)?(\d{3}/\d{2,3}R\d{2})', re.IGNORECASE)
PRICE_PATTERN = re.compile(r'C?\$(\d{2,4}(?:\.\d{2})?)')


def parse_tdot_listings(content, brand, model):
    """
    Parse TDot Performance search results.
    Structure per product:
      1. ![image] with title containing size
      2. [Product Name - Model (SIZE SPEC)]
      3. * Tire Size: XXX/XXRXX
      4. C$NNN.NN
    """
    results = []
    lines = content.split('\n')

    current_size = None
    current_price = None
    in_product = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for tire size in bullet list
        if '* Tire Size:' in stripped:
            m = SIZE_PATTERN.search(stripped)
            if m:
                current_size = m.group(1).upper()
                in_product = True
                continue

        # Check for size in product title line (backup)
        if not current_size and ('[' in stripped and model.split()[0].lower() in stripped.lower()):
            m = SIZE_PATTERN.search(stripped)
            if m:
                current_size = m.group(1).upper()
                in_product = True

        # Check for price
        if in_product and current_size:
            m = PRICE_PATTERN.search(stripped)
            if m:
                price = float(m.group(1))
                if 50 <= price <= 1500:
                    results.append({
                        "shop": "TDot Performance",
                        "brand": brand,
                        "model": model,
                        "size": current_size,
                        "price": price
                    })
                    current_size = None
                    current_price = None
                    in_product = False

        # Reset on new product (numbered list item)
        if re.match(r'^\d+\.', stripped) and in_product:
            current_size = None
            in_product = False

    # Deduplicate
    seen = set()
    deduped = []
    for r in results:
        key = (r["size"], r["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def scrape_model(brand, model, progress):
    """Scrape a single model from TDot with pagination."""
    task_key = f"{brand}:{model}"
    if task_key in progress["completed"]:
        existing = [r for r in progress["results"] if r["brand"] == brand and r["model"] == model]
        print(f"  [skip] {brand} {model} - already done ({len(existing)} entries)")
        return existing

    query = f"{brand} {model}"
    encoded = urllib.parse.quote_plus(query)

    all_entries = []

    # Page 1
    url = f"https://www.tdotperformance.ca/catalogsearch/result/?q={encoded}"
    print(f"  [scrape] {brand} {model} (page 1)")
    result = _try_scrapling(url, timeout=60000)

    if not result["success"]:
        print(f"    FAILED: {result.get('error', 'unknown')}")
        progress["completed"].append(task_key)
        save_progress(progress)
        return []

    entries = parse_tdot_listings(result["content"], brand, model)
    all_entries.extend(entries)
    print(f"    Page 1: {len(entries)} entries found")

    # Check if there are more pages (look for page links)
    content = result["content"]
    # TDot pagination: look for "Page 2", "Next" links, or p=2 params
    page_links = re.findall(r'p=(\d+)', content)
    max_page = 1
    for p in page_links:
        try:
            pn = int(p)
            if pn > max_page:
                max_page = pn
        except ValueError:
            pass

    # Scrape additional pages (up to 5 to be safe)
    for page in range(2, min(max_page + 1, 6)):
        time.sleep(2)
        page_url = f"{url}&p={page}"
        print(f"  [scrape] {brand} {model} (page {page})")
        result = _try_scrapling(page_url, timeout=60000)

        if not result["success"]:
            print(f"    Page {page} FAILED, stopping pagination")
            break

        entries = parse_tdot_listings(result["content"], brand, model)
        if not entries:
            print(f"    Page {page}: no entries, stopping pagination")
            break

        all_entries.extend(entries)
        print(f"    Page {page}: {len(entries)} entries found")

    # Deduplicate
    seen = set()
    deduped = []
    for r in all_entries:
        key = (r["size"], r["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    progress["results"].extend(deduped)
    progress["completed"].append(task_key)
    save_progress(progress)

    return deduped


def main():
    print("=" * 60)
    print("TDot Performance - Full Model Scrape")
    print(f"Target: {len(TARGET_MODELS)} models")
    print("=" * 60)

    progress = load_progress()
    print(f"Progress: {len(progress['completed'])} done, {len(progress['results'])} entries")

    all_results = []

    for brand, model in TARGET_MODELS:
        entries = scrape_model(brand, model, progress)
        all_results.extend(entries)
        time.sleep(3)  # Rate limiting between models

    # Deduplicate final results
    seen = set()
    deduped = []
    for r in all_results:
        key = (r["shop"], r["brand"], r["model"], r["size"], r["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(deduped, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(deduped)} entries saved to {OUTPUT_FILE}")

    # Summary by model
    by_model = {}
    for r in deduped:
        key = f"{r['brand']} {r['model']}"
        by_model.setdefault(key, 0)
        by_model[key] += 1

    for model, count in sorted(by_model.items()):
        print(f"  {model}: {count} sizes")

    print("=" * 60)


if __name__ == "__main__":
    main()
