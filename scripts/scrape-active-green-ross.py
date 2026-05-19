#!/usr/bin/env python3
"""
Scrape Active Green + Ross tire catalog pages for pricing data.
Fetches all 4 category pages (passenger, performance, lighttruck, evc),
parses brand/model/size/price, and filters for our target brands only.

Page structure per tire block:
  #### [ModelName](https://...activegreenross.com/tire-selector/<brand_slug>/id-model)
  P205/55R16              (size — standalone line)
  ...
  **Manufacturer:** Michelin
  ...
  $229.95 / tire...       (price line)
  [Tire Details](...)
"""
import json
import os
import re
import sys
import time

os.environ["FIRECRAWL_API_KEY"] = "fc-2d1cf8b9d6e84fc797d360879036f290"
sys.path.insert(0, "/Users/indigochild/.hermes/extensions/substation/scripts")
from scrape_utils import smart_scrape

SHOP = "Active Green + Ross"
OUTPUT = "/Users/indigochild/.hermes/extensions/substation/scripts/local-prices-active-green-ross.json"

TARGET_BRANDS = {
    "bridgestone", "continental", "falken", "firestone",
    "hankook", "michelin", "pirelli", "toyo", "yokohama"
}

BRAND_DISPLAY = {
    "bridgestone": "Bridgestone",
    "continental": "Continental",
    "falken": "Falken",
    "firestone": "Firestone",
    "hankook": "Hankook",
    "michelin": "Michelin",
    "pirelli": "Pirelli",
    "toyo": "Toyo",
    "yokohama": "Yokohama",
}

CATEGORY_URLS = [
    "https://www.activegreenross.com/tire-selector/view-all-tires?type=passenger",
    "https://www.activegreenross.com/tire-selector/view-all-tires?type=performance",
    "https://www.activegreenross.com/tire-selector/view-all-tires?type=lighttruck",
    "https://www.activegreenross.com/tire-selector/view-all-tires?type=evc",
]

# Heading: #### [ModelName](url) — brand slug is in the URL path
HEADING_RE = re.compile(
    r'^####\s+\[(.+?)\]\(https://www\.activegreenross\.com/tire-selector/([^/]+)/',
    re.IGNORECASE
)
# Bare size line: P205/55R16, LT265/70R17, 225/45ZR17, etc.
SIZE_LINE_RE = re.compile(
    r'^\s*(?:P|LT|ST|C)?(\d{3}/\d{2}[A-Z]{0,2}\d{2})\s*$',
    re.IGNORECASE
)
# Manufacturer detail line
MFGR_RE = re.compile(r'\*\*Manufacturer:\*\*\s*(.+)', re.IGNORECASE)
# Price line: starts with $NNN.NN / tire
PRICE_RE = re.compile(r'^\$\s*([\d,]+\.\d{2})\s*/', re.IGNORECASE)


def parse_entries(markdown: str) -> list[dict]:
    """
    Parse tire entries. We scan through blocks anchored by #### headings.
    Within each block we collect: model (from heading text), brand_slug (from URL),
    size (standalone size line), manufacturer (from Manufacturer: line), price (from $ line).
    """
    results = []
    lines = markdown.split("\n")
    n = len(lines)

    # State for the current block
    current_model = None
    current_brand_slug = None   # from URL in heading
    current_brand_mfgr = None   # from **Manufacturer:** line
    current_size = None

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        # --- Heading: start of a new tire block ---
        m = HEADING_RE.match(line)
        if m:
            current_model = m.group(1).strip()
            current_brand_slug = m.group(2).lower().strip()
            # Reset per-block fields
            current_brand_mfgr = None
            current_size = None
            continue

        # --- Size line ---
        ms = SIZE_LINE_RE.match(line)
        if ms:
            # Only capture first size per block (if we already have one, keep it)
            if current_model and current_size is None:
                current_size = ms.group(1).upper()
            continue

        # --- Manufacturer line ---
        mm = MFGR_RE.search(line)
        if mm:
            current_brand_mfgr = mm.group(1).strip()
            continue

        # --- Price line ---
        mp = PRICE_RE.match(line)
        if mp:
            if current_model is None or current_size is None:
                continue  # incomplete block

            price = float(mp.group(1).replace(",", ""))
            if not (30 < price < 3000):
                continue

            # Determine brand: prefer Manufacturer line, fall back to URL slug
            brand_key = None
            if current_brand_mfgr:
                brand_key = current_brand_mfgr.lower()
            elif current_brand_slug:
                brand_key = current_brand_slug

            if brand_key not in TARGET_BRANDS:
                # Reset and move on
                current_model = None
                current_size = None
                current_brand_mfgr = None
                current_brand_slug = None
                continue

            brand_display = BRAND_DISPLAY[brand_key]

            results.append({
                "brand": brand_display,
                "model": current_model,
                "size": current_size,
                "price": price,
                "shop": SHOP,
            })

            # After emitting, reset size (but keep model+brand for next sizes of same tire
            # if the page lists each size as its own block — which it does)
            current_size = None
            current_brand_mfgr = None
            # Keep current_model and current_brand_slug in case next block is same model
            continue

    return results


def dedupe(entries: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for e in entries:
        key = (e["brand"], e["model"], e["size"], e["price"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def main():
    all_entries = []

    for idx, url in enumerate(CATEGORY_URLS):
        if idx > 0:
            print(f"[agr] sleeping 3s before next fetch...", file=sys.stderr)
            time.sleep(3)

        category = url.split("type=")[-1]
        print(f"[agr] fetching {category}: {url}", file=sys.stderr)
        result = smart_scrape(url)

        if not result["success"]:
            print(f"[agr] FAILED {category}: {result['error']}", file=sys.stderr)
            continue

        content = result["content"]
        print(f"[agr] got {len(content)} chars from {category}", file=sys.stderr)

        entries = parse_entries(content)
        print(f"[agr] parsed {len(entries)} entries from {category}", file=sys.stderr)
        all_entries.extend(entries)

    deduped = dedupe(all_entries)
    print(f"\n[agr] total after dedup: {len(deduped)} entries", file=sys.stderr)

    from collections import Counter
    brand_counts = Counter(e["brand"] for e in deduped)
    for brand, count in sorted(brand_counts.items()):
        print(f"  {brand}: {count}", file=sys.stderr)

    with open(OUTPUT, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"\n[agr] saved to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
