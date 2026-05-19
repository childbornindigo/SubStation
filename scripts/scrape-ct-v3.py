#!/usr/bin/env python3
"""
Canadian Tire tire price scraper — v3.
Strategy:
  1. Firecrawl search → find main product page URL for each tire model
  2. Scrape main page → extract per-size variant URLs from the size table
  3. Scrape a sample of size variant pages → get per-tire prices
  4. Deduplicate and save
"""
import json
import os
import sys
import re
import time

FIRECRAWL_API_KEY = "fc-2d1cf8b9d6e84fc797d360879036f290"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "local-prices-canadian-tire.json")

from firecrawl import FirecrawlApp
fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

MODELS_TO_SEARCH = [
    ("Bridgestone", "Alenza Sport AS"),
    ("Bridgestone", "WeatherPeak"),
    ("Continental", "ExtremeContact DWS06 Plus"),
    ("Continental", "CrossContact LX Sport"),
    ("Continental", "CrossContact RX"),
    ("Falken", "Azenis FK460 AS"),
    ("Firestone", "Destination LE3"),
    ("Firestone", "Weathergrip"),
    ("Hankook", "Kinergy GT"),
    ("Hankook", "Ventus S1 noble2"),
    ("Michelin", "Defender LTX M/S2"),
    ("Michelin", "Pilot Sport AS 4"),
    ("Michelin", "Primacy Tour AS"),
    ("Pirelli", "P Zero AS Plus 3"),
    ("Pirelli", "Scorpion Zero All Season"),
    ("Toyo", "Extensa AS II"),
    ("Yokohama", "Geolandar X-AT"),
    ("Yokohama", "Geolandar X-CV"),
]

SIZE_RE = re.compile(r'\b(\d{3}/\d{2}[A-Z]?\d{2})\b')
# CT price format: "$359.99  each" or "$359.99 each"
# After scraping a specific size page, price appears as "$NNN.NN  each"
EACH_PRICE_RE = re.compile(r'\$([\d,]+\.\d{2})\s+each', re.IGNORECASE)
RANGE_PRICE_RE = re.compile(r'\$([\d,]+\.\d{2})\s*\\?[-–]\s*\$([\d,]+\.\d{2})')

# Size variant URL pattern in CT markdown table
# Example: /en/pdp/p265-70r17-defltx2-0074955p.0074955.html
SIZE_URL_RE = re.compile(
    r'https://www\.canadiantire\.ca/en/pdp/[a-z0-9\-]+p\.\d+\.html'
)

# CT size description pattern on product page: "Size: 265/70R17 116T"
SIZE_DESC_RE = re.compile(r'Size:\s*(\d{3}/\d{2}[A-Z]?\d{2})')


def log(msg):
    print(msg, file=sys.stderr)


def fc_search(query, limit=5):
    """Firecrawl search, returns list of (url, title, desc)."""
    try:
        result = fc.search(query, limit=limit)
        items = []
        if hasattr(result, 'web') and result.web:
            for item in result.web:
                url = item.url if hasattr(item, 'url') else ''
                title = item.title if hasattr(item, 'title') else ''
                desc = item.description if hasattr(item, 'description') else ''
                items.append((url, title, desc or ''))
        return items
    except Exception as e:
        log(f"  [search] error: {e}")
        return []


def fc_scrape(url, wait_for_ms=5000):
    """Scrape URL, return markdown or None."""
    try:
        result = fc.scrape(url, formats=['markdown'], wait_for=wait_for_ms)
        if hasattr(result, 'markdown'):
            return result.markdown or None
        elif isinstance(result, dict):
            return result.get('markdown') or None
    except Exception as e:
        log(f"  [scrape] {url[:60]}... error: {e}")
    return None


def find_main_product_url(brand, model):
    """Find the main CT product page URL via Firecrawl search."""
    query = f'site:canadiantire.ca/en/pdp "{brand} {model}" tire'
    items = fc_search(query, limit=5)

    for url, title, desc in items:
        # Main product page — no variant suffix (no .XXXXXXXX.html at end of the product code)
        if 'canadiantire.ca/en/pdp/' in url:
            # Prefer English URLs
            if '/fr/' not in url:
                log(f"    Found: {url}")
                return url

    # Try broader search
    query2 = f'canadiantire.ca {brand} {model} all-season tire'
    items2 = fc_search(query2, limit=5)
    for url, title, desc in items2:
        if 'canadiantire.ca/en/pdp/' in url and '/fr/' not in url:
            log(f"    Found (broad): {url}")
            return url

    return None


def extract_size_variant_urls(md):
    """Extract per-size variant URLs from main product page markdown."""
    # CT uses a table with size variant links like:
    # /en/pdp/p265-70r17-defltx2-0074955p.0074955.html
    urls = SIZE_URL_RE.findall(md)
    return list(dict.fromkeys(urls))  # dedupe, preserve order


def scrape_size_variant(url, brand, model):
    """Scrape a per-size CT product page. Returns dict or None."""
    md = fc_scrape(url, wait_for_ms=5000)
    if not md:
        return None

    # Extract size from page
    size_match = SIZE_DESC_RE.search(md)
    if not size_match:
        # Also try from URL: p265-70r17 → 265/70R17
        url_size = re.search(r'/p(\d{3})-(\d{2}[a-z])(\d{2})-', url, re.IGNORECASE)
        if url_size:
            size = f"{url_size.group(1)}/{url_size.group(2).upper()}{url_size.group(3)}"
        else:
            return None
    else:
        size = size_match.group(1)

    # Extract price — look for "$NNN.NN  each"
    price_match = EACH_PRICE_RE.search(md)
    if price_match:
        price = float(price_match.group(1).replace(',', ''))
        price_note = "per tire (uninstalled)"
    else:
        # Fall back to first reasonable price in page header area
        # Find first dollar sign in the first 5000 chars
        early_md = md[:8000]
        # Look for price range: take the lower (first) value
        range_match = RANGE_PRICE_RE.search(early_md)
        if range_match:
            price = float(range_match.group(1).replace(',', ''))
            price_note = "per tire (range - lower bound)"
        else:
            return None

    if not (50 < price < 2000):
        return None

    return {
        "brand": brand,
        "model": model,
        "size": size,
        "price": price,
        "shop": "Canadian Tire",
        "price_note": price_note,
        "source_url": url,
    }


def main():
    all_results = []
    seen_keys = set()

    def add(r):
        key = (r["brand"], r["model"], r["size"])
        if key not in seen_keys:
            seen_keys.add(key)
            all_results.append(r)

    for brand, model in MODELS_TO_SEARCH:
        log(f"\n--- {brand} {model} ---")

        # Step 1: Find main product page
        main_url = find_main_product_url(brand, model)
        if not main_url:
            log(f"  No URL found for {brand} {model}")
            time.sleep(1)
            continue

        # Step 2: Scrape main page to get size variant URLs
        time.sleep(1.5)
        main_md = fc_scrape(main_url, wait_for_ms=6000)
        if not main_md:
            log(f"  Could not scrape main page")
            time.sleep(1)
            continue

        log(f"  Main page: {len(main_md)} chars")

        # Step 3: Extract size variant URLs
        variant_urls = extract_size_variant_urls(main_md)
        log(f"  Found {len(variant_urls)} size variants")

        if not variant_urls:
            # Try extracting price directly from main page — some tires show a single size
            size_match = SIZE_DESC_RE.search(main_md)
            price_match = EACH_PRICE_RE.search(main_md)
            if size_match and price_match:
                size = size_match.group(1)
                price = float(price_match.group(1).replace(',', ''))
                if 50 < price < 2000:
                    add({
                        "brand": brand,
                        "model": model,
                        "size": size,
                        "price": price,
                        "shop": "Canadian Tire",
                        "price_note": "per tire (uninstalled)",
                        "source_url": main_url,
                    })
                    log(f"  Got 1 price from main page")
            time.sleep(1)
            continue

        # Step 4: Scrape up to 8 size variants (to get a spread of sizes/prices)
        # Pick every N-th variant to get a spread
        step = max(1, len(variant_urls) // 8)
        sampled = variant_urls[::step][:8]
        log(f"  Sampling {len(sampled)} of {len(variant_urls)} variants")

        scraped_count = 0
        for var_url in sampled:
            time.sleep(2)
            result = scrape_size_variant(var_url, brand, model)
            if result:
                add(result)
                scraped_count += 1
                log(f"    {result['size']}: ${result['price']}")
            else:
                log(f"    No price from {var_url[:60]}")

        log(f"  Got {scraped_count} prices")
        time.sleep(1)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\n=== Done: {len(all_results)} total prices ===")
    log(f"Saved to: {OUTPUT_PATH}")

    from collections import Counter
    by_brand = Counter(r["brand"] for r in all_results)
    for brand, count in sorted(by_brand.items()):
        log(f"  {brand}: {count}")

    found_models = {(r["brand"], r["model"]) for r in all_results}
    missing = [(b, m) for b, m in MODELS_TO_SEARCH if (b, m) not in found_models]
    if missing:
        log(f"\nNot found ({len(missing)}):")
        for b, m in missing:
            log(f"  - {b} {m}")


if __name__ == "__main__":
    main()
