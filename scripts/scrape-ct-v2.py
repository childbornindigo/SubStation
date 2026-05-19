#!/usr/bin/env python3
"""
Canadian Tire tire price scraper — v2.
Approaches:
  1. Firecrawl search to find real product page URLs
  2. Scrape individual product pages via Firecrawl
"""
import json
import os
import sys
import re
import time
import urllib.parse
import urllib.request
import urllib.error

FIRECRAWL_API_KEY = "fc-2d1cf8b9d6e84fc797d360879036f290"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "local-prices-canadian-tire.json")

os.environ["FIRECRAWL_API_KEY"] = FIRECRAWL_API_KEY

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
PRICE_RE = re.compile(r'\$\s?([\d,]+\.?\d{0,2})')


def log(msg):
    print(msg, file=sys.stderr)


def firecrawl_search(query, limit=5):
    """Use Firecrawl search to find URLs. Returns list of (url, title, description)."""
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


def firecrawl_scrape(url, wait_for_ms=5000):
    """Scrape a URL via Firecrawl and return markdown text or None."""
    try:
        result = fc.scrape(url, formats=['markdown'], wait_for=wait_for_ms)
        if hasattr(result, 'markdown'):
            return result.markdown or None
        elif isinstance(result, dict):
            return result.get('markdown') or None
    except Exception as e:
        log(f"  [scrape] error on {url}: {e}")
    return None


def extract_prices_from_markdown(md, brand, model, source_url=""):
    """Parse size+price pairs from markdown. Returns list of result dicts."""
    results = []
    if not md:
        return results

    # Check if prices are set-of-4
    is_set_of_4 = bool(re.search(r'set of 4|per set|for 4|4 tires', md, re.IGNORECASE))
    installed = bool(re.search(r'install|mounted|with install', md, re.IGNORECASE))

    lines = md.split("\n")
    for i, line in enumerate(lines):
        sizes = SIZE_RE.findall(line)
        if not sizes:
            continue
        # Look for prices in ±4 lines around the size
        context = "\n".join(lines[max(0, i - 4): min(len(lines), i + 5)])
        prices = PRICE_RE.findall(context)
        if not prices:
            continue
        for size in sizes:
            for p in prices:
                price = float(p.replace(",", ""))
                if 50 < price < 3000:
                    if is_set_of_4:
                        price = round(price / 4, 2)
                        note = "per tire (normalized from set-of-4)"
                    elif installed:
                        note = "per tire (may include installation)"
                    else:
                        note = "per tire"
                    results.append({
                        "brand": brand,
                        "model": model,
                        "size": size,
                        "price": price,
                        "shop": "Canadian Tire",
                        "price_note": note,
                        "source_url": source_url,
                    })
                    break  # one price per size
    return results


def find_ct_product_urls(brand, model):
    """Search for CT product pages for a tire model."""
    # Try site-specific search
    query = f'site:canadiantire.ca "{brand} {model}" tire'
    log(f"  [search] {brand} {model}")
    items = firecrawl_search(query, limit=5)

    ct_urls = []
    for url, title, desc in items:
        if 'canadiantire.ca' in url and '/en/pdp/' in url:
            ct_urls.append(url)
            log(f"    Found: {url[:80]}")

    # If no pdp URLs, try broader search
    if not ct_urls:
        query2 = f'canadiantire.ca {brand} {model} tire pdp'
        items2 = firecrawl_search(query2, limit=5)
        for url, title, desc in items2:
            if 'canadiantire.ca' in url and '/en/pdp/' in url:
                ct_urls.append(url)
                log(f"    Found (broad): {url[:80]}")

    return list(dict.fromkeys(ct_urls))  # dedupe


def scrape_ct_product_page(url, brand, model):
    """Scrape a CT product page and extract all sizes + prices."""
    log(f"  [pdp] {url[:80]}")
    md = firecrawl_scrape(url, wait_for_ms=6000)
    if not md:
        log(f"    No content")
        return []

    log(f"    Got {len(md)} chars")
    results = extract_prices_from_markdown(md, brand, model, source_url=url)
    log(f"    Extracted {len(results)} prices")

    # On a CT product PDP, we might only get one size.
    # CT shows a size selector — we may need to scrape a "see all sizes" link.
    # Check if there's a link to a size listing page
    size_page_pattern = re.search(
        r'https://www\.canadiantire\.ca/en/pdp/[^\s\)\"\']+sizes[^\s\)\"\']*\.html',
        md
    )
    if size_page_pattern:
        sizes_url = size_page_pattern.group(0)
        log(f"    Found sizes page: {sizes_url[:80]}")
        md2 = firecrawl_scrape(sizes_url, wait_for_ms=6000)
        if md2:
            more = extract_prices_from_markdown(md2, brand, model, source_url=sizes_url)
            results.extend(more)

    return results


def main():
    all_results = []
    seen_keys = set()

    def add_results(new_results):
        for r in new_results:
            key = (r["brand"], r["model"], r["size"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_results.append(r)

    # Phase 1: Find product URLs via search, then scrape each PDP
    log("\n=== Finding CT product URLs via Firecrawl search ===")
    all_product_urls = []  # (brand, model, url)

    for brand, model in MODELS_TO_SEARCH:
        urls = find_ct_product_urls(brand, model)
        for url in urls[:3]:  # max 3 per model
            all_product_urls.append((brand, model, url))
        time.sleep(1.0)

    log(f"\n=== Scraping {len(all_product_urls)} product pages ===")
    for brand, model, url in all_product_urls:
        results = scrape_ct_product_page(url, brand, model)
        if results:
            add_results(results)
        time.sleep(2.0)

    # Phase 2: For brands/models with no results, try CT's search page directly
    no_data_models = [(b, m) for b, m in MODELS_TO_SEARCH
                      if not any(r["brand"] == b and r["model"] == m for r in all_results)]

    if no_data_models:
        log(f"\n=== Phase 2: Direct CT search pages for {len(no_data_models)} models ===")
        for brand, model in no_data_models:
            query = f"{brand} {model} tire"
            ct_search_url = f"https://www.canadiantire.ca/en/search#{urllib.parse.quote(query)}"
            log(f"  [CT search] {brand} {model}")
            md = firecrawl_scrape(ct_search_url, wait_for_ms=7000)
            if md and len(md) > 200:
                log(f"    Got {len(md)} chars")
                # Look for product URLs in the rendered page
                pdp_urls = re.findall(
                    r'https://www\.canadiantire\.ca/en/pdp/[^\s\)\"\']+\.html', md
                )
                pdp_urls = list(dict.fromkeys(pdp_urls))
                log(f"    Found {len(pdp_urls)} product URLs in search results")
                for pdp_url in pdp_urls[:2]:
                    results = scrape_ct_product_page(pdp_url, brand, model)
                    add_results(results)
                    time.sleep(2.0)
            time.sleep(2.0)

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\n=== Done: {len(all_results)} total prices ===")
    log(f"Saved to: {OUTPUT_PATH}")

    from collections import Counter
    by_brand = Counter(r["brand"] for r in all_results)
    for brand, count in sorted(by_brand.items()):
        log(f"  {brand}: {count}")

    missing = [(b, m) for b, m in MODELS_TO_SEARCH
               if not any(r["brand"] == b and r["model"] == m for r in all_results)]
    if missing:
        log(f"\nNot found ({len(missing)}):")
        for b, m in missing:
            log(f"  - {b} {m}")


if __name__ == "__main__":
    main()
