#!/usr/bin/env python3
"""
Canadian Tire tire price scraper — final version.
Strategy:
  1. Use known main product page URLs (pre-collected via earlier Firecrawl search)
  2. Scrapling stealthy-fetch with full stealth flags on main pages → extract size variant URLs
  3. Scrapling stealthy-fetch on sampled size variant pages → extract per-tire prices
"""
import json
import os
import sys
import re
import time
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "local-prices-canadian-tire.json")

# Product page URLs — verified via Firecrawl search + Google/Bing search
# URLs confirmed to return actual product content (not 404)
KNOWN_MAIN_URLS = [
    ("Bridgestone", "Alenza Sport AS",
     "https://www.canadiantire.ca/en/pdp/bridgestone-alenza-sport-a-s-sp-tire-for-passenger-cuv-3085959p.html"),
    ("Bridgestone", "WeatherPeak",
     "https://www.canadiantire.ca/en/pdp/bridgestone-weatherpeak-all-weather-tire-for-passenger-cuv-5081209p.html"),
    ("Continental", "ExtremeContact DWS06 Plus",
     "https://www.canadiantire.ca/en/pdp/continental-extremecontact-dws06-plus-performance-tire-for-passenger-cuv-0063017p.html"),
    ("Continental", "CrossContact LX Sport",
     "https://www.canadiantire.ca/en/pdp/continental-crosscontact-lx-sport-all-season-tire-for-passenger-cuv-3080035p.html"),
    ("Continental", "CrossContact RX",
     "https://www.canadiantire.ca/en/pdp/continental-crosscontact-rx-all-season-tire-3080097p.html"),
    ("Firestone", "Destination LE3",
     "https://www.canadiantire.ca/en/pdp/firestone-destination-le3-all-season-tire-for-truck-suv-3085677p.html"),
    ("Firestone", "Weathergrip",
     "https://www.canadiantire.ca/en/pdp/firestone-weathergrip-all-weather-tire-for-passenger-cuvs-3085641p.html"),
    ("Hankook", "Kinergy GT",
     "https://www.canadiantire.ca/en/pdp/hankook-kinergy-gt-all-season-tire-for-passenger-cuv-4080100p.html"),
    ("Hankook", "Ventus S1 noble2",
     "https://www.canadiantire.ca/en/pdp/hankook-ventus-s1-noble2-performance-tire-for-passenger-cuv-4080109p.html"),
    ("Michelin", "Defender LTX M/S2",
     "https://www.canadiantire.ca/en/pdp/michelin-defender-ltx-m-s-2-all-season-tire-for-truck-suv-0074955p.html"),
    ("Michelin", "Pilot Sport AS 4",
     "https://www.canadiantire.ca/en/pdp/michelin-pilot-sport-a-s-4-all-season-tire-for-passenger-5084480p.html"),
    ("Pirelli", "P Zero AS Plus 3",
     "https://www.canadiantire.ca/en/pdp/pirelli-p-zero-all-season-plus-3-tire-5082339p.html"),
    ("Pirelli", "Scorpion Zero All Season",
     "https://www.canadiantire.ca/en/pdp/pirelli-scorpiontm-zero-all-season-tire-for-passenger-cuv-4082545p.html"),
    # Michelin Primacy Tour AS — URL from earlier Firecrawl search
    ("Michelin", "Primacy Tour AS",
     "https://www.canadiantire.ca/en/pdp/michelin-primacy-tour-a-s-tire-for-passenger-cuv-3085961p.html"),
    # Yokohama — URLs from earlier Firecrawl search
    ("Yokohama", "Geolandar X-AT",
     "https://www.canadiantire.ca/en/pdp/yokohama-geolandar-x-at-all-terrain-tire-for-light-truck-suv-3085974p.html"),
    ("Yokohama", "Geolandar X-CV",
     "https://www.canadiantire.ca/en/pdp/yokohama-geolandar-x-cv-all-season-tire-for-passenger-cuv-3085975p.html"),
    # Falken Azenis FK460 AS and Toyo Extensa AS II — not found on CT
]

# CT price on individual size page — various formats seen:
# 1. "$359.99  each"   (regular price, space before "each")
# 2. "$377.49\n\n~~$399.99~~...\n\n each\n"  (sale price on its own line, "each" follows later)
# 3. "$377.49\n\n each\n"  (sale price without strikethrough)
EACH_PRICE_RE = re.compile(r'\$([\d,]+\.\d{2})\s+each', re.IGNORECASE)
# Sale: price on own line, then 0-4 lines of other text, then "each"
# Pattern: dollar amount, optional newlines/text, then " each" within 200 chars
PRICE_THEN_EACH_RE = re.compile(r'\$([\d,]+\.\d{2})\n(?:[^\n]*\n){0,4}\s*each\b', re.IGNORECASE)
# Size from page content "Size: 265/70R17 116T"
SIZE_DESC_RE = re.compile(r'Size:\s*(\d{3}/\d{2}[A-Z]?\d{2})')
# Per-size variant relative URL pattern
VARIANT_URL_RE = re.compile(r'/en/pdp/[a-z0-9\-]+p\.\d+\.html')

SCRAPLING = "/opt/homebrew/bin/scrapling"
STEALTH_FLAGS = ["--real-chrome", "--solve-cloudflare", "--block-webrtc", "--hide-canvas"]
BASE_URL = "https://www.canadiantire.ca"


def log(msg):
    print(msg, file=sys.stderr)


def scrapling_fetch(url, timeout_ms=30000):
    """Fetch a URL with scrapling stealthy mode. Returns text content or None."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            SCRAPLING, "extract", "stealthy-fetch",
            url, tmp_path,
            "--timeout", str(timeout_ms),
        ] + STEALTH_FLAGS

        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout_ms / 1000 + 45
        )

        if proc.returncode != 0:
            log(f"  [scrapling] failed ({proc.returncode}): {proc.stderr.strip()[:200]}")
            return None

        with open(tmp_path) as f:
            content = f.read()

        if not content.strip() or "General PDP Template" in content[:500]:
            log(f"  [scrapling] empty or template page")
            return None

        return content

    except subprocess.TimeoutExpired:
        log(f"  [scrapling] timed out: {url[:60]}")
        return None
    except Exception as e:
        log(f"  [scrapling] error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def get_size_variant_urls(main_page_content):
    """Extract per-size variant relative URLs from main product page content."""
    rel_urls = VARIANT_URL_RE.findall(main_page_content)
    # Convert to absolute, deduplicate
    full_urls = [BASE_URL + u for u in rel_urls]
    return list(dict.fromkeys(full_urls))


def scrape_size_variant(url, brand, model):
    """Scrape a per-size CT product page. Returns result dict or None."""
    content = scrapling_fetch(url, timeout_ms=25000)
    if not content:
        return None

    # Extract size
    size_m = SIZE_DESC_RE.search(content)
    if not size_m:
        # Fallback: parse size from URL slug e.g. /p265-70r17-
        url_size_m = re.search(r'/p(\d{3})-(\d{2})[a-z](\d{2})-', url, re.IGNORECASE)
        if url_size_m:
            size = f"{url_size_m.group(1)}/{url_size_m.group(2)}R{url_size_m.group(3)}"
        else:
            return None
    else:
        size = size_m.group(1)

    # Extract price
    # Try most specific pattern first (price directly followed by "each")
    each_m = EACH_PRICE_RE.search(content)
    if each_m:
        price = float(each_m.group(1).replace(',', ''))
        price_note = "per tire (uninstalled)"
    else:
        # Try: price on own line, then "each" within next few lines
        pthen_m = PRICE_THEN_EACH_RE.search(content)
        if pthen_m:
            price = float(pthen_m.group(1).replace(',', ''))
            # Check if there's a struck-through original price after (means this is sale price)
            after_snippet = content[pthen_m.start():pthen_m.start()+300]
            if '~~$' in after_snippet:
                price_note = "per tire (sale price)"
            else:
                price_note = "per tire (uninstalled)"
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

    for brand, model, main_url in KNOWN_MAIN_URLS:
        log(f"\n--- {brand} {model} ---")

        # Step 1: Scrape main page to get size variant URLs
        log(f"  Scraping main page: {main_url}")
        main_content = scrapling_fetch(main_url, timeout_ms=35000)

        # Retry once if we got General PDP Template (rate limited)
        if not main_content:
            log(f"  Retrying after 10s...")
            time.sleep(10)
            main_content = scrapling_fetch(main_url, timeout_ms=35000)

        if not main_content:
            log(f"  Failed to get main page")
            time.sleep(5)
            continue

        log(f"  Main page: {len(main_content)} chars")

        # Step 2: Extract size variant URLs
        variant_urls = get_size_variant_urls(main_content)
        log(f"  Found {len(variant_urls)} size variants")

        if not variant_urls:
            # Try extracting price from main page directly (for single-size tires)
            size_m = SIZE_DESC_RE.search(main_content)
            each_m = EACH_PRICE_RE.search(main_content)
            if size_m and each_m:
                price = float(each_m.group(1).replace(',', ''))
                if 50 < price < 2000:
                    add({
                        "brand": brand, "model": model,
                        "size": size_m.group(1),
                        "price": price, "shop": "Canadian Tire",
                        "price_note": "per tire (uninstalled)",
                        "source_url": main_url,
                    })
                    log(f"  Got price from main page: {size_m.group(1)} ${price}")
            time.sleep(3)
            continue

        # Step 3: Sample up to 8 variants spread across the size range
        step = max(1, len(variant_urls) // 8)
        sampled = variant_urls[::step][:8]
        log(f"  Sampling {len(sampled)}/{len(variant_urls)} variants")

        scraped_count = 0
        for var_url in sampled:
            time.sleep(5)
            result = scrape_size_variant(var_url, brand, model)
            if result:
                add(result)
                scraped_count += 1
                log(f"    {result['size']}: ${result['price']} ({result['price_note']})")
            else:
                log(f"    No data from {var_url[-60:]}")

        log(f"  Got {scraped_count} prices")
        time.sleep(5)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\n=== Done: {len(all_results)} total prices ===")
    log(f"Saved to: {OUTPUT_PATH}")

    from collections import Counter
    by_brand = Counter(r["brand"] for r in all_results)
    for bname, count in sorted(by_brand.items()):
        log(f"  {bname}: {count}")

    found_models = {(r["brand"], r["model"]) for r in all_results}
    all_models = [(b, m) for b, m, _ in KNOWN_MAIN_URLS]
    missing = [(b, m) for b, m in all_models if (b, m) not in found_models]
    if missing:
        log(f"\nNot found ({len(missing)}):")
        for b, m in missing:
            log(f"  - {b} {m}")


if __name__ == "__main__":
    main()
