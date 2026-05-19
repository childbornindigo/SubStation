#!/usr/bin/env python3
"""
Re-scrape local GTA market using Scrapling for bot-walled sites
and Google Shopping for broad market data.

Methodology: For each brand/model/size in our inventory,
get prices from as many local sources as possible, average them.
"""

import json
import os
import sys
import subprocess
import tempfile
import time
import re
from collections import defaultdict

SCRAPLING_BIN = "/opt/homebrew/bin/scrapling"
SCRIPTS_DIR = "/Users/indigochild/.hermes/extensions/substation/scripts"

def scrapling_fetch(url, mode="fetch", timeout=30000, css_selector=None):
    """Fetch a URL using Scrapling CLI. Returns markdown content or None."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [SCRAPLING_BIN, "extract", mode, url, tmp_path, "--timeout", str(timeout)]
        if mode == "stealthy-fetch":
            cmd += ["--block-webrtc", "--hide-canvas"]
        if css_selector:
            cmd += ["--css-selector", css_selector]
        if mode in ("fetch", "stealthy-fetch"):
            cmd += ["--network-idle"]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout/1000 + 30)

        if proc.returncode != 0:
            print(f"  FAIL [{mode}] {url}: {proc.stderr.strip()[:200]}", file=sys.stderr)
            return None

        with open(tmp_path, "r") as f:
            content = f.read()

        if not content.strip():
            print(f"  EMPTY [{mode}] {url}", file=sys.stderr)
            return None

        print(f"  OK [{mode}] {url} ({len(content)} chars)", file=sys.stderr)
        return content

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{mode}] {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR [{mode}] {url}: {e}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def parse_prices_from_text(text, brand, model):
    """Extract prices from scraped text. Returns list of {size, price} dicts."""
    results = []
    if not text:
        return results

    # Look for tire size patterns near price patterns
    # Common size formats: 205/55R16, P205/55R16, 225/45R17, etc.
    size_pattern = r'[PL]?T?\d{3}/\d{2}[RZ]\d{2}'
    price_pattern = r'\$\s*(\d{2,4}(?:\.\d{2})?)'

    lines = text.split('\n')
    for i, line in enumerate(lines):
        sizes = re.findall(size_pattern, line)
        prices = re.findall(price_pattern, line)

        if sizes and prices:
            for size in sizes:
                # Normalize size
                s = size.strip().upper()
                for prefix in ['P', 'LT']:
                    if s.startswith(prefix) and s[len(prefix):len(prefix)+1].isdigit():
                        s = s[len(prefix):]

                price = float(prices[0])
                if 50 < price < 2000:  # sanity check
                    results.append({"size": s, "price": price})

        # Also check next line for price if this line has a size
        if sizes and not prices and i + 1 < len(lines):
            next_prices = re.findall(price_pattern, lines[i + 1])
            if next_prices:
                for size in sizes:
                    s = size.strip().upper()
                    for prefix in ['P', 'LT']:
                        if s.startswith(prefix) and s[len(prefix):len(prefix)+1].isdigit():
                            s = s[len(prefix):]
                    price = float(next_prices[0])
                    if 50 < price < 2000:
                        results.append({"size": s, "price": price})

    return results


def scrape_google_shopping(brand, model, size):
    """Search Google Shopping for a specific tire and extract prices."""
    query = f"{brand} {model} {size} tire"
    url = f"https://www.google.ca/search?q={query.replace(' ', '+')}&tbm=shop&gl=ca&hl=en"

    content = scrapling_fetch(url, mode="stealthy-fetch", timeout=20000)
    if not content:
        return []

    results = []
    # Google Shopping shows prices in format like "$289.99" near product listings
    price_pattern = r'\$(\d{2,4}(?:\.\d{2})?)'
    prices = re.findall(price_pattern, content)

    # Also look for store names near prices
    for price_str in prices:
        price = float(price_str)
        if 50 < price < 2000:
            results.append({
                "price": price,
                "source": "Google Shopping CA"
            })

    return results


def scrape_canadian_tire(brand, model, size):
    """Scrape Canadian Tire using Scrapling's dynamic fetch for JS-rendered SPA."""
    query = f"{brand}+{model}+{size}".replace(' ', '+')
    url = f"https://www.canadiantire.ca/en/search-results.html?q={query}"

    content = scrapling_fetch(url, mode="fetch", timeout=30000)
    if not content:
        # Try stealthy mode
        content = scrapling_fetch(url, mode="stealthy-fetch", timeout=30000)

    if not content:
        return []

    return parse_prices_from_text(content, brand, model)


def scrape_ok_tire(brand, model, size):
    """Scrape OK Tire using stealthy fetch."""
    query = f"{brand}+{model}+{size}".replace(' ', '+')
    url = f"https://www.oktire.com/tires?search={query}"

    content = scrapling_fetch(url, mode="stealthy-fetch", timeout=30000)
    if not content:
        return []

    return parse_prices_from_text(content, brand, model)


def scrape_kal_tire(brand, model, size):
    """Scrape Kal Tire using stealthy fetch."""
    query = f"{brand}+{model}+{size}".replace(' ', '+')
    url = f"https://www.kaltire.com/en/search/?q={query}"

    content = scrapling_fetch(url, mode="stealthy-fetch", timeout=30000)
    if not content:
        return []

    return parse_prices_from_text(content, brand, model)


def scrape_costco_tire(brand, model, size):
    """Scrape Costco Tire Centre."""
    query = f"{brand}+{model}+{size}+tire".replace(' ', '+')
    url = f"https://www.costco.ca/CatalogSearch?keyword={query}&dept=All"

    content = scrapling_fetch(url, mode="stealthy-fetch", timeout=30000)
    if not content:
        return []

    return parse_prices_from_text(content, brand, model)


def main():
    # Load inventory
    with open(os.path.join(SCRIPTS_DIR, "inventory-dump.json")) as f:
        inventory = json.load(f)

    # Group by brand/model, pick representative sizes (small, medium, large)
    from itertools import groupby

    model_sizes = defaultdict(list)
    for t in inventory:
        key = (t['brand'], t['model'])
        model_sizes[key].append(t)

    # For each model, pick up to 5 representative sizes
    test_tires = []
    for (brand, model), tires in sorted(model_sizes.items()):
        sorted_tires = sorted(tires, key=lambda t: t.get('price_wholesale', 0) or 0)
        # Pick: cheapest, 25th percentile, median, 75th percentile, most expensive
        n = len(sorted_tires)
        indices = [0, n//4, n//2, 3*n//4, n-1]
        seen = set()
        for idx in indices:
            t = sorted_tires[idx]
            key = (brand, model, t['size'])
            if key not in seen:
                seen.add(key)
                test_tires.append(t)

    print(f"Testing {len(test_tires)} representative tires across {len(model_sizes)} models")

    # First test: can we get Canadian Tire with Scrapling?
    print("\n=== TESTING CANADIAN TIRE (JS SPA) ===")
    test_size = test_tires[0]
    ct_results = scrape_canadian_tire(test_size['brand'], test_size['model'], test_size['size'])
    print(f"Canadian Tire test: {len(ct_results)} prices found for {test_size['brand']} {test_size['model']} {test_size['size']}")

    # Test Google Shopping
    print("\n=== TESTING GOOGLE SHOPPING ===")
    gs_results = scrape_google_shopping(test_size['brand'], test_size['model'], test_size['size'])
    print(f"Google Shopping test: {len(gs_results)} prices found for {test_size['brand']} {test_size['model']} {test_size['size']}")

    # Test OK Tire
    print("\n=== TESTING OK TIRE (BOT WALL) ===")
    ok_results = scrape_ok_tire(test_size['brand'], test_size['model'], test_size['size'])
    print(f"OK Tire test: {len(ok_results)} prices found")

    # Test Kal Tire
    print("\n=== TESTING KAL TIRE (BOT WALL) ===")
    kal_results = scrape_kal_tire(test_size['brand'], test_size['model'], test_size['size'])
    print(f"Kal Tire test: {len(kal_results)} prices found")

    # Test Costco
    print("\n=== TESTING COSTCO ===")
    costco_results = scrape_costco_tire(test_size['brand'], test_size['model'], test_size['size'])
    print(f"Costco test: {len(costco_results)} prices found")

    # Report what works
    print("\n=== RESULTS ===")
    working = []
    if ct_results: working.append("Canadian Tire")
    if gs_results: working.append("Google Shopping")
    if ok_results: working.append("OK Tire")
    if kal_results: working.append("Kal Tire")
    if costco_results: working.append("Costco")

    print(f"Working sources: {', '.join(working) if working else 'NONE'}")
    print(f"Failed sources: {', '.join(s for s in ['Canadian Tire', 'Google Shopping', 'OK Tire', 'Kal Tire', 'Costco'] if s not in working)}")

    # Save raw results for inspection
    test_results = {
        "test_tire": f"{test_size['brand']} {test_size['model']} {test_size['size']}",
        "canadian_tire": ct_results,
        "google_shopping": gs_results,
        "ok_tire": ok_results,
        "kal_tire": kal_results,
        "costco": costco_results,
    }

    output_path = os.path.join(SCRIPTS_DIR, "scrapling-test-results.json")
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
    main()
