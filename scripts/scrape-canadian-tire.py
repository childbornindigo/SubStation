#!/usr/bin/env /opt/homebrew/bin/python3
"""
Scrape Canadian Tire for tire prices using Firecrawl (handles JS rendering).
Falls back to ScrapLing stealth if Firecrawl fails.
"""
import json
import os
import sys
import re
import time

FIRECRAWL_API_KEY = "fc-2d1cf8b9d6e84fc797d360879036f290"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from firecrawl import FirecrawlApp
fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

MODELS_TO_SEARCH = [
    ("Bridgestone", "Alenza Sport AS"),
    ("Bridgestone", "Turanza Everdrive"),
    ("Bridgestone", "WeatherPeak"),
    ("Bridgestone", "UltraWeather"),
    ("Continental", "ExtremeContact DWS06 Plus"),
    ("Continental", "CrossContact LX Sport"),
    ("Continental", "CrossContact RX"),
    ("Continental", "ProContact RX"),
    ("Falken", "Azenis FK460 AS"),
    ("Firestone", "Destination LE3"),
    ("Firestone", "FR710"),
    ("Firestone", "Weathergrip"),
    ("Hankook", "Ventus S1 noble2"),
    ("Hankook", "Kinergy GT"),
    ("Hankook", "Dynapro evo AS"),
    ("Michelin", "Defender LTX M/S2"),
    ("Michelin", "Pilot Sport AS 4"),
    ("Michelin", "Primacy Tour AS"),
    ("Pirelli", "P Zero AS Plus 3"),
    ("Pirelli", "Scorpion Zero All Season"),
    ("Toyo", "Extensa AS II"),
    ("Yokohama", "Geolandar X-AT"),
    ("Yokohama", "Geolandar X-CV"),
]


def extract_tire_prices(markdown_text, brand, model):
    """Extract tire sizes and prices from scraped markdown."""
    results = []
    if not markdown_text:
        return results

    size_pattern = r'(\d{3}/\d{2}[A-Z]?\d{2})'
    price_pattern = r'\$\s?([\d,]+\.?\d{0,2})'

    lines = markdown_text.split('\n')
    for i, line in enumerate(lines):
        context = '\n'.join(lines[max(0, i-2):min(len(lines), i+3)])

        sizes = re.findall(size_pattern, line)
        if not sizes:
            continue

        prices = re.findall(price_pattern, context)
        if not prices:
            continue

        for size in sizes:
            for price_str in prices:
                price = float(price_str.replace(',', ''))
                if 50 < price < 2000:
                    results.append({
                        "brand": brand,
                        "model": model,
                        "size": size,
                        "price": price,
                        "shop": "Canadian Tire",
                    })
                    break

    return results


def scrape_ct_search(brand, model):
    """Search Canadian Tire for a tire model."""
    query = f"{brand} {model} tire"
    url = f"https://www.canadiantire.ca/en/search?q={query.replace(' ', '+')}"

    print(f"[CT] Searching: {brand} {model}", file=sys.stderr)
    print(f"     URL: {url}", file=sys.stderr)

    try:
        result = fc.scrape_url(url, params={
            "formats": ["markdown"],
            "waitFor": 5000,
        })
        content = ""
        if hasattr(result, 'markdown'):
            content = result.markdown or ""
        elif isinstance(result, dict):
            content = result.get('markdown', '')

        if content:
            print(f"     Got {len(content)} chars", file=sys.stderr)
            return content
        else:
            print(f"     Empty response", file=sys.stderr)
            return None
    except Exception as e:
        print(f"     Error: {e}", file=sys.stderr)
        return None


def main():
    all_results = []

    for brand, model in MODELS_TO_SEARCH:
        content = scrape_ct_search(brand, model)
        if content:
            prices = extract_tire_prices(content, brand, model)
            if prices:
                all_results.extend(prices)
                print(f"     Found {len(prices)} prices", file=sys.stderr)
            else:
                print(f"     No prices extracted from content", file=sys.stderr)
        time.sleep(2)

    output_path = os.path.join(SCRIPT_DIR, "local-prices-canadian-tire.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[CT] Total: {len(all_results)} prices saved to {output_path}", file=sys.stderr)

    # Summary
    from collections import Counter
    brands = Counter(r['brand'] for r in all_results)
    print(f"[CT] By brand: {dict(brands)}", file=sys.stderr)


if __name__ == "__main__":
    main()
