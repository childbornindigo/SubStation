#!/usr/bin/env python3
"""
Scrape tire prices from ZRacing Mississauga for all 27 inventory models.
Uses Firecrawl via smart_scrape with 2-second delays between requests.
"""

import os
import sys
import json
import time
import re

os.environ['FIRECRAWL_API_KEY'] = 'fc-2d1cf8b9d6e84fc797d360879036f290'
sys.path.insert(0, '/Users/indigochild/.hermes/extensions/substation/scripts')
from scrape_utils import smart_scrape

OUTPUT_PATH = '/Users/indigochild/.hermes/extensions/substation/scripts/local-prices-zracing.json'
BASE_URL = 'https://zracing.ca/tires/{brand}-tires/{slug}'
SHOP = 'ZRacing Mississauga'

# 27 models with primary slug and fallback slugs
MODELS = [
    # Bridgestone
    {
        "brand": "Bridgestone",
        "model": "Alenza Sport AS",
        "brand_slug": "bridgestone",
        "slugs": ["bridgestone-alenza-sport-a-s", "bridgestone-alenza-sport-as", "bridgestone-alenza-as-ultra", "bridgestone-alenza-a-s-ultra"]
    },
    {
        "brand": "Bridgestone",
        "model": "Dueler HL Alenza Plus",
        "brand_slug": "bridgestone",
        "slugs": ["bridgestone-dueler-hl-alenza-plus", "bridgestone-dueler-hl-alenza", "bridgestone-alenza-hl"]
    },
    {
        "brand": "Bridgestone",
        "model": "Turanza Everdrive",
        "brand_slug": "bridgestone",
        "slugs": ["bridgestone-turanza-everdrive", "bridgestone-turanza-ev"]
    },
    {
        "brand": "Bridgestone",
        "model": "UltraWeather",
        "brand_slug": "bridgestone",
        "slugs": ["bridgestone-ultraweather", "bridgestone-ultra-weather", "bridgestone-ultraweather-as"]
    },
    {
        "brand": "Bridgestone",
        "model": "WeatherPeak",
        "brand_slug": "bridgestone",
        "slugs": ["bridgestone-weatherpeak", "bridgestone-weather-peak"]
    },
    # Continental
    {
        "brand": "Continental",
        "model": "CrossContact LX Sport",
        "brand_slug": "continental",
        "slugs": ["continental-crosscontact-lx-sport", "continental-cross-contact-lx-sport", "continental-crosscontact-lx"]
    },
    {
        "brand": "Continental",
        "model": "CrossContact RX",
        "brand_slug": "continental",
        "slugs": ["continental-crosscontact-rx", "continental-cross-contact-rx"]
    },
    {
        "brand": "Continental",
        "model": "ExtremeContact DWS06 Plus",
        "brand_slug": "continental",
        "slugs": ["continental-extremecontact-dws06-plus", "continental-extreme-contact-dws06-plus", "continental-extremecontact-dws06"]
    },
    {
        "brand": "Continental",
        "model": "ProContact RX",
        "brand_slug": "continental",
        "slugs": ["continental-procontact-rx", "continental-pro-contact-rx"]
    },
    # Falken
    {
        "brand": "Falken",
        "model": "Azenis FK460 AS",
        "brand_slug": "falken",
        "slugs": ["falken-azenis-fk460-a-s", "falken-azenis-fk460-as", "falken-azenis-fk460"]
    },
    # Firestone
    {
        "brand": "Firestone",
        "model": "Destination LE 2",
        "brand_slug": "firestone",
        "slugs": ["firestone-destination-le-2", "firestone-destination-le2", "firestone-destination-le-2-suv"]
    },
    {
        "brand": "Firestone",
        "model": "Destination LE3",
        "brand_slug": "firestone",
        "slugs": ["firestone-destination-le3", "firestone-destination-le-3", "firestone-destination-le3-suv"]
    },
    {
        "brand": "Firestone",
        "model": "FR710",
        "brand_slug": "firestone",
        "slugs": ["firestone-fr710", "firestone-fr-710", "firestone-fr710-all-season"]
    },
    {
        "brand": "Firestone",
        "model": "Weathergrip",
        "brand_slug": "firestone",
        "slugs": ["firestone-weathergrip", "firestone-weather-grip"]
    },
    # Hankook
    {
        "brand": "Hankook",
        "model": "Dynapro evo AS",
        "brand_slug": "hankook",
        "slugs": ["hankook-dynapro-evo-a-s", "hankook-dynapro-evo-as", "hankook-dynapro-evo"]
    },
    {
        "brand": "Hankook",
        "model": "Kinergy GT",
        "brand_slug": "hankook",
        "slugs": ["hankook-kinergy-gt", "hankook-kinergy-gt-h436"]
    },
    {
        "brand": "Hankook",
        "model": "Ventus S1 noble2",
        "brand_slug": "hankook",
        "slugs": ["hankook-ventus-s1-noble2", "hankook-ventus-s1-noble-2", "hankook-ventus-s1-evo2"]
    },
    # Michelin
    {
        "brand": "Michelin",
        "model": "Defender LTX M/S2",
        "brand_slug": "michelin",
        "slugs": ["michelin-defender-ltx-m-s-2", "michelin-defender-ltx-ms2", "michelin-defender-ltx-ms", "michelin-defender-2"]
    },
    {
        "brand": "Michelin",
        "model": "Pilot Sport AS 4",
        "brand_slug": "michelin",
        "slugs": ["michelin-pilot-sport-a-s-4", "michelin-pilot-sport-as-4", "michelin-pilot-sport-as4"]
    },
    {
        "brand": "Michelin",
        "model": "Primacy Tour AS",
        "brand_slug": "michelin",
        "slugs": ["michelin-primacy-tour-a-s", "michelin-primacy-tour-as", "michelin-primacy-as", "michelin-primacy-mxm4"]
    },
    # Pirelli
    {
        "brand": "Pirelli",
        "model": "P Zero AS Plus 3",
        "brand_slug": "pirelli",
        "slugs": ["pirelli-p-zero-a-s-plus-3", "pirelli-p-zero-as-plus-3", "pirelli-p-zero-as-plus", "pirelli-p7-as-plus-3", "pirelli-p-zero-a-s"]
    },
    {
        "brand": "Pirelli",
        "model": "Scorpion Zero All Season",
        "brand_slug": "pirelli",
        "slugs": ["pirelli-scorpion-zero-all-season", "pirelli-scorpion-zero-as", "pirelli-scorpion-weatheractive", "pirelli-scorpion-weather-active"]
    },
    # Toyo
    {
        "brand": "Toyo",
        "model": "Extensa AS II",
        "brand_slug": "toyo",
        "slugs": ["toyo-extensa-a-s-ii", "toyo-extensa-as-ii", "toyo-extensa-as-2", "toyo-extensa-as"]
    },
    {
        "brand": "Toyo",
        "model": "Open Country A50",
        "brand_slug": "toyo",
        "slugs": ["toyo-open-country-a50", "toyo-open-country-a-50", "toyo-open-country-at3", "toyo-open-country-a-t-iii"]
    },
    # Yokohama
    {
        "brand": "Yokohama",
        "model": "Geolandar X-AT",
        "brand_slug": "yokohama",
        "slugs": ["yokohama-geolandar-x-at", "yokohama-geolandar-x-at-g016", "yokohama-geolandar-at", "yokohama-geolandar-a-t-s"]
    },
    {
        "brand": "Yokohama",
        "model": "Geolandar X-CV",
        "brand_slug": "yokohama",
        "slugs": ["yokohama-geolandar-x-cv", "yokohama-geolandar-cv", "yokohama-geolandar-cv-g058"]
    },
]


def parse_price(price_str):
    """Extract float price from string like '$270.00' or '270'."""
    if not price_str:
        return None
    price_str = price_str.strip()
    match = re.search(r'[\$]?\s*([\d,]+\.?\d*)', price_str)
    if match:
        val = match.group(1).replace(',', '')
        try:
            return float(val)
        except ValueError:
            return None
    return None


def parse_tire_table(content, brand, model):
    """
    Parse markdown content from a ZRacing tire page.
    Returns list of {brand, model, size, price, shop} dicts.
    """
    results = []

    # ZRacing uses a table format. Look for tire size patterns like 225/45R17
    # and prices like $270.00

    # Strategy 1: parse markdown table rows
    # Table format: | Size | Type | Load Index | Speed Rating | Price | Status |
    # or similar

    lines = content.split('\n')

    # Find table rows with tire sizes
    size_pattern = re.compile(r'\b(\d{3}/\d{2}[Rr]\d{2}(?:\s*[A-Z]{0,3})?(?:\s*\d{2,3}[A-Z]?)?)\b')
    price_pattern = re.compile(r'\$\s*([\d,]+\.?\d*)')

    # Look for rows that contain a tire size
    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            continue

        size_match = size_pattern.search(line)
        price_match = price_pattern.search(line)

        if size_match and price_match:
            size = size_match.group(1).strip()
            price = float(price_match.group(1).replace(',', ''))

            # Skip if price looks unreasonable (< $50 or > $2000)
            if price < 50 or price > 2000:
                continue

            results.append({
                "brand": brand,
                "model": model,
                "size": size,
                "price": price,
                "shop": SHOP
            })

    # Strategy 2: look for size-price pairs in close proximity (within same paragraph)
    if not results:
        # Split into chunks and find price near size
        text = content

        for size_m in size_pattern.finditer(text):
            size = size_m.group(1).strip()
            # Look for a price within 200 chars after the size
            nearby_text = text[size_m.start():size_m.start()+300]
            price_m = price_pattern.search(nearby_text)
            if price_m:
                price = float(price_m.group(1).replace(',', ''))
                if 50 <= price <= 2000:
                    # Check not already captured
                    already = any(r['size'] == size for r in results)
                    if not already:
                        results.append({
                            "brand": brand,
                            "model": model,
                            "size": size,
                            "price": price,
                            "shop": SHOP
                        })

    return results


def is_404_or_empty(content):
    """Detect if page returned 404 or no product found."""
    if not content or len(content.strip()) < 100:
        return True
    content_lower = content.lower()
    if any(phrase in content_lower for phrase in [
        'page not found', '404', 'nothing was found', 'no products found',
        'oops! that page', "couldn't find", 'not found'
    ]):
        # Double check - if it also has tire sizes, it's probably fine
        if re.search(r'\d{3}/\d{2}[Rr]\d{2}', content):
            return False
        return True
    return False


def scrape_model(model_info):
    """Try all slugs for a model, return list of price records or NOT FOUND entry."""
    brand = model_info['brand']
    model = model_info['model']
    brand_slug = model_info['brand_slug']
    slugs = model_info['slugs']

    for i, slug in enumerate(slugs):
        url = BASE_URL.format(brand=brand_slug, slug=slug)
        print(f"  Trying: {url}", flush=True)

        if i > 0:
            time.sleep(2)

        result = smart_scrape(url)

        if not result['success']:
            print(f"    Failed: {result.get('error', 'unknown')}", flush=True)
            continue

        content = result['content']

        if is_404_or_empty(content):
            print(f"    404/empty for slug: {slug}", flush=True)
            continue

        # Got a page - try to parse
        records = parse_tire_table(content, brand, model)

        if records:
            print(f"    Found {len(records)} sizes from {slug}", flush=True)
            return records, url
        else:
            # Page loaded but no prices found - log content snippet for debugging
            print(f"    Page loaded but no prices parsed. Content snippet:", flush=True)
            print(f"    {content[:500]}", flush=True)
            # Still return empty with the URL so we know the page exists
            return [], url

    # No slug worked
    print(f"  NOT FOUND: {brand} {model}", flush=True)
    return None, None


def main():
    all_results = []
    total = len(MODELS)

    for idx, model_info in enumerate(MODELS):
        brand = model_info['brand']
        model = model_info['model']
        print(f"\n[{idx+1}/{total}] {brand} {model}", flush=True)

        records, found_url = scrape_model(model_info)

        if records is None:
            # Complete NOT FOUND
            all_results.append({
                "brand": brand,
                "model": model,
                "size": None,
                "price": None,
                "shop": SHOP,
                "note": "NOT FOUND on zracing.ca"
            })
        elif len(records) == 0:
            # Page found but no prices parsed
            all_results.append({
                "brand": brand,
                "model": model,
                "size": None,
                "price": None,
                "shop": SHOP,
                "note": f"Page found at {found_url} but prices could not be parsed"
            })
        else:
            all_results.extend(records)

        # 2-second delay between models (not between slug retries)
        if idx < total - 1:
            time.sleep(2)

    # Write results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    found_count = sum(1 for r in all_results if r.get('price') is not None)
    not_found_count = sum(1 for r in all_results if r.get('price') is None)
    print(f"\nDone. {found_count} prices found, {not_found_count} models/sizes not found.", flush=True)
    print(f"Results saved to: {OUTPUT_PATH}", flush=True)

    return all_results


if __name__ == '__main__':
    main()
