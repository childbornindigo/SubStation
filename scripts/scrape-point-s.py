#!/usr/bin/env python3
"""
Scrape Point S Canada for tire prices.
Uses Firecrawl (via smart_scrape) to fetch model pages and parse size/price tables.
"""

import os
import sys
import json
import re
import time

os.environ['FIRECRAWL_API_KEY'] = 'fc-2d1cf8b9d6e84fc797d360879036f290'
sys.path.insert(0, '/Users/indigochild/.hermes/extensions/substation/scripts')
from scrape_utils import smart_scrape

SCRIPTS_DIR = '/Users/indigochild/.hermes/extensions/substation/scripts'
OUTPUT_PATH = os.path.join(SCRIPTS_DIR, 'local-prices-point-s.json')
DELAY = 2  # seconds between requests
BASE_URL = 'https://point-s.ca/en/tires'

MODELS = [
    # brand,  display_model_name,  slug_candidates
    ('Bridgestone', 'Alenza Sport A/S',          ['alenza-sport-a-s', 'alenza-sport-as']),
    ('Bridgestone', 'Dueler HL Alenza Plus',      ['dueler-h-l-alenza-plus', 'dueler-hl-alenza-plus', 'dueler-h-l-alenza']),
    ('Bridgestone', 'Turanza Everdrive',          ['turanza-everdrive']),
    ('Bridgestone', 'UltraWeather',               ['ultraweather', 'ultra-weather']),
    ('Bridgestone', 'WeatherPeak',                ['weatherpeak', 'weather-peak']),
    ('Continental', 'CrossContact LX Sport',      ['crosscontact-lx-sport', 'cross-contact-lx-sport']),
    ('Continental', 'CrossContact RX',            ['crosscontact-rx', 'cross-contact-rx']),
    ('Continental', 'ExtremeContact DWS06+',      ['extremecontact-dws06-plus', 'extremecontact-dws06plus', 'extreme-contact-dws06-plus']),
    ('Continental', 'ProContact RX',              ['procontact-rx', 'pro-contact-rx']),
    ('Falken',      'Azenis FK460 A/S',           ['azenis-fk460-a-s', 'azenis-fk460-as', 'azenis-fk460']),
    ('Firestone',   'Destination LE 2',           ['destination-le-2', 'destination-le2']),
    ('Firestone',   'Destination LE3',            ['destination-le3', 'destination-le-3']),
    ('Firestone',   'FR710',                      ['fr710', 'fr-710']),
    ('Firestone',   'Weathergrip',                ['weathergrip', 'weather-grip']),
    ('Hankook',     'Dynapro evo AS',             ['dynapro-evo-a-s', 'dynapro-evo-as', 'dynapro-evo']),
    ('Hankook',     'Kinergy GT',                 ['kinergy-gt']),
    ('Hankook',     'Ventus S1 noble2',           ['ventus-s1-noble2', 'ventus-s1-evo2', 'ventus-s1-noble-2']),
    ('Michelin',    'Defender LTX M/S2',          ['defender-ltx-m-s-2', 'defender-ltx-ms2', 'defender-ltx-m-s2']),
    ('Michelin',    'Pilot Sport AS 4',           ['pilot-sport-a-s-4', 'pilot-sport-as-4', 'pilot-sport-all-season-4']),
    ('Michelin',    'Primacy Tour A/S',           ['primacy-tour-a-s', 'primacy-tour-as']),
    ('Pirelli',     'P Zero AS Plus 3',           ['p-zero-a-s-plus-3', 'p-zero-as-plus-3', 'p-zero-all-season-plus-3']),
    ('Pirelli',     'Scorpion Zero All Season',   ['scorpion-zero-all-season', 'scorpion-zero-as']),
    ('Toyo',        'Extensa AS II',              ['extensa-a-s-ii', 'extensa-as-ii', 'extensa-a-s-2']),
    ('Toyo',        'Open Country A/T III',       ['open-country-a50', 'open-country-at-iii', 'open-country-a-t-iii', 'open-country-at3']),
    ('Yokohama',    'Geolandar X-AT',             ['geolandar-x-at', 'geolandar-x-at-g016']),
    ('Yokohama',    'Geolandar X-CV',             ['geolandar-x-cv']),
]

# Brand slug mapping for URL construction
BRAND_SLUG = {
    'Bridgestone': 'bridgestone',
    'Continental': 'continental',
    'Falken':      'falken',
    'Firestone':   'firestone',
    'Hankook':     'hankook',
    'Michelin':    'michelin',
    'Pirelli':     'pirelli',
    'Toyo':        'toyo',
    'Yokohama':    'yokohama',
}


def parse_sizes_and_prices(markdown_text):
    """
    Parse tire size/price rows from Point S markdown content.

    Point S format (from Firecrawl rendering):
      - 235/60R18 XL 107H330.99$ per tire
    The size spec (including optional XL/RF and load code like 107H) is immediately
    followed by the price digits, then a literal '$' sign.
    """
    results = []
    if not markdown_text:
        return results

    # Primary pattern: Point S concatenated format  "235/60R18 XL 107H330.99$"
    primary_re = re.compile(
        r'(\d{3}/\d{2}[RZB]\d{2}C?'       # base size e.g. 235/60R18
        r'(?:\s+(?:XL|RF|OWL|BSW|SL|LL))*' # optional modifiers
        r'(?:\s+\d{2,3}[A-Z])?)'            # optional load/speed code e.g. 107H
        r'(\d{3,4}(?:\.\d{2})?)\$'          # price digits immediately before $
    )

    # Fallback: traditional "$xxx.xx" format (prefix dollar sign)
    fallback_size_re = re.compile(r'(\d{3}/\d{2}[RZB]\d{2}C?(?:\s+(?:XL|RF|OWL|BSW|SL|LL))*(?:\s+\d{2,3}[A-Z])?)')
    fallback_price_re = re.compile(r'\$\s*(\d{1,4}(?:,\d{3})?(?:\.\d{2})?)')

    # Try primary pattern on the full text first
    for size_str, price_str in primary_re.findall(markdown_text):
        size_clean = size_str.strip()
        price = float(price_str)
        if 50 < price < 3000:
            results.append({'size': size_clean, 'price': price})

    # If primary found nothing, try fallback line-by-line
    if not results:
        lines = markdown_text.split('\n')
        for i, line in enumerate(lines):
            sizes_found = fallback_size_re.findall(line)
            prices_found = fallback_price_re.findall(line)

            if sizes_found and prices_found:
                size_clean = sizes_found[0].strip()
                price_str = prices_found[0].replace(',', '')
                price = float(price_str)
                if 50 < price < 3000:
                    results.append({'size': size_clean, 'price': price})
            elif sizes_found and not prices_found:
                for j in range(i + 1, min(i + 4, len(lines))):
                    surrounding_prices = fallback_price_re.findall(lines[j])
                    if surrounding_prices:
                        size_clean = sizes_found[0].strip()
                        price_str = surrounding_prices[0].replace(',', '')
                        price = float(price_str)
                        if 50 < price < 3000:
                            results.append({'size': size_clean, 'price': price})
                        break

    # Deduplicate by size+price (keep first occurrence per size)
    seen = {}
    deduped = []
    for item in results:
        if item['size'] not in seen:
            seen[item['size']] = True
            deduped.append(item)

    return deduped


def fetch_model_page(brand, model_name, slugs):
    """
    Try each slug candidate until we get a successful scrape.
    Returns (url_used, markdown_content) or (None, None) on failure.
    """
    brand_slug = BRAND_SLUG.get(brand, brand.lower())

    for slug in slugs:
        url = f"{BASE_URL}/{brand_slug}/{slug}"
        print(f"  Trying: {url}", file=sys.stderr)

        result = smart_scrape(url)

        if not result['success']:
            print(f"    FAIL: {result.get('error', 'unknown error')}", file=sys.stderr)
            time.sleep(DELAY)
            continue

        content = result['content']

        # Check for 404 / not found signals
        not_found_signals = [
            '404',
            'page not found',
            'not found',
            "couldn't find",
            "page doesn't exist",
        ]
        content_lower = content.lower()
        if any(sig in content_lower for sig in not_found_signals) and len(content) < 3000:
            print(f"    404/not found: {url}", file=sys.stderr)
            time.sleep(DELAY)
            continue

        # Check if we got useful tire content
        # Point S uses "330.99$ per tire" (price before $ sign), so check both formats
        has_price = bool(re.search(r'\d{3,4}\.\d{2}\$', content)) or bool(re.search(r'\$\s*\d{2,4}', content))
        has_size = bool(re.search(r'\d{3}/\d{2}[RZ]\d{2}', content))

        if has_price and has_size:
            print(f"    SUCCESS: found prices+sizes at {url}", file=sys.stderr)
            return url, content
        elif has_size and not has_price:
            print(f"    Partial (sizes but no prices - location gated?): {url}", file=sys.stderr)
            # Still return it so we can note it
            return url, content
        else:
            print(f"    No tire data found at {url}", file=sys.stderr)
            time.sleep(DELAY)
            continue

    return None, None


def main():
    all_results = []
    summary = []

    total = len(MODELS)
    for idx, (brand, model_name, slugs) in enumerate(MODELS, 1):
        print(f"\n[{idx}/{total}] {brand} {model_name}", file=sys.stderr)

        url_used, content = fetch_model_page(brand, model_name, slugs)

        if content is None:
            print(f"  ALL SLUGS FAILED for {brand} {model_name}", file=sys.stderr)
            all_results.append({
                'brand': brand,
                'model': model_name,
                'size': 'NOT FOUND',
                'price': None,
                'shop': 'Point S',
                'note': 'All slug candidates failed - model may not be on point-s.ca'
            })
            summary.append((brand, model_name, 0, 'all slugs failed'))
            time.sleep(DELAY)
            continue

        # Parse sizes and prices
        size_prices = parse_sizes_and_prices(content)

        if not size_prices:
            # Check if it's location-gated or genuinely missing
            content_lower = content.lower()
            has_dollar_suffix = bool(re.search(r'\d{3,4}\.\d{2}\$', content))
            if any(kw in content_lower for kw in ['select location', 'choose location', 'find a store', 'location selector', 'enter your postal']):
                note = 'Model exists but pricing requires location selection'
            elif re.search(r'\d{3}/\d{2}[RZ]\d{2}', content) and has_dollar_suffix:
                note = 'Sizes and prices present but regex failed to parse - check page structure'
            elif re.search(r'\d{3}/\d{2}[RZ]\d{2}', content):
                note = 'Sizes listed but no prices found - may be location-gated'
            else:
                note = 'No size/price data found in page content'

            print(f"  No prices parsed: {note}", file=sys.stderr)
            all_results.append({
                'brand': brand,
                'model': model_name,
                'size': 'NOT FOUND - pricing not indexed',
                'price': None,
                'shop': 'Point S',
                'note': note
            })
            summary.append((brand, model_name, 0, note))
        else:
            for sp in size_prices:
                all_results.append({
                    'brand': brand,
                    'model': model_name,
                    'size': sp['size'],
                    'price': sp['price'],
                    'shop': 'Point S',
                })
            print(f"  -> {len(size_prices)} sizes found", file=sys.stderr)
            summary.append((brand, model_name, len(size_prices), 'ok'))

        time.sleep(DELAY)

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== SUMMARY ===", file=sys.stderr)
    total_records = 0
    for brand, model, count, status in summary:
        print(f"  {brand} {model}: {count} sizes ({status})", file=sys.stderr)
        total_records += count

    print(f"\nTotal: {total_records} price records across {len(MODELS)} models", file=sys.stderr)
    print(f"Output: {OUTPUT_PATH}", file=sys.stderr)

    # Also print summary to stdout for visibility
    print(f"\nDone. {total_records} price records written to {OUTPUT_PATH}")
    return all_results


if __name__ == '__main__':
    main()
