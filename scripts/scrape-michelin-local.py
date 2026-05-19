#!/usr/bin/env python3
"""
Scrape 6 local GTA tire shops for Michelin tire pricing on 3 models:
  1. Michelin Pilot Sport AS 4
  2. Michelin Primacy Tour AS
  3. Michelin Defender LTX M/S2

Shops: Point S, TDot Performance, Active Green + Ross, Noble Tire,
       ZRacing, Canadian Tire

Uses smart_scrape (Firecrawl + ScrapLing fallback).
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
OUTPUT_PATH = os.path.join(SCRIPTS_DIR, 'local-prices-michelin.json')
DELAY = 3  # seconds between requests

# ── Models we want ──
MICHELIN_MODELS = [
    'Pilot Sport AS 4',
    'Primacy Tour AS',
    'Defender LTX M/S2',
]

# ── URL plans per shop ──
# Each shop entry: (shop_display_name, list of (model_display, [urls_to_try]))

SHOP_URLS = {
    'Point S': {
        'Pilot Sport AS 4': [
            'https://point-s.ca/en/tires/michelin/pilot-sport-a-s-4',
            'https://point-s.ca/en/tires/michelin/pilot-sport-as-4',
            'https://point-s.ca/en/tires/michelin/pilot-sport-all-season-4',
        ],
        'Primacy Tour AS': [
            'https://point-s.ca/en/tires/michelin/primacy-tour-a-s',
            'https://point-s.ca/en/tires/michelin/primacy-tour-as',
        ],
        'Defender LTX M/S2': [
            'https://point-s.ca/en/tires/michelin/defender-ltx-m-s-2',
            'https://point-s.ca/en/tires/michelin/defender-ltx-ms2',
            'https://point-s.ca/en/tires/michelin/defender-ltx-m-s2',
        ],
    },
    'TDot Performance': {
        'Pilot Sport AS 4': [
            'https://www.tdotperformance.ca/michelin-pilot-sport-all-season-4-tires',
            'https://www.tdotperformance.ca/search?q=michelin+pilot+sport+as+4',
            'https://www.tdotperformance.ca/tires/michelin?q=pilot+sport+as+4',
        ],
        'Primacy Tour AS': [
            'https://www.tdotperformance.ca/michelin-primacy-tour-a-s-tires',
            'https://www.tdotperformance.ca/search?q=michelin+primacy+tour+as',
        ],
        'Defender LTX M/S2': [
            'https://www.tdotperformance.ca/michelin-defender-ltx-m-s-2-tires',
            'https://www.tdotperformance.ca/search?q=michelin+defender+ltx+ms2',
        ],
    },
    'Active Green + Ross': {
        'Pilot Sport AS 4': [
            'https://www.activegreenross.com/tires/search?brand=michelin&model=pilot+sport+as+4',
            'https://www.activegreenross.com/tires/michelin',
        ],
        'Primacy Tour AS': [
            'https://www.activegreenross.com/tires/search?brand=michelin&model=primacy+tour+as',
            'https://www.activegreenross.com/tires/michelin',
        ],
        'Defender LTX M/S2': [
            'https://www.activegreenross.com/tires/search?brand=michelin&model=defender+ltx+ms2',
            'https://www.activegreenross.com/tires/michelin',
        ],
    },
    'Noble Tire': {
        'Pilot Sport AS 4': [
            'https://nobletire.ca/search?q=michelin+pilot+sport+as+4',
            'https://nobletire.ca/collections/michelin',
        ],
        'Primacy Tour AS': [
            'https://nobletire.ca/search?q=michelin+primacy+tour+as',
        ],
        'Defender LTX M/S2': [
            'https://nobletire.ca/search?q=michelin+defender+ltx+ms2',
        ],
    },
    'ZRacing': {
        'Pilot Sport AS 4': [
            'https://zracing.ca/search?q=michelin+pilot+sport+as+4',
            'https://zracing.ca/collections/michelin-tires',
        ],
        'Primacy Tour AS': [
            'https://zracing.ca/search?q=michelin+primacy+tour+as',
        ],
        'Defender LTX M/S2': [
            'https://zracing.ca/search?q=michelin+defender+ltx+ms2',
        ],
    },
    'Canadian Tire': {
        'Pilot Sport AS 4': [
            'https://www.canadiantire.ca/en/search?q=michelin+pilot+sport+as+4',
            'https://www.canadiantire.ca/en/automotive/tires/all-season-tires.html?brand=michelin',
        ],
        'Primacy Tour AS': [
            'https://www.canadiantire.ca/en/search?q=michelin+primacy+tour+as',
        ],
        'Defender LTX M/S2': [
            'https://www.canadiantire.ca/en/search?q=michelin+defender+ltx+ms2',
        ],
    },
}


def extract_tire_size(text):
    """Extract tire size patterns like 225/45R17, 265/70R17, etc."""
    pattern = r'(\d{3}/\d{2,3}[RZB]\d{2}C?)'
    return re.findall(pattern, text)


def extract_prices(text):
    """Extract prices from text. Handles $xxx.xx, xxx.xx$, $x,xxx.xx formats."""
    prices = []
    # $xxx.xx format (most common)
    for m in re.finditer(r'\$\s*(\d{1,4}(?:,\d{3})?(?:\.\d{2})?)', text):
        p = float(m.group(1).replace(',', ''))
        if 50 < p < 3000:
            prices.append((m.start(), p))
    # xxx.xx$ format (Point S)
    for m in re.finditer(r'(\d{3,4}(?:\.\d{2})?)\$', text):
        p = float(m.group(1))
        if 50 < p < 3000:
            prices.append((m.start(), p))
    return prices


def parse_tire_listings(content, model_name, shop_name):
    """
    Generic parser: tries multiple strategies to extract size+price pairs
    from scraped markdown content.
    """
    results = []
    if not content:
        return results

    model_lower = model_name.lower()
    # Model name variants for matching
    model_variants = [model_lower]
    if 'as 4' in model_lower:
        model_variants.extend(['pilot sport a/s 4', 'pilot sport all season 4', 'pilot sport as4', 'ps as 4', 'ps4s', 'pilot sport 4 all season'])
    elif 'primacy tour' in model_lower:
        model_variants.extend(['primacy tour a/s', 'primacy tour all season'])
    elif 'defender ltx' in model_lower:
        model_variants.extend(['defender ltx m/s 2', 'defender ltx ms2', 'defender ltx m/s2', 'defender ltx m/s 2'])

    # ── Strategy 1: Point S concatenated format ──
    # "235/60R18 XL 107H330.99$ per tire"
    ps_re = re.compile(
        r'(\d{3}/\d{2}[RZB]\d{2}C?'
        r'(?:\s+(?:XL|RF|OWL|BSW|SL|LL))*'
        r'(?:\s+\d{2,3}[A-Z])?)'
        r'(\d{3,4}(?:\.\d{2})?)\$'
    )
    for size_str, price_str in ps_re.findall(content):
        size_clean = re.match(r'(\d{3}/\d{2}[RZB]\d{2}C?)', size_str.strip()).group(1)
        price = float(price_str)
        if 50 < price < 3000:
            results.append({'size': size_clean, 'price': price})

    if results:
        return dedupe_by_size(results)

    # ── Strategy 2: Line-by-line size+price proximity ──
    lines = content.split('\n')
    for i, line in enumerate(lines):
        sizes = extract_tire_size(line)
        if not sizes:
            continue

        # Look for price on the same line
        line_prices = extract_prices(line)
        if line_prices:
            results.append({'size': sizes[0], 'price': line_prices[0][1]})
            continue

        # Look for price on nearby lines (within 3 lines)
        for j in range(i + 1, min(i + 4, len(lines))):
            nearby_prices = extract_prices(lines[j])
            if nearby_prices:
                results.append({'size': sizes[0], 'price': nearby_prices[0][1]})
                break

    if results:
        return dedupe_by_size(results)

    # ── Strategy 3: Product card blocks ──
    # Some sites use markdown blocks like "### Product Name\n...225/45R17...\n...$289.99..."
    blocks = re.split(r'\n(?=#{1,4}\s)', content)
    for block in blocks:
        block_lower = block.lower()
        # Check if block is relevant to our model
        if not any(v in block_lower for v in model_variants) and not any(s in block_lower for s in ['michelin']):
            continue
        sizes = extract_tire_size(block)
        prices = extract_prices(block)
        if sizes and prices:
            # Match closest size-price pairs
            for size in sizes:
                if prices:
                    results.append({'size': size, 'price': prices[0][1]})

    if results:
        return dedupe_by_size(results)

    # ── Strategy 4: Table format ──
    # | Size | Price | ... or similar
    table_row_re = re.compile(r'\|[^|]*(\d{3}/\d{2}[RZB]\d{2})[^|]*\|[^|]*\$?\s*(\d{1,4}(?:,\d{3})?(?:\.\d{2})?)[^|]*\|')
    for m in table_row_re.finditer(content):
        size = m.group(1)
        price = float(m.group(2).replace(',', ''))
        if 50 < price < 3000:
            results.append({'size': size, 'price': price})

    return dedupe_by_size(results)


def dedupe_by_size(results):
    """Keep first occurrence per size."""
    seen = set()
    deduped = []
    for item in results:
        if item['size'] not in seen:
            seen.add(item['size'])
            deduped.append(item)
    return deduped


def try_urls_for_model(shop_name, model_name, urls):
    """Try each URL candidate until we get tire pricing data."""
    for url in urls:
        print(f"  Trying: {url}", file=sys.stderr)
        result = smart_scrape(url)

        if not result['success']:
            print(f"    FAIL: {result.get('error', 'unknown')}", file=sys.stderr)
            time.sleep(DELAY)
            continue

        content = result['content']
        if not content or len(content) < 200:
            print(f"    Too short ({len(content or '')} chars)", file=sys.stderr)
            time.sleep(DELAY)
            continue

        # Check for 404
        content_lower = content.lower()
        if any(sig in content_lower for sig in ['404', 'page not found', "couldn't find"]) and len(content) < 3000:
            print(f"    404/not found", file=sys.stderr)
            time.sleep(DELAY)
            continue

        # Try to parse tire data
        listings = parse_tire_listings(content, model_name, shop_name)
        if listings:
            print(f"    SUCCESS: {len(listings)} sizes found", file=sys.stderr)
            return listings, content

        # Even if no parsed listings, check if there's tire-relevant content
        has_size = bool(re.search(r'\d{3}/\d{2}[RZ]\d{2}', content))
        has_price = bool(re.search(r'[\$]\s*\d{2,4}', content)) or bool(re.search(r'\d{3,4}\.\d{2}\$', content))

        if has_size and has_price:
            print(f"    Has size+price data but parser couldn't extract structured listings", file=sys.stderr)
            # Return the content so we can do a second-pass parse
            return [], content
        elif has_size:
            print(f"    Has sizes but no prices visible", file=sys.stderr)
        else:
            print(f"    No tire data found", file=sys.stderr)

        time.sleep(DELAY)

    return [], None


def main():
    all_results = []
    shop_counts = {}
    model_counts = {m: 0 for m in MICHELIN_MODELS}

    for shop_name, model_urls in SHOP_URLS.items():
        shop_total = 0
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"SHOP: {shop_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        for model_name in MICHELIN_MODELS:
            urls = model_urls.get(model_name, [])
            if not urls:
                print(f"\n  [{model_name}] No URLs configured", file=sys.stderr)
                continue

            print(f"\n  [{model_name}]", file=sys.stderr)
            listings, raw_content = try_urls_for_model(shop_name, model_name, urls)

            if listings:
                for item in listings:
                    all_results.append({
                        'shop': shop_name,
                        'brand': 'Michelin',
                        'model': model_name,
                        'size': item['size'],
                        'price': item['price'],
                    })
                shop_total += len(listings)
                model_counts[model_name] += len(listings)
            else:
                note = 'No pricing data found'
                if raw_content:
                    # Check for specific issues
                    cl = raw_content.lower()
                    if any(kw in cl for kw in ['select location', 'choose location', 'enter your postal', 'find a store']):
                        note = 'Pricing requires location selection'
                    elif re.search(r'\d{3}/\d{2}[RZ]\d{2}', raw_content):
                        note = 'Sizes found but no prices parseable'

                all_results.append({
                    'shop': shop_name,
                    'brand': 'Michelin',
                    'model': model_name,
                    'size': 'NOT FOUND',
                    'price': None,
                    'note': note,
                })
                print(f"    -> 0 sizes ({note})", file=sys.stderr)

            time.sleep(DELAY)

        shop_counts[shop_name] = shop_total

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    print(f"\nPer Shop:", file=sys.stderr)
    for shop, count in shop_counts.items():
        print(f"  {shop}: {count} prices", file=sys.stderr)

    print(f"\nPer Model:", file=sys.stderr)
    for model, count in model_counts.items():
        print(f"  Michelin {model}: {count} prices", file=sys.stderr)

    total = sum(shop_counts.values())
    print(f"\nTotal: {total} price records", file=sys.stderr)
    print(f"Output: {OUTPUT_PATH}", file=sys.stderr)
    print(f"\nDone. {total} price records written to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
