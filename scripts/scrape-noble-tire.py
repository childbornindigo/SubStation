#!/usr/bin/env python3
"""
Scrape Noble Tire Brampton for prices matching our inventory sizes.
Uses POST requests to https://nobletire.ca/list-tires
Table structure: #, Size, Manufacturer, Brand, Rating, Price, Type
Price cell has: <span style="color:red;float:left">$110.10</span> (sale price)
"""

import json
import re
import time
import sys
import requests
from bs4 import BeautifulSoup

BRANDS_WE_CARRY = {
    'bridgestone', 'continental', 'falken', 'firestone',
    'hankook', 'michelin', 'pirelli', 'toyo', 'yokohama'
}

BRAND_DISPLAY = {
    'bridgestone': 'Bridgestone',
    'continental': 'Continental',
    'falken': 'Falken',
    'firestone': 'Firestone',
    'hankook': 'Hankook',
    'michelin': 'Michelin',
    'pirelli': 'Pirelli',
    'toyo': 'Toyo',
    'yokohama': 'Yokohama',
}

INVENTORY_PATH = '/Users/indigochild/.hermes/extensions/substation/scripts/inventory-dump.json'
OUTPUT_PATH = '/Users/indigochild/.hermes/extensions/substation/scripts/local-prices-noble-tire.json'


def parse_sizes(inventory):
    """Extract unique width/profile/rim combos from inventory."""
    combos = set()
    pattern = re.compile(r'^(\d+)/(\d+)R(\d+)$')
    for item in inventory:
        m = pattern.match(item['size'])
        if m:
            width, profile, rim = m.group(1), m.group(2), m.group(3)
            combos.add((width, profile, rim))
    return sorted(combos)


def search_noble_tire(width, profile, rim):
    """POST to Noble Tire and return HTML response text."""
    resp = requests.post(
        'https://nobletire.ca/list-tires',
        data={
            'width': str(width),
            'profile': str(profile),
            'rim': str(rim),
            'season': 'all'
        },
        timeout=30,
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://nobletire.ca/',
        }
    )
    resp.raise_for_status()
    return resp.text


def parse_tires(html, width, profile, rim):
    """Parse Noble Tire HTML response, return list of matching tire dicts."""
    soup = BeautifulSoup(html, 'html.parser')
    standard_size = f"{width}/{profile}R{rim}"
    results = []

    # Check for no results
    body_text = soup.get_text()
    if 'TIRE NOT FOUND' in body_text and 'TIRES FOUND' not in body_text:
        return results

    # Find table inside div.table-tires
    table_div = soup.find('div', class_='table-tires')
    if not table_div:
        return results

    table = table_div.find('table')
    if not table:
        return results

    tbody = table.find('tbody')
    if not tbody:
        return results

    for row in tbody.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 6:
            continue

        # Columns: 0=#, 1=Size, 2=Manufacturer, 3=Brand, 4=Rating, 5=Price, 6=Type
        manufacturer = cells[2].get_text(strip=True).upper()
        brand_model = cells[3].get_text(strip=True)

        # Match manufacturer to brands we carry
        matched_brand = None
        for b, display in BRAND_DISPLAY.items():
            if b in manufacturer.lower():
                matched_brand = display
                break

        if not matched_brand:
            continue

        # Extract sale price: <span style="color:red;...">$110.10</span>
        price_cell = cells[5]
        red_span = price_cell.find('span', style=lambda s: s and 'color:red' in s)
        if red_span:
            price_text = red_span.get_text(strip=True)
        else:
            # Fallback: grab first price-like text
            price_text = price_cell.get_text(strip=True)

        price_match = re.search(r'\$?([\d,]+\.\d{2})', price_text)
        if not price_match:
            continue
        price = float(price_match.group(1).replace(',', ''))

        results.append({
            'brand': matched_brand,
            'model': brand_model.title(),
            'size': standard_size,
            'price': price,
            'shop': 'Noble Tire Brampton'
        })

    return results


def main():
    print("Loading inventory...")
    with open(INVENTORY_PATH) as f:
        inventory = json.load(f)

    combos = parse_sizes(inventory)
    total = len(combos)
    print(f"Found {total} unique size combos to search.")

    all_results = []
    errors = []

    for i, (width, profile, rim) in enumerate(combos, 1):
        size_label = f"{width}/{profile}R{rim}"
        print(f"  [{i:3d}/{total}] {size_label} ... ", end='', flush=True)

        try:
            html = search_noble_tire(width, profile, rim)
            tires = parse_tires(html, width, profile, rim)

            # Filter for brands we carry
            matching = [t for t in tires]  # already filtered in parse_tires
            print(f"{len(matching)} matches")

            all_results.extend(matching)

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({'size': size_label, 'error': str(e)})

        # Save progress every 25 combos
        if i % 25 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"    [Checkpoint saved: {len(all_results)} results so far]")

        # Polite delay
        if i < total:
            time.sleep(1)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} total results saved to {OUTPUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e['size']}: {e['error']}")


if __name__ == '__main__':
    main()
