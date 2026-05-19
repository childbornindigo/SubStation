#!/usr/bin/env python3
"""
Scrape TDot Performance and KRAVE Automotive for tire prices.

TDot: Scrape brand pages -> extract product links + titles (sizes in titles) ->
      visit each product page for price.

KRAVE: Scrape collection page -> get product slugs -> visit each product page
       for size/price tables.

Saves results to:
  local-prices-tdot-performance.json
  local-prices-krave.json
"""

import os
import sys
import json
import re
import time
from typing import Optional, List, Dict

# Firecrawl credits are exhausted — use Scrapling directly
# os.environ['FIRECRAWL_API_KEY'] = 'fc-2d1cf8b9d6e84fc797d360879036f290'
sys.path.insert(0, '/Users/indigochild/.hermes/extensions/substation/scripts')
from scrape_utils import _try_scrapling as _scrapling

def smart_scrape(url, **kwargs):
    """Use Scrapling directly since Firecrawl credits are exhausted."""
    return _scrapling(url)

SCRIPTS_DIR = '/Users/indigochild/.hermes/extensions/substation/scripts'
DELAY = 2  # seconds between requests

ALLOWED_BRANDS = {
    'bridgestone', 'continental', 'falken', 'firestone',
    'hankook', 'michelin', 'pirelli', 'toyo', 'yokohama'
}

BRAND_NAMES = {
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

# TDot shows C$XXX.XXea for each price
TDOT_PRICE_RE = re.compile(r'C\$\s*([\d,]+\.?\d+)ea')
# General C$ price (fallback)
CDOLLAR_RE = re.compile(r'C\$\s*([\d,]+\.?\d+)')

# Tire size patterns: standard and LT/P prefix, or numeric like 35X12.5R17
TIRE_SIZE_RE = re.compile(
    r'\b(?:(?:LT|P|ST)?\d{2,3}/\d{2}[RZBDzbd]\d{2}(?:C)?'
    r'|(?:LT)?\d{2}(?:\.\d+)?[Xx]\d{2}(?:\.\d+)?[RZ]\d{2}(?:C)?)\b'
)

# -------------------------------------------------------------------
# TDot Performance scraping
# -------------------------------------------------------------------

def parse_tdot_product_title(title: str) -> Dict:
    """
    Parse TDot product title like:
      'Michelin 17478 - Pilot Sport A/S 4 All Season Tires (245/45ZR20 103Y)'
    Returns {'brand': ..., 'model': ..., 'size': ...}
    """
    # Extract size from parentheses at the end
    size_paren = re.search(r'\(([^)]+)\)\s*$', title)
    raw_size = ''
    if size_paren:
        # The content is like "245/45ZR20 103Y" - get just the size part
        paren_content = size_paren.group(1)
        sizes = TIRE_SIZE_RE.findall(paren_content)
        if sizes:
            raw_size = sizes[0]

    # Try to get size from anywhere in title if paren didn't work
    if not raw_size:
        sizes = TIRE_SIZE_RE.findall(title)
        if sizes:
            raw_size = sizes[0]

    # Remove size-containing parenthetical and leading number at end
    clean_title = re.sub(r'\s*\([^)]+\)\s*$', '', title).strip()

    # First token should be the brand (e.g. "Michelin")
    # Format: "Brand SKUNUM - Model Name ..."
    # Try to split on " - " to separate SKU from model
    dash_parts = clean_title.split(' - ', 1)
    if len(dash_parts) == 2:
        brand_sku = dash_parts[0].strip()
        model_part = dash_parts[1].strip()
        # Brand is first word of brand_sku
        brand_word = brand_sku.split()[0] if brand_sku else ''
        brand_key = brand_word.lower()
        brand = BRAND_NAMES.get(brand_key, brand_word)
        # Clean up model: remove trailing "Tires", "Tire" or season descriptors
        model = re.sub(r'\s+(Tires?|All Season|Winter|All Weather|Summer|All-Season)\s*$', '', model_part, flags=re.IGNORECASE).strip()
        # Remove load/speed index suffix from model if it crept in
        model = re.sub(r'\s+\d{2,3}[A-Z]{1,2}\s*$', '', model).strip()
    else:
        # Fallback: first word is brand
        words = clean_title.split()
        brand = BRAND_NAMES.get(words[0].lower(), words[0]) if words else ''
        model = ' '.join(words[1:]).strip() if len(words) > 1 else ''
        model = re.sub(r'\s+(Tires?)\s*$', '', model, flags=re.IGNORECASE).strip()

    return {'brand': brand, 'model': model, 'size': raw_size}


def scrape_tdot_product_price(url: str) -> Optional[float]:
    """Scrape a TDot product page and return the per-unit price."""
    result = smart_scrape(url)
    if not result['success']:
        return None

    content = result['content']
    # Primary: C$XXX.XXea
    matches = TDOT_PRICE_RE.findall(content)
    if matches:
        # First match is the ea price
        price_str = matches[0].replace(',', '')
        try:
            return float(price_str)
        except ValueError:
            pass

    # Fallback: look for the price block pattern
    # After product title there's usually C$XXX.XX on its own line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        # Looking for a line that is just a C$ price
        m = re.match(r'^C\$\s*([\d,]+\.\d{2})\s*$', line)
        if m:
            price_str = m.group(1).replace(',', '')
            try:
                price = float(price_str)
                if 50 < price < 3000:
                    return price
            except ValueError:
                pass

    return None


def get_tdot_brand_page_products(brand_key: str, page: int = 1) -> List[Dict]:
    """
    Scrape one page of a TDot brand listing.
    Returns list of {'title': ..., 'url': ...}
    """
    brand_name = BRAND_NAMES[brand_key]
    if page == 1:
        url = f'https://www.tdotperformance.ca/brands/{brand_key}.html'
    else:
        url = f'https://www.tdotperformance.ca/brands/{brand_key}.html?p={page}'

    result = smart_scrape(url)
    if not result['success']:
        print(f'  [tdot] Failed to load {url}: {result["error"]}', flush=True)
        return []

    content = result['content']

    # Extract product links with titles
    # Format: [Title](https://www.tdotperformance.ca/products/slug.html)
    product_re = re.compile(
        r'\[([^\]]+tdotperformance[^\]]*|[^\]]*Tires?[^\]]*)\]'
        r'\((https://www\.tdotperformance\.ca/products/[^)\s]+\.html)\)'
    )
    links = product_re.findall(content)

    # Deduplicate by URL
    seen_urls = set()
    products = []
    for title, url in links:
        # Clean title (remove markdown escapes)
        clean_title = title.replace('\\n', ' ').replace('\\\\', '').strip()
        if url not in seen_urls and len(clean_title) > 5:
            seen_urls.add(url)
            products.append({'title': clean_title, 'url': url})

    # Check if we're on the last page
    has_next = bool(re.search(r'Next Page|p=' + str(page + 1), content))

    print(f'  [tdot/{brand_name}] Page {page}: {len(products)} products, has_next={has_next}', flush=True)
    return products


def scrape_tdot() -> List[Dict]:
    """
    Scrape TDot Performance brand pages.
    Pages 1-3 per brand to keep it tractable.
    """
    brands = ['michelin', 'bridgestone', 'continental', 'pirelli',
              'firestone', 'hankook', 'falken', 'toyo', 'yokohama']
    MAX_PAGES = 2  # Scrape up to 2 pages per brand (~40 products per brand)

    all_products = []  # {title, url, brand_key}

    for brand_key in brands:
        brand_name = BRAND_NAMES[brand_key]
        print(f'\n[tdot] Collecting {brand_name} product links...', flush=True)

        for page in range(1, MAX_PAGES + 1):
            if page > 1:
                time.sleep(DELAY)

            products = get_tdot_brand_page_products(brand_key, page)
            for p in products:
                p['brand_key'] = brand_key
            all_products.extend(products)

            # Check if page had results to continue
            if len(products) == 0:
                break

        time.sleep(DELAY)

    print(f'\n[tdot] Total products to price-check: {len(all_products)}', flush=True)

    # Now scrape each product page for price
    entries = []
    seen_keys = set()

    for i, product in enumerate(all_products):
        time.sleep(DELAY)

        parsed = parse_tdot_product_title(product['title'])
        brand = parsed['brand']
        model = parsed['model']
        size = parsed['size']

        # Skip if we couldn't parse meaningfully
        if not brand or not model or not size:
            print(f'  [tdot] Skip (parse failed): {product["title"][:60]}', flush=True)
            continue

        # Deduplicate by (brand, model, size)
        key = (brand.lower(), model.lower(), size.upper())
        if key in seen_keys:
            continue

        print(f'  [tdot] ({i+1}/{len(all_products)}) Pricing {brand} {model} {size}...', flush=True)

        price = scrape_tdot_product_price(product['url'])

        if price is not None:
            seen_keys.add(key)
            entries.append({
                'brand': brand,
                'model': model,
                'size': size,
                'price': price,
                'shop': 'TDot Performance',
            })
            print(f'    -> C${price}', flush=True)
        else:
            print(f'    -> price not found', flush=True)

    return entries


# -------------------------------------------------------------------
# KRAVE Automotive scraping
# -------------------------------------------------------------------

# From collection page scrape, these are the product slugs with allowed brands
KRAVE_PRODUCT_SLUGS = [
    ('toyo',     'toyo-open-country-a-t-iii-tires'),
    ('falken',   'falken-wildpeak-a-t4w-tires'),
    ('yokohama', 'yokohama-m-t-g003-tires'),
    ('yokohama', 'yokohama-geolandar-a-t4-g018-tires'),
    ('yokohama', 'yokohama-m-t-g003-tires-copy'),   # iceGUARD G075
    ('yokohama', 'yokohama-a-t-xd-go17-tires'),     # A/T XD G017
    ('yokohama', 'yokohama-geolandar-x-at-tires'),  # Geolandar X-AT
]


def parse_krave_product_page(content: str, brand: str) -> List[Dict]:
    """
    Parse KRAVE product page. Size/price table appears as:
      XX" - 285/70R17 - $513.00 CAD
    or in a table format.
    Also handles: 'from $XXX' with separate size variants listed.
    """
    entries = []

    # Get product model name from H1 (markdown) or first non-empty line (plain text)
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    model = ''
    raw_model = ''
    if h1_match:
        raw_model = h1_match.group(1).strip()
    else:
        # Scrapling may render the title as plain text on the first line
        for line in content.split('\n'):
            line = line.strip()
            if line and len(line) > 3 and not line.startswith('[') and not line.startswith('!'):
                raw_model = line
                break

    if raw_model:
        # Remove brand prefix
        brand_pattern = re.escape(brand)
        model = re.sub(brand_pattern, '', raw_model, flags=re.IGNORECASE).strip()
        model = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', model).strip()
        # Remove trailing "Tires" and shop name suffixes
        model = re.sub(r'\s+Tires?\s*$', '', model, flags=re.IGNORECASE).strip()
        model = re.sub(r'\s*[–-]\s*KRAVE\s+Automotive\s*$', '', model, flags=re.IGNORECASE).strip()

    if not model:
        return entries

    lines = content.split('\n')

    # Broader tire size pattern that also matches LT-prefix numeric sizes like LT35X12.50R17
    KRAVE_SIZE_RE = re.compile(
        r'\b(?:(?:LT|P|ST)?\d{2,3}/\d{2}[RZBDzbd]\d{2}(?:C)?'
        r'|(?:LT)?\d{2}(?:\.\d+)?[Xx]\d{2}(?:\.\d+)?[RZ]\d{2}(?:C)?)\b'
    )

    # Pattern 1: "XX" - SIZE - $PRICE CAD", "SIZE - $PRICE CAD", or "SIZE [text] - $PRICE CAD"
    size_price_re = re.compile(
        r'(?:\d+["x]?\s*-\s*)?'          # optional diameter prefix
        r'(' + KRAVE_SIZE_RE.pattern + r')'  # size
        r'[^$\n]*?'                          # any chars (variant labels) up to price
        r'\$\s*([\d,]+\.?\d*)'              # price
    )

    for line in lines:
        line = line.strip()
        m = size_price_re.search(line)
        if m:
            size = m.group(1).strip()
            price_str = m.group(2).replace(',', '')
            try:
                price = float(price_str)
                if 50 < price < 3000:
                    entries.append({
                        'brand': brand,
                        'model': model,
                        'size': size,
                        'price': price,
                        'shop': 'KRAVE',
                    })
            except ValueError:
                pass

    # Pattern 2: Table | SIZE | $PRICE |
    for line in lines:
        if '|' in line:
            cols = [c.strip() for c in line.split('|') if c.strip()]
            row_sizes = []
            row_price = None
            for col in cols:
                sizes = TIRE_SIZE_RE.findall(col)
                prices = re.findall(r'\$\s*([\d,]+\.?\d+)', col)
                row_sizes.extend(sizes)
                if prices:
                    row_price = float(prices[0].replace(',', ''))
            if row_sizes and row_price and 50 < row_price < 3000:
                for size in row_sizes:
                    entries.append({
                        'brand': brand,
                        'model': model,
                        'size': size,
                        'price': row_price,
                        'shop': 'KRAVE',
                    })

    return entries


def scrape_krave() -> List[Dict]:
    """Scrape KRAVE Automotive product pages for our allowed brands."""
    all_entries = []
    seen_keys = set()

    # First, also check collection page for any brands we may have missed
    print('\n[krave] Checking collection page for product list...', flush=True)
    coll_result = smart_scrape('https://kraveautomotive.ca/collections/tires')
    if coll_result['success']:
        content = coll_result['content']
        # Find all product links in the collection
        coll_links = re.findall(
            r'((?:Yokohama|Toyo|Falken|Bridgestone|Continental|Firestone|Hankook|Michelin|Pirelli)[^\n\\]+?Tires?)\\\\n\\\\nfrom \$([\d.]+)\]\(https://kraveautomotive\.ca/collections/tires/products/([^)]+)\)',
            content
        )
        for title, price_from, slug in coll_links:
            print(f'  Found: {title.strip()[:60]} from ${price_from} -> {slug}', flush=True)

    time.sleep(DELAY)

    for brand_key, slug in KRAVE_PRODUCT_SLUGS:
        brand = BRAND_NAMES[brand_key]
        url = f'https://kraveautomotive.ca/products/{slug}'
        print(f'\n[krave] Scraping {brand} ({slug})...', flush=True)

        result = smart_scrape(url)
        if not result['success']:
            print(f'  FAILED: {result["error"]}', flush=True)
        else:
            entries = parse_krave_product_page(result['content'], brand)
            new_count = 0
            for e in entries:
                key = (e['brand'].lower(), e['model'].lower(), e['size'].upper())
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_entries.append(e)
                    new_count += 1
            print(f'  Found {new_count} size/price entries', flush=True)
            for e in entries:
                print(f'    {e["size"]} -> ${e["price"]}', flush=True)

        time.sleep(DELAY)

    return all_entries


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def save_results(entries: List[Dict], path: str, shop_name: str):
    """Save entries to JSON with summary."""
    # Filter to allowed brands only
    filtered = [e for e in entries if e['brand'].lower() in ALLOWED_BRANDS]

    # Clean schema
    output = [
        {
            'brand': e['brand'],
            'model': e['model'],
            'size': e['size'],
            'price': e['price'],
            'shop': e['shop'],
        }
        for e in filtered
    ]

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n[{shop_name}] Saved {len(output)} entries to {path}', flush=True)
    from collections import Counter
    brand_counts = Counter(e['brand'] for e in output)
    for brand, count in sorted(brand_counts.items()):
        print(f'  {brand}: {count} entries', flush=True)


if __name__ == '__main__':
    print('=== Scraping TDot Performance ===', flush=True)
    tdot_entries = scrape_tdot()
    tdot_path = os.path.join(SCRIPTS_DIR, 'local-prices-tdot-performance.json')
    save_results(tdot_entries, tdot_path, 'TDot Performance')

    print('\n=== Scraping KRAVE Automotive ===', flush=True)
    krave_entries = scrape_krave()
    krave_path = os.path.join(SCRIPTS_DIR, 'local-prices-krave.json')
    save_results(krave_entries, krave_path, 'KRAVE Automotive')

    print('\nDone.', flush=True)
