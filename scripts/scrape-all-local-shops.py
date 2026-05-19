#!/usr/bin/env python3
"""
Scrape all 8 confirmed local GTA shops for tire pricing.
Uses Firecrawl (API) first, falls back to ScrapLing (stealth browser).

Shops:
1. Canadian Tire (canadiantire.ca) — JS SPA, needs stealth
2. TDot Performance (tdotperformance.ca)
3. Active Green + Ross (activegreenross.com)
4. Point S (point-s.ca)
5. KRAVE Automotive (kraveautomotive.com)
6. Noble Tire (nobletire.ca) — Brampton
7. Fas-Tire (fastire.ca) — Scarborough
8. ZRacing (zracing.ca) — Mississauga
"""

import json
import os
import sys
import re
import time
import subprocess
import tempfile

SCRIPTS_DIR = "/Users/indigochild/.hermes/extensions/substation/scripts"
SCRAPLING_BIN = "/opt/homebrew/bin/scrapling"

sys.path.insert(0, SCRIPTS_DIR)
from scrape_utils import smart_scrape

BRANDS = [
    "Bridgestone", "Continental", "Falken", "Firestone",
    "Hankook", "Michelin", "Pirelli", "Toyo", "Yokohama"
]

MODELS_BY_BRAND = {
    "Bridgestone": ["Alenza Sport AS", "Dueler HL Alenza Plus", "Turanza Everdrive", "UltraWeather", "WeatherPeak"],
    "Continental": ["CrossContact LX Sport", "CrossContact RX", "ExtremeContact DWS06 Plus", "ProContact RX"],
    "Falken": ["Azenis FK460 AS"],
    "Firestone": ["Destination LE 2", "Destination LE3", "FR710", "Weathergrip"],
    "Hankook": ["Dynapro evo AS", "Kinergy GT", "Ventus S1 noble2"],
    "Michelin": ["Defender LTX M/S2", "Pilot Sport AS 4", "Primacy Tour AS"],
    "Pirelli": ["P Zero AS Plus 3", "Scorpion Zero All Season"],
    "Toyo": ["Extensa AS II", "Open Country A50"],
    "Yokohama": ["Geolandar X-AT", "Geolandar X-CV"],
}

SIZE_RE = re.compile(r'[PL]?T?\s*(\d{3})\s*/\s*(\d{2,3})\s*[RZ]\s*(\d{2})', re.IGNORECASE)
PRICE_RE = re.compile(r'\$\s*(\d{2,4}(?:\.\d{2})?)')
LT_SIZE_RE = re.compile(r'(\d{2,3})\s*[xX]\s*(\d{1,2}(?:\.\d{1,2})?)\s*[RZ]?\s*(\d{2})', re.IGNORECASE)


def normalize_size(s):
    s = s.strip().upper()
    for prefix in ["P", "LT"]:
        if s.startswith(prefix) and len(s) > len(prefix) and s[len(prefix)].isdigit():
            s = s[len(prefix):]
    s = s.replace(" ", "")
    return s


def extract_tires_from_text(text, brand=None, model=None):
    """Parse scraped text for tire sizes and prices."""
    results = []
    if not text:
        return results

    lines = text.split('\n')
    for i, line in enumerate(lines):
        sizes = SIZE_RE.findall(line)
        lt_sizes = LT_SIZE_RE.findall(line)
        prices = PRICE_RE.findall(line)

        # Also check 2 lines ahead/behind for prices
        context = line
        for offset in [1, 2, -1, -2]:
            idx = i + offset
            if 0 <= idx < len(lines):
                context += " " + lines[idx]

        if not prices:
            prices = PRICE_RE.findall(context)

        all_sizes = []
        for w, ar, rim in sizes:
            all_sizes.append(f"{w}/{ar}R{rim}")
        for w, ar, rim in lt_sizes:
            all_sizes.append(f"{w}X{ar}R{rim}")

        if all_sizes and prices:
            price = float(prices[0])
            if 50 < price < 3000:
                for size in all_sizes:
                    results.append({
                        "size": normalize_size(size),
                        "price": price,
                        "brand": brand or "",
                        "model": model or "",
                        "raw_line": line.strip()[:200],
                    })

    return results


def scrape_shop(name, url_builder, use_stealth=False, delay=2.0):
    """Scrape a shop for all brands/models. Returns list of {brand, model, size, price}."""
    all_results = []
    for brand in BRANDS:
        for model in MODELS_BY_BRAND.get(brand, []):
            url = url_builder(brand, model)
            print(f"[{name}] Searching: {brand} {model}", file=sys.stderr)

            result = smart_scrape(url, stealth=use_stealth, timeout=30000)
            if result["success"] and result["content"]:
                entries = extract_tires_from_text(result["content"], brand, model)
                for e in entries:
                    e["shop"] = name
                    if not e["brand"]:
                        e["brand"] = brand
                    if not e["model"]:
                        e["model"] = model
                all_results.extend(entries)
                print(f"  -> {len(entries)} prices found", file=sys.stderr)
            else:
                print(f"  -> FAILED: {result.get('error', 'unknown')}", file=sys.stderr)

            time.sleep(delay)

    return all_results


def scrape_canadian_tire():
    """Canadian Tire — JS SPA, search by brand+model."""
    def url_builder(brand, model):
        q = f"{brand} {model} tire".replace(" ", "+")
        return f"https://www.canadiantire.ca/en/search-results.html?q={q}"
    return scrape_shop("Canadian Tire", url_builder, use_stealth=True, delay=3.0)


def scrape_tdot_performance():
    """TDot Performance — standard search."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://www.tdotperformance.ca/catalogsearch/result/?q={q}"
    return scrape_shop("TDot Performance", url_builder, delay=2.0)


def scrape_active_green_ross():
    """Active Green + Ross — search tires."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://www.activegreenross.com/search?q={q}"
    return scrape_shop("Active Green + Ross", url_builder, use_stealth=True, delay=2.0)


def scrape_point_s():
    """Point S Canada — tire search."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://point-s.ca/en/catalogsearch/result/?q={q}"
    return scrape_shop("Point S", url_builder, delay=2.0)


def scrape_krave():
    """KRAVE Automotive — search."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://www.kraveautomotive.com/?s={q}"
    return scrape_shop("KRAVE", url_builder, delay=2.0)


def scrape_noble_tire():
    """Noble Tire Brampton — search."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://nobletire.ca/?s={q}"
    return scrape_shop("Noble Tire Brampton", url_builder, delay=2.0)


def scrape_fas_tire():
    """Fas-Tire Scarborough — search."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://fas-tire.ca/?s={q}"
    return scrape_shop("Fas-Tire Scarborough", url_builder, delay=2.0)


def scrape_zracing():
    """ZRacing Mississauga — search."""
    def url_builder(brand, model):
        q = f"{brand} {model}".replace(" ", "+")
        return f"https://zracing.ca/?s={q}&post_type=product"
    return scrape_shop("ZRacing Mississauga", url_builder, delay=2.0)


SHOP_SCRAPERS = {
    "canadian-tire": scrape_canadian_tire,
    "tdot-performance": scrape_tdot_performance,
    "active-green-ross": scrape_active_green_ross,
    "point-s": scrape_point_s,
    "krave": scrape_krave,
    "noble-tire": scrape_noble_tire,
    "fas-tire": scrape_fas_tire,
    "zracing": scrape_zracing,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shop", help="Scrape only this shop (e.g. 'canadian-tire')")
    parser.add_argument("--test", action="store_true", help="Test mode: only scrape first brand/model")
    args = parser.parse_args()

    if args.shop:
        if args.shop not in SHOP_SCRAPERS:
            print(f"Unknown shop: {args.shop}. Available: {', '.join(SHOP_SCRAPERS.keys())}")
            sys.exit(1)
        shops_to_scrape = {args.shop: SHOP_SCRAPERS[args.shop]}
    else:
        shops_to_scrape = SHOP_SCRAPERS

    for shop_key, scraper_fn in shops_to_scrape.items():
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"SCRAPING: {shop_key}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        results = scraper_fn()

        output_file = os.path.join(SCRIPTS_DIR, f"local-prices-{shop_key}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{shop_key}: {len(results)} prices saved to {output_file}", file=sys.stderr)

    print("\nDone. Run compile-market-pricing.py next.", file=sys.stderr)


if __name__ == "__main__":
    main()
