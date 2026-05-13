#!/usr/bin/env python3
"""
scrape_pmctire.py — Scrape PMCtire's Next.js API for structured tire product data.

Extracts: brand, model, price, wholesale cost, specs, category, season,
and more for all 65 target tire sizes.

Usage:
    python3 scrape_pmctire.py              # full run (65 sizes)
    python3 scrape_pmctire.py --test       # test run (5 sizes)
    python3 scrape_pmctire.py --sizes 10   # partial run
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "market-intelligence")

BUILD_ID = "MTNOvAYQBgPnB-AvxsOFs"
API_BASE = f"https://pmctire.com/_next/data/{BUILD_ID}/en/tires.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-CA,en;q=0.9",
}

PAGES_PER_SIZE = 12
REQUEST_DELAY = 0.8

SIZES = [
    "195/65R15", "205/55R16", "215/60R16", "225/65R17", "235/65R18",
    "245/75R16", "265/70R17", "275/55R20", "215/55R17", "225/60R18",
    "205/65R16", "215/65R17", "225/55R17", "235/55R19", "245/45R18",
    "255/70R18", "265/65R18", "275/60R20", "285/45R22", "195/60R15",
    "205/60R16", "215/45R17", "225/45R17", "235/45R18", "245/40R18",
    "255/55R18", "265/60R18", "275/45R20", "285/65R18", "195/55R16",
    "205/50R17", "215/50R17", "225/40R18", "235/40R19", "245/35R20",
    "255/45R19", "265/50R20", "275/40R20", "285/70R17", "195/50R16",
    "205/45R17", "215/70R16", "225/50R18", "235/70R16", "245/65R17",
    "255/65R18", "265/70R16", "275/65R18", "285/75R16", "175/65R15",
    "185/65R15", "185/60R15", "225/45R18", "235/55R18", "245/60R18",
    "255/50R19", "265/75R16", "275/55R19", "225/60R17", "235/60R18",
    "245/55R19", "255/35R19", "265/60R20", "275/70R18", "225/50R17",
]

FITMENT = {
    "175/65R15": "Honda Fit, Toyota Yaris, Nissan Versa",
    "185/60R15": "Honda Civic, Toyota Corolla, Mazda3",
    "185/65R15": "Honda Civic, Toyota Corolla, Hyundai Elantra",
    "195/50R16": "Mini Cooper, Honda Fit Sport",
    "195/55R16": "VW Golf, Honda Civic, Toyota Corolla",
    "195/60R15": "Honda Civic, Toyota Corolla, Nissan Sentra",
    "195/65R15": "Honda Civic, Toyota Corolla, VW Jetta, Mazda3",
    "205/45R17": "Honda Civic Si, VW GTI, Mazda3 Sport",
    "205/50R17": "Honda Accord, Toyota Camry, Mazda6",
    "205/55R16": "Honda Civic, Toyota Corolla, VW Jetta, Mazda3",
    "205/60R16": "Honda Accord, Toyota Camry, Nissan Altima",
    "205/65R16": "Honda CR-V, Toyota RAV4, Subaru Forester",
    "215/45R17": "Honda Civic Si, Mazda3, VW GTI",
    "215/50R17": "Honda Accord, Mazda6, Subaru Legacy",
    "215/55R17": "Honda Accord, Toyota Camry, Hyundai Sonata",
    "215/60R16": "Honda CR-V, Toyota RAV4, Nissan Rogue",
    "215/65R17": "Honda CR-V, Toyota RAV4, Ford Escape",
    "215/70R16": "Honda CR-V, Toyota RAV4, Subaru Outback",
    "225/40R18": "BMW 3 Series, Audi A4, Mercedes C-Class",
    "225/45R17": "Honda Accord, Toyota Camry, BMW 3 Series",
    "225/45R18": "Audi A4, BMW 3 Series, Mercedes C-Class",
    "225/50R17": "Honda Accord, Toyota Camry, Subaru Outback",
    "225/50R18": "BMW X1, Audi Q3, Mercedes GLA",
    "225/55R17": "Honda Accord, Toyota Camry, Subaru Outback",
    "225/60R17": "Toyota Highlander, Ford Edge, Honda Passport",
    "225/60R18": "Toyota Highlander, Honda Pilot, Ford Edge",
    "225/65R17": "Toyota RAV4, Honda CR-V, Ford Escape, Nissan Rogue",
    "235/40R19": "BMW 4 Series, Audi S4, Mercedes C43 AMG",
    "235/45R18": "BMW 3 Series, Audi A4, Lexus IS",
    "235/55R18": "Acura RDX, Lincoln MKC, BMW X3",
    "235/55R19": "Acura RDX, BMW X3, Mercedes GLC",
    "235/60R18": "Toyota Highlander, Honda Pilot, Ford Explorer",
    "235/65R18": "Toyota Highlander, Chevy Traverse, Ford Explorer",
    "235/70R16": "Toyota 4Runner, Jeep Wrangler, Ford Ranger",
    "245/35R20": "BMW 5 Series, Audi A6, Mercedes E-Class",
    "245/40R18": "BMW 3 Series, Audi A4, Mercedes C-Class",
    "245/45R18": "BMW 5 Series, Audi A6, Mercedes E-Class",
    "245/55R19": "Ford Explorer, Chevy Blazer, Honda Passport",
    "245/60R18": "Toyota Highlander, Chevy Traverse, GMC Acadia",
    "245/65R17": "Toyota 4Runner, Jeep Grand Cherokee, Ford Explorer",
    "245/75R16": "Toyota Tacoma, Jeep Wrangler, Ford F-150",
    "255/35R19": "BMW M3, Audi RS4, Mercedes AMG C63",
    "255/45R19": "BMW X3, Audi Q5, Mercedes GLC",
    "255/50R19": "BMW X5, Audi Q7, Mercedes GLE",
    "255/55R18": "Jeep Grand Cherokee, Ford Explorer, Dodge Durango",
    "255/65R18": "Toyota Tundra, Ford F-150, Chevy Silverado",
    "255/70R18": "Ford F-150, Chevy Silverado, Ram 1500",
    "265/50R20": "Chevy Tahoe, GMC Yukon, Ford Expedition",
    "265/60R18": "Toyota Tundra, Nissan Titan, Ford F-150",
    "265/60R20": "Chevy Tahoe, GMC Yukon, Ford Expedition",
    "265/65R18": "Toyota Tundra, Nissan Titan, Ford F-150",
    "265/70R16": "Toyota Tacoma, Jeep Wrangler, Ford Ranger",
    "265/70R17": "Toyota 4Runner, Jeep Wrangler, Ford F-150",
    "265/75R16": "Ford F-250, Chevy 2500, Ram 2500",
    "275/40R20": "BMW X5, Audi Q7, Porsche Cayenne",
    "275/45R20": "Chevy Tahoe, GMC Yukon, Cadillac Escalade",
    "275/55R19": "Chevy Traverse, GMC Acadia, Buick Enclave",
    "275/55R20": "Ford F-150, Chevy Silverado, Ram 1500",
    "275/60R20": "Ford F-150, Chevy Silverado, Ram 1500",
    "275/65R18": "Ford F-150, Chevy Silverado, Ram 1500, Toyota Tundra",
    "275/70R18": "Ford F-250, Chevy 2500, Ram 2500",
    "285/45R22": "Cadillac Escalade, Chevy Tahoe, GMC Yukon",
    "285/65R18": "Ford F-250, Chevy 2500, Ram 2500",
    "285/70R17": "Ford F-250, Chevy 2500, Ram 2500",
    "285/75R16": "Ford F-250, Chevy 2500, Ram 2500",
}


def fetch_page(size, page_num):
    search_q = urllib.parse.quote(size)
    url = f"{API_BASE}?search={search_q}&page={page_num}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def extract_product(raw, target_size):
    pmc = raw.get("meta", {}).get("pmc_tire", {})
    if pmc.get("size") != target_size:
        return None

    tire_meta = raw.get("meta", {}).get("tire", {})
    global_meta = raw.get("meta", {}).get("global", {})
    variant = raw["variants"][0] if raw.get("variants") else {}
    cost_data = variant.get("inventoryItem", {}).get("unitCost", {})

    return {
        "brand": raw.get("vendor", ""),
        "model": tire_meta.get("model", ""),
        "size": target_size,
        "price_cad": variant.get("price"),
        "wholesale_cost": float(cost_data["amount"]) if cost_data.get("amount") else None,
        "compare_at_price": variant.get("compareAtPrice"),
        "load_index": pmc.get("load_index"),
        "speed_rating": pmc.get("speed_rating", ""),
        "category": pmc.get("category", ""),
        "season": pmc.get("usage", ""),
        "sidewall": pmc.get("sidewall_type", ""),
        "load_range": pmc.get("load_range_ply_rating", ""),
        "tire_type": pmc.get("type", ""),
        "runflat": pmc.get("runflat") == "1",
        "studdable": pmc.get("studdable") == "1",
        "studded": pmc.get("studded") == "1",
        "ev_optimized": pmc.get("electric_vehicle_tire") == "1",
        "original_equipment": pmc.get("original_equipment", ""),
        "sku": variant.get("sku", ""),
        "inventory": variant.get("inventoryQuantity"),
        "rebate_code": global_meta.get("instant_rebate_code", ""),
        "rebate_amount": float(global_meta["instant_rebate_amount"]) if global_meta.get("instant_rebate_amount") else None,
        "vehicle_fitment": FITMENT.get(target_size, ""),
        "retailer": "PMCtire",
        "product_url": f"https://pmctire.com/tires/{raw.get('objectID', '')}",
        "image_url": raw.get("featuredImage", {}).get("src", ""),
    }


def scrape_size(size, verbose=True):
    products = []
    seen_ids = set()

    for page_num in range(1, PAGES_PER_SIZE + 1):
        try:
            data = fetch_page(size, page_num)
            raw_products = data.get("pageProps", {}).get("products", [])

            if not raw_products:
                break

            page_matches = 0
            for raw in raw_products:
                obj_id = raw.get("objectID", "")
                if obj_id in seen_ids:
                    continue
                product = extract_product(raw, size)
                if product:
                    seen_ids.add(obj_id)
                    products.append(product)
                    page_matches += 1

            if verbose:
                print(f"    Page {page_num}: {page_matches} matches", flush=True)

            time.sleep(REQUEST_DELAY)
        except Exception as e:
            if verbose:
                print(f"    Page {page_num}: error - {e}", flush=True)
            break

    return products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test run (5 sizes)")
    parser.add_argument("--sizes", type=int, help="Number of sizes to scrape")
    args = parser.parse_args()

    sizes = SIZES
    if args.test:
        sizes = sizes[:5]
    elif args.sizes:
        sizes = sizes[: args.sizes]

    print(f"PMCtire scraper starting — {len(sizes)} sizes", flush=True)
    start_time = time.time()

    all_data = {}
    total_products = 0

    for i, size in enumerate(sizes, 1):
        print(f"[{i}/{len(sizes)}] {size}", flush=True)
        products = scrape_size(size)
        all_data[size] = products
        total_products += len(products)
        print(f"  → {len(products)} products found", flush=True)

    elapsed = time.time() - start_time

    output = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": "pmctire",
        "total_products": total_products,
        "sizes_scraped": len(sizes),
        "elapsed_seconds": round(elapsed),
        "sizes": all_data,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "pmctire-data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone: {total_products} products across {len(sizes)} sizes in {elapsed:.0f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
