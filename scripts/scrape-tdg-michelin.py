#!/usr/bin/env python3
"""
Scrape TDG Access for Michelin tire stock and pricing.
READ ONLY — no order/buy/cart actions.

Uses f-brand=Michelin URL parameter to filter results directly.

Page text pattern per product:
  Michelin
  Model Name
  Variant Description
    SKU   SIZE   OD   SD   UTQG   STOCK   MSL   PRICE
  (blank)
    MI-XXXXX
    P225/45R17 XL   24.97"   94Y   540 AA A   47 86   $271.00
  $219.51
  Details
  Add to Order

Key: SKU line has just the SKU code. Next line has all the data + MSL.
The standalone $xxx.xx line after that is the DEALER PRICE.
Stock shows two warehouse quantities (e.g., "47 86" = total 133).
"""

import json
import re
import sys
import time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL = "https://www.tdgaccess.ca/"
USERNAME = "robert@luxurylanemotors.ca"
PASSWORD = "Toronto123!"

SIZES = [
    "205/55R16", "215/55R17", "225/45R17", "225/50R17",
    "225/40R18", "235/45R18", "245/40R18", "255/35R18",
    "225/40R19", "235/35R19", "245/40R19", "255/35R19",
    "235/35R20", "245/35R20", "255/30R20", "275/35R20",
    "245/45R20", "255/45R20", "265/50R20",
    "225/65R17", "235/65R18", "245/60R18", "265/60R18",
    "265/70R17", "275/65R18", "275/55R20",
    "255/50R19", "235/55R19",
]

OUTPUT_PATH = "/Users/indigochild/.hermes/extensions/substation/scripts/tdg-michelin-stock.json"
SCRIPTS_DIR = "/Users/indigochild/.hermes/extensions/substation/scripts"


def scrape_michelin():
    all_results = []
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1400, "height": 1000},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # --- Login ---
        print("[*] Navigating to TDG Access...", file=sys.stderr)
        page.goto(URL, wait_until="networkidle", timeout=30000)
        time.sleep(2)

        print("[*] Logging in...", file=sys.stderr)
        page.fill('input[name="username"]', USERNAME)
        page.fill('input[name="password"]', PASSWORD)
        page.locator('button[type="submit"], input[type="submit"]').first.click()
        page.wait_for_load_state("networkidle", timeout=30000)
        time.sleep(3)
        print(f"[*] Post-login URL: {page.url}", file=sys.stderr)

        # --- Search each size with Michelin brand filter ---
        for i, size in enumerate(SIZES):
            print(f"[{i+1}/{len(SIZES)}] Searching: {size}", file=sys.stderr)

            try:
                page_num = 1
                size_results = []

                while True:
                    search_url = (
                        f"https://www.tdgaccess.ca/Catalog/Tires/{page_num}"
                        f"?f-tiresize-front={size.replace('/', '%2F')}"
                        f"&f-brand=Michelin"
                    )
                    page.goto(search_url, wait_until="networkidle", timeout=25000)
                    time.sleep(2)

                    body_text = page.inner_text("body", timeout=10000)

                    if "no tires" in body_text.lower() or "0 tires" in body_text.lower():
                        if page_num == 1:
                            print(f"    -> No Michelin tires for this size", file=sys.stderr)
                        break

                    entries = parse_page(body_text, size)
                    size_results.extend(entries)

                    # Pagination
                    page_info = re.search(r'page\s+(\d+)\s+of\s+(\d+)', body_text, re.IGNORECASE)
                    if page_info:
                        current = int(page_info.group(1))
                        total = int(page_info.group(2))
                        if current < total:
                            page_num += 1
                            continue
                    break

                all_results.extend(size_results)
                stock_count = sum(1 for e in size_results if e["stock"] > 0)
                print(f"    -> {len(size_results)} SKUs ({stock_count} in stock)", file=sys.stderr)

            except PWTimeout as e:
                msg = str(e).split('\n')[0]
                print(f"    -> TIMEOUT: {msg}", file=sys.stderr)
                errors.append({"size": size, "error": msg})
            except Exception as e:
                print(f"    -> ERROR: {e}", file=sys.stderr)
                errors.append({"size": size, "error": str(e)})

            time.sleep(1)

        browser.close()

    return all_results, errors


KNOWN_BRANDS = {
    "Antares", "Bridgestone", "Continental", "Cooper", "Dunlop",
    "Falken", "Firestone", "General", "Goodyear", "Hankook",
    "Kumho", "Michelin", "Nexen", "Nokian", "Pirelli", "Radar",
    "Sailun", "Toyo", "Uniroyal", "Yokohama", "BFGoodrich",
    "GT Radial", "Nitto", "Sumitomo", "Westlake", "Mazzini",
    "Gislaved", "Maxxis", "Minerva", "Landsail", "Federal",
    "Zeta", "Achilles", "Accelera", "Nankang", "Ironman",
}


def parse_page(body_text, search_size):
    """Parse the page text into Michelin tire entries.

    The text follows this pattern per product block:

      Michelin              <- brand header
      Pilot Sport AS 4      <- model line 1
      Black Sidewall 70k... <- model line 2 (variant)
        SKU  SIZE  OD  ...  <- column headers
      (blank line)
        MI-19262            <- SKU code
        P225/45R17 XL  24.97"  94Y  540 AA A  47 86  $271.00   <- data + MSL
      $219.51               <- DEALER PRICE
      Details
      Add to Order

    Some products have multiple SKU rows before "Details".
    """
    results = []
    lines = body_text.split("\n")
    n = len(lines)

    current_model = ""
    in_michelin = False
    i = 0

    while i < n:
        line = lines[i].strip()

        # Detect brand headers
        if line == "Michelin":
            in_michelin = True
            current_model = ""
            # Collect model name from subsequent lines until SKU header
            i += 1
            model_parts = []
            while i < n:
                l = lines[i].strip()
                if "SKU" in l and "SIZE" in l:
                    i += 1  # skip header
                    break
                if l and l not in ("Details", "Add to Order", ""):
                    model_parts.append(l)
                elif l == "":
                    pass  # skip blank
                i += 1
            current_model = " ".join(model_parts)
            continue

        # Check if we've left the Michelin block
        if line in KNOWN_BRANDS and line != "Michelin":
            in_michelin = False
            current_model = ""
            i += 1
            continue

        if not in_michelin:
            i += 1
            continue

        # Inside a Michelin product block -- look for SKU lines
        sku_match = re.match(r'^(MI-\S+)\s*$', line)
        if sku_match:
            sku = sku_match.group(1)

            # Next non-blank line should be the data line
            data_line = ""
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                data_line = lines[j].strip()

            # Next line after data should be the dealer price
            price_line = ""
            k = j + 1
            while k < n and not lines[k].strip():
                k += 1
            if k < n:
                price_line = lines[k].strip()

            entry = build_entry(sku, data_line, price_line, current_model, search_size)
            results.append(entry)

            # Advance past the consumed lines
            i = k + 1 if k < n else j + 1
            continue

        # Also detect SKU on same line as OE markings (e.g., "MI-19262 \n *, MOE")
        sku_match2 = re.match(r'^(MI-\S+)', line)
        if sku_match2 and not re.search(r'\d{3}/\d{2}', line):
            # SKU might have extra text on same line
            sku = sku_match2.group(1)

            # Look ahead for data and price
            data_line = ""
            j = i + 1
            # Skip OE marking lines and blanks
            while j < n:
                l = lines[j].strip()
                if re.search(r'P?\d{3}/\d{2}', l):
                    data_line = l
                    break
                if l and not re.match(r'^[\s*,MOEAQZPRWE\-\s]+$', l) and len(l) > 10:
                    data_line = l
                    break
                j += 1

            price_line = ""
            k = j + 1
            while k < n and not lines[k].strip():
                k += 1
            if k < n:
                price_line = lines[k].strip()

            entry = build_entry(sku, data_line, price_line, current_model, search_size)
            results.append(entry)

            i = k + 1 if k < n else j + 1
            continue

        # Skip noise lines (Details, Add to Order, blanks, header rows)
        if line in ("Details", "Add to Order", "") or ("SKU" in line and "SIZE" in line):
            i += 1
            continue

        # Standalone price line — update the last entry's dealer price
        price_solo = re.match(r'^\$([\d,]+\.\d{2})$', line)
        if price_solo and results:
            results[-1]["dealer_price"] = float(price_solo.group(1).replace(",", ""))
            i += 1
            continue

        # Unrecognized line — skip
        i += 1

    return results


def build_entry(sku, data_line, price_line, model, search_size):
    """Build a single tire entry from parsed components."""
    entry = {
        "size": search_size,
        "brand": "Michelin",
        "model": model.strip(),
        "sku": sku,
        "stock": 0,
        "dealer_price": 0.0,
    }

    # Parse the data line for stock and MSL
    # Data line looks like: P225/45R17 XL  24.97"  94Y  540 AA A  47 86  $271.00
    if data_line:
        # Extract MSL (the $ amount on the data line)
        msl_match = re.search(r'\$([\d,]+\.\d{2})', data_line)
        msl = float(msl_match.group(1).replace(",", "")) if msl_match else 0.0

        # Extract stock: the numbers between UTQG and MSL
        # UTQG pattern: digits followed by letter grades (e.g., "540 AA A" or "-")
        # Stock: one or two numbers after UTQG, before $
        pre_dollar = data_line.split("$")[0] if "$" in data_line else data_line

        # Find the last group of numbers in pre_dollar text
        # These are stock quantities from warehouses
        # Pattern: after all the tire specs, we get stock numbers
        # The stock numbers are the last numbers before the $ sign
        # They appear as "47 86" or "99+" or just "-" or "0"

        # Strategy: extract all standalone numbers from pre_dollar,
        # excluding those in size/OD/SD/UTQG
        # Remove the size pattern first
        cleaned = re.sub(r'P?\d{3}/\d{2}[ZR]+\d{2}\s*(XL)?', '', pre_dollar)
        # Remove OD (like 24.97")
        cleaned = re.sub(r'\d+\.\d+["\s]', '', cleaned)
        # Remove speed rating (like 94Y, 91H)
        cleaned = re.sub(r'\d{2,3}[A-Z]\b', '', cleaned)
        # Remove UTQG (like 540 AA A or 500 A A or just -)
        cleaned = re.sub(r'\d{3}\s+[A-Z]+\s+[A-Z]+', '', cleaned)

        # Now find remaining numbers — these should be stock
        stock_nums = re.findall(r'(\d+)\+?', cleaned)

        if stock_nums:
            total_stock = sum(int(s) for s in stock_nums)
            entry["stock"] = total_stock
        elif "-" in cleaned:
            entry["stock"] = 0

    # Parse the dealer price line
    if price_line:
        dp_match = re.match(r'^\$([\d,]+\.\d{2})', price_line)
        if dp_match:
            entry["dealer_price"] = float(dp_match.group(1).replace(",", ""))
        elif price_line == "-":
            entry["dealer_price"] = 0.0

    return entry


if __name__ == "__main__":
    print("=" * 60, file=sys.stderr)
    print("TDG Access — Michelin Tire Stock Scraper", file=sys.stderr)
    print("READ ONLY — No orders or modifications", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results, errors = scrape_michelin()

    # Deduplicate by SKU
    seen = set()
    unique_results = []
    for r in results:
        key = r["sku"] if r["sku"] else (r["size"], r["model"], r["dealer_price"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # Save JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(unique_results, f, indent=2)

    # --- Summary ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"RESULTS SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Total Michelin SKUs: {len(unique_results)}", file=sys.stderr)

    with_price = [r for r in unique_results if r["dealer_price"] > 0]
    in_stock = [r for r in unique_results if r["stock"] > 0]

    print(f"With dealer price: {len(with_price)}", file=sys.stderr)
    print(f"In stock (stock > 0): {len(in_stock)}", file=sys.stderr)

    sizes_found = set(r["size"] for r in unique_results)
    sizes_with_stock = set(r["size"] for r in in_stock)
    print(f"Sizes with Michelin: {len(sizes_found)} / {len(SIZES)}", file=sys.stderr)
    print(f"Sizes with stock: {len(sizes_with_stock)} / {len(SIZES)}", file=sys.stderr)

    if with_price:
        prices = [r["dealer_price"] for r in with_price]
        print(f"Dealer price range: ${min(prices):.2f} — ${max(prices):.2f}", file=sys.stderr)

    models = sorted(set(r["model"] for r in unique_results if r["model"]))
    print(f"\nUnique models ({len(models)}):", file=sys.stderr)
    for m in models:
        skus = [r for r in unique_results if r["model"] == m]
        in_stock_skus = [r for r in skus if r["stock"] > 0]
        prices_m = [r["dealer_price"] for r in skus if r["dealer_price"] > 0]
        pr = f"${min(prices_m):.2f}-${max(prices_m):.2f}" if prices_m else "no price"
        total_stock = sum(r["stock"] for r in skus)
        print(f"  {m}: {len(skus)} SKUs, {len(in_stock_skus)} in stock (total qty: {total_stock}), {pr}", file=sys.stderr)

    if errors:
        print(f"\nErrors ({len(errors)}):", file=sys.stderr)
        for e in errors:
            print(f"  {e['size']}: {e['error']}", file=sys.stderr)

    print(f"\nSaved to: {OUTPUT_PATH}", file=sys.stderr)
