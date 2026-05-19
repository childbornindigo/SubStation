#!/usr/bin/env python3
"""
Audit TDG Access stock for ALL featured models across ALL brands.
READ ONLY — no orders, no modifications.

Strategy: For each brand we carry, search TDG with brand filter (no size filter),
paginate through ALL results, and collect every model+size+stock entry.
Then compare against our Supabase inventory to find gaps.
"""

import json
import re
import sys
import time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL = "https://www.tdgaccess.ca/"
USERNAME = "robert@luxurylanemotors.ca"
PASSWORD = "Toronto123!"

# Our featured brands and the model names we care about (from inventory dump)
# Map: brand -> list of (our_model_name, tdg_search_terms)
# TDG may use slightly different model names, so we need fuzzy matching
FEATURED_MODELS = {
    "Bridgestone": [
        "Alenza Sport AS",
        "Dueler HL Alenza Plus OWL",
        "Turanza Everdrive",
        "UltraWeather",
        "WeatherPeak",
    ],
    "Continental": [
        "CrossContact LX Sport",
        "CrossContact RX",
        "ExtremeContact DWS06 PLUS",
        "ProContact RX",
    ],
    "Falken": [
        "Azenis FK460 AS",
    ],
    "Firestone": [
        "Destination LE 2",
        "Destination LE3",
        "FR710 All Season",
        "Weathergrip",
    ],
    "Hankook": [
        "Dynapro evo AS",
        "Kinergy GT",
        "Ventus S1 noble2",
    ],
    "Pirelli": [
        "P Zero AS Plus 3",
        "Scorpion Zero All Season",
    ],
    "Toyo": [
        "Extensa AS II",
        "Open Country A50",
    ],
    "Yokohama": [
        "Geolandar X-AT G016",
        "Geolandar X-CV",
    ],
}

KNOWN_BRANDS = {
    "Antares", "Bridgestone", "Continental", "Cooper", "Dunlop",
    "Falken", "Firestone", "General", "Goodyear", "Hankook",
    "Kumho", "Michelin", "Nexen", "Nokian", "Pirelli", "Radar",
    "Sailun", "Toyo", "Uniroyal", "Yokohama", "BFGoodrich",
    "GT Radial", "Nitto", "Sumitomo", "Westlake", "Mazzini",
    "Gislaved", "Maxxis", "Minerva", "Landsail", "Federal",
    "Zeta", "Achilles", "Accelera", "Nankang", "Ironman",
}


def normalize_size(size_str):
    """Normalize TDG size to match our DB format: e.g., 'P225/45R17 XL' -> '225/45R17'"""
    # Remove leading P or LT
    s = re.sub(r'^[PLT]+', '', size_str.strip())
    # Extract the core size pattern
    m = re.match(r'(\d{3}/\d{2}[RZ]+\d{2})', s)
    if m:
        size = m.group(1)
        # Normalize ZR to R
        size = size.replace('ZR', 'R')
        return size
    # Handle LT sizes like "33x12.50R18LT LRF"
    lt_match = re.match(r'(\d+x\d+\.?\d*R\d+)', size_str.strip())
    if lt_match:
        return lt_match.group(1) + "LT LRF"
    return size_str.strip()


def match_model(tdg_model_name, our_models):
    """Try to match a TDG model name to one of our model names.
    Returns the matched model name or None."""
    tdg_lower = tdg_model_name.lower().strip()

    for our_model in our_models:
        our_lower = our_model.lower()

        # Direct containment
        if our_lower in tdg_lower or tdg_lower in our_lower:
            return our_model

        # Key-word matching for specific models
        # Bridgestone
        if "alenza" in our_lower and "sport" in our_lower and "alenza" in tdg_lower and "sport" in tdg_lower:
            return our_model
        if "dueler" in our_lower and "alenza" in our_lower and "dueler" in tdg_lower and "alenza" in tdg_lower:
            return our_model
        if "turanza" in our_lower and "everdrive" in our_lower and "turanza" in tdg_lower and "everdrive" in tdg_lower:
            return our_model
        if "ultraweather" in our_lower and "ultraweather" in tdg_lower:
            return our_model
        if "weatherpeak" in our_lower and "weatherpeak" in tdg_lower:
            return our_model

        # Continental
        if "crosscontact" in our_lower.replace(" ", "") and "lx sport" in our_lower:
            if "crosscontact" in tdg_lower.replace(" ", "") and "lx sport" in tdg_lower:
                return our_model
        if "crosscontact" in our_lower.replace(" ", "") and "rx" in our_lower.split():
            tdg_words = tdg_lower.split()
            if "crosscontact" in tdg_lower.replace(" ", "") and "rx" in tdg_words:
                return our_model
        if "extremecontact" in our_lower.replace(" ", "") and "dws06" in our_lower:
            if "extremecontact" in tdg_lower.replace(" ", "") and "dws06" in tdg_lower:
                return our_model
        if "procontact" in our_lower.replace(" ", "") and "rx" in our_lower.split():
            tdg_words = tdg_lower.split()
            if "procontact" in tdg_lower.replace(" ", "") and "rx" in tdg_words:
                return our_model

        # Falken
        if "fk460" in our_lower and "fk460" in tdg_lower:
            return our_model

        # Firestone
        if "destination" in our_lower and "le 2" in our_lower and "destination" in tdg_lower and ("le 2" in tdg_lower or "le2" in tdg_lower):
            # Make sure it's LE 2, not LE3
            if "le3" not in tdg_lower and "le 3" not in tdg_lower:
                return our_model
        if "destination" in our_lower and "le3" in our_lower and "destination" in tdg_lower and ("le3" in tdg_lower or "le 3" in tdg_lower):
            return our_model
        if "fr710" in our_lower and "fr710" in tdg_lower:
            return our_model
        if "weathergrip" in our_lower and "weathergrip" in tdg_lower:
            return our_model

        # Hankook
        if "dynapro" in our_lower and "evo" in our_lower and "dynapro" in tdg_lower and "evo" in tdg_lower:
            return our_model
        if "kinergy" in our_lower and "gt" in our_lower and "kinergy" in tdg_lower and "gt" in tdg_lower:
            return our_model
        if "ventus" in our_lower and "noble2" in our_lower.replace(" ", "") and "ventus" in tdg_lower and "noble" in tdg_lower:
            # Match but exclude RunFlat variants for the non-runflat model
            if "runflat" not in our_lower and "run" not in tdg_lower and "flat" not in tdg_lower:
                return our_model
            if "runflat" in our_lower:
                return our_model

        # Pirelli
        if "p zero" in our_lower and "plus 3" in our_lower and "p zero" in tdg_lower and "plus 3" in tdg_lower:
            return our_model
        if "p zero" in our_lower and "plus 3" in our_lower and "p zero" in tdg_lower and "plus3" in tdg_lower:
            return our_model
        if "scorpion" in our_lower and "zero" in our_lower and "scorpion" in tdg_lower and "zero" in tdg_lower:
            # Make sure it matches "All Season"
            if "all season" in our_lower and ("all season" in tdg_lower or "as" in tdg_lower.split()):
                return our_model

        # Toyo
        if "extensa" in our_lower and "extensa" in tdg_lower:
            return our_model
        if "open country" in our_lower and "a50" in our_lower and "open country" in tdg_lower and "a50" in tdg_lower:
            return our_model

        # Yokohama
        if "geolandar" in our_lower and "x-at" in our_lower and "geolandar" in tdg_lower and ("x-at" in tdg_lower or "x at" in tdg_lower):
            return our_model
        if "geolandar" in our_lower and "x-cv" in our_lower and "geolandar" in tdg_lower and ("x-cv" in tdg_lower or "x cv" in tdg_lower):
            return our_model

    return None


def parse_brand_page(body_text, brand):
    """Parse TDG page text for a specific brand's tires.
    Returns list of {model, size, stock, sku, dealer_price}"""
    results = []
    lines = body_text.split("\n")
    n = len(lines)

    current_model = ""
    in_brand = False
    i = 0

    while i < n:
        line = lines[i].strip()

        # Detect our brand header
        if line == brand:
            in_brand = True
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
                    pass
                i += 1
            current_model = " ".join(model_parts)
            continue

        # Check if we've left our brand's block
        if line in KNOWN_BRANDS and line != brand:
            in_brand = False
            current_model = ""
            i += 1
            continue

        if not in_brand:
            i += 1
            continue

        # Inside brand block - look for SKU lines
        # SKU patterns by brand prefix
        prefix_map = {
            "Bridgestone": "BS-",
            "Continental": "CT-",
            "Falken": "FK-",
            "Firestone": "FS-",
            "Hankook": "HN-",
            "Pirelli": "PI-",
            "Toyo": "TY-",
            "Yokohama": "YK-",
        }
        prefix = prefix_map.get(brand, "")

        sku_match = re.match(rf'^({re.escape(prefix)}\S+)\s*$', line) if prefix else None
        if not sku_match and prefix:
            sku_match = re.match(rf'^({re.escape(prefix)}\S+)', line)
            if sku_match and re.search(r'\d{3}/\d{2}', line):
                sku_match = None  # Not a standalone SKU line

        if sku_match:
            sku = sku_match.group(1)

            # Next non-blank line should be the data line
            data_line = ""
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                data_line = lines[j].strip()

            # Check if this is an OE marking line (short, no numbers/size pattern)
            # If so, skip to next line for actual data
            if data_line and not re.search(r'\d{3}/\d{2}', data_line) and len(data_line) < 30:
                j += 1
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

            entry = build_entry(sku, data_line, price_line, current_model, brand)
            if entry["size"]:
                results.append(entry)

            i = k + 1 if k < n else j + 1
            continue

        # Standalone price line — update last entry
        price_solo = re.match(r'^\$([\d,]+\.\d{2})$', line)
        if price_solo and results:
            results[-1]["dealer_price"] = float(price_solo.group(1).replace(",", ""))
            i += 1
            continue

        i += 1

    return results


def build_entry(sku, data_line, price_line, model, brand):
    """Build a tire entry from parsed components."""
    entry = {
        "brand": brand,
        "model": model.strip(),
        "sku": sku,
        "size": "",
        "stock": 0,
        "dealer_price": 0.0,
    }

    if data_line:
        # Extract size
        size_match = re.search(r'[PLT]?(\d{2,3}/\d{2}[ZR]+\d{2})', data_line)
        if size_match:
            raw_size = size_match.group(0)
            entry["size"] = normalize_size(raw_size)
        else:
            # LT size pattern
            lt_match = re.search(r'(\d+x\d+\.?\d*R\d+)', data_line)
            if lt_match:
                entry["size"] = normalize_size(lt_match.group(0))

        # Extract MSL
        msl_match = re.search(r'\$([\d,]+\.\d{2})', data_line)

        # Extract stock: numbers between UTQG ratings and $
        pre_dollar = data_line.split("$")[0] if "$" in data_line else data_line

        # Remove size, OD, speed/load rating, UTQG to isolate stock numbers
        cleaned = pre_dollar
        cleaned = re.sub(r'[PLT]?\d{2,3}/\d{2}[ZR]+\d{2}\s*(XL)?', '', cleaned)
        cleaned = re.sub(r'\d+x\d+\.?\d*R\d+\S*', '', cleaned)
        cleaned = re.sub(r'\d+\.\d+["\s]', '', cleaned)  # OD
        cleaned = re.sub(r'\d{2,3}[A-Z()]+\b', '', cleaned)  # speed rating
        cleaned = re.sub(r'\d{3}\s+[A-Z]+\s+[A-Z]+', '', cleaned)  # UTQG
        cleaned = re.sub(r'LR[A-Z]', '', cleaned)  # Load range

        stock_nums = re.findall(r'(\d+)\+?', cleaned)
        if stock_nums:
            entry["stock"] = sum(int(s) for s in stock_nums)
        elif "-" in cleaned.strip():
            entry["stock"] = 0

    # Dealer price
    if price_line:
        dp_match = re.match(r'^\$([\d,]+\.\d{2})', price_line)
        if dp_match:
            entry["dealer_price"] = float(dp_match.group(1).replace(",", ""))

    return entry


def scrape_all_brands():
    all_results = {}  # brand -> list of entries
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

        for brand, models in FEATURED_MODELS.items():
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[*] Scraping brand: {brand} (models: {', '.join(models)})", file=sys.stderr)

            brand_results = []
            page_num = 1
            max_pages = 50  # Safety limit

            while page_num <= max_pages:
                try:
                    search_url = (
                        f"https://www.tdgaccess.ca/Catalog/Tires/{page_num}"
                        f"?f-brand={brand.replace(' ', '+')}"
                    )
                    print(f"  Page {page_num}: {search_url}", file=sys.stderr)
                    page.goto(search_url, wait_until="networkidle", timeout=30000)
                    time.sleep(2)

                    body_text = page.inner_text("body", timeout=10000)

                    # Check for no results
                    if "no tires" in body_text.lower() or "0 tires" in body_text.lower():
                        if page_num == 1:
                            print(f"  -> No tires found for {brand}", file=sys.stderr)
                        break

                    # Parse entries
                    entries = parse_brand_page(body_text, brand)
                    brand_results.extend(entries)

                    print(f"  -> Found {len(entries)} entries on this page", file=sys.stderr)

                    # Check pagination
                    page_info = re.search(r'page\s+(\d+)\s+of\s+(\d+)', body_text, re.IGNORECASE)
                    if page_info:
                        current = int(page_info.group(1))
                        total = int(page_info.group(2))
                        print(f"  -> Page {current} of {total}", file=sys.stderr)
                        if current < total:
                            page_num += 1
                            time.sleep(1)
                            continue
                    break

                except PWTimeout as e:
                    msg = str(e).split('\n')[0]
                    print(f"  -> TIMEOUT on page {page_num}: {msg}", file=sys.stderr)
                    errors.append({"brand": brand, "page": page_num, "error": msg})
                    break
                except Exception as e:
                    print(f"  -> ERROR on page {page_num}: {e}", file=sys.stderr)
                    errors.append({"brand": brand, "page": page_num, "error": str(e)})
                    break

            # Deduplicate by SKU
            seen = set()
            unique = []
            for r in brand_results:
                if r["sku"] not in seen:
                    seen.add(r["sku"])
                    unique.append(r)

            all_results[brand] = unique

            # Filter to our models
            matched = 0
            for entry in unique:
                m = match_model(entry["model"], models)
                if m:
                    matched += 1
                    entry["matched_model"] = m

            in_stock = sum(1 for e in unique if e.get("matched_model") and e["stock"] > 0)
            print(f"  -> Total {brand} SKUs: {len(unique)}, matched to our models: {matched}, in stock: {in_stock}", file=sys.stderr)

            time.sleep(2)

        browser.close()

    return all_results, errors


if __name__ == "__main__":
    print("=" * 60, file=sys.stderr)
    print("TDG Access — Full Brand Audit Scraper", file=sys.stderr)
    print("READ ONLY — No orders or modifications", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results, errors = scrape_all_brands()

    # Save raw results
    output_path = "/Users/indigochild/.hermes/extensions/substation/scripts/tdg-brand-audit-raw.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", file=sys.stderr)
    print("RAW RESULTS SAVED", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for brand, entries in results.items():
        matched = [e for e in entries if e.get("matched_model")]
        in_stock = [e for e in matched if e["stock"] > 0]
        print(f"{brand}: {len(entries)} total SKUs, {len(matched)} matched, {len(in_stock)} in stock", file=sys.stderr)

    if errors:
        print(f"\nErrors: {json.dumps(errors, indent=2)}", file=sys.stderr)

    print(f"\nSaved to: {output_path}", file=sys.stderr)
