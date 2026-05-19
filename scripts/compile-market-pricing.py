#!/usr/bin/env python3
"""
Market-Driven Pricing Analysis for LuxuryLaneTires.ca

Reads inventory dump and local competitor pricing, matches tires by
brand + model (fuzzy) + size (exact), calculates average local market
prices, and recommends pricing adjustments.

Pricing rules:
  - Baseline: wholesale + 30%
  - If local avg > baseline by >$10: RAISE to (local_avg - $5), cap at 60% markup
  - If local avg < baseline by >$10: DROP to (local_avg - $5), floor at 10% markup
  - If local avg within $10 of baseline: KEEP at 30%
  - If no local data: mark "no_data", keep at 30%
"""

import json
import os
import re
from collections import defaultdict
from itertools import groupby

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════
# Model name aliases — maps local shop names → canonical inventory names
# ═══════════════════════════════════════════════════════════════════

MODEL_ALIASES = {
    # ── Bridgestone ──
    "Alenza Sport A/S": "Alenza Sport AS",
    "Alenza Sport A/S All Season": "Alenza Sport AS",
    "Alenza AS Ultra All Season": "Alenza Sport AS",
    "Alenza  Sport As": "Alenza Sport AS",
    "Alenza Sport AS RFT": "Alenza Sport AS",
    "Dueler H/L Alenza Plus": "Dueler HL Alenza Plus OWL",
    "Dueler HL Alenza Plus": "Dueler HL Alenza Plus OWL",
    "Ecopia H/L 422  Plus": "Dueler HL Alenza Plus OWL",
    "Turanza EverDrive": "Turanza Everdrive",
    "Turanza EVERDR": "Turanza Everdrive",
    # ── Continental ──
    "ExtremeContact DWS06 Plus": "ExtremeContact DWS06 PLUS",
    "ExtremeContact DWS06+": "ExtremeContact DWS06 PLUS",
    "Continental Extremecontact Dws  06 Plus": "ExtremeContact DWS06 PLUS",
    "Procontact Rx": "ProContact RX",
    # Note: "Contiprocontact" is a DIFFERENT Continental model, not ProContact RX
    # ── Falken ──
    "Azenis FK460 A/S": "Azenis FK460 AS",
    # ── Firestone ──
    "Destination LE 3": "Destination LE3",
    "Destination LE3 OWL": "Destination LE3",
    "Destination  Le-3": "Destination LE3",
    "Destination  LE-3": "Destination LE3",
    "Destination LE3 All Season": "Destination LE3",
    "Firestone Destination LE2 All Season": "Destination LE 2",
    "FR710": "FR710 All Season",
    "WeatherGrip All Weather": "Weathergrip",
    "Firestone WeatherGrip All Weather": "Weathergrip",
    # Note: "Firestone All Season" and "All Season" from Noble are generic names
    # that likely refer to FR710 but could be other models -- map conservatively
    "Firestone All Season": "FR710 All Season",
    "All Season": "FR710 All Season",
    "All  Season": "FR710 All Season",
    # ── Hankook ──
    "H436 Kinergy GT": "Kinergy GT",
    "H436B Kinergy GT  RunFlat": "Kinergy GT RunFlat",
    "H452 Ventus S1 noble2": "Ventus S1 noble2",
    "H452B Ventus S1 noble2 RunFlat": "Ventus S1 noble2 RunFlat",
    "RA21 Dynapro evo AS": "Dynapro evo AS",
    "Kinergy GT (HT436) All Season": "Kinergy GT",
    "Ventus S1 noble2 (H452) All Season": "Ventus S1 noble2",
    "Ventus S1 Noble2": "Ventus S1 noble2",
    # ── Michelin ──
    "Defender LTX M/S 2": "Defender LTX M/S2",
    "Pilot Sport A/S 4": "Pilot Sport AS 4",
    "Pilot Sport All Season 4": "Pilot Sport AS 4",
    "Michelin Pilot Sport A/S 4": "Pilot Sport AS 4",
    "Primacy Tour A/S": "Primacy Tour AS",
    "Primacy Tour AS": "Primacy Tour AS",
    # Note: Noble's "Defender 2" / "Defender-2" is the Michelin Defender 2 (passenger)
    # which is a DIFFERENT tire from our "Defender LTX M/S2" (light truck). Do NOT alias.
    # ── Pirelli ──
    "P Zero A/S Plus 3": "P Zero AS Plus 3",
    "P Zero A/S  Plus 3": "P Zero AS Plus 3",
    "P Zero A/S +3": "P Zero AS Plus 3",
    "P Zero All Season Plus 3": "P Zero AS Plus 3",
    "P7  As Plus 3": "P Zero AS Plus 3",
    "P7 As Plus 3": "P Zero AS Plus 3",
    "Scorpion Zero AS": "Scorpion Zero All Season",
    "Scorpion A/S + Iii": "Scorpion Zero All Season",
    "Scorpion As+Iii": "Scorpion Zero All Season",
    # ── Toyo ──
    "Extensa A/S II": "Extensa AS II",
    "OPA50": "Extensa AS II",
    # Note: "Open Country A/T III" is NOT in our inventory (we have A50 and Extensa AS II)
    # ── Yokohama ──
    "Geolandar X-AT": "Geolandar X AT G016",
    "GEOLANDAR X-AT": "Geolandar X AT G016",
    "Geolandar X-AT G016": "Geolandar X AT G016",
    "Geolandar X AT G016": "Geolandar X AT G016",
    "Geolandar X-CV All Season": "Geolandar X CV",
    "Geolandar X-CV": "Geolandar X CV",
    "GEOLANDAR X-CV": "Geolandar X CV",
    "Geolandar X CV G057C": "Geolandar X CV",
    "Geolandar X-CV G057C": "Geolandar X CV",
}


# ═══════════════════════════════════════════════════════════════════
# Size normalization
# ═══════════════════════════════════════════════════════════════════

def normalize_size(size_str):
    """
    Extract and normalize just the tire size from a string that may
    include load index and speed rating (e.g. "235/65R17 104H" -> "235/65R17").
    Rim diameter is always 2 digits (15-24), so we limit to exactly 2 digits after R.
    """
    s = size_str.strip().upper()

    # Standard sizes: 205/55R16, LT275/65R18, P205/55R16
    # Match with or without spaces; rim is 2 digits
    m = re.match(r'((?:LT|P)?\d{3}/\d{2,3}R\d{2})\b', s)
    if m:
        return m.group(1).replace(" ", "")

    # Without word boundary (for cases where load index is glued on)
    m = re.match(r'((?:LT|P)?\d{3}/\d{2,3}R\d{2})', s)
    if m:
        return m.group(1).replace(" ", "")

    # LT sizes like 35X12.50R17
    m = re.match(r'(\d+[X][\d.]+R\d{2})\b', s)
    if m:
        return m.group(1).replace(" ", "")

    m = re.match(r'(\d+[X][\d.]+R\d{2})', s)
    if m:
        return m.group(1).replace(" ", "")

    return s.replace(" ", "")


def normalize_model(model_str):
    """Normalize model name via alias lookup, stripping extra whitespace."""
    m = model_str.strip()
    # Try direct alias
    if m in MODEL_ALIASES:
        return MODEL_ALIASES[m]
    # Try case-insensitive match
    m_lower = m.lower()
    for alias_key, alias_val in MODEL_ALIASES.items():
        if alias_key.lower() == m_lower:
            return alias_val
    return m


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_inventory():
    with open(os.path.join(SCRIPTS_DIR, "inventory-dump.json")) as f:
        return json.load(f)


def load_shop_data(filename, shop_label):
    """Load a shop's JSON data and normalize entries."""
    filepath = os.path.join(SCRIPTS_DIR, filename)
    if not os.path.exists(filepath):
        return []

    with open(filepath) as f:
        data = json.load(f)

    entries = []
    for item in data:
        price = item.get("price")
        if price is None or price == "CALL" or price == 0:
            continue
        if isinstance(price, str):
            try:
                price = float(price.replace("$", "").replace(",", "").strip())
            except ValueError:
                continue
        price = float(price)
        if price <= 0:
            continue

        brand = item.get("brand", "").strip()
        model = normalize_model(item.get("model", ""))
        size = normalize_size(item.get("size", ""))

        if not brand or not model or not size:
            continue

        entries.append({
            "brand": brand,
            "model": model,
            "size": size,
            "price": price,
            "shop": shop_label,
        })

    return entries


def load_all_local_prices():
    """Load all local pricing files."""
    shops = [
        ("local-prices-canadian-tire.json", "Canadian Tire"),
        ("local-prices-tdot-performance.json", "TDot Performance"),
        ("local-prices-point-s.json", "Point S"),
        ("local-prices-noble-tire.json", "Noble Tire Brampton"),
        ("local-prices-zracing.json", "ZRacing Mississauga"),
        ("local-prices-krave.json", "KRAVE"),
        ("local-prices-active-green-ross.json", "Active Green+Ross"),
        ("local-prices-fas-tire.json", "FAS Tire"),
    ]

    all_prices = []
    for filename, label in shops:
        entries = load_shop_data(filename, label)
        all_prices.extend(entries)
        if entries:
            print(f"  {label}: {len(entries)} priced entries")

    return all_prices


# ═══════════════════════════════════════════════════════════════════
# Matching and pricing logic
# ═══════════════════════════════════════════════════════════════════

def build_local_index(local_prices):
    """Index local prices by (brand_lower, model_normalized, size)."""
    index = defaultdict(list)
    for item in local_prices:
        key = (item["brand"].lower(), item["model"].lower(), item["size"])
        index[key].append(item)
    return index


def find_local_matches(brand, model, size, local_index):
    """Find all local price entries matching brand + model + size."""
    key = (brand.lower(), model.lower(), size)
    return local_index.get(key, [])


def compute_recommendation(wholesale, current_retail, local_matches):
    """
    Apply pricing rules and return the full result dict fields.
    """
    baseline = round(wholesale * 1.30, 2)
    current_markup_pct = round(((current_retail / wholesale) - 1) * 100, 2) if wholesale > 0 else 30.0

    if not local_matches:
        return {
            "local_avg": None,
            "local_prices": [],
            "num_sources": 0,
            "recommended_price": baseline,
            "recommended_markup_pct": 30.0,
            "change_from_30_pct": "0.0%",
            "direction": "no_data",
        }

    # Deduplicate: take cheapest price per shop (if a shop listed same tire twice)
    shop_best = {}
    for m in local_matches:
        shop = m["shop"]
        if shop not in shop_best or m["price"] < shop_best[shop]:
            shop_best[shop] = m["price"]

    local_prices_list = sorted(
        [{"shop": s, "price": round(p, 2)} for s, p in shop_best.items()],
        key=lambda x: x["price"]
    )
    num_sources = len(shop_best)
    local_avg = round(sum(shop_best.values()) / num_sources, 2)

    diff = local_avg - baseline

    if abs(diff) <= 10:
        # Within $10 of 30% -- keep
        recommended_price = baseline
        direction = "keep"
    elif diff > 0:
        # Local avg higher than our 30% -- raise
        recommended_price = round(local_avg - 5, 2)
        max_price = round(wholesale * 1.60, 2)
        if recommended_price > max_price:
            recommended_price = max_price
        direction = "raise"
    else:
        # Local avg lower than our 30% -- drop
        recommended_price = round(local_avg - 5, 2)
        min_price = round(wholesale * 1.20, 2)
        if recommended_price < min_price:
            recommended_price = min_price
        direction = "drop"

    recommended_markup_pct = round(((recommended_price / wholesale) - 1) * 100, 2) if wholesale > 0 else 30.0
    change = round(recommended_markup_pct - 30.0, 1)
    change_str = f"+{change}%" if change >= 0 else f"{change}%"

    return {
        "local_avg": local_avg,
        "local_prices": local_prices_list,
        "num_sources": num_sources,
        "recommended_price": recommended_price,
        "recommended_markup_pct": recommended_markup_pct,
        "change_from_30_pct": change_str,
        "direction": direction,
    }


# ═══════════════════════════════════════════════════════════════════
# Summary generation
# ═══════════════════════════════════════════════════════════════════

def generate_summary(results):
    """Generate human-readable summary text."""
    raises = [r for r in results if r["direction"] == "raise"]
    drops = [r for r in results if r["direction"] == "drop"]
    keeps = [r for r in results if r["direction"] == "keep"]
    no_data = [r for r in results if r["direction"] == "no_data"]
    with_data = raises + drops + keeps

    lines = []
    lines.append("=" * 80)
    lines.append("MARKET-DRIVEN PRICING ANALYSIS -- LuxuryLaneTires.ca")
    lines.append("Generated: 2026-05-18")
    lines.append("=" * 80)
    lines.append("")

    # ── Overview ──
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total tires analyzed:      {len(results)}")
    lines.append(f"Tires with local data:     {len(with_data)}")
    lines.append(f"Tires with no data:        {len(no_data)}")
    pct = round(len(with_data) / len(results) * 100, 1) if results else 0
    lines.append(f"Coverage:                  {pct}%")
    lines.append("")

    # ── Recommendation counts ──
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append(f"RAISE  (local avg > our 30%):   {len(raises)}")
    lines.append(f"DROP   (local avg < our 30%):   {len(drops)}")
    lines.append(f"KEEP   (within $10 of 30%):     {len(keeps)}")
    lines.append(f"NO DATA:                        {len(no_data)}")
    lines.append("")

    # ── Revenue impact ──
    total_current = sum(r["current_price"] for r in results)
    total_recommended = sum(r["recommended_price"] for r in results)
    delta = total_recommended - total_current
    lines.append("REVENUE IMPACT (per-unit, full catalog)")
    lines.append("-" * 40)
    lines.append(f"Total current revenue (per unit):      ${total_current:,.2f}")
    lines.append(f"Total recommended revenue (per unit):  ${total_recommended:,.2f}")
    lines.append(f"Delta:                                 ${delta:+,.2f}")
    if total_current > 0:
        lines.append(f"Change:                                {delta / total_current * 100:+.1f}%")
    lines.append("")

    # ── Top 20 biggest raises ──
    lines.append("=" * 80)
    lines.append("TOP 20 BIGGEST RAISES (by % above 30% markup)")
    lines.append("=" * 80)
    header = f"{'Brand':<14} {'Model':<32} {'Size':<14} {'Markup%':<9} {'Change':<9} {'Rec$':<10} {'Src'}"
    lines.append(header)
    lines.append("-" * len(header))

    raises_sorted = sorted(raises, key=lambda r: r["recommended_markup_pct"], reverse=True)
    for r in raises_sorted[:20]:
        lines.append(
            f"{r['brand']:<14} {r['model']:<32} {r['size']:<14} "
            f"{r['recommended_markup_pct']:.1f}%    "
            f"{r['change_from_30_pct']:<9} "
            f"${r['recommended_price']:<9.2f} {r['num_sources']}"
        )
    lines.append("")

    # ── Top 20 biggest drops ──
    lines.append("=" * 80)
    lines.append("TOP 20 BIGGEST DROPS (by % below 30% markup)")
    lines.append("=" * 80)
    lines.append(header)
    lines.append("-" * len(header))

    drops_sorted = sorted(drops, key=lambda r: r["recommended_markup_pct"])
    for r in drops_sorted[:20]:
        lines.append(
            f"{r['brand']:<14} {r['model']:<32} {r['size']:<14} "
            f"{r['recommended_markup_pct']:.1f}%    "
            f"{r['change_from_30_pct']:<9} "
            f"${r['recommended_price']:<9.2f} {r['num_sources']}"
        )
    lines.append("")

    # ── Brand-level summary ──
    lines.append("=" * 80)
    lines.append("BRAND-LEVEL SUMMARY")
    lines.append("=" * 80)
    brand_hdr = f"{'Brand':<14} {'Avg Markup%':<13} {'With Data':<11} {'Total':<8} {'Coverage%':<10} {'Raises':<8} {'Drops':<8} {'Keeps'}"
    lines.append(brand_hdr)
    lines.append("-" * len(brand_hdr))

    brand_groups = defaultdict(list)
    for r in results:
        brand_groups[r["brand"]].append(r)

    for brand in sorted(brand_groups.keys()):
        items = brand_groups[brand]
        avg_markup = round(sum(r["recommended_markup_pct"] for r in items) / len(items), 1)
        wd = len([r for r in items if r["direction"] != "no_data"])
        total = len(items)
        cov = round(wd / total * 100, 1) if total > 0 else 0
        r_count = len([r for r in items if r["direction"] == "raise"])
        d_count = len([r for r in items if r["direction"] == "drop"])
        k_count = len([r for r in items if r["direction"] == "keep"])
        lines.append(
            f"{brand:<14} {avg_markup:.1f}%        {wd:<11} {total:<8} {cov:.1f}%      "
            f"{r_count:<8} {d_count:<8} {k_count}"
        )
    lines.append("")

    # ── Local shop source counts ──
    lines.append("=" * 80)
    lines.append("LOCAL SHOP SOURCE COUNTS")
    lines.append("=" * 80)
    shop_counts = defaultdict(int)
    for r in results:
        for lp in r.get("local_prices", []):
            shop_counts[lp["shop"]] += 1
    for shop, count in sorted(shop_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {shop:<30} {count} matched tires")
    lines.append("")

    # ── Matched tires detail ──
    lines.append("=" * 80)
    lines.append("MATCHED TIRES DETAIL (raises and drops)")
    lines.append("=" * 80)
    for r in results:
        if r["direction"] in ("raise", "drop"):
            baseline_30 = round(r["wholesale"] * 1.30, 2)
            lines.append("")
            lines.append(f"  {r['brand']} {r['model']} -- {r['size']}")
            lines.append(f"    Wholesale: ${r['wholesale']:.2f}  |  30% price: ${baseline_30:.2f}  |  Local avg: ${r['local_avg']:.2f}")
            lines.append(f"    Current: ${r['current_price']:.2f} ({r['current_markup_pct']:.1f}%)  -->  Recommended: ${r['recommended_price']:.2f} ({r['recommended_markup_pct']:.1f}%)")
            lines.append(f"    Direction: {r['direction'].upper()}  |  Change from 30%: {r['change_from_30_pct']}")
            for lp in r["local_prices"]:
                lines.append(f"      - {lp['shop']}: ${lp['price']:.2f}")

    lines.append("")
    lines.append("-- End of Report --")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Loading inventory...")
    inventory = load_inventory()
    print(f"  {len(inventory)} tires in inventory")

    print("Loading local prices...")
    local_prices = load_all_local_prices()
    print(f"  Total: {len(local_prices)} local price entries")

    print("Building index...")
    local_index = build_local_index(local_prices)
    print(f"  {len(local_index)} unique brand/model/size combos with local data")

    print("Matching and calculating recommendations...")
    results = []

    # Featured product slugs on the live site (Michelin is NOT among them)
    LIVE_BRANDS = {"Bridgestone", "Continental", "Falken", "Firestone", "Hankook", "Michelin", "Pirelli", "Toyo", "Yokohama"}

    for tire in inventory:
        brand = (tire.get("brand") or tire.get("brand_name", "")).strip()
        model = normalize_model(tire["model"].strip())
        size = normalize_size(tire["size"].strip())
        wholesale = tire.get("price_wholesale", 0) or 0
        current_retail = tire.get("price_retail", 0) or 0

        if wholesale <= 0:
            continue

        # Skip Michelin — not live on site yet
        if brand not in LIVE_BRANDS:
            continue

        matches = find_local_matches(brand, model, size, local_index)
        rec = compute_recommendation(wholesale, current_retail, matches)

        # Yokohama without local data: keep current markup (50% X-AT, 35% X-CV)
        # instead of defaulting to 30%
        if brand == "Yokohama" and not matches:
            if "X-AT" in model:
                rec["recommended_price"] = round(wholesale * 1.50, 2)
                rec["recommended_markup_pct"] = 50.0
                rec["change_from_30_pct"] = "+20.0%"
                rec["direction"] = "no_data"
            elif "X-CV" in model:
                rec["recommended_price"] = round(wholesale * 1.35, 2)
                rec["recommended_markup_pct"] = 35.0
                rec["change_from_30_pct"] = "+5.0%"
                rec["direction"] = "no_data"

        current_markup_pct = round(((current_retail / wholesale) - 1) * 100, 2)

        result = {
            "brand": brand,
            "model": model,
            "size": size,
            "wholesale": wholesale,
            "current_price": current_retail,
            "current_markup_pct": current_markup_pct,
            **rec,
        }
        results.append(result)

    # Sort: matched first (by brand, model, size), then no_data
    results.sort(key=lambda r: (r["direction"] == "no_data", r["brand"], r["model"], r["size"]))

    # Write JSON results
    output_path = os.path.join(SCRIPTS_DIR, "market-pricing-results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to market-pricing-results.json")

    # Write summary
    summary_text = generate_summary(results)
    summary_path = os.path.join(SCRIPTS_DIR, "market-pricing-summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Wrote summary to market-pricing-summary.txt")

    # Quick stdout summary
    raises = [r for r in results if r["direction"] == "raise"]
    drops = [r for r in results if r["direction"] == "drop"]
    keeps = [r for r in results if r["direction"] == "keep"]
    no_data = [r for r in results if r["direction"] == "no_data"]
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(raises)} raises, {len(drops)} drops, {len(keeps)} keeps, {len(no_data)} no_data")
    total_current = sum(r["current_price"] for r in results)
    total_rec = sum(r["recommended_price"] for r in results)
    print(f"Revenue impact: ${total_rec - total_current:+,.2f} per unit across catalog")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
