#!/usr/bin/env python3
"""
Michelin-Only Market-Driven Pricing for LuxuryLaneTires.ca

Processes ONLY Michelin tires from inventory against local competitor data.

Pricing rules:
  - Baseline: wholesale + 30%
  - If local avg > baseline by >$10: RAISE to (local_avg - $5), cap at 60% markup
  - If local avg < baseline by >$10: DROP to (local_avg - $5), floor at 20% markup
  - If local avg within $10 of baseline: KEEP at 30%
  - If no local data: mark "no_data", keep at 30%
  - DELIST: if wholesale >= local retail average (can't compete)
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Markup parameters
MARKUP_FLOOR = 0.20   # 20% minimum
MARKUP_CAP = 0.60     # 60% maximum
MARKUP_DEFAULT = 0.30  # 30% default (no data)

# ═══════════════════════════════════════════════════════════════════
# Model name aliases — Michelin-relevant entries
# ═══════════════════════════════════════════════════════════════════

MODEL_ALIASES = {
    "Defender LTX M/S 2": "Defender LTX M/S2",
    "Pilot Sport A/S 4": "Pilot Sport AS 4",
    "Pilot Sport All Season 4": "Pilot Sport AS 4",
    "Michelin Pilot Sport A/S 4": "Pilot Sport AS 4",
    "Primacy Tour A/S": "Primacy Tour AS",
}


# ═══════════════════════════════════════════════════════════════════
# Size normalization
# ═══════════════════════════════════════════════════════════════════

def normalize_size(size_str):
    s = size_str.strip().upper()
    m = re.match(r'((?:LT|P)?\d{3}/\d{2,3}R\d{2})\b', s)
    if m:
        return m.group(1).replace(" ", "")
    m = re.match(r'((?:LT|P)?\d{3}/\d{2,3}R\d{2})', s)
    if m:
        return m.group(1).replace(" ", "")
    m = re.match(r'(\d+[X][\d.]+R\d{2})\b', s)
    if m:
        return m.group(1).replace(" ", "")
    m = re.match(r'(\d+[X][\d.]+R\d{2})', s)
    if m:
        return m.group(1).replace(" ", "")
    return s.replace(" ", "")


def normalize_model(model_str):
    m = model_str.strip()
    if m in MODEL_ALIASES:
        return MODEL_ALIASES[m]
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

        # Only keep Michelin entries
        if brand.lower() != "michelin":
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
    shops = [
        ("local-prices-canadian-tire.json", "Canadian Tire"),
        ("local-prices-tdot-performance.json", "TDot Performance"),
        ("local-prices-point-s.json", "Point S"),
        ("local-prices-noble-tire.json", "Noble Tire Brampton"),
        ("local-prices-zracing.json", "ZRacing Mississauga"),
        ("local-prices-krave.json", "KRAVE"),
        ("local-prices-active-green-ross.json", "Active Green+Ross"),
        ("local-prices-fas-tire.json", "FAS Tire"),
        ("local-prices-michelin.json", "Michelin (aggregated)"),
    ]

    all_prices = []
    for filename, label in shops:
        entries = load_shop_data(filename, label)
        all_prices.extend(entries)
        if entries:
            print(f"  {label}: {len(entries)} Michelin entries")

    return all_prices


# ═══════════════════════════════════════════════════════════════════
# Matching and pricing logic
# ═══════════════════════════════════════════════════════════════════

def build_local_index(local_prices):
    index = defaultdict(list)
    for item in local_prices:
        key = (item["brand"].lower(), item["model"].lower(), item["size"])
        index[key].append(item)
    return index


def find_local_matches(brand, model, size, local_index):
    key = (brand.lower(), model.lower(), size)
    return local_index.get(key, [])


def compute_recommendation(wholesale, current_retail, local_matches):
    baseline = round(wholesale * (1 + MARKUP_DEFAULT), 2)

    if not local_matches:
        # No local data: default to 30%
        return {
            "local_avg": None,
            "local_prices": [],
            "num_sources": 0,
            "recommended_price": baseline,
            "recommended_markup_pct": MARKUP_DEFAULT * 100,
            "change_from_30_pct": "0.0%",
            "direction": "no_data",
        }

    # Deduplicate: take cheapest price per shop
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

    # DELIST CHECK: if wholesale >= local retail average, we can't compete
    if wholesale >= local_avg:
        return {
            "local_avg": local_avg,
            "local_prices": local_prices_list,
            "num_sources": num_sources,
            "recommended_price": 0,
            "recommended_markup_pct": 0,
            "change_from_30_pct": "DELIST",
            "direction": "delist",
        }

    diff = local_avg - baseline

    if abs(diff) <= 10:
        recommended_price = baseline
        direction = "keep"
    elif diff > 0:
        # Local avg higher -- raise
        recommended_price = round(local_avg - 5, 2)
        max_price = round(wholesale * (1 + MARKUP_CAP), 2)
        if recommended_price > max_price:
            recommended_price = max_price
        direction = "raise"
    else:
        # Local avg lower -- drop
        recommended_price = round(local_avg - 5, 2)
        min_price = round(wholesale * (1 + MARKUP_FLOOR), 2)
        if recommended_price < min_price:
            recommended_price = min_price
        direction = "drop"

    recommended_markup_pct = round(((recommended_price / wholesale) - 1) * 100, 2) if wholesale > 0 else MARKUP_DEFAULT * 100
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
    raises = [r for r in results if r["direction"] == "raise"]
    drops = [r for r in results if r["direction"] == "drop"]
    keeps = [r for r in results if r["direction"] == "keep"]
    no_data = [r for r in results if r["direction"] == "no_data"]
    delisted = [r for r in results if r["direction"] == "delist"]
    with_data = raises + drops + keeps + delisted

    lines = []
    lines.append("=" * 80)
    lines.append("MICHELIN MARKET-DRIVEN PRICING ANALYSIS -- LuxuryLaneTires.ca")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Rules: {int(MARKUP_FLOOR*100)}% floor, {int(MARKUP_CAP*100)}% cap, {int(MARKUP_DEFAULT*100)}% default (no data)")
    lines.append("=" * 80)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total Michelin tires analyzed: {len(results)}")
    lines.append(f"Tires with local data:        {len(with_data)}")
    lines.append(f"Tires with no data:           {len(no_data)}")
    pct = round(len(with_data) / len(results) * 100, 1) if results else 0
    lines.append(f"Coverage:                     {pct}%")
    lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append(f"RAISE  (local avg > our 30%):   {len(raises)}")
    lines.append(f"DROP   (local avg < our 30%):   {len(drops)}")
    lines.append(f"KEEP   (within $10 of 30%):     {len(keeps)}")
    lines.append(f"DELIST (wholesale >= local avg): {len(delisted)}")
    lines.append(f"NO DATA (default 30%):          {len(no_data)}")
    lines.append("")

    # Revenue impact (exclude delisted from totals)
    active = [r for r in results if r["direction"] != "delist"]
    total_current = sum(r["current_price"] for r in active)
    total_recommended = sum(r["recommended_price"] for r in active)
    delta = total_recommended - total_current
    lines.append("REVENUE IMPACT (per-unit, active catalog only)")
    lines.append("-" * 40)
    lines.append(f"Active tires (not delisted):          {len(active)}")
    lines.append(f"Total current revenue (per unit):     ${total_current:,.2f}")
    lines.append(f"Total recommended revenue (per unit): ${total_recommended:,.2f}")
    lines.append(f"Delta:                                ${delta:+,.2f}")
    if total_current > 0:
        lines.append(f"Change:                               {delta / total_current * 100:+.1f}%")
    lines.append("")

    # Delisted tires
    if delisted:
        lines.append("=" * 80)
        lines.append(f"DELISTED TIRES ({len(delisted)}) -- wholesale >= local retail avg")
        lines.append("=" * 80)
        header = f"{'Model':<28} {'Size':<14} {'Wholesale':<12} {'Local Avg':<12} {'Gap':<10} {'Src'}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in sorted(delisted, key=lambda x: (x["model"], x["size"])):
            gap = round(r["wholesale"] - r["local_avg"], 2)
            lines.append(
                f"{r['model']:<28} {r['size']:<14} "
                f"${r['wholesale']:<11.2f} ${r['local_avg']:<11.2f} ${gap:<9.2f} {r['num_sources']}"
            )
        lines.append("")

    # Model-level summary
    lines.append("=" * 80)
    lines.append("MODEL-LEVEL SUMMARY")
    lines.append("=" * 80)
    model_hdr = f"{'Model':<28} {'Avg Markup%':<13} {'With Data':<11} {'Total':<8} {'Raises':<8} {'Drops':<8} {'Keeps':<8} {'Delist'}"
    lines.append(model_hdr)
    lines.append("-" * len(model_hdr))

    model_groups = defaultdict(list)
    for r in results:
        model_groups[r["model"]].append(r)

    for model in sorted(model_groups.keys()):
        items = model_groups[model]
        active_items = [r for r in items if r["direction"] != "delist"]
        avg_markup = round(sum(r["recommended_markup_pct"] for r in active_items) / len(active_items), 1) if active_items else 0
        wd = len([r for r in items if r["direction"] not in ("no_data",)])
        total = len(items)
        r_count = len([r for r in items if r["direction"] == "raise"])
        d_count = len([r for r in items if r["direction"] == "drop"])
        k_count = len([r for r in items if r["direction"] == "keep"])
        dl_count = len([r for r in items if r["direction"] == "delist"])
        lines.append(
            f"{model:<28} {avg_markup:.1f}%        {wd:<11} {total:<8} "
            f"{r_count:<8} {d_count:<8} {k_count:<8} {dl_count}"
        )
    lines.append("")

    # Top raises
    lines.append("=" * 80)
    lines.append("TOP 20 BIGGEST RAISES (by % above 30% markup)")
    lines.append("=" * 80)
    header = f"{'Model':<28} {'Size':<14} {'Markup%':<9} {'Change':<9} {'Rec$':<10} {'Local Avg':<12} {'Src'}"
    lines.append(header)
    lines.append("-" * len(header))

    raises_sorted = sorted(raises, key=lambda r: r["recommended_markup_pct"], reverse=True)
    for r in raises_sorted[:20]:
        lines.append(
            f"{r['model']:<28} {r['size']:<14} "
            f"{r['recommended_markup_pct']:.1f}%    "
            f"{r['change_from_30_pct']:<9} "
            f"${r['recommended_price']:<9.2f} ${r['local_avg']:<11.2f} {r['num_sources']}"
        )
    lines.append("")

    # Top drops
    lines.append("=" * 80)
    lines.append("TOP 20 BIGGEST DROPS (by % below 30% markup)")
    lines.append("=" * 80)
    lines.append(header)
    lines.append("-" * len(header))

    drops_sorted = sorted(drops, key=lambda r: r["recommended_markup_pct"])
    for r in drops_sorted[:20]:
        lines.append(
            f"{r['model']:<28} {r['size']:<14} "
            f"{r['recommended_markup_pct']:.1f}%    "
            f"{r['change_from_30_pct']:<9} "
            f"${r['recommended_price']:<9.2f} ${r['local_avg']:<11.2f} {r['num_sources']}"
        )
    lines.append("")

    # Local shop source counts
    lines.append("=" * 80)
    lines.append("LOCAL SHOP SOURCE COUNTS")
    lines.append("=" * 80)
    shop_counts = defaultdict(int)
    for r in results:
        for lp in r.get("local_prices", []):
            shop_counts[lp["shop"]] += 1
    for shop, count in sorted(shop_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {shop:<30} {count} matched sizes")
    lines.append("")

    # Matched tires detail (raises and drops)
    lines.append("=" * 80)
    lines.append("MATCHED TIRES DETAIL (raises and drops)")
    lines.append("=" * 80)
    for r in results:
        if r["direction"] in ("raise", "drop"):
            baseline_30 = round(r["wholesale"] * 1.30, 2)
            lines.append("")
            lines.append(f"  Michelin {r['model']} -- {r['size']}")
            lines.append(f"    Wholesale: ${r['wholesale']:.2f}  |  30% price: ${baseline_30:.2f}  |  Local avg: ${r['local_avg']:.2f}")
            lines.append(f"    Current: ${r['current_price']:.2f} ({r['current_markup_pct']:.1f}%)  -->  Recommended: ${r['recommended_price']:.2f} ({r['recommended_markup_pct']:.1f}%)")
            lines.append(f"    Direction: {r['direction'].upper()}  |  Change from 30%: {r['change_from_30_pct']}")
            for lp in r["local_prices"]:
                lines.append(f"      - {lp['shop']}: ${lp['price']:.2f}")

    lines.append("")
    lines.append("-- End of Michelin Pricing Report --")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Loading inventory (Michelin only)...")
    inventory = load_inventory()
    michelin_inv = [t for t in inventory if t["brand"].strip() == "Michelin"]
    print(f"  {len(michelin_inv)} Michelin tires in inventory (of {len(inventory)} total)")

    print("Loading local prices (Michelin only)...")
    local_prices = load_all_local_prices()
    print(f"  Total: {len(local_prices)} Michelin local price entries")

    print("Building index...")
    local_index = build_local_index(local_prices)
    print(f"  {len(local_index)} unique model/size combos with local data")

    print("Matching and calculating recommendations...")
    results = []

    for tire in michelin_inv:
        brand = tire["brand"].strip()
        model = normalize_model(tire["model"].strip())
        size = normalize_size(tire["size"].strip())
        wholesale = tire.get("price_wholesale", 0) or 0
        current_retail = tire.get("price_retail", 0) or 0

        if wholesale <= 0:
            continue

        matches = find_local_matches(brand, model, size, local_index)
        rec = compute_recommendation(wholesale, current_retail, matches)

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

    # Sort: delisted first, then matched (by model, size), then no_data
    direction_order = {"delist": 0, "raise": 1, "drop": 1, "keep": 1, "no_data": 2}
    results.sort(key=lambda r: (direction_order.get(r["direction"], 9), r["model"], r["size"]))

    # Write JSON results
    output_path = os.path.join(SCRIPTS_DIR, "michelin-pricing-results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to michelin-pricing-results.json")

    # Write summary
    summary_text = generate_summary(results)
    summary_path = os.path.join(SCRIPTS_DIR, "michelin-pricing-summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Wrote summary to michelin-pricing-summary.txt")

    # Quick stdout summary
    raises = [r for r in results if r["direction"] == "raise"]
    drops = [r for r in results if r["direction"] == "drop"]
    keeps = [r for r in results if r["direction"] == "keep"]
    no_data = [r for r in results if r["direction"] == "no_data"]
    delisted = [r for r in results if r["direction"] == "delist"]
    active = [r for r in results if r["direction"] != "delist"]

    print(f"\n{'=' * 60}")
    print(f"MICHELIN RESULTS:")
    print(f"  {len(raises)} raises, {len(drops)} drops, {len(keeps)} keeps")
    print(f"  {len(delisted)} delisted (wholesale >= local avg)")
    print(f"  {len(no_data)} no_data (default 30%)")
    total_current = sum(r["current_price"] for r in active)
    total_rec = sum(r["recommended_price"] for r in active)
    print(f"Revenue impact: ${total_rec - total_current:+,.2f} per unit across active catalog")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
