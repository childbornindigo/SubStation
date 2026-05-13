#!/usr/bin/env python3
"""
analyze_tires.py — Brand ranking analysis for Jun's tire business.

Reads scraped tire data from market-intelligence/ and produces
composite brand rankings: popularity, coverage, value, and category leaders.

Usage:
    python analyze_tires.py
    python analyze_tires.py --data-dir /path/to/market-intelligence
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict | list | None:
    """Load a JSON file, return None if missing or corrupt."""
    if not os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} not found", file=sys.stderr)
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"  [ok]   {os.path.basename(path)} loaded", file=sys.stderr)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [fail] {os.path.basename(path)}: {e}", file=sys.stderr)
        return None


def load_all_data(data_dir: str) -> tuple[list, dict, dict]:
    """Load retail, marketplace, and trends data. Returns (retail, marketplace, trends)."""
    print("Loading data files...", file=sys.stderr)
    retail = load_json(os.path.join(data_dir, "retail-data.json"))
    marketplace = load_json(os.path.join(data_dir, "marketplace-demand.json"))
    trends = load_json(os.path.join(data_dir, "trends-data.json"))

    # Normalize retail to flat list of listings
    retail_listings = normalize_retail(retail) if retail else []
    marketplace_data = marketplace if isinstance(marketplace, dict) else {}
    trends_data = trends if isinstance(trends, dict) else {}

    return retail_listings, marketplace_data, trends_data


def normalize_retail(retail_raw) -> list[dict]:
    """
    Normalize retail data into a flat list of listing dicts.

    Accepts either:
      - A list of listing dicts directly
      - A dict keyed by tire size, each value a list of listings
      - A dict with a "results" or "data" key wrapping either format
    """
    # Unwrap common envelope keys
    if isinstance(retail_raw, dict):
        for key in ("results", "data", "listings", "sizes"):
            if key in retail_raw and isinstance(retail_raw[key], (list, dict)):
                retail_raw = retail_raw[key]
                break

    listings = []

    if isinstance(retail_raw, list):
        # Flat list of listings
        for item in retail_raw:
            if isinstance(item, dict):
                listings.append(_normalize_listing(item))
    elif isinstance(retail_raw, dict):
        # Dict keyed by size
        for size, items in retail_raw.items():
            # Handle {size: {listings: [...]}} or {size: [...]}
            if isinstance(items, dict) and "listings" in items:
                items = items["listings"]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        listing = _normalize_listing(item)
                        if not listing.get("size"):
                            listing["size"] = size
                        listings.append(listing)
    return listings


def _normalize_listing(item: dict) -> dict:
    """Ensure a listing has the standard fields."""
    price = item.get("price")
    if isinstance(price, str):
        price = price.replace("$", "").replace(",", "").strip()
        try:
            price = float(price)
        except ValueError:
            price = None

    rating = item.get("rating")
    if isinstance(rating, str):
        try:
            rating = float(rating)
        except ValueError:
            rating = None

    return {
        "brand": _clean_brand(item.get("brand", "")),
        "model": item.get("model", ""),
        "price": price,
        "rating": rating,
        "type": _normalize_type(item.get("type", "")),
        "source": item.get("source", item.get("retailer", "")),
        "size": item.get("size", ""),
    }


# ---------------------------------------------------------------------------
# Brand / type normalization
# ---------------------------------------------------------------------------

# Known brand aliases and common misspellings
BRAND_ALIASES = {
    "bf goodrich": "BFGoodrich",
    "bfgoodrich": "BFGoodrich",
    "bf_goodrich": "BFGoodrich",
    "general tire": "General",
    "gt radial": "GT Radial",
    "gtradial": "GT Radial",
    "hankook tire": "Hankook",
    "nokian tyres": "Nokian",
    "nokian tires": "Nokian",
    "kumho tire": "Kumho",
    "nitto tire": "Nitto",
    "nitto tires": "Nitto",
    "michelin tires": "Michelin",
    "bridgestone tires": "Bridgestone",
    "goodyear tires": "Goodyear",
    "continental tires": "Continental",
}

# Known winter-specialist brands (fallback when type data is sparse)
WINTER_BRANDS = {"Nokian", "Gislaved"}
PERFORMANCE_BRANDS = {"Pirelli"}  # brands strongly associated with performance


def _clean_brand(brand: str) -> str:
    """Normalize brand name to canonical form."""
    if not brand:
        return "Unknown"
    brand = brand.strip()
    lower = brand.lower()
    if lower in BRAND_ALIASES:
        return BRAND_ALIASES[lower]
    # Title-case, but preserve known casing
    return brand.title() if brand == brand.lower() or brand == brand.upper() else brand


TYPE_MAP = {
    "all-season": "all_season",
    "all season": "all_season",
    "allseason": "all_season",
    "all_season": "all_season",
    "a/s": "all_season",
    "touring": "all_season",
    "winter": "winter",
    "snow": "winter",
    "ice": "winter",
    "studded": "winter",
    "studless": "winter",
    "performance": "performance",
    "sport": "performance",
    "ultra high performance": "performance",
    "uhp": "performance",
    "summer": "performance",
    "max performance": "performance",
    "all-weather": "all_season",
    "all weather": "all_season",
    "truck": "all_season",
    "suv": "all_season",
    "highway": "all_season",
    "h/t": "all_season",
    "a/t": "all_season",
    "all-terrain": "all_season",
    "mud": "all_season",
    "m/t": "all_season",
}


def _normalize_type(tire_type: str) -> str:
    """Normalize tire type to one of: all_season, winter, performance, or empty."""
    if not tire_type:
        return ""
    lower = tire_type.lower().strip()
    for key, val in TYPE_MAP.items():
        if key in lower:
            return val
    return ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of {key: raw_score} to 0-100 scale."""
    if not scores:
        return {}
    max_val = max(scores.values())
    if max_val == 0:
        return {k: 0.0 for k in scores}
    return {k: round((v / max_val) * 100, 1) for k, v in scores.items()}


def compute_popularity(listings: list[dict], trends: dict, marketplace: dict) -> dict[str, float]:
    """
    Brand Popularity Score.
    - Listing count per brand (more retailer listings = more demand signal)
    - Weighted by Google Trends interest for sizes they appear in
    - Weighted by marketplace listing counts for their sizes
    """
    brand_listing_count = defaultdict(int)
    brand_sizes = defaultdict(set)

    for listing in listings:
        brand = listing["brand"]
        if brand == "Unknown":
            continue
        brand_listing_count[brand] += 1
        if listing["size"]:
            brand_sizes[brand].add(listing["size"])

    # Base score: listing count
    raw = {}
    for brand, count in brand_listing_count.items():
        score = float(count)

        # Boost by trends interest for sizes this brand covers
        if trends:
            trends_boost = 0.0
            sizes = brand_sizes.get(brand, set())
            for size in sizes:
                interest = _get_trends_interest(trends, size)
                if interest:
                    trends_boost += interest
            if sizes:
                trends_boost = trends_boost / len(sizes)  # avg interest
                score *= (1 + trends_boost / 100)  # scale: 100 trend = 2x

        # Boost by marketplace demand for sizes this brand covers
        if marketplace:
            mp_boost = 0.0
            sizes = brand_sizes.get(brand, set())
            for size in sizes:
                mp_count = _get_marketplace_count(marketplace, size)
                if mp_count:
                    mp_boost += mp_count
            if sizes:
                mp_boost = mp_boost / len(sizes)
                score *= (1 + min(mp_boost, 100) / 100)

        raw[brand] = score

    return normalize_scores(raw)


def _get_trends_interest(trends: dict, size: str) -> float:
    """Extract trends interest for a tire size. Handles various data shapes."""
    # Try direct key match
    if size in trends:
        val = trends[size]
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            return float(val.get("interest", val.get("value", val.get("score", 0))))
        if isinstance(val, list) and val:
            # Take the latest/max value
            nums = [x for x in val if isinstance(x, (int, float))]
            return max(nums) if nums else 0.0

    # Try nested "sizes" or "data" key
    for container_key in ("sizes", "data", "results"):
        if container_key in trends and isinstance(trends[container_key], dict):
            return _get_trends_interest(trends[container_key], size)

    return 0.0


def _get_marketplace_count(marketplace: dict, size: str) -> float:
    """Extract marketplace listing count for a tire size."""
    if size in marketplace:
        val = marketplace[size]
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            total = 0
            for platform in ("kijiji", "ebay", "facebook", "total", "count"):
                if platform in val and isinstance(val[platform], (int, float)):
                    total += val[platform]
            # If we got specific platforms, use that; otherwise try "total"
            if total > 0:
                return float(total)
            # Try any numeric value
            nums = [v for v in val.values() if isinstance(v, (int, float))]
            return sum(nums) if nums else 0.0

    for container_key in ("sizes", "data", "results"):
        if container_key in marketplace and isinstance(marketplace[container_key], dict):
            return _get_marketplace_count(marketplace[container_key], size)

    return 0.0


def compute_coverage(listings: list[dict]) -> dict[str, dict]:
    """
    Brand Coverage Score.
    Returns {brand: {score, sizes_count, retailers, sizes_list}} normalized to 0-100.
    """
    brand_sizes = defaultdict(set)
    brand_retailers = defaultdict(set)

    for listing in listings:
        brand = listing["brand"]
        if brand == "Unknown":
            continue
        if listing["size"]:
            brand_sizes[brand].add(listing["size"])
        if listing["source"]:
            brand_retailers[brand].add(listing["source"])

    raw_scores = {}
    details = {}
    for brand in brand_sizes:
        size_count = len(brand_sizes[brand])
        retailer_count = len(brand_retailers[brand])
        # Coverage = sizes * retailer breadth
        raw_scores[brand] = size_count * (1 + retailer_count * 0.5)
        details[brand] = {
            "sizes_count": size_count,
            "retailers": sorted(brand_retailers[brand]),
            "retailer_count": retailer_count,
        }

    normed = normalize_scores(raw_scores)
    result = {}
    for brand in normed:
        result[brand] = {
            "score": normed[brand],
            **details[brand],
        }
    return result


def compute_value(listings: list[dict]) -> dict[str, dict]:
    """
    Brand Value Score.
    Returns {brand: {score, avg_price, price_stdev, tier, listing_count}}.
    """
    brand_prices = defaultdict(list)

    for listing in listings:
        brand = listing["brand"]
        if brand == "Unknown":
            continue
        if listing["price"] and listing["price"] > 0:
            brand_prices[brand].append(listing["price"])

    details = {}
    for brand, prices in brand_prices.items():
        avg = mean(prices)
        sd = stdev(prices) if len(prices) > 1 else 0.0
        consistency = 1 - min(sd / avg, 1.0) if avg > 0 else 0.0  # 1=very consistent

        if avg < 120:
            tier = "budget"
        elif avg < 200:
            tier = "midrange"
        else:
            tier = "premium"

        # Value score: higher price with good consistency = more margin potential
        # Budget brands score lower on margin potential
        margin_potential = avg * consistency
        details[brand] = {
            "avg_price": round(avg, 2),
            "price_stdev": round(sd, 2),
            "consistency": round(consistency, 2),
            "tier": tier,
            "listing_count": len(prices),
            "margin_potential": margin_potential,
        }

    # Normalize margin_potential to 0-100 as the value score
    raw = {b: d["margin_potential"] for b, d in details.items()}
    normed = normalize_scores(raw)
    for brand in normed:
        details[brand]["score"] = normed[brand]

    return details


def compute_category_leaders(listings: list[dict]) -> dict[str, list]:
    """
    Top 5 brands per category: all_season, winter, performance.
    Falls back to brand knowledge if type data is sparse.
    """
    category_brands = defaultdict(lambda: defaultdict(int))

    typed_count = 0
    for listing in listings:
        brand = listing["brand"]
        if brand == "Unknown":
            continue
        tire_type = listing.get("type", "")
        if tire_type:
            category_brands[tire_type][brand] += 1
            typed_count += 1

    # If <20% of listings have type data, supplement with brand knowledge
    type_coverage = typed_count / len(listings) if listings else 0
    sparse_types = type_coverage < 0.2

    if sparse_types:
        # Add known specialists to their categories
        all_brands_seen = {l["brand"] for l in listings if l["brand"] != "Unknown"}
        for brand in all_brands_seen:
            if brand in WINTER_BRANDS:
                category_brands["winter"][brand] = max(category_brands["winter"].get(brand, 0), 1)
            if brand in PERFORMANCE_BRANDS:
                category_brands["performance"][brand] = max(category_brands["performance"].get(brand, 0), 1)

    result = {}
    for category in ("all_season", "winter", "performance"):
        brands = category_brands.get(category, {})
        sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:5]
        result[category] = [{"brand": b, "listings": c} for b, c in sorted_brands]

    return result


def compute_composite(
    popularity: dict[str, float],
    coverage: dict[str, dict],
    value: dict[str, dict],
) -> list[dict]:
    """
    Overall Top 10 Brands.
    Composite = popularity (40%) + coverage (30%) + value/margin potential (30%).
    """
    all_brands = set(popularity.keys()) | set(coverage.keys()) | set(value.keys())

    scored = []
    for brand in all_brands:
        pop_score = popularity.get(brand, 0)
        cov_score = coverage[brand]["score"] if brand in coverage else 0
        val_score = value[brand]["score"] if brand in value else 0

        composite = pop_score * 0.4 + cov_score * 0.3 + val_score * 0.3

        entry = {
            "brand": brand,
            "composite_score": round(composite, 1),
            "popularity": round(pop_score, 1),
            "coverage": round(cov_score, 1),
            "value": round(val_score, 1),
            "avg_price": value[brand]["avg_price"] if brand in value else None,
            "tier": value[brand]["tier"] if brand in value else "unknown",
            "sizes_available": coverage[brand]["sizes_count"] if brand in coverage else 0,
            "retailers": coverage[brand]["retailer_count"] if brand in coverage else 0,
        }
        scored.append(entry)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    # Add rank
    for i, entry in enumerate(scored):
        entry["rank"] = i + 1

    return scored[:10]


# ---------------------------------------------------------------------------
# Size opportunity analysis
# ---------------------------------------------------------------------------

def compute_size_opportunities(
    listings: list[dict], trends: dict, marketplace: dict
) -> dict:
    """Find highest-demand sizes and best margin sizes."""
    size_prices = defaultdict(list)
    size_brands = defaultdict(set)

    for listing in listings:
        size = listing.get("size", "")
        if not size:
            continue
        if listing["price"] and listing["price"] > 0:
            size_prices[size].append(listing["price"])
        if listing["brand"] != "Unknown":
            size_brands[size].add(listing["brand"])

    # Highest demand: combine trends + marketplace signals
    demand_scores = {}
    all_sizes = set(size_prices.keys())
    if trends:
        for size in all_sizes:
            interest = _get_trends_interest(trends, size)
            mp_count = _get_marketplace_count(marketplace, size) if marketplace else 0
            demand_scores[size] = interest + mp_count * 2  # marketplace counts weighted higher
    elif marketplace:
        for size in all_sizes:
            mp_count = _get_marketplace_count(marketplace, size)
            demand_scores[size] = mp_count
    else:
        # Fall back to listing count as proxy
        for size in all_sizes:
            demand_scores[size] = len(size_prices[size])

    demand_sorted = sorted(demand_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    highest_demand = []
    for size, score in demand_sorted:
        entry = {
            "size": size,
            "demand_score": round(score, 1),
            "avg_price": round(mean(size_prices[size]), 2) if size_prices[size] else None,
            "brand_count": len(size_brands.get(size, set())),
        }
        if trends:
            entry["trends_interest"] = _get_trends_interest(trends, size)
        if marketplace:
            entry["marketplace_listings"] = _get_marketplace_count(marketplace, size)
        highest_demand.append(entry)

    # Best margin: highest average price with reasonable demand
    margin_scores = {}
    for size, prices in size_prices.items():
        if len(prices) < 2:
            continue  # need enough data points
        avg = mean(prices)
        demand = demand_scores.get(size, 1)
        # Margin score: price * sqrt(demand) — high price + decent demand
        margin_scores[size] = avg * (demand ** 0.5) if demand > 0 else avg

    margin_sorted = sorted(margin_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    best_margin = []
    for size, score in margin_sorted:
        best_margin.append({
            "size": size,
            "margin_score": round(score, 1),
            "avg_price": round(mean(size_prices[size]), 2),
            "price_range": f"${min(size_prices[size]):.0f}-${max(size_prices[size]):.0f}",
            "brand_count": len(size_brands.get(size, set())),
        })

    return {
        "highest_demand_sizes": highest_demand,
        "best_margin_sizes": best_margin,
    }


# ---------------------------------------------------------------------------
# Data quality assessment
# ---------------------------------------------------------------------------

def assess_quality(listings: list[dict], trends: dict, marketplace: dict) -> dict:
    """Produce a data quality summary."""
    sources = set()
    sizes = set()
    for listing in listings:
        if listing.get("source"):
            sources.add(listing["source"])
        if listing.get("size"):
            sizes.add(listing["size"])

    # Count how many listings have price data
    priced = sum(1 for l in listings if l.get("price") and l["price"] > 0)
    typed = sum(1 for l in listings if l.get("type"))

    quality = {
        "total_listings": len(listings),
        "listings_with_price": priced,
        "listings_with_type": typed,
        "sizes_scraped": len(sizes),
        "sources_found": sorted(sources),
        "has_trends_data": bool(trends),
        "has_marketplace_data": bool(marketplace),
    }

    if len(listings) < 50:
        quality["confidence"] = "low"
        quality["note"] = "Fewer than 50 listings — rankings may not be representative"
    elif len(listings) < 200:
        quality["confidence"] = "medium"
    else:
        quality["confidence"] = "high"

    return quality


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output(
    top_10: list[dict],
    popularity: dict[str, float],
    coverage: dict[str, dict],
    value: dict[str, dict],
    categories: dict[str, list],
    opportunities: dict,
    quality: dict,
) -> dict:
    """Build the full output JSON structure."""
    # by_demand: top brands sorted by popularity
    by_demand = sorted(
        [{"brand": b, "score": round(s, 1)} for b, s in popularity.items()],
        key=lambda x: x["score"],
        reverse=True,
    )[:15]

    # by_coverage: top brands sorted by coverage
    by_coverage = sorted(
        [{"brand": b, "score": d["score"], "sizes": d["sizes_count"], "retailers": d["retailer_count"]}
         for b, d in coverage.items()],
        key=lambda x: x["score"],
        reverse=True,
    )[:15]

    # by_value_tier
    by_value_tier = {"premium": [], "midrange": [], "budget": []}
    for brand, data in value.items():
        tier = data["tier"]
        if tier in by_value_tier:
            by_value_tier[tier].append({
                "brand": brand,
                "avg_price": data["avg_price"],
                "score": data["score"],
                "consistency": data["consistency"],
            })
    for tier in by_value_tier:
        by_value_tier[tier].sort(key=lambda x: x["score"], reverse=True)
        by_value_tier[tier] = by_value_tier[tier][:10]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_quality": quality,
        "top_10_overall": top_10,
        "by_demand": by_demand,
        "by_coverage": by_coverage,
        "by_value_tier": by_value_tier,
        "by_category": categories,
        "size_opportunities": opportunities,
    }


def print_summary(output: dict):
    """Print human-readable summary to stdout."""
    print()
    print("=" * 55)
    print("  TIRE BRAND RANKINGS FOR JUN")
    print("=" * 55)
    print()

    quality = output["data_quality"]
    conf = quality.get("confidence", "unknown")
    print(f"Data: {quality['total_listings']} listings | "
          f"{quality['sizes_scraped']} sizes | "
          f"Confidence: {conf.upper()}")
    if quality.get("note"):
        print(f"Note: {quality['note']}")
    print(f"Sources: {', '.join(quality.get('sources_found', []))}")
    print()

    # Top 10
    print("TOP 10 BRANDS (Overall):")
    print("-" * 55)
    for entry in output["top_10_overall"]:
        price_str = f"${entry['avg_price']:.0f}" if entry.get("avg_price") else "N/A"
        tier = entry.get("tier", "?").capitalize()
        print(f"  {entry['rank']:>2}. {entry['brand']:<20} "
              f"Score: {entry['composite_score']:>5.1f} | "
              f"Avg: {price_str:>5} | "
              f"{entry['sizes_available']} sizes | "
              f"{tier}")
    print()

    # Demand
    demand = output.get("size_opportunities", {}).get("highest_demand_sizes", [])
    if demand:
        print("HIGHEST DEMAND SIZES:")
        print("-" * 55)
        for i, s in enumerate(demand[:8], 1):
            parts = [f"{s['size']}"]
            if s.get("trends_interest"):
                parts.append(f"Trends: {s['trends_interest']:.0f}")
            if s.get("marketplace_listings"):
                parts.append(f"{s['marketplace_listings']:.0f} marketplace listings")
            if s.get("avg_price"):
                parts.append(f"Avg: ${s['avg_price']:.0f}")
            print(f"  {i}. {' | '.join(parts)}")
        print()

    # Margin
    margin = output.get("size_opportunities", {}).get("best_margin_sizes", [])
    if margin:
        print("BEST MARGIN OPPORTUNITIES:")
        print("-" * 55)
        for i, s in enumerate(margin[:8], 1):
            print(f"  {i}. {s['size']:<14} "
                  f"Avg: ${s['avg_price']:.0f} | "
                  f"Range: {s['price_range']} | "
                  f"{s['brand_count']} brands")
        print()

    # Category leaders
    categories = output.get("by_category", {})
    for cat_key, cat_name in [("all_season", "ALL-SEASON"), ("winter", "WINTER"), ("performance", "PERFORMANCE")]:
        leaders = categories.get(cat_key, [])
        if leaders:
            print(f"TOP {cat_name} BRANDS:")
            for i, entry in enumerate(leaders, 1):
                print(f"  {i}. {entry['brand']} ({entry['listings']} listings)")
            print()

    # Value tiers
    tiers = output.get("by_value_tier", {})
    for tier_key, tier_name in [("premium", "PREMIUM"), ("midrange", "MID-RANGE"), ("budget", "BUDGET")]:
        brands = tiers.get(tier_key, [])
        if brands:
            names = ", ".join(b["brand"] for b in brands[:5])
            price_range = f"${min(b['avg_price'] for b in brands):.0f}-${max(b['avg_price'] for b in brands):.0f}"
            print(f"  {tier_name}: {names} ({price_range})")
    print()
    print("=" * 55)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze scraped tire data and produce brand rankings for Jun."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "market-intelligence"),
        help="Path to market-intelligence directory (default: ../market-intelligence/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for JSON results (default: <data-dir>/tire-brand-rankings.json)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_path = args.output or os.path.join(data_dir, "tire-brand-rankings.json")

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        print("Run the tire scraper first to populate market-intelligence/", file=sys.stderr)
        sys.exit(1)

    # Load data
    listings, marketplace, trends = load_all_data(data_dir)

    if not listings:
        print("Error: no retail listings found. Cannot produce rankings.", file=sys.stderr)
        print("Ensure retail-data.json exists and contains listing data.", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnalyzing {len(listings)} listings...", file=sys.stderr)

    # Compute scores
    popularity = compute_popularity(listings, trends, marketplace)
    coverage = compute_coverage(listings)
    value = compute_value(listings)
    categories = compute_category_leaders(listings)
    top_10 = compute_composite(popularity, coverage, value)
    opportunities = compute_size_opportunities(listings, trends, marketplace)
    quality = assess_quality(listings, trends, marketplace)

    # Build output
    output = build_output(top_10, popularity, coverage, value, categories, opportunities, quality)

    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}", file=sys.stderr)

    # Print human-readable summary
    print_summary(output)


if __name__ == "__main__":
    main()
