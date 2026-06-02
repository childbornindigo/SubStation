# LuxuryLane — CT local-market price scrape (above-MSRP comps)

## Goal
Get local GTA market comps (Canadian Tire) for the ~536 above-MSRP rows (455 Michelin + 81 BFG)
so the above-MSRP reprice can re-anchor them below MSRP where CT genuinely undercuts. Free path via
Scrapling (CT's Akamai cleared by StealthyFetcher browser). No Firecrawl credits.

## State (2026-06-01)
- Scraper: scripts/scrape-ct-prices.py (complete, reverse-engineered CT price API). RUNNING bg PID 24920.
- Targets: data/pricing/above-msrp-scrape-targets.json (536 rows, 51 models)
- Checkpoint: data/pricing/ct-price-ckpt.json (resume-safe per-model)
- Output: data/pricing/ct-price-comps.json (121 comps; 73 old-shape from earlier pass + 48 new)
- 45/51 models resolved to CT pcodes, 6 no-pdp.

## KNOWN ISSUE to fix after run
- CrossClimate2 / CrossClimate2 CUV / CrossClimate2 A/W CUV all map to CT pcode 3086342 and matched
  0 sizes despite 112 skus. Likely price API returned no priced skus OR name-size parse miss.
  These are high-volume Michelin = big coverage hole. After run: reset those models in ckpt 'done'
  + re-run, OR diagnose JS_PRICE return for that pcode.

## Michelin manual override (DONE)
- Pilot Sport AS 4 205/50R17 (id 3e4d3c62...) = $249.00 flat, price_basis=manual-override.
  cost 211.41 (+17.8%), MSRP 261. Locked so reprice won't clobber (must skip manual-override).

## Next after scrape completes
1. Merge ct-price-comps.json into scripts/market-data-*.json prices[] (or point reprice at it)
2. Re-run scripts/above-msrp-local-market.py --apply (snapshot first) — MUST exclude price_basis=manual-override
3. Verify: above-MSRP rows with CT comp now <MSRP show % OFF; rest stay "Best Market Price"
4. /tires is ISR revalidate=60 -> live in ~60s, NO deploy needed for DB-only reprice

## COMPLETE (2026-06-01 ~13:15)
- Scrape finished: 384/498 target sizes covered (77%). CrossClimate2 landed 11/23 (early 0 was transient).
- Merged CT comps into scripts/market-data-2026-06-01.json (490 old + 386 CT = 876 prices).
- Pointed above-msrp-local-market.py at merged file; market_source label -> local-gta-2026-06-01.
- APPLIED 137 rows. 49 dropped below MSRP (vs 10 before scrape). Snapshot:
  data/snapshots/above-msrp-localmkt-snap-1780333866.json (reversible).
- Live DB: 5939 active priced | 5433 below MSRP (% OFF badge) | 506 at/above MSRP (Best Market Price).
  Of the 506: only 86 have NO comp at all (no-market->target30) — CT doesn't carry those sizes +
  23 null-tdg_card_model rows. Rest are legit capped (CT/market genuinely above us = correct).
- Michelin Pilot Sport AS 4 205/50R17 = $249 manual-override INTACT (untouched by reprice).
- /tires is ISR revalidate=60 -> live in ~60s. NO deploy needed (DB-only).
- Residual gap (diminishing returns): 86 no-comp rows. Could chase via blackcircles/4tires but CT is
  the free GTA source and these sizes/models it doesn't stock. They correctly show "Best Market Price".
