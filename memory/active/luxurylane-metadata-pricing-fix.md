# LuxuryLane — Metadata / Grouping / $0 / MSRP fix

**Goal:** 4 fixes Dee asked for (2026-06-01):
1. Fix $0 broken sizes
2. Align/format metadata for all tires
3. Highlight "X% off MSRP" when listing under MSRP
4. Fix grouping bug (multiple distinct models mashed into one product page → same size, two prices)

## ROOT CAUSE (verified via DB audit, data/audit_metadata.py + audit_raw.py)
`tdg_card_model` is the ONLY authoritative product name. Both `display_name` AND the `model`
slug got contaminated by TDG **search-labels** on a huge fraction of rows. Example: 65 rows have
`model='extremecontact-sport-02'`, `display='ExtremeContact Sport 02'`, but `card='ExtremeContact
DWS06 PLUS'` — DWS06 tires wearing a Sport-02 slug. Two real tires (two costs) collapse into one
product page → "same size, two prices."

## Audit numbers (7,211 active tires)
- **$0 rows: 51 total, ALL active=False** (parked/red/discontinued, mostly no cost). Don't render to
  customers (size pickers filter active=true). Landmines to clean, not live bugs. A few Radar (hide brand)
  have cost but pr=0.
- **Grouping collisions: 38 groups** with >1 distinct tdg_card_model; **75 groups** same-size-different-cost.
  Real distinct-tire mashups: ExtremeContact DWS06+Sport02, P Zero+P Zero Winter, Celsius II+Extensa AS II,
  UltraWeather+WeatherPeak (27 rows are actually WeatherPeak mislabeled), Defender LTX+Premier LTX,
  Geolandar A/T4+X CV, Falken FK460+FK510, BFG All-Terrain KO2+Trail-Terrain.
- **Contaminated slugs: 1,684** rows where tdg_card_model tokens missing from slug.
- **MSRP: 6,968/7,211 have msl_price. 6,295 show %off (good). 673 price ABOVE MSRP (bad look). 243 no MSRP.**

## Render paths (deploy implications)
- `/tires` grid = ISR revalidate=60 → groups by `display_name` (makeGroupKey). DB fix self-refreshes ~60s.
- `/tires/[slug]` detail = ISR revalidate=60 → size picker = getTireSiblings keyed on raw `model` slug.
  **So fixing display_name alone won't fix the detail size picker — must canonicalize `model` slug too.**
- MSRP "% OFF MSRP" badge ALREADY CODED in TireCard + TireGroupCard (src/components/TireCard.tsx).
  Driven by tire.msrp from msl-prices.ts static map + DB msl_price. No UI build needed.
- image_url WINS over resolveImageUrl(model). Changing `model` only affects image_url-NULL rows. CHECK COUNT.

## PLAN
1. **Canonical name pass (DB-only, reversible, snapshot first):** for every row, set
   `display_name := tdg_card_model` (clean) and canonicalize `model` slug ← tdg_card_model so grid +
   detail group consistently. Image-safe where image_url set. → fixes grouping (task 4) + customer-facing
   metadata (task 2). Self-refreshes via ISR, NO deploy.
2. **$0 rows:** all inactive; null junk / leave hidden so no landmine. Low risk DB.
3. **MSRP:** verify badge coverage (done — works). Optionally cap the 673 over-MSRP rows OR confirm intended.
4. **Photo/image de-collision:** DEFER — needs eyes-on verification per skill (don't bulk-rewrite img paths).

## STATUS (2026-06-01) — APPLIED & LIVE (DB), code fix STAGED (not deployed)
- apply_fix.py --apply ran. Snapshot: data/snapshots/tires-snapshot-1780327731.json (all 10077 rows).
- 7595 names canonicalized (display_name+model := tdg_card_model). 1168 redundant rows deactivated.
- LIVE DB verified: 6043 active | 301 groups | 0 same-size-two-price.
  Falken split SN250 AS / SN250A AS; ExtremeContact -> DWS06 Plus/Sport/Sport 02; P Zero -> 10 variants;
  WeatherPeak/UltraWeather separated. 0 featured slugs deactivated (hard-guarded).
- 1 residual on grid only: makeGroupKey strips '+', merged Proxes Sport AS + AS+. FIXED in code
  (src/lib/tires.ts makeGroupKey: '+'->'plus' before normalize). tsc --noEmit passes. NEEDS DEPLOY.
- Rollback: re-PATCH from snapshot (active + display_name + model per id).
- ISR revalidate=60 -> DB changes already self-refreshed live, no deploy needed for tasks 1/2/3/4-DB.

## 2026-06-01 ROUND 2 (Dee follow-up)
- **makeGroupKey '+'->'plus' deploy:** running via vercel --prod (bg task box5ss8s3). Splits X-Ice Snow
  vs Snow+ on grid. VERIFIED data: X-Ice Snow=104 sizes, X-Ice Snow SUV=35, X-Ice Snow+=16 — 3 distinct
  models, clean card=display=model. => 3 SEPARATE CARDS on grid; click-in shows that model's SIZES only.
- **Q: X-Ice Snow / Snow+ / SUV = multiple cards or one card w/ options?** ANSWER: multiple cards (each
  model own card; getTireSiblings keys on model slug => only sizes inside, not other models). Correct: genuinely diff tires/costs.
- **"Best Market Price" label for above-MSRP:** ALREADY LIVE. TireCard.tsx L156-164: savingsPct>0 =>
  "X% OFF MSRP", else => "✦ Best Market Price". Same in TireGroupCard L334-353. No build needed.
- **Compare above-MSRP vs LOCAL market (Dee directive):** audited 565 at/above MSRP.
  - 333 already market-compared (market >= our price => genuinely best price, honest).
  - 232 = blind +30% (price_basis no-market->target30) w/ NO market check <- the ones to fix.
  - Local GTA data scripts/market-data-2025-05-19.json (PMC,TDot,Noble,ZRacing; 490 prices/26 models).
    Covered 10/233 by exact size -> ALL 10 dropped below MSRP, repriced match-local-market. APPLIED.
    Snapshot data/snapshots/above-msrp-localmkt-snap-1780328466.json. Script scripts/above-msrp-local-market.py.
  - 223 remain (CrossClimate2, Pilot Sport EV/AS4, Trail-Terrain, Primacy Tour, large/LT sizes) NOT in
    May-19 local data -> NEED FRESH local-market scrape (40 model-pages, free via scrapling). PROPOSED to Dee.

## OPEN QUESTIONS FOR DEE
- Run fresh local-market scrape for the 223 remaining above-MSRP rows (40 model-pages)? Confirms their
  pricing vs local GTA market so none sit above MSRP on a blind +30%.

## 2026-06-01 ROUND 3 — "prices not aligned with local data" (DWS06 $405 vs $373.99)
Dee example: Continental ExtremeContact DWS06 PLUS 245/35R20 showed $405; blackcircles $373.99.
ROOT CAUSE (verified): NOT the size normalizer — `norm_size` (re.search, `\D{0,3}` handles ZR/W/Y/LT/P)
was already correct; `245/35 ZR20` canonicalizes to `245/35R20` fine. The real cause was a STALE comp:
the DB had market_price=$427.99 (old blackcircles pass, basis market-above->cap30 => held at cost×1.30
= $404.90). A FRESH blackcircles scrape captured the true $373.99, but the catalog-wide reprice using
those fresh comps had not yet been applied to the DB.
- FRESH BC scrape (scripts/scrape-bc-prices.py, scrapling, free): 311 comps over 368 sizes (311/1902
  target rows; BC doesn't carry every model/size). Output data/pricing/bc-price-comps.json.
- Catalog-wide reprice (scripts/catalog-wide-localmkt-reprice.py) APPLIED against fresh BC + market-data.
  945 rows matched local market; 583 MOVED (396 dropped cheaper, 187 raised toward local mkt within band).
  Snapshot: data/snapshots/catalog-localmkt-snap-1780338171.json (rollback = re-PATCH customer_price/
  market_price/market_source/price_basis per id).
  DWS06 245/35R20 NOW $373.99 (match-local-market, src local-gta-2026-06-01) — matches blackcircles.
- BAND AUDIT (5,951 active priced): 0 below floor, 0 above cap. Manual-override $249 Michelin INTACT.
- COVERAGE: 4,993 rows comp-based basis (84%) | 958 without (941 no-market->target30 + 16 + 1 override).
  Honest gap: blackcircles/CT don't carry every odd/LT/EV size => those stay flat +30% "Best Market Price".
- TDG LEAK fixed: 1,725 active (2,233 total) DB rows had customer-facing feature "TDG Access in-stock —
  ships same/next-day" rendered on /tires/[slug]. Scrubbed -> "In stock — ships same/next-day"
  (scripts/scrub-tdg-features.py, snapshot data/snapshots/tdg-feature-scrub-snap-1780338140.json).
  ALSO scrubbed 6 static src/data/*.ts files (tdg-expansion/new/products/availability/msl + wheels.ts;
  6,328 customer-facing replacements; static fallback render path). wheels DB table = 0 leaks (was
  static-only). tsc --noEmit passes. 0 rendered TDG strings anywhere in src/; 0 TDG leaks across all
  tires+wheels DB rows (only code comments retain "TDG").
- ISR revalidate=60 => DB changes live in ~60s. NO deploy run (DB-only). Static-file scrub needs a deploy
  only if the static fallback is ever hit (DB rows win); flag for next vercel --prod.

## Creds: .env.local.prod (SUPABASE_SERVICE_ROLE_KEY). Repo: /Users/indigochild/luxurylanetires.ca
## Deploy: vercel --prod --scope luxury-lane-tires-ca-s-projects (token .env*.vercel). Show-before-ship.
