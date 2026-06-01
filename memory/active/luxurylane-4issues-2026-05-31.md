# LuxuryLane — Dee's 4 storefront issues + mismatch check (2026-05-31)

## Goal
Diagnose & fix 4 issues Dee reported on the live tire storefront, and verify no photo mismatch.

## Verified diagnosis (all evidence from DB + May-29 scrape, not guessed)

### #1 TITLES — ROOT CAUSE FOUND, code fix STAGED (needs deploy)
- `src/lib/tires.ts` `mapToTire` rendered `model: row.model` — the slug-contaminated field
  (e.g. "scorpion-winter-2", "blizzak-dm-v2", "ziex-ct60-as").
- Clean `display_name` ("Scorpion Winter 2", "Blizzak DM-V2") IS set on all 2,653 new rows.
- FIX APPLIED (local, not deployed): added `display_name`/`tdg_card_model` to interface;
  `model: row.display_name ?? row.tdg_card_model ?? row.model`. Image resolution still keys on raw `row.model`.

### #2 / #3 OUT-OF-STOCK — two compounding causes
- DB availability matches May-29 scrape PERFECTLY by SKU: 3,234/3,234 agree, 0 flips. Data not corrupt.
- BUT: (a) scrape is 2 days stale; Dee checks TDG live → stock churns red↔green.
- (b) The prior activation flipped 2,659 RED rows active. Of 205 active model-groups, 173 are 100% red.
- LOCKED spec: 100%-red never-stocked models → HIDDEN (active=false); red-but-normally-stocked → "Request" button.
  Bare "Out of Stock" badge on red items VIOLATES spec (red = "Request", per Dee's earlier correction).
- Card (`TireCard.tsx` L283) shows "Out of Stock" badge for any !inStock regardless of requestable → eyesore.
- Scrape ratio: 2464 green / 946 blue / 5336 red. Active set is 79% red (inverted) → activation over-selected red.

### #4 IMAGES white bg — confirmed
- 2,369 active rows point to .jpg (Canadian Tire white-bg). 991 are .png (transparent curated). 180 distinct jpg dirs.
- Fix: bg-removal pass (rembg birefnet) on the 180 jpg dirs → transparent png → repoint image_url.
  Check disk for existing png twins first (cheap repoint) before re-rendering.

### MISMATCH — CLEAN ✅
- 0 true photo mismatches among 3,360 active rows. 37 flagged are all legit model-code aliases
  (Hankook H436=Kinergy GT, H452=Ventus, Firestone FR710=all-season). Photo belongs to the tire.

## *** CORRECTION + RESOLUTION (2026-05-31 PM) ***

### #2/#3 ROOT CAUSE WAS WRONG — it's NOT staleness. Dee was right.
- My "3,234/3,234 DB matches scrape" claim was MISLEADING. The DB faithfully mirrors the scrape
  by SKU (0 disagree), but that's the trap: TDG lists MULTIPLE SKU variants per (model,size) —
  a base SKU that's green/in-stock (e.g. BS-001145) PLUS aged/alt SKUs that are red
  (BS-008310, BS-001145AGED-22). Ingest collapsed each size to ONE SKU and systematically kept
  a RED variant even when an in-stock one existed for the same size.
- BLAST RADIUS (proven): 831 active rows falsely red (in-stock variant exists same model+size).
  134 active models showed 100%-OOS while actually having in-stock sizes. = Dee's exact complaint.
- FIX APPLIED: for each active red row, set availability to BEST variant (green>blue>red) across
  scrape siblings. 831 flipped (607→green, 224→blue). Rollback: data/pricing/ROLLBACK-availability-*.json
- Card reads `availability` (tires.ts:187-189), group badge = sizes.some(inStock) (L514) → fix renders.
- CAVEAT: `in_stock` is a GENERATED column (from stock_qty, NOT availability) → can't update directly.
  Main catalog uses availability so it's fine; secondary featured/related queries (.eq in_stock true,
  L402/453) still under-count. TODO: sync stock_qty or change those queries to availability.
- CAVEAT: prices for the 831 were computed from the RED variant's dealer_cost. In-stock variant cost
  may differ → revisit price basis for these in CRM review. Availability-only fix did NOT touch prices.

### #1 TITLES — FIXED + DEPLOYED. commit 78d659d (display_name ?? tdg_card_model ?? model).
### #4 IMAGES — 1,793 active .jpg rows repointed to existing transparent .png twins.
  Only 37 dirs still need bg-removal pass. Rollback: data/pricing/ROLLBACK-imgrepoint-*.json
### MISMATCH — RE-CONFIRMED CLEAN post-fix: 0 null price / 0 missing img / 0 missing file / 0 size-slug mismatch.

## Remaining
- 37 dirs need real bg-removal (rembg) → transparent png → repoint.
- Activation: still 1,828 red rows active showing OOS. Spec says red→"Request" not bare OOS badge (UI work).
- Revisit price basis on the 831 availability-corrected rows.
- Sync `in_stock`/stock_qty for featured/related queries.

## Files
- scrape: `data/scrapes/tdg-tires-canonical-2026-05-29.json`
- rollbacks: `data/pricing/ROLLBACK-availability-*.json`, `ROLLBACK-imgrepoint-*.json`
- title fix: `src/lib/tires.ts` (deployed)

## *** X-ICE SNOW PROOF — VERIFIED CLEAN (2026-05-31 evening) ***
Dee asked: fix X-Ice photo (white bg), verify pricing vs local data, THEN scale to all models.
Findings after full verification (NOT guessed — DB + live blackcircles + live screenshot):
- PHOTO: `/images/tires/michelin-x-ice-snow/angle-1.png` is GENUINELY transparent (37.7% transparent
  px, all corners RGBA 0,0,0,0). Live prod serves byte-identical PNG. Live screenshot = clean tire on
  neutral card, NO white box. X-Ice needed NO photo fix. Dee's white-bg complaint is REAL but on OTHER
  models (the 37 dirs below), not X-Ice.
- PRICING: 86 active rows. Internal math 0 errors (every customer_price exactly matches locked band
  given cost+market). under_market=true correct on all 3 cap30 rows. 0 null market prices.
  Validated vs LIVE blackcircles (scrapling): **0 of 62 sizes mismatch**. Margins correct.
  TRAP I hit + corrected: blackcircles lists PRICE-THEN-SIZE (`$162.11\n195/65 R15`). My first audit
  parser took price-AFTER-size → false "56 mismatches". The EXISTING engine grabs price-BEFORE-size =
  CORRECT. Verified raw structure before touching data. DO NOT "fix" the engine's price association.
- basis dist (X-Ice active): 82 floor15, 1 match-market, 3 cap30. Most floor because Michelin dealer
  cost > blackcircles retail → floored at +15% (spec-correct, but means we're priced above blackcircles
  on those — competitiveness note for Dee/CRM, not a bug).

## *** SCALE JOB — measured, in progress ***
### PHOTOS (the real white-bg issue): 37 distinct dirs / 576 ACTIVE rows still on white-bg .jpg.
  ZERO have png twins on disk (cheap repoints already done last session) → ALL 37 need real bg-removal.
  Tool: `scripts/fast-bg-remove.py --model <slug>` (birefnet, 800px inference). List saved:
  `data/pricing/active-jpg-scan.json`. Flagships incl: michelin-crossclimate-2, michelin-pilot-sport-
  all-season-4, falken-wildpeak-a-t3w, bridgestone-blizzak-icepeak, toyo-open-country-at-iii, etc.
  Pass: bg-remove 37 → verify transparent → repoint 576 rows .jpg→.png (snapshot first) → commit+deploy.
### PRICING: engine verified correct on X-Ice. Need to (a) spot-audit market_price across all active
  models vs blackcircles, (b) revisit the 831 availability-corrected rows (priced off RED variant cost).
### Inspect scripts created: scripts/xice-inspect.mjs, xice-price-validate.mjs, xice-price-audit2.py
  (correct), xice-blackcircles-spotcheck.py, scan-active-jpg-photos.mjs, shot.py.
  NOTE: scripts/xice-price-audit.py has REVERSED price assoc (wrong) — use audit2.py.

## UPDATE (go-fast pass, session continued)
- APPLIED sku-reconcile-dryrun.mjs --apply: 1006 updates (238 titles null->true TDG name, 5 red->in-stock sibling recoveries, 763 availability truth-aligns to canonical scrape). 0 err.
- CONFIRMED data layer already correct: lib/tires.ts L188-189 red->requestable (Request UI), green/blue->inStock. Active rows now 0 null availability => "Out of Stock" wall gone; reds render existing "Request" button.
- APPLIED activate-instock-parked.mjs --apply: 1231 rows / 122 models activated (gated: in-stock + price + sku-in-canonical + photo-on-disk + photo-matches-model, all git-tracked/deployed). Store ~6214 -> ~7445 active.
- Photos: bg batch (batch-bg-tail.py) still running ~14 dirs for the 16 white-bg active dirs (61 rows). Need: commit transparent PNGs + repoint 61 jpg->png + deploy.
- Photo-SKU mismatch reconcile: 0 true mismatches (photos clean).
- TODO next: finish batch -> commit+repoint+deploy photos -> verify live -> then scale to remaining parked models.

## VERIFIED MILESTONE (go-fast pass complete, photos pending)
- Margins: 7313 in-band [15-30%], 0 belowFloor, 0 aboveCap, 0 belowCost, 132 curated untouched. reprice-out-of-band.mjs fixed 130 violators caused by variant-recovery cost swaps.
- Stock recovery: 311 red->in-stock sibling SKU swaps (variant recovery across 2 reconcile passes).
- Activation: store 6214->7445 active / 328 models.
- Live verified (topaz alias, cache-busted): "Out of Stock" count=0, titles clean (no slugs).
- Apex loads on Dee's phone (my machine caches stale apex IP, irrelevant).
- PENDING: bg batch PID 69515 finishing 16 white-bg dirs (68 active jpg rows). Waiter bccshpsaw will notify. Then: commit PNGs + deploy + repoint jpg->png.
- reprice-out-of-band.mjs is NEW reusable script (band reprice, curated-guarded).

## TOYO PHOTO-CONTENT MISMATCH — CAUGHT + FIXED (own-eyes verify; gate missed it)
- The gated activation checks "photo-matches-model" by NAME, but a dir named toyo-open-country-at-iii
  CONTAINED the wrong tire's photo (Extensa HP II, md5 30e9f06). Same wrong photo in toyo-celsius-ii.
  Name-match passed; CONTENT was wrong. Only caught by opening the images.
- Verified-correct dirs (by sidewall): toyo-extensa-as-ii, toyo-extensa-hp-ii, toyo-open-country-a-t-iii
  (dashes, NOT at-iii), toyo-open-country-ht-ii, toyo-proxes-st-iii. toyo-celsius-ii has NO correct photo.
- Repointed 251+19 active rows keyed STRICTLY on display_name/tdg_card_model (model slug is contaminated):
  133 Open Country A/T III(+EV)->a-t-iii; 24 Extensa AS II->extensa-as-ii; 3 HT II->ht-ii; 4 Proxes->proxes-st-iii;
  100 Celsius II->NULL placeholder (logged data/pricing/REPHOTO-QUEUE.json). 0 err.
  Scripts: toyo-decollide.mjs (v1, keyed model-slug, BUGGY-superseded), toyo-decollide2.mjs (v2 display_name, CORRECT).
  Rollbacks: ROLLBACK-toyo-decollide*.json. Wrong files -> data/photo-quarantine/ (reversible).
- Collision audit catalog-wide (collision-audit.mjs): after fix, 10 shared-photo dirs remain = ALL legit
  same-tire aliases (LE3/OWL, X-CV/G057C, 4S/4-S, primacy spelling, alenza RFT, mud-terrain spelling,
  crossclimate spelling, H436 Kinergy, H452 Ventus). 0 real cross-model mismatches left.
- Pricing re-confirmed catalog-wide (price-sanity.mjs, all 7445 active): 0 null, 0 below floor, 0 above cap.
- LESSON: a name-based photo-match gate is NOT sufficient — dir name can lie about file content.
  De-collision/mismatch checks must verify image CONTENT (eyes or hash-vs-known-good), keyed on
  display_name/tdg_card_model, never the contaminated model slug.

## *** MULTI-AGENT WRITE CONFLICT (2026-05-31 ~17:40) — ESCALATED TO DEE ***
- TWO+ agent sessions edited the SAME Supabase DB + repo concurrently this session. Evidence:
  active count grew 3392->6214->7445 between my own consecutive reads; commits 07ee915 + 45cb940
  ("transparent PNGs for all white-bg tires + photo repoints") appeared that I did NOT author;
  memory file co-edited; 4 claude-agent-sdk procs running.
- The PARALLEL pass repoints photos BY NAME (model slug -> dir) with no content check, and it
  REVERTED my verified toyo de-collision: it re-created toyo-celsius-ii/angle-1.png as a GRAY-bg
  Extensa HP II photo (still the WRONG tire, failed transparency) and COMMITTED it (45cb940),
  re-pointing Celsius II rows back to it. I re-applied my display_name-keyed fix; rows flip between
  reads => active write-race. Cannot land a stable fix while both agents run.
- LIVE RISK MITIGATED FOR NOW: Celsius II rows currently image_url=null => tires.ts serves clean
  placeholder, NOT the gray Extensa. Risk returns if parallel pass re-points Celsius again.
- 45cb940/07ee915 committed locally (main ahead 1 of origin); deploys are vercel --prod (NOT git),
  so not necessarily live yet. DID NOT DEPLOY.
- DECISION NEEDED FROM DEE: serialize the work (pause the other agent / one owner for photos) before
  any vercel --prod, else deploy ships an unstable toyo mismatch + a committed gray-bg wrong photo.
- My verified-correct toyo map (for whoever owns the fix): keyed on display_name/tdg_card_model ->
  Open Country A/T III(+EV)=toyo-open-country-a-t-iii; Extensa AS II=toyo-extensa-as-ii;
  Extensa HP II=toyo-extensa-hp-ii; Open Country HT II=toyo-open-country-ht-ii; Proxes ST III=
  toyo-proxes-st-iii; Celsius II=NO clean photo on disk => null+re-photo. Script: toyo-decollide2.mjs.

## *** GO-FAST CONTINUATION (2026-05-31 night) — contamination recurrence + titles ***
- BG batch (batch-bg-tail) finished 13/14 dirs. BUT it re-rendered angle-1.png from LEFTOVER
  contaminated source JPGs in toyo-celsius-ii + toyo-open-country-at-iii (md5 838685949... ==
  extensa-hp-ii.jpg). Unpushed commit 45cb940 had committed those 2 wrong PNGs (2975624 bytes each).
  LESSON REINFORCED: quarantine the SOURCE jpg too, not just the png — batch re-contaminates from jpg.
- FIX: nulled 98 Celsius II rows (rephoto queue, no correct photo exists), repointed 1 stray
  "Extensa AS II" 255/70R16 row from at-iii->toyo-extensa-as-ii (committed/correct). git rm'd the 4
  contaminated files (celsius+at-iii png&jpg), committed a98ed1f, pushed. Rollback: ROLLBACK-toyo-contam-*.json
- DEPLOY-READINESS AUDIT (all 7445 active): 7340 tracked+deployable photos, 105 null placeholders,
  0 missing-on-disk, 0 untracked-404-risk, 0 pointing-to-deleted. CLEAN.
- TITLES polish: 62 rows / 19 names had leading TDG internal codes (H436 Kinergy GT, RF12 Dynapro,
  HA32 Solus 4S, OPA50...). Stripped leading code -> clean name (Kinergy GT, Dynapro AT2 Xtreme,
  Solus 4S, Open Country A50); Kumho HT51 reordered -> "Crugen HT51" (non-lossy). 0 code-prefixed left.
  Live via ISR. Rollback: ROLLBACK-titles-*.json. Genuine model-codes (Blizzak WS90, Turanza T005,
  ExtremeContact DWS06, AltiMax RT45) correctly LEFT as-is.
- Deploy: pushed a98ed1f; vercel --prod building (qq2s7pmnx). Removes contaminated files (hygiene;
  no active row referenced them so no customer-facing change from the deploy itself).
- STATE: 7445 active / 0 OOS-wall / 0 null price / 0 active jpg / 0 code-titles / mismatch clean.
  Remaining: 105 null-photo rows (rephoto queue, mostly Celsius II — need real source photos).

## ORCHESTRATION + DIRECTIVES (2026-05-31 ~19:45)
- DEE DIRECTIVE: missing/junk tire photos are pulled from CANADIAN TIRE. Tooling exists: scripts/scrape-ct-images.py (Playwright search canadiantire.ca → download → birefnet bg-removal → angle-1.png). --model SLUG mode.
- RUNNING PROCESSES (orchestrator audit): 4 claude-agent-sdk procs, ALL cwd=substation (Telegram sessions, NOT deliberate workers). Started 4:47/5:01/5:34/6:04 PM. 1356=substation index.js. NONE hold luxurylane file handles, NONE writing now. Earlier 7445→6600 drift was THIS session's own sequential writes, not a parallel writer. The "another agent writing" theory was WRONG.
- PARKED INVENTORY TRUTH (3,477 rows / 315 models): NOT all OOS.
  - 2,571 red (genuinely out of stock at TDG)
  - 464 green + 144 blue = 608 IN STOCK but parked
  - Why in-stock parked: held by activation gate — 1,200 unpriced, 1,603 need-review, 1,598 no photo, 791 white-bg jpg.
  - So 608 sellable rows are being withheld only because they fail photo/price/review gate, not because OOS.
- ACTIVE missing-photo: 105 rows null image (100 Celsius II + ~7 Yokohama Geolandar X-CV junk) → REPHOTO-QUEUE.json → pull from Canadian Tire.
