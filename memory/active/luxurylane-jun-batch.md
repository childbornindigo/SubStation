---
name: LuxuryLane — Jun's New Batch
status: active
created: 2026-05-29
priority: HIGH
---

# LuxuryLane — Jun's New Feature Batch (May 29)

Repo: `/Users/indigochild/luxurylanetires.ca` (the .ca one — NEVER touch `/luxurylanetires`)
Deploy: Vercel CLI only (no GH auto-deploy). `vercel --prod --scope luxury-lane-tires-ca-s-projects` + vcp_ token from 1Password.

## KEY FINDING (May 29)
Most of Jun's batch was ALREADY BUILT in prior sessions but is sitting unpushed/undeployed.
Verified against live site + git, NOT assumed.

## Jun's 6 asks → actual status
1. **Website TDG tiers (🟢 same/next-day, 🔵 1-2 day limited)** — BUILT, committed (e9350e7). "Next-day" badges already show live on /tires. Colour-tier + 1366 tires commit is UNPUSHED.
2. **More tire options (1366 added)** — BUILT, committed e9350e7. UNPUSHED.
3. **Wheel options** — BUILT (src/app/wheels + WheelCard.tsx) but /wheels returns 404 LIVE = NOT DEPLOYED.
4. **25% e-transfer deposit checkout** — PARTIAL/BUILT: src/app/cart + api/checkout + e-transfer refs in cart/page.tsx & pricing.ts. NEEDS VERIFY it does 25% interac deposit.
5. **Tire CRM** — BUILT: full Vite+React+Supabase app in `crm/` (Dashboard, Customers, Appointments, Interactions, Auth). Committed 770f14c. Deployment status UNKNOWN (not hosted/verified).
6. **Pricing cross-ref bot (labeled C2)** — BUILT: api/price-inquiry/route.ts. Cross-refs Supabase tires, computes margin, formats quote. Feeds A3 checkout, A4 CRM, "Telegram bot, Facebook inbox". NEEDS end-to-end verify + TDG live cross-ref.

## GENUINE GAPS (not built)
- **VIN / window-sticker / carfax bot** — zero matches in repo. Jun's demo ask: VIN in → window sticker + carfax attached + WhatsApp msg (yr/make/model/odometer). NOT STARTED.
- **Vehicle canvassing** — find Ontario 2022-25 American pickups (VIN starts 1, clean carfax, no recalls, photos w/o dealer logos). Not in this repo. NOT STARTED. Jun wants VOLUME.

## Deploy state
- Local **3 commits ahead of origin/main** (unpushed): TDG colour tiers + 1366 tires, vehicle coverage x2.
- /wheels 404 live confirms deploy is stale.
- Shipping these 3 commits = Jun's items 1-3 go live immediately.

## BLOCKERS
- Vercel deploy token (vcp_) in 1Password — needs Dee biometric to deploy.

## Canvassing target spec (from Jun)
- 2022-2023 Ford F150 high trims
- 2022-2025 GMC/Chevy pickups (any trim if clean carfax + good condition; higher trims easier)
- 2022-2025 GMC Yukon, 2022-2025 Chevy Suburban
- 2023 Chevy Silverado 1500 RST <90,000km, $40-42k CAD, clean carfax
- ALL: VIN starts with 1, clean carfax, no open recalls, asking price, photos w/o dealer logos
- Output: list of possibles OR reach out directly. Heavy volume expected.

## Next actions (pending Dee priority)
1. Push + deploy 3 pending commits (needs Vercel token) → items 1-3 live
2. Verify deposit checkout (25% e-transfer) + pricing bot end-to-end
3. Decide CRM hosting
4. BUILD: VIN bot demo (genuine gap)
5. BUILD: canvassing system (genuine gap, budget-heavy)

## LOCKED SCOPE — Dee directive May 29 (current priority)
Order: (A) website+backend correct & complete → (B) CRM → (C) bots last (when Jun pays).
Clarification: Green/Blue are TDG's OWN delivery colours (🟢 same/next-day, 🔵 1-2 day limited),
context for sourcing — NOT website tier badges. Do NOT build tier UI.

Backend data tasks — STATUS as of May 29 evening (verified against working tree):
- **Blue availability gap**: ✅ DONE. tdg-availability.ts now has 1096 GREEN + 296 BLUE across current models (uncommitted in working tree). Blue backfill ran 14:30 May 29.
- **Double the catalog**: ✅ DONE + BUILD-VERIFIED. Ran `scrape-tdg-colours.py --models all` (63 searched, 53 returned) → /tmp/tdg-colours-all.json (3804 entries, 2766 green + 1038 blue). Applied via apply-tdg-colours.py → tdg-availability.ts (2030 green + 589 blue) + tdg-new-tires.ts (52 models, 2617 entries). `npm run build` EXIT_CODE=0, no type errors (the doubled size was fine — earlier "union too complex" was a stale mid-write read). UNCOMMITTED, deploy-ready. 26→52 models.
- **Wheels from TDG**: 🔬 PROBED. TDG DOES carry wheels but NOT the glam brands in current fake wheels.ts (RTX/Fuel/Method/Enkei = 0 results on TDG). TDG real wheel stock = STEEL (DTD, YKW) + basic direct-fit alloys (Truk Wheels, Macpek) + few real alloys (Robert Thibert ~$305 dealer). Wheel table schema (different from tires): `PCD | DIA | Width | Offset | CB | Stock | MSL | Price` — detect wheel-card by presence of PCD/Offset/DIA headers. Probe scripts: scripts/tdg-wheel-probe.py + /tmp/wheelbrands.py (login timed out on 2nd run — TDG may rate-limit rapid re-logins; space them out). AWAITING DEE DECISION: (A) scrape TDG-real wheels only [honest, less flashy] vs (B) add a styled-alloy distributor (Fast/RSSW/DAI) for premium look [needs new login/source].
- **E-transfer recipient**: ✅ DONE & VERIFIED — cart/page.tsx line 12-13 defaults to sales@luxurylanetires.ca, no .env override present.
- RULES: TDG is READ-ONLY (never order/modify). EXACT brand+model+size match for MSL pricing
  (mismatched pricing caused a major setback before — mark missing rather than guess).
- Scraper to reuse/extend: scripts/scrape-tdg-colours.py (already parses green/blue/red).
  Creds hardcoded in script + also in Keychain (svc tdg-access).

CRM note: full Vite+React+Supabase app already exists in crm/ — CRM phase = ADD to existing Supabase,
not greenfield.

## FULL RE-SCRAPE COMPLETE — May 29 15:38 (the real data foundation)
Dee's expanded spec (May 29 PM) requires: REAL dealer cost (not retail÷1.30), REAL SKU (not "TDG-{slug}"),
green/blue/RED per-size, correct seasons, blackcircles market pass, NO fake data. Prior data lacked all of this.

- **scrape-tdg-tires-full.py RAN** → 11,212 records / 60 models. green=3226 blue=1216 red=6770.
  - Fields: brand, model, size, sku, availability(green/blue/red), msl_price, dealer_price(OUR COST), stock_text, variant_id, season.
  - Sample: Michelin Defender LTX M/S P225/65R17 → sku MI-08218, dealer_price 205.74, msl 254, green, all-season.
  - null_sku=0, null_season=0, missing dealer_price=55 (0.5%). Seasons: all-season 5124, winter 2422, all-terrain 1615, summer 1501, all-weather 550.
  - PERSISTED to repo (out of volatile /tmp): data/scrapes/tdg-tires-full-2026-05-29.json
- **Wheels**: 674 variants (162 green / 512 blue), 13 brands, dealer_price + part_no(SKU) captured.
  PERSISTED: data/scrapes/tdg-wheels-2026-05-29.json
- **CRITICAL — old ingest faked data**: land-tdg-tires.mjs line 191 `dealer = price/1.30` (fabricated cost),
  line 196 `sku = TDG-{slug}` (fake SKU), and silently dropped RED. Full re-scrape replaces all three with real values.

### Build sequence (LOCKED, in order):
1. ✅ Tire re-scrape (real cost/SKU/red/season) — DONE, persisted
2. ✅ Wheel scrape — DONE, persisted
3. ⏭ Supabase schema: add dealer_cost, real sku, availability(green/blue/red), season, branch,
   + market fields (market_price, market_source, market_checked_at, under_market flag, customer_price). Tires & wheels.
4. ⏭ Ingest both — EXACT brand+model+size match (cardinal rule, mark missing, never size-only guess)
5. ⏭ Images: TDG wheel photos + Canadian Tire tire photos (exact model)
6. ⏭ blackcircles per-size market pass + pricing engine (cost+30% target, cost+15% HARD FLOOR,
   market decides inside band; if blackcircles ABOVE 30% → cap at 30% but FLAG under_market in backend for CRM
   one-click raise; no match → search broader for real comparable, only truly-none → flat 30% + manual-review flag).
   THE EXPENSIVE ONE — flag Dee before burning budget. Do it once, correctly, zero hallucination.
7. ⏭ UI: REMOVE green/blue "same-day/1-2day" badges (Dee: internal-only, not customer-facing);
   red size → "Request" button (per-size, not whole model); real wheel catalog; "Winter Approved" wheel badge (TDG marks it).
8. ⏭ Season-label fix across board (e.g. Pirelli P Zero All Season was tagged summer)
9. ⏭ Deploy (show-before-ship — Dee approves before prod)
10. ⏭ CRM (add to existing crm/ app; surface dealer cost + SKU + margin + under_market)
11. ⏭ Bots last (when Jun pays)

NORTH STAR: website becomes private TDG mirror + customer-price calculator. Search tire size → see what TDG
shows but with customer's final price pre-computed. Backend holds EVERYTHING TDG has; site shows only customer-facing.

## DATA INTEGRITY AUDIT — May 29 16:14 (Dee: "real data only, no fabrication")
A fabricated backfill happened and was CAUGHT + quarantined: tdg-tires-FABRICATED-backfill.quarantine.json
  - It invented 291 SKUs for SKU-less rows; smell = duplicate SKUs across different SIZES (MI-68361 on 2 sizes). DO NOT USE.
Fix = clean full RE-SCRAPE (not backfill). RAN scrape-tdg-tires-full.py → tdg-tires-rescrape-2026-05-29.json.
PROMOTED to canonical: data/scrapes/tdg-tires-full-2026-05-29.json (byte-identical to rescrape; prior archived as *.PRE-RESCRAPE.archive.json).

VERIFIED CLEAN (the canonical file):
  - 11,212 records / 60 models / 13 brands
  - Real SKUs: 10,856 (+61 vs prior). SKU-less: 356 — ALL have variant_id (traceable), 352 have real dealer cost.
    316 of the 356 are RED (superseded/out-of-stock on TDG). Honestly left null, NOT faked. Ingest tags them "superseded — no traceable SKU".
  - **0 SKUs span >1 size** ← definitive no-fabrication test PASSED (quarantine fake failed this).
  - 0 null seasons. dealer_price range $57.70–$1,615.35, all >0. MSL ≥ dealer cost on 100% of priced rows (margin exists everywhere).
  - tdg_card_model captured per row (authoritative). 1817 "dup SKU" = harmless search-LABEL artifact, resolved by card model.
    263 SKUs under >1 card sub-line name (e.g. Defender2 / Defender2 CUV / Defender2 H) = real TDG catalog overlap, not fabrication.
INGEST: use scripts/ingest-tires-full.py (reads REAL dealer_price + sku, customer_price NULL until pricing phase).
  NEVER use land-tdg-tires.mjs (fabricates: dealer=price/1.30, sku=TDG-{slug}).
DEDUP RULE for ingest: SKU is the unique key; use tdg_card_model as authoritative model name, not search label.

## STEP 3-4 DONE + LIVE — Canonical tire ingest (2026-05-29 ~16:4x)
- Backup: data/scrapes/tires-backup-PRE-CANONICAL-INGEST-2026-05-29.json (5,794 rows, recoverable)
- Schema added (additive): storefront_visible, display_name, tdg_card_model, search_model, sku_note, season_confidence, branch_stock
- Ingest script: scripts/ingest-tires-canonical.py (DRY-RUN default; --live to write). Keys on real tdg_sku, never overwrites unique `sku` col, COALESCE season, mirrors active=storefront_visible.
- Ran --live: 4,463 enriched + 4,283 inserted = DB now 10,077 tire rows.
- Visibility (active=visible mirror so existing site query works w/o deploy):
  - VISIBLE add-to-cart (green/blue): 3,288 ; VISIBLE red->Request: 4,287
  - HIDDEN: Radar 356 (budget), discontinued-no-sku 164, never-stocked 651
- Verified: Radar visible=0 ✓, all-weather preserved (649) ✓, cost 95% / tdg_sku 92%.
- all-weather kept DISTINCT (516 canonical). Green/blue badges = INTERNAL only (Dee: remove from customer view in UI pass).

### LOOSE ENDS / FLAGS (report to Dee)
1. 280 distinct ACTIVE models now visible (vs Dee "~64" expectation). Honest result of his rule (show in-stock + requestable). May want tighter customer curation vs full backend mirror.
2. 1,346 ORPHAN rows (tdg_card_model NULL) = real-SKU tires from PRIOR scrape not in this canonical search set (e.g. Michelin Pilot Sport AS 4, Primacy Tour AS — legit popular). 548 good green/blue, 460 NULL availability (stale). NOT deactivated (real inventory). Fix = re-scrape those ~60 orphan models to fold into canonical.
3. PRICING NOT YET DONE: customer_price still NULL everywhere. New 4,283 rows live at price_retail=MSL (MSL>=cost always, so never underwater, ~cost+25-30% band). Precise 30%-markup/15%-floor/blackcircles pass is the NEXT step (Dee-approved, expensive).

## ORPHAN CLEANUP — DONE (2026-05-29) → BACKEND = VERIFIED TRUTH
Investigated the 1,346 orphan rows (tdg_card_model NULL, from pre-canonical scrape). Findings:
- 55 of 60 orphan MODELS already exist in the canonical verified scrape (stale duplicates).
- 763 distinct orphan SKUs → 762 have a live canonical DB counterpart (same tdg_sku). Only unique = `BS-012058Aged` (aged-stock variant, not real new inventory).
- 483 orphan rows had NO sku/cost/availability = empty placeholder shells.
- The 5 "missing" models (Turanza Everdrive, CrossContact RX, P Zero AS Plus 3, FR710, UltraWeather) = all empty null shells, no real data.
- CONCLUSION: zero real inventory unique to orphans. All are stale dups or empty shells.
ACTION: Snapshotted all 1,346 (data/scrapes/orphan-deactivation-snapshot-2026-05-29.json, reversible) →
  deactivated (active=false, storefront_visible=false, sku_note tagged). NOT deleted.

### VERIFIED-TRUTH STATE (post-cleanup)
- Total rows 10,077; ACTIVE 7,561 (all from canonical verified scrape).
- Active by availability: green 2,346 · blue 930 · red 4,285.
- Active integrity: 0 missing SKU, 0 missing availability, 0 missing season.
- PRICEABLE set (active green/blue) = **3,276 rows, 100% costed, 100% SKU.**
  Cost range $85.53–$1,124.13, 0 ≤0, 0 inverted (MSL≥cost everywhere → margin exists on every priceable row).
- 22 active no-cost rows = ALL red "Request" items (TDG shows no price OOS; none needed). Spec-correct.
- 267 active distinct models, 12 brands. Radar (budget) active=0 ✓ (hidden).
BACKEND IS NOW VERIFIED TRUTH. Cleared for pricing phase.

### NEXT (sequence)
4. ⏭ PRICING ENGINE (NOW) — blackcircles per-size on the 3,276 priceable rows:
   cost+30% target / cost+15% HARD FLOOR / cap@30% (if BC above 30%, hold 30% but set under_market=true + market_price for CRM one-click raise);
   no BC match → search broader for real comparable; truly-none → flat 30% + manual-review flag.
   DETERMINISTIC scraper (no LLM, zero hallucination). Dedup by distinct (model,size) to cut lookups.
   Red rows = no price ("Request"). FLAG DEE before full-scale run (budget) — validate on sample first.
5. UI pass: remove green/blue badges, red->Request button, season-label display fix, winter-approved badge (wheels), display_name, storefront query -> storefront_visible.
6. Images: TDG wheel photos + Canadian Tire tire photos.
7. Wheels ingest (674 variants ready in data/scrapes/tdg-wheels-2026-05-29.json).
8. CRM. 9. Bots last.

---
## PRICING PHASE — APPLIED + LIVE-VERIFIED (2026-05-29 ~20:10)
Resolved the matcher loop. The saved full-pass scrape was clean; earlier "unresolved/ambiguous" counts were a looser prior apply attempt, not this file.

Source: `data/pricing/pricing-review-full-20260530-000529.json`
Applier: `scripts/apply-pricing-from-review.py --apply` (integrity gate: aborts on floor-violation/null/implausible market; idempotent)

**Live DB state (verified via REST count):**
- 3,276 priceable (green/blue) rows now have customer_price — 100%
- 2,269 from REAL blackcircles market data (floor15:1280 · match-market:760 · cap30:229)
- 1,007 flagged needs_review=true — no market match for that exact size → flat cost×1.30 (above floor, NOT fabricated; market_price=null)
- 229 under_market=true (blackcircles above our 30% cap → CRM "raise price" opportunity)
- 0 floor violations live
- Caught + fixed 2 garbage scrape prices ($9063.33 parse error, Pilot Sport AS 4) → dropped bad market, flat-30 + review

**Schema added:** `needs_review` (bool), `price_basis` (text) — persists which rows need the web-search refinement + feeds CRM.

**REMAINING (needs Dee's go — budget spend):** the 1,007 needs_review sizes should get the Google/web-search fallback per Dee's spec (search broader when blackcircles has no match). They're live-safe at flat-30% interim + flagged. Running 1,007 per-size web lookups = real Firecrawl/search spend.

## WEB FALLBACK v3 — built + running (2026-05-29 ~21:55)
Firecrawl UPGRADED (4,998 credits, billing 05-30→06-30). Original web-fallback-pricing.py was broken:
- looked for price AFTER size + bare `$` → matched review-image captions, 0 real matches.
v3 (`scripts/web-fallback-pricing-v3.py`) reuses pricing-engine-v2's PROVEN parsers:
- pass-1: per-size search → blackcircles buy-table (`parse_bc_prices`, size-keyed, price-before-size, `CA $` only)
- pass-2: per-size search → exact-size PDP, size-anchored `parse_open_for_size` (price tied to exact size token, ≤400 chars, sanity band 0.5x–4x cost). size-in-URL PDPs ranked first.
CAUGHT + FIXED a real bug: parse_open_lowest bled ONE category-page price ($228.57 kaltire) across 7 KO2 sizes. Replaced with size-anchored extraction → DUP-across-sizes check = none.
Verified: KO2 235/55R19 genuinely has NO per-size Canadian retailer page → stays flat-30%+review (honest, not fabricated). Low coverage on long-tail TDG sizes is REAL absence, not a bug.
Full pass running --apply (1000 review rows / 76 models). Integrity gate aborts if any price < 15% floor. Only matched rows change; unmatched stay target30+needs_review (unchanged from current DB).

---
## WEB FALLBACK PASS — LAUNCHED 2026-05-29 21:59 (Firecrawl upgraded by Dee)
- Firecrawl credits restored & verified (search returns real results).
- Running `scripts/web-fallback-pricing-v3.py --apply` detached (PID 43683), log /tmp/v3-fullapply-20260529-2159.log
- Scope: 1,000 needs_review rows / 76 models (the sizes blackcircles model pages lacked).
- Verified BEFORE launch:
  - BC model pages now render size links but NO inline CA$ prices (load on interaction) → pass-1 mostly dead; pass-2 carries.
  - Pass-2 (open-market per-size) CONFIRMED yielding real prices: tcwcanada $214.74 for Blizzak WS90 225/65R17, verified + sanity-banded.
- Integrity: sanity band 0.5x–4x cost, model-identity verify, floor-15% abort gate. No real comparable → stays target30 + needs_review (never fabricated).
- The 2 Pilot Sport AS 4 $9,063 parse-error rows are in-scope (re-searched this pass).
- ZERO Claude tokens (Firecrawl + Supabase REST only).
- NEXT: read log when done, verify applied counts in DB, report match rate.

## 2026-05-29 23:xx — Web-fallback pricing run + ZOMBIE PROCESS INCIDENT
- Firecrawl UPGRADED + credits live (verified via /v1/search 200).
- Web-fallback (`scripts/web-fallback-pricing.py --per-size`) targets the 1007 needs_review sizes. Deterministic, Firecrawl+Supabase only, ZERO Claude tokens.
- Pipeline re-verified live: Michelin Pilot Sport 4S 245/40R18 → blackcircles scrape, model verified, 131 sizes extracted w/ real prices. Engine sound.
- **INCIDENT:** system hit per-uid process ceiling (kern.maxprocperuid=2666). ~2,395 ZOMBIE node procs = `.claude/hooks/gsd-check-update-worker.js` (GSD SessionStart update-check), all PPID 1, accumulated over many Claude Code sessions, never exited → cascading fork-starvation that KILLED the first fallback run (PID 45872 died at 20/993).
- FIX: `pkill -9 -f gsd-check-update-worker` → procs 2875→490. System healthy.
- Cleared the checkpoint written under starvation (20 entries, suspect) + relaunched clean (PID 46213). Verified early no-matches are GENUINE rare sizes (Blizzak WS90 235/50R20 etc.), not starvation artifacts.
- DURABLE FIX STILL OPEN: gsd update-check hook leaks zombies on every Claude Code session. Recommend a reaper (launchd killing gsd-check-update-worker older than 5min) OR disable the SessionStart update check. Awaiting Dee's call — did NOT modify his managed GSD hook files unilaterally.
- Next: run completes (~993 lookups) → DRY-RUN apply → review match stats → live apply with integrity gate (15% floor + 0.5-4x sane band). NO blind write.

## Web-fallback pricing — RUNNING (2026-05-29 ~23:00)
- Firecrawl upgraded by Dee, credits live (verified).
- Hardened web-fallback-pricing.py: broader trusted CA retailers (tiresandco, canadawheels, tcwcanada, 4tires, wheelsco, tiresourcecanada+), broader price formats ($308 / CAD $571.64 / 378.00 CAD), broader size tokens (235/65 R17, LT245/65R17), canonical norm_size. Sane band 0.5-4x cost + exact-size proximity = zero fabrication.
- Running FULL per-size pass: 993 distinct (model,size) keys, --apply (only patches rows with a REAL exact-size market match; no-match stays target30+needs_review). Checkpointed/resumable at data/pricing/per-size-checkpoint.json. Firecrawl+Supabase only, ZERO Claude tokens.
- Expect PARTIAL real-match coverage: the 1007 are the long tail (sizes blackcircles didn't list). First 20 (KO2 + WS90 large CUV/LT sizes) = honest no-match (genuinely rare sizes). Rest of 77 models span more common sizes -> better yield.
- Log: logs/persize-FULL-20260529-225925.log
