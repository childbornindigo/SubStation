---
name: LuxuryLane — Parked Reintro + Wheel/Photo Quality
status: active
created: 2026-05-30
priority: HIGH
---

# LuxuryLane — Parked Reintro + Quality Pass

Repo: `/Users/indigochild/luxurylanetires.ca` (the .ca one — NEVER touch `/luxurylanetires`)
Deploy: Vercel CLI only. `vercel --prod --scope luxury-lane-tires-ca-s-projects` + vcp_ token (1Password).
DB query helper: `scripts/dbq.py` (persistent — `/tmp` wipes on reboot).

## ROOT CAUSE (settled)
May-29 TDG ingest collapsed multiple real tires onto shared slugs/photo dirs → wrong titles, wrong shared photos, white-bg jpgs, OOS-per-size shown as whole-model OOS, "TDG" customer-facing leaks. NOT degradation of prior work — a contaminating flood on top of the curated 32.

## DEE'S DIRECTIVE (do not re-litigate)
"Parked" ≠ ignore. Parked set IS the job — it's broken inventory that was good, fix to same bar as curated 32, then reactivate. For EVERYTHING (curated + parked + wheels):
- photos correct + background removed properly (transparent PNG, like curated set)
- NO mismatched photos (sidewall must match model)
- prices = 15%floor / match / 30%cap vs LOCAL market data (use the scrape numbers)
- WHEELS: clean stock photos ONLY. No vehicle, no box, no warehouse, no watermark, no crop. If none exists → SVG placeholder, never junk.

## DONE (persisted in DB, survived reboot)
- Curated 32 rolled back live; TDG flood parked (active=false, reversible).
- De-collision naming applied to 8,047 parked rows: clean display_name + own unique slug/photo dir (kills shared-photo mismatch at root). Snapshot reversible.
- "TDG" customer language scrubbed from code (wheel desc/features, tire detail, cart btn).
- 2 live tire mismatches fixed (Firestone Destination LE2, Yokohama Geolandar X-CV G057C) + deployed.
- Wheel junk pass: 72 junk files quarantined → SVG placeholder. Reversible.

## IN FLIGHT (background jobs, detached — survive session, NOT reboot-safe for /tmp)
1. **CT tire photos** (supervisor PID 3276 → scraper 3285): Canadian Tire native search → transparent PNG cutouts → `public/images/tires/{slug}/angle-1.png`. ckpt `data/parked/ct-ckpt.json`. ~30/35 ok. 273 models total. Self-healing wrapper (rembg crashes ~every 28).
2. **Parked pricing** (PID 3493): `web-fallback-final.py --parked --apply --checkpoint data/pricing/parked-fallback-ckpt.json`. 342 model-groups / 6,090 rows. 15/30 band, gated (no fabrication, 15% floor, sanity band), free Scrapling. Skips rows that already have market_price (2,482 done earlier). Slow: each model = many sizes × DDG+retailer fetches. Resumable. Log `data/pricing/parked-full.log`.
3. **Wheel photo sourcing** (supervisor PID 9029 → `source-wheel-images.py`): Bing Images via StealthyFetcher, 3-query/model (white-bg + product + plain), catalog-CDN domain bias, marketplace/on-car penalized. Downloads ≤6 candidates/model → `data/wheels/candidates/`. ckpt `data/wheels/source-ckpt.json`. 55 distinct models (44 + finish variants). NO DB writes here.

## WHEEL PIPELINE (next steps after sourcing completes)
1. `scripts/build-wheel-cand-sheets.py` → per-model contact sheet in `data/wheels/cand-sheets/`.
2. Dispatch parallel vision agents: pick cleanest full-rim STUDIO index per model, or "NONE" (reject box/vehicle/rack/watermark/crop). Can't verify exact spoke design w/o reference → flag low-confidence.
3. Apply: chosen candidate → composite/clean → save to slug dir → repoint DB image_url. "NONE" → keep SVG placeholder.
4. Wheel queue: `data/wheels/rephoto-queue.json` (128 rows: 5 junk files + 107 null-image, 44 models).
5. Wheels are SSG → need `vercel --prod` to show.

## KEY FACTS
- bg-removal model cached at `~/.cache/u2net-local/birefnet-general.onnx` (972MB). Point U2NET_HOME there to avoid download hang.
- DDG image API (i.js) 403s. DDG *html* endpoint 200s. Bing images mediaurl= pattern works via StealthyFetcher.
- Keep-set: 290 keep / 135 cut (Radar 13, red-noise 118, no-sku 4). 273 keep models need verified photo.
- image_url in DB = LOCAL paths, not CDN → files must be deployed to render.
- IMAGE_EXT maps dirs to png/jpg; tires.ts:211 prefers DB image_url over resolver.

## ACTIVATION GATE (per-model, nothing live until all 3 pass)
Photo verified (sidewall-match) + price applied (in-band) + de-collided name/slug → flip active=true → build-verify → deploy on Dee's go.

## NEXT
- Monitor 3 background jobs. When sourcing done → build sheets → vision-gate → apply wheels.
- When CT photos done → sidewall vision-check → activate parked models in batches.
- Hold all deploys for Dee's go (show-before-ship).

## 2026-05-31 — getAllTires 1000-row cap fix
- BUG: `src/lib/tires.ts getAllTires()` used `.range(0,1999)` but Supabase/PostgREST hard-caps every response at 1000 rows → browse `/tires` only ever showed the 1000 cheapest active tires. With 3360 active, 2360 (incl. newly-activated parked set) were invisible to browsing (detail pages worked fine via direct URL).
- FIX: paginate getAllTires in 1000-row chunks until exhausted. Verified: now returns 3360. Commit aa8f164, deployed.
- LESSON: any Supabase query expecting >1000 rows must paginate; `.range()` alone does NOT override the server max-rows cap.
