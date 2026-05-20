---
# Market-Driven Pricing — Active Thread
**Created:** 2026-05-18
**Status:** IN PROGRESS

## Goal
Build accurate per-size, per-model, per-brand pricing based on LOCAL GTA market averages. 30% markup is the benchmark — adjust up or down based on what local shops actually charge.

## Methodology
1. Pull full inventory from Supabase (brand, model, size, wholesale, current retail)
2. Scrape 8 LOCAL GTA shops for every model/size we carry
3. Average local prices per brand/model/size (never mix sizes or models)
4. Compare our wholesale + 30% vs local average
5. Report: % increase or decrease from 30% for each tire
6. DO NOT APPLY — Dee reviews numbers first

## Local Shops (confirmed by Dee)
1. Canadian Tire (Brampton, Mississauga, Scarborough)
2. TDot Performance (GTA-based)
3. Active Green + Ross (GTA chain)
4. KRAVE Automotive (GTA)
5. Point S (Brampton/Mississauga)
6. Noble Tire (Brampton)
7. Fas-Tire (Scarborough)
8. ZRacing (Mississauga)

## Explicitly REMOVED (not local)
tire.ca, 4tires.ca, wheelsco, pmctire, 1010tires, blackcircles, quattrotires

## Current State
- Database is at flat 30% (applied earlier this session)
- Need local market data to validate/adjust
- Dee wants numbers only, will review before any changes

## Rules
- Per size comparison only (never average across sizes)
- Per model comparison only (never bleed across models)
- Show % increase or decrease from 30% baseline
- 30% is the goal — deviate only with local evidence

## CRITICAL LESSON (May 18 — DO NOT REPEAT)
Previous scrape searched by SIZE and grabbed random brands' prices, attaching them to wrong models.
This corrupted the entire DB. The fix took a full manual rebuild (707 tires, one model at a time).
**NEVER search by size alone.** Always search by exact brand + model, then record only sizes listed under that model.
Cross-brand contamination = data destruction. Verify every entry maps to the correct brand/model before touching DB.
Also: Michelin was MISSED in local market scraping. Dee caught it manually ($260 vs $249 local). ALL brands must be scraped, no exceptions.

## Open Loops
- [ ] Pull full inventory
- [ ] Scrape all 8 shops
- [ ] Build averages
- [ ] Present report to Dee

## Next Actions
Scraping in progress — 8 shops being hit in parallel
