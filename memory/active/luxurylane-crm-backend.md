# LuxuryLane CRM Backend Prep

## Goal
Prepare the LuxuryLane CRM backend (build-order step 6): extend existing Supabase to surface
backend-only fields (cost, SKU, margin, under_market, needs_review). North star = Jun's private
TDG mirror + customer-price calculator: type a size → see TDG truth (cost/SKU/availability/branch
stock) + customer's computed price + margin.

## Current State (2026-05-31)
- **Schema already live & applied** in Supabase: customers, leads, interactions, orders, order_items,
  team_members (1 row). Defined in `supabase/schema.sql`. All EMPTY except team_members.
- No `quotes` table.
- TIRES: 10,077 total / 3,392 active / 1,221 under_market / 2,534 needs_review.
- Real TDG-mirror columns on `tires`: wholesale_cost, tdg_sku, availability(green/blue/red), branch,
  branch_stock, season, msl_price, market_price, market_source, market_checked_at, under_market,
  customer_price, price_basis, needs_review, display_name, tdg_card_model, search_model, storefront_visible.
- Same market fields on `wheels`.
- **STALE**: `src/app/api/price-inquiry/route.ts` reads OLD dead columns (price_wholesale/price_retail/
  in_stock/stock_qty) + estimateWholesaleFromMSL. Needs replacing with real-column lookup.
- No /crm or /admin route. No middleware/auth gate yet.

## Decisions Made
- CRM lives in same Next.js app (same Supabase, deploys with site) — most efficient.
- Build the internal inventory-intelligence lib first (pure server-side, zero fork, zero risk).

## Open Loops / Forks for Dee
- AUTH model for CRM (exposes wholesale cost/SKU/margin — must NOT be public):
  shared password (matches Dee's dashboard infra) vs Supabase auth vs Vercel deployment protection.
- Build lib + API route locally; DO NOT deploy until Dee reviews (show-before-ship, live client site).

## DONE (2026-05-31, verified end-to-end on live DB, NOT deployed)
- `src/lib/crm/inventory.ts` — internal intelligence layer on REAL columns. lookupInventory(),
  underMarketOpportunities(), needsReviewQueue(), inventoryStats(). Computes margin $/% + markup %.
  Server-side only (exposes cost/SKU). tsc + eslint clean.
- `src/lib/crm/auth.ts` — shared-secret gate (CRM_API_TOKEN env, Bearer or x-crm-token header).
  FAILS CLOSED (no token env = 503). timing-safe compare.
- `src/app/api/crm/lookup/route.ts` (GET+POST) and `src/app/api/crm/queues/route.ts`
  (type=stats|under_market|needs_review). Both auth-gated.
- VERIFIED: 401 no-token & wrong-token; lookup returns cost/SKU/margin/basis correctly; stats =
  10077/3392 active/6685 parked/1221 under_market/2534 needs_review/1200 unpriced; raise-price queue works.
- Files UNTRACKED (not committed, not deployed). Test token removed from .env.local; dev server killed.

## OPEN — needs Dee's call before next step
1. AUTH model for production: shared secret (built, default) vs Supabase auth vs Vercel protection.
   For deploy, CRM_API_TOKEN must be set in Vercel env.
2. CRM UI: build the /crm dashboard pages (lookup box, queues, lead capture) — UI is step after backend.
3. Lead/customer/interaction write helpers (tables live+empty) — build when CRM UI is scoped.

## Why It Matters
This is the calculator Jun actually uses day-to-day. Backend correctness gates the CRM UI + bots.

## Why It Matters
This is the calculator Jun actually uses day-to-day. Backend correctness gates the CRM UI + bots.
