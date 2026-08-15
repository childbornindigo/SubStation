# MyPeptide Ad Engine — ACTIVE THREAD

**Goal:** Build a compounding, traceable ad factory inside the existing MyPeptide admin CRM (`~/mypeptide-supplier-dashboard` → `admin.mypeptide.org`), grounded in real sourced videos + live MyPeptide economics. Not a compliance box — a learning flywheel: hopper/intake → chests/libraries → crafting bench → RUO/quality judge → dispenser → numbers gate → Whop/Meta rail, with winners re-fed and losers retired.

**Repo:** `~/mypeptide-supplier-dashboard` (NOT phonegate-build; stale lane cwd trap).
**Supabase (correct):** `wgcskxcklibuacqvawdy`. **Wrong project:** `kicptsvodbfexmczumid`. Every deploy: grep bundle+Functions — wgcs present, kicpts zero.

## Current State (2026-08-14)

**Phases P1–P4 all committed, build-only (NOT deployed, migrations staged not applied):**
- P1 `d15f21a` — chest tables (avatars, angles, formats, hooks, frameworks, swipe, scripts, ads)
- P2 `8daeb0b` — Judge Panel: server-side RUO hard-gate + soft judges + logged overrides (`/api/ad-engine-judge.js`)
- P3 `151b9d3` — generation packaging (`/api/ad-engine-generate.js`); static in-engine + video hand-off contract
- P4 `a92f675` — Numbers Gate / learning ledger

**Numbers Gate grounded config** (`knowledge/mypeptide-ad-engine/NUMBERS-GATE-GROUNDED.md`):
- King goal = **ROAS 3.0** (default). CPA is derived kill/budget lever: `target_cpa = AOV / target_roas`.
- Live MyPeptide economics: AOV seed **$120 CAD**, gross margin ~80%, net ~68%, breakeven CPA $96 gross / $82 net, **target CPA $40 CAD**.
- **Two budget tiers:**
  - LOW (<$500/day, where MyPeptide launches): manual kill — judge after **2–3× target CPR**; single CBO; winner signal = competitor ad **still running >60 days** (adslibrary.ai has the duration filter raw Meta Ad Library lacks).
  - SCALING (>$500/day, Sam Piliero): mechanical kill — adset spend-limit = 1× target CPA; testing cap 20%; judgment window 7–14 days; scale bands +5-10 / +20-30 / +50% by distance over target.

## Source corpus (all vaulted)
7 X sources + Oliver Merrick IG reel `DbnZB9bSTOG` + Moonlighters scaling video + 2 Sam Piliero videos + 2 AirDropped silent recordings (OCR'd) + Whop/WAP material + reel `Db_VSbgs_uW` ("scale ads on low budget", pulled via public embed route, no auth — gave the low-budget tier).

## The real open holes (undesigned)
1. **Internal component promote/retire thresholds** — external half grounded (competitor >60d = replicate candidate). Internal half (avatar/angle/hook promote on ROAS-beat across ≥3 ads; retire when component's ads collectively spend 1× CPA @ zero conv) is my unconfirmed v1 proposal. THIS is what closes the flywheel. Needs Dee's call: ship v1 now, or wait for ≥30 tagged ads for empirical cut.
2. **Bridge page + subdomain** (`go.mypeptide.org`) — non-scrollable image landing masking Ecwid store from Meta's spider. Gates the dispatch node. Undesigned, 100% in our control, rail-agnostic.
3. **Warmup + 80/20 governance node** — between Dispenser (P3) and Whop dispatch. Not scoped.

## Staged / blocked
- **Deploy** (needs Dee go): apply P1–P4 migrations to wgcs + deploy Marketing tab to admin.mypeptide.org. Post-deploy verify wgcs present/kicpts zero + authenticated Marketing mount.
- **P3→Whop dispatch node** — blocked on Whop/wobb creds + BM/Page/Pixel IDs (Dee-owned).
- **P1 competitor-intel node** — approved to draft: browser provider (adslibrary.ai / Meta Ad Library, page IDs Growth Guys / Peptide Warehouse / Direct Peptides), Kalodata API stubbed behind same seam. Open: batch now or with Whop.

## Guardrails
RUO framing / GLP masking (MP1-S/MP2-T/MP3-R). Never leak Ecwid storefront link in creative. No body/outcome/result claims. No-deploy-without-Dee-go.

## Key files
- `knowledge/mypeptide-ad-engine/NUMBERS-GATE-GROUNDED.md`
- `knowledge/mypeptide-ad-engine/AD-ENGINE-SPEC.md` (fused spec)
- `knowledge/mypeptide-ad-engine/WHOP-PROVISIONING-CHECKLIST.md`
- `knowledge/mypeptide-ad-engine/reel-lowbudget-scaling-GROUNDED.md`
- `~/ObsidianBrain/Knowledge/peptides/2026-05-28-pep-business-summit-day-1.md` (Orange Trail rail source)

## Next decisions from Dee
1. Internal component promote/retire — ship v1 or wait for volume?
2. adslibrary.ai as P1 intel provider — draft the WO now, or bundle with Whop?
3. Deploy go / hold?
