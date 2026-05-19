---
name: LuxuryLane Tires
status: active
created: 2026-05-11
---

# LuxuryLane Tires — Jun's Tire Business

## Goal
Build and launch LuxuryLaneTires.ca — tire sales + installation in Vaughan, ON. Automated sales pipeline.

## Deployment
- Site: https://luxurylanetires.ca/
- Hosting: Vercel under Jun's account (LuxuryLaneTires project)
- Access: Need 1Password for Jun's Vercel credentials
- Wholesale: TDG Access (tdgaccess.ca) — READ ONLY, never order/modify
- Payment: Stripe recommended (fastest Next.js integration)

## Business Model
- Buy wholesale tires, mark up ~$80/tire
- Installation: costs $20/tire, charge $40 → $20 profit/tire
- Per set of 4: ~$400 net (markup + install profit)
- Jun's team gets 20% rev share on sets sold
- Target: $10-15k/month off tires alone

## Deliverables

### TODAY (May 11)
- [x] Scrape tire market data — which brands + sizes sell best (search volume) → DONE, saved to luxurylane-tire-research.md
- [ ] Build LuxuryLaneTires.ca website — Next.js, dark luxury aesthetic, TireBuyer layout clone
  - Reference: tirebuyer.com/tires (design brief at /tmp/tirebuyer-design-brief.md)
  - Deploy to Vercel on temp URL tonight
  - Domain: LuxuryLaneTires.ca (Jun purchasing when ready)
- [ ] Scraper: Kijiji GTA + Canadian Tire/Costco/Walmart retail pricing → price intelligence report

### This Week
- [ ] Price chart: popular brands/sizes vs wholesale pricing
- [ ] Facebook Marketplace bot — post brand+size combos hourly
- [ ] Auto-response bot for marketplace inquiries
- [ ] Booking flow (consultation/appointment setter)

### UX Reference (saved May 12)
- Site design assets saved to `market-intelligence/site-assets/`
- **Flow page**: `flow-icons/full-flow-as-easy-as-123.jpg` — "As Easy as 1, 2, 3" section
  - Step 1: FIND YOUR MATCH — search by vehicle, size, or brand
  - Step 2: FREE DELIVERY — sourced from brand partners, delivered to Vaughan facility
  - Step 3: EASY INSTALLATION — book online, $40/tire mounting + balancing
- **Step icons**: `step1-search.jpg`, `step2-delivery.jpg` (gold accents, dark tire imagery)
- **Top-Rated Tires grid**: `flow-icons/top-rated-tires-grid.jpg` — product cards with category badges (All-Season/All-Terrain), brand logo, model name, size, speed/load specs, star ratings + review counts
- **Google Reviews badges**: 4.6 stars version + 5.0 stars version (dark + light variants)
- **Brand logo strip**: `flow-icons/brand-logo-strip.jpg` — Pirelli, Continental, Ironman, General, Toyo, Falken in colored circles
- **Individual brand logos** (8): Hankook, Bridgestone, Michelin, Falken, Toyo, General Tire, Ironman, Pirelli
- **Nav bar**: Premium Brands | Expert Installation | Free Consultation | TIRES | DEALS | HELP & ADVICE | SERVICES | (905) 555-TIRE | BOOK NOW
- Build this flow once site design is approved by Jun

### Future
- [ ] Facebook/Meta ads once initial clientele established
- [ ] LuxuryLane Detailing (mobile + dealership targeting)
- [ ] Move into bigger unit with hoist (dealership + shop + tire store)

## Strategy Notes
- Brand-specific, size-specific listings (NOT generic "wholesale tires")
- Include "installation available on-site at Vaughan location"
- High-end performance tires (21" 305s for supercars) = good margin
- Seasonal: performance tires now (summer), snow tires in fall (Michelin X-Ice, Yokohama, Bridgestone Blizzak)
- Target dealerships in Vaughan + surrounding areas, not just retail
- Niche hunting > broad ads for launch phase

## Key Details
- Domain: LuxuryLaneTires.ca (not yet purchased)
- Location: Vaughan, ON (unit in a complex, neighbors are car dealers)
- Client: Jun (mentor/client)
- Revenue model: 20% on tire sets for Dee's team
