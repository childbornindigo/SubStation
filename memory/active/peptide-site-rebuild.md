# Peptide Site Rebuild — Medvi Layout

## Goal
Rebuild mypeptide.org landing page to match Medvi (home.medvi.org) visual layout. Keep existing peptide detail pages.

## Plan (confirmed with Dee 2026-05-11)

### What to KEEP
- All 34 peptide detail pages (CategoryPage, PeptidePage)
- React 19 + Vite + Tailwind + GSAP + Lenis stack
- Routing structure (/peptides/:category/:peptide)

### Landing Page Redesign
- **Method**: Fetch Medvi's actual HTML/CSS as skeleton, reskin with mypeptide brand/content
- NOT eyeballing from screenshots — clone the real markup
- Match Medvi's layout, spacing, sections, card styles exactly

### Brand Assets (3 logos Dee provided)
- **Wordmark** (text-only logo) → navbar header + footer
- **MP icon** → favicon (replace current generic one)
- **Full logo** → where appropriate
- All 3 must be visibly used on the site

### Copy Changes
- ❌ "physician-prescribed" — NOT accurate for this business
- ✅ "doctor-guided" — correct framing
- Update all copy referencing prescriptions/physicians

### Peptide Images
- Each of the 34 peptides needs a real image on its detail page
- Category cards also need images (not gradient+icon placeholders)

## Research Complete (2026-05-11)

### Medvi Design System (from /tmp/medvi-design-notes.md)
- Built with Framer — hashed classes, not semantic HTML
- **Page order**: Navbar (60px transparent) → Hero (524px, dark green, 70px headline, giant watermark) → 4 Category Cards (216x190, colored top + grey bottom, product image overlapping) → Trust Bar (marquee scroll, 4 icons) → 8 Category Sections (identical template: left product card + right content) → Testimonials (scrolling cards) → Footer
- **Fonts**: Onest (headings), Red Hat Text (body), Montserrat 800 (display)
- **Colors**: Primary green #2E936F, text #242220, off-white #FAF9F7, each category has own color
- **Category cards**: 216x190, split colored top/grey bottom, product image floats above, 20px radius
- **Category sections**: 1120px container, 380px left (product image + benefits checklist) + 640px right (title + photos + description + pill CTA), pastel circle decoration behind
- Full HTML at /tmp/medvi-homepage.html (463K)
- Screenshots at /tmp/medvi-desktop-full.png, /tmp/medvi-mobile-full.png
- Structural data at /tmp/medvi-deep-data.json, /tmp/medvi-final-data.json

### Brand Asset Audit
- Favicon: placeholder SVG text, NOT the real MP monogram — needs replacing
- Footer: plain text, not wordmark image — needs fixing
- Navbar: correctly uses wordmark PNG ✅
- All logos are 1254x1254 square
- No og:image for social sharing

## Dee's Correction (2026-05-11)
**ORCHESTRATE = spawn agents for ALL heavy work. Main thread is dispatch only.**

## Image Generation (Dee generating — 2026-05-11)

Medvi uses AI-generated imagery throughout. Dee is generating matching images.

### Shot List

**Peptide Vials (10 — one per category)**
Clean product photography style, white/light background, professional medical aesthetic:
1. Recovery & Healing — vial + bandage/healing visual
2. Growth Hormone Support — vial + growth/vitality visual
3. Weight Management — vial + fitness/slim visual
4. Men's Health — vial + masculine wellness visual
5. Women's Health — vial + feminine wellness visual
6. Anti-Aging & Longevity — vial + youthful/radiant visual
7. Cognitive & Neurological — vial + brain/focus visual
8. Sleep & Recovery — vial + calm/restful visual
9. Immune Support — vial + shield/protection visual
10. Skin & Hair — vial + skin/hair beauty visual

**Lifestyle Images (5-7 — for category sections)**
Real-looking AI people, warm tones, clean backgrounds (like Medvi's weight loss section):
- Smiling woman in athletic wear (weight management)
- Person self-injecting peptide (product in action — like Medvi's second screenshot)
- Man looking healthy/confident (men's health)
- Woman glowing/radiant (women's health / anti-aging)
- Person sleeping peacefully (sleep & recovery)
- Person in consultation with doctor (trust/social proof)
- Lab/pharmacy aesthetic (trust/credibility)

**Hero Image (1)**
- Either a clean product arrangement (multiple vials on white surface) or a powerful lifestyle shot

### Status: IMAGES DONE — 71 vials + 16 lifestyle organized in public/images/

## Peptide Page Buy Boxes (confirmed 2026-05-11)

Each peptide detail page gets TWO purchase paths:

**Buy Box 1 — Direct Purchase (experienced buyers)**
- Add to Cart button
- Buy Now button
- For people who already know what they want — just let them spend money

**Buy Box 2 — Consultation CTA (first-timers)**
- "Begin Your Consultation" or "Speak to a Doctor" button
- For newcomers who need guidance before purchasing
- Doctor-guided framing, not prescription-required

Both boxes on every peptide page. Two user types, two paths, one page.

## E-Commerce / Checkout (confirmed 2026-05-11)

**Pricing**: Show prices on peptide pages. People price shop — let them see everything.

**Checkout flow**:
- Browse freely (pages, prices, details — all visible)
- To actually CHECK OUT → must create account (or mandatory checkbox agreeing to sign up during checkout)
- Captures email + phone number for retargeting
- Goal: build a customer list from every purchase

**Cart/Payment integration**: Needs a real checkout. Dee's first store build.
- Option A: **Shopify Buy Button / Storefront API** — embed into existing React site, Shopify handles cart + checkout + payments. No migration needed.
- Option B: **Stripe Checkout** — more control, but need to build cart logic ourselves.
- Option C: **Snipcart** — drop-in cart for any site, handles checkout + accounts.
- **DECISION: Shopify Buy Button** (confirmed 2026-05-11)
- $5/mo Starter plan
- Shopify JS SDK embeds into React site
- Products + prices managed in Shopify admin
- Checkout hosted by Shopify (payments, taxes, shipping)
- Mandatory account creation at checkout = email + phone captured
- Abandoned cart flows, retargeting, upsells all built into Shopify

**Setup sequence**:
1. Dee creates Shopify account + Starter plan
2. Dee adds products in Shopify admin (or we bulk-import)
3. Dee provides Storefront API access token + domain
4. Build agent wires up Shopify Buy Button SDK into React site
5. Each peptide page gets real Add to Cart / Buy Now buttons linked to Shopify products

## References
- Site: `/Users/indigochild/mypeptide.org`
- Medvi reference: https://home.medvi.org/
- Current site: https://mypeptide.org/
- Medvi screenshots: `/Users/indigochild/.hermes/image_cache/img_35f58cce882c.jpg`, `img_1bf951efbef4.jpg`
- Logos: `public/logo-full.png`, `logo-icon.png`, `logo-wordmark.png` + white variants
