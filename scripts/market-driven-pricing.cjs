/**
 * Market-Driven Pricing Script — LOCAL GTA MARKET DATA
 *
 * Philosophy: 30% markup is the benchmark.
 * - If 30% makes us overpriced vs local market → drop to competitor - $5
 * - If 30% still puts us under market → raise to capture margin (competitor - $5)
 * - Always be the best price in the GTA while maximizing margin
 *
 * LOCAL data sources (May 18, 2026):
 * - KRAVE Automotive (GTA) — Yokohama specialist
 * - Point S Canada — GTA locations (Brampton, Mississauga, Scarborough)
 * - Quattro Tires — GTA dealer
 * - TireDirect.ca — Canadian retailer
 * - tire.ca — Canadian retailer, ships from ON
 * - 4tires.ca — Ontario-based, free ON delivery
 * - Noble Tire (Brampton) — 266 Rutherford Rd S
 * - Costco Tire Centre — GTA (Etobicoke, Markham, Downsview) estimates
 * - ZRacing (Mississauga) — starting prices
 */

const { createClient } = require('@supabase/supabase-js');

const sb = createClient(
  'https://leyjdephnjcdhkykinmi.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxleWpkZXBobmpjZGhreWtpbm1pIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk1MjMxMiwiZXhwIjoyMDk0NTI4MzEyfQ.1zv3GEkf_ULsY9b6ZO4FBzRT7GN4vJEfJqoLf1y4yaw'
);

// ============================================================
// COMPETITOR PRICE DATABASE
// Per-size competitor prices from audit + fresh GTA scrapes
// ============================================================

const COMPETITOR_PRICES = {
  // YOKOHAMA X-AT G016 — KRAVE (GTA), Point S (GTA), Quattro (GTA), TireDirect
  'Geolandar X-AT G016': {
    // From KRAVE Automotive (GTA)
    '265/70R17': 351.00,
    '285/70R17': 432.00,
    '295/70R17': 491.40,
    '35x12.50R17': 496.80,
    '37x12.50R17': 540.00,
    // From Point S Canada
    '265/60R18': 452.99,
    '275/65R18': 442.99,
    '275/70R18': 416.99,
    '285/65R18': 486.99,
    '285/75R18': 510.99,
    '295/70R18': 528.99,
    '33x12.50R20LT LRF': 594.99,
    '35x12.50R20': 632.99,
    '35x13.50R20LT LRE': 657.99,
    '37x12.50R20LT LRE': 668.99,
    '37x13.50R20LT LRE': 668.99,
    // From TireDirect
    '285/70R17': 418.60, // lower than KRAVE, use this
    // From Quattro Tires (audit)
    '265/70R17LT': 380.80,
  },

  // YOKOHAMA X-CV — Point S (GTA), wheelsco.ca (Canadian)
  // NOTE: wheelsco flat $308 sale is clearance — only use for sizes where our wholesale < $250
  'Geolandar X-CV': {
    '215/70R16': 308.04, '195/75R16': 308.04, '215/85R16': 308.04,
    '235/65R16': 308.04, // small sizes where $308 is a valid benchmark
    '215/60R17': 308.04, '215/50R17': 308.04, '235/55R17': 308.04,
    '215/50R18': 308.04, '215/55R18': 308.04, '225/50R18': 308.04,
    '225/55R19': 308.04, // under $250 wholesale — wheelsco is valid
    '275/40R21': 406.99, // Point S
    '285/50R22': 453.00, // wheelsco regular (not the $308 sale)
  },

  // CONTINENTAL ExtremeContact DWS06+ — 4tires.ca (Ontario)
  'ExtremeContact DWS06 PLUS': {
    '195/50R16': 224.99, '225/50R16': 210.80, '225/55R16': 274.99,
    '205/50R17': 246.99, '215/45R17': 240.99, '225/45R17': 254.99,
    '235/45R17': 258.99, '245/40R17': 288.99, '245/45R17': 286.99,
    '245/50R17': 355.99, '255/40R17': 290.99, '255/45R17': 331.99,
    '215/45R18': 274.99, '245/35R18': 317.99, '245/55R18': 337.99,
    '255/35R18': 323.99, '265/35R18': 363.99, '265/40R18': 355.99,
    '275/35R18': 210.80, '275/40R18': 391.99, '285/35R18': 413.99,
    '295/35R18': 436.99,
    '235/55R19': 350.99, '245/35R19': 384.99, '255/35R19': 346.99,
    '265/35R19': 408.99, '275/30R19': 444.99, '285/30R19': 476.99,
    '285/35R19': 471.99, '295/30R19': 210.80,
    '245/40R20': 398.99, '245/45R20': 359.99, '255/35R20': 379.99,
    '255/45R20': 363.99, '265/45R20': 444.99, '275/35R20': 419.99,
    '275/40R20': 427.99, '275/45R20': 461.99, '285/30R20': 456.99,
    '295/40R20': 476.99, '295/45R20': 583.99, '305/30R20': 584.99,
    '315/35R20': 494.99, '335/25R20': 669.99,
    '245/35R21': 484.99, '295/35R21': 611.99, '295/40R21': 612.99,
    '255/30R22': 575.99, '265/35R22': 522.99, '275/40R22': 583.99,
    '285/30R22': 583.99, '285/35R22': 610.99, '295/25R22': 610.99,
  },

  // CONTINENTAL CrossContact RX — Point S, tire.ca
  'CrossContact RX': {
    '225/75R16': 240.00, '235/60R17': 234.51,
    '255/55R19': 450.00, '275/35R21': 575.00, '275/40R22': 580.00,
  },

  // CONTINENTAL CrossContact LX Sport — tire.ca, 4tires.ca (Ontario)
  'CrossContact LX Sport': {
    '215/70R16': 265.00, '235/65R17': 323.15,
    '235/65R18': 290.67, // tire.ca — exact model match (NOTE: below our wholesale $316!)
    '255/50R19': 376.25,
    '255/45R20': 456.19, '265/40R21': 537.40,
    '275/45R21': 518.08, // tire.ca — exact model match
    '285/40R21': 505.92, '275/40R22': 580.20,
  },

  // CONTINENTAL ProContact RX — tire.ca (exact model matches)
  'ProContact RX': {
    '185/65R15': 160.00,
    '215/60R17': 235.00,
    '225/45R17': 286.73, // tire.ca — exact ProContact RX match
    '245/40R19': 420.00, '285/40R21': 500.00, '275/35R22': 580.00,
  },

  // PIRELLI Scorpion Zero All Season — Quattro, V1Auto, Point S (GTA)
  'Scorpion Zero All Season': {
    '245/60R18': 244.39, '255/50R19': 313.29,
    '255/40R20': 394.99, '285/40R21': 420.00, '285/40R22': 550.00,
  },

  // PIRELLI P Zero AS Plus 3 — 4tires.ca (Ontario)
  'P Zero AS Plus 3': {
    '235/40R17': 231.73, '245/45R18': 243.92, '255/45R19': 301.16,
    '245/40R20': 326.60,
  },

  // FALKEN Azenis FK460 AS — tire.ca (local exact match)
  'Azenis FK460 AS': {
    '275/45R21': 433.76, // tire.ca — exact Azenis FK460 A/S match
  },

  // BRIDGESTONE Alenza Sport AS — tire.ca, 4tires.ca (Ontario)
  'Alenza Sport AS': {
    '235/65R17': 270.86,
    '235/65R18': 328.88, // tire.ca — exact Alenza Sport A/S match
    '255/50R19': 433.82,
    '255/50R20': 338.58, '255/40R21': 400.00,
  },

  // FIRESTONE Weathergrip — competitive at 30% per local data
  'Weathergrip': {},

  // FIRESTONE FR710 — 4tires.ca (Ontario)
  'FR710 All Season': {
    '235/55R19': 249.99, '255/50R20': 285.99,
  },

  // FIRESTONE Destination LE3 — 4tires.ca (Ontario), Point S (GTA)
  'Destination LE3': {
    '235/75R15': 195.99,
    '265/70R17': 281.99, // 4tires.ca — exact model, Ontario
    '235/65R18': 274.99, // Point S — exact model, GTA locations
  },

  // HANKOOK Kinergy GT — no local GTA competitor data available
  'Kinergy GT': {},

  // HANKOOK Ventus S1 noble2 RunFlat — Point S (GTA)
  'Ventus S1 noble2 RunFlat': {
    '245/45R18': 241.00,
    '255/45R19': 316.00,
    '285/35R20': 375.00,
  },

  // HANKOOK Dynapro evo AS — limited local data
  'Dynapro evo AS': {
    '275/35R21': 700.00, '275/40R22': 475.00,
  },
};

// ============================================================
// TIERED MARKUP TABLE
// 30% is the benchmark for ALL brands. Only Yokohama X-AT and X-CV
// deviate — supported by LOCAL GTA competitor data (KRAVE, Point S).
// For all other brands, sizes with LOCAL competitor data will adjust
// via the COMPETITOR_PRICES table; everything else stays at 30%.
// ============================================================

const TIERED_MARKUPS = {
  'Yokohama_X-AT': {
    14: 0.55, 15: 0.55, 16: 0.55,
    17: 0.50, 18: 0.45,
    19: 0.42, 20: 0.40,
    21: 0.38, 22: 0.35,
    23: 0.30, 24: 0.30,
  },
  'Yokohama_X-CV': {
    14: 0.38, 15: 0.38, 16: 0.38,
    17: 0.35, 18: 0.32,
    19: 0.30, 20: 0.28,
    21: 0.25, 22: 0.20,
    23: 0.18, 24: 0.18,
  },
};

// Default: 30% for any brand/size not in the table
const DEFAULT_MARKUP = 0.30;

function normalizeSize(size) {
  // Normalize size strings for matching
  // Remove LT prefix, LR suffixes, etc. for matching
  return size
    .replace(/LT\s*/i, '')
    .replace(/\s*LR[A-Z]$/i, '')
    .trim();
}

function getCompetitorPrice(model, size) {
  const modelData = COMPETITOR_PRICES[model];
  if (!modelData) return null;

  // Try exact match first
  if (modelData[size]) return modelData[size];

  // Try normalized match
  const normSize = normalizeSize(size);
  for (const [compSize, price] of Object.entries(modelData)) {
    if (compSize.startsWith('_')) continue;
    if (normalizeSize(compSize) === normSize) return price;
  }

  // Try partial match (e.g., "265/70R17" matches "265/70R17LT")
  for (const [compSize, price] of Object.entries(modelData)) {
    if (compSize.startsWith('_')) continue;
    if (normSize.includes(normalizeSize(compSize)) || normalizeSize(compSize).includes(normSize)) {
      return price;
    }
  }

  // Falken has a flat competitor price
  if (modelData._flat_competitor) return modelData._flat_competitor;

  // X-CV has a default competitor price
  if (modelData._default_competitor) return modelData._default_competitor;

  return null;
}

function getTieredMarkup(brandName, model, rimDiameter) {
  let key = brandName;

  // Special handling for Yokohama models
  if (brandName === 'Yokohama') {
    if (model.includes('X-AT')) key = 'Yokohama_X-AT';
    else if (model.includes('X-CV')) key = 'Yokohama_X-CV';
    else return DEFAULT_MARKUP;
  }

  const tiers = TIERED_MARKUPS[key];
  if (!tiers) return DEFAULT_MARKUP;

  // Clamp rim diameter to range
  const rd = Math.max(14, Math.min(24, rimDiameter));
  return tiers[rd] !== undefined ? tiers[rd] : DEFAULT_MARKUP;
}

function calculatePrice(wholesale, brandName, model, size, rimDiameter) {
  const competitorPrice = getCompetitorPrice(model, size);
  const tieredMarkup = getTieredMarkup(brandName, model, rimDiameter);
  const price30pct = wholesale * 1.30;
  const priceTiered = wholesale * (1 + tieredMarkup);

  let finalPrice;
  let method;

  const MAX_MARKUP = 0.60; // Never go above 60% even if competitor allows
  const maxPrice = wholesale * (1 + MAX_MARKUP);

  if (competitorPrice) {
    // We have actual competitor data — price to beat them by $5
    const targetPrice = competitorPrice - 5;
    const minPrice = wholesale * 1.10; // Floor: never below 10% margin

    if (targetPrice < minPrice) {
      finalPrice = minPrice;
      method = 'FLOOR_10PCT (competitor too low)';
    } else if (targetPrice > maxPrice) {
      finalPrice = maxPrice;
      method = `CAP_60% (comp: $${competitorPrice.toFixed(2)}, capped from ${((targetPrice/wholesale-1)*100).toFixed(0)}%)`;
    } else {
      finalPrice = targetPrice;
      method = `COMPETITOR-$5 (comp: $${competitorPrice.toFixed(2)})`;
    }
  } else {
    // No competitor data — use tiered markup (already capped in tier table)
    finalPrice = Math.min(priceTiered, maxPrice);
    method = `TIERED_${(tieredMarkup * 100).toFixed(0)}%`;
  }

  // Round to 2 decimal places
  finalPrice = Math.round(finalPrice * 100) / 100;

  return { finalPrice, method, competitorPrice, tieredMarkup };
}

async function main() {
  const DRY_RUN = process.argv.includes('--dry-run');
  const APPLY = process.argv.includes('--apply');

  if (!DRY_RUN && !APPLY) {
    console.log('Usage: node market-driven-pricing.js --dry-run | --apply');
    console.log('  --dry-run: Show what would change without updating');
    console.log('  --apply:   Actually update Supabase');
    process.exit(1);
  }

  // Get brands
  const { data: brands } = await sb.from('brands').select('id, name');
  const brandMap = {};
  brands.forEach(b => brandMap[b.id] = b.name);

  // Get all active tires
  const { data: tires, error } = await sb.from('tires')
    .select('id, brand_id, model, size, rim_diameter, price_wholesale, price_retail')
    .eq('active', true)
    .order('brand_id')
    .order('model')
    .order('rim_diameter');

  if (error) { console.error('Supabase error:', error); return; }

  console.log(`\nAnalyzing ${tires.length} active tires...\n`);

  const changes = [];
  const noChanges = [];
  const byBrand = {};

  for (const tire of tires) {
    const brandName = brandMap[tire.brand_id] || 'Unknown';
    const { finalPrice, method, competitorPrice, tieredMarkup } =
      calculatePrice(tire.price_wholesale, brandName, tire.model, tire.size, tire.rim_diameter);

    const currentMarkup = ((tire.price_retail - tire.price_wholesale) / tire.price_wholesale * 100).toFixed(1);
    const newMarkup = ((finalPrice - tire.price_wholesale) / tire.price_wholesale * 100).toFixed(1);
    const diff = finalPrice - tire.price_retail;

    // Track by brand
    if (!byBrand[brandName]) byBrand[brandName] = { raises: 0, drops: 0, unchanged: 0, totalDiff: 0 };

    if (Math.abs(diff) > 0.50) { // Only change if more than $0.50 difference
      changes.push({
        id: tire.id,
        brand: brandName,
        model: tire.model,
        size: tire.size,
        rim: tire.rim_diameter,
        wholesale: tire.price_wholesale,
        oldRetail: tire.price_retail,
        newRetail: finalPrice,
        diff: diff,
        oldMarkup: currentMarkup,
        newMarkup: newMarkup,
        method: method,
      });

      if (diff > 0) byBrand[brandName].raises++;
      else byBrand[brandName].drops++;
      byBrand[brandName].totalDiff += diff;
    } else {
      noChanges.push({ brand: brandName, model: tire.model, size: tire.size });
      byBrand[brandName].unchanged++;
    }
  }

  // Print summary by brand
  console.log('=== BRAND SUMMARY ===\n');
  for (const [brand, stats] of Object.entries(byBrand).sort()) {
    const total = stats.raises + stats.drops + stats.unchanged;
    const avgDiff = stats.totalDiff / (stats.raises + stats.drops || 1);
    console.log(`${brand}: ${total} tires | ${stats.raises} raises, ${stats.drops} drops, ${stats.unchanged} unchanged | avg change: $${avgDiff.toFixed(2)}`);
  }

  // Print all changes grouped by model
  console.log(`\n=== ${changes.length} PRICE CHANGES ===\n`);

  let currentModel = '';
  for (const c of changes.sort((a,b) => {
    if (a.brand !== b.brand) return a.brand.localeCompare(b.brand);
    if (a.model !== b.model) return a.model.localeCompare(b.model);
    return a.rim - b.rim;
  })) {
    const modelKey = `${c.brand} ${c.model}`;
    if (modelKey !== currentModel) {
      currentModel = modelKey;
      console.log(`\n--- ${modelKey} ---`);
    }
    const arrow = c.diff > 0 ? '↑' : '↓';
    console.log(`  ${c.size.padEnd(22)} W:$${c.wholesale.toFixed(2).padStart(7)} | $${c.oldRetail.toFixed(2).padStart(7)} → $${c.newRetail.toFixed(2).padStart(7)} (${arrow}$${Math.abs(c.diff).toFixed(2).padStart(6)}) | ${c.oldMarkup}% → ${c.newMarkup}% | ${c.method}`);
  }

  console.log(`\n=== ${noChanges.length} UNCHANGED ===`);

  // Total impact
  const totalRaises = changes.filter(c => c.diff > 0);
  const totalDrops = changes.filter(c => c.diff < 0);
  const raiseRevenue = totalRaises.reduce((sum, c) => sum + c.diff, 0);
  const dropRevenue = totalDrops.reduce((sum, c) => sum + c.diff, 0);

  console.log(`\n=== REVENUE IMPACT ===`);
  console.log(`Raises: ${totalRaises.length} tires, avg +$${(raiseRevenue/totalRaises.length||0).toFixed(2)}/tire, total +$${raiseRevenue.toFixed(2)}`);
  console.log(`Drops:  ${totalDrops.length} tires, avg -$${(Math.abs(dropRevenue)/(totalDrops.length||1)).toFixed(2)}/tire, total -$${Math.abs(dropRevenue).toFixed(2)}`);
  console.log(`Net:    $${(raiseRevenue + dropRevenue).toFixed(2)} per full inventory sell-through`);

  if (APPLY && changes.length > 0) {
    console.log(`\n=== APPLYING ${changes.length} CHANGES ===\n`);

    let success = 0;
    let errors = 0;

    // Batch updates in groups of 50
    for (let i = 0; i < changes.length; i += 50) {
      const batch = changes.slice(i, i + 50);
      const promises = batch.map(c =>
        sb.from('tires')
          .update({ price_retail: c.newRetail, updated_at: new Date().toISOString() })
          .eq('id', c.id)
      );

      const results = await Promise.all(promises);
      for (const r of results) {
        if (r.error) { errors++; console.error('Error:', r.error); }
        else success++;
      }

      process.stdout.write(`  Updated ${Math.min(i + 50, changes.length)}/${changes.length}\r`);
    }

    console.log(`\nDone: ${success} updated, ${errors} errors`);
  } else if (DRY_RUN) {
    console.log('\n--- DRY RUN: No changes applied. Use --apply to update Supabase ---');
  }
}

main().catch(console.error);
