const { createClient } = require('@supabase/supabase-js');
const marketData = require('./market-pricing-results.json');

const sb = createClient(
  'https://leyjdephnjcdhkykinmi.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxleWpkZXBobmpjZGhreWtpbm1pIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk1MjMxMiwiZXhwIjoyMDk0NTI4MzEyfQ.1zv3GEkf_ULsY9b6ZO4FBzRT7GN4vJEfJqoLf1y4yaw'
);

// Tires to delist (not on TDG, can't restock)
const DELIST = [
  { brand: 'Falken', model: 'Azenis FK460 AS', size: '265/45R20' },
  { brand: 'Pirelli', model: 'P Zero AS Plus 3', size: '235/40R19' },
  { brand: 'Firestone', model: 'Destination LE3', size: '245/70R17' },
  { brand: 'Bridgestone', model: 'Dueler HL Alenza Plus OWL', size: '225/65R17' },
  { brand: 'Continental', model: 'ProContact RX', size: '255/45R19' },
];

// Firestone wholesale fixes (TDG actual prices)
const WHOLESALE_FIXES = [
  { brand: 'Firestone', model: 'Destination LE3', size: '265/65R17', new_wholesale: 225.12 },
  { brand: 'Firestone', model: 'Destination LE3', size: '265/70R18', new_wholesale: 231.42 },
  { brand: 'Firestone', model: 'Destination LE3', size: '245/75R16', new_wholesale: 193.36 },
];

function normalizeSize(s) {
  return s.trim().toUpperCase()
    .replace(/LT\s*/i, '')
    .replace(/\s*LR[A-Z]$/i, '')
    .replace(/\s+/g, '');
}

async function main() {
  const DRY_RUN = process.argv.includes('--dry-run');
  const APPLY = process.argv.includes('--apply');

  if (!DRY_RUN && !APPLY) {
    console.log('Usage: node apply-market-pricing.cjs --dry-run | --apply');
    process.exit(1);
  }

  // Get brands
  const { data: brands } = await sb.from('brands').select('id, name');
  const brandMap = {};
  const brandIdMap = {};
  brands.forEach(b => { brandMap[b.id] = b.name; brandIdMap[b.name] = b.id; });

  // Get all active tires (paginated — Supabase default limit is 1000)
  let tires = [];
  let from = 0;
  const PAGE = 1000;
  while (true) {
    const { data, error: err } = await sb.from('tires')
      .select('id, brand_id, model, size, rim_diameter, price_wholesale, price_retail')
      .eq('active', true)
      .order('id')
      .range(from, from + PAGE - 1);
    if (err) { console.error('Error:', err); return; }
    tires = tires.concat(data);
    if (data.length < PAGE) break;
    from += PAGE;
  }
  console.log(`Loaded ${tires.length} active tires\n`);

  // ===== STEP 1: DELIST =====
  console.log('=== STEP 1: DELIST (not on TDG) ===\n');
  const delistIds = [];
  for (const dl of DELIST) {
    const match = tires.find(t =>
      brandMap[t.brand_id] === dl.brand &&
      t.model === dl.model &&
      t.size === dl.size
    );
    if (match) {
      delistIds.push(match.id);
      console.log(`  DELIST: ${dl.brand} ${dl.model} ${dl.size}`);
    } else {
      console.log(`  NOT FOUND: ${dl.brand} ${dl.model} ${dl.size}`);
    }
  }

  // ===== STEP 2: FIX WHOLESALE =====
  console.log('\n=== STEP 2: FIX WHOLESALE (TDG actual) ===\n');
  const wholesaleUpdates = [];
  for (const fix of WHOLESALE_FIXES) {
    const match = tires.find(t =>
      brandMap[t.brand_id] === fix.brand &&
      t.model === fix.model &&
      t.size === fix.size
    );
    if (match) {
      console.log(`  FIX: ${fix.brand} ${fix.model} ${fix.size}: $${match.price_wholesale} → $${fix.new_wholesale}`);
      wholesaleUpdates.push({ id: match.id, new_wholesale: fix.new_wholesale });
      match.price_wholesale = fix.new_wholesale;
    }
  }

  // ===== STEP 3: APPLY PRICES FROM market-pricing-results.json =====
  // NO logic reimplementation — just read the recommended_price and apply it
  console.log('\n=== STEP 3: APPLY MARKET PRICING (from compile script) ===\n');

  // Build lookup from market data
  const lookup = {};
  for (const entry of marketData) {
    const key = `${entry.brand}|${entry.model}|${normalizeSize(entry.size)}`;
    lookup[key] = entry;
  }

  const changes = [];
  const unchanged = [];
  const byBrand = {};

  // Only process non-Michelin, non-delisted tires
  const activeTires = tires.filter(t =>
    !delistIds.includes(t.id) &&
    brandMap[t.brand_id] !== 'Michelin'
  );

  for (const tire of activeTires) {
    const brand = brandMap[tire.brand_id];
    const key = `${brand}|${tire.model}|${normalizeSize(tire.size)}`;
    const market = lookup[key];

    if (!byBrand[brand]) byBrand[brand] = { raises: 0, drops: 0, unchanged: 0, totalDiff: 0, count: 0 };
    byBrand[brand].count++;

    if (!market) {
      // No entry in compile results — keep current price
      byBrand[brand].unchanged++;
      unchanged.push({ brand, model: tire.model, size: tire.size, reason: 'not in compile results' });
      continue;
    }

    const newPrice = market.recommended_price;
    // Compare against 30% BASELINE (not current DB price) for reporting
    // Yokohama X-AT baseline is 50%, X-CV is 35%
    let baselineMarkup = 0.30;
    if (brand === 'Yokohama' && tire.model.includes('X-AT')) baselineMarkup = 0.50;
    else if (brand === 'Yokohama' && tire.model.includes('X-CV')) baselineMarkup = 0.35;
    const baselinePrice = Math.round(tire.price_wholesale * (1 + baselineMarkup) * 100) / 100;
    const diffFromBaseline = newPrice - baselinePrice;

    if (Math.abs(diffFromBaseline) > 0.50) {
      changes.push({
        id: tire.id,
        brand,
        model: tire.model,
        size: tire.size,
        wholesale: tire.price_wholesale,
        baselineRetail: baselinePrice,
        newRetail: newPrice,
        diff: diffFromBaseline,
        baselinePct: baselineMarkup * 100,
        newPct: market.recommended_markup_pct,
        direction: market.direction,
        method: market.direction === 'no_data' ? 'KEEP_CURRENT' : `${market.direction.toUpperCase()} (local avg: $${market.local_avg}, ${market.num_sources} src)`,
      });
      if (diffFromBaseline > 0) byBrand[brand].raises++;
      else byBrand[brand].drops++;
      byBrand[brand].totalDiff += diffFromBaseline;
    } else {
      byBrand[brand].unchanged++;
      unchanged.push({ brand, model: tire.model, size: tire.size, reason: 'within $0.50' });
    }
  }

  // Print brand summary
  console.log('BRAND SUMMARY (excluding Michelin):');
  console.log('─'.repeat(90));
  let totalRaises = 0, totalDrops = 0, totalUnchanged = 0, grandTotal = 0;
  for (const [brand, s] of Object.entries(byBrand).sort()) {
    console.log(`  ${brand.padEnd(16)} ${String(s.count).padStart(4)} tires | ${String(s.raises).padStart(3)}↑ ${String(s.drops).padStart(3)}↓ ${String(s.unchanged).padStart(4)}= | net: $${s.totalDiff.toFixed(2)}`);
    totalRaises += s.raises;
    totalDrops += s.drops;
    totalUnchanged += s.unchanged;
    grandTotal += s.totalDiff;
  }
  console.log('─'.repeat(90));
  console.log(`  TOTAL          ${activeTires.length} tires | ${totalRaises}↑ ${totalDrops}↓ ${totalUnchanged}= | net: $${grandTotal.toFixed(2)}\n`);

  // Print changes
  const raises = changes.filter(c => c.diff > 0).sort((a,b) => b.diff - a.diff);
  const drops = changes.filter(c => c.diff < 0).sort((a,b) => a.diff - b.diff);

  console.log(`RAISES (${raises.length}, +$${raises.reduce((s,c) => s+c.diff, 0).toFixed(2)}):`);
  for (const c of raises.slice(0, 20)) {
    console.log(`  ${c.brand.padEnd(14)} ${c.model.padEnd(28)} ${c.size.padEnd(14)} $${c.baselineRetail.toFixed(2)} → $${c.newRetail.toFixed(2)} (+$${c.diff.toFixed(2)}) | ${c.baselinePct.toFixed(0)}% → ${c.newPct.toFixed(0)}%`);
  }
  if (raises.length > 20) console.log(`  ... and ${raises.length - 20} more`);

  console.log(`\nDROPS (${drops.length}, -$${Math.abs(drops.reduce((s,c) => s+c.diff, 0)).toFixed(2)}):`);
  for (const c of drops.slice(0, 20)) {
    console.log(`  ${c.brand.padEnd(14)} ${c.model.padEnd(28)} ${c.size.padEnd(14)} $${c.baselineRetail.toFixed(2)} → $${c.newRetail.toFixed(2)} (-$${Math.abs(c.diff).toFixed(2)}) | ${c.baselinePct.toFixed(0)}% → ${c.newPct.toFixed(0)}%`);
  }
  if (drops.length > 20) console.log(`  ... and ${drops.length - 20} more`);

  // Floor tires
  const floorTires = changes.filter(c => c.newPct <= 20.5 && c.newPct >= 19.5);
  if (floorTires.length > 0) {
    console.log(`\nFLOOR TIRES (${floorTires.length} at 20% floor):`);
    for (const c of floorTires) {
      console.log(`  ${c.brand.padEnd(14)} ${c.model.padEnd(28)} ${c.size.padEnd(14)} 20% = $${c.newRetail.toFixed(2)} | ${c.method}`);
    }
  }

  // Revenue impact
  const raiseTotal = raises.reduce((s,c) => s+c.diff, 0);
  const dropTotal = drops.reduce((s,c) => s+c.diff, 0);
  console.log(`\nREVENUE IMPACT:`);
  console.log(`  Raises: ${raises.length} tires, +$${raiseTotal.toFixed(2)}`);
  console.log(`  Drops:  ${drops.length} tires, -$${Math.abs(dropTotal).toFixed(2)}`);
  console.log(`  Net:    $${(raiseTotal + dropTotal).toFixed(2)}`);
  console.log(`  Floor:  20% | Cap: 60%`);

  // Apply
  if (APPLY) {
    console.log(`\n=== APPLYING ===\n`);

    // Delists
    for (const id of delistIds) {
      const { error } = await sb.from('tires').update({ active: false }).eq('id', id);
      if (error) console.error('Delist error:', error);
    }
    console.log(`Delisted: ${delistIds.length}`);

    // Wholesale fixes
    for (const u of wholesaleUpdates) {
      const { error } = await sb.from('tires')
        .update({ price_wholesale: u.new_wholesale, updated_at: new Date().toISOString() })
        .eq('id', u.id);
      if (error) console.error('Wholesale fix error:', error);
    }
    console.log(`Wholesale fixed: ${wholesaleUpdates.length}`);

    // Price changes
    let success = 0, errors = 0;
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
    }
    console.log(`Prices updated: ${success}, errors: ${errors}`);
  } else {
    console.log('\n--- DRY RUN: No changes applied. Use --apply to update. ---');
  }
}

main().catch(console.error);
