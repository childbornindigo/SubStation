const { createClient } = require('@supabase/supabase-js');
const michelinData = require('./michelin-pricing-results.json');

const sb = createClient(
  'https://leyjdephnjcdhkykinmi.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxleWpkZXBobmpjZGhreWtpbm1pIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk1MjMxMiwiZXhwIjoyMDk0NTI4MzEyfQ.1zv3GEkf_ULsY9b6ZO4FBzRT7GN4vJEfJqoLf1y4yaw'
);

function normalizeSize(s) {
  return s.trim().toUpperCase()
    .replace(/LT\s*/i, '')
    .replace(/\s*LR[A-Z]$/i, '')
    .replace(/\s+/g, '');
}

async function main() {
  console.log('=== MICHELIN MARKET-DRIVEN PRICING — LIVE APPLY ===\n');

  // Get brands
  const { data: brands } = await sb.from('brands').select('id, name');
  const brandMap = {};
  brands.forEach(b => { brandMap[b.id] = b.name; });
  const michelinBrandId = brands.find(b => b.name === 'Michelin')?.id;
  if (!michelinBrandId) {
    console.error('ERROR: Michelin brand not found in DB!');
    process.exit(1);
  }
  console.log(`Michelin brand_id: ${michelinBrandId}\n`);

  // Get all active Michelin tires (paginated)
  let tires = [];
  let from = 0;
  const PAGE = 1000;
  while (true) {
    const { data, error: err } = await sb.from('tires')
      .select('id, brand_id, model, size, rim_diameter, price_wholesale, price_retail')
      .eq('active', true)
      .eq('brand_id', michelinBrandId)
      .order('id')
      .range(from, from + PAGE - 1);
    if (err) { console.error('Error fetching tires:', err); return; }
    tires = tires.concat(data);
    if (data.length < PAGE) break;
    from += PAGE;
  }
  console.log(`Loaded ${tires.length} active Michelin tires from DB\n`);

  // Build lookup from michelin pricing data
  const lookup = {};
  for (const entry of michelinData) {
    const key = `${entry.model}|${normalizeSize(entry.size)}`;
    lookup[key] = entry;
  }

  // Separate delists from price updates
  const delistEntries = michelinData.filter(e => e.direction === 'delist');
  const priceEntries = michelinData.filter(e => e.direction !== 'delist');

  // ===== STEP 1: DELIST =====
  console.log('=== STEP 1: DELIST (wholesale > local avg, can\'t compete) ===\n');
  const delistIds = [];
  for (const dl of delistEntries) {
    const match = tires.find(t =>
      t.model === dl.model &&
      normalizeSize(t.size) === normalizeSize(dl.size)
    );
    if (match) {
      delistIds.push(match.id);
      console.log(`  DELIST: Michelin ${dl.model} ${dl.size} | wholesale: $${dl.wholesale} vs local avg: $${dl.local_avg}`);
    } else {
      console.log(`  NOT FOUND (maybe already inactive): Michelin ${dl.model} ${dl.size}`);
    }
  }
  console.log(`\n  Total to delist: ${delistIds.length}\n`);

  // ===== STEP 2: PRICE UPDATES =====
  console.log('=== STEP 2: PRICE UPDATES ===\n');
  const updates = [];
  const unchanged = [];
  const notFound = [];
  let totalOldRevenue = 0;
  let totalNewRevenue = 0;

  for (const tire of tires) {
    // Skip tires being delisted
    if (delistIds.includes(tire.id)) continue;

    const key = `${tire.model}|${normalizeSize(tire.size)}`;
    const market = lookup[key];

    if (!market) {
      notFound.push({ model: tire.model, size: tire.size, currentPrice: tire.price_retail });
      continue;
    }

    const oldPrice = tire.price_retail;
    const newPrice = market.recommended_price;
    const diff = newPrice - oldPrice;
    const markupPct = market.recommended_markup_pct;

    totalOldRevenue += oldPrice;
    totalNewRevenue += newPrice;

    if (Math.abs(diff) < 0.01) {
      unchanged.push({ model: tire.model, size: tire.size, price: oldPrice, markupPct });
      continue;
    }

    updates.push({
      id: tire.id,
      model: tire.model,
      size: tire.size,
      wholesale: tire.price_wholesale,
      oldPrice,
      newPrice,
      diff,
      markupPct,
      direction: market.direction,
      numSources: market.num_sources,
      localAvg: market.local_avg,
    });
  }

  // Sort: raises first (desc), then drops (asc)
  const raises = updates.filter(u => u.diff > 0).sort((a, b) => b.diff - a.diff);
  const drops = updates.filter(u => u.diff < 0).sort((a, b) => a.diff - b.diff);

  console.log(`RAISES (${raises.length}):`);
  for (const u of raises) {
    console.log(`  Michelin ${u.model.padEnd(28)} ${u.size.padEnd(14)} $${u.oldPrice.toFixed(2)} -> $${u.newPrice.toFixed(2)} (+$${u.diff.toFixed(2)}) | ${u.markupPct.toFixed(1)}% markup | ${u.numSources} src, local avg: $${u.localAvg}`);
  }

  console.log(`\nDROPS (${drops.length}):`);
  for (const u of drops.slice(0, 30)) {
    console.log(`  Michelin ${u.model.padEnd(28)} ${u.size.padEnd(14)} $${u.oldPrice.toFixed(2)} -> $${u.newPrice.toFixed(2)} (-$${Math.abs(u.diff).toFixed(2)}) | ${u.markupPct.toFixed(1)}% markup | ${u.numSources} src, local avg: $${u.localAvg}`);
  }
  if (drops.length > 30) console.log(`  ... and ${drops.length - 30} more drops`);

  console.log(`\nUNCHANGED: ${unchanged.length} tires (price already matches recommended)`);
  if (notFound.length > 0) {
    console.log(`NOT IN RESULTS: ${notFound.length} tires (no matching entry in michelin-pricing-results.json)`);
    for (const nf of notFound) {
      console.log(`  Michelin ${nf.model.padEnd(28)} ${nf.size.padEnd(14)} $${nf.currentPrice.toFixed(2)} (KEPT AS-IS)`);
    }
  }

  // ===== SUMMARY =====
  const raiseTotal = raises.reduce((s, u) => s + u.diff, 0);
  const dropTotal = drops.reduce((s, u) => s + u.diff, 0);

  console.log('\n=== SUMMARY ===\n');
  console.log(`  Total Michelin in DB:     ${tires.length}`);
  console.log(`  To delist:                ${delistIds.length}`);
  console.log(`  Price raises:             ${raises.length} (+$${raiseTotal.toFixed(2)} total)`);
  console.log(`  Price drops:              ${drops.length} (-$${Math.abs(dropTotal).toFixed(2)} total)`);
  console.log(`  Unchanged:                ${unchanged.length}`);
  console.log(`  Not in results (kept):    ${notFound.length}`);
  console.log(`  Net revenue impact:       $${(raiseTotal + dropTotal).toFixed(2)} per unit sold`);
  console.log(`  Total updates to apply:   ${updates.length}`);

  // Sanity checks
  const allUpdates = [...raises, ...drops];
  const minMarkup = allUpdates.length > 0 ? Math.min(...allUpdates.map(u => u.markupPct)) : 0;
  const maxMarkup = allUpdates.length > 0 ? Math.max(...allUpdates.map(u => u.markupPct)) : 0;
  console.log(`\n  Markup range: ${minMarkup.toFixed(1)}% — ${maxMarkup.toFixed(1)}%`);
  if (minMarkup < 15) {
    console.error('\n  WARNING: Some markups below 15%! Aborting for safety.');
    process.exit(1);
  }
  if (maxMarkup > 80) {
    console.error('\n  WARNING: Some markups above 80%! Aborting for safety.');
    process.exit(1);
  }

  // ===== APPLY =====
  console.log('\n=== APPLYING TO SUPABASE ===\n');

  // Delists
  let delistSuccess = 0;
  for (const id of delistIds) {
    const { error } = await sb.from('tires').update({ active: false, updated_at: new Date().toISOString() }).eq('id', id);
    if (error) {
      console.error(`  Delist error (id ${id}):`, error);
    } else {
      delistSuccess++;
    }
  }
  console.log(`  Delisted: ${delistSuccess}/${delistIds.length}`);

  // Price updates (batch 50 at a time)
  let priceSuccess = 0;
  let priceErrors = 0;
  const allPriceUpdates = [...raises, ...drops];

  for (let i = 0; i < allPriceUpdates.length; i += 50) {
    const batch = allPriceUpdates.slice(i, i + 50);
    const promises = batch.map(u =>
      sb.from('tires')
        .update({ price_retail: u.newPrice, updated_at: new Date().toISOString() })
        .eq('id', u.id)
    );
    const results = await Promise.all(promises);
    for (let j = 0; j < results.length; j++) {
      if (results[j].error) {
        priceErrors++;
        console.error(`  Error updating ${batch[j].model} ${batch[j].size}:`, results[j].error);
      } else {
        priceSuccess++;
      }
    }
  }

  console.log(`  Prices updated: ${priceSuccess}/${allPriceUpdates.length}`);
  if (priceErrors > 0) console.log(`  Errors: ${priceErrors}`);

  console.log('\n=== DONE ===');
  console.log(`  ${delistSuccess} delisted, ${priceSuccess} prices updated, ${unchanged.length} unchanged`);
}

main().catch(console.error);
