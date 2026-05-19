const { createClient } = require('@supabase/supabase-js');
const audit = require('./missing-sizes-audit.json');
const tdgRaw = require('./tdg-brand-audit-raw.json');
const marketData = require('./market-pricing-results.json');

const SUPABASE_URL = 'https://leyjdephnjcdhkykinmi.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxleWpkZXBobmpjZGhreWtpbm1pIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk1MjMxMiwiZXhwIjoyMDk0NTI4MzEyfQ.1zv3GEkf_ULsY9b6ZO4FBzRT7GN4vJEfJqoLf1y4yaw';
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const DRY_RUN = process.argv.includes('--dry-run');
const MARKUP_TARGET = 0.30;
const MARKUP_FLOOR = 0.20;
const MARKUP_CAP = 0.60;

function parseSize(size) {
  // Handle LT sizes like "35x13.50R20LT LRF"
  let match = size.match(/(\d+)x([\d.]+)R(\d+)/i);
  if (match) {
    return { width: parseInt(match[1]), aspect_ratio: parseInt(parseFloat(match[2])), rim_diameter: parseInt(match[3]) };
  }
  match = size.match(/(\d+)\/([\d.]+)R(\d+)/i);
  if (match) {
    return { width: parseInt(match[1]), aspect_ratio: parseInt(match[2]), rim_diameter: parseInt(match[3]) };
  }
  return { width: null, aspect_ratio: null, rim_diameter: null };
}

function generateSlug(brand, model, size) {
  const sizeClean = size.replace(/\//g, '-').replace(/\s+/g, '-').toLowerCase();
  return `${brand}-${model}-${sizeClean}`
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function findTDGPrice(brand, model, size) {
  const brandData = tdgRaw[brand] || [];
  // Try exact model match first
  for (const t of brandData) {
    if (t.size === size) {
      // Match by key words in model name
      const modelWords = model.toLowerCase().split(/\s+/);
      const tdgModel = t.model.toLowerCase();
      const matches = modelWords.filter(w => w.length > 2 && tdgModel.includes(w)).length;
      if (matches >= 2 || (modelWords.length <= 2 && matches >= 1)) {
        return t;
      }
    }
  }
  // Fallback: just match brand + size
  for (const t of brandData) {
    if (t.size === size) {
      return t;
    }
  }
  return null;
}

function getMarketPrice(brand, model, size) {
  if (!marketData || !marketData.by_tire) return null;
  for (const tire of marketData.by_tire) {
    if (tire.brand === brand && tire.model === model && tire.size === size) {
      return tire.market_avg;
    }
  }
  return null;
}

function calculateRetail(wholesale, brand, model, size) {
  const marketAvg = getMarketPrice(brand, model, size);
  const targetRetail = wholesale * (1 + MARKUP_TARGET);

  if (!marketAvg) {
    // No market data — use brand/model-specific defaults
    let defaultMarkup = MARKUP_TARGET;
    if (brand === 'Yokohama' && (model.includes('X-AT') || model.includes('Geolandar X-AT'))) defaultMarkup = 0.50;
    else if (brand === 'Yokohama' && (model.includes('X-CV') || model.includes('Geolandar X-CV'))) defaultMarkup = 0.35;
    const defaultRetail = wholesale * (1 + defaultMarkup);
    return { retail: Math.round(defaultRetail * 100) / 100, markup: defaultMarkup, source: `default-${Math.round(defaultMarkup*100)}%` };
  }

  const competitorMinus5 = marketAvg - 5;

  if (targetRetail <= competitorMinus5) {
    // 30% is under market — keep 30% (or raise if gap is huge)
    const maxRetail = wholesale * (1 + MARKUP_CAP);
    const retail = Math.min(competitorMinus5, maxRetail);
    const markup = (retail - wholesale) / wholesale;
    return { retail: Math.round(retail * 100) / 100, markup: Math.round(markup * 1000) / 1000, source: 'market-raise' };
  } else {
    // 30% is above market — drop to market minus $5, but respect floor
    const floorRetail = wholesale * (1 + MARKUP_FLOOR);
    const retail = Math.max(competitorMinus5, floorRetail);
    const markup = (retail - wholesale) / wholesale;
    return { retail: Math.round(retail * 100) / 100, markup: Math.round(markup * 1000) / 1000, source: markup <= MARKUP_FLOOR ? 'floor-20%' : 'market-drop' };
  }
}

async function main() {
  console.log(DRY_RUN ? '=== DRY RUN ===' : '=== LIVE RUN ===');

  // Get brand_id and category_id mappings from existing tires
  const { data: existingTires, error: fetchErr } = await supabase
    .from('tires')
    .select('brand_id, category_id, model, description, features, fits_vehicles, season, speed_rating, load_index, tread_depth, mileage_warranty, rating, review_count')
    .eq('active', true);

  if (fetchErr) { console.error('Fetch error:', fetchErr); return; }

  // Build model → metadata map
  const modelMeta = {};
  for (const t of existingTires) {
    if (!modelMeta[t.model]) {
      modelMeta[t.model] = t;
    }
  }

  const toInsert = [];
  const notFound = [];

  for (const [slug, data] of Object.entries(audit.by_model)) {
    if (data.missing_count === 0) continue;

    const meta = modelMeta[data.model];
    if (!meta) {
      console.log(`WARNING: No metadata found for model "${data.model}" — skipping ${data.missing_count} sizes`);
      notFound.push({ model: data.model, count: data.missing_count });
      continue;
    }

    for (const size of data.missing_sizes) {
      const tdg = findTDGPrice(data.brand, data.model, size);
      if (!tdg) {
        console.log(`WARNING: No TDG price for ${data.brand} ${data.model} ${size}`);
        notFound.push({ brand: data.brand, model: data.model, size });
        continue;
      }

      const { width, aspect_ratio, rim_diameter } = parseSize(size);
      const tireSlug = generateSlug(data.brand, data.model, size);
      const pricing = calculateRetail(tdg.dealer_price, data.brand, data.model, size);

      toInsert.push({
        brand_id: meta.brand_id,
        category_id: meta.category_id,
        model: data.model,
        slug: tireSlug,
        sku: tdg.sku || null,
        size: size,
        width,
        aspect_ratio,
        rim_diameter,
        load_index: meta.load_index,
        speed_rating: meta.speed_rating,
        season: meta.season,
        tread_depth: meta.tread_depth,
        mileage_warranty: meta.mileage_warranty,
        price_wholesale: tdg.dealer_price,
        price_retail: pricing.retail,
        description: meta.description,
        features: meta.features,
        fits_vehicles: meta.fits_vehicles,
        rating: meta.rating,
        review_count: meta.review_count,
        active: true,
        _markup: pricing.markup,
        _source: pricing.source,
        _brand: data.brand,
      });
    }
  }

  console.log(`\n=== SUMMARY ===`);
  console.log(`Tires to add: ${toInsert.length}`);
  console.log(`Not found / skipped: ${notFound.length}`);

  // Show breakdown by brand
  const byBrand = {};
  for (const t of toInsert) {
    if (!byBrand[t._brand]) byBrand[t._brand] = { count: 0, raises: 0, drops: 0, floor: 0, default30: 0 };
    byBrand[t._brand].count++;
    if (t._source === 'market-raise') byBrand[t._brand].raises++;
    else if (t._source === 'market-drop') byBrand[t._brand].drops++;
    else if (t._source === 'floor-20%') byBrand[t._brand].floor++;
    else byBrand[t._brand].default30++;
  }

  console.log('\nBy brand:');
  for (const [brand, stats] of Object.entries(byBrand)) {
    console.log(`  ${brand}: ${stats.count} tires (${stats.default30} at 30%, ${stats.raises} raised, ${stats.drops} dropped, ${stats.floor} at 20% floor)`);
  }

  // Show markup distribution
  const markups = toInsert.map(t => t._markup);
  const avgMarkup = markups.reduce((a, b) => a + b, 0) / markups.length;
  const minMarkup = Math.min(...markups);
  const maxMarkup = Math.max(...markups);
  console.log(`\nMarkup range: ${(minMarkup * 100).toFixed(1)}% - ${(maxMarkup * 100).toFixed(1)}%`);
  console.log(`Average markup: ${(avgMarkup * 100).toFixed(1)}%`);

  // Show the floor tires
  const floorTires = toInsert.filter(t => t._source === 'floor-20%');
  if (floorTires.length > 0) {
    console.log(`\n20% floor tires (${floorTires.length}):`);
    for (const t of floorTires) {
      console.log(`  ${t._brand} ${t.model} ${t.size}: W:$${t.price_wholesale} → R:$${t.price_retail} (${(t._markup * 100).toFixed(1)}%)`);
    }
  }

  // Show some samples
  console.log('\nSample entries:');
  for (const t of toInsert.slice(0, 5)) {
    console.log(`  ${t._brand} ${t.model} ${t.size}: W:$${t.price_wholesale} → R:$${t.price_retail} (${(t._markup * 100).toFixed(1)}%, ${t._source})`);
  }

  if (notFound.length > 0) {
    console.log('\nSkipped:');
    for (const n of notFound) {
      console.log(`  ${n.brand || ''} ${n.model} ${n.size || `(${n.count} sizes)`}`);
    }
  }

  if (DRY_RUN) {
    console.log('\n=== DRY RUN — nothing written ===');
    return;
  }

  // Clean up internal fields before insert
  const cleanInserts = toInsert.map(t => {
    const { _markup, _source, _brand, ...clean } = t;
    return clean;
  });

  // Insert in batches of 50
  let inserted = 0;
  let errors = 0;
  for (let i = 0; i < cleanInserts.length; i += 50) {
    const batch = cleanInserts.slice(i, i + 50);
    const { data, error } = await supabase.from('tires').insert(batch);
    if (error) {
      console.error(`Batch ${i/50 + 1} error:`, error.message);
      // Try one by one
      for (const tire of batch) {
        const { error: singleErr } = await supabase.from('tires').insert(tire);
        if (singleErr) {
          console.error(`  FAILED: ${tire.slug} — ${singleErr.message}`);
          errors++;
        } else {
          inserted++;
        }
      }
    } else {
      inserted += batch.length;
    }
  }

  console.log(`\n=== DONE ===`);
  console.log(`Inserted: ${inserted}`);
  console.log(`Errors: ${errors}`);
}

main().catch(console.error);
