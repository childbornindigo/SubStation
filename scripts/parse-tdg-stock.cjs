const data = require('./tdg-stock-check.json');

const targets = [
  { size: '265/45R20', brand: 'Pirelli', model: 'Scorpion Zero' },
  { size: '265/45R20', brand: 'Falken', model: 'FK460' },
  { size: '265/65R17', brand: 'Firestone', model: 'Destination LE3' },
  { size: '275/45R20', brand: 'Continental', model: 'CrossContact LX Sport' },
  { size: '235/40R19', brand: 'Pirelli', model: 'P Zero' },
  { size: '265/70R18', brand: 'Firestone', model: 'Destination LE3' },
  { size: '245/75R16', brand: 'Firestone', model: 'Destination LE3' },
  { size: '245/70R17', brand: 'Firestone', model: 'Destination LE3' },
  { size: '225/75R16', brand: 'Firestone', model: 'Destination LE3' },
  { size: '285/45R21', brand: 'Bridgestone', model: 'Alenza' },
  { size: '225/65R17', brand: 'Bridgestone', model: 'Dueler' },
  { size: '235/65R18', brand: 'Continental', model: 'CrossContact LX Sport' },
  { size: '255/45R19', brand: 'Continental', model: 'ProContact RX' },
  { size: '255/45R19', brand: 'Pirelli', model: 'P Zero' },
];

data.forEach(sizeResult => {
  const body = sizeResult.body_snippet || '';
  const relevant = targets.filter(t => t.size === sizeResult.size);
  if (!relevant.length) return;

  console.log('\n' + '='.repeat(60));
  console.log('SIZE: ' + sizeResult.size);

  // Split body into product blocks — each starts with a brand name
  const brands = ['Continental', 'Pirelli', 'Firestone', 'Bridgestone', 'Falken', 'Hankook', 'Yokohama', 'Michelin', 'Toyo', 'Kumho', 'Nexen', 'Radar', 'Sailun', 'GT Radial', 'Nitto', 'BFGoodrich'];
  const brandPattern = new RegExp(`^(${brands.join('|')})$`, 'gm');
  const matches = [...body.matchAll(brandPattern)];

  const products = [];
  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : body.length;
    const block = body.substring(start, end);
    const brandName = matches[i][1];

    // Extract model name (usually next line after brand)
    const lines = block.split('\n').map(l => l.trim()).filter(l => l);
    const modelName = lines.length > 1 ? lines[1] : '';

    // Look for stock — pattern is like "43 49" or "- " or "28 99+"
    const stockRegex = /STOCK\s*MSL\s*PRICE[\s\S]*?\n\s*\S+\s+\S+\s+[\d."]+\s+\S+\s+[\S ]+\s+([\d\-][\d ]*\+?)/;
    const stockMatch2 = block.match(stockRegex);

    // Simpler: find the line with numbers that looks like stock after UTQG
    const utqgIdx = block.indexOf('UTQG');
    let stockVal = 'unknown';
    let dealerPrice = 'unknown';

    if (utqgIdx > -1) {
      const afterUtqg = block.substring(utqgIdx);
      // Stock appears between UTQG ratings and price
      // Pattern: 740 A A    43 49    $533.00  $309.14
      const dataLine = afterUtqg.split('\n').slice(2).join(' ');
      // Find stock numbers — they're between UTQG rating and $ price
      const stockPriceMatch = dataLine.match(/[A-Z]\s+([\d\-][\d\s]*?\+?)\s+\$[\d,.]+\s+\$([\d,.]+)/);
      if (stockPriceMatch) {
        stockVal = stockPriceMatch[1].trim();
        dealerPrice = '$' + stockPriceMatch[2];
      } else {
        // Try alternative: just find the price pattern
        const priceMatch = dataLine.match(/\$([\d,.]+)\s+\$([\d,.]+)/);
        if (priceMatch) {
          dealerPrice = '$' + priceMatch[2];
        }
      }
    }

    products.push({
      brand: brandName,
      model: modelName,
      stock: stockVal,
      dealerPrice: dealerPrice,
      block: block.substring(0, 400)
    });
  }

  relevant.forEach(target => {
    const matching = products.filter(p => {
      if (p.brand !== target.brand) return false;
      const modelLower = (p.model || '').toLowerCase();
      const targetParts = target.model.toLowerCase().split(' ');
      return targetParts.some(part => modelLower.includes(part));
    });

    if (matching.length === 0) {
      console.log(`  ${target.brand} ${target.model}: NOT FOUND ON TDG for this size`);
    } else {
      matching.forEach(m => {
        const isOutOfStock = m.stock === '-' || m.stock === '0' || m.stock === '0 0';
        console.log(`  ${target.brand} ${m.model}:`);
        console.log(`    Stock: ${m.stock} ${isOutOfStock ? '❌ OUT OF STOCK' : '✅'}`);
        console.log(`    Dealer price: ${m.dealerPrice}`);
      });
    }
  });
});
