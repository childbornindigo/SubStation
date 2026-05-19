import json, re, os

# Load all local price data
all_prices = []
files = ['local-prices-point-s.json', 'local-prices-zracing.json', 'local-prices-noble-tire.json', 
         'local-prices-tdot-performance.json', 'local-prices-canadian-tire.json', 'local-prices-krave.json',
         'local-prices-michelin.json']
for f in files:
    try:
        data = json.load(open(f))
        if isinstance(data, list):
            all_prices.extend(data)
    except:
        pass

# Model name normalization map — local shop names → our DB model names
MODEL_NORMALIZE = {
    # Bridgestone
    'Alenza Sport A/S': 'Alenza Sport AS',
    'Alenza  Sport As': 'Alenza Sport AS',
    'Alenza Sport A/S All Season': 'Alenza Sport AS',
    # Continental
    'ExtremeContact DWS06+': 'ExtremeContact DWS06 PLUS',
    'ExtremeContact DWS06 Plus': 'ExtremeContact DWS06 PLUS',
    'Continental Extremecontact Dws  06 Plus': 'ExtremeContact DWS06 PLUS',
    'Procontact Rx': 'ProContact RX',
    # Falken
    'Azenis FK460 A/S': 'Azenis FK460 AS',
    # Firestone
    'All  Season': 'FR710',
    'Firestone All Season': 'FR710',
    'Destination  Le-3': 'Destination LE3',
    'Destination LE3 All Season': 'Destination LE3',
    'Firestone Destination LE2 All Season': 'Destination LE 2',
    'Firestone WeatherGrip All Weather': 'Weathergrip',
    'WeatherGrip All Weather': 'Weathergrip',
    # Hankook
    'Kinergy GT (HT436) All Season': 'H436 Kinergy GT',
    'Kinergy GT': 'H436 Kinergy GT',
    'Ventus S1 noble2 (H452) All Season': 'H452 Ventus S1 noble2',
    'Ventus S1 noble2': 'H452 Ventus S1 noble2',
    'Dynapro evo AS': 'RA21 Dynapro evo AS',
    # Michelin
    'Pilot Sport A/S 4': 'Pilot Sport AS 4',
    'Michelin Pilot Sport A/S 4': 'Pilot Sport AS 4',
    'Primacy Tour A/S': 'Primacy Tour AS',
    # Pirelli
    'P Zero A/S  Plus 3': 'P Zero AS Plus 3',
    'P Zero A/S +3': 'P Zero AS Plus 3',
    'P7  As Plus 3': 'P Zero AS Plus 3',
    # Toyo
    'Open Country A/T III': 'OPA50',
    'Open Country A/T III All Terrain': 'OPA50',
    'Open Country A50': 'OPA50',
    # Yokohama
    'Geolandar X-AT': 'Geolandar X AT G016',
    'Geolandar X-CV': 'Geolandar X CV',
    'Geolandar X-CV All Season': 'Geolandar X CV',
}

# Our DB models
OUR_MODELS = {
    'Bridgestone': ['Alenza Sport AS', 'Turanza Everdrive', 'UltraWeather', 'WeatherPeak'],
    'Continental': ['CrossContact LX Sport', 'CrossContact RX', 'ExtremeContact DWS06 PLUS', 'ProContact RX'],
    'Falken': ['Azenis FK460 AS'],
    'Firestone': ['Destination LE 2', 'Destination LE3', 'FR710', 'Weathergrip'],
    'Hankook': ['H436 Kinergy GT', 'H452 Ventus S1 noble2', 'RA21 Dynapro evo AS'],
    'Michelin': ['Defender LTX M/S2', 'Pilot Sport AS 4', 'Primacy Tour AS'],
    'Pirelli': ['P Zero AS Plus 3', 'Scorpion Zero All Season'],
    'Toyo': ['Extensa AS II', 'OPA50'],
    'Yokohama': ['Geolandar X AT G016', 'Geolandar X CV'],
}

def normalize_size(size_str):
    """Extract just the tire size like 205/45R17 from various formats"""
    m = re.search(r'(\d{3}/\d{2}[A-Z]*R?\d{2})', str(size_str).upper().replace(' ', ''))
    if m:
        s = m.group(1)
        if 'R' not in s:
            # Insert R before rim diameter
            s = re.sub(r'(\d{2,3}/\d{2})(\d{2})$', r'\1R\2', s)
        return s
    # LT sizes
    m = re.search(r'(LT?\d{3}/\d{2}R\d{2})', str(size_str).upper().replace(' ', ''))
    if m:
        return m.group(1)
    # 33x12.50R20 format
    m = re.search(r'(\d{2,3}X\d+\.?\d*R\d{2})', str(size_str).upper().replace(' ', ''))
    if m:
        return m.group(1)
    return size_str.strip()

# Normalize and filter to our models only
matched_prices = []
for p in all_prices:
    brand = p.get('brand', '')
    model = p.get('model', '')
    price = p.get('price', 0)
    shop = p.get('shop', '')
    size = p.get('size', '')
    
    if not brand or not model or not price or price <= 0:
        continue
    
    # Normalize model name
    norm_model = MODEL_NORMALIZE.get(model, model)
    
    # Check if this matches one of our models
    if brand in OUR_MODELS and norm_model in OUR_MODELS[brand]:
        norm_size = normalize_size(size)
        matched_prices.append({
            'brand': brand,
            'model': norm_model,
            'size': norm_size,
            'price': float(price),
            'shop': shop,
        })

print(f"Total matched local prices: {len(matched_prices)}")

# Group by brand/model/size and compute average
from collections import defaultdict
grouped = defaultdict(list)
for p in matched_prices:
    key = f"{p['brand']}|{p['model']}|{p['size']}"
    grouped[key].append(p)

averages = {}
for key, prices in grouped.items():
    brand, model, size = key.split('|')
    avg_price = sum(p['price'] for p in prices) / len(prices)
    shops = list(set(p['shop'] for p in prices))
    averages[key] = {
        'brand': brand,
        'model': model,
        'size': size,
        'avg_local_price': round(avg_price, 2),
        'num_shops': len(shops),
        'shops': shops,
        'prices': [(p['shop'], p['price']) for p in prices],
    }

print(f"Unique brand/model/size combos with averages: {len(averages)}")

# Load our inventory
inv = json.load(open('inventory-dump.json'))
print(f"Our inventory: {len(inv)} tires")

# Cross-reference
results = []
for tire in inv:
    brand = tire.get('brand', '')
    model = tire.get('model', '')
    size = normalize_size(tire.get('size', ''))
    wholesale = tire.get('wholesale', 0) or 0
    current_retail = tire.get('retail', 0) or 0
    
    if wholesale <= 0:
        continue
    
    baseline_30 = round(wholesale * 1.30, 2)
    
    key = f"{brand}|{model}|{size}"
    market = averages.get(key)
    
    if market:
        avg_local = market['avg_local_price']
        # Target: competitor avg - $5
        target = avg_local - 5
        
        # Apply 20% floor
        floor_price = round(wholesale * 1.20, 2)
        
        if target < floor_price:
            final_price = floor_price
            action = 'FLOOR_20%'
        elif target > baseline_30:
            final_price = target
            action = 'RAISE'
        else:
            final_price = target
            action = 'DROP'
        
        markup_pct = round((final_price / wholesale - 1) * 100, 1)
        change_from_30 = round(final_price - baseline_30, 2)
        change_pct = round((final_price - baseline_30) / baseline_30 * 100, 1)
        
        results.append({
            'brand': brand,
            'model': model,
            'size': size,
            'wholesale': wholesale,
            'baseline_30': baseline_30,
            'avg_local': avg_local,
            'num_shops': market['num_shops'],
            'shops': market['shops'],
            'final_price': round(final_price, 2),
            'markup_pct': markup_pct,
            'action': action,
            'change_from_30': change_from_30,
            'change_pct': change_pct,
        })
    else:
        results.append({
            'brand': brand,
            'model': model,
            'size': size,
            'wholesale': wholesale,
            'baseline_30': baseline_30,
            'avg_local': None,
            'num_shops': 0,
            'shops': [],
            'final_price': baseline_30,
            'markup_pct': 30.0,
            'action': 'NO_DATA',
            'change_from_30': 0,
            'change_pct': 0,
        })

# Summary
raises = [r for r in results if r['action'] == 'RAISE']
drops = [r for r in results if r['action'] == 'DROP']
floors = [r for r in results if r['action'] == 'FLOOR_20%']
no_data = [r for r in results if r['action'] == 'NO_DATA']

print(f"\n=== SUMMARY ===")
print(f"Total tires: {len(results)}")
print(f"Raises: {len(raises)} (+${sum(r['change_from_30'] for r in raises):,.2f})")
print(f"Drops: {len(drops)} (${sum(r['change_from_30'] for r in drops):,.2f})")
print(f"Floor (20%): {len(floors)} (${sum(r['change_from_30'] for r in floors):,.2f})")
print(f"No data (stay 30%): {len(no_data)}")
print(f"Net change: ${sum(r['change_from_30'] for r in results):,.2f}")

# Save full results
json.dump(results, open('market-pricing-v2-results.json', 'w'), indent=2)

# Print per-brand breakdown
print(f"\n=== BY BRAND ===")
brands = sorted(set(r['brand'] for r in results))
for brand in brands:
    br = [r for r in results if r['brand'] == brand]
    br_raises = [r for r in br if r['action'] == 'RAISE']
    br_drops = [r for r in br if r['action'] == 'DROP']
    br_floors = [r for r in br if r['action'] == 'FLOOR_20%']
    br_nodata = [r for r in br if r['action'] == 'NO_DATA']
    net = sum(r['change_from_30'] for r in br)
    print(f"{brand}: {len(br)} tires | {len(br_raises)}↑ {len(br_drops)}↓ {len(br_floors)} floor | {len(br_nodata)} no data | net ${net:,.2f}")

# Print the specific example
print(f"\n=== SPECIFIC: Michelin Pilot Sport AS 4 205/45R17 ===")
specific = [r for r in results if 'Pilot Sport' in r['model'] and '205/45R17' in r['size']]
for s in specific:
    print(f"  Wholesale: ${s['wholesale']}")
    print(f"  30% baseline: ${s['baseline_30']}")
    print(f"  Local avg: ${s['avg_local']} ({s['num_shops']} shops: {s['shops']})")
    print(f"  Final price: ${s['final_price']} ({s['markup_pct']}% markup)")
    print(f"  Action: {s['action']} ({s['change_pct']}% from 30%)")
