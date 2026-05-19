#!/usr/bin/env python3
"""Check TDG Access stock for specific tire sizes. READ-ONLY — no orders, no changes."""

import asyncio
import json
import re
from playwright.async_api import async_playwright

SIZES_TO_CHECK = [
    "265/45R20",
    "265/65R17",
    "275/45R20",
    "235/40R19",
    "265/70R18",
    "245/75R16",
    "245/70R17",
    "225/75R16",
    "285/45R21",
    "225/65R17",
    "235/65R18",
    "255/45R19",
]

async def main():
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("Logging in to TDG Access...")
        await page.goto("https://www.tdgaccess.ca/Login")
        await page.wait_for_load_state("networkidle")
        await page.fill('input[name="username"]', "robert@luxurylanemotors.ca")
        await page.fill('input[name="password"]', "Toronto123!")
        await page.click('button[type="submit"]')
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)
        print(f"Logged in. URL: {page.url}")

        for size in SIZES_TO_CHECK:
            print(f"\n{'='*60}")
            print(f"Searching: {size}")
            try:
                # Go back to homepage for fresh search
                await page.goto("https://www.tdgaccess.ca/")
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(1)

                # Use the tire size search input
                tire_input = page.locator('#frontTireSize')
                await tire_input.clear()
                await tire_input.fill(size)
                await asyncio.sleep(1)

                # Click the "Find tires for sale" button (the one under Search by size)
                # It's the second submit button — the one with formaction for BySize
                size_search_btn = page.locator('button:has-text("Find tires for sale")')
                if await size_search_btn.count() > 0:
                    await size_search_btn.first.click()
                else:
                    # Fallback: submit via form action URL directly
                    size_str = size.replace("/", "").replace("R", "")
                    await page.goto(f"https://www.tdgaccess.ca/Tires/Search?Mode=BySize&frontTireSize={size}")

                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(3)

                print(f"  Results URL: {page.url}")

                # Get full page text
                body = await page.inner_text("body")

                # Extract product details
                products = []

                # Parse the text for tire listings
                lines = body.split('\n')
                current_product = {}
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_product:
                            products.append(current_product)
                            current_product = {}
                        continue

                    # Look for brand/model patterns
                    for brand in ["Continental", "Pirelli", "Firestone", "Bridgestone", "Falken", "Hankook", "Yokohama", "Michelin", "Toyo"]:
                        if brand in line and len(line) < 200:
                            current_product['name'] = line

                    # Look for price
                    price_match = re.search(r'\$(\d+\.?\d{0,2})', line)
                    if price_match and 'name' in current_product:
                        current_product['price'] = price_match.group(0)

                    # Look for stock
                    stock_match = re.search(r'(\d+)\s*(?:in stock|available|qty)', line, re.I)
                    if stock_match and 'name' in current_product:
                        current_product['stock'] = stock_match.group(0)

                    if 'out of stock' in line.lower() or 'backorder' in line.lower():
                        if 'name' in current_product:
                            current_product['stock'] = 'OUT OF STOCK'

                if current_product:
                    products.append(current_product)

                # Also get raw prices and stock from body
                all_prices = re.findall(r'\$[\d,]+\.?\d{0,2}', body)
                stock_info = re.findall(r'(?:\d+\s*(?:in stock|available)|out of stock|back.?order|special order)', body, re.I)

                brands_found = [b for b in ["Continental", "Pirelli", "Firestone", "Bridgestone", "Falken", "Hankook", "Yokohama", "Michelin", "Toyo"] if b.lower() in body.lower()]

                result = {
                    "size": size,
                    "url": page.url,
                    "brands_found": brands_found,
                    "all_prices": all_prices[:30],
                    "stock_info": stock_info[:20],
                    "products": [p for p in products if 'name' in p][:30],
                    "body_snippet": body[:4000]
                }
                results.append(result)

                print(f"  Brands: {brands_found}")
                print(f"  Stock: {stock_info[:5]}")
                print(f"  Prices: {all_prices[:10]}")
                print(f"  Products parsed: {len([p for p in products if 'name' in p])}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"size": size, "error": str(e)})

        await browser.close()

    with open("tdg-stock-check.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n\nResults saved to tdg-stock-check.json")

asyncio.run(main())
