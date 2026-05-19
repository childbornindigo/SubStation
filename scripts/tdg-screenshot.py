#!/usr/bin/env python3
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.tdgaccess.ca/")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)
        await page.screenshot(path="tdg-login.png", full_page=True)

        # Get all input elements
        inputs = await page.locator("input").all()
        for inp in inputs:
            name = await inp.get_attribute("name")
            typ = await inp.get_attribute("type")
            placeholder = await inp.get_attribute("placeholder")
            inp_id = await inp.get_attribute("id")
            print(f"Input: name={name} type={typ} placeholder={placeholder} id={inp_id}")

        # Get all buttons
        buttons = await page.locator("button").all()
        for btn in buttons:
            text = await btn.inner_text()
            typ = await btn.get_attribute("type")
            print(f"Button: text='{text}' type={typ}")

        # Also check for forms
        forms = await page.locator("form").count()
        print(f"\nForms found: {forms}")

        # Check page title and URL
        print(f"Title: {await page.title()}")
        print(f"URL: {page.url}")

        await browser.close()

asyncio.run(main())
