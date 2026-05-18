#!/usr/bin/env python3
"""Generate Facebook Marketplace listing images for LuxuryLane Tires."""

import urllib.request
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# Brand colors
GOLD = (212, 175, 55)
DARK_BG = (18, 18, 22)
CARD_BG = (28, 28, 34)
WHITE = (255, 255, 255)
LIGHT_GRAY = (180, 180, 190)
PRICE_GREEN = (34, 197, 94)
ACCENT_GOLD = (255, 215, 0)

TIRES = [
    {
        "brand": "Pirelli",
        "model": "P Zero All Season",
        "size": "235/45R18",
        "wholesale": 153.94,
        "price": 235,
        "season": "All-Season",
        "load": "94V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/c317f5ac840dbd0b88c44033f075a08daf60474e_p_zero_all_season_b92a4718-ea5a-47b4-9f56-2d1078b5a6f1.jpg?v=1718308562",
        "vehicles": "BMW 3 Series, Mercedes C-Class, Audi A4/A5, Lexus IS, Infiniti Q50"
    },
    {
        "brand": "Hankook",
        "model": "WeatherFlex GT",
        "size": "225/50R17",
        "wholesale": 157.20,
        "price": 239,
        "season": "All-Weather",
        "load": "98V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/Hankook-Weatherflex-GT-H755A-All-Weather-235-55R20-102V-Passenger-Tire.7bea506c04a770e151f4e7bc0a485cd2_1d84d0c4-f3c6-460c-a5c6-875fa3daaef5.webp?v=1752070595",
        "vehicles": "Honda Accord, Toyota Camry, Mazda 6, Kia K5, Hyundai Sonata"
    },
    {
        "brand": "Hankook",
        "model": "Kinergy 4S2",
        "size": "225/40R18",
        "wholesale": 149.01,
        "price": 229,
        "season": "All-Weather",
        "load": "92W",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/a68c7526e2dcd5ae4b8f2abb8dcd32e8a4cf563b_kinergy_4_s2_h750_2f2c9733-dd07-472c-a686-62e6e0078871.jpg?v=1718306258",
        "vehicles": "VW Golf GTI, Honda Civic Si, Mazda 3, Subaru WRX, Hyundai Elantra N"
    },
    {
        "brand": "Hankook",
        "model": "WeatherFlex GT",
        "size": "235/40R19",
        "wholesale": 190.24,
        "price": 270,
        "season": "All-Weather",
        "load": "96V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/Hankook-Weatherflex-GT-H755A-All-Weather-235-55R20-102V-Passenger-Tire.7bea506c04a770e151f4e7bc0a485cd2_a9688fb9-ba95-45f3-b6c6-535b8d7db572.webp?v=1752070600",
        "vehicles": "BMW 3/4 Series, Mercedes C-Class, Audi S4/S5, Genesis G70"
    },
    {
        "brand": "Hankook",
        "model": "WeatherFlex GT",
        "size": "255/45R19",
        "wholesale": 226.80,
        "price": 309,
        "season": "All-Weather",
        "load": "104V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/Hankook-Weatherflex-GT-H755A-All-Weather-235-55R20-102V-Passenger-Tire.7bea506c04a770e151f4e7bc0a485cd2_cf95ee44-470b-4bdd-ab43-fe1b234d5cbf.webp?v=1752070600",
        "vehicles": "BMW X3/X4, Mercedes GLC, Audi Q5, Porsche Macan"
    },
    {
        "brand": "Pirelli",
        "model": "Scorpion Verde AS",
        "size": "235/55R18",
        "wholesale": 175.28,
        "price": 255,
        "season": "All-Season",
        "load": "100V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/625906636fa4167aa431ac53a7639f40d4deb3c0_scorpion_verde_all_season_0d189054-cb7b-4e9c-a0f7-4fe76cfceba4.jpg?v=1718308198",
        "vehicles": "Hyundai Tucson, Kia Sportage, Ford Escape, Chevy Equinox, Toyota RAV4"
    },
    {
        "brand": "Hankook",
        "model": "Kinergy XP",
        "size": "245/40R18",
        "wholesale": 198.69,
        "price": 279,
        "season": "All-Season",
        "load": "97W",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/HANKOOK-Kinergy-XP-H446-225-60R18-100V-Quantity-of-1.3c9c5b30f3dd3cbff568e165937da237_d02936f4-c63a-4f42-8fc9-71948962a84c.png?v=1741360472",
        "vehicles": "BMW 3 Series, Lexus IS, Infiniti Q50, Genesis G70, Acura TLX"
    },
    {
        "brand": "Pirelli",
        "model": "P Zero AS Plus 3",
        "size": "255/50R19",
        "wholesale": 247.29,
        "price": 329,
        "season": "All-Season",
        "load": "107V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/pirelli-p-zero-all-season-plus-3_61dd06fa-5bdf-4229-a4de-8aa05841b33b.png?v=1718313842",
        "vehicles": "BMW X5, Mercedes GLE, Audi Q7, Volvo XC90, Porsche Cayenne"
    },
    {
        "brand": "Hankook",
        "model": "WeatherFlex GT",
        "size": "235/55R19",
        "wholesale": 172.55,
        "price": 255,
        "season": "All-Weather",
        "load": "101V",
        "image_url": "https://cdn.shopify.com/s/files/1/0515/0851/0900/files/Hankook-Weatherflex-GT-H755A-All-Weather-235-55R20-102V-Passenger-Tire.7bea506c04a770e151f4e7bc0a485cd2_476278fe-6235-45c4-a6ad-4caac1cd396c.webp?v=1752070598",
        "vehicles": "Toyota Highlander, Hyundai Santa Fe, Kia Sorento, Ford Edge, Subaru Outback"
    },
]


def download_image(url):
    """Download image from URL, return PIL Image."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGBA")
    except Exception as e:
        print(f"  Failed to download {url[:60]}...: {e}")
        return None


def get_font(size, bold=False):
    """Get system font."""
    fonts_to_try = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    if bold:
        fonts_to_try = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ] + fonts_to_try

    for fp in fonts_to_try:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size, index=1 if bold and fp.endswith('.ttc') else 0)
            except Exception:
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=0):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def create_tire_listing(tire, index):
    """Create a professional FB Marketplace listing image."""
    W, H = 1200, 1200
    img = Image.new("RGB", (W, H), DARK_BG)
    draw = ImageDraw.Draw(img)

    # Gradient background effect
    for y in range(H):
        r = int(18 + (y / H) * 12)
        g = int(18 + (y / H) * 10)
        b = int(22 + (y / H) * 14)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Gold accent line at top
    draw.rectangle([(0, 0), (W, 5)], fill=GOLD)

    # "LUXURYLANE TIRES" header
    font_header = get_font(32, bold=True)
    draw.text((W // 2, 35), "LUXURYLANE TIRES", font=font_header, fill=GOLD, anchor="mt")

    # Thin separator
    draw.line([(100, 65), (W - 100, 65)], fill=(60, 60, 70), width=1)

    # Season badge
    font_badge = get_font(22, bold=True)
    season_text = tire["season"].upper()
    badge_w = draw.textlength(season_text, font=font_badge) + 30
    badge_x = W // 2 - badge_w // 2
    draw_rounded_rect(draw, (badge_x, 78, badge_x + badge_w, 110), radius=15, fill=(45, 45, 55), outline=GOLD, width=1)
    draw.text((W // 2, 94), season_text, font=font_badge, fill=GOLD, anchor="mm")

    # Tire image
    tire_img = download_image(tire["image_url"])
    if tire_img:
        tire_size = 520
        tire_img = tire_img.resize((tire_size, tire_size), Image.LANCZOS)
        paste_x = (W - tire_size) // 2
        paste_y = 130
        if tire_img.mode == "RGBA":
            img.paste(tire_img, (paste_x, paste_y), tire_img)
        else:
            img.paste(tire_img, (paste_x, paste_y))
    else:
        # Placeholder
        font_placeholder = get_font(48, bold=True)
        draw.text((W // 2, 390), "TIRE IMAGE", font=font_placeholder, fill=LIGHT_GRAY, anchor="mm")

    # Brand + Model
    font_brand = get_font(52, bold=True)
    font_model = get_font(38)
    draw.text((W // 2, 680), tire["brand"].upper(), font=font_brand, fill=WHITE, anchor="mt")
    draw.text((W // 2, 738), tire["model"], font=font_model, fill=LIGHT_GRAY, anchor="mt")

    # Size — big and prominent
    font_size = get_font(72, bold=True)
    draw.text((W // 2, 800), tire["size"], font=font_size, fill=WHITE, anchor="mt")

    # Specs line
    font_specs = get_font(24)
    specs_text = f"Load/Speed: {tire['load']}  •  {tire['season']}"
    draw.text((W // 2, 880), specs_text, font=font_specs, fill=LIGHT_GRAY, anchor="mt")

    # Price box
    price_text = f"${tire['price']}"
    font_price = get_font(64, bold=True)
    font_per = get_font(24)
    price_w = draw.textlength(price_text, font=font_price)
    per_w = draw.textlength("/tire", font=font_per)
    total_price_w = price_w + per_w + 10

    box_w = total_price_w + 60
    box_h = 90
    box_x = (W - box_w) // 2
    box_y = 920
    draw_rounded_rect(draw, (box_x, box_y, box_x + box_w, box_y + box_h), radius=12, fill=(20, 60, 20), outline=PRICE_GREEN, width=2)

    price_start_x = W // 2 - total_price_w // 2
    draw.text((price_start_x, box_y + 12), price_text, font=font_price, fill=PRICE_GREEN)
    draw.text((price_start_x + price_w + 8, box_y + 42), "/tire", font=font_per, fill=(120, 200, 120))

    # Vehicles fits line
    font_fits = get_font(20)
    fits_text = f"Fits: {tire['vehicles']}"
    if draw.textlength(fits_text, font=font_fits) > W - 80:
        # Truncate
        vehicles = tire["vehicles"].split(", ")
        fits_text = f"Fits: {', '.join(vehicles[:4])} + more"
    draw.text((W // 2, 1030), fits_text, font=font_fits, fill=LIGHT_GRAY, anchor="mt")

    # Bottom bar
    draw.rectangle([(0, 1090), (W, H)], fill=(25, 25, 32))
    draw.rectangle([(0, 1090), (W, 1092)], fill=GOLD)

    font_bottom = get_font(22, bold=True)
    font_loc = get_font(20)
    draw.text((W // 2, 1118), "PROFESSIONAL INSTALLATION AVAILABLE", font=font_bottom, fill=GOLD, anchor="mt")
    draw.text((W // 2, 1150), "Vaughan, ON  •  luxurylanetires.ca", font=font_loc, fill=LIGHT_GRAY, anchor="mt")

    # Gold accent line at bottom
    draw.rectangle([(0, H - 5), (W, H)], fill=GOLD)

    filename = f"tire_{index+1}_{tire['size'].replace('/','_')}_{tire['brand'].lower()}.jpg"
    filepath = os.path.join(IMG_DIR, filename)
    img.convert("RGB").save(filepath, "JPEG", quality=95)
    print(f"  Saved: {filename}")
    return filepath


def create_general_warehouse(variant):
    """Create a general 'tire warehouse' FB Marketplace image."""
    W, H = 1200, 1200
    img = Image.new("RGB", (W, H), DARK_BG)
    draw = ImageDraw.Draw(img)

    # Gradient
    for y in range(H):
        r = int(18 + (y / H) * 15)
        g = int(18 + (y / H) * 12)
        b = int(22 + (y / H) * 18)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Gold top bar
    draw.rectangle([(0, 0), (W, 6)], fill=GOLD)

    if variant == 1:
        # "ALL BRANDS • ALL SIZES" listing
        font_main = get_font(72, bold=True)
        font_sub = get_font(42, bold=True)
        font_body = get_font(28)
        font_list = get_font(26)
        font_cta = get_font(36, bold=True)

        draw.text((W // 2, 100), "LUXURYLANE", font=font_main, fill=GOLD, anchor="mt")
        draw.text((W // 2, 180), "TIRES", font=font_main, fill=WHITE, anchor="mt")

        draw.line([(200, 265), (W - 200, 265)], fill=GOLD, width=2)

        draw.text((W // 2, 300), "ALL BRANDS • ALL SIZES", font=font_sub, fill=GOLD, anchor="mt")
        draw.text((W // 2, 355), "Wholesale Pricing • Professional Installation", font=font_body, fill=LIGHT_GRAY, anchor="mt")

        # Brand list in two columns
        brands_left = ["Pirelli", "Michelin", "Continental", "Bridgestone", "Goodyear"]
        brands_right = ["Hankook", "Toyo", "Falken", "General Tire", "Ironman"]

        y_start = 430
        for i, (bl, br) in enumerate(zip(brands_left, brands_right)):
            y = y_start + i * 45
            draw.text((250, y), f"✦  {bl}", font=font_list, fill=WHITE)
            draw.text((650, y), f"✦  {br}", font=font_list, fill=WHITE)

        # Sizes section
        draw.line([(200, 680), (W - 200, 680)], fill=(60, 60, 70), width=1)

        sizes_popular = [
            "15\" – 22\" RIM SIZES IN STOCK",
            "Sedans • SUVs • Trucks • Sports Cars",
            "All-Season • All-Weather • Performance"
        ]
        for i, line in enumerate(sizes_popular):
            draw.text((W // 2, 720 + i * 45), line, font=font_body, fill=LIGHT_GRAY, anchor="mt")

        # Price range
        draw_rounded_rect(draw, (200, 870, W - 200, 960), radius=15, fill=(20, 60, 20), outline=PRICE_GREEN, width=2)
        draw.text((W // 2, 895), "FROM $149/TIRE", font=font_cta, fill=PRICE_GREEN, anchor="mt")
        draw.text((W // 2, 940), "Sets of 4 available • Volume discounts", font=get_font(20), fill=(120, 200, 120), anchor="mt")

        # CTA
        draw_rounded_rect(draw, (250, 1000, W - 250, 1070), radius=12, fill=GOLD)
        draw.text((W // 2, 1035), "MESSAGE FOR YOUR SIZE & QUOTE", font=get_font(28, bold=True), fill=DARK_BG, anchor="mm")

    elif variant == 2:
        # "NEW ARRIVALS • HUGE INVENTORY" listing
        font_main = get_font(60, bold=True)
        font_sub = get_font(36, bold=True)
        font_body = get_font(26)
        font_cta = get_font(32, bold=True)

        draw.text((W // 2, 80), "LUXURYLANE TIRES", font=font_main, fill=GOLD, anchor="mt")
        draw.line([(200, 150), (W - 200, 150)], fill=GOLD, width=2)

        draw.text((W // 2, 180), "VAUGHAN'S TIRE WAREHOUSE", font=font_sub, fill=WHITE, anchor="mt")
        draw.text((W // 2, 228), "Premium Tires at Wholesale Prices", font=font_body, fill=LIGHT_GRAY, anchor="mt")

        # Feature boxes
        features = [
            ("BRAND NEW", "Factory-sealed, full warranty"),
            ("WHOLESALE PRICING", "Save 30-50% vs retail dealers"),
            ("EXPERT INSTALLATION", "$40/tire — mount + balance"),
            ("ANY SIZE AVAILABLE", "If we don't have it, we'll get it"),
            ("FREE CONSULTATION", "Tell us your vehicle, we'll find the fit"),
        ]

        y_start = 300
        for i, (title, desc) in enumerate(features):
            y = y_start + i * 110
            draw_rounded_rect(draw, (80, y, W - 80, y + 95), radius=10, fill=CARD_BG, outline=(50, 50, 60), width=1)
            draw.text((130, y + 20), f"▸  {title}", font=get_font(28, bold=True), fill=GOLD)
            draw.text((160, y + 55), desc, font=get_font(22), fill=LIGHT_GRAY)

        # Sizes we carry
        draw.text((W // 2, 870), "POPULAR SIZES IN STOCK NOW:", font=get_font(24, bold=True), fill=WHITE, anchor="mt")
        sizes_row1 = "225/65R17  •  235/55R18  •  265/70R17  •  215/55R17"
        sizes_row2 = "235/45R18  •  245/40R18  •  255/45R19  •  225/50R17"
        draw.text((W // 2, 910), sizes_row1, font=get_font(22), fill=LIGHT_GRAY, anchor="mt")
        draw.text((W // 2, 940), sizes_row2, font=get_font(22), fill=LIGHT_GRAY, anchor="mt")

        # CTA
        draw_rounded_rect(draw, (200, 990, W - 200, 1070), radius=12, fill=GOLD)
        draw.text((W // 2, 1010), "TEXT YOUR VEHICLE INFO", font=font_cta, fill=DARK_BG, anchor="mt")
        draw.text((W // 2, 1048), "We'll send you options + pricing in minutes", font=get_font(22), fill=(40, 40, 50), anchor="mt")

    # Bottom bar
    draw.rectangle([(0, 1100), (W, H)], fill=(25, 25, 32))
    draw.rectangle([(0, 1100), (W, 1102)], fill=GOLD)
    font_loc = get_font(22)
    draw.text((W // 2, 1130), "Vaughan, ON  •  luxurylanetires.ca", font=font_loc, fill=LIGHT_GRAY, anchor="mt")
    draw.text((W // 2, 1165), "Professional Installation On-Site", font=get_font(20), fill=GOLD, anchor="mt")
    draw.rectangle([(0, H - 5), (W, H)], fill=GOLD)

    filename = f"general_warehouse_{variant}.jpg"
    filepath = os.path.join(IMG_DIR, filename)
    img.convert("RGB").save(filepath, "JPEG", quality=95)
    print(f"  Saved: {filename}")
    return filepath


if __name__ == "__main__":
    print("Generating 9 tire listing images...")
    for i, tire in enumerate(TIRES):
        print(f"\n[{i+1}/9] {tire['brand']} {tire['model']} - {tire['size']}")
        create_tire_listing(tire, i)

    print("\nGenerating 2 general warehouse images...")
    create_general_warehouse(1)
    create_general_warehouse(2)

    print("\n✓ All images generated in:", IMG_DIR)
