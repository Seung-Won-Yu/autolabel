from playwright.sync_api import sync_playwright


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(
        viewport={"width": 1200, "height": 630},
        color_scheme="dark",
        reduced_motion="reduce",
        device_scale_factor=1,
    )
    page.goto("http://127.0.0.1:4174", wait_until="networkidle")
    page.add_style_tag(
        content="""
        html, body { width: 1200px; height: 630px; overflow: hidden !important; }
        nav { position: relative; }
        nav .links { display: none; }
        nav .wrap { max-width: 1080px; }
        .hero { min-height: 572px; padding-top: 38px; }
        .hero::before { height: 572px; }
        .hero-flow { margin-bottom: 12px; }
        h1 { font-size: 80px; }
        .tagline { margin-top: 16px; font-size: 18px; line-height: 1.45; }
        .badges { margin-top: 14px; }
        .cta, .stage { display: none; }
        .band { margin-top: 26px; padding-block: 14px; }
        .band b { font-size: 29px; }
        .band span { font-size: 12px; }
        """
    )
    page.screenshot(path="docs/og.png")
    browser.close()
