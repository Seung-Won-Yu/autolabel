import json
import sys

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:4174"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "latest"


def inspect_page(browser, name, viewport, color_scheme, reduced_motion="no-preference"):
    page = browser.new_page(
        viewport=viewport,
        color_scheme=color_scheme,
        reduced_motion=reduced_motion,
    )
    console_errors = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: console_errors.append(str(error)))
    page.goto(BASE_URL, wait_until="networkidle")
    page.wait_for_timeout(500)
    page.evaluate(
        """
        async () => {
          const step = Math.max(320, Math.floor(innerHeight * 0.72))
          for (let y = 0; y < document.documentElement.scrollHeight; y += step) {
            scrollTo(0, y)
            await new Promise((resolve) => setTimeout(resolve, 45))
          }
          scrollTo(0, 0)
          await new Promise((resolve) => setTimeout(resolve, 180))
        }
        """
    )
    page.wait_for_timeout(1000)

    result = page.evaluate(
        """
        () => {
          const ids = new Set([...document.querySelectorAll('[id]')].map((el) => el.id))
          const anchors = [...document.querySelectorAll('a[href^="#"]')].map((el) => el.getAttribute('href'))
          const invalidAnchors = anchors.filter((href) => href !== '#' && !ids.has(href.slice(1)))
          const visibleReveals = [...document.querySelectorAll('.reveal, tbody')]
            .filter((el) => el.classList.contains('in')).length
          return {
            title: document.title,
            h1: document.querySelector('h1')?.innerText,
            landmarks: {
              nav: document.querySelectorAll('nav').length,
              main: document.querySelectorAll('main').length,
              header: document.querySelectorAll('header').length,
              footer: document.querySelectorAll('footer').length,
            },
            skipLink: Boolean(document.querySelector('.skip-link[href="#content"]')),
            links: document.links.length,
            anchors: anchors.length,
            invalidAnchors,
            scrollWidth: document.documentElement.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
            horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
            revealCount: document.querySelectorAll('.reveal, tbody').length,
            visibleReveals,
          }
        }
        """
    )

    page.keyboard.press("Tab")
    skip_link = page.locator(".skip-link")
    skip_box = skip_link.bounding_box()
    skip_active = skip_link.evaluate("(el) => document.activeElement === el")
    result["skipLinkActive"] = skip_active
    result["skipLinkY"] = None if skip_box is None else round(skip_box["y"], 1)
    result["skipLinkKeyboard"] = (
        skip_active
        and skip_box is not None
        and skip_box["y"] >= 0
    )

    slider = page.get_by_role("slider")
    if slider.count():
        slider.focus()
        before = slider.get_attribute("aria-valuenow")
        page.keyboard.press("ArrowRight")
        after = slider.get_attribute("aria-valuenow")
        result["sliderKeyboard"] = before != after

    page.screenshot(
        path=f".design/landing-page/screenshots/{LABEL}-{name}-fold.png",
        full_page=False,
    )
    page.screenshot(
        path=f".design/landing-page/screenshots/{LABEL}-{name}.png",
        full_page=True,
    )
    result["consoleErrors"] = console_errors
    page.close()
    return result


def inspect_without_javascript(browser):
    context = browser.new_context(
        viewport={"width": 390, "height": 844},
        color_scheme="dark",
        java_script_enabled=False,
    )
    page = context.new_page()
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    visible_headings = sum(1 for heading in page.locator("h2").all() if heading.is_visible())
    page.screenshot(
        path=f".design/landing-page/screenshots/{LABEL}-no-js-mobile.png",
        full_page=True,
    )
    context.close()
    return {"visibleHeadings": visible_headings, "expectedHeadings": 8}


with sync_playwright() as playwright:
    chromium = playwright.chromium.launch(headless=True)
    report = {
        "desktopDark": inspect_page(chromium, "desktop-dark", {"width": 1440, "height": 900}, "dark"),
        "desktopLight": inspect_page(chromium, "desktop-light", {"width": 1440, "height": 900}, "light"),
        "mobileDark": inspect_page(chromium, "mobile-dark", {"width": 390, "height": 844}, "dark"),
        "mobileNarrow": inspect_page(chromium, "mobile-narrow", {"width": 320, "height": 720}, "dark"),
        "mobileReducedMotion": inspect_page(
            chromium, "mobile-reduced-motion", {"width": 390, "height": 844}, "dark", "reduce"
        ),
        "noJavaScript": inspect_without_javascript(chromium),
    }
    chromium.close()

print(json.dumps(report, ensure_ascii=False, indent=2))
