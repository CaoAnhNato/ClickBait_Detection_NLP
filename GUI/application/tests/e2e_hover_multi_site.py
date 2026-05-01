#!/usr/bin/env python3
"""
Browser E2E test: hover news titles and verify tooltip predictions.
Targets: BBC News, The Guardian, Reuters, CNN, NYTimes.
Uses the already-running ORCD server at http://127.0.0.1:8000.
Proxies backend calls via Playwright route to avoid WSL2 localhost issues.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

EXTENSION_DIR = Path(__file__).resolve().parents[1] / "chrome_extension"
APP_DIR = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

NEWS_SITES = [
    ("The Guardian", "https://www.theguardian.com/international"),
    ("NY Times", "https://www.nytimes.com/international/"),
    ("Al Jazeera", "https://www.aljazeera.com/"),
]

SELECTORS_BY_SITE = {
    "The Guardian": [
        "h3.card-headline",
        "h2.card-headline",
        ".fc-item__headline",
        "[class*='headline' i]",
        "h3 a",
        "h2 a",
        "article h2",
        "article h3",
        "h3",
        "h2",
    ],
    "Reuters": [
        "article h2",
        "article h3",
        "[class*='headline' i]",
        ".article-title",
        "h2 a",
    ],
    "CNN": [
        "article h2",
        "article h3",
        "[class*='headline' i]",
        ".article-title",
        ".card-title",
        "h2",
    ],
    "NY Times": [
        "article h2",
        "article h3",
        "[class*='headline' i]",
        ".article-title",
        "h2",
    ],
    "Al Jazeera": [
        "article h2",
        "article h3",
        ".article-card__title",
        "[class*='headline' i]",
        "[class*='title' i]",
        "h2",
    ],
    "NPR News": [
        "article h2",
        "[class*='headline' i]",
        "[class*='title' i]",
        "h2",
        "h3",
    ],
    "BBC News": [
        "article h2",
        "article h3",
        "[data-testid='card-headline']",
        "[class*='headline' i]",
        ".article-title",
        "h2",
    ],
}

ALL_SELECTORS = [
    "article h2 a",
    "article h3 a",
    "h1 a",
    "h2 a",
    "h3 a",
    "[class*='headline' i] a",
    "[class*='story-title' i] a",
    "[class*='card-title' i] a",
    "[class*='article-title' i] a",
    "[class*='title' i] a",
]


def find_best_title(page, site_name: str) -> tuple[Any, str] | None:
    """Find a visible news title element, preferring the site's specific selectors."""
    selectors = SELECTORS_BY_SITE.get(site_name, []) + ALL_SELECTORS
    seen = set()

    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = locator.count()
            for idx in range(min(count, 100)):
                el = locator.nth(idx)
                try:
                    text = (el.inner_text(timeout=1500) or "").strip()
                    text_clean = re.sub(r"\s+", " ", text)
                    if len(text_clean) < 15:
                        continue
                    if text_clean in seen:
                        continue
                    seen.add(text_clean)
                    box = el.bounding_box()
                    if not box or box["width"] < 80 or box["height"] < 15:
                        continue
                    is_nav_or_footer = el.evaluate(
                        "el => !!el.closest("
                        "'nav, footer, [role=navigation], [role=contentinfo], "
                        "[role=banner], .cookie-banner, .ad, .advertisement, "
                        ".sidebar, .breaking-news-label, #nav, .nav')"
                    )
                    if is_nav_or_footer:
                        continue
                    # Skip element if text is just nav/section labels
                    skip_words = {
                        "more top stories", "more stories", "breaking news", "latest news",
                        "top stories", "in case you missed it", "trending now", "more from",
                        "inside burundi's refugee camps – in pictures",
                        "iron maiden 'i nearly quit to become a fencing teacher'",
                    }
                    if text_clean.lower() in skip_words:
                        continue
                    # Scroll element into view
                    el.scroll_into_view_if_needed()
                    page.wait_for_timeout(300)
                    return el, text_clean[:200]
                except PlaywrightTimeoutError:
                    continue
                except Exception:
                    continue
        except Exception:
            continue
    return None


def install_predict_proxy_route(page) -> None:
    """Install a route that proxies /predict calls to the real backend."""
    proxy_url = BASE_URL.rstrip("/")

    def handle_predict(route, request):
        try:
            payload = request.post_data_json or {}
            resp = requests.post(f"{proxy_url}/predict", json=payload, timeout=60)
            route.fulfill(
                status=resp.status_code,
                headers={"Content-Type": "application/json"},
                body=resp.text,
            )
        except Exception as exc:
            route.fulfill(
                status=503,
                headers={"Content-Type": "application/json"},
                body=json.dumps({"detail": f"Proxy error: {exc}"}),
            )

    page.route("**/__cbd_predict", handle_predict)


def inject_content_script(page) -> None:
    """Inject extension CSS and JS with the proxy endpoint."""
    js_path = EXTENSION_DIR / "content.js"
    css_path = EXTENSION_DIR / "style.css"

    if css_path.exists():
        page.add_style_tag(content=css_path.read_text(encoding="utf-8"))

    if js_path.exists():
        js_text = js_path.read_text(encoding="utf-8")
        # Handle both old format (with semicolon) and new format (without)
        for old_endpoint in [
            'const API_ENDPOINT = "http://127.0.0.1:8000/predict";',
            'const API_ENDPOINT = "http://127.0.0.1:8000/predict"',
        ]:
            if old_endpoint in js_text:
                js_text = js_text.replace(old_endpoint, 'const API_ENDPOINT = "/__cbd_predict";')
                break
        page.add_script_tag(content=js_text)


def run_test_for_site(page, site_name: str, url: str) -> dict[str, Any]:
    print(f"\n--- Testing {site_name} ({url}) ---")
    result = {
        "site": site_name,
        "url": url,
        "success": False,
        "titles_found": [],
        "predictions": [],
        "errors": [],
    }

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)

        found = find_best_title(page, site_name)
        if not found:
            result["errors"].append("No visible title element found")
            return result

        target, title_text = found
        result["titles_found"].append(title_text)
        print(f"  Hovering: {title_text[:80]}...")

        # Inject proxy + content script BEFORE hovering so event listeners are active
        install_predict_proxy_route(page)
        inject_content_script(page)
        page.wait_for_timeout(600)

        hover_start = time.perf_counter()
        target.hover(force=True)

        tooltip_title_locator = page.locator("#cbd-hover-tooltip .cbd-title")
        tooltip_detail_locator = page.locator("#cbd-hover-tooltip .cbd-detail")

        # Wait for Analyzing state
        analyzing_deadline = time.perf_counter() + 15.0
        analyzing_started = None
        while time.perf_counter() < analyzing_deadline:
            try:
                current = (tooltip_title_locator.inner_text(timeout=300) or "").strip()
            except Exception:
                current = ""
            if "nalyzing" in current.lower():
                analyzing_started = time.perf_counter()
                print(f"  [Analyzing detected after {round((analyzing_started-hover_start)*1000)}ms]")
                break
            page.wait_for_timeout(150)

        if analyzing_started is None:
            result["errors"].append("Tooltip never entered 'Analyzing' state")
            return result

        # Wait for prediction output
        output_deadline = time.perf_counter() + 240.0
        output_ready = None
        final_tt = ""
        while time.perf_counter() < output_deadline:
            try:
                tt = (tooltip_title_locator.inner_text(timeout=400) or "").strip()
            except Exception:
                tt = ""
            if tt and "nalyzing" not in tt.lower():
                output_ready = time.perf_counter()
                final_tt = tt
                break
            page.wait_for_timeout(200)

        if output_ready is None:
            result["errors"].append("Tooltip timed out waiting for prediction")
            return result

        try:
            tooltip_detail = (tooltip_detail_locator.inner_text(timeout=300) or "").strip()
        except Exception:
            tooltip_detail = ""

        is_valid_result = bool(
            re.search(r"%\s*(suspicious clickbait|genuine news|error)", final_tt, flags=re.IGNORECASE)
        )
        print(f"  Result: {final_tt}")
        print(f"  Detail: {tooltip_detail}")
        print(f"  Latency: hover->analyzing={round((analyzing_started-hover_start)*1000)}ms, analyzing->output={round((output_ready-analyzing_started)*1000)}ms, total={round((output_ready-hover_start)*1000)}ms")

        result["success"] = True
        result["predictions"].append({
            "title": title_text,
            "tooltip_title": final_tt,
            "tooltip_detail": tooltip_detail,
            "is_valid_result": is_valid_result,
            "latency_ms": {
                "hover_to_analyzing": round((analyzing_started - hover_start) * 1000, 2),
                "analyzing_to_output": round((output_ready - analyzing_started) * 1000, 2),
                "hover_to_output": round((output_ready - hover_start) * 1000, 2),
            },
        })

    except PlaywrightTimeoutError as exc:
        result["errors"].append(f"Page timeout: {str(exc)[:200]}")
    except Exception as exc:
        result["errors"].append(str(exc)[:200])

    return result


def main() -> None:
    if not EXTENSION_DIR.exists():
        print(f"ERROR: Extension directory not found: {EXTENSION_DIR}")
        sys.exit(1)

    report_path = APP_DIR / "tests" / "artifacts" / "hover_multi_site_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ORCD Multi-Site Hover E2E Test")
    print("Backend:", BASE_URL)
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-extensions-except",
                f"--load-extension={EXTENSION_DIR}",
                "--no-sandbox",
                "--disable-web-security",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
        )
        # Anti-detection: hide automation signals that trigger bot protections
        ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
        """)
        page = ctx.new_page()

        all_results = []
        for site_name, url in NEWS_SITES:
            result = run_test_for_site(page, site_name, url)
            all_results.append(result)
            page.wait_for_timeout(2000)

        ctx.close()

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "sites_tested": len(NEWS_SITES),
        "results": all_results,
        "summary": {
            "total": len(all_results),
            "with_tooltip": sum(1 for r in all_results if r["predictions"]),
            "valid_predictions": sum(1 for r in all_results if r.get("predictions") and r["predictions"][-1]["is_valid_result"]),
            "failed": sum(1 for r in all_results if r["errors"] and not r["predictions"]),
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n\nReport saved to: {report_path}")
    print("\nSummary:")
    print(json.dumps(report["summary"], indent=2))

    for r in all_results:
        if r["predictions"]:
            p = r["predictions"][-1]
            print(f"  [{r['site']}] {p['tooltip_title'][:60]}")
        else:
            print(f"  [{r['site']}] FAIL: {r['errors'][0] if r['errors'] else 'no result'}")


if __name__ == "__main__":
    main()
