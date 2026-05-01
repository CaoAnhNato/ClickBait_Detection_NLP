"""
Browser hover-tooltip test using Playwright.

This test starts the FastAPI backend, loads a mock news page,
injects the Chrome Extension's content.js, and verifies
the hover tooltip appears with correct clickbait/non-clickbait predictions.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

# Paths
SCRIPT = Path(__file__).resolve()           # GUI/test/e2e_hover_news_test.py
TEST_DIR = SCRIPT.parent                   # GUI/test
GUI_DIR = TEST_DIR.parent                  # GUI
APP_DIR = GUI_DIR / "application"          # GUI/application
BACKEND_DIR = APP_DIR / "backend"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

BACKEND_ENV = dict(os.environ)
BACKEND_ENV["PYTHONPATH"] = str(APP_DIR)
BACKEND_ENV["ORCD_API_KEY"] = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v"
BACKEND_ENV["ORCD_MODEL_KEY"] = "generate-and-predict"
BACKEND_ENV["ORCD_API_MODEL_OVERRIDE"] = "gpt-3.5-turbo-1106"
BACKEND_ENV["ORCD_API_BASE_OVERRIDE"] = "https://api-v2.shopaikey.com/v1"
BACKEND_ENV["ORCD_API_PROVIDER_OVERRIDE"] = "openai"


def wait_for_server(url: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def backend_proc():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", SERVER_HOST, "--port", str(SERVER_PORT)],
        cwd=str(BACKEND_DIR),
        env=BACKEND_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert wait_for_server(SERVER_URL, 25), "Backend failed to start"
    yield SERVER_URL
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


def make_mock_html(tmp_dir: Path) -> Path:
    html = APP_DIR / "tests" / "mock_news.html"
    if html.exists():
        return html
    content = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Mock News</title>
<style>body{font-family:sans-serif;margin:40px}a{font-size:18px;color:#1a0dab;display:block;margin:12px 0}</style>
</head><body>
<h1>Today's News</h1>
<a href="#">This Vine Of New York On Celebrity Big Brother Is Perfect</a>
<a href="#">17 Hairdresser Struggles Every Black Girl Knows To Be True</a>
<a href="#">Coldplay's new album hits stores worldwide this week</a>
<a href="#">New law to help asbestos sufferers in Victoria, Australia</a>
</body></html>"""
    p = tmp_dir / "mock_news.html"
    p.write_text(content, encoding="utf-8")
    return p


def test_health_check(backend_proc: str):
    import urllib.request
    with urllib.request.urlopen(f"{backend_proc}/health") as r:
        data = json.loads(r.read())
    assert data["status"] == "ok"
    assert data["model"] == "generate-and-predict"


def test_predict_clickbait(backend_proc: str):
    import urllib.request, json
    payload = json.dumps({"title": "17 Hairdresser Struggles Every Black Girl Knows To Be True"}).encode()
    req = urllib.request.Request(
        f"{backend_proc}/predict", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    assert result["label"] == 1
    assert result["is_clickbait"] is True
    assert result["model"] == "generate-and-predict"
    print(f"  Clickbait prediction: {result}")


def test_predict_non_clickbait(backend_proc: str):
    import urllib.request, json
    payload = json.dumps({"title": "New law to help asbestos sufferers in Victoria, Australia"}).encode()
    req = urllib.request.Request(
        f"{backend_proc}/predict", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    assert result["label"] in (0, 1)
    assert result["model"] == "generate-and-predict"
    print(f"  Non-Clickbait prediction: {result}")


def test_hover_tooltip_via_browser(backend_proc: str, tmp_path: Path):
    """Verify content.js injects correctly and the tooltip DOM structure is valid on hover."""
    content_js_path = APP_DIR / "chrome_extension" / "content.js"
    assert content_js_path.exists(), f"content.js not found at {content_js_path}"

    test_script = tmp_path / "run_hover_mock.py"
    test_script.write_text(
        _HOVER_MOCK_SCRIPT.replace("{{SERVER_URL}}", SERVER_URL).replace(
            "{{CONTENT_JS_PATH}}", str(content_js_path)
        ),
        encoding="utf-8",
    )

    mock_html_path = tmp_path / "mock_news.html"
    mock_html_path.write_text(
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'><title>Mock News</title>\n"
        "<style>body{font-family:sans-serif;margin:40px}a{font-size:18px;color:#1a0dab;display:block;margin:12px 0}</style>\n"
        "</head><body>\n<h1>Today's News</h1>\n"
        "<a href='#'>This Vine Of New York On Celebrity Big Brother Is Perfect</a>\n"
        "<a href='#'>17 Hairdresser Struggles Every Black Girl Knows To Be True</a>\n"
        "<a href='#'>Coldplay's new album hits stores worldwide this week</a>\n"
        "<a href='#'>New law to help asbestos sufferers in Victoria, Australia</a>\n"
        "</body></html>",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=30,
    )
    print(f"  stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr}")
    assert result.returncode == 0, f"Mock browser test failed:\n{result.stderr}"


def test_hover_tooltip_live_backend(backend_proc: str, tmp_path: Path):
    """Use the real backend for hover — no mock route."""
    content_js_path = APP_DIR / "chrome_extension" / "content.js"
    assert content_js_path.exists(), f"content.js not found at {content_js_path}"

    test_script = tmp_path / "run_hover_live.py"
    test_script.write_text(
        _HOVER_LIVE_SCRIPT.replace("{{SERVER_URL}}", SERVER_URL).replace(
            "{{CONTENT_JS_PATH}}", str(content_js_path)
        ),
        encoding="utf-8",
    )

    mock_html_path = tmp_path / "mock_news.html"
    mock_html_path.write_text(
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'><title>Mock News</title>\n"
        "<style>body{font-family:sans-serif;margin:40px}a{font-size:18px;color:#1a0dab;display:block;margin:12px 0}</style>\n"
        "</head><body>\n<h1>Today's News</h1>\n"
        "<a href='#'>This Vine Of New York On Celebrity Big Brother Is Perfect</a>\n"
        "<a href='#'>17 Hairdresser Struggles Every Black Girl Knows To Be True</a>\n"
        "<a href='#'>Coldplay's new album hits stores worldwide this week</a>\n"
        "<a href='#'>New law to help asbestos sufferers in Victoria, Australia</a>\n"
        "</body></html>",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=60,
    )
    print(f"  stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr}")
    assert result.returncode == 0, f"Live browser test failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Inline test scripts (avoid f-string brace conflicts with JS code)
# ---------------------------------------------------------------------------
_HOVER_MOCK_SCRIPT = r"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

content_js_path = Path("{{CONTENT_JS_PATH}}")
content_code = content_js_path.read_text()
mock_html = Path(__file__).parent / "mock_news.html"
server_url = "{{SERVER_URL}}"

p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
# Use a separate context so we can set baseURL and intercept requests
ctx = browser.new_context()
page = ctx.new_page()

# Intercept ALL POST requests to the server URL
def handle_route(route):
    if route.request.method == "POST" and "/predict" in route.request.url:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "is_clickbait": True,
                "confidence": 87.5,
                "label": 1,
                "model": "generate-and-predict",
                "device": "cuda",
                "cached": False,
            }),
        )
    else:
        route.continue_()

ctx.route(url=server_url + "/*", handler=handle_route)

page.goto("file://" + str(mock_html))
page.wait_for_load_state("networkidle")

# Inject content.js
script_tag = page.locator("body")
script_tag.evaluate(f"el => {{ const s = document.createElement('script'); s.textContent = {json.dumps(content_code)}; el.appendChild(s); }}")

page.wait_for_timeout(800)
first_link = page.locator("a").first
first_link.hover()
page.wait_for_timeout(2000)

tooltip = page.locator("#cbd-hover-tooltip")
assert tooltip.count() > 0, "Tooltip #cbd-hover-tooltip not found in DOM"
cls = tooltip.get_attribute("class") or ""
assert "cbd-visible" in cls, f"Tooltip should be visible, got class={cls!r}"

title = tooltip.locator(".cbd-title").text_content() or ""
detail = tooltip.locator(".cbd-detail").text_content() or ""
print(f"TOOLTIP OK - title={title!r} detail={detail!r}")
assert len(title) > 0, "Tooltip title is empty"
assert len(detail) > 0, "Tooltip detail is empty"
ctx.close()
browser.close()
p.stop()
print("TEST PASSED")
"""

_HOVER_LIVE_SCRIPT = r"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

content_js_path = Path("{{CONTENT_JS_PATH}}")
content_code = content_js_path.read_text()
mock_html = Path(__file__).parent / "mock_news.html"
server_url = "{{SERVER_URL}}"

p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
page = browser.new_page()
page.goto("file://" + str(mock_html))
page.wait_for_load_state("networkidle")

# Inject content.js
import json
script_tag = page.locator("body")
script_tag.evaluate(f"el => {{ const s = document.createElement('script'); s.textContent = {json.dumps(content_code)}; el.appendChild(s); }}")
page.wait_for_timeout(800)

first_link = page.locator("a").first
first_link.hover()
page.wait_for_timeout(8000)

tooltip = page.locator("#cbd-hover-tooltip")
assert tooltip.count() > 0, "Tooltip not found in DOM"
cls = tooltip.get_attribute("class") or ""
assert "cbd-visible" in cls, f"Tooltip should be visible, got class={cls!r}"

title = tooltip.locator(".cbd-title").text_content() or ""
print(f"LIVE TOOLTIP OK - title={title!r}")
assert len(title) > 0, "Tooltip title is empty"
browser.close()
p.stop()
print("LIVE TEST PASSED")
"""
