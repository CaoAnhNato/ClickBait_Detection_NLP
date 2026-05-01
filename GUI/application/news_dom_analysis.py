from playwright.sync_api import sync_playwright
import json
import time

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"

def get_css_selector(element):
    """Generate a full CSS selector path for an element"""
    path = []
    current = element
    while current:
        selector = current.evaluate("""
            el => {
                if (el.id) return '#' + el.id;
                let path = el.tagName.toLowerCase();
                if (el.className && typeof el.className === 'string') {
                    path += '.' + el.className.split(' ')[0];
                }
                return path;
            }
        """)
        path.insert(0, selector)
        try:
            parent = current.evaluate_handle("el => el.parentElement")
            current = parent.as_element() if parent else None
        except:
            break
    return ' > '.join(path)

def get_element_html(element):
    """Get outerHTML of an element"""
    return element.evaluate("el => el.outerHTML")

def get_element_text(element):
    """Get text content of an element"""
    return element.evaluate("el => el.textContent.trim()")

def get_bounding_box(element):
    """Get bounding box of an element"""
    return element.evaluate("""el => {
        const rect = el.getBoundingClientRect();
        return {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height)
        };
    }""")

def get_tag_name(element):
    """Get tag name of an element"""
    return element.evaluate("el => el.tagName")

def accept_cookies_if_present(page):
    """Try to accept cookies on various news sites"""
    time.sleep(1)
    
    for selector in [
        '[aria-label="Accept all"]',
        '[data-testid="accept-terms"]',
        '#onetrust-accept-btn-handler',
        '[aria-label="Accept Cookies"]',
        '.accept-cookies-button'
    ]:
        try:
            btn = page.query_selector(selector)
            if btn and btn.is_visible():
                btn.click()
                time.sleep(0.5)
                print(f"    [Accepted cookies: {selector}]")
                return
        except:
            pass

def analyze_page(page, url):
    """Analyze a news homepage and extract article card structure"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {url}")
    print('='*80)
    
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        time.sleep(3)
        accept_cookies_if_present(page)
        time.sleep(1)
        
        print("\n--- STEP 1: Finding article card structure ---")
        
        article_found = False
        
        for pattern in [
            '[data-gu-section="news"]',
            '[data-component="card"]',
            '.fc-item',
            '.article-card',
            '[class*="article"]',
            '[class*="card"]',
            'section > ul > li',
            '.story',
            '.post',
            'article'
        ]:
            try:
                elems = page.query_selector_all(pattern)
                if len(elems) > 0:
                    print(f"\nFound {len(elems)} elements with selector: {pattern}")
                    first = elems[0]
                    html = get_element_html(first)
                    print(f"\nFirst element HTML (first 1500 chars):")
                    print(html[:1500])
                    print("\n" + "-"*60)
                    
                    links = first.query_selector_all('a')
                    headers = first.query_selector_all('h1, h2, h3, h4')
                    print(f"\nInside first article card:")
                    print(f"  - Links: {len(links)}")
                    print(f"  - Headers: {len(headers)}")
                    
                    for h in headers[:2]:
                        try:
                            tag = get_tag_name(h)
                            print(f"\n  Header <{tag}>:")
                            print(f"    class: {h.get_attribute('class') or ''}")
                            print(f"    text: {get_element_text(h)[:100]}")
                            print(f"    selector: {get_css_selector(h)}")
                        except Exception as e:
                            print(f"  Header error: {e}")
                    
                    for a in links[:3]:
                        text = get_element_text(a)
                        if len(text) > 10:
                            try:
                                print(f"\n  Link <a>:")
                                print(f"    href: {(a.get_attribute('href') or '')[:80]}")
                                print(f"    class: {a.get_attribute('class') or ''}")
                                print(f"    text: {text[:100]}")
                                print(f"    selector: {get_css_selector(a)}")
                            except Exception as e:
                                print(f"  Link error: {e}")
                    
                    article_found = True
                    break
            except Exception as e:
                print(f"  Selector error: {e}")
                continue
        
        if not article_found:
            print("\nNo article cards found with standard selectors.")
            main = page.query_selector('main') or page.query_selector('[role="main"]')
            if main:
                headers = main.query_selector_all('h1, h2, h3')[:5]
                print(f"\nMain content area headers:")
                for h in headers:
                    try:
                        tag = get_tag_name(h)
                        print(f"  <{tag}>: {get_element_text(h)[:80]}")
                    except: pass
        
        print("\n--- STEP 2: Finding hoverable title elements ---")
        
        all_anchors = page.query_selector_all('a')
        anchor_candidates = []
        
        for a in all_anchors:
            text = get_element_text(a)
            words = text.split()
            if len(words) >= 5 and len(text) > 30:
                try:
                    bbox = get_bounding_box(a)
                    if bbox['width'] > 50 and bbox['height'] > 20:
                        anchor_candidates.append({
                            'element': a,
                            'text': text,
                            'words': len(words),
                            'bbox': bbox
                        })
                except:
                    pass
        
        print(f"\nAnchor tags with >= 5 words, visible (w>50, h>20): {len(anchor_candidates)}")
        print("\nFirst 3 candidates (full CSS selector + text):")
        for i, candidate in enumerate(anchor_candidates[:3]):
            try:
                selector = get_css_selector(candidate['element'])
                print(f"\n  Candidate {i+1}:")
                print(f"    CSS Selector: {selector}")
                print(f"    Text: {candidate['text'][:150]}")
                print(f"    Words: {candidate['words']}")
                print(f"    BBox: x={candidate['bbox']['x']}, y={candidate['bbox']['y']}, w={candidate['bbox']['width']}, h={candidate['bbox']['height']}")
            except Exception as e:
                print(f"  Error: {e}")
        
        all_headers = page.query_selector_all('h1, h2, h3')
        header_candidates = []
        for h in all_headers:
            try:
                bbox = get_bounding_box(h)
                if bbox['width'] > 100:
                    header_candidates.append({
                        'element': h,
                        'text': get_element_text(h),
                        'bbox': bbox
                    })
            except:
                pass
        
        print(f"\nVisible Header tags (h1, h2, h3): {len(header_candidates)}")
        print("\nFirst 3 headers (full CSS selector + text):")
        for i, h in enumerate(header_candidates[:3]):
            try:
                tag = get_tag_name(h['element'])
                selector = get_css_selector(h['element'])
                print(f"\n  Header {i+1}:")
                print(f"    CSS Selector: {selector}")
                print(f"    Text: {h['text'][:150]}")
                print(f"    BBox: x={h['bbox']['x']}, y={h['bbox']['y']}, w={h['bbox']['width']}, h={h['bbox']['height']}")
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n--- STEP 3: First 3 visible <a> tags with textContent > 15 chars ---")
        
        visible_anchors = []
        for a in all_anchors:
            try:
                text = get_element_text(a)
                if len(text) > 15:
                    bbox = get_bounding_box(a)
                    if bbox['width'] > 30 and bbox['height'] > 10:
                        visible_anchors.append({'element': a, 'text': text, 'bbox': bbox})
                        if len(visible_anchors) >= 3:
                            break
            except:
                pass
        
        for i, item in enumerate(visible_anchors):
            print(f"\n  Anchor {i+1}:")
            html = get_element_html(item['element'])
            print(f"    outerHTML:")
            print(f"    {html[:600]}")
            if len(html) > 600:
                print("    ... [truncated]")
            
            print(f"    Bounding Box: x={item['bbox']['x']}, y={item['bbox']['y']}, width={item['bbox']['width']}, height={item['bbox']['height']}")
            print(f"    textContent: {item['text'][:100]}")
            
            try:
                print(f"    CSS Selector: {get_css_selector(item['element'])}")
            except:
                pass
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"Error analyzing {url}: {e}")
        import traceback
        traceback.print_exc()

def main():
    urls = [
        "https://www.theguardian.com/international",
        "https://www.nytimes.com/international/",
        "https://www.aljazeera.com/"
    ]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-web-security",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        ctx = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900}
        )
        
        ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
        """)
        
        page = ctx.new_page()
        page.set_default_timeout(60000)
        
        for url in urls:
            analyze_page(page, url)
            time.sleep(2)
        
        browser.close()

if __name__ == "__main__":
    main()
