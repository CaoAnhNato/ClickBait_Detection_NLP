(() => {
  const API_ENDPOINT = "http://127.0.0.1:8000/predict";
  const HOVER_DELAY_MS = 500;
  const REQUEST_TIMEOUT_MS = 45000;
  const MESSAGE_TIMEOUT_MS = 8000;
  const MESSAGE_RETRY_COUNT = 1;
  const MAX_TITLE_LENGTH = 512;
  const MIN_TITLE_LENGTH = 8;
  const MAX_CACHE_ENTRIES = 500;

  // Tier 1: Semantic HTML5 / ARIA headings
  // Tier 2: Class-pattern matches
  // Tier 3: Links with news/story keywords
  const SELECTORS = [
    "h1",
    "h2",
    "h3",
    "article h1",
    "article h2",
    "article h3",
    "[role='heading']",
    "[itemprop='headline']",
    "[itemprop='name']",
    "[class*='title' i]",
    "[class*='headline' i]",
    "[class*='card-title' i]",
    "[class*='story-title' i]",
    "[class*='entry-title' i]",
    "[class*='post-title' i]",
    "[class*='news-title' i]",
    "[class*='article-title' i]",
    "[class*='section-title' i]",
    "[class*='feature-title' i]",
    "[class*='breaking-title' i]",
    "a[href][class*='title']",
    "a[href][class*='headline']",
    "a[href][class*='news']",
    "a[href][class*='story']",
    "a[href][class*='article']",
  ];

  // Elements that should NOT be the target for text extraction
  // (they appear inside card/article wrappers and carry non-title text)
  const BAD_TEXT_SELECTORS = [
    "script", "style", "noscript", "iframe", "svg",
    "button", "input", "select", "textarea",
    // Byline / meta elements
    ".byline", ".author", "[class*='byline' i]", "[class*='author' i]",
    ".timestamp", ".time", "[class*='timestamp' i]", "[class*='date' i]",
    ".section", "[class*='section-label' i]",
    ".kicker", "[class*='kicker' i]",
    ".tag", "[class*='tag' i]",
    "time",
    // Images
    "img",
    // Icons / media
    "[class*='icon' i]", "[class*='svg-icon' i]",
    ".media-icon",
  ];

  // Elements that are LIKELY to contain the actual headline text
  const TITLE_TEXT_SELECTORS = [
    "h1", "h2", "h3", "h4",
    "[itemprop='headline']",
    "[itemprop='name']",
    "[class*='headline' i]",
    "[class*='card-title' i]",
    "[class*='story-title' i]",
    "[class*='entry-title' i]",
    "[class*='post-title' i]",
    "[class*='news-title' i]",
    "[class*='article-title' i]",
    "[class*='section-title' i]",
    "[class*='feature-title' i]",
    "[class*='breaking-title' i]",
    "[class*='title-text' i]",
    "[class*='item-title' i]",
    // Specific news site headline classes
    ".headline",
    ".card-headline",
    ".fc-item__headline",
    "[data-testid='card-headline']",
    "[data-testid='headline']",
    // Guardian-specific
    "[class*='headline-text' i]",
    ".tickerHeadline",
  ];

  const BAD_ANCESTOR_SELECTOR = "nav, footer, [role='navigation'], [role='banner'], [role='contentinfo'], .nav, #nav, .sidebar, .ad, .advertisement, [class*='ad-' i], [id*='ad-' i]";

  let hoverTimer = null;
  let activeTarget = null;
  let activeTitle = "";
  let activeController = null;
  let tooltipEl = null;
  let titleEl = null;
  let detailEl = null;
  const pointer = { x: 0, y: 0 };
  const localCache = new Map();

  // ─── Element filtering helpers ───────────────────────────────────────────

  function isVisible(element) {
    if (!element) return false;
    const style = window.getComputedStyle(element);
    if (style.display === "none") return false;
    if (style.visibility === "hidden") return false;
    if (parseFloat(style.opacity) === 0) return false;
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    return true;
  }

  function isInsideBadAncestor(element) {
    return element.closest(BAD_ANCESTOR_SELECTOR) !== null;
  }

  function getScore(element) {
    let score = 0;
    if (element.closest("article")) score += 10;
    if (element.closest("[itemprop='headline'],[itemprop='name']")) score += 8;
    if (element.hasAttribute("itemprop")) score += 5;
    if (element.tagName === "H1") score += 6;
    else if (element.tagName === "H2") score += 4;
    else if (element.tagName === "H3") score += 3;
    const cls = (element.className || "").toLowerCase();
    if (/title/i.test(cls)) score += 3;
    if (/headline|article|story|news/i.test(cls)) score += 2;
    if (/card|feature|breaking/i.test(cls)) score += 2;
    return score;
  }

  // ─── Text extraction helpers ──────────────────────────────────────────────

  /**
   * Remove noise elements and return textContent of a cloned node.
   */
  function stripNoise(node) {
    const clone = node.cloneNode(true);
    clone.querySelectorAll(BAD_TEXT_SELECTORS.join(",")).forEach((el) => el.remove());
    return (clone.textContent || "").replace(/\s+/g, " ").trim();
  }

  /**
   * Find the most specific nested element that likely contains the headline text.
   * This solves the "outer card <a> with section label + headline + byline" problem
   * found on Guardian/NYT homepages.
   *
   * Search order:
   * 1. The root element itself (if it's a heading or has itemprop)
   * 2. Direct children that are headings or have news-like classes
   * 3. Descendants within a reasonable depth
   */
  function findNestedTitleElement(root) {
    // 1. Root itself
    if (root instanceof Element) {
      const rootTag = root.tagName;
      if (["H1", "H2", "H3", "H4", "H5", "H6"].includes(rootTag)) {
        const text = stripNoise(root);
        if (text.length >= MIN_TITLE_LENGTH) return root;
      }
      if (root.hasAttribute("itemprop")) {
        const text = stripNoise(root);
        if (text.length >= MIN_TITLE_LENGTH) return root;
      }
    }

    // 2. Direct children (limit depth to avoid deep DOM traversal)
    const depthLimit = 5;
    function searchIn(node, depth) {
      if (depth > depthLimit) return null;
      const children = node.children;
      for (const child of children) {
        const childTag = child.tagName;
        if (["H1", "H2", "H3", "H4", "H5", "H6"].includes(childTag)) {
          const text = stripNoise(child);
          if (text.length >= MIN_TITLE_LENGTH) return child;
        }
        const childClass = child.className || "";
        if (
          /headline|title|card/i.test(childClass) &&
          !/byline|author|date|time|tag|section|kicker/i.test(childClass)
        ) {
          const text = stripNoise(child);
          if (text.length >= MIN_TITLE_LENGTH) return child;
        }
        const result = searchIn(child, depth + 1);
        if (result) return result;
      }
      return null;
    }

    return searchIn(root, 0);
  }

  /**
   * Extract title text from an element.
   * 1. aria-label (exact, unambiguous)
   * 2. Nested heading/itemprop/headline element inside the target (most specific)
   * 3. Direct textContent of the target itself
   */
  function extractTitleText(target) {
    // 1. aria-label — always preferred
    const aria = target.getAttribute("aria-label");
    if (aria && aria.trim().length >= MIN_TITLE_LENGTH) {
      return aria.trim().replace(/\s+/g, " ").slice(0, MAX_TITLE_LENGTH);
    }

    // 2. Nested title element (solves homepage card wrapper problem)
    const nested = findNestedTitleElement(target);
    if (nested) {
      const text = stripNoise(nested);
      if (text.length >= MIN_TITLE_LENGTH) {
        return text.slice(0, MAX_TITLE_LENGTH);
      }
    }

    // 3. Direct textContent of target
    const raw = stripNoise(target);
    if (raw.length < MIN_TITLE_LENGTH) return "";
    return raw.slice(0, MAX_TITLE_LENGTH);
  }

  // ─── Target element selection ─────────────────────────────────────────────

  /**
   * Find the best candidate element to use as the title target from a mouseover event.
   *
   * Strategy:
   * 1. If the hovered element itself is a heading, use it directly.
   * 2. If the hovered element is inside a heading, use the heading (not the outer card <a>).
   * 3. Walk up to find a matching candidate; if that candidate is a card wrapper <a> and
   *    contains a nested heading, prefer the nested heading.
   * 4. Otherwise fall back to the best scoring candidate on the page.
   */
  function findTargetElement(eventTarget, clientX, clientY) {
    if (!(eventTarget instanceof Element)) return null;
    if (eventTarget.id === "cbd-hover-tooltip" || eventTarget.closest("#cbd-hover-tooltip")) {
      return null;
    }

    // Skip if the target is a noise element
    for (const sel of BAD_TEXT_SELECTORS) {
      try {
        if (eventTarget.matches(sel)) return null;
      } catch (_) { /* invalid */ }
    }

    // ── Helper: check if a candidate element is the one at clientX/clientY ──
    // Using elementFromPoint is more reliable than contains() because some sites
    // (Guardian) overlay iframes that intercept events and break contains().
    function isElementAtPoint(el) {
      try {
        const at = document.elementFromPoint(clientX, clientY);
        if (!at) return false;
        return el === at || el.contains(at);
      } catch (_) {
        return false;
      }
    }

    // ── Case 1: Hovered element is a heading ───────────────────────────────
    const tag = eventTarget.tagName;
    if (["H1", "H2", "H3", "H4", "H5", "H6"].includes(tag)) {
      const text = extractTitleText(eventTarget);
      if (text) return eventTarget;
    }

    // ── Case 2: Hovered element is inside a heading ───────────────────────
    const headingAncestor = eventTarget.closest("h1, h2, h3, h4, [role='heading']");
    if (headingAncestor) {
      const text = extractTitleText(headingAncestor);
      if (text) return headingAncestor;
    }

    // ── Case 3: Hovered element is inside a nested title element ────────────
    const nestedTitle = eventTarget.closest(TITLE_TEXT_SELECTORS.join(","));
    if (nestedTitle) {
      const text = extractTitleText(nestedTitle);
      if (text) return nestedTitle;
    }

    // ── Case 4: Walk up to find a page-level candidate ─────────────────────
    // Only consider candidates that contain (or are) the event target position
    let current = eventTarget;
    while (current && current !== document.documentElement) {
      const isCandidate = SELECTORS.some((sel) => {
        try {
          return current.matches(sel);
        } catch (_) {
          return false;
        }
      });

      if (isCandidate && isVisible(current) && !isInsideBadAncestor(current)) {
        const text = extractTitleText(current);
        if (text) return current;
      }
      current = current.parentElement;
    }

    // ── Case 5: Last resort — use elementFromPoint at the cursor ───────────
    // This handles cases where an overlay iframe blocks contains() but
    // document.elementFromPoint still gives us the real underlying element
    try {
      const fromPoint = document.elementFromPoint(clientX, clientY);
      if (fromPoint && fromPoint instanceof Element && fromPoint !== document.documentElement) {
        // Check if fromPoint itself is a suitable candidate
        if (!fromPoint.matches(BAD_TEXT_SELECTORS.join(","))) {
          const text = extractTitleText(fromPoint);
          if (text) return fromPoint;
        }
        // Also check its heading ancestors
        const headingFromPoint = fromPoint.closest("h1, h2, h3, h4, [role='heading']");
        if (headingFromPoint) {
          const text = extractTitleText(headingFromPoint);
          if (text) return headingFromPoint;
        }
        const nestedFromPoint = fromPoint.closest(TITLE_TEXT_SELECTORS.join(","));
        if (nestedFromPoint) {
          const text = extractTitleText(nestedFromPoint);
          if (text) return nestedFromPoint;
        }
      }
    } catch (_) { /* elementFromPoint not available */ }

    return null;
  }

  // ─── Messaging / Request helpers (unchanged) ─────────────────────────────

  function hasRuntimeMessaging() {
    return typeof chrome !== "undefined" && !!chrome.runtime && typeof chrome.runtime.sendMessage === "function";
  }

  function createTypedError(kind, message) {
    const err = new Error(message);
    err.kind = kind;
    return err;
  }

  function isRuntimeUnavailableError(error) {
    if (!error || typeof error !== "object") return false;
    if (error.kind === "extension") return true;
    if (!(error instanceof Error)) return false;
    return /Receiving end does not exist|Extension context invalidated|message port closed before a response was received|runtime messaging error/i.test(
      error.message || ""
    );
  }

  function shouldUseDirectFetchFallback(error) {
    return isRuntimeUnavailableError(error) && window.location.protocol === "http:";
  }

  function sendPredictMessageOnce(title, signal) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const clearListeners = () => {
        if (signal && onAbort) signal.removeEventListener("abort", onAbort);
      };
      const settleResolve = (value) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeoutId);
        clearListeners();
        resolve(value);
      };
      const settleReject = (error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeoutId);
        clearListeners();
        reject(error);
      };
      const timeoutId = window.setTimeout(() => {
        settleReject(createTypedError("extension", "Extension runtime timeout while sending prediction request"));
      }, MESSAGE_TIMEOUT_MS);
      let onAbort = null;
      if (signal) {
        if (signal.aborted) {
          settleReject(new DOMException("Aborted", "AbortError"));
          return;
        }
        onAbort = () => settleReject(new DOMException("Aborted", "AbortError"));
        signal.addEventListener("abort", onAbort, { once: true });
      }
      try {
        chrome.runtime.sendMessage({ type: "CBD_PREDICT", title }, (response) => {
          const runtimeError = chrome.runtime.lastError;
          if (runtimeError) {
            settleReject(createTypedError("extension", runtimeError.message || "Runtime messaging error"));
            return;
          }
          settleResolve(response);
        });
      } catch (error) {
        settleReject(
          createTypedError("extension", error instanceof Error && error.message ? error.message : "Runtime messaging error")
        );
      }
    });
  }

  async function sendPredictMessage(title, signal) {
    let lastError = null;
    for (let attempt = 0; attempt <= MESSAGE_RETRY_COUNT; attempt += 1) {
      try {
        return await sendPredictMessageOnce(title, signal);
      } catch (error) {
        lastError = error;
        if (error instanceof DOMException && error.name === "AbortError") throw error;
        if (!isRuntimeUnavailableError(error) || attempt >= MESSAGE_RETRY_COUNT) throw error;
        await new Promise((resolve) => window.setTimeout(resolve, 120));
      }
    }
    throw lastError || createTypedError("extension", "Runtime messaging error");
  }

  function buildAbortPromise(signal) {
    return new Promise((_, reject) => {
      if (!signal) return;
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    });
  }

  async function requestPredictionDirect(title, signal) {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
      signal,
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        if (Array.isArray(payload.detail)) {
          detail = payload.detail.map((item) => (item && typeof item === "object" && "msg" in item ? String(item.msg) : String(item))).join("; ");
        } else {
          detail = payload.detail || payload.error || detail;
        }
      } catch (_) { /* Keep fallback */ }
      const err = new Error(detail);
      err.kind = "backend";
      throw err;
    }
    return await response.json();
  }

  async function requestPrediction(title, signal) {
    if (hasRuntimeMessaging()) {
      try {
        const response = await Promise.race([sendPredictMessage(title), buildAbortPromise(signal)]);
        if (!response || !response.ok) {
          const err = createTypedError(response?.kind || "backend", response?.error || "Prediction request failed");
          throw err;
        }
        return response.data;
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") throw error;
        if (shouldUseDirectFetchFallback(error)) return await requestPredictionDirect(title, signal);
        if (isRuntimeUnavailableError(error) && window.location.protocol === "https:") {
          throw createTypedError("extension", "Extension runtime unavailable. Reload extension and refresh this tab.");
        }
        throw error;
      }
    }
    return await requestPredictionDirect(title, signal);
  }

  // ─── Tooltip helpers ─────────────────────────────────────────────────────

  function ensureTooltip() {
    if (tooltipEl) return tooltipEl;
    tooltipEl = document.createElement("div");
    tooltipEl.id = "cbd-hover-tooltip";
    titleEl = document.createElement("div");
    titleEl.className = "cbd-title";
    detailEl = document.createElement("div");
    detailEl.className = "cbd-detail";
    tooltipEl.appendChild(titleEl);
    tooltipEl.appendChild(detailEl);
    document.documentElement.appendChild(tooltipEl);
    return tooltipEl;
  }

  function showTooltip(title, detail, tone) {
    const node = ensureTooltip();
    node.className = `cbd-tooltip cbd-visible cbd-${tone}`;
    titleEl.textContent = title;
    detailEl.textContent = detail;
    positionTooltip(pointer.x, pointer.y);
  }

  function hideTooltip() {
    if (!tooltipEl) return;
    tooltipEl.className = "cbd-tooltip";
  }

  function positionTooltip(x, y) {
    if (!tooltipEl) return;
    const margin = 12;
    const offsetX = 14;
    const offsetY = 18;
    let left = x + offsetX;
    let top = y + offsetY;
    const rect = tooltipEl.getBoundingClientRect();
    if (left + rect.width + margin > window.innerWidth) left = x - rect.width - offsetX;
    if (top + rect.height + margin > window.innerHeight) top = y - rect.height - offsetY;
    left = Math.max(margin, left);
    top = Math.max(margin, top);
    tooltipEl.style.left = `${left}px`;
    tooltipEl.style.top = `${top}px`;
  }

  function setCache(key, value) {
    if (localCache.has(key)) localCache.delete(key);
    localCache.set(key, value);
    if (localCache.size > MAX_CACHE_ENTRIES) {
      const oldestKey = localCache.keys().next().value;
      localCache.delete(oldestKey);
    }
  }

  // ─── Hover timer / analysis flow ──────────────────────────────────────────

  function clearHoverTimer() {
    if (hoverTimer !== null) {
      window.clearTimeout(hoverTimer);
      hoverTimer = null;
    }
  }

  function abortInFlightRequest() {
    if (!activeController) return;
    activeController.abort();
    activeController = null;
  }

  function resetActiveState() {
    clearHoverTimer();
    abortInFlightRequest();
    activeTarget = null;
    activeTitle = "";
  }

  function scheduleAnalysis(target, title) {
    clearHoverTimer();
    activeTarget = target;
    activeTitle = title;
    showTooltip("Analyzing...", "Running ORCD reasoning pipeline", "info");
    hoverTimer = window.setTimeout(() => {
      hoverTimer = null;
      void analyzeTitle(target, title);
    }, HOVER_DELAY_MS);
  }

  async function analyzeTitle(target, title) {
    if (target !== activeTarget || title !== activeTitle) return;
    if (localCache.has(title)) {
      renderPrediction(localCache.get(title));
      return;
    }
    abortInFlightRequest();
    const controller = new AbortController();
    activeController = controller;
    const timeoutId = window.setTimeout(() => controller.abort("timeout"), REQUEST_TIMEOUT_MS);
    try {
      const result = await requestPrediction(title, controller.signal);
      setCache(title, result);
      if (target === activeTarget && title === activeTitle) renderPrediction(result);
    } catch (error) {
      if (
        controller.signal.aborted &&
        controller.signal.reason !== "timeout" &&
        !(error instanceof DOMException && error.name === "AbortError")
      ) {
        return;
      }
      const errorKind = error && typeof error === "object" ? error.kind : "";
      if ((controller.signal.aborted && controller.signal.reason === "timeout") || errorKind === "timeout") {
        showTooltip("Server Timeout", "Backend took too long to respond", "error");
        return;
      }
      if (errorKind === "extension") {
        showTooltip("Extension Not Ready", "Reload extension and refresh this tab", "error");
        return;
      }
      const isOffline = error instanceof TypeError || errorKind === "network";
      if (isOffline) {
        showTooltip("Server Offline", "Cannot connect to local backend", "error");
        return;
      }
      const message = error instanceof Error ? error.message : "Unexpected backend error";
      showTooltip("Prediction Error", message.slice(0, 140), "error");
    } finally {
      window.clearTimeout(timeoutId);
      if (activeController === controller) activeController = null;
    }
  }

  function renderPrediction(result) {
    const confidence = Number(result.confidence || 0);
    const isClickbait = Boolean(result.is_clickbait);
    const confidenceText = Number.isFinite(confidence) ? confidence.toFixed(1) : "0.0";
    if (isClickbait) {
      showTooltip(
        `${confidenceText}% suspicious clickbait`,
        `Model: ${result.model || "orcd"} | ${result.cached ? "cached" : "fresh"}`,
        "danger"
      );
      return;
    }
    showTooltip(
      `${confidenceText}% genuine news`,
      `Model: ${result.model || "orcd"} | ${result.cached ? "cached" : "fresh"}`,
      "success"
    );
  }

  // ─── Event listeners ──────────────────────────────────────────────────────

  /**
   * Walk into same-origin iframes to find the real element at (x, y).
   * Some sites (e.g. Guardian) overlay invisible iframes at hover positions.
   * This recursively probes inside same-origin iframes to find the actual element.
   */
  function getRealElementAtPoint(x, y) {
    let el;
    try {
      el = document.elementFromPoint(x, y);
    } catch (_) {
      return null;
    }
    if (!el || el === document.documentElement || el === document.body) {
      return null;
    }
    // If it's an iframe, try to pierce through same-origin iframes
    while (el && el.tagName === "IFRAME") {
      try {
        const iframeWin = el.contentWindow;
        if (!iframeWin) break;
        const iframeDoc = iframeWin.document;
        // Convert coordinates to iframe's coordinate space
        const iframeRect = el.getBoundingClientRect();
        const relX = x - iframeRect.left;
        const relY = y - iframeRect.top;
        const inner = iframeDoc.elementFromPoint(relX, relY);
        if (inner && inner !== iframeDoc.documentElement && inner !== iframeDoc.body) {
          el = inner;
        } else {
          break;
        }
      } catch (_) {
        // Cross-origin iframe — cannot pierce
        break;
      }
    }
    return el instanceof Element ? el : null;
  }

  /**
   * Find the real element at (x, y), piercing same-origin iframes.
   * Some sites (e.g. Guardian) overlay invisible iframes at hover positions.
   */

  function handleMouseOver(event) {
    // Some sites (e.g. Guardian) overlay invisible iframes that capture events.
    // getRealElementAtPoint pierces through same-origin iframes.
    // If it's cross-origin (cannot pierce), fall back to walking up from event.target.
    const fromPoint = getRealElementAtPoint(event.clientX, event.clientY);
    let realTarget = fromPoint && fromPoint.tagName !== "IFRAME" ? fromPoint : null;
    // If piercing failed (cross-origin iframe), walk up from event.target
    if (!realTarget && event.target instanceof Element) {
      let cur = event.target;
      while (cur && cur !== document.documentElement) {
        const text = extractTitleText(cur);
        if (text) { realTarget = cur; break; }
        cur = cur.parentElement;
      }
    }
    if (!realTarget) return;

    const target = findTargetElement(realTarget, event.clientX, event.clientY);
    if (!target) return;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    const title = extractTitleText(target);
    if (!title) return;
    if (target === activeTarget && title === activeTitle) return;
    scheduleAnalysis(target, title);
  }

  function handleMouseMove(event) {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    if (tooltipEl && tooltipEl.classList.contains("cbd-visible")) {
      positionTooltip(pointer.x, pointer.y);
    }
  }

  function handleMouseOut(event) {
    if (!activeTarget) return;
    if (!event.relatedTarget || !activeTarget.contains(event.relatedTarget)) {
      resetActiveState();
      hideTooltip();
    }
  }

  function handleVisibilityChange() {
    if (!document.hidden) return;
    resetActiveState();
    hideTooltip();
  }

  document.addEventListener("mouseover", handleMouseOver, true);
  document.addEventListener("mousemove", handleMouseMove, { capture: true, passive: true });
  document.addEventListener("mouseout", handleMouseOut, true);
  document.addEventListener("visibilitychange", handleVisibilityChange, true);
})();
