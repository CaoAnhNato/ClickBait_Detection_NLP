(() => {
  const API_ENDPOINT = "http://127.0.0.1:8000/predict";
  const HOVER_DELAY_MS = 500;
  const REQUEST_TIMEOUT_MS = 45000;
  const MESSAGE_TIMEOUT_MS = 8000;
  const MESSAGE_RETRY_COUNT = 1;
  const MAX_TITLE_LENGTH = 320;
  const MAX_CACHE_ENTRIES = 500;
  const TARGET_SELECTOR = "a, h1, h2, h3, h4, [class*='title' i], [class*='headline' i]";

  let hoverTimer = null;
  let activeTarget = null;
  let activeTitle = "";
  let activeController = null;
  let tooltipEl = null;
  let titleEl = null;
  let detailEl = null;
  const pointer = { x: 0, y: 0 };
  const localCache = new Map();

  function hasRuntimeMessaging() {
    return typeof chrome !== "undefined" && !!chrome.runtime && typeof chrome.runtime.sendMessage === "function";
  }

  function createTypedError(kind, message) {
    const err = new Error(message);
    err.kind = kind;
    return err;
  }

  function isRuntimeUnavailableError(error) {
    if (!error || typeof error !== "object") {
      return false;
    }

    if (error.kind === "extension") {
      return true;
    }

    if (!(error instanceof Error)) {
      return false;
    }

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
        if (signal && onAbort) {
          signal.removeEventListener("abort", onAbort);
        }
      };

      const settleResolve = (value) => {
        if (settled) {
          return;
        }
        settled = true;
        window.clearTimeout(timeoutId);
        clearListeners();
        resolve(value);
      };

      const settleReject = (error) => {
        if (settled) {
          return;
        }
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

        onAbort = () => {
          settleReject(new DOMException("Aborted", "AbortError"));
        };
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
          createTypedError(
            "extension",
            error instanceof Error && error.message ? error.message : "Runtime messaging error"
          )
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

        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }

        if (!isRuntimeUnavailableError(error) || attempt >= MESSAGE_RETRY_COUNT) {
          throw error;
        }

        await new Promise((resolve) => {
          window.setTimeout(resolve, 120);
        });
      }
    }

    throw lastError || createTypedError("extension", "Runtime messaging error");
  }

  function buildAbortPromise(signal) {
    return new Promise((_, reject) => {
      if (!signal) {
        return;
      }

      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }

      signal.addEventListener(
        "abort",
        () => {
          reject(new DOMException("Aborted", "AbortError"));
        },
        { once: true }
      );
    });
  }

  async function requestPredictionDirect(title, signal) {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ title }),
      signal
    });

    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        if (Array.isArray(payload.detail)) {
          detail = payload.detail
            .map((item) => {
              if (item && typeof item === "object" && "msg" in item) {
                return String(item.msg);
              }
              return String(item);
            })
            .join("; ");
        } else {
          detail = payload.detail || payload.error || detail;
        }
      } catch (_error) {
        // Keep fallback detail when response body is not JSON.
      }

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
        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }

        // Falling back to direct fetch on HTTPS can trigger mixed-content failures.
        if (shouldUseDirectFetchFallback(error)) {
          return await requestPredictionDirect(title, signal);
        }

        if (isRuntimeUnavailableError(error) && window.location.protocol === "https:") {
          throw createTypedError("extension", "Extension runtime unavailable. Reload extension and refresh this tab.");
        }

        throw error;
      }
    }

    return await requestPredictionDirect(title, signal);
  }

  function ensureTooltip() {
    if (tooltipEl) {
      return tooltipEl;
    }

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
    if (!tooltipEl) {
      return;
    }
    tooltipEl.className = "cbd-tooltip";
  }

  function positionTooltip(x, y) {
    if (!tooltipEl) {
      return;
    }

    const margin = 12;
    const offsetX = 14;
    const offsetY = 18;
    let left = x + offsetX;
    let top = y + offsetY;

    const rect = tooltipEl.getBoundingClientRect();

    if (left + rect.width + margin > window.innerWidth) {
      left = x - rect.width - offsetX;
    }
    if (top + rect.height + margin > window.innerHeight) {
      top = y - rect.height - offsetY;
    }

    left = Math.max(margin, left);
    top = Math.max(margin, top);

    tooltipEl.style.left = `${left}px`;
    tooltipEl.style.top = `${top}px`;
  }

  function setCache(key, value) {
    if (localCache.has(key)) {
      localCache.delete(key);
    }
    localCache.set(key, value);

    if (localCache.size > MAX_CACHE_ENTRIES) {
      const oldestKey = localCache.keys().next().value;
      localCache.delete(oldestKey);
    }
  }

  function findTargetElement(node) {
    if (!(node instanceof Element)) {
      return null;
    }

    if (node.id === "cbd-hover-tooltip" || node.closest("#cbd-hover-tooltip")) {
      return null;
    }

    return node.closest(TARGET_SELECTOR);
  }

  function extractTitleText(target) {
    const raw = (target.getAttribute("aria-label") || target.textContent || "").replace(/\s+/g, " ").trim();
    if (!raw || raw.length < 8) {
      return "";
    }
    return raw.slice(0, MAX_TITLE_LENGTH);
  }

  function clearHoverTimer() {
    if (hoverTimer !== null) {
      window.clearTimeout(hoverTimer);
      hoverTimer = null;
    }
  }

  function abortInFlightRequest() {
    if (!activeController) {
      return;
    }

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
    if (target !== activeTarget || title !== activeTitle) {
      return;
    }

    if (localCache.has(title)) {
      renderPrediction(localCache.get(title));
      return;
    }

    abortInFlightRequest();
    const controller = new AbortController();
    activeController = controller;

    const timeoutId = window.setTimeout(() => {
      controller.abort("timeout");
    }, REQUEST_TIMEOUT_MS);

    try {
      const result = await requestPrediction(title, controller.signal);
      setCache(title, result);

      if (target === activeTarget && title === activeTitle) {
        renderPrediction(result);
      }
    } catch (error) {
      if (
        controller.signal.aborted &&
        controller.signal.reason !== "timeout" &&
        !(error instanceof DOMException && error.name === "AbortError")
      ) {
        return;
      }

      const errorKind = error && typeof error === "object" ? error.kind : "";

      if (controller.signal.aborted && controller.signal.reason === "timeout") {
        showTooltip("Server Timeout", "Backend took too long to respond", "error");
        return;
      }

      if (errorKind === "timeout") {
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
      if (activeController === controller) {
        activeController = null;
      }
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

  function handleMouseOver(event) {
    const target = findTargetElement(event.target);
    if (!target) {
      return;
    }

    pointer.x = event.clientX;
    pointer.y = event.clientY;

    const title = extractTitleText(target);
    if (!title) {
      return;
    }

    if (target === activeTarget && title === activeTitle) {
      return;
    }

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
    if (!activeTarget) {
      return;
    }

    const fromTarget = findTargetElement(event.target);
    if (!fromTarget) {
      return;
    }

    const relatedTarget = event.relatedTarget;
    const nextTarget = findTargetElement(relatedTarget);

    if (nextTarget === activeTarget) {
      return;
    }

    resetActiveState();
    hideTooltip();
  }

  function handleVisibilityChange() {
    if (!document.hidden) {
      return;
    }
    resetActiveState();
    hideTooltip();
  }

  document.addEventListener("mouseover", handleMouseOver, true);
  document.addEventListener("mousemove", handleMouseMove, { capture: true, passive: true });
  document.addEventListener("mouseout", handleMouseOut, true);
  document.addEventListener("visibilitychange", handleVisibilityChange, true);
})();
