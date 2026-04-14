(() => {
  const API_ENDPOINTS = [
    "http://127.0.0.1:8000/predict",
    "http://localhost:8000/predict"
  ];
  const REQUEST_TIMEOUT_MS = 45000;

  function parseBackendDetail(payload, fallbackStatus) {
    if (payload && typeof payload === "object") {
      if (Array.isArray(payload.detail)) {
        return payload.detail
          .map((item) => {
            if (item && typeof item === "object" && "msg" in item) {
              return String(item.msg);
            }
            return String(item);
          })
          .join("; ");
      }

      if (payload.detail) {
        return String(payload.detail);
      }

      if (payload.error) {
        return String(payload.error);
      }
    }

    return `Request failed (${fallbackStatus})`;
  }

  async function predictThroughEndpoint(endpoint, title) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort("timeout");
    }, REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ title }),
        signal: controller.signal
      });

      const raw = await response.text();
      let payload = null;
      try {
        payload = raw ? JSON.parse(raw) : null;
      } catch (_err) {
        payload = null;
      }

      if (!response.ok) {
        return {
          ok: false,
          kind: "backend",
          status: response.status,
          error: parseBackendDetail(payload, response.status)
        };
      }

      return {
        ok: true,
        data: payload ?? {}
      };
    } catch (error) {
      if (controller.signal.aborted && controller.signal.reason === "timeout") {
        return {
          ok: false,
          kind: "timeout",
          error: "Backend took too long to respond"
        };
      }

      return {
        ok: false,
        kind: "network",
        error: error instanceof Error ? error.message : "Cannot connect to local backend"
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || message.type !== "CBD_PREDICT") {
      return;
    }

    const title = typeof message.title === "string" ? message.title.trim() : "";
    if (!title) {
      sendResponse({
        ok: false,
        kind: "backend",
        status: 400,
        error: "Missing title"
      });
      return;
    }

    (async () => {
      let lastNetworkError = {
        ok: false,
        kind: "network",
        error: "Cannot connect to local backend"
      };

      for (const endpoint of API_ENDPOINTS) {
        const result = await predictThroughEndpoint(endpoint, title);
        if (result.ok) {
          sendResponse(result);
          return;
        }

        if (result.kind === "backend" || result.kind === "timeout") {
          sendResponse(result);
          return;
        }

        lastNetworkError = result;
      }

      sendResponse(lastNetworkError);
    })();

    return true;
  });
})();
