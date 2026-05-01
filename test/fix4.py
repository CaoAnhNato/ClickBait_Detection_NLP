import re
with open('GUI/application/tests/e2e_hover_news_test.py', 'r', encoding='utf-8') as f:
    text = f.read()
lines = text.split('\n')
for i in range(len(lines)):
    if 'def handle_predict_proxy(' in lines[i]:
        start = i
    if 'page.route(' in lines[i]:
        end = i

replacement = '''    def handle_predict_proxy(route, request) -> None:
        try:
            if request.method == "OPTIONS":
                route.fulfill(status=200, headers={"Access-Control-Allow-Origin":"*", "Access-Control-Allow-Methods":"*", "Access-Control-Allow-Headers":"*"})
                return
            payload = request.post_data_json or {}
            response = requests.post(proxy_url, json=payload, timeout=45)
            route.fulfill(
                status=response.status_code,
                headers={"Content-Type": response.headers.get("content-type", "application/json"), "Access-Control-Allow-Origin":"*"},
                body=response.text,
            )
        except Exception as exc:  # noqa: BLE001
            route.fulfill(status=503, headers={"Content-Type": "application/json", "Access-Control-Allow-Origin":"*"}, body='{"detail": "proxy error"}')

'''
lines = lines[:start] + replacement.split('\n') + lines[end:]
with open('GUI/application/tests/e2e_hover_news_test.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
