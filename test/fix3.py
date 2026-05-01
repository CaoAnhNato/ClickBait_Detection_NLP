import re

with open('GUI/application/tests/e2e_hover_news_test.py', 'r', encoding='utf-8') as f:
    text = f.read()

fixed = re.sub(
    r'if request.method ==.*?(?=route.fulfill\()',
    'if request.method == "OPTIONS":\n                route.fulfill(status=200, headers={"Access-Control-Allow-Origin":"*", "Access-Control-Allow-Methods":"*", "Access-Control-Allow-Headers":"*"})\n                return\n            response = requests.post(proxy_url, json=payload, timeout=45)\n            ',
    text,
    flags=re.DOTALL
)

with open('GUI/application/tests/e2e_hover_news_test.py', 'w', encoding='utf-8') as f:
    f.write(fixed)
