import re

with open('GUI/application/tests/e2e_hover_news_test.py', 'r', encoding='utf-8') as f:
    text = f.read()

fixed = re.sub(
    r'= requests\.post\(proxy.*?timeout=45\)',
    '',
    text,
    flags=re.DOTALL
)

with open('GUI/application/tests/e2e_hover_news_test.py', 'w', encoding='utf-8') as f:
    f.write(fixed)
