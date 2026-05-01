import re
with open('GUI/application/tests/e2e_hover_news_test.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('js_text = js_text.replace(endpoint_literal, \'const API_ENDPOINT = "/__cbd_predict";\')', '')
with open('GUI/application/tests/e2e_hover_news_test.py', 'w', encoding='utf-8') as f:
    f.write(text)
