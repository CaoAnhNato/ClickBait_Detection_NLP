with open('test_model_service.py', 'r') as f:
    content = f.read()
import re
content = re.sub(r'assert set\(service\.available_models\(\)\.keys\(\)\) == \{[^}]+\}', 'assert set(service.available_models().keys()) == {\n        "bart-mnli",\n        "roberta",\n        "ggbert",\n        "sheepdog",\n        "gpt4o-zero",\n        "gpt4o-few",\n        "orcd-gpt35",\n        "orcd-gpt4o",\n        "gemini-zero",\n        "qwen-zero",\n        "llama-zero",\n    }', content)
with open('test_model_service.py', 'w') as f:
    f.write(content)
