import litellm
import json

try:
    response = litellm.completion(
        model="gemini/gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://api-v2.shopaikey.com/v1",
        custom_llm_provider="gemini",
        api_key="dummy"
    )
    print(response)
except Exception as e:
    print("Error:", e)

