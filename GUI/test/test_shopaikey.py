import litellm
import os

try:
    response = litellm.completion(
        model="gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": "Rate this out of 100. Return JSON with 'score'."}],
        api_base="https://api-v2.shopaikey.com/v1",
        custom_llm_provider="openai",
        api_key="sk-123456789",  # dummy
    )
    print("Without response_format:")
    print(response)
except Exception as e:
    print("Error:", e)
