import litellm
import os

try:
    response = litellm.completion(
        model="gemini-3.1-flash-lite-preview",
        messages=[
            {"role": "system", "content": "You are a helpful and expert news analyst."},
            {"role": "user", "content": "Goal: Re-score... output format [int]."}
        ],
        max_tokens=16,
        temperature=0.0,
        api_key="sk-xxxx", # shopaikey token?
        api_base="https://api-v2.shopaikey.com/v1",
        custom_llm_provider="openai",
        timeout=3.0
    )
    print("Success:", response)
except Exception as e:
    print("Error during score:", e)
