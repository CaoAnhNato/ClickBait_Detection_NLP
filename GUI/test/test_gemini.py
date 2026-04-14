import litellm
import os

api_key = os.environ.get("GEMINI_API_KEY", "")

try:
    response = litellm.completion(
        model="gemini/gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": "Rate this out of 100: 'The dog is good'. Return JSON with 'score'."}],
        api_key=api_key,
        response_format={"type": "json_object"}
    )
    print("With response_format:")
    print(response.choices[0].message.content)
except Exception as e:
    print("Error with response_format:", e)

try:
    response = litellm.completion(
        model="gemini/gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": "Rate this out of 100: 'The dog is good'. Return JSON with 'score'."}],
        api_key=api_key
    )
    print("Without response_format:")
    print(response.choices[0].message.content)
except Exception as e:
    print("Error without response_format:", e)
