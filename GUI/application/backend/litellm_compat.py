from __future__ import annotations

from typing import Any

from openai import OpenAI


def completion(**kwargs: Any) -> dict[str, Any]:
    model = kwargs.get("model")
    messages = kwargs.get("messages") or []
    api_key = kwargs.get("api_key")
    api_base = kwargs.get("api_base") or kwargs.get("base_url")
    timeout_s = kwargs.get("timeout")
    max_tokens = kwargs.get("max_tokens")
    temperature = kwargs.get("temperature", 0)
    top_p = kwargs.get("top_p", 1)
    response_format = kwargs.get("response_format")

    if not model:
        raise ValueError("completion() requires 'model'")
    if not isinstance(messages, list) or not messages:
        raise ValueError("completion() requires non-empty 'messages'")

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if api_base:
        client_kwargs["base_url"] = api_base
    if timeout_s:
        client_kwargs["timeout"] = timeout_s

    client = OpenAI(**client_kwargs)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        request_kwargs["response_format"] = response_format

    response = client.chat.completions.create(**request_kwargs)
    text = ""
    if response and response.choices:
        message = response.choices[0].message
        text = str(getattr(message, "content", "") or "")

    return {
        "choices": [
            {
                "message": {
                    "content": text,
                }
            }
        ]
    }
