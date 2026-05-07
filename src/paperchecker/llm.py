"""LLM backends — DeepSeek, OpenAI, Claude."""

import json
import urllib.request
import urllib.error
from paperchecker.config import Config


def _call_deepseek(config: Config, prompt: str) -> str | None:
    """Call DeepSeek API using urlopen (no external client needed)."""
    payload = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.llm_max_tokens,
            "temperature": config.llm_temperature,
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {config.deepseek_api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=config.llm_timeout)
        body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _call_openai(config: Config, prompt: str) -> str | None:
    """Call OpenAI API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.openai_api_key, timeout=config.llm_timeout)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception:
        return None


def _call_claude(config: Config, prompt: str) -> str | None:
    """Call Anthropic Claude API."""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=config.anthropic_api_key, timeout=config.llm_timeout)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return None


BACKENDS = {
    "deepseek": _call_deepseek,
    "openai": _call_openai,
    "claude": _call_claude,
}


def call_llm(config: Config, prompt: str, backend: str | None = None) -> str | None:
    """Call the LLM with the given prompt.

    Uses the specified backend, or the first available one (DeepSeek preferred).
    Returns the response text or None on failure.
    """
    if backend and backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")

    if backend:
        fn = BACKENDS[backend]
        return fn(config, prompt)

    for name in ["deepseek", "openai", "claude"]:
        if name in config.available_backends:
            return BACKENDS[name](config, prompt)

    return None
