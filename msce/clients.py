"""Provider-neutral HTTP helpers for MSCE.

The repository intentionally ships without model endpoints or API keys. Set the
environment variables documented in `.env.example` before running the pipeline.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import requests


DEFAULT_LLM_MODEL = os.environ.get("MSCE_LLM_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL = os.environ.get("MSCE_EMBEDDING_MODEL", "bge-m3")
DEFAULT_EMBEDDING_DIM = int(os.environ.get("MSCE_EMBEDDING_DIM", "1024"))


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "See .env.example for the provider-neutral configuration names."
        )
    return value


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("MSCE_CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def parse_jsonish(content: str) -> Any:
    text = (content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        lo, hi = text.find("{"), text.rfind("}")
        if lo >= 0 and hi > lo:
            return json.loads(text[lo:hi + 1])
        raise


def chat_completion(
    system: str,
    user: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    retries: int = 3,
    timeout: int = 120,
    model: str | None = None,
) -> str:
    url = _required_env("MSCE_CHAT_COMPLETIONS_URL")
    payload = {
        "model": model or DEFAULT_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=_auth_headers(),
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            if attempt == retries - 1:
                response.raise_for_status()
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    return ""


def chat_completion_json(
    system: str,
    user: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    retries: int = 3,
    timeout: int = 120,
    model: str | None = None,
) -> Any:
    content = chat_completion(
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        timeout=timeout,
        model=model,
    )
    return parse_jsonish(content)


def embedding_batch(
    texts: list[str],
    batch_size: int = 16,
    retries: int = 4,
    timeout: int = 120,
) -> np.ndarray:
    url = _required_env("MSCE_EMBEDDINGS_URL")
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = [t[:6000] if len(t) > 6000 else t for t in texts[i:i + batch_size]]
        payload = {"model": DEFAULT_EMBEDDING_MODEL, "input": chunk}
        for attempt in range(retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=_auth_headers(),
                    timeout=timeout,
                )
                if response.status_code == 200:
                    data = response.json()["data"]
                    out.append(
                        np.asarray([d["embedding"] for d in data], dtype=np.float32)
                    )
                    break
                if attempt == retries - 1:
                    response.raise_for_status()
            except Exception:
                if attempt == retries - 1:
                    raise
            time.sleep(2**attempt)
    full = (
        np.concatenate(out, axis=0)
        if out
        else np.zeros((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)
    )
    norms = np.linalg.norm(full, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return full / norms


def embedding_one(text: str) -> np.ndarray:
    return embedding_batch([text], batch_size=1, timeout=60)[0]
