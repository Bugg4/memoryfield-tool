import json
import os
import urllib.request

import ollama

MODEL_CODE = "nomic-embed-text-v1.5"
OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_OPENAI_MODEL = "nomic-embed-text-v1.5"

# Backend selection.  By default embeddings come from the local Ollama API.
# Setting MEMORYFIELD_EMBED_BASE_URL switches to an OpenAI-compatible
# `/embeddings` endpoint (llama.cpp's llama-server, LM Studio, vLLM, ...).
# The base URL must include any `/v1` suffix the server expects, for example
# `http://127.0.0.1:8090/v1`.
BASE_URL_ENV = "MEMORYFIELD_EMBED_BASE_URL"
MODEL_ENV = "MEMORYFIELD_EMBED_MODEL"

_EMBED_FAILED = False


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    return list(ollama.embed(model=OLLAMA_MODEL, input=texts, truncate=True)["embeddings"])


def _embed_openai(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read())
    data = sorted(body["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    global _EMBED_FAILED
    if _EMBED_FAILED:
        return None
    try:
        base_url = os.environ.get(BASE_URL_ENV, "").strip()
        if base_url:
            model = os.environ.get(MODEL_ENV, "").strip() or DEFAULT_OPENAI_MODEL
            return _embed_openai(texts, base_url, model)
        return _embed_ollama(texts)
    except Exception as e:
        print(f"embedding failed: {e}")
        _EMBED_FAILED = True
        return None
