import json

import pytest

from memoryfield_tool import embed


@pytest.fixture(autouse=True)
def _reset_embed_failed():
    embed._EMBED_FAILED = False
    yield
    embed._EMBED_FAILED = False


def test_embed_success(monkeypatch):
    monkeypatch.setattr(
        embed.ollama,
        "embed",
        lambda model, input, truncate: {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
    )
    result = embed.embed_texts(["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_failure_short_circuits(monkeypatch):
    calls = {"n": 0}

    def boom(model, input, truncate):
        calls["n"] += 1
        raise RuntimeError("ollama down")

    monkeypatch.setattr(embed.ollama, "embed", boom)

    assert embed.embed_texts(["a"]) is None
    assert embed.embed_texts(["b"]) is None
    assert calls["n"] == 1


def test_embed_model_constants():
    assert embed.MODEL_CODE == "nomic-embed-text-v1.5"
    assert embed.OLLAMA_MODEL == "nomic-embed-text"


def test_embed_openai_backend_selected_by_env(monkeypatch):
    monkeypatch.setenv(embed.BASE_URL_ENV, "http://127.0.0.1:8090/v1")
    monkeypatch.setenv(embed.MODEL_ENV, "nomic-embed-text-v1.5")

    calls = {}

    def fake_openai(texts, base_url, model):
        calls.update(texts=texts, base_url=base_url, model=model)
        return [[0.1], [0.2]]

    monkeypatch.setattr(embed, "_embed_openai", fake_openai)
    monkeypatch.setattr(embed.ollama, "embed", lambda *a, **k: pytest.fail("ollama used"))

    assert embed.embed_texts(["a", "b"]) == [[0.1], [0.2]]
    assert calls == {
        "texts": ["a", "b"],
        "base_url": "http://127.0.0.1:8090/v1",
        "model": "nomic-embed-text-v1.5",
    }


def test_embed_openai_model_defaults(monkeypatch):
    monkeypatch.setenv(embed.BASE_URL_ENV, "http://127.0.0.1:8090/v1")
    monkeypatch.delenv(embed.MODEL_ENV, raising=False)

    seen = {}

    def fake_openai(texts, base_url, model):
        seen["model"] = model
        return [[0.5]]

    monkeypatch.setattr(embed, "_embed_openai", fake_openai)

    assert embed.embed_texts(["x"]) == [[0.5]]
    assert seen["model"] == embed.DEFAULT_OPENAI_MODEL


def test_embed_openai_failure_short_circuits(monkeypatch):
    monkeypatch.setenv(embed.BASE_URL_ENV, "http://127.0.0.1:9/v1")
    calls = {"n": 0}

    def boom(texts, base_url, model):
        calls["n"] += 1
        raise RuntimeError("server down")

    monkeypatch.setattr(embed, "_embed_openai", boom)

    assert embed.embed_texts(["a"]) is None
    assert embed.embed_texts(["b"]) is None
    assert calls["n"] == 1


def test_embed_openai_request_and_parsing(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(embed.urllib.request, "urlopen", fake_urlopen)

    result = embed._embed_openai(["hello", "world"], "http://127.0.0.1:8090/v1", "m")

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "http://127.0.0.1:8090/v1/embeddings"
    assert captured["payload"] == {"model": "m", "input": ["hello", "world"]}
    assert captured["timeout"] == 60
