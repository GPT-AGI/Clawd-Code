"""Smoke tests for the Ollama provider.

Live tests require a running Ollama instance on localhost:11434; they
are skipped otherwise so CI stays hermetic.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from src.providers import PROVIDER_INFO, get_provider_class


def _ollama_alive() -> bool:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=1
        ) as r:
            json.load(r)
        return True
    except (urllib.error.URLError, TimeoutError, Exception):
        return False


def test_provider_registered():
    assert "ollama" in PROVIDER_INFO
    assert "huihui_ai/qwen3.5-abliterated:4B" in PROVIDER_INFO["ollama"]["available_models"]


def test_get_provider_class_resolves():
    cls = get_provider_class("ollama")
    assert cls.__name__ == "OllamaProvider"


def test_provider_constructs_without_ollama_running():
    from src.providers.ollama_provider import OllamaProvider

    prov = OllamaProvider(base_url="http://localhost:11434", model="llama3.2:3b")
    assert prov.base_url == "http://localhost:11434"
    assert prov.model == "llama3.2:3b"
    assert prov.think is False  # default


def test_think_env_var(monkeypatch):
    from src.providers.ollama_provider import OllamaProvider

    monkeypatch.setenv("OLLAMA_THINK", "true")
    prov = OllamaProvider()
    assert prov.think is True


@pytest.mark.skipif(not _ollama_alive(), reason="Ollama not running on localhost:11434")
def test_list_models_live():
    from src.providers.ollama_provider import OllamaProvider

    prov = OllamaProvider()
    models = prov.get_available_models()
    assert isinstance(models, list)
    # Local test host has at least one model
    assert len(models) >= 1


@pytest.mark.skipif(not _ollama_alive(), reason="Ollama not running on localhost:11434")
def test_chat_non_streaming_live():
    from src.providers.ollama_provider import OllamaProvider

    prov = OllamaProvider(model="llama3.2:3b")
    resp = prov.chat(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=20,
        temperature=0.0,
    )
    assert isinstance(resp.content, str)
    assert len(resp.content) > 0
    assert resp.finish_reason in ("stop", "length")
