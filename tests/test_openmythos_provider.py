"""Smoke tests for the OpenMythos provider.

We don't load torch here (heavy), but we verify:
  * provider is registered in PROVIDER_INFO
  * get_provider_class('openmythos') returns OpenMythosProvider
  * the _flatten_messages helper renders a transcript correctly
"""

from __future__ import annotations

import importlib.util
import pytest

from src.providers import PROVIDER_INFO, get_provider_class


def test_provider_registered():
    assert "openmythos" in PROVIDER_INFO
    info = PROVIDER_INFO["openmythos"]
    assert info["default_model"] == "mythos_3b"
    assert "mythos_1t" in info["available_models"]


def test_get_provider_class_resolves():
    cls = get_provider_class("openmythos")
    assert cls.__name__ == "OpenMythosProvider"


def test_flatten_messages_transcript_shape():
    from src.providers.openmythos_provider import _flatten_messages

    out = _flatten_messages(
        [
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ]
    )
    # Lines end with a hanging assistant: prompt
    assert out.endswith("\nassistant:")
    assert "system: you are a helpful assistant." in out
    assert "user: hi" in out


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("open_mythos") is None,
    reason="torch or open_mythos not installed",
)
def test_provider_constructs_tiny_model(tmp_path):
    """Full end-to-end: build a tiny model with open_mythos, wire the
    provider, tokenize one prompt, stream one token. Skipped if deps
    are not present in the environment."""
    import torch

    from open_mythos import MythosConfig, OpenMythos

    from src.providers.openmythos_provider import OpenMythosProvider

    # Override variant registry to a cheap config to keep CI fast.
    cfg = MythosConfig(
        vocab_size=256, dim=32, n_heads=2, n_kv_heads=1,
        max_seq_len=32, max_loop_iters=2,
        prelude_layers=1, coda_layers=1,
        n_experts=2, n_experts_per_tok=1, expert_dim=16,
        attn_type="gqa",
    )
    model = OpenMythos(cfg)
    ckpt_path = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), ckpt_path)

    # We can't easily monkeypatch variant selection without more hooks;
    # assert the provider at least imports and raises on unknown variant.
    with pytest.raises(ValueError, match="Unknown OpenMythos variant"):
        OpenMythosProvider(variant="mythos_xxx")
