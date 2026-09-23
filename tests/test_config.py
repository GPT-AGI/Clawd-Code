"""Tests for configuration management."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import base64
import os

from src.config import (
    get_config_path,
    get_default_config,
    load_config,
    save_config,
    get_provider_config,
    set_api_key,
    set_default_provider,
    get_default_provider,
    use_model_profile,
    _encode_api_key,
    _decode_api_key
)


class TestConfigPath(unittest.TestCase):
    """Test configuration path functions."""

    def test_get_config_path(self):
        """Test getting config path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.config.Path.home', return_value=Path(temp_dir)):
                path = get_config_path()
                expected = Path(temp_dir) / ".clawd" / "config.json"
                self.assertEqual(path, expected)

    def test_config_dir_created(self):
        """Test that config directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            config_dir = home / ".clawd"
            self.assertFalse(config_dir.exists())

            with patch('src.config.Path.home', return_value=home):
                get_config_path()
                self.assertTrue(config_dir.exists())


class TestDefaultConfig(unittest.TestCase):
    """Test default configuration."""

    def test_get_default_config(self):
        """Test getting default config."""
        config = get_default_config()

        self.assertIn("default_provider", config)
        self.assertIn("providers", config)
        self.assertIn("anthropic", config["providers"])
        self.assertIn("openai", config["providers"])
        self.assertIn("glm", config["providers"])
        self.assertIn("qwen", config["providers"])

    def test_default_provider_is_anthropic(self):
        """Test that default provider is Anthropic."""
        config = get_default_config()
        self.assertEqual(config["default_provider"], "anthropic")

    def test_default_models(self):
        """Test default models for providers."""
        config = get_default_config()
        self.assertEqual(
            config["providers"]["anthropic"]["default_model"],
            "claude-sonnet-4-6"
        )
        self.assertEqual(
            config["providers"]["openai"]["default_model"],
            "gpt-5.4"
        )
        self.assertEqual(
            config["providers"]["glm"]["default_model"],
            "zai/glm-5"
        )
        self.assertEqual(config["providers"]["qwen"]["default_model"], "ms-mnhdj86z")
        self.assertTrue(config["providers"]["qwen"]["base_url"].endswith("/ms-mnhdj86z/v1"))

    def test_existing_config_is_migrated_with_qwen_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps({"default_provider": "glm", "providers": {"glm": {"api_key": ""}}}),
                encoding="utf-8",
            )
            with patch('src.config.get_config_path', return_value=config_path):
                config = load_config()

            self.assertIn("qwen", config["providers"])
            self.assertEqual(config["providers"]["glm"]["default_model"], "zai/glm-5")


class TestAPIKeyEncoding(unittest.TestCase):
    """Test API key encoding/decoding."""

    def test_encode_api_key(self):
        """Test API key encoding."""
        api_key = "test_api_key_123"
        encoded = _encode_api_key(api_key)
        expected = base64.b64encode(api_key.encode()).decode()
        self.assertEqual(encoded, expected)

    def test_decode_api_key(self):
        """Test API key decoding."""
        api_key = "test_api_key_123"
        encoded = base64.b64encode(api_key.encode()).decode()
        decoded = _decode_api_key(encoded)
        self.assertEqual(decoded, api_key)

    def test_decode_plain_text(self):
        """Test decoding plain text (not encoded)."""
        plain_key = "plain_api_key"
        decoded = _decode_api_key(plain_key)
        self.assertEqual(decoded, plain_key)


class TestLoadSaveConfig(unittest.TestCase):
    """Test loading and saving configuration."""

    def test_save_and_load_config(self):
        """Test save and load roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                config = {
                    "default_provider": "glm",
                    "providers": {
                        "glm": {
                            "api_key": "test_key",
                            "base_url": "https://example.com",
                            "default_model": "glm-4"
                        }
                    }
                }

                save_config(config)
                loaded = load_config()

                self.assertEqual(loaded["default_provider"], "glm")
                self.assertEqual(
                    loaded["providers"]["glm"]["api_key"],
                    "test_key"
                )

    def test_load_config_creates_default(self):
        """Test that loading non-existent config creates default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                config = load_config()

                self.assertIn("default_provider", config)
                self.assertIn("providers", config)
                self.assertTrue(config_path.exists())

    def test_api_keys_encoded_on_save(self):
        """Test that API keys are encoded when saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                config = {
                    "default_provider": "glm",
                    "providers": {
                        "glm": {
                            "api_key": "plain_text_key",
                            "base_url": "https://example.com",
                            "default_model": "glm-4"
                        }
                    }
                }

                save_config(config)

                # Read raw file to check encoding
                with open(config_path, 'r') as f:
                    raw_data = json.load(f)

                encoded_key = raw_data["providers"]["glm"]["api_key"]
                self.assertNotEqual(encoded_key, "plain_text_key")

                # Verify it can be decoded
                decoded_key = _decode_api_key(encoded_key)
                self.assertEqual(decoded_key, "plain_text_key")

    def test_api_keys_decoded_on_load(self):
        """Test that API keys are decoded when loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            # Create config with encoded key
            encoded_key = _encode_api_key("secret_key")
            raw_config = {
                "default_provider": "glm",
                "providers": {
                    "glm": {
                        "api_key": encoded_key,
                        "base_url": "https://example.com",
                        "default_model": "glm-4"
                    }
                }
            }

            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(raw_config, f)

            with patch('src.config.get_config_path', return_value=config_path):
                config = load_config()

                # API key should be decoded
                self.assertEqual(
                    config["providers"]["glm"]["api_key"],
                    "secret_key"
                )

    @unittest.skipIf(os.name == "nt", "POSIX file permission semantics differ on Windows")
    def test_config_file_permissions_restricted_on_save(self):
        """Test that saved config uses owner-only permissions on POSIX systems."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                save_config(get_default_config())
                mode = config_path.stat().st_mode & 0o777
                self.assertEqual(mode, 0o600)


class TestProviderConfig(unittest.TestCase):
    """Test provider-specific configuration."""

    def test_get_provider_config(self):
        """Test getting provider config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                glm_config = get_provider_config("glm")

                self.assertIn("api_key", glm_config)
                self.assertIn("base_url", glm_config)
                self.assertIn("default_model", glm_config)

    def test_get_unknown_provider(self):
        """Test getting unknown provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                with self.assertRaises(ValueError) as context:
                    get_provider_config("unknown")

                self.assertIn("Unknown provider", str(context.exception))


class TestSetAPIKey(unittest.TestCase):
    """Test setting API keys."""

    def test_set_api_key(self):
        """Test setting API key for provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_api_key("glm", "new_api_key")

                config = load_config()
                self.assertEqual(
                    config["providers"]["glm"]["api_key"],
                    "new_api_key"
                )

    def test_set_api_key_with_options(self):
        """Test setting API key with base URL and model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_api_key(
                    "glm",
                    "new_api_key",
                    base_url="https://custom.url",
                    default_model="custom-model"
                )

                config = load_config()
                self.assertEqual(
                    config["providers"]["glm"]["api_key"],
                    "new_api_key"
                )
                self.assertEqual(
                    config["providers"]["glm"]["base_url"],
                    "https://custom.url"
                )
                self.assertEqual(
                    config["providers"]["glm"]["default_model"],
                    "custom-model"
                )


class TestDefaultProvider(unittest.TestCase):
    """Test default provider management."""

    def test_set_default_provider(self):
        """Test setting default provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_default_provider("openai")

                provider = get_default_provider()
                self.assertEqual(provider, "openai")

    def test_get_default_provider(self):
        """Test getting default provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                provider = get_default_provider()
                self.assertEqual(provider, "anthropic")

    def test_model_profiles_switch_protocol_endpoint_and_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            with patch('src.config.get_config_path', return_value=config_path):
                use_model_profile("qwen3.5")
                self.assertEqual(get_default_provider(), "qwen")
                qwen = get_provider_config("qwen")
                self.assertEqual(qwen["default_model"], "ms-mnhdj86z")

                use_model_profile("glm5")
                self.assertEqual(get_default_provider(), "anthropic")
                glm = get_provider_config("anthropic")
                self.assertEqual(glm["base_url"], "https://api.z.ai/api/anthropic")
                self.assertEqual(glm["default_model"], "glm-5.2")

    def test_profile_switch_preserves_saved_api_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            with patch('src.config.get_config_path', return_value=config_path):
                set_api_key("anthropic", "glm-secret")
                set_api_key("qwen", "qwen-secret")
                use_model_profile("qwen")
                use_model_profile("glm")
                config = load_config()

            self.assertEqual(config["providers"]["anthropic"]["api_key"], "glm-secret")
            self.assertEqual(config["providers"]["qwen"]["api_key"], "qwen-secret")


if __name__ == '__main__':
    unittest.main()
