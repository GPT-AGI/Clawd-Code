"""Configuration management for Clawd Codex."""

from __future__ import annotations

import json
import base64
import os
from pathlib import Path
from typing import Any, Optional


MODEL_PROFILES: dict[str, dict[str, str]] = {
    "glm5": {
        "provider": "anthropic",
        "base_url": "https://api.z.ai/api/anthropic",
        "default_model": "glm-5.2",
    },
    "qwen3.5": {
        "provider": "qwen",
        "base_url": (
            "https://ms-mnhdj86z-100034032793-sw.gw.ap-zhongwei.ti.tencentcs.com/"
            "ms-mnhdj86z/v1"
        ),
        "default_model": "ms-mnhdj86z",
    },
}

MODEL_PROFILE_ALIASES = {
    "glm": "glm5",
    "glm5": "glm5",
    "glm-5": "glm5",
    "glm5.2": "glm5",
    "zhipu-glm5": "glm5",
    "qwen": "qwen3.5",
    "qwen3.5": "qwen3.5",
}


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    config_dir = Path.home() / ".clawd"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def _get_default_config_from_providers() -> dict[str, Any]:
    """Build default config using provider info registry."""
    from src.providers import PROVIDER_INFO

    return {
        "default_provider": "anthropic",
        "active_profile": None,
        "providers": {
            name: {
                "api_key": "",
                "base_url": info["default_base_url"],
                "default_model": info["default_model"],
            }
            for name, info in PROVIDER_INFO.items()
        },
        "session": {
            "auto_save": True,
            "max_history": 300
        }
    }


def get_default_config() -> dict[str, Any]:
    """Generate default configuration."""
    return _get_default_config_from_providers()


def _merge_default_config(config: dict[str, Any]) -> bool:
    """Add newly introduced providers/settings without overwriting user values."""
    defaults = get_default_config()
    changed = False
    for key in ("default_provider", "active_profile", "session"):
        if key not in config:
            config[key] = defaults[key]
            changed = True

    providers = config.get("providers")
    if not isinstance(providers, dict):
        config["providers"] = defaults["providers"]
        return True
    for name, provider_defaults in defaults["providers"].items():
        provider = providers.get(name)
        if not isinstance(provider, dict):
            providers[name] = dict(provider_defaults)
            changed = True
            continue
        for key, value in provider_defaults.items():
            if key not in provider:
                provider[key] = value
                changed = True
    return changed


def _encode_api_key(api_key: str) -> str:
    """Encode API key for basic obfuscation."""
    return base64.b64encode(api_key.encode()).decode()


def _decode_api_key(encoded_key: str) -> str:
    """Decode API key."""
    try:
        return base64.b64decode(encoded_key.encode()).decode()
    except Exception:
        # If decoding fails, return as-is (might be plain text)
        return encoded_key


def load_config() -> dict[str, Any]:
    """Load configuration from file.

    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config
        config = get_default_config()
        save_config(config)
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("configuration root must be an object")
        changed = _merge_default_config(config)

        # Decode API keys
        for provider_name, provider_config in config.get("providers", {}).items():
            if provider_config.get("api_key"):
                provider_config["api_key"] = _decode_api_key(provider_config["api_key"])

        if changed:
            save_config(config)

        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid modifying the original
    config_copy = json.loads(json.dumps(config))

    # Encode API keys
    for provider_name, provider_config in config_copy.get("providers", {}).items():
        if provider_config.get("api_key"):
            provider_config["api_key"] = _encode_api_key(provider_config["api_key"])

    if os.name == "nt":
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, indent=2, ensure_ascii=False)
    else:
        fd = os.open(
            config_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, indent=2, ensure_ascii=False)
        os.chmod(config_path, 0o600)


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific provider.

    Args:
        provider: Provider name (anthropic, openai, glm, minimax)

    Returns:
        Provider configuration dictionary
    """
    from src.providers import normalize_provider_name

    provider = normalize_provider_name(provider)
    config = load_config()
    providers = config.get("providers", {})

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider]


def set_api_key(provider: str, api_key: str, base_url: Optional[str] = None,
                default_model: Optional[str] = None) -> None:
    """Set API key for a provider.

    Args:
        provider: Provider name (anthropic, openai, glm, minimax)
        api_key: API key to set
        base_url: Optional base URL override
        default_model: Optional default model override
    """
    from src.providers import normalize_provider_name

    provider = normalize_provider_name(provider)
    config = load_config()

    if provider not in config.get("providers", {}):
        # Add new provider if it doesn't exist
        if "providers" not in config:
            config["providers"] = {}
        config["providers"][provider] = {}

    config["providers"][provider]["api_key"] = api_key

    if base_url is not None:
        config["providers"][provider]["base_url"] = base_url

    if default_model is not None:
        config["providers"][provider]["default_model"] = default_model

    save_config(config)


def set_default_provider(provider: str) -> None:
    """Set the default provider.

    Args:
        provider: Provider name
    """
    from src.providers import PROVIDER_INFO, normalize_provider_name

    provider = normalize_provider_name(provider)
    if provider not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider}")
    config = load_config()
    config["default_provider"] = provider
    config["active_profile"] = None
    save_config(config)


def use_model_profile(profile: str) -> dict[str, str]:
    """Activate a named model profile without changing its saved API key."""
    normalized = MODEL_PROFILE_ALIASES.get(profile.strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown model profile: {profile}; choose one of {choices}")
    selected = MODEL_PROFILES[normalized]
    config = load_config()
    provider = selected["provider"]
    provider_config = config["providers"][provider]
    provider_config["base_url"] = selected["base_url"]
    provider_config["default_model"] = selected["default_model"]
    config["default_provider"] = provider
    config["active_profile"] = normalized
    save_config(config)
    return {"name": normalized, **selected}


def get_default_provider() -> str:
    """Get the default provider.

    Returns:
        Default provider name
    """
    config = load_config()
    return config.get("default_provider", "anthropic")
