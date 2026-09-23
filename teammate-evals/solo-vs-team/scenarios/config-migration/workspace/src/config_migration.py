from __future__ import annotations

from typing import Any, Mapping


def migrate_config(raw: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    version = raw.get("version", 1)
    if version != 1:
        return dict(raw)
    endpoint = str(raw.get("endpoint", "")).replace(
        "${SERVICE_HOST}", env.get("SERVICE_HOST", "")
    )
    return {
        "version": 3,
        "service": {"url": endpoint, "auth": {"token": raw.get("token")}},
        "retry": {"max_attempts": int(raw.get("retries", 3))},
        "features": raw.get("features", {}),
    }


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if "token" in result:
        result["token"] = "[REDACTED]"
    return result
