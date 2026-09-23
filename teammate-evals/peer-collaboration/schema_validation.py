from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str) and not _matches_type(value, expected):
        return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in value and isinstance(child, dict):
                errors.extend(_validate(value[key], child, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{index}]"))
    return errors


def validate_file(document: str | Path, schema_name: str) -> list[str]:
    value = json.loads(Path(document).read_text(encoding="utf-8"))
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    return _validate(value, schema, "$")


def validate_run(run_dir: str | Path) -> None:
    directory = Path(run_dir)
    errors = [
        *validate_file(directory / "manifest.json", "manifest.schema.json"),
        *validate_file(directory / "result.json", "result.schema.json"),
    ]
    if errors:
        raise ValueError("peer run schema validation failed:\n" + "\n".join(errors))
