from __future__ import annotations

import re
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolRegistry, ToolSpec


class ToolSearchTool:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ToolSearch",
            description="Search registered tools by name or keywords. Use '*' to list all tools.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        max_results = tool_input.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 50:
            raise ToolInputError("max_results must be an integer between 1 and 50")

        q = query.strip()
        lowered = q.lower()
        if lowered.startswith("select:"):
            name = q.split(":", 1)[1].strip()
            tool = self._registry.get(name)
            specs = [tool.spec()] if tool else []
            return self._result(query, specs)

        all_specs = self._registry.list_specs()
        if lowered in {"*", "all", "list all", "all tools"}:
            return self._result(query, all_specs[:max_results], total_matches=len(all_specs))

        query_terms = self._query_terms(lowered)
        scored: list[tuple[int, int, str, ToolSpec]] = []
        for spec in all_specs:
            name = spec.name.lower()
            aliases = " ".join(spec.aliases).lower()
            hay = f"{name} {aliases} {spec.description.lower()}"
            matched_terms = sum(1 for term in query_terms if term in hay)
            if lowered == name:
                rank = 0
            elif lowered in name:
                rank = 1
            elif lowered in hay:
                rank = 2
            elif matched_terms:
                rank = 3
            else:
                continue
            scored.append((rank, -matched_terms, name, spec))
        scored.sort(key=lambda item: item[:3])
        matched_specs = [item[3] for item in scored]
        return self._result(query, matched_specs[:max_results], total_matches=len(matched_specs))

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        terms = set(re.findall(r"[a-z0-9_]+", query))
        synonyms = {
            "cat": {"read"},
            "content": {"read"},
            "execute": {"bash"},
            "filesystem": {"file"},
            "local": {"file"},
            "modify": {"edit", "write"},
            "run": {"bash"},
            "shell": {"bash"},
        }
        expanded = set(terms)
        for term in terms:
            expanded.update(synonyms.get(term, set()))
        return expanded

    @staticmethod
    def _tool_details(spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": dict(spec.input_schema),
            **({"aliases": list(spec.aliases)} if spec.aliases else {}),
        }

    def _result(
        self,
        query: str,
        specs: list[ToolSpec],
        *,
        total_matches: int | None = None,
    ) -> ToolResult:
        return ToolResult(
            name="ToolSearch",
            output={
                "matches": [spec.name for spec in specs],
                "tools": [self._tool_details(spec) for spec in specs],
                "query": query,
                "total_matches": len(specs) if total_matches is None else total_matches,
                "total_deferred_tools": 0,
            },
        )
