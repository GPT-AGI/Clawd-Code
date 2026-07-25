from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolExecutionError
from ..protocol import ToolResult
from ..registry import ToolSpec


class SkillTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Skill",
            description="Execute a prompt-based SKILL.md skill or a legacy Python skill module.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "skill": {"type": "string"},
                    "args": {"type": "string"},
                    "name": {"type": "string"},
                    "input": {"type": "object"},
                },
                "anyOf": [
                    {"required": ["skill"]},
                    {"required": ["name"]},
                ],
            },
            is_destructive=False,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if "skill" in tool_input:
            return self._run_markdown_skill(tool_input, context)
        return self._run_legacy_python_skill(tool_input, context)

    def _run_markdown_skill(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        skill_name = tool_input.get("skill")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ToolInputError("skill must be a non-empty string")
        args = tool_input.get("args", "")
        if not isinstance(args, str):
            raise ToolInputError("args must be a string when provided")

        normalized = skill_name.strip()
        if normalized.startswith("/"):
            normalized = normalized[1:]

        from ...skills.argument_substitution import substitute_arguments
        from ...skills.loader import get_all_skills

        cwd = context.cwd or context.workspace_root
        skills = get_all_skills(project_root=cwd)
        skill = next((s for s in skills if s.name == normalized), None)
        if skill is None:
            return ToolResult(
                name="Skill",
                output={"success": False, "error": f"unknown skill: {normalized}", "commandName": normalized},
                is_error=True,
            )
        if skill.disable_model_invocation:
            return ToolResult(
                name="Skill",
                output={
                    "success": False,
                    "error": f"skill {normalized} cannot be invoked (disable-model-invocation: true)",
                    "commandName": normalized,
                },
                is_error=True,
            )

        content = skill.markdown_content
        content = substitute_arguments(
            content,
            args,
            append_if_no_placeholder=True,
            argument_names=skill.arg_names,
        )
        if skill.skill_root:
            content = f"Base directory for this skill: {skill.skill_root}\n\n{content}"
            skill_dir = skill.skill_root.replace("\\", "/")
            content = content.replace("${CLAUDE_SKILL_DIR}", skill_dir)

        return ToolResult(
            name="Skill",
            output={
                "success": True,
                "commandName": normalized,
                "status": "inline",
                "allowedTools": list(skill.allowed_tools) if skill.allowed_tools else [],
                "model": skill.model,
                "loadedFrom": skill.loaded_from,
                "skillRoot": skill.skill_root,
                "prompt": content,
            },
        )

    def _run_legacy_python_skill(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = tool_input.get("name")
        if not isinstance(name, str) or not name:
            raise ToolInputError("name must be a non-empty string")
        payload = tool_input.get("input") or {}
        if not isinstance(payload, dict):
            raise ToolInputError("input must be an object when provided")

        clawd_skills_dir = os.environ.get("CLAWD_SKILLS_DIR")
        if clawd_skills_dir:
            skill_dir = Path(clawd_skills_dir).expanduser().resolve()
        else:
            for d in (Path.home() / ".clawd" / "skills", Path.home() / ".claude" / "skills"):
                if d.exists() and d.is_dir():
                    skill_dir = d
                    break
            else:
                skill_dir = Path.home() / ".clawd" / "skills"
        file_path = (skill_dir / f"{name}.py").resolve()
        if not file_path.exists():
            return ToolResult(name="Skill", output={"error": f"skill not found: {name}"}, is_error=True)

        module = _load_module(file_path, module_prefix="clawd_skill_")
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            raise ToolExecutionError(f"skill {name} does not export a callable run(input, context)")

        out = run_fn(payload, context)
        return ToolResult(name="Skill", output={"name": name, "output": out})


def _load_module(path: Path, *, module_prefix: str) -> types.ModuleType:
    module_name = f"{module_prefix}{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ToolExecutionError(f"failed to import: {path}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, types.ModuleType)
    spec.loader.exec_module(module)
    return module
