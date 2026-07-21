from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "teammate-evals"
    / "nl2repo-pilot"
    / "reward_repair.py"
)
SPEC = importlib.util.spec_from_file_location("nl2repo_reward_repair", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def case_result(task: str, quality: float, *, error: str | None = None) -> dict[str, object]:
    hidden: dict[str, object] = {
        "pytest": {
            "expected": 10,
            "passed": int(quality / 10),
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "returncode": 1,
            "quality_score": quality,
            "all_passed": False,
        }
    }
    if error:
        hidden["error"] = error
    return {
        "task": task,
        "difficulty": "Easy",
        "mode": "adaptive",
        "quality_score": quality,
        "success": False,
        "agent_elapsed_s": 1.0,
        "usage": {"total_tokens": 0},
        "team": {"agents": [], "peer_messages": 0},
        "protocol_ok": True,
        "hidden_tests": hidden,
    }


class TestRewardRepair(unittest.TestCase):
    def test_main_always_routes_reward_repairs_through_global_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SystemExit, "global_pool_supervisor.py"):
                repair.main(["--run", str(root)])

    def test_discovery_selects_only_explicit_infrastructure_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "broken" / "adaptive" / "result.json",
                case_result("broken", 0, error="Docker build failed"),
            )
            write_json(
                root / "valid-zero" / "adaptive" / "result.json",
                case_result("valid-zero", 0),
            )
            failures = repair.discover_infrastructure_failures(root)

        self.assertEqual(failures, [("broken", "adaptive", "Docker build failed")])

    def test_refresh_aggregate_prefers_repaired_case_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = case_result("sample", 0, error="old failure")
            current = case_result("sample", 70)
            write_json(root / "sample" / "adaptive" / "result.json", current)
            write_json(
                root / "results.json",
                {
                    "run_id": "run",
                    "upstream_ref": "ref",
                    "results": [old],
                },
            )

            changed = repair.refresh_aggregate(root)
            aggregate = json.loads((root / "results.json").read_text(encoding="utf-8"))
            report = (root / "REPORT.md").read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertEqual(aggregate["results"][0]["quality_score"], 70)
        self.assertIn("70", report)


if __name__ == "__main__":
    unittest.main()
