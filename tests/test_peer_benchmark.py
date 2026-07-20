from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "teammate-evals" / "peer-collaboration"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PeerBenchmarkTests(unittest.TestCase):
    def test_scripted_coupled_smoke_and_schemas(self) -> None:
        smoke = load_module("peer_scripted_smoke", BENCHMARK / "scripted_smoke.py")
        validator = load_module(
            "peer_schema_validation", BENCHMARK / "schema_validation.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = smoke.run_smoke(Path(temporary))
            run_dir = Path(result["result_path"]).parent
            validator.validate_run(run_dir)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["acceptance"]["exit_code"], 0)
            self.assertTrue(result["acceptance"]["stderr"])
            self.assertEqual(len(result["messages"]), 1)
            self.assertEqual(result["messages"][0]["status"], "consumed")
            self.assertEqual(result["orphan_threads"], [])

    def test_real_runner_requires_explicit_arguments(self) -> None:
        source = (BENCHMARK / "runner.py").read_text(encoding="utf-8")
        self.assertIn("--provider", source)
        self.assertIn("--model", source)
        self.assertNotIn("run_smoke(", source)


if __name__ == "__main__":
    unittest.main()
