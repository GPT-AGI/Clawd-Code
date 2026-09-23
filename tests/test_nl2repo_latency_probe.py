from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "teammate-evals"
    / "nl2repo-pilot"
    / "latency_probe.py"
)
SPEC = importlib.util.spec_from_file_location("nl2repo_latency_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class TestNL2RepoLatencyProbe(unittest.TestCase):
    def test_live_probe_is_disabled_even_when_global_pool_is_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SystemExit, "global-pool-only"):
                probe.main(["--upstream-root", str(root)])

    def test_dry_run_does_not_claim_model_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(probe, "find_upstream_root", return_value=root), patch.object(
                probe, "load_tasks", return_value=[]
            ), patch.object(probe, "reject_if_global_pool_active") as guard:
                with self.assertRaisesRegex(ValueError, "cannot select"):
                    probe.main(["--dry-run", "--subset-size", "1"])
                guard.assert_not_called()

    def test_parse_concurrency_sweep(self) -> None:
        self.assertEqual(probe.parse_concurrency_sweep("1,2,4,8"), [1, 2, 4, 8])
        with self.assertRaisesRegex(Exception, "positive"):
            probe.parse_concurrency_sweep("1,0,4")
        with self.assertRaisesRegex(Exception, "unique"):
            probe.parse_concurrency_sweep("1,2,2")

    def test_task_selection_is_deterministic_and_respects_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [
                probe.TaskPrompt(
                    name=f"task-{index:02d}",
                    path=root / f"task-{index:02d}.md",
                    prompt_bytes=index * 100,
                    difficulty="Easy",
                )
                for index in range(1, 41)
            ]
            selected_a, eligible_a = probe.select_tasks(tasks, 8, 1234, 2_000)
            selected_b, eligible_b = probe.select_tasks(reversed(tasks), 8, 1234, 2_000)

        self.assertEqual(eligible_a, 20)
        self.assertEqual(eligible_b, 20)
        self.assertEqual(
            [task.name for task in selected_a],
            [task.name for task in selected_b],
        )
        self.assertTrue(all(task.prompt_bytes <= 2_000 for task in selected_a))

    def test_summary_reports_latency_tokens_and_errors(self) -> None:
        results = [
            probe.RequestResult(
                task="one",
                prompt_bytes=100,
                difficulty="Easy",
                success=True,
                ttft_seconds=1.0,
                latency_seconds=2.0,
                input_tokens=10,
                output_tokens=2,
                output_chars=5,
                finish_reason="stop",
            ),
            probe.RequestResult(
                task="two",
                prompt_bytes=200,
                difficulty="Medium",
                success=True,
                ttft_seconds=3.0,
                latency_seconds=4.0,
                input_tokens=20,
                output_tokens=3,
                output_chars=6,
                finish_reason="stop",
            ),
            probe.RequestResult(
                task="three",
                prompt_bytes=300,
                difficulty="Hard",
                success=False,
                ttft_seconds=None,
                latency_seconds=0.5,
                input_tokens=0,
                output_tokens=0,
                output_chars=0,
                finish_reason=None,
                error_type="RateLimitError",
                error_status=429,
            ),
        ]

        summary = probe.summarize(results, duration_seconds=5.0)

        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["error_breakdown"], {"RateLimitError:429": 1})
        self.assertEqual(summary["input_tokens"], 30)
        self.assertEqual(summary["output_tokens"], 5)
        self.assertAlmostEqual(summary["requests_per_second"], 0.4)
        self.assertAlmostEqual(summary["ttft_seconds"]["p50"], 2.0)
        self.assertAlmostEqual(summary["latency_seconds"]["p95"], 3.9)

    def test_comparison_includes_scaling_efficiency_and_paired_ratios(self) -> None:
        baseline_results = [
            probe.RequestResult("one", 1, "", True, 1.0, 2.0, 1, 1, 1, "stop"),
            probe.RequestResult("two", 1, "", True, 1.0, 4.0, 1, 1, 1, "stop"),
        ]
        concurrent_results = [
            probe.RequestResult("one", 1, "", True, 2.0, 4.0, 1, 1, 1, "stop"),
            probe.RequestResult("two", 1, "", True, 2.0, 12.0, 1, 1, 1, "stop"),
        ]
        baseline = probe.summarize(baseline_results, 8.0)
        concurrent = probe.summarize(concurrent_results, 2.0)

        comparison = probe.compare_runs(
            baseline,
            concurrent,
            baseline_concurrency=1,
            concurrency=4,
            baseline_results=baseline_results,
            concurrent_results=concurrent_results,
        )

        self.assertAlmostEqual(comparison["throughput_scale"], 4.0)
        self.assertAlmostEqual(comparison["scaling_efficiency"], 1.0)
        self.assertAlmostEqual(comparison["ttft_p50_ratio"], 2.0)
        self.assertAlmostEqual(comparison["paired_latency_ratio"]["p50"], 2.5)


if __name__ == "__main__":
    unittest.main()
