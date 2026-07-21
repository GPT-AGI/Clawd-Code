from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCENARIOS = {
    "crash-resume": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_recovers_expired_in_progress_lease"
    ),
    "retry": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_automatically_retries_transient_failure"
    ),
    "review-reject": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_reviewer_rejection_can_drive_repair_and_re_review"
    ),
    "cancel": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_cooperative_cancel_is_observed_after_active_model_call"
    ),
    "worker-stop": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_lead_stops_one_worker_without_cancelling_team"
    ),
    "budgets": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_turn_budget_limits_model_round_trips"
    ),
    "parallel": (
        "tests.test_teammate_resilience.TestTeammateResilience."
        "test_ready_tasks_run_in_parallel_without_lost_updates"
    ),
    "worktree": (
        "tests.test_teammate_resilience.TestTeammateWorktree."
        "test_auto_integrates_isolated_teammate_changes"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate teammate runtime resilience.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Run only the selected scenario; repeat for multiple scenarios.",
    )
    args = parser.parse_args()
    selected = args.scenario or list(SCENARIOS)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromName(SCENARIOS[name]) for name in selected
    )
    print(f"Teammate resilience scenarios: {', '.join(selected)}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"Resilience evaluation: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
