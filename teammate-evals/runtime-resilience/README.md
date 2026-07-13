# Teammate Runtime Resilience Evaluation

This deterministic evaluation exercises scheduler behavior that is difficult
to verify reliably with one live-model task:

- expired-lease crash recovery
- automatic and lead-requested retries
- reviewer rejection followed by coder repair and re-review
- cooperative cancellation
- timeout, token, and turn budgets
- parallel ready-task execution without lost task updates
- isolated git worktree execution and integration

Run every scenario from the repository root:

```bash
.venv/bin/python teammate-evals/runtime-resilience/evaluate.py
```

Run one scenario:

```bash
.venv/bin/python teammate-evals/runtime-resilience/evaluate.py --scenario crash-resume
```

The evaluator uses scripted providers and temporary workspaces, so it does not
require an API key and does not modify the repository under test.
