# GLM-5.2 Baseline: Solo vs Team

Date: 2026-07-14

This is the first real-model baseline for the five-scenario benchmark. Each
scenario was run once in solo mode and once in team mode against the configured
Anthropic-compatible endpoint with model `glm-5.2`. Team mode used a serial
analyst -> implementer -> reviewer workflow with required handoff messages.
This records the original forced-pipeline v1 protocol; the current benchmark's
primary comparison is solo vs adaptive lead-controlled orchestration.

## Results

| Scenario | Solo quality | Team quality | Solo sec | Team sec | Solo tokens | Team tokens | Team protocol |
|---|---:|---:|---:|---:|---:|---:|:---:|
| config-migration | 100 | 100 | 103.689 | 534.832 | 15,596 | 88,180 | yes |
| csv-reconciliation | 100 | 100 | 155.862 | 495.247 | 38,744 | 98,291 | yes |
| invoice-allocation | 100 | 100 | 83.769 | 664.008 | 22,127 | 131,458 | yes |
| permission-policy | 100 | 100 | 183.874 | 839.041 | 41,740 | 241,899 | no |
| webhook-idempotency | 100 | 100 | 112.159 | 649.926 | 17,935 | 123,035 | no |

Aggregate results:

- Business acceptance: solo 5/5, team 5/5.
- Strict success, including the execution protocol: solo 5/5, team 3/5.
- Mean quality delta: 0 points.
- Mean elapsed time: solo 127.9 seconds, team 636.6 seconds (4.98x).
- Mean token use: solo 27,228, team 136,573 (5.02x).
- Protected task, requirements, and test files remained unchanged in all runs.

## What Happened

The team workflow did real collaborative work: all five runs created three
agents and three dependent tasks, completed all tasks, and persisted all three
required handoffs. On these small, explicit tasks, however, the extra analysis
and independent review did not improve acceptance quality because solo already
reached 100%.

Two team runs missed the strict protocol gate after producing correct code:

- `permission-policy` exceeded the team's 600 second timeout after the final
  task had completed. The lead then verified the tests and disbanded the team,
  leaving the historical team status as `cancelled`.
- `webhook-idempotency` exceeded its 300 second team timeout after all three
  tasks had completed and remained `failed`.

The runtime checks timeout and usage budgets before checking whether all tasks
are complete. That ordering can turn a just-completed workflow into a failure at
the next scheduler iteration. The benchmark now preserves metrics even when
`TeamDelete` removes the active team pointer by reading historical team state.

The traces also exposed a task API usability issue. Workers naturally used the
stable keys requested by the prompt, such as `analysis` and `review`, when
calling `TaskUpdate`, but that tool currently accepts only internal task IDs.
Workers recovered by listing tasks and retrying, at unnecessary token and tool
cost. One worker also tried the intuitive status value `done` instead of
`completed`.

## Decision

For work of this size, solo should remain the default. A three-role serial team
cost about five times as much without a quality gain in this sample. Team mode
is better reserved for work with genuinely separable parallel investigation,
large context that one agent cannot hold reliably, distinct ownership areas, or
high-risk changes where independent review is worth the added cost.

The next implementation priorities are:

1. Check all-tasks-complete before timeout and budget failure in `TeamRun`.
2. Resolve `TaskUpdate.taskId` by either internal ID or stable task key, and make
   accepted status values clearer.
3. Add adaptive orchestration and reviewer budgets so small tasks stay solo and
   expensive review loops are bounded.
4. Build benchmark v2 with hidden held-out tests, larger cross-module changes,
   parallelizable research, and repeated trials per cell.

## Limitations

This is a smoke benchmark, not a statistically significant model evaluation.
There is one sample per cell, and acceptance tests are visible to the agents
although their integrity is hash-checked. The baseline is useful for catching
large regressions and workflow problems; quality claims need repeated trials and
held-out tests.
