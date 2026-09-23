# Team Protocol v2

Protocol v2 is the repository-generation harness used by NL2Repo `adaptive-team-v2`
and `forced-team`. Its goal is to make delegation useful rather than ceremonial: a Team
receives credit only when independently owned work is actually executed, integrated, and
accepted by harness-controlled checks.

## Control flow

```text
TeamCreate(quality_gates=true)
  -> TeamPlan(replace, expected_revision)
  -> TeamRun
       -> task claimed
       -> worker writes audited against owned_files
       -> task produced
       -> task acceptance checks
       -> task accepted
       -> clean install/import/integration verification
  -> completed
```

`TeamPlan` is an atomic manifest. It contains the architecture contract, all workers, the
complete task DAG, file ownership, acceptance checks, validation profile, and execution
budget. Invalid plans return structured issues without materializing a partial Team. A
plan revision is immutable while work is running, and legacy incremental mutation tools
cannot alter a v2 plan.

## Planning invariants

- At least two distinct workers own real implementation tasks.
- At least two implementation owners can start immediately; frozen interfaces enable
  parallel work, while handoff interfaces add explicit DAG dependencies.
- Concrete `owned_files` may overlap only between tasks owned by the same worker.
- Persistent project tests are deliverables and must appear in `owned_files`. Each task
  also receives `.clawd/task-tests/<task-id>/` for disposable self-tests; that subtree
  is private to the task and excluded from the delivered repository. For compatibility
  with common test workflows, a task may create a previously absent, unreserved
  `tests/test_*.py` self-test; existing or task-reserved tests remain protected.
- Every implementation task has a behavioral acceptance check; file-existence, no-op,
  and fail-open commands are rejected.
- Worker models must use the configured endpoint/model policy.
- Replacing a failed plan clears stale execution settings and increments the revision.

## Runtime invariants

- Each write/edit is checked before execution. Shell commands are audited by a guarded
  before/after workspace snapshot. An out-of-scope mutation is sticky and moves the Team
  to `repair_required`.
- Task-private scratch and newly created local self-tests are included in the ownership
  audit, so a worker cannot modify another task's or any pre-existing test through a
  shell command.
- A worker completion produces evidence; it does not directly accept its task.
- The harness runs task acceptance checks, then changes `produced` to `accepted`.
- Final verification always uses a fresh environment and runs install, import, and
  integration stages. Cleanup runs even when verification fails.
- Provider, AGS, and transport failures pause the Team as retryable infrastructure work;
  they do not become candidate failures.
- `completed` is terminal for `TeamRun`. Repair requires a new plan revision.
  `TeamAbort` is terminal and cannot be resumed or replanned.

## Evaluation metrics

The benchmark reports three orthogonal aggregates:

- `Q = mean(code_quality_score)` over scoreable hidden-test results.
- `P = passed protocol cases / protocol-eligible cases`.
- `E = mean(effective_quality_score)`, where valid delivery with a failed Team protocol
  receives zero protocol credit.

Infrastructure failures have `protocol_status=not_evaluated`, null Q/E, and all metric
eligibility flags false. They remain retryable. A protocol failure can still have a valid
Q, which lets the evaluation distinguish poor code from poor collaboration discipline.

## Compatibility

Protocol-v1 Teams keep the incremental
`TeamConfigure -> TeammateCreate -> TaskCreate -> TeamRun -> TeamVerify` workflow.
Protocol v2 deliberately blocks those mutation paths after an atomic plan is committed.
