# Teammate Evaluation Task

Repair the member discount behavior in this order calculator. You must perform
the work as a teammate workflow rather than solving it as a single agent.

Use exactly these teammates:

- `researcher`: read `requirements.md` and the existing implementation, then
  identify every pricing rule and edge case. This teammate must not edit files.
- `coder`: wait until the researcher sends its findings, then implement the
  repair and run the acceptance checks.
- `reviewer`: wait for the coder's task, independently inspect the diff and run
  all acceptance checks. If any defect remains, send a concrete message to the
  coder and require a repair before approving.

The lead agent must:

1. Create the team and all three teammate records.
2. Create separate analysis, implementation, and review tasks with explicit
   dependencies.
3. Ensure the researcher sends its findings to the coder with `SendMessage`.
4. Ensure the coder reports its implementation and test result to the reviewer.
5. Ensure the reviewer reports approval or requested changes to the lead.
6. Wait for all tasks to complete before producing the final response.
7. Summarize changed files, test results, messages exchanged, and each
   teammate's contribution.

Do not modify `requirements.md`, `TASK.md`, or files under `checks/`. Do not
weaken or delete tests. The final command below must pass:

```bash
python -m unittest checks.order_acceptance -v
```
