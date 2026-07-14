# GLM 5.2 Baseline

Date: 2026-07-12

Configuration:

- Provider: Anthropic-compatible z.ai endpoint
- Model: `glm-5.2`
- Entry point: normal `clawd --stream` REPL
- Workspace: this fixture directory

Observed behavior before implementing the teammate runtime:

1. GLM 5.2 received the task and successfully used `Glob` to list the fixture.
2. It initially attempted to read local files with z.ai's HTTP-only `webReader`.
3. Repeated `ToolSearch` queries did not give the model a usable local `Read`,
   `Write`, `Edit`, or `Bash` invocation path.
4. No team or teammate sessions were created.
5. No messages or delegated tasks were produced.
6. The model stopped honestly and reported the missing capabilities rather than
   claiming that the task had succeeded.
7. Source files were unchanged after the run.

Initial acceptance result:

```text
Ran 6 tests
FAILED (failures=2)
```

Expected initial failures:

- Member discount incorrectly reduces shipping: actual `54.00`, expected
  `55.00`.
- Store credit incorrectly reduces shipping: actual `0.00`, expected `10.00`.

This baseline establishes that a successful future run must demonstrate both
working local engineering tools and genuine teammate orchestration.

## Tool-discovery fix verification

After improving `ToolSearch`, adding local-tool system guidance, and serializing
Anthropic-compatible tool results as JSON text, a second authorized GLM 5.2
smoke test succeeded:

```text
ToolSearch (Read)
Read (./requirements.md) - lines 1-16/16
```

GLM 5.2 returned the first pricing rule directly from the `Read` result. It did
not invoke `Bash`, `cat`, `webReader`, or another web tool. A separate smoke run
also invoked `Bash` successfully and reported the expected acceptance result of
four passing and two failing checks.

At that point, the remaining evaluation blocker was genuine teammate spawning,
messaging, and task scheduling rather than local engineering tool discovery.

## Native teammate run

Date: 2026-07-13

After adding the persistent teammate runtime and native trace instrumentation,
a fresh-copy GLM 5.2 run completed the full workflow without manual source
edits. The lead created exactly the required researcher, coder, and reviewer;
scheduled the dependency chain; and received all three required message
handoffs.

Combined evaluator result:

```text
Collaboration evidence: PASSED
Ran 6 tests
OK
Business acceptance: PASSED
```

Persisted trace summary:

- 237 native events
- 59 tool calls
- 3 teammate messages
- 67,945 aggregate input and output tokens
- 5 minutes 44 seconds elapsed
- all agents, tasks, and the team completed

Eight tool attempts failed during the run, including incorrect path casing,
unavailable Python aliases, expected failing baseline tests, and one malformed
review command. Each failure was visible in the trace and the agents recovered
without operator intervention. The trace was recorded natively rather than
reconstructed from teammate sessions.

## Resilient runtime release gate

Date: 2026-07-14

The release gate was repeated in a clean fixture copy after adding recovery,
leases, retries, cancellation, budgets, parallel scheduling, and worktree
support. The normal user configuration selected the Anthropic-compatible z.ai
endpoint and `glm-5.2`; no credential was copied into the fixture or repository.

The clean baseline again produced the two expected business failures and no
collaboration evidence. A subsequent real-model run created three teammates,
persisted a three-task dependency chain, exchanged the three required handoff
messages, changed only `src/order.py`, and completed without operator edits.

Runtime settings selected by the lead:

- 3 parallel workers
- 10-minute timeout
- 50-turn budget
- 2 automatic retries
- 15-minute task leases

Combined evaluator result:

```text
Collaboration evidence: PASSED
Ran 6 tests
OK
Business acceptance: PASSED
```

Persisted trace summary:

- 201 native events
- 49 tool calls: 47 completed and 2 failed
- 3 teammate messages
- 40,165 teammate tokens across 19 turns
- 3 minutes 16 seconds elapsed inside `TeamRun`
- every agent and task completed on its first attempt

The two failed tool calls were the unavailable `python` alias and the expected
failing baseline test command. The lead recovered by using `python3`, and the
coder and reviewer independently obtained six passing acceptance checks.
