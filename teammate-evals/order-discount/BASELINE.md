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

The remaining evaluation blocker is now genuine teammate spawning, messaging,
and task scheduling rather than local engineering tool discovery.
