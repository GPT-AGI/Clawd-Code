# Order Discount Teammate Evaluation

This fixture evaluates whether a lead agent can coordinate researcher, coder,
and reviewer teammates to repair a small order-pricing module.

The implementation intentionally starts with pricing defects. A clean fixture
must fail some acceptance checks before the teammate run and pass all checks
after a successful run.

## Baseline check

From this directory:

```bash
../../.venv/bin/python -m unittest checks.order_acceptance -v
```

## Real-model run

Run this evaluation in a clean copy because the agent intentionally edits
`src/order.py` and persists its team state under `.clawd/`. The normal Clawd
configuration is used. With an Anthropic-compatible z.ai provider configured,
run the task directly from this directory:

```bash
../../.venv/bin/clawd run \
  --provider anthropic \
  --model glm-5.2 \
  --prompt-file TASK.md \
  --max-turns 100
```

No API key is stored in this fixture. Clawd reads it from the user's normal
`~/.clawd/config.json` configuration or from `ANTHROPIC_AUTH_TOKEN` and
`ANTHROPIC_BASE_URL` environment variables.

After the run, execute the baseline command again and inspect the team state in
`.clawd/teams/`. The expected collaboration evidence is described in
`acceptance.json`.

Run the combined automated evaluator from this directory:

```bash
../../.venv/bin/python evaluate.py
```

It verifies the business checks plus teammate records, independent sessions,
task ownership and dependencies, message handoffs, event history, and final
team status.

Inspect the complete run locally after or during execution:

```bash
../../.venv/bin/clawd trace . --open
```

The viewer shows the event timeline, agent lanes, task graph, tool inputs and
results, message handoffs, failures, timing, and token usage. It follows a run
through server-sent events when opened before execution starts.
