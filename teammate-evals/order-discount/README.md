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

The normal Clawd configuration is used. Set `anthropic.default_model` to
`glm-5.2`, start Clawd from this directory, and paste the contents of
`TASK.md` into the REPL:

```bash
../../.venv/bin/clawd --stream
```

No API key is stored in this fixture. Clawd reads it from the user's normal
`~/.clawd/config.json` configuration.

After the run, execute the baseline command again and inspect the team state in
`.clawd/teams/`. The expected collaboration evidence is described in
`acceptance.json`.
