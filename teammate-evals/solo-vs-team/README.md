# Solo vs Team Benchmark

This benchmark compares one-agent execution with a persistent teammate workflow
on the same five repository repair tasks:

The first real `glm-5.2` run and its conclusions are recorded in
[`BASELINE_GLM52.md`](BASELINE_GLM52.md).

1. `invoice-allocation`: money, ordering, partial allocation, and aging rules.
2. `webhook-idempotency`: concurrency, retries, ordering, and tenant isolation.
3. `config-migration`: multi-version migration, validation, interpolation, and redaction.
4. `permission-policy`: inheritance, wildcard matching, tenant scoping, and deny precedence.
5. `csv-reconciliation`: parsing, deterministic matching, ambiguity, and malformed data.

Each run gets a clean workspace. `TASK.md`, `requirements.md`, and acceptance
tests are hashed before and after execution. Solo and team receive the same
business task; only the execution protocol differs. Quality is the percentage
of deterministic acceptance tests passed. The report also records elapsed time,
tokens, model/tool calls, and team collaboration evidence.

Validate the intentionally broken fixtures without using a model:

```bash
.venv/bin/python teammate-evals/solo-vs-team/benchmark.py --validate-fixtures
```

Run all ten comparisons with the configured Anthropic-compatible GLM endpoint:

```bash
.venv/bin/python teammate-evals/solo-vs-team/benchmark.py \
  --provider anthropic \
  --model glm-5.2
```

Run one scenario or one mode while iterating:

```bash
.venv/bin/python teammate-evals/solo-vs-team/benchmark.py \
  --scenario webhook-idempotency \
  --mode team
```

Artifacts are written under `runs/<UTC timestamp>/`: each isolated workspace,
the exact generated prompt, stdout/stderr, per-run JSON, aggregate `results.json`,
and `REPORT.md`. Credentials are inherited from normal Clawd configuration or
environment variables and are never written by the benchmark.

Recompute acceptance, integrity, protocol, and usage metrics from an existing
run without making another model request:

```bash
.venv/bin/python teammate-evals/solo-vs-team/benchmark.py \
  --rescore-output teammate-evals/solo-vs-team/runs/<run-id>
```
