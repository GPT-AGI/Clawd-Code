# Peer-native collaboration benchmark

This directory evaluates whether complete coding agents can coordinate as equal
peers without a privileged LLM lead, a pre-created task graph, assigned owners,
or fixed professions. It is an inference-only benchmark: no model training,
fine-tuning, reinforcement learning, learned routing, or learned policy is used.

The peer runtime is separate from the existing lead-controlled teammate runtime.
A non-intelligent supervisor only creates sessions and workspaces, enforces
budgets and timeouts, records events, wakes idle sessions, validates Git
revisions, and stops the run after the first valid submission.

## Conditions

- `solo`: one agent; no peer message tools.
- `independent` / `none`: multiple agents with the same mission and no message tools.
- `artifact-only`: multiple agents with no message tools; coordination can occur
  only through repository state, commits, or other artifacts visible under the
  selected workspace mode.
- `star`: every peer has the same code capabilities and budget, but message
  transport permits only edges that touch the designated coordinator peer.
- `p2p`: any peer can directly message or broadcast to any other peer.

Communication policy is enforced in the tool registry and transport ACL. Prompt
text is not used as an access-control mechanism. Workspace visibility is an
orthogonal experimental variable: `shared` deliberately exposes concurrent file
effects, while `worktree` starts each peer at the same Git revision in an
isolated detached worktree. Consequently, `independent + shared` still exposes
incidental file effects even though it has no messaging channel; researchers
should normally use `independent + worktree` for a strict independent baseline.

## Scripted smoke

The deterministic coupled fixture requires one peer to publish an item
normalization interface and another peer to consume it before implementing the
client. It exercises persistent sessions, direct delivery, idle wakeup, shared
workspace edits, Git submission, acceptance testing, and schema validation
without an API key:

```bash
.venv/bin/python teammate-evals/peer-collaboration/scripted_smoke.py \
  --output-dir /tmp/clawd-peer-smoke
```

The command must exit zero and the saved `result.json` must show a consumed
message, an accepted submission, no orphan threads, and acceptance exit code 0.

## Real-model pilot

Real calls are explicit opt-in and are never run by the ordinary test suite. For
the configured GLM-5.2 Anthropic-compatible endpoint:

```bash
.venv/bin/python teammate-evals/peer-collaboration/runner.py \
  --repo /path/to/clean/git/repository \
  --prompt-file TASK.md \
  --peers 2 \
  --communication p2p \
  --workspace-mode worktree \
  --provider anthropic \
  --model glm-5.2 \
  --timeout-seconds 600 \
  --max-turns 20 \
  --token-budget 100000 \
  --output-dir /tmp/glm52-peer-pilot \
  --acceptance-command 'python -m pytest -q'
```

The equivalent product CLI is:

```bash
.venv/bin/clawd peer run \
  --repo /path/to/repository \
  --prompt-file TASK.md \
  --peers 2 \
  --communication p2p \
  --workspace-mode worktree \
  --provider anthropic \
  --model glm-5.2 \
  --output-dir /tmp/peer-run
```

## Persistent artifacts

Each run directory contains:

- `manifest.json`: mission, repository revision, condition, peer count,
  provider/model, budgets, workspace mode, backend, participant IDs, and exact
  tool surfaces;
- `run.json`: mutable lifecycle, aggregate usage, stop reason, and accepted
  submission;
- `participants/*.json`: stable IDs, session IDs, status timestamps, workspace,
  and per-peer usage;
- `sessions/*.json`: persistent conversation state and model-boundary index;
- `messages/*.json`: payload plus created, delivered, and consumed state;
- `broadcasts/*.json`: broadcast ID and the exact per-recipient deliveries;
- `submissions/*.json`: accepted, rejected, and already-submitted attempts;
- `events.jsonl`: auditable events with UTC wall-clock and monotonic timestamps;
- `result.json`: terminal summary, commit attribution, acceptance stdout/stderr
  and exit code, usage, wall time, and cleanup outcome.

Schemas are in `schemas/`; `schema_validation.py` validates the saved manifest
and result without adding a runtime dependency.

The trace supports offline reconstruction of message edges, delivery and
consumption latency, response latency, volume, policy rejections, aggregate and
per-peer tokens/calls, workspace heads, commits, submit races, wall time, and
acceptance quality. Repeated work, stale work, conflicts, and rework remain
analysis-layer proxies derived from tool/file/commit events; the runtime does
not declare causal or scaling conclusions.

## Limitations

- The first backend uses threads and cooperative stop checks. It is not process
  isolation. A provider call that ignores its own network timeout cannot be
  safely killed by Python; the guarded registry still rejects later tool calls.
- Claude Code CLI and Codex CLI process adapters are not implemented. The small
  session backend protocol is designed to accept them later.
- Local Git repositories support both workspace modes. Remote sandbox worktree
  isolation requires one sandbox per peer or an explicit synchronization layer.
- Token budgets stop new work after reported usage crosses the limit; already
  in-flight concurrent calls can cause bounded overshoot.
- Dynamic peer recruitment and resizing are outside this version.
- One run is one observation. Statistical aggregation and causal claims belong
  in a separate analysis layer.
