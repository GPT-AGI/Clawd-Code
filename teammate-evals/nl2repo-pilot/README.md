# NL2Repo Pilot Benchmark

This adapter evaluates Clawd against a pinned external checkout of
[NL2Repo-Bench](https://github.com/multimodal-art-projection/NL2RepoBench).
The upstream task documents and hidden tests are not copied into this repository.
The default pinned commit is `781a1da1ee41fb8edb0bed22f586d69111610edf`.

The pilot compares solo execution with adaptive lead-controlled collaboration.
`forced-team` remains a runtime diagnostic, but now enables strict quality gates:
at least two independently runnable teammate tasks, non-overlapping file ownership,
explicit interface contracts, peer acknowledgement on dependency edges, and a fresh
environment install/import/integration verification before the Team can complete.
Role names, permissions, workspaces, and the concrete task graph remain task-specific.

Reward is skipped when the rollout, specification-integrity check, or Team protocol is
incomplete. This prevents an unfinished strict Team from entering the AGS/Docker reward
pool merely because the lead model stopped. Results and the dashboard expose a stable
`failure_class` such as `dependency_environment`, `collection_error`,
`cross_module_contract`, `team_validation`, or `functional_test_failure`.

## Docker environment

The scorer requires a running Docker-compatible daemon with `linux/amd64`
emulation. On Apple Silicon with Colima:

```bash
brew install colima docker
colima start --cpu 6 --memory 12 --disk 100 --vm-type vz --vz-rosetta
docker run --rm --platform linux/amd64 alpine:3.20 uname -m
```

The first benchmark invocation clones the pinned upstream repository into
`~/.cache/clawd-code/nl2repo-bench/`. Set `NL2REPO_BENCH_ROOT` or pass
`--upstream-root` to use an existing checkout.

## Tencent AGS environment

AGS can replace Docker for agent execution, scoring, or both. Install the
customized SWE-ReX checkout next to this repository and the AGS transport
dependencies:

```bash
cd /Users/dexter/Desktop/workspace/Multi-agent/Clawd-Code
.venv/bin/python -m pip install -e ../sandbox/SWE-ReX
.venv/bin/python -m pip install -e '.[ags]'
```

Configure `AGS_SECRET_ID`, `AGS_SECRET_KEY`, and preferably an existing
`AGS_TOOL_ID`. The launcher automatically discovers `../sandbox/ags/.env`, or
you can select it explicitly with `--ags-env-file`. Credentials are never
copied into the sandbox or written to benchmark results.

Validate imports, credentials, task metadata, and image selection without
starting a cloud instance. The recommended first configuration runs the agent
in AGS and keeps the existing network-isolated Docker scorer:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --task jsonlines \
  --execution-backend ags \
  --score-backend docker \
  --ags-env-file ../sandbox/ags/.env \
  --validate
```

Run the agent in AGS:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --provider anthropic \
  --model glm-5.2 \
  --task jsonlines \
  --mode solo \
  --execution-backend ags \
  --score-backend docker \
  --ags-env-file ../sandbox/ags/.env \
  --ags-timeout 3h \
  --ags-cpu 2 \
  --ags-memory 4Gi
```

The agent instance is reset before `start.md` is uploaded, so it cannot see the
image's hidden tests. After the agent exits, its repository is downloaded and
the instance is stopped. The sandbox TTL is a cleanup backstop if the local
process is killed.

To move scoring to AGS too, create a second reusable SandboxTool configured
with `NetworkConfiguration.NetworkMode=SANDBOX` and save its ID as
`AGS_SCORE_TOOL_ID`. A normal `PUBLIC` tool is deliberately rejected: Tencent
AGS instances inherit their Tool's network mode, while the Docker scorer uses
`--network none`.

```bash
cd ../sandbox/ags
uv run python main.py \
  --task jsonlines \
  --network-mode SANDBOX \
  --create-tool \
  --cmd 'true'
# Copy the printed tool ID into .env as AGS_SCORE_TOOL_ID=...

cd ../../Clawd-Code
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --task jsonlines \
  --mode solo \
  --execution-backend ags \
  --score-backend ags \
  --ags-env-file ../sandbox/ags/.env
```

With AGS scoring enabled, each case uses a second fresh instance. It keeps the
official task image intact, receives only a stripped candidate overlay, and
runs an outbound-network probe before the hidden suite. The run fails closed
if that probe detects Internet access.

The AGS image convention is
`swebenchdocker.tencentcloudcr.com/swebench/nl2repo:<task>-1.0`; Docker uses the
upstream GHCR image. Verify score parity for a representative task before a
large run whenever either registry is updated.

Cloud teammates share `/workspace`. Worktree teammates are intentionally
rejected for AGS runs because a single remote workspace cannot provide the
local worktree isolation and integration semantics yet.

## Usage

List upstream tasks and mark the five pilot selections:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py --list
```

Validate task metadata and official GHCR images without calling a model:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --task jsonlines --task tinydb --validate
```

Run a small comparison:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --provider anthropic \
  --model glm-5.2 \
  --task jsonlines \
  --mode both
```

The primary five-task pilot is `jsonlines`, `tinydb`, `aiofiles`,
`flask-restful`, and `fastapi-users`. Runs default to 300 lead turns, 80 turns
per worker, and a two-hour agent timeout because these are long-horizon tasks.
Each model turn may use up to 16,384 output tokens by default so large file-write
tool arguments are not truncated; override this with `--max-output-tokens`.

Each run starts from a Git repository containing only `start.md`. The scorer
removes generated tests and packaging files exactly as the upstream harness
does, overlays the implementation onto the official task image, and installs
declared build dependencies while building the score image. It then disables
network access before running the hidden pytest suite. Results also record token
use, agent roles and permissions, task completion, direct peer messages, and
lead stop/resume/reassign/retry interventions.

Structured model streaming is enabled by default. While the lead runs,
`progress.jsonl` records model, text-stream, and tool events incrementally, so
provider stalls can be distinguished from active repository work. Use
`--no-stream` only when diagnosing a provider without compatible streaming.
Qwen streaming requests explicitly enable the terminal usage chunk. New result
files separate lead and worker token counts and set `usage.complete`; aggregate
token comparisons should include only complete measurements. Historical Qwen
streaming runs cannot be backfilled exactly because the gateway did not return
Lead usage unless it was requested during generation.

## 32-task rollout pool

The `qwen32` task set is the exact deterministic subset used by the concurrency
probe: 32 tasks sampled with seed `20260715` after limiting `start.md` to 64 KiB.
It defaults to adaptive mode, 300 lead turns, and eight concurrent rollouts.
Inspect the resolved plan without launching agents or sandboxes:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --task-set qwen32 \
  --plan
```

Run the Qwen evaluation in AGS with an independent four-worker reward pool:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/benchmark.py \
  --task-set qwen32 \
  --mode adaptive \
  --provider qwen \
  --model ms-mnhdj86z \
  --max-turns 300 \
  --rollout-concurrency 8 \
  --reward-concurrency 4 \
  --execution-backend ags \
  --score-backend ags \
  --ags-env-file ../sandbox/ags/.env
```

The rollout executor always keeps up to eight agent cases active. As soon as
one rollout finishes and its workspace has been downloaded, that slot starts
the next queued case. Hidden-test reward evaluation is submitted to a separate
executor and never consumes a rollout slot. With both phases on AGS, the example
can therefore have up to 12 cloud sandboxes active at once: eight rollout
instances plus four isolated reward instances. Set `--score-backend docker` to
keep reward evaluation local while retaining the same scheduling semantics.

`scheduler.jsonl` records rollout and reward start/completion events. Individual
case results are persisted immediately below `<task>/<mode>/result.json`, so
completed cases remain recoverable if the aggregate run is interrupted.

### Live evaluation dashboard

Every new run writes `run-metadata.json` before the first rollout starts. The
read-only dashboard combines that manifest with `scheduler.jsonl`, incremental
`progress.jsonl` events, and per-case results, so queued tasks are visible before
their workspace exists and corrected reward results take precedence over older
scheduler events.

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/dashboard.py \
  --run teammate-evals/nl2repo-pilot/runs/<run-id> \
  --port 8765
```

Open `http://127.0.0.1:8765/`. The console refreshes every two seconds and offers
task-table, rollout/reward pipeline, and scheduler-timeline views. Selecting a
task opens its recent structured events, agent response, hidden-test output, and
Docker logs. The server only reads benchmark artifacts and has no run-control
or sandbox-delete endpoints.

To compare modes across compatible batches, add a comparison link to the
selected run's `run-metadata.json`. Results in the selected run take precedence;
the sibling baseline only fills missing task/mode pairs:

```json
{
  "comparison": {
    "modes": ["adaptive", "forced-team"],
    "baseline_runs": {"adaptive": "20260715-qwen32-pool8-v2"}
  }
}
```

The COMPARE view reports paired quality, rollout time, calls, token coverage,
and per-task deltas. Cross-run latency is labeled because concurrency and service
load may differ. Historical zero token counts from streaming responses are
treated as missing measurements.

### Continuous evaluation queue

For sustained utilization, run a queue worker instead of launching fixed-size
batches. Its SQLite/WAL state can be updated from another process while eight
rollout workers and four independent reward workers stay alive:

```bash
RUN=teammate-evals/nl2repo-pilot/runs/qwen-continuous

# Long-lived process. It waits instead of exiting when the queue is empty.
.venv/bin/python teammate-evals/nl2repo-pilot/evaluation_queue.py \
  --run "$RUN" serve \
  --provider qwen \
  --model ms-mnhdj86z \
  --rollout-concurrency 8 \
  --reward-concurrency 4 \
  --execution-backend ags \
  --score-backend docker \
  --ags-env-file ../sandbox/ags/.env
```

Add `--start-after <previous-run-dir>` to prepare and fill the next queue while
the current fixed batch is still running. Workers will not claim queued cases
until the predecessor writes `results.json`, preventing the two runs from
competing for the same model capacity during handoff.

Append validated tasks at any time, including while the worker is running:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/evaluation_queue.py \
  --run "$RUN" add --task tinydb --task tablib --mode adaptive

# Add the deterministic 32-task set, or every upstream task.
.venv/bin/python teammate-evals/nl2repo-pilot/evaluation_queue.py \
  --run "$RUN" add --task-set qwen32 --mode adaptive

# Add only the 72 tasks outside qwen32.
.venv/bin/python teammate-evals/nl2repo-pilot/evaluation_queue.py \
  --run "$RUN" add --task-set remaining-qwen32 --mode forced-team

.venv/bin/python teammate-evals/nl2repo-pilot/evaluation_queue.py \
  --run "$RUN" status
```

Cases are deduplicated by task and mode within a continuous run. A completed or
failed case can be intentionally rerun with `retry`; its previous artifacts are
archived below `_attempts/` before the new rollout starts. Higher-priority cases
can be inserted with `add --priority N`.

The state machine is `queued → rollout → reward_pending → rewarding → done`.
After a runner restart, an interrupted rollout returns to the queue, while an
interrupted reward resumes from its persisted rollout artifact. The dashboard
reads `queue.sqlite3` directly, includes newly appended tasks automatically, and
shows a `QUEUE LOW` warning when fewer tasks are waiting than rollout slots.
The continuous runner retries scorer exceptions and explicit
`hidden_tests.error` results three times by default; configure this with
`--reward-attempts` and `--reward-retry-delay`. Ordinary hidden-test failures
are valid rewards and are never retried.

For a fixed batch started by an older scorer process, watch and repair only its
infrastructure failures without repeating any agent rollout:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/reward_repair.py \
  --run teammate-evals/nl2repo-pilot/runs/<run-id> \
  --watch --concurrency 4
```

When the batch finishes, the repair process also rebuilds `results.json` and
`REPORT.md` from the corrected per-case results so stale in-memory reward values
do not survive in the aggregate report.

## Qwen concurrency latency probe

Before spending sandbox time on 32 complete agents, measure model-service
queueing with a fixed subset of real NL2Repo specifications:

```bash
.venv/bin/python teammate-evals/nl2repo-pilot/latency_probe.py --dry-run
.venv/bin/python teammate-evals/nl2repo-pilot/latency_probe.py \
  --subset-size 32 \
  --baseline-concurrency 1 \
  --concurrency 32

# Find the useful operating point with the same 32 prompts at every level.
.venv/bin/python teammate-evals/nl2repo-pilot/latency_probe.py \
  --subset-size 32 \
  --sweep 1,2,4,8
```

By default the probe deterministically samples 32 tasks whose `start.md` is no
larger than 64 KiB, using seed `20260715`. It sends the same prompts serially
and concurrently, disables Qwen thinking, requests only a short acknowledgement,
streams without retries, and records TTFT, total latency, throughput, token use,
and errors. It does not execute agents or score repositories, so its results
isolate the model endpoint rather than AGS, tool calls, or hidden tests.

Artifacts are written below `latency-runs/<timestamp>/`. `results.json` contains
only task names, prompt sizes, metrics, and sanitized errors; prompt contents and
the configured AuthToken are not persisted. Use `--output` to choose a stable
result directory.

The upstream repository does not currently include a root license file. Keep it
as an external pinned dependency unless its maintainers publish redistribution
terms.
