# NL2Repo Pilot Benchmark

This adapter evaluates Clawd against a pinned external checkout of
[NL2Repo-Bench](https://github.com/multimodal-art-projection/NL2RepoBench).
The upstream task documents and hidden tests are not copied into this repository.
The default pinned commit is `781a1da1ee41fb8edb0bed22f586d69111610edf`.

The pilot compares solo execution with adaptive lead-controlled collaboration.
`forced-team` remains a runtime diagnostic: it requires delegation but does not
prescribe agent count, roles, models, permissions, workspaces, task graph, or
communication topology.

## Environment

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
does, overlays the implementation onto the official task image, disables
network access, and runs the hidden pytest suite. Results also record token use,
agent roles and permissions, task completion, direct peer messages, and lead
stop/resume/reassign/retry interventions.

Structured model streaming is enabled by default. While the lead runs,
`progress.jsonl` records model, text-stream, and tool events incrementally, so
provider stalls can be distinguished from active repository work. Use
`--no-stream` only when diagnosing a provider without compatible streaming.

The upstream repository does not currently include a root license file. Keep it
as an external pinned dependency unless its maintainers publish redistribution
terms.
