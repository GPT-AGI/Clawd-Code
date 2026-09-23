# OpenMythos Provider

Drive Clawd-Code against a local [OpenMythos](https://github.com/kyegomez/OpenMythos)
Recurrent-Depth Transformer instead of a remote API.

## When to use this

- You are training or evaluating an OpenMythos checkpoint and want an
  interactive REPL around it.
- You want to smoke-test the Clawd-Code plumbing without a network
  round-trip or an API key.
- You are experimenting with depth extrapolation (inference-time
  `n_loops` > training `max_loop_iters`).

## Caveat

OpenMythos has no pretrained weights shipped. Without a checkpoint the
model is randomly initialized and will produce noise. Useful for wiring
checks; not useful for "chat" until you train it.

## Install

```bash
pip install -e .[openmythos]
# or
uv pip install -e .[openmythos]
```

This adds `open-mythos >= 0.4.0` and `torch` as optional extras.

## Configure

```bash
clawd login openmythos
# — or skip login and set environment directly:
export OPENMYTHOS_VARIANT=mythos_3b          # or any mythos_<size>
export OPENMYTHOS_CHECKPOINT=/path/state.pt  # optional; skip for random init
export OPENMYTHOS_N_LOOPS=16                 # inference-time loop depth
export OPENMYTHOS_DEVICE=cuda                # auto-detected if unset
export OPENMYTHOS_TOKENIZER=openai/gpt-oss-20b
```

The checkpoint may be either a plain `state_dict()` or the composite
`{"model": ..., "optimizer": ...}` shape produced by the upstream
training script at `training/3b_fine_web_edu.py` — both are detected.

## Run

```bash
clawd --provider openmythos
```

Or programmatically:

```python
from src.providers import get_provider_class

Provider = get_provider_class("openmythos")
provider = Provider(
    variant="mythos_3b",
    checkpoint="/path/state.pt",
    n_loops=16,
)
for chunk in provider.chat_stream(
    [{"role": "user", "content": "Explain the Parcae stability proof."}],
    max_new_tokens=256,
):
    print(chunk, end="", flush=True)
```

## How streaming works

Each decode step runs one recurrent forward pass with the full
`n_loops` depth. The KV cache is reused across steps so decode cost is
proportional to one token per step, not the growing sequence. Tokens
are emitted one at a time; BPE partial pieces are buffered and flushed
as soon as they resolve into printable characters.

The provider does **not** implement tool calling. Clawd-Code's agent
loop falls back to plain chat when a provider does not support tools.

## Knobs (per call)

```python
provider.chat_stream(messages,
    max_new_tokens=512,     # default 512, clamped to max_seq_len - prompt_len
    temperature=0.8,        # softmax temperature
    top_k=50,               # top-K truncation; 0 = disabled
    n_loops=32,             # inference-time loop depth (may exceed training)
    eos_token_id=None,      # defaults to tokenizer.eos_token_id
)
```

## Known limitations

1. No tool calling (architectural — OpenMythos has no tool-output head).
2. No pretrained chat template; transcript is a plain
   `role: content` string terminated with `assistant:`.
3. No speculative decoding or beam search — single-sample multinomial.
4. CPU inference is viable for tiny configs (mythos_1b or smaller); the
   1B+ variants want a GPU.
5. The `vocab_size` in `MythosConfig` is overwritten at provider init
   to match the tokenizer. If you load a checkpoint whose embedding
   matrix was sized to a different vocab, loading uses `strict=False`
   so the mismatch is surfaced as a warning rather than a crash — but
   the model will be partially initialized.
