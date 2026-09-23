"""OpenMythos local-model provider for Clawd-Code.

Wraps a local PyTorch `open_mythos.OpenMythos` model behind the Clawd
`BaseProvider` interface so you can drive Clawd-Code against a
locally-trained (or randomly-initialized) Recurrent-Depth Transformer.

Design constraints:
- Heavy imports (torch, open_mythos) are deferred to __init__ so Clawd
  users without PyTorch are unaffected at package-import time.
- Messages are flattened into a simple `role: content` transcript with
  a trailing `assistant:` prompt. There is no built-in chat template —
  OpenMythos does not ship pretrained chat weights, so any template
  would be fictional.
- Streaming is token-by-token via incremental generate() calls that
  reuse a KV cache across tokens (same model state; one token per step).
- Tool calling is not supported in this provider. Clawd's agent loop
  falls back to text-only mode when tool schemas aren't honored.

Environment:
    OPENMYTHOS_VARIANT      One of mythos_1b, mythos_3b, mythos_10b, ...
                            Falls back to `mythos_3b`.
    OPENMYTHOS_CHECKPOINT   Optional path to a state_dict .pt file. If
                            unset the model runs with random-init
                            weights (useful only for wiring checks —
                            output is noise).
    OPENMYTHOS_DEVICE       cuda / mps / cpu. Auto-detected if unset.
    OPENMYTHOS_N_LOOPS      Recurrent depth at inference. Defaults to
                            cfg.max_loop_iters.
    OPENMYTHOS_TOKENIZER    HuggingFace model id for MythosTokenizer.
                            Defaults to openai/gpt-oss-20b (matches the
                            training script).
"""

from __future__ import annotations

import os
from typing import Any, Generator, Optional

from .base import BaseProvider, ChatResponse, MessageInput


def _autoselect_device() -> str:
    """Pick the best available device without importing torch at module load."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _flatten_messages(messages: list[MessageInput]) -> str:
    """Render a transcript as a single prompt string.

    OpenMythos has no pretrained chat template. We use a minimal
    `<role>: <content>` form and append a trailing `assistant:` so the
    model continues the turn.
    """
    lines: list[str] = []
    for m in messages:
        d = m if isinstance(m, dict) else m.to_dict()
        role = d.get("role", "user")
        content = d.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


class OpenMythosProvider(BaseProvider):
    """Run inference against a local OpenMythos model.

    This provider is useful for three workflows:
    1. Integration dev — verify the provider + REPL plumbing works
       without needing an API key or network.
    2. Local evaluation of a model you trained via `training/` in the
       OpenMythos repo.
    3. Research — drive the recurrent-depth model from an interactive
       shell instead of writing one-off scripts.

    NOTE: If you do not pass `checkpoint` (or set OPENMYTHOS_CHECKPOINT),
    the model is random-init and will produce noise. This is expected
    and documented.
    """

    def __init__(
        self,
        api_key: str = "",  # unused; local model
        base_url: Optional[str] = None,  # unused
        model: Optional[str] = None,
        *,
        variant: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        n_loops: Optional[int] = None,
        tokenizer_id: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model)

        import torch

        try:
            from open_mythos import MythosTokenizer, OpenMythos  # noqa: F401
            from open_mythos import variants as om_variants
        except ImportError as exc:
            raise ImportError(
                "open_mythos is not installed. Install it with "
                "`pip install open-mythos` (>= 0.4.0) or `pip install -e path/to/OpenMythos`."
            ) from exc

        variant_name = (
            variant
            or os.environ.get("OPENMYTHOS_VARIANT")
            or (model if model and model.startswith("mythos_") else None)
            or "mythos_3b"
        )
        variant_fn = getattr(om_variants, variant_name, None)
        if variant_fn is None:
            available = [n for n in dir(om_variants) if n.startswith("mythos_")]
            raise ValueError(
                f"Unknown OpenMythos variant {variant_name!r}; "
                f"available: {sorted(available)}"
            )
        cfg = variant_fn()

        device_str = device or os.environ.get("OPENMYTHOS_DEVICE") or _autoselect_device()
        self.device = torch.device(device_str)
        self._n_loops = (
            n_loops
            if n_loops is not None
            else int(os.environ.get("OPENMYTHOS_N_LOOPS", cfg.max_loop_iters))
        )

        ckpt_path = checkpoint or os.environ.get("OPENMYTHOS_CHECKPOINT")
        tokenizer_id = tokenizer_id or os.environ.get("OPENMYTHOS_TOKENIZER") or "openai/gpt-oss-20b"

        # Build the tokenizer first so cfg.vocab_size aligns before the
        # model's embedding is allocated. This avoids the adv-14 foot-gun
        # (32k default vocab vs 200k tokenizer).
        self.tokenizer = MythosTokenizer(model_id=tokenizer_id)
        cfg.vocab_size = self.tokenizer.vocab_size

        self.cfg = cfg
        self.variant_name = variant_name
        self.mythos_model = OpenMythos(cfg)

        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # training/3b_fine_web_edu.py saves {'model': ..., 'optimizer': ..., 'step': ...}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.mythos_model.load_state_dict(state, strict=False)

        self.mythos_model.to(self.device)
        self.mythos_model.eval()

        self._torch = torch

    # ------------------------------------------------------------------
    # BaseProvider surface
    # ------------------------------------------------------------------

    def get_available_models(self) -> list[str]:
        """Enumerate known local variants. The active one is self.variant_name."""
        return [
            "mythos_1b",
            "mythos_3b",
            "mythos_10b",
            "mythos_50b",
            "mythos_100b",
            "mythos_500b",
            "mythos_1t",
        ]

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Non-streaming: collect the full response and return it."""
        chunks: list[str] = []
        for chunk in self.chat_stream(messages, tools=tools, **kwargs):
            chunks.append(chunk)
        text = "".join(chunks)
        return ChatResponse(
            content=text,
            model=self.variant_name,
            usage={
                "prompt_tokens": 0,  # not separately tracked here
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            finish_reason="stop",
        )

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time by invoking generate() with
        max_new_tokens=1 and reusing the KV cache across calls.

        Kwargs supported: max_new_tokens (default 512), temperature
        (default 0.8), top_k (default 50), n_loops (overrides __init__
        default), eos_token_id (defaults to tokenizer.eos_token_id).
        """
        torch = self._torch

        prompt = _flatten_messages(self._prepare_messages(messages))
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            return
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        max_new_tokens = int(kwargs.get("max_new_tokens", 512))
        temperature = float(kwargs.get("temperature", 0.8))
        top_k = int(kwargs.get("top_k", 50))
        n_loops = int(kwargs.get("n_loops", self._n_loops))
        eos_token_id = kwargs.get("eos_token_id", self.tokenizer.eos_token_id)

        max_seq_len = self.cfg.max_seq_len
        budget = max_seq_len - input_ids.shape[1]
        max_new_tokens = min(max_new_tokens, max(0, budget))

        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            last_decoded = ""
            for step in range(max_new_tokens):
                if step == 0:
                    cur_ids = input_ids
                    start_pos = 0
                else:
                    cur_ids = input_ids[:, -1:]
                    start_pos = prompt_len + step - 1

                logits = self.mythos_model(
                    cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
                )
                next_logits = logits[:, -1, :] / max(temperature, 1e-6)
                if top_k > 0:
                    effective_k = min(top_k, next_logits.shape[-1])
                    v, _ = next_logits.topk(effective_k)
                    next_logits[next_logits < v[:, -1:]] = float("-inf")

                probs = next_logits.softmax(dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                tok_id = int(next_tok.item())

                if eos_token_id is not None and tok_id == eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_tok], dim=1)

                # Incremental decode: decode the full generated tail and
                # emit only the newly-added suffix. Some BPE tokenizers
                # (gpt-oss-20b included) produce partial pieces that
                # resolve into characters only after the next token, so
                # decoding the full tail each step is the safe path.
                full_text = self.tokenizer.decode(input_ids[0, prompt_len:].tolist())
                if len(full_text) > len(last_decoded):
                    new_piece = full_text[len(last_decoded):]
                    last_decoded = full_text
                    yield new_piece
