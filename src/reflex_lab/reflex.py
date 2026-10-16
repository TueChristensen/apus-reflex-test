"""APUS Reflex decision engine: exact candidate distribution via mlx-lm.

Adapted from the model repo's examples/openjev_mlx.py. The candidate labels A..P
are single tokens; the distribution is the softmax of their final-position logits.
"""

import os

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from reflex_lab.contracts import format_response, label_mapping, render_prompt

# HF repo id or a local MLX folder (e.g. one under ~/.lmstudio/models)
REFLEX_MODEL = os.environ.get("REFLEX_MODEL", "apus-ailab/APUS-Reflex-9B-MLX-4bit")
MAX_TOKENS = 8192
PREFILL_STEP = 512  # chunked prefill keeps long prompts within unified memory


class Reflex:
    def __init__(self, model_path=REFLEX_MODEL):
        self.model, self.tokenizer = load(model_path)

    def prompt(self, request):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": render_prompt(request)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def compile(self, request):
        mapping = label_mapping(request)
        prompt = self.prompt(request)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > MAX_TOKENS:
            raise ValueError("input exceeds 8192 tokens; no truncation is performed")
        candidates = []
        for label in mapping:
            token = self.tokenizer.encode(label, add_special_tokens=False)
            if (
                len(token) != 1
                or self.tokenizer.encode(prompt + label, add_special_tokens=False)
                != ids + token
            ):
                raise ValueError(
                    "candidate label is not a single token at the answer boundary"
                )
            candidates.append(token[0])
        return ids, candidates

    def decide(self, request):
        ids, candidates = self.compile(request)
        cache = make_prompt_cache(self.model)
        for start in range(0, len(ids) - 1, PREFILL_STEP):
            self.model(
                mx.array([ids[start : min(start + PREFILL_STEP, len(ids) - 1)]]),
                cache=cache,
            )
            mx.eval([c.state for c in cache])
        logits = self.model(mx.array([ids[-1:]]), cache=cache)[0, -1].astype(
            mx.float32
        )[mx.array(candidates)]
        response = format_response(request, mx.softmax(logits).tolist())
        response.update(prompt_tokens=len(ids), calibrated=False)
        return response
