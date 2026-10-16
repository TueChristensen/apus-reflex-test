# reflex-lab

Small test bench for [APUS-Reflex-9B-MLX-4bit](https://huggingface.co/apus-ailab/APUS-Reflex-9B-MLX-4bit) on Apple Silicon.

Reflex is a **decision model**, not a chat model. It takes a request with a `state`, `instructions` and
2–16 `criteria` (candidates). It returns a probability for each candidate, from the logits of the labels A..P.
A small local LLM (Qwen3-4B-Instruct, MLX 4-bit) turns your free-form question into that request.

```
question ──► Qwen3-4B (translator) ──► OpenJev request JSON ──► APUS Reflex 9B ──► candidate probabilities
```

## Run

```sh
uv sync
uv run reflex-lab                          # Gradio UI at http://127.0.0.1:7860
uv run reflex-lab --ask "Is a tomato a fruit?"   # one-shot in the terminal
```

The first run downloads about 8 GB of weights into `~/.cache/huggingface`. Both models together use about 10 GB of unified memory.

Type a question and press **Ask**: you get the answer. The structured question (editable, with **Re-ask**), the per-candidate scores and the raw prompt are in collapsible sections below it.

## Models

Override either model with a Hugging Face repo id or a local MLX folder, for example one in LM Studio:

```sh
REFLEX_MODEL=~/.lmstudio/models/apus-ailab/APUS-Reflex-9B-MLX-4bit \
TRANSLATOR_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit uv run reflex-lab
```

## Layout

- `src/reflex_lab/contracts.py`: the model's training contract, copied verbatim from the model repo (Apache-2.0). Don't edit it.
- `src/reflex_lab/reflex.py`: exact candidate distribution, adapted from the repo's `examples/openjev_mlx.py`.
- `src/reflex_lab/translator.py`: question → request (`choice` for multiple choice, `noul` for yes/no propositions).
- `src/reflex_lab/app.py`: the Gradio UI.

Probabilities are **not calibrated**: compare them within one request, not across requests.
