# reflex-lab

A small local test bench for [APUS-Reflex-9B-MLX-4bit](https://huggingface.co/apus-ailab/APUS-Reflex-9B-MLX-4bit) on Apple Silicon.

Type a question and get an answer with a probability. Everything runs locally through [mlx-lm](https://github.com/ml-explore/mlx-lm).

## How it works

APUS Reflex is a **decision model**, not a chat model. It doesn't write text. Each request gives it a context (`state`),
an `instructions` sentence and 2–16 candidates. It returns a probability for each candidate, taken from the
final-position logits of the candidate labels A..P. So a second, small LLM first turns your question into that structure:

```
your question
   │
   ▼
Qwen3-4B-Instruct (MLX 4-bit)       structures the question: yes/no ("noul") or multiple choice ("choice")
   │                                with 2–8 candidates. It never answers.
   ▼
APUS Reflex 9B (MLX 4-bit)          scores every candidate
   │
   ▼
answer + probability                the top candidate, filled in by code (no LLM writes the answer)
```

Example: "What is the capital of Australia?" becomes

```json
{
  "primitive": "choice",
  "state": "The question asks for the capital city of Australia.",
  "instructions": "Select the capital of Australia.",
  "criteria": [
    {"id": "sydney", "description": "Sydney."},
    {"id": "canberra", "description": "Canberra."},
    {"id": "melbourne", "description": "Melbourne."}
  ]
}
```

Reflex answers **Canberra**, with a probability for every candidate.

## Requirements

- **macOS on Apple Silicon (M1 or later).** MLX doesn't run on Intel Macs, Linux or Windows, so neither model can load there.
- At least 16 GB of unified memory. Both models together use about 10 GB, so 16 GB works but leaves little room for other apps.
- Python 3.12 or newer. uv installs it automatically if it's missing.
- [uv](https://docs.astral.sh/uv/)
- About 8 GB of disk for the weights, downloaded into `~/.cache/huggingface` on first run.

## Run

```sh
git clone https://github.com/TueChristensen/apus-reflex-test.git
cd apus-reflex-test
uv sync
uv run reflex-lab                                 # web UI at http://127.0.0.1:7860
uv run reflex-lab --ask "Is a tomato a fruit?"    # one question in the terminal, prints JSON
```

### The UI

Type a question and press **Ask** (or Enter). The answer appears with its probability, for example:

> **Rest at home and monitor symptoms.**
> *p = 0.6493 (uncalibrated)*

Below it are three collapsible sections:

1. **Structured question:** the request the small LLM produced. It is editable: fix a candidate or add a missing one, then press **Re-ask**.
2. **Reflex scores:** the probability of every candidate. These add up to 1.
3. **Raw prompt & response:** the exact text Reflex saw, and its raw output.

## Configuration

Either model can be replaced with a Hugging Face repo id or a local MLX folder, for example one downloaded with LM Studio:

| Variable | Default |
|---|---|
| `REFLEX_MODEL` | `apus-ailab/APUS-Reflex-9B-MLX-4bit` |
| `TRANSLATOR_MODEL` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |

```sh
TRANSLATOR_MODEL=~/.lmstudio/models/some-org/some-model-MLX-4bit uv run reflex-lab
```

The models must be in MLX format. GGUF files won't load.

## Caveats

- **Probabilities are not calibrated.** Compare them within one question, not across questions.
- **Reflex can only pick from the candidates it is given.** If the small LLM leaves out the right answer, the result is wrong
  however confident it looks. Check the structured question when an answer seems off.
- The 4-bit build matches the full-precision model on 85% of the model card's 80-question panel (68/80; full precision gets 75/80).

## Project layout

```
src/reflex_lab/
├── __init__.py     CLI entry point (reflex-lab, --ask)
├── app.py          Gradio UI
├── translator.py   question → structured request (Qwen3-4B)
├── reflex.py       candidate probabilities (APUS Reflex), adapted from the model repo's examples/openjev_mlx.py
└── contracts.py    the model's prompt/training contract, copied verbatim from the model repo. Don't edit it.
```

## License

Apache-2.0, see [LICENSE](LICENSE). `contracts.py` and the inference logic in `reflex.py` are adapted from the
[APUS Reflex model repo](https://huggingface.co/apus-ailab/APUS-Reflex-9B-MLX-4bit), which is also Apache-2.0.
