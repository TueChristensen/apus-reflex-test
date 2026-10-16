"""Follow-up to random7.py: does a model still over-pick 7 when it can see its earlier picks?

Two modes, both in a single chat (unlike random7.py, where every sample is a fresh chat):
  turns: ask the same question N times in one growing conversation; the model sees
         all its previous answers.
  list:  ask once for N random numbers in a single reply.

Run:  TRANSLATOR_MODEL=<hf repo> uv run python experiments/random7_chat.py --mode turns
Writes: experiments/results/random7-chat-<mode>-<timestamp>/
"""

import argparse
import collections
import csv
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

MODEL = os.environ.get("TRANSLATOR_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TURN_PROMPT = "Pick a random number from 1 to 10. Reply with only the number."
LIST_PROMPT = (
    "Pick {n} random numbers from 1 to 10. "
    "Reply with only the numbers, separated by commas."
)


def parse_first(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def run_turns(model, tokenizer, n, sampler, out_dir):
    """One conversation, N user turns; the KV cache is reused across turns."""
    cache = make_prompt_cache(model)
    cached = []  # tokens currently held in the cache
    messages = []
    rows = []
    t0 = time.time()
    for i in range(n):
        messages.append({"role": "user", "content": TURN_PROMPT})
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        common = 0
        while common < min(len(full), len(cached)) and full[common] == cached[common]:
            common += 1
        if len(cached) > common:
            trim_prompt_cache(cache, len(cached) - common)
        new = full[common:]
        if not new:  # always feed at least one token
            trim_prompt_cache(cache, 1)
            common -= 1
            new = full[common:]
        out_tokens, text = [], ""
        for resp in stream_generate(
            model, tokenizer, prompt=new, max_tokens=8, sampler=sampler, prompt_cache=cache
        ):
            out_tokens.append(resp.token)
            text += resp.text
        cached = (full[:common] + new + out_tokens)[: cache[0].offset]
        reply = text.strip()
        messages.append({"role": "assistant", "content": reply})
        rows.append({"i": i, "raw": reply, "parsed": parse_first(reply)})
        if (i + 1) % 100 == 0:
            print(f"  turn {i + 1}/{n} ({time.time() - t0:.0f}s, context {cache[0].offset} tokens)")
    (out_dir / "conversation.json").write_text(json.dumps(messages, indent=1, ensure_ascii=False))
    return rows


def run_list(model, tokenizer, n, sampler, out_dir):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": LIST_PROMPT.format(n=n)}], tokenize=False, add_generation_prompt=True
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=4 * n + 50, sampler=sampler)
    (out_dir / "reply.txt").write_text(raw)
    return [{"i": i, "raw": s, "parsed": int(s)} for i, s in enumerate(re.findall(r"\d+", raw))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["turns", "list"], default="turns")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "results" / time.strftime(f"random7-chat-{args.mode}-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    print(f"{MODEL}: mode {args.mode}, n {args.n}, temperature {args.temp}, seed {args.seed}")
    model, tokenizer = load(MODEL)
    sampler = make_sampler(temp=args.temp)
    mx.random.seed(args.seed)
    run = run_turns if args.mode == "turns" else run_list
    rows = run(model, tokenizer, args.n, sampler, out_dir)

    with open(out_dir / "samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "raw", "parsed"])
        writer.writeheader()
        writer.writerows(rows)

    counts = collections.Counter(r["parsed"] for r in rows)
    share_7 = counts[7] / len(rows) if rows else 0.0
    summary = {
        "model": MODEL,
        "mode": args.mode,
        "prompt": TURN_PROMPT if args.mode == "turns" else LIST_PROMPT.format(n=args.n),
        "n_requested": args.n,
        "n_parsed": len(rows),
        "temperature": args.temp,
        "seed": args.seed,
        "counts": {str(k): v for k, v in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        "share_7": share_7,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ncounts: {summary['counts']}  ({len(rows)} parsed, 7 = {share_7:.1%})")
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
