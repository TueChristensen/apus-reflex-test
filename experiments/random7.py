"""One-off test: do LLMs over-pick 7, and what does APUS Reflex believe about it?

1. Ask the translator LLM (Qwen3-4B) to pick a random number 1..10, N times at
   temperature 1.0 (the model's unmodified distribution), and record every reply.
2. Ask APUS Reflex fixed yes/no propositions about it, including a sweep of
   "picks 7 more than x% of the time" for x = 0..50.

Run:  uv run python experiments/random7.py
Writes: experiments/results/random7-<timestamp>/
"""

import argparse
import collections
import csv
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from reflex_lab.contracts import BINARY_CRITERIA, render_prompt
from reflex_lab.reflex import Reflex
from reflex_lab.translator import Translator

LLM_PROMPT = "Pick a random number from 1 to 10. Reply with only the number."
LLM_STATE = "A language model is asked to pick a random number from 1 to 10."


def noul(rid, state, proposition):
    return {
        "id": rid,
        "group_id": "random7",
        "primitive": "noul",
        "state": state,
        "instructions": f"Judge whether the proposition is true: {proposition}",
        "criteria": BINARY_CRITERIA,
    }


def reflex_requests(xs):
    requests = [
        noul(
            "fair_rng_7",
            "A fair random number generator picks an integer from 1 to 10.",
            "7 is more likely than any other number.",
        ),
        noul(
            "llm_7_gt_10",
            LLM_STATE,
            "When a language model is asked to pick a random number from 1 to 10, "
            "it picks 7 more than 10% of the time.",
        ),
    ]
    for x in xs:
        requests.append(
            noul(
                f"llm_7_gt_{x:02d}",
                LLM_STATE,
                "When a language model is asked to pick a random number from 1 to 10, "
                f"it picks 7 more than {x}% of the time.",
            )
        )
    return requests


def sample_llm(n, temp, seed, out_dir):
    translator = Translator()
    model, tokenizer = translator.model, translator.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": LLM_PROMPT}], tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temp)
    mx.random.seed(seed)
    rows = []
    t0 = time.time()
    for i in range(n):
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=8, sampler=sampler)
        m = re.search(r"\d+", raw)
        parsed = int(m.group()) if m else None
        rows.append({"i": i, "raw": raw, "parsed": parsed})
        if (i + 1) % 100 == 0:
            print(f"  LLM sample {i + 1}/{n} ({time.time() - t0:.0f}s)")
    with open(out_dir / "llm_samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "raw", "parsed"])
        writer.writeheader()
        writer.writerows(rows)
    del translator, model
    mx.clear_cache()
    return rows


def ask_reflex(requests, out_dir):
    reflex = Reflex()
    results = []
    with open(out_dir / "reflex.jsonl", "w") as f:
        for request in requests:
            response = reflex.decide(request)
            row = {"request": request, "prompt": render_prompt(request), "response": response}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
            print(f"  {request['id']}: p(yes) = {response['yes_probability']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "results" / time.strftime("random7-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    print(f"LLM: {args.n} samples at temperature {args.temp}, seed {args.seed}")
    rows = sample_llm(args.n, args.temp, args.seed, out_dir)
    counts = collections.Counter(r["parsed"] for r in rows)
    share_7 = counts[7] / args.n

    print("Reflex:")
    results = ask_reflex(reflex_requests(range(0, 51)), out_dir)

    summary = {
        "llm": {
            "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            "prompt": LLM_PROMPT,
            "n": args.n,
            "temperature": args.temp,
            "seed": args.seed,
            "counts": {str(k): v for k, v in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
            "share_7": share_7,
        },
        "reflex": {r["request"]["id"]: r["response"]["yes_probability"] for r in results},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nLLM counts: {summary['llm']['counts']}  (7 = {share_7:.1%})")
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
