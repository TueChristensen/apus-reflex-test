"""Small local LLM that turns a free-form question into an OpenJev request."""

import json
import os
import re
import uuid

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from reflex_lab.contracts import BINARY_CRITERIA, validate_request

# HF repo id or a local MLX folder (e.g. one under ~/.lmstudio/models)
TRANSLATOR_MODEL = os.environ.get("TRANSLATOR_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")

SYSTEM_PROMPT = """\
You convert a user's question into a JSON request for a decision model.
The decision model cannot write free text. It only picks one candidate from a list,
so you must supply the context and the candidates.

Output ONLY a JSON object with these keys:
- "primitive": "choice" or "noul".
  Use "noul" ONLY when the user's own question is a yes/no question
  ("Is...", "Does...", "Can..."). The proposition must restate the user's claim,
  never an answer you came up with.
  Questions starting with What/Which/Who/Where/When/How/Why are ALWAYS "choice".
  You never answer the question yourself; the decision model does.
- "state": the facts/context needed to decide, written as plain sentences.
  Include everything from the question; do not answer it.
- "instructions": one short imperative sentence saying what to decide,
  e.g. "Select the capital of Australia." For "noul", phrase it as a proposition
  to judge, e.g. "Judge whether the proposition is true: The Earth is flat."
- "criteria": only for "choice". 2 to 8 candidates, each
  {"id": "snake_case_id", "description": "one sentence"}.
  Candidates must be mutually exclusive, plausible, and include the correct answer.
  Do not order them so the correct answer is always first.
  NEVER reveal which candidate is correct: each description only names the option
  (e.g. "Mercury."), with no facts, distances or justification.
  The "state" must not contain the answer either.

Example input: My order arrived and the customer says the problem is fixed. What should support do next?
Example output:
{"primitive": "choice",
 "state": "The order was delivered. The customer confirms that the issue is resolved.",
 "instructions": "Select the next support action.",
 "criteria": [
  {"id": "escalate", "description": "Escalate to a human agent."},
  {"id": "close_ticket", "description": "Close the ticket as resolved."},
  {"id": "refund", "description": "Issue a refund."}]}

Example input: What is the capital of Canada?
Example output:
{"primitive": "choice",
 "state": "The question asks for the capital city of Canada.",
 "instructions": "Select the capital of Canada.",
 "criteria": [
  {"id": "toronto", "description": "Toronto."},
  {"id": "ottawa", "description": "Ottawa."},
  {"id": "montreal", "description": "Montreal."},
  {"id": "vancouver", "description": "Vancouver."}]}

Example input: Do penguins live at the North Pole?
Example output:
{"primitive": "noul",
 "state": "The question asks whether penguins live at the North Pole.",
 "instructions": "Judge whether the proposition is true: Penguins live at the North Pole."}
"""


def _extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in translator output: {text!r}")
    return json.loads(text[start : end + 1])


def normalize(raw):
    """Fill the bookkeeping fields and canonical yes/no criteria, then validate."""
    rid = raw.get("id") or f"q-{uuid.uuid4().hex[:8]}"
    request = {
        "id": rid,
        "group_id": raw.get("group_id") or rid,
        "primitive": raw.get("primitive", "choice"),
        "state": raw.get("state", ""),
        "instructions": raw.get("instructions", ""),
        "criteria": raw.get("criteria"),
    }
    if request["primitive"] != "choice":
        request["criteria"] = BINARY_CRITERIA
    validate_request(request)
    return request


class Translator:
    def __init__(self, model_path=TRANSLATOR_MODEL):
        self.model, self.tokenizer = load(model_path)
        self.sampler = make_sampler(temp=0.0)

    def translate(self, question, retries=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        error = None
        for _ in range(retries + 1):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=800,
                sampler=self.sampler,
            )
            try:
                return normalize(_extract_json(text))
            except (ValueError, TypeError) as e:
                error = e
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Invalid: {e}. Output only the corrected JSON."},
                ]
        raise ValueError(f"translator failed: {error}")
