# Vendored verbatim from apus-ailab/APUS-Reflex-9B-MLX-4bit (openjev_contracts.py), Apache-2.0.
# This is the training contract: do not edit, or prompts will drift from what the model saw.

"""Small shared contract. Prompts use a strict whitelist of input fields."""

import json
import math

PROMPT_VERSION = "jev.dynamic.prompt.v2"
LABELS = tuple("ABCDEFGHIJKLMNOP")
BINARY_CRITERIA = [
    {"id": "yes", "description": "The stated proposition is true."},
    {"id": "no", "description": "The stated proposition is false."},
]


def validate_request(record):
    for key in ("id", "group_id", "state", "instructions"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if record.get("primitive") not in ("choice", "noul", "score_level"):
        raise ValueError("unsupported primitive")
    criteria = record.get("criteria")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= len(LABELS):
        raise ValueError("criteria must contain 2..16 candidates")
    ids = []
    for candidate in criteria:
        if not isinstance(candidate, dict):
            raise TypeError("candidate must be an object")
        for key in ("id", "description"):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ValueError(f"candidate {key} must be nonempty")
        ids.append(candidate["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids")
    if record["primitive"] != "choice" and criteria != BINARY_CRITERIA:
        raise ValueError("noul and score_level require canonical yes/no criteria")


def validate_record(record):
    validate_request(record)
    if record.get("gold") not in [c["id"] for c in record["criteria"]]:
        raise ValueError("gold must be a candidate id")
    if not isinstance(record.get("provenance"), dict):
        raise TypeError("provenance must be an object")


def label_mapping(record):
    validate_request(record)
    return dict(zip(LABELS, (c["id"] for c in record["criteria"])))


def render_prompt_parts(record):
    """Text prefix/suffix; callers MUST check tokenizer boundary equivalence."""
    validate_request(record)
    prefix = "Shared state:\n" + record["state"] + "\n\n"
    task = {
        "primitive": record["primitive"],
        "instructions": record["instructions"],
        "criteria": [
            {"label": label, "description": candidate["description"]}
            for label, candidate in zip(LABELS, record["criteria"])
        ],
    }
    suffix = json.dumps(task, ensure_ascii=False, sort_keys=True)
    suffix += (
        "\nReturn only the selected letter: "
        + ", ".join(LABELS[: len(record["criteria"])])
        + ".\nAnswer:"
    )
    return prefix, suffix


def render_prompt(record):
    return "".join(render_prompt_parts(record))


def to_messages(record):
    validate_record(record)
    inverse = {candidate: label for label, candidate in label_mapping(record).items()}
    return {
        "messages": [
            {"role": "user", "content": render_prompt(record)},
            {"role": "assistant", "content": inverse[record["gold"]]},
        ]
    }


def format_response(record, probabilities):
    """Map ordered candidate probabilities; score_level is NOT aggregate Score."""
    mapping = label_mapping(record)
    values = list(probabilities)
    if len(values) != len(mapping) or any(
        not math.isfinite(p) or p < 0 or p > 1 for p in values
    ):
        raise ValueError("invalid probabilities")
    if not math.isclose(sum(values), 1, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    distribution = dict(zip(mapping.values(), values))
    result = {"type": record["primitive"], "probabilities": distribution}
    if record["primitive"] == "choice":
        result["choice"] = max(distribution, key=distribution.get)
    else:
        result["yes_probability"] = distribution["yes"]
    return result
