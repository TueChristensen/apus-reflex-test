"""Gradio UI: question -> (small LLM) -> editable OpenJev request -> APUS Reflex."""

import json
from functools import cache

import gradio as gr

from reflex_lab.contracts import render_prompt
from reflex_lab.reflex import Reflex
from reflex_lab.translator import Translator, normalize


@cache
def translator():
    return Translator()


@cache
def reflex():
    return Reflex()


def do_translate(question):
    if not question.strip():
        raise gr.Error("Type a question first.")
    try:
        request = translator().translate(question)
    except ValueError as e:
        raise gr.Error(str(e))
    return json.dumps(request, indent=2, ensure_ascii=False)


def answer_text(request, result):
    """Turn Reflex's pick into the answer shown to the user (no LLM involved)."""
    if result["type"] == "choice":
        top = result["choice"]
        answer = next(c["description"] for c in request["criteria"] if c["id"] == top)
    else:
        top = "yes" if result["yes_probability"] >= 0.5 else "no"
        claim = request["instructions"].split("proposition is true:")[-1].strip()
        verdict = "true" if top == "yes" else "false"
        answer = f"{top.capitalize()}.\n*Judged {verdict}: \u201c{claim}\u201d*"
    return f"## {answer}\n*p = {result['probabilities'][top]:.4f} (uncalibrated)*"


def do_ask(request_json):
    try:
        request = normalize(json.loads(request_json))
        result = reflex().decide(request)
    except (ValueError, TypeError) as e:
        raise gr.Error(str(e))
    descriptions = {c["id"]: c["description"] for c in request["criteria"]}
    probs = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
    label = {descriptions[cid]: p for cid, p in probs}
    return answer_text(request, result), label, render_prompt(request), result


def do_both(question):
    request_json = do_translate(question)
    return request_json, *do_ask(request_json)


with gr.Blocks(title="Reflex Lab") as demo:
    gr.Markdown("# Reflex Lab\nQuestion → Qwen3-4B structures it → **APUS Reflex 9B** answers.")
    question = gr.Textbox(label="Question", lines=2, placeholder="Is a tomato a fruit?")
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Markdown()

    with gr.Accordion("1 · Structured question (editable)", open=False):
        request_box = gr.Code(language="json", interactive=True)
        rerun_btn = gr.Button("Re-ask with this request")
    with gr.Accordion("2 · Reflex scores", open=False):
        scores = gr.Label(show_label=False, num_top_classes=16)
    with gr.Accordion("Raw prompt & response", open=False):
        prompt_box = gr.Textbox(label="Rendered prompt", lines=8)
        raw = gr.JSON(label="Response")

    outputs = [answer, scores, prompt_box, raw]
    ask_btn.click(do_both, question, [request_box, *outputs])
    question.submit(do_both, question, [request_box, *outputs])
    rerun_btn.click(do_ask, request_box, outputs)


def launch():
    demo.queue(default_concurrency_limit=1).launch()
