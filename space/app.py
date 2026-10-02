"""Naluri demo: ask typed questions about a text or JSON state (Hugging Face Space, CPU)."""
import importlib.util
import json
import sys

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

REPO = "jerichosiahaya/naluri-xlmr-base"
torch.set_num_threads(2)  # free CPU Spaces have 2 vCPUs

spec = importlib.util.spec_from_file_location("naluri", hf_hub_download(REPO, "naluri.py"))
naluri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(naluri)
tok, model = naluri.from_pretrained(REPO, device="cpu")


def parse_state(text):
    text = text.strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def parse_options(qtype, text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if qtype == "score":
        return lines
    crit = {}
    for l in lines:
        key, _, desc = l.partition(":")
        crit[key.strip()] = desc.strip()
    return crit


def ask(state_text, qtype, instructions, options_text):
    if not state_text.strip() or not instructions.strip():
        raise gr.Error("Fill in the state and the question.")
    q = {"type": qtype, "instructions": instructions.strip()}
    crit = parse_options(qtype, options_text)
    if qtype == "choice" and len(crit) < 2:
        raise gr.Error("A choice question needs at least 2 options, one per line (key: description).")
    if qtype == "score" and len(crit) < 2:
        raise gr.Error("A score question needs at least 2 levels, one per line, lowest first.")
    if crit and qtype in ("choice", "score"):
        q["criteria"] = crit
    elif crit and qtype == "noul":
        q["criteria"] = {k.lower(): v for k, v in crit.items() if k.lower() in ("true", "false")}
    probs = naluri.predict(tok, model, parse_state(state_text), {"q": q}, device="cpu")["q"]
    if qtype == "score":
        expected = sum(k * v for k, v in probs.items())
        labels = {f"{k}: {crit[k]}": v for k, v in probs.items()}
        return labels, f"Expected level: {expected:.2f} (0 = {crit[0]}, {len(crit) - 1} = {crit[-1]})"
    if qtype == "noul":
        return {"true (yes)": probs["true"], "false (no)": probs["false"]}, f"p(true) = {probs['true']:.2f}"
    best = max(probs, key=probs.get)
    return probs, f"Answer: {best} ({probs[best]:.2f})"


EXAMPLES = [
    ['{"message": "Kak, barangnya baru sampai tapi layarnya retak. Tolong kirim unit pengganti ya, saya butuh buat kerja besok.",\n "order": {"id": "INV-2291", "item": "Monitor 24 inch", "status": "delivered", "paid": true}}',
     "choice", "What does the customer want?",
     "replacement: get a new unit for a damaged item\nrefund: get their money back\ntrack_order: know where the order is\nproduct_question: ask about a product"],
    ["Kalau sampai besok refund saya belum masuk, saya batalkan semua langganan dan pindah ke kompetitor.",
     "noul", "Is the customer threatening to cancel or leave?", ""],
    ["The blender stopped working after two days and nobody answers my emails. Worst purchase ever.",
     "score", "How upset is the customer?", "calm\nannoyed\nfrustrated\nangry"],
    ["memek kau, slot anjing, penipu",
     "choice", "What is the sentiment of this message?",
     "positive: happy or praising\nneutral: no clear emotion\nnegative: angry, insulting or complaining"],
]

with gr.Blocks(title="Naluri demo") as demo:
    gr.Markdown(
        "# Naluri: instinctive, multilingual decisions in one pass\n"
        "Give it a **state** (text or JSON) and a **typed question** with its options written out. Naluri reads them "
        "together and returns a probability per option. Model: "
        f"[{REPO}](https://huggingface.co/{REPO}) (CC BY-NC 4.0) · "
        "[code + report](https://github.com/jerichosiahaya/naluri). Runs on a free CPU, about 0.1–0.3 s per question "
        "(the first request after the Space wakes up is slower).")
    with gr.Row():
        with gr.Column():
            state = gr.Textbox(label="State (text or JSON)", lines=6)
            qtype = gr.Radio(["choice", "noul", "score"], value="choice", label="Question type",
                             info="choice = pick one · noul = yes/no · score = ordered levels")
            instructions = gr.Textbox(label="Question", lines=1)
            options = gr.Textbox(label="Options (one per line)", lines=5,
                                 info="choice: key: description · score: levels, lowest first · noul: optional "
                                      "'true: …' / 'false: …' lines")
            btn = gr.Button("Ask Naluri", variant="primary")
        with gr.Column():
            out = gr.Label(label="Probabilities", num_top_classes=10)
            summary = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[state, qtype, instructions, options])
    gr.Markdown("**Known limits (zero-shot):** routing to your own custom labels can be hit-and-miss, and yes/no "
                "questions about structured JSON (e.g. 'is this transfer risky?') react only weakly to the data. "
                "Fine-tuning on your own labeled examples fixes most of this. Probabilities close to each other mean "
                "the model is unsure, which is a signal to escalate.")
    btn.click(ask, [state, qtype, instructions, options], [out, summary])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)  # --share: temporary public link while this runs
