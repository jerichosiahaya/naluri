"""Naluri (Neural Answering via Label Understanding and Rapid Inference): System One on XLM-R.

Input format (same layout as Laya, so /v1/systemone requests map onto it):
    <s> {type} question: {instructions} </s> <mask> opt0 <mask> opt1 ... </s> {state} </s>
Each option is scored from the encoder vector at its own <mask> by a small MLP; softmax over a
question's options gives the answer distribution. No extra transformer layers: XLM-R is
bidirectional, so every option already attends to the question, the other options and the state.
"""
import json

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

BASE = "FacebookAI/xlm-roberta-base"
MAX_LEN = 512          # XLM-R position limit
HEAD_MAX_LEN = 192     # question + options budget; the state gets the rest
OPT_MAX = 48           # tokens per option


def render_options(q):
    """Option texts in label order. noul is always [false, true]; score levels are ordered."""
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        return [str(k) if not v else f"{k}: {v}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    crit = crit or {}
    return [f"false: {crit.get('false') or 'no, the statement does not hold'}",
            f"true: {crit.get('true') or 'yes, the statement holds'}"]


def serialize_state(state):
    if isinstance(state, str):
        return state
    if isinstance(state, dict):  # nested values as JSON, not Python repr
        return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                         for k, v in state.items())
    return json.dumps(state, ensure_ascii=False)


def build_sequence(tok, state, q, max_len=MAX_LEN):
    """-> (input ids, marker positions). One marker per option, in render_options order."""
    enc = lambda s, n=None: tok(s, add_special_tokens=False, truncation=n is not None, max_length=n)["input_ids"]
    head = enc(f"{q['type']} question: {q['instructions']}")
    opts = [[tok.mask_token_id] + enc(" " + o.replace(tok.mask_token, " "), OPT_MAX) for o in render_options(q)]
    budget = HEAD_MAX_LEN - sum(map(len, opts))
    if budget < 16:  # many options: shrink each so the question keeps some room
        per = max(4, (HEAD_MAX_LEN - 16) // len(opts))
        opts = [o[:per] for o in opts]
        budget = HEAD_MAX_LEN - sum(map(len, opts))
    ids = [tok.cls_token_id] + head[:max(8, budget)] + [tok.sep_token_id]
    markers = []
    for o in opts:
        markers.append(len(ids))
        ids += o
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    ids += enc(serialize_state(state).replace(tok.mask_token, " "))[:room] + [tok.sep_token_id]
    return ids, markers


class SystemOne(nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base, attn_implementation="sdpa")
        d = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("temperature", torch.ones(3))  # per type: choice, score, noul

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = h.gather(1, marker_pos[..., None].expand(-1, -1, h.size(-1)))
        return self.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)


QTYPE = {"choice": 0, "score": 1, "noul": 2}


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    k = max(len(it["markers"]) for it in items)
    b = {"input_ids": torch.full((n, L), pad_id), "attention_mask": torch.zeros((n, L), dtype=torch.long),
         "marker_pos": torch.zeros((n, k), dtype=torch.long), "marker_mask": torch.zeros((n, k), dtype=torch.bool)}
    for i, it in enumerate(items):
        b["input_ids"][i, :len(it["ids"])] = torch.tensor(it["ids"])
        b["attention_mask"][i, :len(it["ids"])] = 1
        b["marker_pos"][i, :len(it["markers"])] = torch.tensor(it["markers"])
        b["marker_mask"][i, :len(it["markers"])] = True
    return b


def load(path, device="cuda"):
    """Load a trained checkpoint dir written by train.py."""
    tok = AutoTokenizer.from_pretrained(path)
    model = SystemOne(path)
    model.scorer.load_state_dict(torch.load(f"{path}/scorer.pt", map_location="cpu"))
    model.temperature.copy_(torch.load(f"{path}/temperature.pt", map_location="cpu"))
    return tok, model.to(device).eval()


@torch.no_grad()
def predict(tok, model, state, questions, device="cuda"):
    """Answer {qid: question} for one state -> {qid: {option: prob}} (noul options are false/true)."""
    items = []
    for q in questions.values():
        ids, markers = build_sequence(tok, state, q)
        items.append({"ids": ids, "markers": markers})
    b = {k: v.to(device) for k, v in collate(items, tok.pad_token_id).items()}
    # bf16 on GPU; plain fp32 on CPU (bf16 is emulated and slower on most CPUs)
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits = model(**b)
    out = {}
    for (qid, q), z in zip(questions.items(), logits):
        labels = ["false", "true"] if q["type"] == "noul" else (
            list(q["criteria"]) if q["type"] == "choice" else list(range(len(q["criteria"]))))
        p = torch.softmax(z[:len(labels)] / model.temperature[QTYPE[q["type"]]], -1).tolist()
        out[qid] = dict(zip(labels, p))
    return out
