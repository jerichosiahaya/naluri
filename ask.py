"""Ask Naluri typed questions about your own data, on CPU or GPU.

  python ask.py                                  # built-in examples
  python ask.py --state "text or JSON" --questions questions.json
  python ask.py --model runs/naluri-xlmr-base-m2-td --device cuda

--state takes plain text or a JSON object/array (parsed if it looks like JSON).
--questions is a JSON file: {"id": {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}
"""
import argparse
import json
import time

import torch

import s1

EXAMPLES = [
    ("Halo kak, saya sudah transfer tapi pesanan belum diproses. Tolong dicek segera, besok mau dipakai!",
     {"intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"payment_check": "confirm a payment or order status", "refund": "get money back",
                              "product_question": "ask about a product", "complaint": "complain about service"}},
      "urgent": {"type": "noul", "instructions": "Is this request urgent?"},
      "mood": {"type": "score", "instructions": "How upset is the customer?",
               "criteria": ["calm", "slightly annoyed", "frustrated", "angry"]}}),
    ({"order": {"id": "A-1029", "total": 450, "status": "delayed", "days_late": 6}, "customer": {"tier": "VIP", "orders": 38},
      "note": "Customer asked twice already, threatening to cancel."},
     {"action": {"type": "choice", "instructions": "What should support do next?",
                 "criteria": {"apologize": "send an apology only", "discount": "offer a discount voucher",
                              "escalate": "escalate to a supervisor now", "wait": "wait for the courier update"}},
      "churn_risk": {"type": "noul", "instructions": "Is this customer likely to leave?"}}),
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/naluri-xlmr-base-m2", help="run dir (M2 = best general model)")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
ap.add_argument("--state", help="text, or a JSON object/array")
ap.add_argument("--questions", help="path to a questions JSON file")
ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

t = time.perf_counter()
tok, model = s1.load(args.model, device=args.device)
print(f"loaded {args.model} on {args.device} in {time.perf_counter() - t:.1f}s")

if args.state:
    state = args.state.strip()
    if state[:1] in "{[":
        state = json.loads(state)
    cases = [(state, json.load(open(args.questions, encoding="utf-8")))]
else:
    cases = EXAMPLES

s1.predict(tok, model, "warm-up", {"q": {"type": "noul", "instructions": "ok?"}}, device=args.device)
for state, questions in cases:
    t = time.perf_counter()
    answers = s1.predict(tok, model, state, questions, device=args.device)
    ms = (time.perf_counter() - t) * 1000
    print(f"\nstate: {json.dumps(state, ensure_ascii=False)[:120]}")
    for qid, probs in answers.items():
        q = questions[qid]
        if q["type"] == "noul":
            summary = f"p(true) = {probs['true']:.2f}"
        elif q["type"] == "score":
            exp = sum(k * v for k, v in probs.items())
            summary = f"score {exp:.2f} of {len(probs) - 1} ({q['criteria'][max(probs, key=probs.get)]})"
        else:
            best = max(probs, key=probs.get)
            summary = f"{best} ({probs[best]:.2f})"
        print(f"  {qid:12} {q['type']:6} -> {summary}   {{{', '.join(f'{k}: {v:.2f}' for k, v in probs.items())}}}")
    print(f"  ({len(questions)} questions in {ms:.0f} ms)")
