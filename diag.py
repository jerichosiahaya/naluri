"""Is the model reading the state? A shortcut check on the typed-decisions test split.

For every question: the answer with the real state vs with an EMPTY state. If they mostly agree, the model is
answering from the option wording alone. Also prints, per choice question, the gold vs predicted answer mix
(a model that collapses onto one option shows up immediately). No API calls.
Usage: python diag.py runs/<run>
"""
import collections
import json
import sys

from datasets import load_dataset

import s1

tok, model = s1.load(sys.argv[1], device="cuda")
same, acc, tot = collections.Counter(), collections.Counter(), collections.Counter()
mix = collections.defaultdict(lambda: (collections.Counter(), collections.Counter()))
for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test"):
    state, qs, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
    real, empty = s1.predict(tok, model, state, qs), s1.predict(tok, model, "", qs)
    for k, q in qs.items():
        t = q["type"]
        g = str(gold[k]["label"]).lower() if t == "noul" else (int(gold[k]["label"]) if t == "score" else str(gold[k]["label"]))
        guess = ("true" if real[k]["true"] >= 0.5 else "false") if t == "noul" else max(real[k], key=real[k].get)
        acc[t] += guess == g
        same[t] += max(real[k], key=real[k].get) == max(empty[k], key=empty[k].get)
        tot[t] += 1
        if t == "choice":
            mix[(r["workflow"], k)][0][g] += 1
            mix[(r["workflow"], k)][1][guess] += 1
for t in ("choice", "noul", "score"):
    print(f"{t:7} accuracy {acc[t] / tot[t]:.3f} | same answer as with an EMPTY state: {same[t] / tot[t]:.0%}")
print(f"overall accuracy {sum(acc.values()) / sum(tot.values()):.3f}")
for (wf, k), (g, p) in sorted(mix.items()):
    print(f"  {wf} / {k}\n    gold {dict(g.most_common())}\n    pred {dict(p.most_common())}")
