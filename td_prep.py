"""typed-decisions train split -> data/td/{train,val}.jsonl for fine-tuning (test split untouched).

Each question becomes one record with the teacher's soft distribution as "target" (in render_options order)
and its label as "gold". 10% of cases (whole cases) are held out for calibration and epoch selection.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data/td"
rows = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
random.Random(20260930).shuffle(rows)
n_val = len(rows) // 10


def records(case):
    state, qs, gold = json.loads(case["state"]), json.loads(case["questions"]), json.loads(case["gold"])
    for qid, q in qs.items():
        g = gold[qid]
        if q["type"] == "noul":
            labels = ["false", "true"]
        elif q["type"] == "choice":
            labels = list(q["criteria"])
        else:
            labels = [str(i) for i in range(len(q["criteria"]))]
        target = [float(g["probabilities"].get(l, 0.0)) for l in labels]
        total = sum(target) or 1.0
        yield {"task": case["workflow"], "lang": "en", "state": state, "q": q,
               "gold": labels.index(str(g["label"]).lower() if q["type"] == "noul" else str(g["label"])),
               "target": [t / total for t in target]}


OUT.mkdir(parents=True, exist_ok=True)
for name, part in [("val", rows[:n_val]), ("train", rows[n_val:])]:
    recs = [r for case in part for r in records(case)]
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(f"{name}: {len(part)} cases, {len(recs)} questions")
