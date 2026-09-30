"""Public benchmarks with published Jev numbers: typed-decisions (test, 400 cases / 2,000 decisions) and AG News.

Scoring follows Laya's fine-tuning notebook: choice = argmax option, noul = p(true) >= 0.5, score = argmax level,
all against the gold "label"; score MAE uses the expected level vs the gold score.
Jev and Laya reference numbers are third-party / published (see REPORT.md), not measured here.
Usage: python bench.py runs/naluri-xlmr-base-m2 [--laya] [--ag-n 2000]
"""
import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import s1

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--laya", action="store_true", help="also run laya-multilingual zero-shot")
ap.add_argument("--ag-n", type=int, default=2000)
ap.add_argument("--jev", action="store_true", help="also run TypeSafe Jev live (needs JEV_API_KEY in .env; capped, cached)")
ap.add_argument("--td-limit", type=int, default=0, help="first N typed-decisions cases only (0 = all 400)")
ap.add_argument("--no-naluri", action="store_true")
args = ap.parse_args()

AG_Q = {"topic": {"type": "choice", "instructions": "What is the topic of this news article?",
                  "criteria": {"World": "world news and politics", "Sports": "sports",
                               "Business": "business and economy", "Sci/Tech": "science and technology"}}}
PUBLISHED = {"typed_decisions": {"Jev 1.13.0 (published)": 0.727, "Laya fine-tuned (published)": 0.766,
                                 "Laya-multilingual zero-shot (published)": 0.342, "teacher self-agreement": 0.735},
             "ag_news": {"Jev 1.13.0 (published)": 0.910, "Laya routed (published)": 0.950}}


def ece(conf, correct, bins=15):
    conf, correct = np.array(conf), np.array(correct, dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


# --- predictors: (state, questions) -> {qid: {label: prob}}; noul labels "false"/"true", score labels ints ---
tok, model = (None, None) if args.no_naluri else s1.load(args.run)
predictors = {} if args.no_naluri else {"naluri": lambda state, qs: s1.predict(tok, model, state, qs)}
if args.jev:
    import jev
    predictors["jev (live)"] = jev.predict
if args.laya:
    import laya
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")

    def laya_ml(state, qs):
        out = {}
        for qid, a in agent.predict(state, qs)["answers"].items():
            if qs[qid]["type"] == "noul":
                out[qid] = {"false": 1 - a["noul"], "true": a["noul"]}
            elif qs[qid]["type"] == "score":
                out[qid] = {int(k): v for k, v in a["probabilities"].items()}
            else:
                out[qid] = a["probabilities"]
        return out

    predictors["laya-multilingual (zero-shot)"] = laya_ml


def score(cases, fn, desc):
    """cases: [(group, state, questions, gold)] -> metrics. gold[qid] has "label" (+ "score" for score)."""
    by_type, by_group, conf, correct, mae, ms = defaultdict(list), defaultdict(list), [], [], [], []
    for group, state, qs, gold in tqdm(cases, desc=desc, unit="case", leave=False):
        t = time.perf_counter()
        pred = fn(state, qs)
        ms.append((time.perf_counter() - t) * 1000)
        for qid, q in qs.items():
            p = pred[qid]
            if q["type"] == "score":
                p = {int(k): v for k, v in p.items()}
                mae.append(abs(sum(k * v for k, v in p.items()) - gold[qid].get("score", gold[qid]["label"])))
                label = int(gold[qid]["label"])
            else:
                label = str(gold[qid]["label"]).lower() if q["type"] == "noul" else str(gold[qid]["label"])
            guess = ("true" if p["true"] >= 0.5 else "false") if q["type"] == "noul" else max(p, key=p.get)
            ok = guess == label
            by_type[q["type"]].append(ok)
            by_group[group].append(ok)
            conf.append(max(p.values()))
            correct.append(ok)
    res = {"acc": round(float(np.mean(correct)), 3), "n_decisions": len(correct), "ece": round(ece(conf, correct), 3),
           "p50_ms_per_case": round(statistics.median(ms), 1),
           "by_type": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_type.items())},
           "by_group": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_group.items())}}
    if mae:
        res["score_mae"] = round(float(np.mean(mae)), 3)
    return res


td = [(r["workflow"], json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"]))
      for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test")]
if args.td_limit:
    td = td[:args.td_limit]
ag_rows = list(load_dataset("fancyzhx/ag_news", split="test"))
random.Random(0).shuffle(ag_rows)
names = ["World", "Sports", "Business", "Sci/Tech"]
ag = [("ag_news", r["text"], AG_Q, {"topic": {"label": names[r["label"]]}}) for r in ag_rows[:args.ag_n]]

results = {}
for bench, cases in [("typed_decisions", td), ("ag_news", ag)]:
    results[bench] = {"published": PUBLISHED[bench]}
    for pname, fn in predictors.items():
        res = score(cases, fn, f"{bench} | {pname}")
        results[bench][pname] = res
        print(f"{bench:16} {pname:32} acc {res['acc']:.3f}  ece {res['ece']:.3f}  "
              f"mae {res.get('score_mae', float('nan')):.3f}  p50 {res['p50_ms_per_case']} ms/case  "
              f"by type {res['by_type']}  by group {res['by_group'] if bench == 'typed_decisions' else ''}")
    print(f"{'':16} published: {PUBLISHED[bench]}")

if args.jev:
    results["jev_usage"] = jev.used
    print("jev usage:", jev.used)
tag = ("jev_" if args.jev else "") + Path(args.run).name + (f"_td{args.td_limit}" if args.td_limit else "")
out = HERE / "results" / f"bench_{tag}.json"
json.dump(results, open(out, "w"), indent=2)
print(f"-> {out}")
