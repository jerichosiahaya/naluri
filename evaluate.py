"""Evaluate a trained run and zero-shot laya-multilingual on the same eval questions.

Per eval file: accuracy (overall + per language), random baseline, ECE, and for choice questions an
order-consistency score (same answer after shuffling the options). Latency is single-question p50.
Usage: python evaluate.py runs/m0 [--no-laya] [--limit N]
"""
import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import s1

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--no-laya", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="max items per eval file (0 = all)")
ap.add_argument("--order-n", type=int, default=300, help="choice items per file for the order test")
ap.add_argument("--only", nargs="*", help="eval file stems to run (default: all)")
args = ap.parse_args()
rng = random.Random(0)


def labels_of(q):
    if q["type"] == "score":
        return list(range(len(q["criteria"])))
    return ["false", "true"] if q["type"] == "noul" else list(q["criteria"])


def shuffled(q):
    keys = list(q["criteria"])
    rng.shuffle(keys)
    return q | {"criteria": {k: q["criteria"][k] for k in keys}}


def ece(conf, correct, bins=15):
    conf, correct = np.array(conf), np.array(correct, dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


# --- predictors: (state, q) -> {label: prob} ---------------------------------------------------------
tok, model = s1.load(args.run)


def ours(state, q):
    return s1.predict(tok, model, state, {"q": q})["q"]


predictors = {"naluri": ours}
if not args.no_laya:
    import laya
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")

    def laya_ml(state, q):
        a = agent.predict(state, {"q": q})["answers"]["q"]
        if q["type"] == "noul":
            return {"false": 1 - a["noul"], "true": a["noul"]}
        if q["type"] == "score":
            return {int(k): v for k, v in a["probabilities"].items()}
        return a["probabilities"]

    predictors["laya-multilingual (zero-shot)"] = laya_ml

results = {}
for path in sorted((HERE / "data/eval").glob("*.jsonl")):
    if args.only and path.stem not in args.only:
        continue
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if args.limit:
        rng.shuffle(recs)
        recs = recs[:args.limit]
    name = path.stem
    results[name] = {"n": len(recs), "random": round(statistics.mean(1 / len(labels_of(r["q"])) for r in recs), 3)}
    for pname, fn in predictors.items():
        per_lang, conf, correct, ms, abs_err = defaultdict(list), [], [], [], []
        for r in tqdm(recs, desc=f"{name} | {pname}", unit="q", leave=False):
            t = time.perf_counter()
            probs = fn(r["state"], r["q"])
            ms.append((time.perf_counter() - t) * 1000)
            pred = max(probs, key=probs.get)
            ok = pred == labels_of(r["q"])[r["gold"]]
            per_lang[r["lang"]].append(ok)
            conf.append(max(probs.values()))
            correct.append(ok)
            if r["q"]["type"] == "score":  # expected level vs gold level
                abs_err.append(abs(sum(k * v for k, v in probs.items()) - r["gold"]))
        res = {"acc": round(float(np.mean(correct)), 3), "ece": round(ece(conf, correct), 3),
               "p50_ms": round(statistics.median(ms), 1),
               "per_lang": {l: round(float(np.mean(v)), 3) for l, v in sorted(per_lang.items())}}
        if abs_err:
            res["score_mae"] = round(float(np.mean(abs_err)), 3)
        choice = [r for r in recs if r["q"]["type"] == "choice"][:args.order_n]
        if choice:
            same = 0
            for r in choice:
                a, b = fn(r["state"], r["q"]), fn(r["state"], shuffled(r["q"]))
                same += max(a, key=a.get) == max(b, key=b.get)
            res["order_consistency"] = round(same / len(choice), 3)
        results[name][pname] = res
        print(f"{name:24} {pname:32} acc {res['acc']:.3f}  mae {res.get('score_mae', float('nan')):.2f}  (random {results[name]['random']:.3f})  "
              f"ece {res['ece']:.3f}  order {res.get('order_consistency', float('nan')):.3f}  p50 {res['p50_ms']} ms")

out = HERE / "results" / f"{Path(args.run).name}{'_' + '_'.join(args.only) if args.only else ''}.json"
out.parent.mkdir(exist_ok=True)
json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
print(f"-> {out}")
