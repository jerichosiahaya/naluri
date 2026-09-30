"""Print the results tables for REPORT.md from results/*.json (no hand-copied numbers).

Usage: python make_table.py > results/tables.md
Laya's scores come from the M1 results file (same frozen eval sets for every run).
"""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
LAYA = "laya-multilingual (zero-shot)"
EVALS = [  # (file stem, label, fair vs zero-shot Laya?)
    ("massive_seen_langs", "MASSIVE intent, trained languages", "no: task trained"),
    ("massive_heldout_langs", "MASSIVE intent, unseen languages", "no: task trained"),
    ("xnli_seen_langs", "XNLI logic, trained languages", "no: task trained"),
    ("xnli_heldout_langs", "XNLI logic, unseen languages", "no: task trained"),
    ("amazon_score", "Amazon stars (1-5, exact)", "no: task trained"),
    ("sentiment_new_domain", "Sentiment, new domain + languages", "partly: similar skill"),
    ("xcopa_unseen_task", "XCOPA cause/effect (unseen task)", "yes"),
    ("sib200_unseen_task", "SIB-200 topic (unseen task**)", "yes**"),
    ("belebele_unseen_task", "Belebele reading (unseen task*)", "yes*"),
]


def load(*names):
    out = {}
    for n in names:
        p = R / n
        if p.exists():
            for k, v in json.load(open(p)).items():
                out.setdefault(k, {}).update(v)
    return out


runs = {"M0": load("naluri-xlmr-base-m0.json", "naluri-xlmr-base-m0_new_evals.json"),
        "M1": load("naluri-xlmr-base-m1.json"),
        "M2": load("naluri-xlmr-base-m2.json"),
        "M3": load("naluri-xlmr-base-m3.json"),
        "M4": load("naluri-xlmr-base-m4.json"),  # mean of 2 seeds
        "M5": load("naluri-xlmr-base-m5.json"),
        "M6": load("naluri-xlmr-base-m6.json")}
laya = load("naluri-xlmr-base-m1.json")


def ours(res, name):
    r = res.get(name, {})
    return r.get("naluri") or r.get("xlmr-system-one")  # key renamed after M1


def cell(r, key="acc", pct=True):
    if not r or key not in r:
        return "–"
    return f"{r[key] * 100:.1f}%" if pct else f"{r[key]:.3f}"


present = [k for k, v in runs.items() if v]
print("| Eval set | Random | " + " | ".join(f"Naluri {k}" for k in present) + " | Laya (zero-shot) | Fair vs Laya? |")
print("|---|---|" + "---|" * len(present) + "---|---|")
for stem, label, fair in EVALS:
    rnd = laya.get(stem, {}).get("random")
    accs = {k: ours(runs[k], stem) for k in present}
    lay = laya.get(stem, {}).get(LAYA)
    best = max([a["acc"] for a in accs.values() if a] + ([lay["acc"]] if lay else []))
    fmt = lambda r: (f"**{cell(r)}**" if r and r["acc"] == best else cell(r))
    extra = lambda r: f" (MAE {r['score_mae']:.2f})" if r and "score_mae" in r else ""
    print(f"| {label} | {rnd * 100:.1f}% | " + " | ".join(fmt(accs[k]) + extra(accs[k]) for k in present)
          + f" | {fmt(lay)}{extra(lay)} | {fair} |")

print("\n| Metric (range over eval sets) | " + " | ".join(f"Naluri {k}" for k in present) + " | Laya (zero-shot) |")
print("|---|" + "---|" * (len(present) + 1))
for key, label in [("ece", "Calibration error (ECE, lower is better)"), ("order_consistency", "Order consistency (higher is better)"),
                   ("p50_ms", "Latency p50, ms (RTX 5050)")]:
    def rng(get):
        vals = [get(s)[key] for s, _, _ in EVALS if get(s) and key in get(s)]
        return f"{min(vals):.2f}–{max(vals):.2f}" if vals else "–"
    print(f"| {label} | " + " | ".join(rng(lambda s, k=k: ours(runs[k], s)) for k in present)
          + f" | {rng(lambda s: laya.get(s, {}).get(LAYA))} |")
