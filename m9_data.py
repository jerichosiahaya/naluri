"""M9 data = M8's field layout, pushed harder where M8 was still weak, plus yes/no reading questions.

M8 diagnosis: field-layout NLI as a choice question lagged the plain layout (65% vs 82% on val), option
descriptions that talk about the fields confused it, and two-field yes/no questions leaned to "false".
 - multi-field tasks (xnli, race) go to the field layout 70% of the time, with descriptions that name the fields
 - new yes/no reading tasks, balanced: SQuAD 2.0 answerability, MultiRC answer checking, RACE option checking

Usage: python m9_data.py  -> data/m9/{train,val,gen_dev}.jsonl
"""
import json
import random
import shutil
from pathlib import Path

from datasets import load_dataset

import m8_data as m8

SRC, OUT = Path("data/m6"), Path("data/m9")
rng = m8.rng
P_FIELD = {"xnli": 0.7, "race": 0.7}  # others stay at 0.5
N_NEW, N_VAL = 6000, 150

NLI_DESCS = [("`{p}` implies `{h}`", "`{p}` neither implies nor contradicts `{h}`", "`{p}` contradicts `{h}`"),
             ("the {h} follows from the {p}", "the {p} does not settle the {h}", "the {h} conflicts with the {p}"),
             ("`{h}` must be true if `{p}` is true", "`{h}` could be true or false", "`{h}` cannot be true if `{p}` is true")]
NLI_ORDER = {n: i for names in [("entailment", "neutral", "contradiction"), ("yes", "maybe", "no"),
                                ("true", "unknown", "false"), ("follows", "unrelated", "contradicts"),
                                ("implied", "not determined", "contradicted")] for i, n in enumerate(names)}
PASSAGE = ["passage", "context", "document", "article", "text"]


def fielded(r):
    new = m8.fielded(r)
    if new and r["task"] == "xnli" and new["q"]["type"] == "choice":
        p, h = [k for pair in m8.NLI_FIELDS if set(pair) == set(new["state"]) for k in pair]
        descs = rng.choice(NLI_DESCS + [("", "", "")])
        new["q"]["criteria"] = {k: descs[NLI_ORDER[k]].format(p=p, h=h) for k in new["q"]["criteria"]}
    return new


def noul(task, passage, fields, field_ins, plain_ins, gold):
    """One yes/no reading row: 70% as named fields, else the passage as a bare text with the rest in the question."""
    passage = passage[:1500]  # ponytail: char clip so the short fields survive the 512-token cut; sentence-aware clip if it matters
    if rng.random() < 0.7:
        pf = rng.choice(PASSAGE)
        state = m8.shuffled({pf: passage, **fields})
        ins = rng.choice(field_ins).format(p=pf)
    else:
        state, ins = passage, rng.choice(plain_ins).format(**fields)
    return {"task": task, "lang": "en", "state": state, "q": {"type": "noul", "instructions": ins}, "gold": int(gold)}


def balanced(rows, n):
    pos, neg = [r for r in rows if r["gold"]], [r for r in rows if not r["gold"]]
    k = min(n // 2, len(pos), len(neg))
    return rng.sample(pos, k) + rng.sample(neg, k)


def squad(split, n):  # is the question answerable from this paragraph? SQuAD 2.0 marks the unanswerable ones
    ds = load_dataset("rajpurkar/squad_v2", split=split).shuffle(seed=0).select(range(n * 3))
    return balanced([noul("squad_answerable", r["context"], {"question": r["question"]},
                          ["Does `{p}` contain the answer to `question`?", "Can `question` be answered from `{p}`?",
                           "Is `{p}` enough to answer `question`?"],
                          ['Does the text contain the answer to "{question}"?', 'Can "{question}" be answered from the text?'],
                          len(r["answers"]["text"]) > 0) for r in ds], n)


def multirc(split, n):  # is this candidate answer right?
    ds = load_dataset("aps/super_glue", "multirc", split=split).shuffle(seed=0).select(range(n * 3))
    return balanced([noul("multirc", r["paragraph"], {"question": r["question"], "answer": r["answer"]},
                          ["According to `{p}`, is `answer` a correct answer to `question`?",
                           "Does `answer` correctly answer `question`, given `{p}`?", "Is `answer` right for `question`? Check `{p}`."],
                          ['According to the text, is "{answer}" a correct answer to "{question}"?',
                           'Given the text, does "{answer}" correctly answer "{question}"?'],
                          r["label"] == 1) for r in ds], n)


def race_tf(rows, n):  # existing RACE rows: is this one option the right answer?
    out = []
    for r in rng.sample(rows, min(n, len(rows))):
        opts = list(r["q"]["criteria"])
        right = rng.random() < 0.5
        ans = opts[r["gold"]] if right else rng.choice([o for i, o in enumerate(opts) if i != r["gold"]])
        out.append(noul("race_tf", r["state"], {"question": r["q"]["instructions"], "answer": ans},
                        ["According to `{p}`, is `answer` the correct answer to `question`?", "Is `answer` right for `question`? Check `{p}`."],
                        ['According to the text, is "{answer}" the right answer to "{question}"?'], right))
    return out


def convert(name, extra):
    rows = [json.loads(l) for l in open(SRC / name)]
    out, n = [], 0
    for r in rows:
        new = fielded(r) if rng.random() < P_FIELD.get(r["task"], 0.5) else None
        n += new is not None
        out.append(new or r)
    out += extra + race_tf([r for r in rows if r["task"] == "race" and r["q"]["type"] == "choice"], len(extra) // 2)
    with open(OUT / name, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(out)} rows, {n} re-rendered, {len(out) - len(rows)} new yes/no reading rows")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    t = fielded({"task": "xnli", "lang": "en", "state": "P.", "gold": 0,
                 "q": {"type": "choice", "instructions": 'How does the statement "H" relate to the text?',
                       "criteria": {"contradicts": "x", "follows": "y", "unrelated": "z"}}})
    assert list(t["q"]["criteria"]) == ["contradicts", "follows", "unrelated"], t  # option order (and so gold) is untouched
    d = t["q"]["criteria"]
    assert not d["follows"] or ("implies" in d["follows"] or "follows" in d["follows"] or "must be true" in d["follows"]), d
    convert("train.jsonl", squad("train", N_NEW) + multirc("train", N_NEW))
    convert("val.jsonl", squad("validation", N_VAL) + multirc("validation", N_VAL))
    shutil.copy(SRC / "gen_dev.jsonl", OUT / "gen_dev.jsonl")
