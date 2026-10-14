"""M8 data: half of the public-task rows of data/m6 re-rendered in the field layout that /v1/systemone
callers use (state = named fields, the question points at them with `backticks`). M6 only ever saw a
bare text as the state, so it ignores inputs like {"premise": ..., "hypothesis": ...}.

No new sources and no new labels. Usage: python m8_data.py  -> data/m8/{train,val,gen_dev}.jsonl
"""
import json
import random
import re
import shutil
from pathlib import Path

SRC, OUT = Path("data/m6"), Path("data/m8")
rng = random.Random(0)

NLI_FIELDS = [("premise", "hypothesis"), ("text", "statement"), ("context", "claim"), ("passage", "sentence")]
NLI_CHOICE = ["What is the relation of `{h}` to `{p}`?", "Given `{p}`, is `{h}` true, false or unknown?",
              "Compare `{h}` with `{p}`: which relation holds?", "Judge `{h}` against `{p}`."]
NLI_NOUL = ["Does `{p}` imply `{h}`?", "Is `{h}` supported by `{p}`?", "Can `{h}` be concluded from `{p}`?",
            "Based on `{p}`, is `{h}` true?"]
RACE_FIELDS = [("passage", "question"), ("article", "question"), ("document", "query"), ("context", "question")]
RACE_INS = ["Using `{p}`, which option answers `{q}`?", "Pick the correct answer to `{q}` according to `{p}`.",
            "What is the answer to `{q}`? Read `{p}`."]
TEXT_FIELDS = {"toxicity": ["comment", "tweet", "text", "message", "post"],
               "emotion": ["text", "message", "tweet", "entry"],
               "sms_spam": ["sms", "message", "text"],
               "langid": ["text", "sentence", "input"],
               "massive": ["utterance", "request", "user_message", "customer_message", "query"]}
POINTER = re.compile(r"\b(this|the) (text|message|comment|sms|request|review)\b", re.I)
WRAP = ["In `{f}`: {ins}", "{ins} (see `{f}`)", "About `{f}`: {ins}"]


def point_at(ins, field):
    """Make the instruction name the field: swap 'this text' for `field`, else wrap it."""
    new, n = POINTER.subn(f"`{field}`", ins, count=1)
    return new if n else rng.choice(WRAP).format(f=field, ins=ins)


def shuffled(d):
    items = list(d.items())
    rng.shuffle(items)
    return dict(items)


def fielded(r):
    """One row in the field layout, or None when the task has no such form (synthetic rows already have one)."""
    task, state, q = r["task"], r["state"], dict(r["q"])
    if task == "xnli" and isinstance(state, str):
        m = re.search(r'"(.*)"', q["instructions"], re.S)
        if not m:
            return None
        p, h = rng.choice(NLI_FIELDS)
        q["instructions"] = rng.choice(NLI_NOUL if q["type"] == "noul" else NLI_CHOICE).format(p=p, h=h)
        state = shuffled({p: state, h: m.group(1)})
    elif task == "race" and isinstance(state, str):
        p, qf = rng.choice(RACE_FIELDS)
        state, q["instructions"] = {p: state, qf: q["instructions"]}, rng.choice(RACE_INS).format(p=p, q=qf)
    elif task in TEXT_FIELDS and isinstance(state, str):
        f = rng.choice(TEXT_FIELDS[task])
        state, q["instructions"] = {f: state}, point_at(q["instructions"], f)
    elif isinstance(state, dict) and not task.startswith("synth") and state:
        f = max(state, key=lambda k: len(str(state[k])))  # the field carrying the text
        q["instructions"] = point_at(q["instructions"], f)
    else:
        return None
    return {**r, "state": state, "q": q}


def convert(name):
    rows = [json.loads(l) for l in open(SRC / name)]
    n = 0
    with open(OUT / name, "w") as out:
        for r in rows:
            new = fielded(r) if rng.random() < 0.5 else None
            n += new is not None
            out.write(json.dumps(new or r, ensure_ascii=False) + "\n")
    print(f"{name}: {n}/{len(rows)} rows in field layout")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    # self-check: the hypothesis moves out of the question into its own field, nothing is lost
    t = fielded({"task": "xnli", "lang": "en", "state": "P.", "gold": 0,
                 "q": {"type": "noul", "instructions": 'Does the text imply that "H "x" y"?'}})
    assert sorted(t["state"].values()) == ['H "x" y', "P."] and '"' not in t["q"]["instructions"], t
    assert point_at("Is this text toxic?", "tweet") == "Is `tweet` toxic?"
    convert("train.jsonl")
    convert("val.jsonl")
    shutil.copy(SRC / "gen_dev.jsonl", OUT / "gen_dev.jsonl")  # unseen-task dev set stays as it was
