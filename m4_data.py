"""M4 data: M3's mix (public tasks + synthetic decisions) + public topic data, and a mixed dev set.

Topic data (none of it is SIB-200 or AG News, whose test sets stay held out):
  MasakhaNEWS (16 African languages, 7 news topics), Yahoo Answers (10 topics), DBpedia (14 categories).
Dev set for epoch selection: SIB-200 + XCOPA validation splits + 300 held-out synthetic decisions,
scored as the mean of per-task accuracies.
"""
import json
import random
import re

from datasets import get_dataset_config_names, load_dataset

from prep import choice_record, noul_record

OUT = "data/m4"
rng = random.Random(20261001)
TOPIC_Q = ["What is the topic of this text?", "Which category does this belong to?", "What is this mainly about?",
           "Classify the subject of this text."]


def topic_records(task, lang, state, labels, gold, k_min=4):
    if rng.random() < 0.3:
        positive = rng.random() < 0.5
        asked = gold if positive else rng.choice([l for l in labels if l != gold])
        return noul_record(task, lang, state, f"Is this text mainly about {asked}?", positive)
    k = rng.randint(min(k_min, len(labels)), len(labels))
    options = [gold] + rng.sample([l for l in labels if l != gold], k - 1)
    return choice_record(task, lang, state, rng.choice(TOPIC_Q), options, gold)


def masakhanews(per_lang=300):
    out = []
    for lang in get_dataset_config_names("masakhane/masakhanews"):
        rows = list(load_dataset("masakhane/masakhanews", lang, split="train"))
        labels = sorted({r["category"] for r in rows})
        rng.shuffle(rows)
        for r in rows[:per_lang]:
            state = {"headline": r["headline"], "text": r["text"][:1500]}
            out.append(topic_records("topic_news", lang, state, labels, r["category"]))
    return out


def yahoo(n=4000):
    ds = load_dataset("community-datasets/yahoo_answers_topics", split="train")
    names = [t.lower() for t in ds.features["topic"].names]
    idx = rng.sample(range(len(ds)), n)
    return [topic_records("topic_qa", "en", {"question": f"{r['question_title']} {r['question_content']}".strip(),
                                             "answer": r["best_answer"][:800]}, names, names[r["topic"]])
            for r in (ds[i] for i in idx)]


def dbpedia(n=3000):
    ds = load_dataset("fancyzhx/dbpedia_14", split="train")
    names = [re.sub(r"(?<!^)(?=[A-Z])", " ", c).lower() for c in ds.features["label"].names]
    idx = rng.sample(range(len(ds)), n)
    return [topic_records("topic_wiki", "en", {"title": r["title"], "text": r["content"]}, names, names[r["label"]])
            for r in (ds[i] for i in idx)]


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    topics = masakhanews() + yahoo() + dbpedia()
    print(f"topic records: {len(topics)}")
    with open(f"{OUT}/train.jsonl", "w", encoding="utf-8") as f:
        f.write(open("data/m3/train.jsonl", encoding="utf-8").read())
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in topics)
    open(f"{OUT}/val.jsonl", "w", encoding="utf-8").write(open("data/m3/val.jsonl", encoding="utf-8").read())
    synth_val = [l for l in open("data/m3/val.jsonl", encoding="utf-8") if json.loads(l)["task"] == "synth"]
    with open(f"{OUT}/gen_dev.jsonl", "w", encoding="utf-8") as f:
        f.write(open("data/gen_dev.jsonl", encoding="utf-8").read())
        f.writelines(synth_val)
    n = lambda p: sum(1 for _ in open(p, encoding="utf-8"))
    print(f"m4 train {n(f'{OUT}/train.jsonl')} | val {n(f'{OUT}/val.jsonl')} | gen-dev {n(f'{OUT}/gen_dev.jsonl')} "
          f"(sib200 + xcopa + {len(synth_val)} synthetic)")
