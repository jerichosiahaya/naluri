"""Build M0 data: typed questions from public multilingual datasets.

Train (seen tasks):  XNLI (noul + choice), MASSIVE intent (choice with 2-12 options + noul),
                     Amazon reviews (score 1-5 stars + choice), multilingual toxicity, language ID,
                     emotion, RACE reading comprehension, SMS spam. (PAWS-X dropped in M2: never learned.)
Generality dev:      SIB-200 + XCOPA *validation* splits, used only to pick the best epoch.
Eval, seen tasks:    XNLI / MASSIVE test in trained and held-out languages; Amazon test;
                     multilingual sentiment (new domain: tweets/news, new languages incl. id/ms/hi/ar).
Eval, unseen tasks:  SIB-200 topic (7-way), Belebele reading (4-way), XCOPA cause/effect (2-way).
Eval files are frozen: an existing file is never rewritten, so runs stay comparable.

Every record: {"task", "lang", "state", "q": {"type", "instructions", "criteria"}, "gold": option index}
(option index follows s1.render_options order; noul: 0 = false, 1 = true).
Augmentation on train: shuffled options, renamed labels, paraphrased instructions, varied option counts.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUT = Path(__file__).parent / "data"
SEED = 20260929
rng = random.Random(SEED)

XNLI_TRAIN_LANGS = ["en", "es", "fr", "de", "ru", "zh", "ar", "hi", "tr", "vi"]
XNLI_HELDOUT_LANGS = ["sw", "th", "ur", "bg", "el"]
MASSIVE_TRAIN_LANGS = ["en", "id", "es", "fr", "de", "ru", "zh-CN", "ja", "ko", "ar", "hi", "tr", "vi", "th",
                       "pt", "it", "nl", "pl", "fa", "ms"]
MASSIVE_HELDOUT_LANGS = ["sw", "tl", "jv", "km", "am"]
SIB_LANGS = ["eng_Latn", "ind_Latn", "zho_Hans", "arb_Arab", "hin_Deva", "rus_Cyrl", "vie_Latn", "tha_Thai",
             "swh_Latn", "jav_Latn", "yor_Latn", "amh_Ethi"]
BELEBELE_LANGS = SIB_LANGS

N_XNLI, N_MASSIVE, N_PAWSX, N_AMAZON = 2000, 800, 0, 1500      # train questions per language
N_TOX, N_LANGID, N_EMOTION, N_RACE, N_SPAM = 700, 6000, 5000, 8000, 3000
TOX_LANGS = ["en", "ru", "uk", "de", "es", "am", "zh", "ar", "hi", "it", "fr", "he", "tt", "ja"]
LANG_NAMES = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
              "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
              "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu",
              "vi": "Vietnamese", "zh": "Chinese"}
PAWSX_LANGS = ["en", "de", "es", "fr", "ja", "ko", "zh"]
AMAZON_LANGS = ["en", "de", "es", "fr", "ja", "zh"]
SENTIMENT_EVAL_LANGS = ["indonesian", "malay", "hindi", "arabic", "portuguese", "italian", "english"]
XCOPA_LANGS = ["id", "et", "ht", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]
N_EVAL = 200                         # eval questions per language (seen tasks)

# --- augmentation pools ---------------------------------------------------------------------------
NLI_NAMES = [("entailment", "neutral", "contradiction"), ("yes", "maybe", "no"), ("true", "unknown", "false"),
             ("follows", "unrelated", "contradicts"), ("implied", "not determined", "contradicted")]
NLI_DESC = ("the text implies the statement", "the text neither implies nor contradicts the statement",
            "the text contradicts the statement")
NLI_NOUL = ['Does the text imply that "{h}"?', 'Based on the text, is it true that "{h}"?',
            'Can we conclude from the text that "{h}"?', 'Is "{h}" supported by the text?']
NLI_CHOICE = ['How does the statement "{h}" relate to the text?', 'Given the text, is "{h}" true, false or unknown?',
              'What is the relationship between the text and "{h}"?']
INTENT_CHOICE = ["What does the user want?", "Which intent best matches this request?",
                 "Classify the user's request.", "What is the user trying to do?", "Pick the user's intent."]
INTENT_NOUL = ["Is the user asking to {i}?", "Does this request mean: {i}?", "Is the intent of this message '{i}'?"]


PARA_NOUL = ["Do sentence 1 and sentence 2 mean the same thing?", "Are these two sentences paraphrases?",
             "Do both sentences express the same meaning?"]
PARA_NAMES = [("same meaning", "different meaning"), ("paraphrase", "not a paraphrase"), ("equivalent", "different")]
STARS = [["1 star: very negative", "2 stars: negative", "3 stars: mixed or neutral", "4 stars: positive",
          "5 stars: very positive"],
         ["terrible", "bad", "okay", "good", "excellent"],
         ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"]]
RATE = ["How many stars would this review give?", "Rate the reviewer's satisfaction.",
        "How positive is this review?"]
SENTI_NAMES = [("negative", "neutral", "positive"), ("bad", "neutral", "good"), ("unhappy", "neutral", "happy")]
SENTI_CHOICE = ["What is the sentiment of this text?", "How does the writer feel?", "Is this text positive, negative or neutral?"]


TOX_NAMES = [("toxic", "not toxic"), ("offensive", "acceptable"), ("abusive", "polite"), ("hateful or rude", "fine")]
TOX_NOUL = ["Is this text toxic or offensive?", "Does this message contain insults, hate or abuse?",
            "Would a moderator remove this comment?"]
LANGID_CHOICE = ["Which language is this text written in?", "What language is this?", "Identify the language of the text."]
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_CHOICE = ["Which emotion does the writer express?", "How does the writer feel?", "What is the main emotion?"]
SPAM_NOUL = ["Is this message spam?", "Is this an unwanted promotional or scam message?", "Should this SMS go to the spam folder?"]


def human(label):
    return label.replace("_", " ").replace("qa", "question").replace("iot", "smart home")


def choice_record(task, lang, state, instructions, options, gold_key, descs=None):
    """options: list of keys (shuffled here); descs: optional {key: description}."""
    keys = options[:]
    rng.shuffle(keys)
    crit = {k: (descs or {}).get(k, "") for k in keys}
    return {"task": task, "lang": lang, "state": state,
            "q": {"type": "choice", "instructions": instructions, "criteria": crit}, "gold": keys.index(gold_key)}


def noul_record(task, lang, state, instructions, truth, crit=None):
    q = {"type": "noul", "instructions": instructions}
    if crit:
        q["criteria"] = crit
    return {"task": task, "lang": lang, "state": state, "q": q, "gold": int(truth)}


# --- XNLI -----------------------------------------------------------------------------------------
def xnli(lang, split, n, augment):
    ds = load_dataset("facebook/xnli", lang, split=split, streaming=True).shuffle(seed=SEED, buffer_size=10_000)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        prem, hyp, label = row["premise"], row["hypothesis"], row["label"]
        if augment and rng.random() < 0.5:
            out.append(noul_record("xnli", lang, prem, rng.choice(NLI_NOUL).format(h=hyp), label == 0))
        else:
            names = rng.choice(NLI_NAMES) if augment else NLI_NAMES[0]
            descs = dict(zip(names, NLI_DESC)) if (not augment or rng.random() < 0.6) else None
            ins = rng.choice(NLI_CHOICE) if augment else NLI_CHOICE[0]
            out.append(choice_record("xnli", lang, prem, ins.format(h=hyp), list(names), names[label], descs))
    return out


# --- MASSIVE --------------------------------------------------------------------------------------
def massive(lang, split, n, augment, k_eval=8):
    rows = list(load_dataset("mteb/amazon_massive_intent", lang, split=split))
    rng.shuffle(rows)
    intents = sorted({r["label"] for r in rows})
    out = []
    for r in rows[:n]:
        gold = r["label"]
        if augment and rng.random() < 0.3:
            positive = rng.random() < 0.5
            asked = gold if positive else rng.choice([i for i in intents if i != gold])
            out.append(noul_record("massive", lang, r["text"], rng.choice(INTENT_NOUL).format(i=human(asked)), positive))
            continue
        k = rng.randint(2, 12) if augment else k_eval
        options = [gold] + rng.sample([i for i in intents if i != gold], k - 1)
        ins = rng.choice(INTENT_CHOICE) if augment else INTENT_CHOICE[0]
        rec = choice_record("massive", lang, r["text"], ins, [human(o) for o in options], human(gold))
        out.append(rec)
    return out


# --- PAWS-X and Amazon reviews -------------------------------------------------------------------
def pawsx(lang, n):
    rows = list(load_dataset("google-research-datasets/paws-x", lang, split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        state = {"sentence 1": r["sentence1"], "sentence 2": r["sentence2"]}
        same = r["label"] == 1
        if rng.random() < 0.6:
            out.append(noul_record("pawsx", lang, state, rng.choice(PARA_NOUL), same))
        else:
            names = rng.choice(PARA_NAMES)
            out.append(choice_record("pawsx", lang, state, "Do the two sentences have the same meaning?",
                                     list(names), names[0] if same else names[1]))
    return out


def toxicity(lang, n):
    rows = list(load_dataset("textdetox/multilingual_toxicity_dataset", split=lang))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        toxic = int(r["toxic"]) == 1
        if rng.random() < 0.5:
            out.append(noul_record("toxicity", lang, r["text"], rng.choice(TOX_NOUL), toxic))
        else:
            names = rng.choice(TOX_NAMES)
            out.append(choice_record("toxicity", lang, r["text"], "Is this text acceptable or offensive?",
                                     list(names), names[0] if toxic else names[1]))
    return out


def langid(n):
    rows = list(load_dataset("papluca/language-identification", split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        gold = LANG_NAMES[r["labels"]]
        if rng.random() < 0.3:
            positive = rng.random() < 0.5
            asked = gold if positive else rng.choice([l for l in LANG_NAMES.values() if l != gold])
            out.append(noul_record("langid", r["labels"], r["text"], f"Is this text written in {asked}?", positive))
        else:
            k = rng.randint(2, 8)
            options = [gold] + rng.sample([l for l in LANG_NAMES.values() if l != gold], k - 1)
            out.append(choice_record("langid", r["labels"], r["text"], rng.choice(LANGID_CHOICE), options, gold))
    return out


def emotion(n):
    rows = list(load_dataset("dair-ai/emotion", "split", split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        gold = EMOTIONS[r["label"]]
        if rng.random() < 0.3:
            positive = rng.random() < 0.5
            asked = gold if positive else rng.choice([e for e in EMOTIONS if e != gold])
            out.append(noul_record("emotion", "en", r["text"], f"Does the writer feel {asked}?", positive))
        else:
            out.append(choice_record("emotion", "en", r["text"], rng.choice(EMOTION_CHOICE), EMOTIONS, gold))
    return out


def race(n):
    rows = list(load_dataset("ehovy/race", "all", split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n:
            break
        opts = r["options"]
        if len(set(opts)) < len(opts) or "_" in r["question"]:
            continue  # duplicate options collide as keys; fill-in-the-blank questions read oddly as questions
        gold = opts["ABCD".index(r["answer"])]
        out.append(choice_record("race", "en", r["article"], r["question"], opts, gold))
    return out


def sms_spam(n):
    rows = list(load_dataset("ucirvine/sms_spam", split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        spam = r["label"] == 1
        if rng.random() < 0.6:
            out.append(noul_record("sms_spam", "en", r["sms"], rng.choice(SPAM_NOUL), spam))
        else:
            out.append(choice_record("sms_spam", "en", r["sms"], "What kind of message is this?",
                                     ["spam", "normal message"], "spam" if spam else "normal message"))
    return out


def score_record(task, lang, state, instructions, levels, gold):
    return {"task": task, "lang": lang, "state": state,
            "q": {"type": "score", "instructions": instructions, "criteria": levels}, "gold": gold}


def amazon(lang, split, n, augment):
    rows = list(load_dataset("json", data_files=f"hf://datasets/goosmanlei/amazon_reviews_multi/{lang}/{split}.jsonl.gz",
                             split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        state = {"title": r["review_title"], "review": r["review_body"]}
        stars = r["stars"] - 1
        if not augment or rng.random() < 0.6:
            levels = rng.choice(STARS) if augment else STARS[0]
            out.append(score_record("amazon", lang, state, rng.choice(RATE) if augment else RATE[0], levels, stars))
        elif stars != 2:  # polarity choice; skip 3-star reviews as ambiguous
            names = rng.choice(SENTI_NAMES)
            out.append(choice_record("amazon", lang, state, rng.choice(SENTI_CHOICE), [names[0], names[2]],
                                     names[2] if stars > 2 else names[0]))
    return out


# --- held-out tasks -------------------------------------------------------------------------------
def sib200(lang, split="test"):
    rows = load_dataset("Davlan/sib200", lang, split=split)
    cats = sorted({r["category"] for r in rows})
    return [choice_record("sib200", lang, r["text"], "What is the topic of this text?", cats, r["category"]) for r in rows]


def belebele(lang, n=150):
    rows = list(load_dataset("facebook/belebele", lang, split="test"))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        answers = [r[f"mc_answer{i}"] for i in range(1, 5)]
        gold = answers[int(r["correct_answer_num"]) - 1]
        if len(set(answers)) < 4:
            continue  # duplicate answer texts would collide as option keys
        out.append(choice_record("belebele", lang, r["flores_passage"], r["question"], answers, gold))
    return out


def sentiment_eval(n=150):
    rows = [r for r in load_dataset("tasksource/multilingual-sentiments", split="test") if r["language"] in SENTIMENT_EVAL_LANGS]
    rng.shuffle(rows)
    out, count = [], {}
    for r in rows:
        if count.get(r["language"], 0) >= n:
            continue
        count[r["language"]] = count.get(r["language"], 0) + 1
        names = SENTI_NAMES[0]
        out.append(choice_record("sentiment", r["language"], r["text"], SENTI_CHOICE[0], list(names), names[r["label"]]))
    return out


def xcopa(lang, split="test"):
    out = []
    for r in load_dataset("cambridgeltl/xcopa", lang, split=split):
        ins = "What was the cause of this?" if r["question"] == "cause" else "What happened as a result?"
        gold = r["choice1"] if r["label"] == 0 else r["choice2"]
        if r["choice1"] != r["choice2"]:
            out.append(choice_record("xcopa", lang, r["premise"], ins, [r["choice1"], r["choice2"]], gold))
    return out


def write(name, records, frozen=False):
    path = OUT / name
    if frozen and path.exists():
        print(f"{name}: kept existing (frozen)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(records)}")


if __name__ == "__main__":
    train = []
    for lang in tqdm(XNLI_TRAIN_LANGS, desc="xnli train"):
        train += xnli(lang, "train", N_XNLI, augment=True)
    for lang in tqdm(MASSIVE_TRAIN_LANGS, desc="massive train"):
        train += massive(lang, "train", N_MASSIVE, augment=True)
    for lang in tqdm(AMAZON_LANGS, desc="amazon train"):
        train += amazon(lang, "train", N_AMAZON, augment=True)
    for lang in tqdm(TOX_LANGS, desc="toxicity train"):
        train += toxicity(lang, N_TOX)
    train += langid(N_LANGID) + emotion(N_EMOTION) + race(N_RACE) + sms_spam(N_SPAM)
    rng.shuffle(train)
    n_val = 1000
    write("val.jsonl", train[:n_val])
    write("train.jsonl", train[n_val:])

    # Generality dev: validation splits of the unseen task types, only for picking the best epoch.
    rng.seed(SEED + 2)
    write("gen_dev.jsonl", [r for l in SIB_LANGS for r in sib200(l, "validation")] +
                           [r for l in XCOPA_LANGS for r in xcopa(l, "validation")])

    rng.seed(SEED + 1)  # eval sampling independent of how much train data was drawn
    builders = {
        "xnli_seen_langs": lambda: [r for l in XNLI_TRAIN_LANGS[:5] for r in xnli(l, "test", N_EVAL, augment=False)],
        "xnli_heldout_langs": lambda: [r for l in XNLI_HELDOUT_LANGS for r in xnli(l, "test", N_EVAL, augment=False)],
        "massive_seen_langs": lambda: [r for l in MASSIVE_TRAIN_LANGS[:5] for r in massive(l, "test", N_EVAL, augment=False)],
        "massive_heldout_langs": lambda: [r for l in MASSIVE_HELDOUT_LANGS for r in massive(l, "test", N_EVAL, augment=False)],
        "sib200_unseen_task": lambda: [r for l in SIB_LANGS for r in sib200(l)],
        "belebele_unseen_task": lambda: [r for l in BELEBELE_LANGS for r in belebele(l)],
        "amazon_score": lambda: [r for l in AMAZON_LANGS for r in amazon(l, "test", 150, augment=False)],
        "xcopa_unseen_task": lambda: [r for l in XCOPA_LANGS for r in xcopa(l)],
        "sentiment_new_domain": sentiment_eval,
    }
    for name, build in builders.items():
        path = f"eval/{name}.jsonl"
        if (OUT / path).exists():  # frozen: never rebuilt, so every run is scored on the same questions
            print(f"{path}: kept existing (frozen)")
        else:
            write(path, build())
