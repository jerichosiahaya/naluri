# Naluri

**Instinctive, multilingual decisions in one pass.**

Naluri (Indonesian for *instinct*; **N**eural **A**nswering via **L**abel **U**nderstanding and **R**apid **I**nference) is a small *System One* decision model built on XLM-RoBERTa base (≈280M parameters). You give it a text or JSON state and a typed question with its options written out. It returns calibrated probabilities over those options in a single forward pass: about **5 ms per question on a laptop GPU** and about 50–60 ms on a CPU.

```text
state:     "memek kau, slot anjing, penipu"
question:  What is the sentiment of this message?   positive / neutral / negative
answer:    negative 0.80   neutral 0.16   positive 0.04
```

It speaks the same three question types as TypeSafe Jev and Laya: `noul` (yes/no), `choice` (pick one) and `score` (an ordered scale).

## Results at a glance

All numbers were measured in this repo (single runs). Jev was called live through its API with the same questions. Details, per-language results and caveats are in [REPORT.md](REPORT.md).

| | Naluri | Laya-multilingual | Jev 1.13.0 |
|---|---|---|---|
| **typed-decisions**, after a 17-minute fine-tune on its train split | 72.5% | 76.6% *(published)* | 73.8% *(zero-shot)* |
| typed-decisions, zero-shot | 39.3% (M4) | 35.2% | **73.8%** |
| XCOPA cause/effect, unseen task (11 languages) | **57.2%** (M4) | 54.7% | – |
| SIB-200 topic (12 languages; M4 trains on *other* topic datasets) | **74.4%** (M4) | 70.8% | – |
| AG News topic | 76.0% (M4) | **92.0%** | 88.0% |
| AG News calibration error (lower is better) | **0.034** (M4) | 0.052 | 0.080 |
| Intent in 5 languages never trained on (a task Naluri trained on) | **75.7%** (M2) | 37.6% | – |
| Calibration error (ECE, lower is better) on typed-decisions, fine-tuned | **0.020** | – | 0.042 |
| Calibration error across the 9 eval sets in REPORT.md | **0.03–0.07** (M2) | 0.12–0.56 | – |
| Median time per typed-decisions case | **31–54 ms** (laptop GPU) | 94 ms | 307 ms (API) |

- On typed-decisions, the fine-tuned model and Jev are a **statistical tie** (exact McNemar test, p = 0.29).
- **Where Naluri wins:** speed, cost ($0 per call, self-hosted), calibrated confidence on the tasks it knows, robustness to option order, and cross-lingual transfer.
- **Where it doesn't yet:** zero-shot accuracy on brand-new task types, where Jev is clearly ahead. Closing that gap is the current work (see *Roadmap*).

## Quickstart

```bash
git clone https://github.com/jerichosiahaya/naluri && cd naluri
uv sync
```

Model weights are not in the repo yet, so train a model first (see *Reproduce*). Then:

```bash
.venv/bin/python ask.py                                   # built-in examples, CPU
.venv/bin/python ask.py --state "Pesanan saya belum sampai!" --questions my_questions.json
.venv/bin/python ask.py --model runs/naluri-xlmr-base-m4-s0 --device cuda
```

`my_questions.json`:

```json
{
  "intent": {"type": "choice", "instructions": "What does the customer want?",
             "criteria": {"order_status": "ask about an order", "refund": "get money back", "other": "anything else"}},
  "urgent": {"type": "noul", "instructions": "Is this urgent?"},
  "mood":   {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}
}
```

From Python:

```python
import s1
tok, model = s1.load("runs/naluri-xlmr-base-m2", device="cpu")
s1.predict(tok, model, {"order": {"status": "delayed", "days_late": 6}},
           {"late": {"type": "noul", "instructions": "Is the order late?"}}, device="cpu")
# {'late': {'false': ..., 'true': ...}}
```

## How it works

```text
<s> {type} question: {instructions} </s> <mask> option 1 <mask> option 2 ... </s> {state} </s>
        │
        ▼  XLM-R base (bidirectional: every option sees the question, the other options and the state)
        ▼  small MLP at each <mask>  →  one score per option  →  softmax  →  probabilities
```

- **The options are input, not fixed output slots.** Naluri reads what each option *means*, so new questions and label sets need no new head.
- **Training:** cross-entropy (soft targets when a teacher provides probabilities), followed by one calibration temperature per question type.
- **Data:** public multilingual tasks (XNLI, MASSIVE, Amazon reviews, toxicity, language ID, emotion, RACE, SMS spam) and teacher-labeled synthetic business decisions, with options shuffled and labels and instructions paraphrased throughout.
- **Epoch selection:** each run keeps the epoch that scores best on *unseen* task types (the SIB-200 and XCOPA validation splits), not simply the last epoch.

| Milestone | Change | Result |
|---|---|---|
| M0 | 2 public tasks | the idea works |
| M1 | more data on the same tasks | better on trained tasks, less general |
| **M2** | 8 task types and the most-general epoch | the best general model so far |
| M2 + fine-tune | typed-decisions train split | ties Jev on typed-decisions |
| M3 | + 4.7k teacher-labeled synthetic decisions | +9.6 points zero-shot on typed-decisions |
| **M4** | from M2 + balanced sampling, 3× synthetic, topic data (2 seeds) | **the best model so far:** SIB-200 74.4%, above Laya |

## Reproduce

```bash
.venv/bin/python prep.py                                        # public data -> typed questions (eval sets frozen)
.venv/bin/python train.py --name naluri-xlmr-base-m2 --epochs 3 --max-len 384
.venv/bin/python evaluate.py runs/naluri-xlmr-base-m2           # 9 eval sets vs zero-shot Laya
.venv/bin/python bench.py runs/naluri-xlmr-base-m2              # typed-decisions + AG News (add --jev to call Jev live)
.venv/bin/python make_table.py                                  # tables for REPORT.md
```

| File | Purpose |
|---|---|
| `s1.py` | input format, model, `load()` / `predict()` |
| `prep.py`, `td_prep.py`, `synth.py` | public data, typed-decisions fine-tune data, synthetic data (generate → label → export) |
| `train.py` | training, generality-based epoch selection, temperature calibration |
| `evaluate.py`, `bench.py`, `make_table.py` | evaluation, benchmarks, report tables |
| `jev.py`, `llm.py` | API clients for Jev and a teacher LLM (LiteLLM proxy), with caching and hard cost caps |
| `ask.py` | try your own data on CPU or GPU |

The API clients read keys from a local `.env` file, which is not committed: `JEV_API_KEY`, `LITELLM_API_URL` and `LITELLM_API_KEY`.

## Roadmap

- Synthetic data with more agent-trace, workflow and policy decisions over structured states, scaled to 20–50k decisions (M4 showed that better use of the current 4.7k isn't enough for typed-decisions)
- An **mmBERT-base** backbone (8k context), and XLM-R large
- A `/v1/systemone`-compatible server for side-by-side battles with Jev, Laya and others

## License

The code is released under the [Apache License 2.0](LICENSE). The training datasets keep their own licenses: XNLI appears to be non-commercial, several are CC-BY or CC-BY-SA, and some are listed as "other" or "unknown". Check them before training or releasing a model for commercial use. Model weights are not distributed here yet.

## Acknowledgements

Inspired by [Laya](https://huggingface.co/convaiinnovations/laya) (whose input format and fine-tuning recipe Naluri follows), [Contrastive Language Models](https://github.com/Contrastive-LM/CLM), and TypeSafe's Jev. Built on [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base).
