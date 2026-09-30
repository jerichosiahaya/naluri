"""Train XLM-R System One on data/train.jsonl (from prep.py) -> runs/<name>/

Cross-entropy over each question's options, bf16, gradient checkpointing, length-bucketed batches.
If data/gen_dev.jsonl exists (unseen task types, validation splits), it is scored after every epoch
and the epoch with the best score is kept (the most general one, not the last one).
After training, one temperature per question type is fitted on data/val.jsonl.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from s1 import BASE, QTYPE, SystemOne, build_sequence, collate

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="e.g. naluri-xlmr-base-m2")
ap.add_argument("--base", default=BASE)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--tokens-per-batch", type=int, default=6144)
ap.add_argument("--lr-encoder", type=float, default=2e-5)
ap.add_argument("--lr-head", type=float, default=5e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data", help="dir with train.jsonl / val.jsonl (and optional gen_dev.jsonl)")
ap.add_argument("--init", help="start from a trained run dir instead of the base model")
ap.add_argument("--task-alpha", type=float, help="resample each epoch by task, p(task) ~ size^alpha (e.g. 0.5)")
ap.add_argument("--boost", nargs="*", default=[], help="extra task weight on top of --task-alpha, e.g. synth=3")
args = ap.parse_args()
OUT = HERE / "runs" / args.name
random.seed(args.seed)
torch.manual_seed(args.seed)

tok = AutoTokenizer.from_pretrained(args.base)


def encode(path):
    items = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        ids, markers = build_sequence(tok, r["state"], r["q"], args.max_len)
        items.append({"ids": ids, "markers": markers, "gold": r["gold"], "qtype": QTYPE[r["q"]["type"]], "task": r["task"],
                      "target": r.get("target")})  # optional soft target (teacher distribution)
    return items


def batches(items, shuffle):
    """Group similar lengths so padding stays small; cap each batch by padded token count."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out, cur, longest = [], [], 0
    for i in order:
        longest = max(longest, len(items[i]["ids"]))
        if cur and longest * (len(cur) + 1) > args.tokens_per_batch:
            out.append(cur)
            cur, longest = [], len(items[i]["ids"])
        cur.append(i)
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return [[items[i] for i in b] for b in out]


def epoch_items(items):
    """With --task-alpha: draw len(items) examples, task t with probability ~ n_t^alpha, so small tasks
    (synthetic, RACE, spam) are not drowned out by big ones. Synthetic domains count as one task."""
    if not args.task_alpha:
        return items
    groups = {}
    for it in items:
        groups.setdefault(it["task"].split(":")[0], []).append(it)
    names = sorted(groups)
    boost = {k: float(v) for k, v in (b.split("=") for b in args.boost)}
    weights = [len(groups[n]) ** args.task_alpha * boost.get(n, 1.0) for n in names]
    picks = random.choices(names, weights=weights, k=len(items))
    return [random.choice(groups[n]) for n in picks]


def run(model, chunk, device):
    b = {k: v.to(device) for k, v in collate(chunk, tok.pad_token_id).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**b)


@torch.no_grad()
def val_logits(model, items, device):
    model.eval()
    out = []
    for chunk in batches(items, shuffle=False):
        logits = run(model, chunk, device)
        out += [(it, logits[j, :len(it["markers"])].cpu()) for j, it in enumerate(chunk)]
    model.train()
    return out


def accuracy(preds):
    return sum(int(z.argmax()) == it["gold"] for it, z in preds) / len(preds)


def macro(preds):
    """Mean of per-task accuracies (so a big dev task cannot dominate epoch selection)."""
    tasks = sorted({it["task"].split(":")[0] for it, _ in preds})
    return sum(accuracy([p for p in preds if p[0]["task"].split(":")[0] == t]) for t in tasks) / len(tasks)


def report(tag, preds):
    tasks = sorted({it["task"].split(":")[0] for it, _ in preds})
    per = " | ".join(f"{t} {accuracy([p for p in preds if p[0]['task'].split(':')[0] == t]):.3f}" for t in tasks)
    print(f"{tag}: val acc {accuracy(preds):.3f}, macro {macro(preds):.3f}  ({per})")


t0 = time.time()
DATA = HERE / args.data
train_items, val_items = encode(DATA / "train.jsonl"), encode(DATA / "val.jsonl")
gen_path = DATA / "gen_dev.jsonl"
gen_items = encode(gen_path) if gen_path.exists() else []
print(f"encoded {len(train_items)} train / {len(val_items)} val / {len(gen_items)} gen-dev in {time.time() - t0:.0f}s")

device = "cuda"
model = SystemOne(args.init or args.base)
if args.init:  # encoder weights came from the run dir; restore its scorer too
    model.scorer.load_state_dict(torch.load(Path(args.init) / "scorer.pt", map_location="cpu"))
    tok = AutoTokenizer.from_pretrained(args.init)
model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(device).train()
opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr_encoder},
                         {"params": model.scorer.parameters(), "lr": args.lr_head}], weight_decay=0.01)
steps = len(batches(train_items, shuffle=False)) * args.epochs
warmup = max(1, int(0.06 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

report("before training", val_logits(model, val_items, device))
if gen_items:
    report("before training (gen-dev)", val_logits(model, gen_items, device))
best_score, best_epoch = -1.0, 0
BEST = OUT / "best_epoch.pt"  # best weights go to disk, not RAM
step = 0
for epoch in range(args.epochs):
    bar = tqdm(batches(epoch_items(train_items), shuffle=True), desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch")
    total = 0.0
    for i, chunk in enumerate(bar, 1):
        logits = run(model, chunk, device)
        # per item: the teacher's soft distribution when there is one, else one-hot on the gold option
        target = torch.zeros_like(logits)
        for j, it in enumerate(chunk):
            if it["target"]:
                target[j, :len(it["target"])] = torch.tensor(it["target"], device=device)
            else:
                target[j, it["gold"]] = 1.0
        loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        total += loss.item()
        step += 1
        bar.set_postfix(loss=f"{total / i:.4f}")
    report(f"epoch {epoch + 1}", val_logits(model, val_items, device))
    if gen_items:
        gen = val_logits(model, gen_items, device)
        report(f"epoch {epoch + 1} (gen-dev)", gen)
        score = macro(gen)
        if score > best_score:  # keep the most general epoch
            best_score, best_epoch = score, epoch + 1
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST)

if BEST.exists():
    model.load_state_dict(torch.load(BEST, map_location=device))
    BEST.unlink()
    print(f"keeping epoch {best_epoch} (gen-dev macro acc {best_score:.3f})")


# --- temperature per question type, fitted on val -------------------------------------------------
def fit_temp(pairs):
    if len(pairs) < 20:
        return 1.0
    k = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), k), -1e4)
    for i, (z, _) in enumerate(pairs):
        Z[i, :len(z)] = z
    y = torch.tensor([g for _, g in pairs])
    log_t = torch.zeros(1, requires_grad=True)
    lbfgs = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        lbfgs.zero_grad()
        loss = F.cross_entropy(Z / log_t.exp(), y)
        loss.backward()
        return loss

    lbfgs.step(closure)
    return float(log_t.detach().exp().clamp(0.1, 10.0))


preds = val_logits(model, val_items, device)
temps = [fit_temp([(z, it["gold"]) for it, z in preds if it["qtype"] == t]) for t in range(3)]
print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
model.temperature.copy_(torch.tensor(temps))

OUT.mkdir(parents=True, exist_ok=True)
model.encoder.save_pretrained(OUT)
tok.save_pretrained(OUT)
torch.save(model.scorer.state_dict(), OUT / "scorer.pt")
torch.save(model.temperature.cpu(), OUT / "temperature.pt")
json.dump(vars(args) | {"train_items": len(train_items), "minutes": round((time.time() - t0) / 60, 1),
                        "best_epoch": best_epoch, "gen_dev_acc": round(best_score, 3)},
          open(OUT / "train_args.json", "w"), indent=2)
print(f"saved -> {OUT}")
