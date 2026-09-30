"""Minimal TypeSafe Jev client for benchmarks: key from .env, on-disk cache, hard input-token cap.

The cache (results/jev_cache.jsonl) is keyed by the request body, so re-running a benchmark never pays twice.
The cap stops the run before the next paid call once the input-token total reaches it.
"""
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
URL = "https://api.typesafe.ai/v1/systemone"
CACHE = HERE / "results" / "jev_cache.jsonl"
TOKEN_CAP = 1_500_000  # input tokens per process; published price ~$0.05/1M, so ~$0.08 (well under a $3 balance)


class BudgetExceeded(RuntimeError):
    pass


_key = next((l.split("=", 1)[1].strip() for l in open(HERE / ".env") if l.startswith("JEV_API_KEY=")), None)
_cache = {}
if CACHE.exists():
    for line in open(CACHE, encoding="utf-8"):
        r = json.loads(line)
        _cache[r["key"]] = r
used = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0}


def call(state, questions, model="jev-latest"):
    """-> (answers dict, latency ms or None when served from cache)."""
    body = {"model": model, "state": state, "questions": questions}
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if k in _cache:
        used["cached"] += 1
        return _cache[k]["answers"], None
    if used["input_tokens"] >= TOKEN_CAP:
        raise BudgetExceeded(f"input-token cap {TOKEN_CAP:,} reached; not sending more requests")
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read())
            ms = (time.perf_counter() - t) * 1000
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"Jev HTTP {e.code}: {e.read()[:300]!r}") from None
        except (TimeoutError, urllib.error.URLError) as e:  # slow or dropped connection: retry
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"Jev request failed after 3 attempts: {e}") from None
    u = out.get("usage", {})
    used["input_tokens"] += u.get("input_tokens", 0)
    used["output_tokens"] += u.get("output_tokens", 0)
    used["calls"] += 1
    rec = {"key": k, "model": out.get("model"), "answers": out["answers"], "usage": u, "ms": round(ms, 1)}
    _cache[k] = rec
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out["answers"], ms


def predict(state, questions):
    """Same output shape as the other bench predictors: {qid: {label: prob}}."""
    answers, _ = call(state, questions)
    out = {}
    for qid, q in questions.items():
        a = answers[qid]
        if q["type"] == "noul":
            out[qid] = {"false": 1 - a["noul"], "true": a["noul"]}
        elif q["type"] == "score":
            out[qid] = {int(k): v for k, v in a["probabilities"].items()}
        else:
            out[qid] = a["probabilities"]
    return out
