"""Teacher LLM client over the LiteLLM proxy (OpenAI-compatible): JSON replies, disk cache, live cost cap.

Key/URL from .env (LITELLM_API_KEY, LITELLM_API_URL). Cost = usage x the proxy's configured prices below.
The cap is cumulative across runs (earlier spend is re-counted from the cache) and is checked before each paid
call, so the total stops (BudgetExceeded) before going over it.
"""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ENV = dict(l.strip().split("=", 1) for l in open(HERE / ".env") if "=" in l and not l.startswith("#"))
BASE = ENV["LITELLM_API_URL"].rstrip("/").removesuffix("/v1")
PRICES = {  # $ per 1M tokens (input, output), as configured on the proxy's /v1/model/info
    "azure_ai/deepseek-v4-flash": (0.19, 0.51),
    "azure_ai/deepseek-v4-pro": (1.74, 3.48),
}
CACHE = HERE / "data" / "synth" / "llm_cache.jsonl"
CAP_USD = 5.0  # cumulative across all runs; raise via set_cap() for a new, approved budget

spent = {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0, "errors": 0}
_lock = threading.Lock()
_cache = {}
CACHE.parent.mkdir(parents=True, exist_ok=True)
if CACHE.exists():
    for line in open(CACHE, encoding="utf-8"):
        r = json.loads(line)
        _cache[r["key"]] = r["content"]
        pin, pout = PRICES.get(r.get("model"), (0, 0))
        u = r.get("usage", {})
        spent["usd"] += (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
spent["usd_before_this_run"] = round(spent["usd"], 4)


class BudgetExceeded(RuntimeError):
    pass


def set_cap(usd):
    global CAP_USD
    CAP_USD = usd


def chat(prompt, model="azure_ai/deepseek-v4-flash", temperature=0.0, max_tokens=2000, seed_tag=""):
    """-> reply text. seed_tag makes otherwise identical prompts cache separately (e.g. generation batches)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
            "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    key = hashlib.sha256((json.dumps(body, sort_keys=True) + seed_tag).encode()).hexdigest()
    with _lock:
        if key in _cache:
            spent["cached"] += 1
            return _cache[key]
        if spent["usd"] >= CAP_USD:
            raise BudgetExceeded(f"cost cap ${CAP_USD} reached (spent ${spent['usd']:.3f})")
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {ENV['LITELLM_API_KEY']}",
                                          "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if attempt < 3 and (code is None or code in (408, 429, 500, 502, 503, 504)):
                time.sleep(2 ** attempt * 3)
                continue
            with _lock:
                spent["errors"] += 1
            raise RuntimeError(f"LLM call failed: {e}") from None
    content = out["choices"][0]["message"].get("content") or ""
    u = out.get("usage", {})
    pin, pout = PRICES[model]
    with _lock:
        spent["input_tokens"] += u.get("prompt_tokens", 0)
        spent["output_tokens"] += u.get("completion_tokens", 0)
        spent["usd"] += (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
        spent["calls"] += 1
        _cache[key] = content
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "content": content, "usage": u}, ensure_ascii=False) + "\n")
    return content


def parse_json(text):
    """Tolerant JSON parse (strips code fences); None if unparseable."""
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None
