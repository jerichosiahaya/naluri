"""Serve Naluri at POST /v1/systemone (the System One request and response shape).

Request:  {"model": ..., "state": ..., "questions": {qid: {"type", "instructions", "criteria"?}}}
Response: {"model": ..., "answers": {qid: {"noul": p} | {"choice": top, "probabilities": {...}}
                                          | {"score": top, "probabilities": {"0": ..., ...}}}}

Usage: python serve.py [--model jerichosiahaya/Naluri-308M-encoder | runs/<dir>] [--port 8000] [--device cuda]
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import s1

TYPES = {"noul", "choice", "score"}


def answer(q, p):
    """One question's predict() output -> its System One answer."""
    if q["type"] == "noul":
        return {"noul": p["true"]}
    probs = {str(k): v for k, v in p.items()}
    top = max(p, key=p.get)
    return {q["type"]: top, "probabilities": probs}


def check(body):
    """-> error message for a request predict() can't take, else None."""
    qs = body.get("questions") if isinstance(body, dict) else None
    if not isinstance(qs, dict) or not qs:
        return "questions must be a non-empty object"
    for qid, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in TYPES or not isinstance(q.get("instructions"), str):
            return f"question {qid!r} needs type (noul, choice or score) and instructions"
        crit = q.get("criteria")
        if q["type"] == "choice" and not (isinstance(crit, dict) and len(crit) >= 2):
            return f"choice question {qid!r} needs criteria with 2+ options"
        if q["type"] == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            return f"score question {qid!r} needs criteria with 2+ levels"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jerichosiahaya/Naluri-308M-encoder", help="HF repo id or a local run/export dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tok, model = s1.load(a.model, a.device) if Path(a.model).is_dir() else s1.from_pretrained(a.model, a.device)
    name = Path(a.model).name
    lock = threading.Lock()  # ponytail: one forward pass at a time; batch across requests if throughput matters

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"    # keep-alive: no new TCP connection per request
        disable_nagle_algorithm = True   # headers and body go out in separate writes; Nagle + delayed ACK adds ~40 ms

        def reply(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self.reply(404, {"error": "POST /v1/systemone"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                return self.reply(400, {"error": "body must be JSON"})
            err = check(body)
            if err:
                return self.reply(400, {"error": err})
            qs = body["questions"]
            with lock:
                out = s1.predict(tok, model, body.get("state", ""), qs, device=a.device)
            self.reply(200, {"model": name, "answers": {qid: answer(qs[qid], p) for qid, p in out.items()}})

        def log_message(self, *args):  # quiet: the benchmark sends thousands of requests
            pass

    print(f"Naluri ({name}) on http://{a.host}:{a.port}/v1/systemone", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    # self-check of the answer mapping and request validation (no model needed): python serve.py --check
    import sys
    if sys.argv[1:] == ["--check"]:
        assert answer({"type": "noul"}, {"false": 0.3, "true": 0.7}) == {"noul": 0.7}
        assert answer({"type": "choice"}, {"a": 0.2, "b": 0.8}) == {"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}
        assert answer({"type": "score"}, {0: 0.1, 1: 0.6, 2: 0.3})["probabilities"] == {"0": 0.1, "1": 0.6, "2": 0.3}
        assert check({"questions": {"q": {"type": "noul", "instructions": "x"}}}) is None
        assert check({"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": ""}}}})
        assert check({"questions": {}}) and check([]) and check({"questions": {"q": {"type": "maybe", "instructions": "x"}}})
        print("ok")
    else:
        main()
