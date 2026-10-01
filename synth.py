"""M3 synthetic typed decisions: an LLM writes varied cases, a teacher LLM labels them with soft probabilities.

  python synth.py quality --cases 20        teacher vs gold on typed-decisions *train* cases (test untouched)
  python synth.py gen --cases 1500          write data/synth/cases.jsonl (situations + questions)
  python synth.py label                     teacher probabilities -> data/synth/labels.jsonl
  python synth.py export                    -> data/synth/records.jsonl (training records with soft targets)

Labels: the teacher states a probability per option in JSON, twice with the options in different orders; the two
answers are averaged (cancels position bias). The proxy does not return logprobs, so probabilities are verbalized.
Honesty rule: typed-decisions' own four workflows are excluded from generation, so its test stays zero-shot.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import llm

HERE = Path(__file__).parent
OUT = HERE / "data" / "synth"  # overridden by --out-dir
TEACHER = "azure_ai/deepseek-v4-flash"
WORKERS = 16

DOMAINS = [  # deliberately excludes agent-trace observability, customer-support tickets, invoices, security incidents
    "e-commerce order fulfilment", "warehouse inventory", "last-mile delivery", "freight logistics", "HR recruiting",
    "employee leave requests", "payroll exceptions", "clinic appointment triage", "health insurance claims",
    "pharmacy stock", "consumer loan applications", "credit card fraud review", "insurance underwriting",
    "real estate listings", "property maintenance requests", "hotel reservations", "airline disruptions",
    "restaurant food delivery", "school admissions", "online course moderation", "student academic integrity",
    "marketing campaign approval", "social media content moderation", "app store review", "CI/CD pipeline results",
    "cloud cost anomalies", "database migration planning", "product feature requests", "SaaS subscription churn",
    "manufacturing quality control", "energy grid alerts", "agricultural crop reports", "legal contract review",
    "regulatory compliance checks", "procurement vendor selection", "event ticketing", "ride-hailing trips",
    "news article classification", "job candidate screening", "charity donation processing", "smart home automation",
    "game community reports", "research paper review", "travel visa applications", "banking KYC review",
]
LANGS = [("English", 30), ("Indonesian", 10), ("Spanish", 6), ("French", 5), ("German", 5), ("Portuguese", 5),
         ("Chinese", 5), ("Japanese", 4), ("Arabic", 4), ("Hindi", 4), ("Russian", 4), ("Vietnamese", 3),
         ("Thai", 3), ("Turkish", 3), ("Korean", 3), ("Swahili", 2), ("Italian", 2)]
# M5 "ops" profile: operational decisions over structured run/event logs. Same question archetypes as
# typed-decisions (next action, outcome grade, risk level, needs review) but none of its four workflows
# (agent traces, customer support, invoices, security incidents).
OPS_DOMAINS = [
    "CI/CD pipeline runs", "nightly ETL / data pipeline jobs", "robotic process automation bot runs",
    "IoT sensor telemetry alerts", "warehouse robot missions", "delivery drone flights", "database backup jobs",
    "cloud autoscaling events", "Kubernetes deployment rollouts", "ML model training jobs", "batch payroll runs",
    "expense report approvals", "purchase order approval workflows", "employee access requests",
    "loan underwriting pipelines", "insurance claim processing workflows", "fraud rule engine hits",
    "content moderation queue decisions", "SLA breach monitoring", "manufacturing line quality checks",
    "energy grid control actions", "trading bot order executions", "email marketing campaign sends",
    "medical prior-authorization workflows", "laboratory sample processing", "logistics route optimization runs",
    "customer data export requests (privacy)", "feature flag rollouts", "vendor onboarding checks",
    "scheduled maintenance of fleet vehicles",
]
OPS_ARCHETYPES = """Across the cases, cover these operational question archetypes (2-4 per case, mixed types):
  * next action for the system/operator (choice, e.g. continue / monitor / pause / roll back / escalate / stop)
  * outcome grade of the run (choice, e.g. success / partial / failure / harmful or out-of-policy)
  * risk or severity level (score, 3-5 ordered levels with concrete descriptions)
  * whether a human must review it (noul), whether a policy/constraint was violated (noul)
  * who should own it / which queue it goes to (choice)
Make the correct answer depend on the numbers and events in the state (error counts, retries, thresholds,
irreversible steps, constraint violations, durations, amounts), and include some clean successful runs."""
OPS_FORMATS = ["a nested JSON run record with config, constraints, a list of step events (with status, retries,"
               " durations) and a summary block",
               "a nested JSON object with metrics, thresholds, recent alerts and actions already taken",
               "a JSON workflow record with the request, policy rules, approvals so far and anomalies",
               "a JSON log excerpt (list of timestamped events) plus a short operator note in free text"]
# M6 "pairs" profile: contrastive minimal pairs. One set of questions, several versions of the state that
# differ in a few decisive facts so the right answers change. The only way to score well is to read the state.
PAIRS_PROMPT = """Create {n} decision groups for a fast decision model in the domain: {domain}.
Write human-readable text in {lang}; JSON keys and option keys in English snake_case.

Each group has:
- "questions": 2 to 3 questions shared by every version, mixing types:
    choice: {{"type": "choice", "instructions": ..., "criteria": {{"<option_key>": "<description>", ...}}}} with 3-5 options
    noul:   {{"type": "noul", "instructions": ..., "criteria": {{"true": ..., "false": ...}}}}
    score:  {{"type": "score", "instructions": ..., "criteria": ["<level 0>", "<level 1>", ...]}} with 3-5 levels
  Use operational or business judgment questions: next action, outcome grade, risk level, needs review,
  policy violation, routing / owner queue.
- "states": 3 or 4 versions of {fmt}. All versions describe the same kind of situation and look alike, but
  change a FEW decisive details (numbers vs thresholds, error or retry counts, amounts vs limits, a policy
  violation, an irreversible step, a missing approval, customer tier, time left) so that the correct answer to
  at least one choice question is DIFFERENT between versions. Across the versions, make different options the
  correct one; include at least one clean, low-risk version.
Do NOT include the answers.

Return JSON: {{"groups": [{{"questions": {{...}}, "states": [<state>, <state>, <state>]}}, ...]}}"""
# Targeted pairs (M7): decision archetypes the model still collapses on. Domains avoid typed-decisions' own
# workflows (invoices, customer support), so the benchmark stays zero-shot for its domains.
FOCUS = {
    "dispositions": ("approval dispositions: approve / hold / manual review / reject (4 options, pick the fitting "
                     "subset); the right one must depend on amounts vs limits, missing documents, duplicates, "
                     "mismatches, policy flags and approver status",
                     ["expense report approvals", "purchase order approvals", "loan applications", "insurance claims",
                      "vendor onboarding", "grant applications", "refund-free warranty claims for field equipment",
                      "travel booking approvals", "overtime requests", "procurement contract renewals"]),
    "responses": ("request handling actions: answer directly / ask for more information / escalate to a human "
                  "specialist / take an automated action (e.g. reset, reschedule, reissue); the right one must depend "
                  "on what is already known, missing details, risk, sentiment and permissions",
                  ["HR employee requests", "IT helpdesk requests from employees", "tenant maintenance requests",
                   "patient portal messages to a clinic", "student requests to a university office",
                   "B2B partner integration requests", "government permit inquiries", "internal facilities requests",
                   "bank branch internal requests", "event attendee requests"]),
}
FORMATS = ["a nested JSON object with realistic fields, ids, numbers, timestamps and statuses",
           "a nested JSON object with a list of recent events or line items",
           "a plain-text message or email written by a person",
           "a short chat transcript as a JSON list of {\"role\", \"text\"} turns",
           "a JSON object mixing structured fields with a free-text note"]


def gen_prompt(domain, lang, fmt, n, extra=""):
    return f"""Create {n} realistic, varied decision cases for a fast decision model in the domain: {domain}.
Write all human-readable text (state text, instructions, option descriptions) in {lang}. JSON keys and option
keys stay in English snake_case.

Each case has:
- "state": {fmt}. Make it specific and realistic (names, amounts, dates, statuses), and make the right answers
  depend on the details. Vary difficulty: some clear-cut, some genuinely ambiguous.
- "questions": 2 to 4 questions keyed by short snake_case ids. Mix these types:
  * "noul": a yes/no statement. {{"type": "noul", "instructions": "<statement or yes/no question>",
    "criteria": {{"true": "<what yes means>", "false": "<what no means>"}}}}
  * "choice": pick one of 2-6 options. {{"type": "choice", "instructions": "<question>",
    "criteria": {{"<option_key>": "<description>", ...}}}}
  * "score": an ordered scale. {{"type": "score", "instructions": "<question>",
    "criteria": ["<level 0 description>", "<level 1 description>", ...]}} with 3-5 levels.
  Questions should need judgment (risk, priority, routing, next action, policy fit, urgency, sentiment), not
  simple lookups. Do NOT include the answers.
{extra}

Return JSON: {{"cases": [{{"state": ..., "questions": {{...}}}}, ...]}}"""


def options(q):
    """Option keys in canonical order: noul [false, true], score ["0".."n-1"], choice as given."""
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def describe(q, order):
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        desc = {"false": crit.get("false") or "no", "true": crit.get("true") or "yes"}
    elif q["type"] == "score":
        desc = {str(i): c for i, c in enumerate(crit)}
    else:
        desc = crit
    return "\n".join(f"  - {k}: {desc[k]}" for k in order)


def label_prompt(state, questions, rng):
    blocks = []
    for qid, q in questions.items():
        order = options(q)
        rng.shuffle(order)
        kind = {"noul": "yes/no (options false/true)", "choice": "pick one", "score": "ordered scale (option = level)"}[q["type"]]
        blocks.append(f"[{qid}] ({kind}) {q['instructions']}\n{describe(q, order)}")
    state_txt = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    return f"""You are an expert, careful decision-maker. Read the STATE and answer every question.
For each question give a probability for EVERY option (they must sum to 1), meaning how likely that option is the
right answer given the state. Be calibrated: use intermediate probabilities when the state is ambiguous, near 0/1
only when clear-cut. Use the exact option keys shown.

STATE:
{state_txt}

QUESTIONS:
{chr(10).join(blocks)}

Return JSON: {{"<question_id>": {{"<option_key>": <probability>, ...}}, ...}}"""


def valid_question(q):
    t, c = q.get("type"), q.get("criteria")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        return False
    if t == "noul":
        return c is None or (isinstance(c, dict) and set(map(str.lower, c)) <= {"true", "false"})
    if t == "choice":
        return isinstance(c, dict) and 2 <= len(c) <= 8 and all(isinstance(v, str) for v in c.values())
    if t == "score":
        return isinstance(c, list) and 3 <= len(c) <= 6 and all(isinstance(v, str) for v in c)
    return False


def label_case(state, questions, model=TEACHER, rotations=2, tag=""):
    """-> {qid: {option: prob}} averaged over `rotations` differently-ordered prompts; missing qids dropped."""
    sums, counts = {}, {}
    for r in range(rotations):
        reply = llm.parse_json(llm.chat(label_prompt(state, questions, random.Random(f"{tag}{r}")), model=model,
                                        max_tokens=150 + 120 * len(questions)))
        if not isinstance(reply, dict):
            continue
        for qid, q in questions.items():
            p = reply.get(qid)
            keys = options(q)
            if not isinstance(p, dict):
                continue
            vals = [max(0.0, float(p.get(k, 0) or 0)) if isinstance(p.get(k, 0), (int, float)) else 0.0 for k in keys]
            if sum(vals) <= 0:
                continue
            vals = [v / sum(vals) for v in vals]
            sums[qid] = [a + b for a, b in zip(sums.get(qid, [0.0] * len(keys)), vals)]
            counts[qid] = counts.get(qid, 0) + 1
    return {qid: dict(zip(options(questions[qid]), [v / counts[qid] for v in s])) for qid, s in sums.items()}


def run_parallel(fn, jobs, desc):
    results = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            try:
                results.append(f.result())
            except llm.BudgetExceeded as e:
                print("STOP:", e)
                ex.shutdown(cancel_futures=True)
                break
            except Exception as e:  # one bad reply should not kill the batch
                results.append(None)
                print("skip:", str(e)[:120])
    return results


# --- commands ---------------------------------------------------------------------------------------------------
def cmd_quality(args):
    from datasets import load_dataset
    rows = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
    random.Random(1).shuffle(rows)
    rows = rows[:args.cases]
    for model in (TEACHER, "azure_ai/deepseek-v4-pro"):
        def job(r):
            qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
            pred = label_case(json.loads(r["state"]), qs, model=model, tag=r["id"])
            hits = [max(p, key=p.get) == str(gold[q]["label"]).lower() for q, p in pred.items()]
            return hits, len(qs) - len(pred)
        res = [x for x in run_parallel(job, [(r,) for r in rows], model) if x]
        hits = [h for hs, _ in res for h in hs]
        print(f"{model}: agrees with gold on {sum(hits)}/{len(hits)} = {sum(hits) / max(1, len(hits)):.3f} "
              f"(missing answers: {sum(m for _, m in res)})  | spent so far ${llm.spent['usd']:.3f}")


def cmd_gen_pairs(args):
    rng = random.Random(args.seed)
    langs = [l for l, w in LANGS for _ in range(w)]
    per_call = 2
    domains = OPS_DOMAINS + [d for d in DOMAINS if d not in ("news article classification", "research paper review")]
    focus_text = ""
    if args.focus:
        focus_text, domains = FOCUS[args.focus]
        focus_text = (f"\nEvery group must include one choice question about {focus_text}. Across the versions, "
                      "each option should be the correct one at least once.")
    jobs = [(rng.choice(domains), rng.choice(langs), rng.choice(OPS_FORMATS + FORMATS[:2]), i)
            for i in range(args.cases // per_call)]

    def job(domain, lang, fmt, i):
        reply = llm.parse_json(llm.chat(PAIRS_PROMPT.format(n=per_call, domain=domain, lang=lang, fmt=fmt) + focus_text,
                                        temperature=1.0, max_tokens=6000, seed_tag=f"pairs{args.seed}gen{i}"))
        cases = []
        for g_i, g in enumerate((reply or {}).get("groups", [])):
            qs = {k: v for k, v in (g.get("questions") or {}).items() if isinstance(v, dict) and valid_question(v)}
            states = [x for x in (g.get("states") or []) if x]
            if not qs or len(states) < 2:
                continue
            for q in qs.values():
                if q["type"] == "noul" and q.get("criteria"):
                    q["criteria"] = {k.lower(): v for k, v in q["criteria"].items()}
            for v_i, st in enumerate(states):
                cases.append({"domain": domain, "lang": lang, "group": f"p{args.seed}_{i}_{g_i}", "variant": v_i,
                              "state": st, "questions": qs})
        return cases

    cases = [c for batch in run_parallel(job, jobs, "generate pairs") if batch for c in batch]
    with open(OUT / "cases.jsonl", "w", encoding="utf-8") as f:
        for i, c in enumerate(cases):
            f.write(json.dumps({"id": f"v{i}"} | c, ensure_ascii=False) + "\n")
    groups = len({c["group"] for c in cases})
    print(f"{groups} groups, {len(cases)} state versions, {sum(len(c['questions']) for c in cases)} questions "
          f"| spent ${llm.spent['usd']:.3f}")


def cmd_gen(args):
    if args.profile == "pairs":
        return cmd_gen_pairs(args)
    rng = random.Random(args.seed)
    langs = [l for l, w in LANGS for _ in range(w)]
    per_call = 4
    ops = args.profile == "ops"
    domains, formats = (OPS_DOMAINS, OPS_FORMATS) if ops else (DOMAINS, FORMATS)
    jobs = [(rng.choice(domains), rng.choice(langs), rng.choice(formats), i) for i in range(args.cases // per_call)]

    def job(domain, lang, fmt, i):
        prompt = gen_prompt(domain, lang, fmt, per_call, OPS_ARCHETYPES if ops else "")
        reply = llm.parse_json(llm.chat(prompt, temperature=1.0, max_tokens=5000,
                                        seed_tag=f"{args.profile}{args.seed}gen{i}"))
        cases = []
        for c in (reply or {}).get("cases", []):
            qs = {k: v for k, v in (c.get("questions") or {}).items() if isinstance(v, dict) and valid_question(v)}
            if c.get("state") and qs:
                for q in qs.values():
                    if q["type"] == "noul" and q.get("criteria"):
                        q["criteria"] = {k.lower(): v for k, v in q["criteria"].items()}
                cases.append({"domain": domain, "lang": lang, "state": c["state"], "questions": qs})
        return cases

    cases = [c for batch in run_parallel(job, jobs, "generate") if batch for c in batch]
    with open(OUT / "cases.jsonl", "w", encoding="utf-8") as f:
        for i, c in enumerate(cases):
            f.write(json.dumps({"id": f"s{i}"} | c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases, {sum(len(c['questions']) for c in cases)} questions | spent ${llm.spent['usd']:.3f}")


def cmd_label(args):
    cases = [json.loads(l) for l in open(OUT / "cases.jsonl", encoding="utf-8")]

    def job(c):  # one pass: option order is shuffled per case, so position bias averages out over the dataset
        return {"id": c["id"], "teacher": args.teacher,
                "labels": label_case(c["state"], c["questions"], model=args.teacher, rotations=args.rotations, tag=c["id"])}

    labels = [x for x in run_parallel(job, [(c,) for c in cases], "label") if x]
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as f:
        for x in labels:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"labeled {len(labels)}/{len(cases)} cases | spent ${llm.spent['usd']:.3f}")


def cmd_export(args):
    cases = {c["id"]: c for c in map(json.loads, open(OUT / "cases.jsonl", encoding="utf-8"))}
    # pairs: which questions actually change their teacher answer across a group's versions
    answers = {}
    for x in map(json.loads, open(OUT / "labels.jsonl", encoding="utf-8")):
        c = cases[x["id"]]
        for qid, probs in x["labels"].items():
            answers.setdefault((c.get("group"), qid), set()).add(max(probs, key=probs.get))
    flips = {}
    for (g, qid), a in answers.items():
        if g and len(a) > 1:
            flips.setdefault(g, set()).add(qid)
    if any(c.get("group") for c in cases.values()):
        print(f"contrastive: {sum(len(v) for v in flips.values())} group-questions change answer across versions "
              f"({len(flips)} of {len({c.get('group') for c in cases.values()})} groups)")
    n = 0
    with open(OUT / "records.jsonl", "w", encoding="utf-8") as f:
        for x in map(json.loads, open(OUT / "labels.jsonl", encoding="utf-8")):
            c = cases[x["id"]]
            for qid, probs in x["labels"].items():
                q = c["questions"][qid]
                if c.get("group") and args.contrastive_only and qid not in flips.get(c["group"], set()):
                    continue
                target = [probs[k] for k in options(q)]
                f.write(json.dumps({"task": f"{args.task_prefix}:" + c["domain"], "lang": c["lang"], "state": c["state"], "q": q,
                                    "gold": max(range(len(target)), key=target.__getitem__), "target": target},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} training records -> {OUT / 'records.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quality", "gen", "label", "export"])
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default="azure_ai/deepseek-v4-pro", help="labeling model (label command)")
    ap.add_argument("--rotations", type=int, default=1, help="differently-ordered labeling passes per case")
    ap.add_argument("--profile", default="general", choices=["general", "ops", "pairs"], help="generation profile (gen)")
    ap.add_argument("--focus", choices=["dispositions", "responses"], help="pairs: target one decision archetype")
    ap.add_argument("--out-dir", default=str(OUT), help="where cases/labels/records live")
    ap.add_argument("--task-prefix", default="synth", help="task name prefix in exported records")
    ap.add_argument("--contrastive-only", action="store_true",
                    help="export: keep only pair questions whose teacher answer changes across versions")
    ap.add_argument("--cap-usd", type=float, default=llm.CAP_USD, help="cumulative spend cap across all runs")
    a = ap.parse_args()
    OUT = Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    llm.set_cap(a.cap_usd)
    globals()[f"cmd_{a.cmd}"](a)
    print("llm usage:", {k: round(v, 4) if isinstance(v, float) else v for k, v in llm.spent.items()})
