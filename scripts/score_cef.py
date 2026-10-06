#!/usr/bin/env python
"""Cross-Examination Framework (CEF) over a metric-input table. Reference-free.

    python scripts/score_cef.py results/metric_inputs_X/perturb.jsonl -o results/cef_X/perturb.jsonl

Implements Raha et al., "Cross-Examination Framework: A Task-Agnostic Diagnostic for
Information Fidelity in Text-to-Text Generation" (M42 Health, arXiv:2601.19350). The source
and the candidate are treated as independent knowledge bases:

  1. An LLM writes N = 10 closed-ended YES-only questions from the SOURCE and N from the
     CANDIDATE (always in English, temperature 0, JSON). Prompts are the paper's, Appendix A.5.
  2. Source questions are put to the candidate, candidate questions to the source; each
     answer is YES / NO / IDK, "grounded on the text content only".
  3. Scores (all 0-100, HIGHER is better):
       Coverage    = 100 - %(source questions the candidate answers IDK)   omissions
       Conformity  = 100 - %(source questions the candidate answers NO)    contradictions
       Consistency = 100 - %(candidate questions the source answers IDK)   hallucinations

Differences from the paper, stated plainly:
  * JUDGE. The paper selected DeepSeek-V3 after a stability analysis (ADR / ADS) of five
    models; Qwen3-235B was the LEAST stable of the five there. This run uses whatever model
    the vLLM endpoint serves (Qwen3.8-27B), which that analysis does not cover. Judge
    stability should be checked (see --judge-check) rather than assumed.
  * RESOLUTION. N = 10 questions per text is the paper's choice (its ablation: N=10 is as
    stable as N=20). A PARROT report holds many more than 10 checkable facts, so one wrong
    fact is sampled only if a question happens to touch it. Treat CEF as a coverage-style
    diagnostic, not a per-fact detector.
  * Source questions are generated once per distinct source and shared by every system.

Caches (resumable): questions and answers are appended to <output>.cache.jsonl.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

QGEN = ("Please formulate {n} critical, concise and closed-ended questions (in a YES/NO format) in English that thoroughly "
        "scrutinize the document. The questions generated should ALWAYS result in a 'YES' based on the given text. Questions should be "
        "about the content of the document and not include any qualifier of the clarity, justification or definition.\n**Note**\n"
        "The questions have to be STRICTLY closed-ended and should not be subjective or open to human interpretation.\n"
        "You should return in a JSON format. The JSON should be a list of dictionaries where each dictionary will have two keys:\n"
        "- 'question': specifying the question\n- 'answer': either YES or NO.\nThe given text should be able to answer 'YES' for each "
        "generated question.\nDocument:\n{text}\nJSON:")
QANS = ("Answer the following question with a YES, NO or IDK, grounded on the text content only. Do not use any external knowledge. "
        "If you cannot answer the question based on the provided text, please respond with 'IDK'.\n**Note**\n"
        "You should respond either YES, NO or IDK. You should respond with a single word and only in English.\nText:\n{text}\nQuestion:\n{question}\nAnswer:")
SYSTEM = "You are a helpful assistant."


def h(*parts) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


class Judge:
    def __init__(self, env_file: Path, workers: int):
        env = dict(l.strip().split("=", 1) for l in env_file.open() if "=" in l and not l.startswith("#"))
        self.url, self.key, self.model = env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], env["MODEL_NAME"]
        self.session = requests.Session()
        self.session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))

    def chat(self, user: str, max_tokens: int) -> str:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(5):
            try:
                r = self.session.post(f"{self.url}/chat/completions", headers={"Authorization": f"Bearer {self.key}"},
                                      json=body, timeout=180)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt); continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(2 ** attempt)
        raise RuntimeError("judge endpoint unreachable after 5 attempts")


def parse_questions(raw: str, n: int) -> list[str]:
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    a, b = s.find("["), s.rfind("]")
    if a >= 0 and b >= 0:
        try:
            data = json.loads(s[a:b + 1])
            qs = [d["question"].strip() for d in data if isinstance(d, dict) and isinstance(d.get("question"), str)]
            if qs:
                return qs[:n]
        except json.JSONDecodeError:
            pass
    # The judge occasionally emits slightly malformed JSON (e.g. an item missing its opening
    # brace; seen in ~1-4% of generations). The questions are still intact, so recover them
    # directly rather than discard all ten and silently score that text on nothing.
    PARSE_FALLBACKS[0] += 1
    return [q.strip() for q in re.findall(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', s)][:n]


PARSE_FALLBACKS = [0]


def parse_answer(raw: str) -> str:
    m = re.search(r"[A-Za-z]+", raw)
    w = m.group(0).upper() if m else ""
    return w if w in ("YES", "NO", "IDK") else "INVALID"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--env", type=Path, default=Path("../vllm_serve/endpoint.env"))
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output.with_suffix(".cache.jsonl")

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    judge = Judge(args.env, args.workers)
    print(f"preflight: judge {judge.model} at {judge.url} | {len(rows)} rows | N={args.n_questions}", flush=True)

    qcache: dict[str, list[str]] = {}
    acache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            x = json.loads(line)
            (qcache if x["k"] == "q" else acache)[x["key"]] = x["val"]
        print(f"preflight: resumed cache, {len(qcache)} question sets, {len(acache)} answers", flush=True)
    lock = threading.Lock()
    cache_f = cache_path.open("a", encoding="utf-8")

    def remember(kind, key, val):
        with lock:
            (qcache if kind == "q" else acache)[key] = val
            cache_f.write(json.dumps({"k": kind, "key": key, "val": val}, ensure_ascii=False) + "\n"); cache_f.flush()

    def run_pool(label, items, fn):
        t0, done = time.time(), 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(fn, *it) for it in items]
            for f in cf.as_completed(futs):
                f.result(); done += 1
                if done % 2000 == 0 or done == len(futs):
                    el = time.time() - t0
                    print(f"  {label}: {done}/{len(futs)} ({el:.0f}s, ETA {el / done * (len(futs) - done):.0f}s)", flush=True)

    # 1. questions, once per distinct text
    texts = {}
    for r in rows:
        for t in (r["src"], r["hyp"]):
            texts.setdefault(h("q", str(args.n_questions), t), t)
    todo = [(k, t) for k, t in texts.items() if k not in qcache]

    def gen(k, t):
        qs = parse_questions(judge.chat(QGEN.format(n=args.n_questions, text=t), 1800), args.n_questions)
        remember("q", k, qs)
    print(f"questions: {len(texts)} distinct texts, {len(todo)} to generate", flush=True)
    run_pool("question generation", todo, gen)

    # 2. answers, once per (text, question)
    need = {}
    for r in rows:
        for q_text, a_text in ((r["src"], r["hyp"]), (r["hyp"], r["src"])):
            for q in qcache[h("q", str(args.n_questions), q_text)]:
                need.setdefault(h("a", a_text, q), (a_text, q))
    todo = [(k, v) for k, v in need.items() if k not in acache]

    def ans(k, v):
        remember("a", k, parse_answer(judge.chat(QANS.format(text=v[0], question=v[1]), 8)))
    print(f"answers: {len(need)} distinct (text, question) pairs, {len(todo)} to ask", flush=True)
    run_pool("cross-examination", todo, ans)
    cache_f.close()

    # 3. scores
    gen_fail = invalid = total = 0
    stats = {"coverage": [], "conformity": [], "consistency": []}
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            qs_src = qcache[h("q", str(args.n_questions), r["src"])]
            qs_hyp = qcache[h("q", str(args.n_questions), r["hyp"])]
            a_sh = [(q, acache[h("a", r["hyp"], q)]) for q in qs_src]      # source questions put to the candidate
            a_hs = [(q, acache[h("a", r["src"], q)]) for q in qs_hyp]      # candidate questions put to the source
            gen_fail += (not qs_src) + (not qs_hyp)
            total += len(a_sh) + len(a_hs); invalid += sum(a == "INVALID" for _, a in a_sh + a_hs)
            v_sh = [(q, a) for q, a in a_sh if a != "INVALID"]; v_hs = [(q, a) for q, a in a_hs if a != "INVALID"]
            pct = lambda xs, lab: 100.0 * sum(a == lab for _, a in xs) / len(xs) if xs else None
            cov = None if pct(v_sh, "IDK") is None else 100 - pct(v_sh, "IDK")
            con = None if pct(v_sh, "NO") is None else 100 - pct(v_sh, "NO")
            cns = None if pct(v_hs, "IDK") is None else 100 - pct(v_hs, "IDK")
            for k, v in (("coverage", cov), ("conformity", con), ("consistency", cns)):
                if v is not None:
                    stats[k].append(v)
            f.write(json.dumps({
                "set": r["set"], "system": r["system"], "doc_id": r["doc_id"], "step": r["step"], "direction": r["direction"],
                "cef_coverage": cov, "cef_conformity": con, "cef_consistency": cns,
                "n_q_src": len(v_sh), "n_q_hyp": len(v_hs),
                "cef_omitted": [q for q, a in v_sh if a == "IDK"], "cef_contradicted": [q for q, a in v_sh if a == "NO"],
                "cef_hallucinated": [q for q, a in v_hs if a == "IDK"],
            }, ensure_ascii=False) + "\n")
    print(f"done: {args.output} | question-generation failures {gen_fail} | malformed-JSON recoveries {PARSE_FALLBACKS[0]} | unparseable answers {invalid}/{total}")
    print("means: " + ", ".join(f"{k} {statistics.mean(v):.1f}" for k, v in stats.items() if v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
