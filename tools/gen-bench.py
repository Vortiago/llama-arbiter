#!/usr/bin/env python3
"""Generate speed of one llama-server backend, or of two in an A/B.

Sends a fixed prompt set straight to a backend's /completion, with
cache_prompt false and n_predict 256, at temperature 0 and at the backends'
sampling defaults (temp 1.0, top-p 0.95, top-k 20, min-p 0). Writes one JSON
line per request as it goes, then prints a summary.

    python3 tools/gen-bench.py URL --out run/exp/gen.jsonl
    python3 tools/gen-bench.py --a URL --b URL --reps 3 --out run/exp/gen-ab.jsonl
    python3 tools/gen-bench.py --summarize run/exp/gen.jsonl

With two URLs the order alternates per rep: A B, then B A. The prompts are
built once, from the first URL (chat template and token count), and both
URLs get the same text. The first request to each URL is a warmup and is not
recorded.

The gen_* fields come from the exp/gentrace llama.cpp patch. A build without
it writes rows without them, and the step breakdown leaves those rows out.
No dependencies.
"""
import argparse, json, pathlib, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The parts of a generate step, in loop order. Their sum is the wall time
# from the first generated token to the last. decode_wait is not a part: it
# is the share of decode spent in scheduler syncs (GGML_OP_STATS=1 only).
PARTS = ("gap", "prep", "yield", "draft", "ckpt", "decode", "sync", "spec", "sample", "emit")

SAMPLING = {
    0.0: {"temperature": 0.0},
    1.0: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
}


# ---------------------------------------------------------------- summary math

def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def acceptance(timings):
    drafted = timings.get("draft_n") or 0
    if drafted <= 0:
        return None
    return timings.get("draft_n_accepted", 0) / drafted


def ab_order(labels, rep):
    """A B on even reps, B A on odd reps."""
    return list(labels) if rep % 2 == 0 else list(reversed(labels))


def summarize(rows):
    """Per (label, prompt, temp): median, min, max and spread of generate
    tok/s; median acceptance and prompt tok/s. Spread is (max - min) / median."""
    groups = {}
    for r in rows:
        groups.setdefault((r["label"], r["prompt"], r["temp"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        tps = [r["gen_tps"] for r in rs if r.get("gen_tps") is not None]
        med = median(tps)
        out[key] = {
            "n": len(rs),
            "gen_tps": med,
            "gen_tps_min": min(tps) if tps else None,
            "gen_tps_max": max(tps) if tps else None,
            "gen_tps_spread": (max(tps) - min(tps)) / med if med else None,
            "acceptance": median([r.get("acceptance") for r in rs]),
            "prompt_tps": median([r.get("prompt_tps") for r in rs]),
        }
    return out


def breakdown(rows):
    """Per (label, temp), pooled over prompts and reps: each part's mean ms
    per step and its share of the sum of the parts. tokens_per_step counts the
    tokens after the first (the first comes from the prompt's logits); tps is
    what tokens_per_step and ms_per_step imply. coverage is the sum of the
    parts over the server's own predicted_ms."""
    groups = {}
    for r in rows:
        if r.get("gen_steps"):
            groups.setdefault((r["label"], r["temp"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        steps = sum(r["gen_steps"] for r in rs)
        totals = {p: sum(r.get(f"gen_{p}_ms", 0.0) for r in rs) for p in PARTS}
        totals = {p: v for p, v in totals.items() if any(f"gen_{p}_ms" in r for r in rs)}
        total = sum(totals.values())
        tokens = sum(max(0, r.get("predicted_n", 0) - 1) for r in rs)
        predicted = sum(r.get("predicted_ms", 0.0) for r in rs)
        ms_per_step = total / steps
        b = {
            "steps": steps,
            "ms_per_step": ms_per_step,
            "tokens_per_step": tokens / steps,
            "tps": tokens / total * 1000 if total else None,
            "coverage": total / predicted if predicted else None,
            "parts": {p: {"ms_per_step": v / steps, "pct": 100 * v / total if total else 0.0}
                      for p, v in totals.items()},
        }
        waits = [r["gen_decode_wait_ms"] for r in rs if "gen_decode_wait_ms" in r]
        if waits:
            b["decode_wait_ms_per_step"] = sum(waits) / steps
        out[key] = b
    return out


def print_summary(rows, file=sys.stdout):
    def f(v, spec):
        return "n/a" if v is None else format(v, spec)

    print("\n== generate tok/s and acceptance (median; spread = (max-min)/median)", file=file)
    print(f"{'label':<6} {'prompt':<8} {'temp':>4} {'n':>3} {'gen t/s':>8} {'min':>7} {'max':>7}"
          f" {'spread':>7} {'accept':>7} {'pp t/s':>8}", file=file)
    for (label, prompt, temp), s in sorted(summarize(rows).items()):
        print(f"{label:<6} {prompt:<8} {temp:>4.1f} {s['n']:>3} {f(s['gen_tps'], '8.2f')}"
              f" {f(s['gen_tps_min'], '7.2f')} {f(s['gen_tps_max'], '7.2f')}"
              f" {f(s['gen_tps_spread'] and 100 * s['gen_tps_spread'], '6.1f')}%"
              f" {f(s['acceptance'] and 100 * s['acceptance'], '6.1f')}%"
              f" {f(s['prompt_tps'], '8.1f')}", file=file)

    bd = breakdown(rows)
    if not bd:
        print("\n(no gen_* fields: the backend is not an exp/gentrace build)", file=file)
        return
    for (label, temp), b in sorted(bd.items()):
        print(f"\n== step breakdown: {label}, temp {temp:.1f}: {b['steps']} steps,"
              f" {b['ms_per_step']:.2f} ms/step, {b['tokens_per_step']:.2f} tok/step,"
              f" {f(b['tps'], '.2f')} tok/s, parts/predicted_ms = {f(b['coverage'], '.3f')}", file=file)
        for p, v in b["parts"].items():
            # tok/s if this part took no time: what removing it would give at most
            rest = b["ms_per_step"] - v["ms_per_step"]
            ceil = b["tokens_per_step"] / rest * 1000 if rest > 0 else None
            print(f"  {p:<7} {v['ms_per_step']:8.2f} ms/step {v['pct']:6.1f}%"
                  f"   tok/s without it: {f(ceil, '7.2f')}", file=file)
        if "decode_wait_ms_per_step" in b:
            print(f"  (decode_wait {b['decode_wait_ms_per_step']:.2f} ms/step, inside decode)", file=file)


# ---------------------------------------------------------------- requests

def post(url, path, body, timeout=900):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def n_tokens(url, text):
    return len(post(url, "/tokenize", {"content": text})["tokens"])


def head_tokens(url, text, target):
    """The longest run of whole lines from the start of text with at most
    target tokens, found by scaling the char count by the token ratio."""
    lines = text.splitlines(keepends=True)
    chars = target * 4
    best = ""
    for _ in range(6):
        cut, size = [], 0
        for line in lines:
            if size + len(line) > chars:
                break
            cut.append(line)
            size += len(line)
        piece = "".join(cut)
        n = n_tokens(url, piece)
        if n <= target:
            best = max(best, piece, key=len)
        if n == 0 or abs(n - target) < target * 0.02 or size == len(text):
            break
        chars = int(size * target / n)
    return best


def build_prompts(url, long_file, long_tokens, edit_file):
    long_text = head_tokens(url, pathlib.Path(long_file).read_text(errors="replace"), long_tokens)
    edit_text = head_tokens(url, pathlib.Path(edit_file).read_text(errors="replace"), 1200)
    asks = {
        "code": "Write a Python module that parses an nginx access log from stdin and prints the ten "
                "slowest request paths with their p50 and p99 latency. Use only the standard library, "
                "with type hints and docstrings.",
        "prose": "Write a short story of about 600 words about a lighthouse keeper who finds a radio "
                 "that receives tomorrow's weather. Plain prose, no headings.",
        "json": "You are a tool-calling agent. The tools are read_file(path), grep(pattern, path) and "
                "run(cmd). The task: find why the unit tests in tests/ fail after a rename of the "
                "module router.pool to router.slots. Reply with only a JSON array of the next ten tool "
                "calls, each {\"tool\": ..., \"args\": {...}, \"why\": ...}.",
        "summary": "Summarize the following source file for a new maintainer: its purpose, its main "
                   "classes and functions, and the invariants it keeps.\n\n```python\n" + long_text + "```",
        "edit": "Output the complete file below again, unchanged except that every occurrence of the "
                "word `backend` becomes `server`. Output only the code.\n\n```python\n" + edit_text + "```",
    }
    prompts = {}
    for pid, text in asks.items():
        try:
            prompts[pid] = post(url, "/apply-template", {"messages": [{"role": "user", "content": text}]})["prompt"]
        except Exception as e:  # a server without the endpoint gets the bare text
            print(f"apply-template failed ({e}); {pid} is sent without a chat template", file=sys.stderr)
            prompts[pid] = text
    return prompts


def run_one(url, prompt, temp, seed, n_predict):
    body = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False,
            "seed": seed, "stream": False, **SAMPLING[temp]}
    t0 = time.monotonic()
    res = post(url, "/completion", body)
    wall = time.monotonic() - t0
    t = res.get("timings", {})
    row = {
        "prompt_tokens": t.get("prompt_n"),
        "prompt_tps": t.get("prompt_per_second"),
        "gen_tps": t.get("predicted_per_second"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "draft_n": t.get("draft_n"),
        "draft_n_accepted": t.get("draft_n_accepted"),
        "acceptance": acceptance(t),
        "stop_type": res.get("stop_type"),
        "http_s": round(wall, 3),
    }
    row.update({k: v for k, v in t.items() if k.startswith("gen_")})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url", nargs="?", help="one backend, e.g. http://127.0.0.1:8097")
    ap.add_argument("--a", help="backend A of an A/B")
    ap.add_argument("--b", help="backend B of an A/B")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temps", default="0,1", help="comma list of 0 and/or 1")
    ap.add_argument("--prompts", default="code,prose,json,summary,edit")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--long-file", default=str(ROOT / "bin/router/pool/pool.py"))
    ap.add_argument("--long-tokens", type=int, default=8000)
    ap.add_argument("--edit-file", default=str(ROOT / "tools/perf-ab.py"))
    ap.add_argument("--out", help="JSONL to append rows to (default: run/exp/gen-bench-<time>.jsonl)")
    ap.add_argument("--summarize", metavar="JSONL", help="print the summary of a written file and exit")
    a = ap.parse_args()

    if a.summarize:
        rows = [json.loads(l) for l in pathlib.Path(a.summarize).read_text().splitlines() if l.strip()]
        print_summary(rows)
        return

    urls = {}
    if a.url:
        urls["a"] = a.url
    if a.a:
        urls["a"] = a.a
    if a.b:
        urls["b"] = a.b
    if not urls:
        ap.error("give a URL, or --a and --b")

    temps = [float(t) for t in a.temps.split(",")]
    for t in temps:
        if t not in SAMPLING:
            ap.error(f"temp {t}: only 0 and 1 are defined")
    pids = a.prompts.split(",")
    out = pathlib.Path(a.out or ROOT / "run/exp" / time.strftime("gen-bench-%Y%m%d-%H%M%S.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)

    first = next(iter(urls.values()))
    prompts = build_prompts(first, a.long_file, a.long_tokens, a.edit_file)
    out.with_suffix(".prompts.json").write_text(json.dumps(prompts, indent=1))

    for label, url in urls.items():
        post(url, "/completion", {"prompt": prompts["code"], "n_predict": 16, "cache_prompt": False})
        print(f"warmup {label} {url} done", file=sys.stderr)

    rows = []
    with out.open("a") as fh:
        for rep in range(a.reps):
            for temp in temps:
                for pid in pids:
                    for label in ab_order(list(urls), rep):
                        r = {"label": label, "url": urls[label], "prompt": pid, "temp": temp,
                             "rep": rep, "seed": a.seed, "time": time.time()}
                        r.update(run_one(urls[label], prompts[pid], temp, a.seed, a.n_predict))
                        rows.append(r)
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                        acc = r["acceptance"]
                        print(f"rep {rep} temp {temp:.0f} {pid:<8} {label}: {r['gen_tps'] or 0:6.2f} tok/s"
                              f"  accept {'n/a' if acc is None else f'{100 * acc:.1f}%'}"
                              f"  pp {r['prompt_tokens']} tok", file=sys.stderr)
    print(f"rows: {out}", file=sys.stderr)
    print_summary(rows)


if __name__ == "__main__":
    main()
