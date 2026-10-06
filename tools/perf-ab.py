#!/usr/bin/env python3
"""Measure one llama.cpp build and append the numbers to a TSV file.

Each run starts one llama-server (or llama-bench, or llama-perplexity), takes
its measurement and stops it, so that two builds never share the CPU. The
`report` command compares every label in the TSV against a reference label.

    perf-ab.py prefill  --label B-main --build DIR
    perf-ab.py generate --label B-main --build DIR
    perf-ab.py kernel   --label B-main --build DIR
    perf-ab.py quality  --label B-main --build DIR [--base]
    perf-ab.py report   --ref B-main
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BENCH = Path(os.environ.get("BENCH_DIR", "/home/atle/llama-arbiter-bench"))
MODELS = Path("/home/atle/local_llm/models")
MODEL = MODELS / "unsloth/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
DRAFT = MODELS / "ggml-org/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
TSV = BENCH / "perf-results.tsv"
SLOTS = BENCH / "slots"
PORT = 18080

# The live backends' context, so the KV cache and its layout match production.
CTX = 262144

# A model of 175 GiB maps in about 90 s from a warm page cache, and several
# minutes from a cold one.
STARTUP_SECONDS = 900

# A read of 8201 tokens at 40 tok/s takes about 200 s. Twice that is a hang.
REQUEST_SECONDS = 1800

# A depth fill reads up to 240k tokens once, at 20 to 40 tok/s: up to about 3.5 h.
FILL_SECONDS = 6 * 3600

COMMON = [
    "--parallel", "1", "--flash-attn", "auto", "--jinja", "--cache-ram", "0",
    "--checkpoint-min-step", "2048", "--ctx-checkpoints", "8",
    "--no-cache-idle-slots", "--device", "none",
    "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
]

# The flags of bin/cpu-prefill.sh and bin/cpu-generate.sh, one slot each.
ROLE_FLAGS = {
    "prefill": ["--threads", "32", "--threads-batch", "32", "--cpu-range", "0-31",
                "--cpu-strict", "1", "--batch-size", "512", "--ubatch-size", "512"],
    "generate": ["--threads", "16", "--threads-batch", "16", "--cpu-range", "0-31",
                 "--cpu-strict", "0", "--prio", "1", "--batch-size", "512",
                 "--ubatch-size", "512"],
}

GENERATE_PROMPTS = {
    "explain": "Explain how a gated delta network differs from softmax attention, "
               "and why that matters for long-context inference.",
    "code": "Write a Python function that parses an ISO 8601 duration such as "
            "P3DT4H12M into a number of seconds. Include type hints and tests.",
    "list": "List twenty European capital cities with one sentence on the river "
            "or coast each one lies on.",
}
SEED = 42

# Wikitext averages about 4 characters a token for this tokenizer.
CHARS_PER_TOKEN = 4

# About 2000 tokens read at depth: four ubatches of 512.
DEPTH_EXTENSION_CHARS = 8000

DEPTH_REPLY_TOKENS = 256

COLUMNS = ["time", "label", "test", "case", "rep", "metric", "value", "tokens",
           "build", "flags", "load", "others_cpu"]

# The box also runs other people's containers. A rep taken while one of them
# compiles reads slow, so each row carries the load it was measured under.
BENCH_PROCESSES = ("llama-server", "llama-bench", "llama-perplexity", "perf-ab.py")
OTHERS_SAMPLE_SECONDS = 0.5


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)

    for name, func in [("prefill", run_prefill), ("pair", run_pair), ("generate", run_generate), ("depth", run_depth),
                       ("kernel", run_kernel), ("quality", run_quality)]:
        p = sub.add_parser(name)
        p.add_argument("--label", required=True)
        p.add_argument("--build", required=True, type=Path,
                       help="a build directory that holds bin/")
        p.add_argument("--draft", type=Path, default=DRAFT,
                       help="the MTP draft GGUF, or none to run without speculation")
        p.add_argument("--extra", default="",
                       help="flags added after the role's own, so they win")
        p.add_argument("--reps", type=int, default=3)
        p.add_argument("--tsv", type=Path, default=TSV)
        p.set_defaults(func=func)
        if name in ("prefill", "pair"):
            p.add_argument("--prompt-file", type=Path, default=BENCH / "prompt-8k.txt")
        if name == "generate":
            p.add_argument("--n-predict", type=int, default=512)
        if name == "depth":
            p.add_argument("--depth", type=int, default=32768,
                           help="tokens of context before the measured read and reply")
            p.set_defaults(reps=2)
        if name == "kernel":
            p.add_argument("--bench-args", default="-p 512,2048 -n 128")
        if name == "quality":
            p.add_argument("--base", action="store_true",
                           help="write the base logits rather than compare to them")
            p.add_argument("--chunks", type=int, default=20)
            p.add_argument("--ctx", type=int, default=512,
                           help="tokens per chunk; each context size has its own base logits")

    p = sub.add_parser("report")
    p.add_argument("--ref", required=True)
    p.add_argument("--tsv", type=Path, default=TSV)
    p.set_defaults(func=run_report)
    return parser.parse_args()


def run_prefill(args):
    prompt = args.prompt_file.read_text()
    with Server(args, "prefill") as server:
        for rep in range(args.reps):
            body = {"prompt": prompt, "n_predict": 1, "cache_prompt": False}
            timings = server.complete(body)["timings"]
            record(args, "prefill", "8k", rep, "prompt_tok_s",
                   timings["prompt_per_second"], timings["prompt_n"])


def run_pair(args):
    """Read two prompts at once on two prefill backends, as pre0 and pre1 share the socket.

    Each backend gets its own text, so neither reads what the other has cached.
    The aggregate is both prompts' tokens over the time until the slower one ends.
    """
    first = args.prompt_file.read_text()
    second = (BENCH / "wiki.test.raw").read_text()[400000:400000 + len(first)]
    with Server(args, "prefill", port=PORT) as a, Server(args, "prefill", port=PORT + 1) as b:
        for rep in range(args.reps):
            bodies = [{"prompt": text, "n_predict": 1, "cache_prompt": False}
                      for text in (first, second)]
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                results = list(pool.map(lambda pair: pair[0].complete(pair[1]),
                                        zip((a, b), bodies)))
            wall = time.monotonic() - started
            tokens = sum(r["timings"]["prompt_n"] for r in results)
            for side, result in zip("ab", results):
                record(args, "pair", f"8k-{side}", rep, "prompt_tok_s",
                       result["timings"]["prompt_per_second"], result["timings"]["prompt_n"])
            record(args, "pair", "8k-both", rep, "prompt_tok_s", tokens / wall, tokens)


def run_generate(args):
    with Server(args, "generate") as server:
        for case, prompt in GENERATE_PROMPTS.items():
            for rep in range(args.reps):
                result = server.complete(sampled(prompt, args.n_predict))
                timings = result["timings"]
                record(args, "generate", case, rep, "gen_tok_s",
                       timings["predicted_per_second"], timings["predicted_n"])
                record(args, "generate", case, rep, "acceptance",
                       acceptance(timings), timings.get("draft_n", 0))
            text = server.complete(greedy(prompt, args.n_predict))["content"]
            record(args, "generate", case, 0, "greedy_sha", digest(text), len(text))


def run_depth(args):
    """Time a read and a reply with a long context already in the slot.

    The context is read once and saved to a slot file. Every rep restores that
    file, so each build starts from the same state, and only the extension is read.
    """
    text = (BENCH / "wiki.test.raw").read_text()
    chars = args.depth * CHARS_PER_TOKEN
    context = text[:chars]
    saved = f"depth{args.depth // 1024}k.bin"
    case = f"d{args.depth // 1024}k"
    with Server(args, "generate", slot_dir=SLOTS) as server:
        if not (SLOTS / saved).exists():
            timings = server.complete({"prompt": context, "n_predict": 0,
                                       "cache_prompt": True}, timeout=FILL_SECONDS)["timings"]
            record(args, "depth", f"fill-{case}", 0, "prompt_tok_s",
                   timings["prompt_per_second"], timings["prompt_n"])
            server.slot_action("save", saved)
        for rep in range(args.reps):
            server.slot_action("restore", saved)
            start = chars + rep * DEPTH_EXTENSION_CHARS
            body = sampled(context + text[start:start + DEPTH_EXTENSION_CHARS],
                           DEPTH_REPLY_TOKENS)
            body.update({"cache_prompt": True, "id_slot": 0})
            timings = server.complete(body)["timings"]
            record(args, "depth", case, rep, "prompt_tok_s",
                   timings["prompt_per_second"], timings["prompt_n"])
            record(args, "depth", case, rep, "gen_tok_s",
                   timings["predicted_per_second"], timings["predicted_n"])
            record(args, "depth", case, rep, "acceptance",
                   acceptance(timings), timings.get("draft_n", 0))


def greedy(prompt, n_predict):
    return {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "cache_prompt": False}


def sampled(prompt, n_predict):
    return {"prompt": prompt, "n_predict": n_predict, "seed": SEED,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "cache_prompt": False}


def acceptance(timings):
    drafted = timings.get("draft_n", 0)
    return timings.get("draft_n_accepted", 0) / drafted if drafted else 0.0


def digest(text):
    """Return a short hash, so that two builds' greedy outputs compare in one cell."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def run_kernel(args):
    command = [str(args.build / "bin/llama-bench"), "-m", str(MODEL), "-t", "32",
               "-C", "0xffffffff", "--cpu-strict", "1", "-fa", "1",
               "-r", str(args.reps), "-o", "jsonl",
               *shlex.split(args.bench_args), *shlex.split(args.extra)]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    for row in parse_bench(output):
        for rep, tok_s in enumerate(row["samples"]):
            record(args, "kernel", row["case"], rep, "tok_s", tok_s, row["tokens"])


def parse_bench(output):
    """Turn llama-bench jsonl into one row for each test it ran."""
    rows = []
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        test = json.loads(line)
        prompt, gen, depth = test["n_prompt"], test["n_gen"], test.get("n_depth", 0)
        case = f"pp{prompt}" if prompt else f"tg{gen}"
        if test.get("n_ubatch") != 512:
            case += f"-ub{test['n_ubatch']}"
        if depth:
            case += f"@d{depth}"
        rows.append({"case": case, "samples": test["samples_ts"], "tokens": prompt or gen})
    return rows


def run_quality(args):
    base = BENCH / ("kld-base.bin" if args.ctx == 512 else f"kld-base-c{args.ctx}.bin")
    text = BENCH / "wiki.test.raw"
    command = [str(args.build / "bin/llama-perplexity"), "-m", str(MODEL),
               "-f", str(text), "-t", "32", "--chunks", str(args.chunks), "-c", str(args.ctx),
               "-fa", "on", "--device", "none", *shlex.split(args.extra)]
    if args.base:
        command += ["--kl-divergence-base", str(base)]
    else:
        command += ["--kl-divergence-base", str(base), "--kl-divergence"]
    log = subprocess.run(command, check=True, capture_output=True, text=True)
    if args.base:
        return
    for metric, value in parse_kld(log.stdout + log.stderr).items():
        record(args, "quality", f"wiki-c{args.ctx}", 0, metric, value, args.chunks)


def parse_kld(text):
    """Read the mean KLD and the top-1 agreement out of llama-perplexity's log."""
    found = {}
    match = re.search(r"Mean\s+KLD:\s+([-\d.]+)", text)
    if match:
        found["mean_kld"] = float(match.group(1))
    match = re.search(r"Same top p:\s+([\d.]+)", text)
    if match:
        found["same_top_p"] = float(match.group(1))
    # PPL tells a harmless reshuffle of hard choices from a real loss, which KLD cannot.
    for name, pattern in (("ppl", r"Mean PPL\(Q\)\s*:\s*([\d.]+)"),
                          ("ppl_base", r"Mean PPL\(base\)\s*:\s*([\d.]+)"),
                          ("ppl_ratio", r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)"),
                          ("ppl_ratio_unc", r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*[\d.]+\s*±\s*([\d.]+)")):
        match = re.search(pattern, text)
        if match:
            found[name] = float(match.group(1))
    if not found:
        raise RuntimeError("expected Mean KLD and Same top p in the perplexity log, found neither")
    return found


def draft_flags(draft):
    """Name the MTP draft, or nothing when the draft is the word none."""
    if str(draft) == "none":
        return []
    return ["--model-draft", str(draft), "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]


class Server:
    """One llama-server on a port, started for a measurement and stopped after it."""

    def __init__(self, args, role, slot_dir=None, port=PORT):
        self.port = port
        self.log_path = BENCH / "run" / f"{args.label}-{role}-{port}.log"
        slots = ["--slot-save-path", f"{slot_dir}/"] if slot_dir else []
        self.command = [str(args.build / "bin/llama-server"), "--model", str(MODEL),
                        "--ctx-size", str(CTX), *COMMON, *draft_flags(args.draft),
                        *ROLE_FLAGS[role], *slots, *shlex.split(args.extra),
                        "--host", "127.0.0.1", "--port", str(port)]
        self.process = None

    def __enter__(self):
        refuse_if_taken(self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"}
        with open(self.log_path, "w") as log:
            self.process = subprocess.Popen(self.command, stdout=log, stderr=log,
                                            env=env, start_new_session=True)
        self.wait_healthy()
        return self

    def __exit__(self, *exc):
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def wait_healthy(self):
        deadline = time.monotonic() + STARTUP_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.process.returncode}, "
                                   f"see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5):
                    return
            except OSError:
                time.sleep(2)
        raise RuntimeError(f"llama-server not healthy after {STARTUP_SECONDS} s")

    def complete(self, body, timeout=REQUEST_SECONDS):
        return self.post("/completion", body, timeout)

    def slot_action(self, action, filename):
        """Save slot 0 to, or restore it from, a file in the slot directory."""
        return self.post(f"/slots/0?action={action}", {"filename": filename})

    def post(self, path, body, timeout=REQUEST_SECONDS):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)


def refuse_if_taken(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"port {port} is in use: stop what holds it first")


def record(args, test, case, rep, metric, value, tokens):
    new = not args.tsv.exists()
    with open(args.tsv, "a") as out:
        if new:
            out.write("\t".join(COLUMNS) + "\n")
        row = [time.strftime("%Y-%m-%dT%H:%M:%S"), args.label, test, case, str(rep),
               metric, format_value(value), str(tokens), build_id(args.build), args.extra,
               load_average(), f"{others_cpu():.0f}"]
        out.write("\t".join(row) + "\n")
    print(f"{args.label:16} {test:9} {case:12} {rep} {metric:12} {format_value(value)}",
          flush=True)


def load_average():
    return Path("/proc/loadavg").read_text().split()[0]


def others_cpu():
    """Return the CPU percent that processes outside the bench use now.

    ps reports CPU averaged over a process's whole life, so a compile that
    has just started would not show. Two /proc samples half a second apart do.
    """
    before = process_ticks()
    time.sleep(OTHERS_SAMPLE_SECONDS)
    after = process_ticks()
    ticks = sum(after[pid][0] - before[pid][0] for pid in after
                if pid in before and not is_bench(after[pid][1]))
    return 100 * ticks / os.sysconf("SC_CLK_TCK") / OTHERS_SAMPLE_SECONDS


def process_ticks():
    """Map each pid to its user plus system clock ticks and its command line."""
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue  # the process ended between the listing and the read
        found[entry.name] = (parse_ticks(stat), command)
    return found


def parse_ticks(stat):
    """Read utime plus stime from a /proc/<pid>/stat line."""
    fields = stat[stat.rindex(")") + 2:].split()
    return int(fields[11]) + int(fields[12])


def is_bench(command):
    return any(name in command for name in BENCH_PROCESSES)


def format_value(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def build_id(build):
    """Name a build by the commit its source tree is at."""
    try:
        return subprocess.run(["git", "-C", str(build.parent), "rev-parse", "--short", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError:
        return build.name


def run_report(args):
    groups = read_groups(args.tsv)
    reference = {key[1:]: values for key, values in groups.items() if key[0] == args.ref}
    if not reference:
        raise SystemExit(f"expected label {args.ref!r} in {args.tsv}, found none")
    print(f"{'label':16} {'test':9} {'case':12} {'metric':12} {'median':>9} "
          f"{'min':>9} {'max':>9} {'vs ref':>8} {'ref spread':>10}")
    for (label, test, case, metric), values in sorted(groups.items()):
        print(format_row(label, test, case, metric, values,
                         reference.get((test, case, metric))))


def read_groups(tsv):
    """Group the numeric values of a results file by label, test, case and metric."""
    groups = {}
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        try:
            value = float(row["value"])
        except ValueError:
            continue  # a greedy hash, compared by eye
        groups.setdefault((row["label"], row["test"], row["case"], row["metric"]),
                          []).append(value)
    return groups


def format_row(label, test, case, metric, values, reference):
    median = statistics.median(values)
    delta = spread = ""
    if reference:
        ref_median = statistics.median(reference)
        if ref_median:
            delta = f"{100 * (median / ref_median - 1):+.1f}%"
            spread = f"{100 * (max(reference) - min(reference)) / ref_median:.1f}%"
    return (f"{label:16} {test:9} {case:12} {metric:12} {median:9.3f} "
            f"{min(values):9.3f} {max(values):9.3f} {delta:>8} {spread:>10}")


if __name__ == "__main__":
    sys.exit(main())
