#!/usr/bin/env python3
"""Offline simulator of a fixed set of hot experts on the card.

Reads the small-batch log of GGML_EXPERT_STATS (<base>.bin and its rotated
<base>.bin.N files). For each budget it chooses a hot set of (layer, expert)
pairs from the first half of the steps and scores it on the second half:

- pick hit rate: routed (token, expert) rows whose expert is hot;
- saved per step: experts a step no longer reads from RAM (the CPU reads
  each expert of a call's union once);
- MB per step before and after, and the CPU time saved at --gbps.

Two ways to spend a budget: "equal" (the same slots in every layer) and
"greedy" (the globally most read pairs, which is optimal for the training
half because the benefit adds up per pair).

A cache is fixed, so only the counts matter, not the order: every metric is a
sum over the hot pairs of per-pair counts from the scoring half.

    tools/expert-cache-sim.py /mnt/storage/qwen-experiment/run/exp/expertstats/gpu0_0
    tools/expert-cache-sim.py BASE --range code=2026-10-15T09:00:2026-10-15T12:00 \
        --range prose=2026-10-15T13:00:2026-10-15T15:00

Stdlib only. Format: run/exp/expertstats-README.md.
"""
import argparse
import collections
import datetime
import glob
import os
import re
import struct
import sys

HEADER_MAGIC = b"GXSTATS1"
HEADER = struct.Struct("<8sQ")    # magic, unix time of the process start (us)
REC = struct.Struct("<IHBBHH")    # t_ms, layer, n_tokens, 0, n_union, offloaded picks

Record = collections.namedtuple("Record", "t_ms layer n_tokens picks offload")


# --- format ---

def encode_record(t_ms, layer, n_tokens, picks, offload=0):
    ids = sorted(picks.items())
    out = REC.pack(t_ms, layer, n_tokens, 0, len(ids), offload)
    return out + b"".join(struct.pack("<H", e | ((min(c, 16) - 1) << 12)) for e, c in ids)


def iter_records(data):
    """(t0_us, generator of Record with t_ms relative to t0). A truncated tail is dropped."""
    if len(data) < HEADER.size:
        raise ValueError("short file")
    magic, t0 = HEADER.unpack_from(data, 0)
    if magic != HEADER_MAGIC:
        raise ValueError("not an expert stats log")

    def gen():
        pos, n = HEADER.size, len(data)
        while pos + REC.size <= n:
            t, layer, nt, _, nu, off = REC.unpack_from(data, pos)
            end = pos + REC.size + 2 * nu
            if end > n:
                return
            ids = struct.unpack_from("<%dH" % nu, data, pos + REC.size)
            yield Record(t, layer, nt, tuple((v & 0xfff, (v >> 12) + 1) for v in ids), off)
            pos = end
    return t0, gen()


def parse(data):
    t0, gen = iter_records(data)
    return t0, list(gen)


def log_files(base):
    if base.endswith(".bin") or re.search(r"\.bin\.\d+$", base):
        return [base]
    rotated = glob.glob(glob.escape(base) + ".bin.*")
    rotated = sorted((f for f in rotated if re.search(r"\.bin\.\d+$", f)),
                     key=lambda f: -int(f.rsplit(".", 1)[1]))
    files = rotated + ([base + ".bin"] if os.path.exists(base + ".bin") else [])
    if not files:
        raise SystemExit("no log at %s.bin" % base)
    return files


def stream(files):
    """Every record of the files, oldest file first, with absolute t_ms (unix ms)."""
    for f in files:
        with open(f, "rb") as fh:
            data = fh.read()
        try:
            t0, gen = iter_records(data)
        except ValueError as e:
            print("skip %s: %s" % (f, e), file=sys.stderr)
            continue
        base = t0 // 1000
        for r in gen:
            yield r._replace(t_ms=base + r.t_ms)


# --- steps ---

class Stepper:
    """A main-model step starts when a layer comes again at or below the last one seen.
    Records of the MTP layer belong to the current step."""

    def __init__(self, mtp_layer):
        self.mtp = mtp_layer
        self.step = -1
        self.last = None

    def __call__(self, r):
        if r.layer != self.mtp:
            if self.last is None or r.layer <= self.last:
                self.step += 1
            self.last = r.layer
        return max(self.step, 0)


def assign_steps(records, mtp_layer):
    s = Stepper(mtp_layer)
    return [s(r) for r in records]


def split_half(steps):
    """The first half of the distinct steps (in order) and the rest."""
    order = sorted(set(steps))
    h = len(order) // 2
    return set(order[:h]), set(order[h:])


# --- counts and hot sets ---

def read_counts(records):
    """{layer: {expert: calls that read it}}"""
    out = {}
    for r in records:
        d = out.setdefault(r.layer, {})
        for e, _ in r.picks:
            d[e] = d.get(e, 0) + 1
    return out


class Agg:
    """Per-pair read and pick counts of a set of records, and its step count."""

    def __init__(self):
        self.read = {}
        self.picks = {}
        self.steps = set()
        self.n_picks = 0
        self.n_reads = 0

    def add(self, r, step):
        self.steps.add(step)
        rd = self.read.setdefault(r.layer, {})
        pk = self.picks.setdefault(r.layer, {})
        for e, c in r.picks:
            rd[e] = rd.get(e, 0) + 1
            pk[e] = pk.get(e, 0) + c
            self.n_picks += c
        self.n_reads += len(r.picks)


def slots_for(budget_gb, expert_mb):
    return int(budget_gb * 1000.0 / expert_mb + 1e-9)


def _ranked(d):
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))


def hot_greedy(counts, n_slots):
    pairs = sorted(((c, l, e) for l, d in counts.items() for e, c in d.items()),
                   key=lambda t: (-t[0], t[1], t[2]))
    out = {}
    for _, l, e in pairs[:n_slots]:
        out.setdefault(l, set()).add(e)
    return out


def hot_equal(counts, n_slots, layers):
    layers = sorted(layers)
    if not layers:
        return {}
    per, rem = divmod(n_slots, len(layers))
    out = {}
    for i, l in enumerate(layers):
        k = per + (1 if i < rem else 0)
        top = [e for e, _ in _ranked(counts.get(l, {}))[:k]]
        if top:
            out[l] = set(top)
    return out


def n_hot(hot):
    return sum(len(s) for s in hot.values())


def overlap(a, b):
    """|A & B| / |A| over (layer, expert) pairs; 1.0 for two empty sets."""
    pa = {(l, e) for l, s in a.items() for e in s}
    pb = {(l, e) for l, s in b.items() for e in s}
    if not pa:
        return 1.0 if not pb else 0.0
    return len(pa & pb) / len(pa)


def evaluate_agg(agg, hot, n_steps, expert_bytes, gbps, layers=None):
    keep = (lambda l: True) if layers is None else (lambda l: l in layers)
    picks = sum(c for l, d in agg.picks.items() if keep(l) for c in d.values())
    reads = sum(c for l, d in agg.read.items() if keep(l) for c in d.values())
    hit_p = sum(agg.picks.get(l, {}).get(e, 0) for l, s in hot.items() if keep(l) for e in s)
    hit_r = sum(agg.read.get(l, {}).get(e, 0) for l, s in hot.items() if keep(l) for e in s)
    n = max(n_steps, 1)
    saved = hit_r / n
    return {
        "pick_hit_rate": hit_p / picks if picks else 0.0,
        "read_per_step": reads / n,
        "saved_per_step": saved,
        "bytes_before": reads / n * expert_bytes,
        "bytes_after": (reads - hit_r) / n * expert_bytes,
        "ms_saved": saved * expert_bytes / (gbps * 1e9) * 1e3,
    }


def evaluate(records, hot, n_steps, expert_bytes, gbps):
    agg = Agg()
    for r in records:
        agg.add(r, 0)
    return evaluate_agg(agg, hot, n_steps, expert_bytes, gbps)


# --- command line ---

def parse_time(s):
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s) * 1000.0
    return datetime.datetime.fromisoformat(s).timestamp() * 1000.0


def parse_range(s):
    name, _, span = s.partition("=")
    # START:END where each is unix seconds or an ISO time (which has colons itself)
    m = re.fullmatch(r"(\d+(?:\.\d+)?|\d{4}-\d\d-\d\dT[\d:]+)\:(\d+(?:\.\d+)?|\d{4}-\d\d-\d\dT[\d:]+)", span)
    if not name or not m:
        raise argparse.ArgumentTypeError("--range NAME=START:END")
    return name, parse_time(m.group(1)), parse_time(m.group(2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="the GGML_EXPERT_STATS path, or one .bin file")
    ap.add_argument("--budgets", default="0.5,1,1.7,2.7,4,6", help="GB (1e9 bytes)")
    ap.add_argument("--expert-mb", type=float, default=5.2224, help="MB of one expert (gate+up+down)")
    ap.add_argument("--gbps", type=float, default=94.0, help="CPU expert read rate, GB/s")
    ap.add_argument("--mtp-layer", type=int, default=48)
    ap.add_argument("--min-tokens", type=int, default=1, help="main-model calls with fewer tokens are skipped")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--range", action="append", type=parse_range, default=[],
                    help="NAME=START:END (unix s or ISO local time): a workload to compare")
    a = ap.parse_args(argv)

    files = log_files(a.base)
    budgets = [float(b) for b in a.budgets.split(",")]
    eb = a.expert_mb * 1e6

    def wanted(r):
        return r.layer == a.mtp_layer or a.min_tokens <= r.n_tokens <= a.max_tokens

    # pass 1: the steps and their start times
    st = Stepper(a.mtp_layer)
    n_rec = 0
    t_first = t_last = None
    for r in stream(files):
        if not wanted(r):
            continue
        st(r)
        n_rec += 1
        t_first = r.t_ms if t_first is None else t_first
        t_last = r.t_ms
    n_steps = st.step + 1
    if n_steps < 2:
        raise SystemExit("need at least 2 steps, have %d (%d records)" % (n_steps, n_rec))
    half = n_steps // 2

    # pass 2: counts per half and per range
    train, test = Agg(), Agg()
    ranges = [(name, lo, hi, Agg()) for name, lo, hi in a.range]
    tokens = collections.Counter()
    st = Stepper(a.mtp_layer)
    for r in stream(files):
        if not wanted(r):
            continue
        s = st(r)
        (train if s < half else test).add(r, s)
        if r.layer != a.mtp_layer:
            tokens[r.n_tokens] += 1
        for _, lo, hi, g in ranges:
            if lo <= r.t_ms < hi:
                g.add(r, s)

    layers = sorted(set(train.read) | set(test.read))
    n_tr, n_te = len(train.steps), len(test.steps)
    fmt_t = lambda ms: datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    print("log: %s (%d files), %d records, %d steps, %s to %s" % (
        a.base, len(files), n_rec, n_steps, fmt_t(t_first), fmt_t(t_last)))
    print("main-model calls by tokens: %s" % dict(sorted(tokens.items())))
    print("train: steps 0-%d (%d), score: steps %d-%d (%d); layers %s; MTP layer %d %s" % (
        half - 1, n_tr, half, n_steps - 1, n_te, "%d-%d" % (layers[0], layers[-1]) if layers else "-",
        a.mtp_layer, "present" if a.mtp_layer in layers else "absent"))
    before = evaluate_agg(test, {}, n_te, eb, a.gbps)
    mtp_before = evaluate_agg(test, {}, n_te, eb, a.gbps, layers={a.mtp_layer})
    print("score half: %.1f experts read a step (MTP layer %.1f), %.0f MB, %.1f ms at %.0f GB/s" % (
        before["read_per_step"], mtp_before["read_per_step"], before["bytes_before"] / 1e6,
        before["bytes_before"] / (a.gbps * 1e9) * 1e3, a.gbps))
    print()
    print("%-6s %-6s %5s | %7s %8s %8s %13s %8s | %8s %8s | %7s %7s" % (
        "GB", "alloc", "slots", "hits", "saved/st", "mtp sv", "MB/st bef>aft", "ms/st", "oracle", "ora ms",
        "overlap", "ovl@25%"))
    for b in budgets:
        n = slots_for(b, a.expert_mb)
        for name in ("equal", "greedy"):
            pick = (lambda c: hot_greedy(c, n)) if name == "greedy" else (lambda c: hot_equal(c, n, layers))
            hot = pick(train.read)
            ora = pick(test.read)
            r = evaluate_agg(test, hot, n_te, eb, a.gbps)
            rm = evaluate_agg(test, hot, n_te, eb, a.gbps, layers={a.mtp_layer})
            ro = evaluate_agg(test, ora, n_te, eb, a.gbps)
            # how stable the top of the set is: the top quarter of each half's ranking
            q = max(n // 4, 1)
            hq = hot_greedy(train.read, q) if name == "greedy" else hot_equal(train.read, q, layers)
            oq = hot_greedy(test.read, q) if name == "greedy" else hot_equal(test.read, q, layers)
            print("%-6g %-6s %5d | %6.1f%% %8.1f %8.2f %6.0f>%-6.0f %8.2f | %8.1f %8.2f | %6.1f%% %6.1f%%" % (
                b, name, n_hot(hot), 100 * r["pick_hit_rate"], r["saved_per_step"], rm["saved_per_step"],
                r["bytes_before"] / 1e6, r["bytes_after"] / 1e6, r["ms_saved"], ro["saved_per_step"],
                ro["ms_saved"], 100 * overlap(hot, ora), 100 * overlap(hq, oq)))
    print("\nhits: routed rows of the score half on a hot expert. saved/st: experts a step no longer"
          " reads (mtp sv: of them, the MTP layer). oracle: the set chosen on the score half itself."
          " overlap: |train set & score set| / |set|, at the budget and for its top quarter.")

    if ranges:
        print("\nworkloads (greedy sets chosen on each range; cell = experts saved a step on the column's range"
              " / overlap with the column's own set)")
        for name, lo, hi, g in ranges:
            print("  %-10s %s to %s: %d steps, %d reads" % (name, fmt_t(lo), fmt_t(hi), len(g.steps), g.n_reads))
        for b in budgets:
            n = slots_for(b, a.expert_mb)
            sets = {name: hot_greedy(g.read, n) for name, _, _, g in ranges}
            print("  %g GB:" % b)
            for name, _, _, _ in ranges:
                cells = []
                for col, _, _, g in ranges:
                    r = evaluate_agg(g, sets[name], len(g.steps), eb, a.gbps)
                    cells.append("%s %.1f/%3.0f%%" % (col, r["saved_per_step"], 100 * overlap(sets[name], sets[col])))
                print("    from %-10s %s" % (name, "  ".join(cells)))


if __name__ == "__main__":
    main()
