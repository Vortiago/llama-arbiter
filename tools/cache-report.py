#!/usr/bin/env python3
"""What the cache did, read from its own event log.

The dashboard shows the present; this answers questions about the past, from
run/cache-events-*.jsonl: which openings paid for themselves, how long wants
starved, how much of every read the prompt cache skipped, what the forks did.

    python3 tools/cache-report.py [--dir run] [--since-hours 24]

Numbers are only as wide as the window asked for. Compare two windows by
running it twice with different --since-hours.
"""
import argparse, json, os, pathlib, statistics, sys, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.environ.get("CACHE_LOG_DIR") or os.environ.get("RUN")
                or str(pathlib.Path(__file__).resolve().parent.parent / "run"))
ap.add_argument("--since-hours", type=float, default=24.0)
a = ap.parse_args()

now = time.time()
floor = now - a.since_hours * 3600
# One file a day, named for the day, so a file whose day ended before the
# window opened holds nothing this report can use. Without this the default
# 24 hour question reads and parses the whole archive to throw almost all of
# it away, and the cost grows with every day the router runs.
floor_day = time.strftime("%Y%m%d", time.localtime(floor))
rows = []
for path in sorted(pathlib.Path(a.dir).glob("cache-events-*.jsonl")):
    if path.stem.rsplit("-", 1)[-1] < floor_day:
        continue
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("ts", 0) >= floor:
            rows.append(row)

if not rows:
    print(f"no cache events in {a.dir} within {a.since_hours:g}h")
    sys.exit(0)


def part(name):
    print(f"\n== {name} ==")


def numbers(rows, field):
    """Every numeric value of this field, skipping the rows that lack it."""
    return [r[field] for r in rows if isinstance(r.get(field), (int, float))]


print(f"{len(rows)} events, {time.strftime('%Y-%m-%d %H:%M', time.localtime(min(r['ts'] for r in rows)))}"
      f" to {time.strftime('%Y-%m-%d %H:%M', time.localtime(max(r['ts'] for r in rows)))}")

requests = [r for r in rows if r["event"] == "request"]
if requests:
    part("turns")
    started = Counter(r.get("started") for r in requests)
    print("started from:  " + "  ".join(f"{k}={v}" for k, v in started.most_common()))
    print("clients:       " + "  ".join(f"{k or '?'}={v}" for k, v in
                                        Counter(r.get("client") for r in requests).most_common()))
    print("identity:      " + "  ".join(f"{k}={v}" for k, v in
                                        Counter(r.get("source") for r in requests).most_common()))
    skipped, processed = numbers(requests, "read_cache_n"), numbers(requests, "read_prompt_n")
    if skipped or processed:
        whole = sum(skipped) + sum(processed)
        share = 100 * sum(skipped) / whole if whole else 0.0
        print(f"read pass:     {sum(processed):>9} tokens processed, "
              f"{sum(skipped):>9} skipped by the cache ({share:.0f}% skipped)")
    took = numbers(requests, "took")
    if took:
        print(f"turn seconds:  median {statistics.median(took):.1f}, "
              f"max {max(took):.1f}")
    left = sum(1 for r in requests if r.get("left"))
    if left:
        print(f"client left:   {left}")

overs = [r for r in rows if r["event"] == "start_over"]
if overs:
    part("start overs")
    print("  ".join(f"{k}={v}" for k, v in
                    Counter(r.get("reason") for r in overs).most_common()))

choices = [r for r in rows if r["event"] == "choice"]
if choices:
    part("what warm_prefix decided")
    print("plans:   " + "  ".join(f"{k}={v}" for k, v in
                                  Counter(r.get("plan") for r in choices).most_common()))
    deeper = [r for r in choices if (r.get("shared") or -1) > (r.get("stored") or -1)]
    print(f"shared deeper than saved: {len(deeper)} of {len(choices)}")

    # The question deeper openings exist to answer, and the reason they are
    # off: a slot holds an opening only while the parent is resident, and
    # there are four slots against dozens of copies. `copied` is the same
    # measurement against the copies on disk, which is the population a fork
    # would really be served from.
    measured = [r for r in choices if "copied" in r]
    from_copy = [r for r in measured
                 if (r.get("copied") or -1) > max(r.get("stored") or -1,
                                                  r.get("shared") or -1)]
    if not measured:
        # Not a zero. A row written before the router recorded this says
        # nothing either way, and reporting it as 0 of 111 is how a missing
        # measurement gets read as a finished one.
        print("a copy held more than anything saved: not recorded in these rows")
    else:
        print(f"a copy held more than anything saved or resident: "
              f"{len(from_copy)} of {len(measured)} measured")
    for r in from_copy[:5]:
        print(f"  {r.get('conv')} could have started at message "
              f"{(r.get('copied') or 0) + 1} from {r.get('copied_from')}, "
              f"saved reached {r.get('stored')}")

forks = [r for r in rows if r["event"] == "fork"]
if forks:
    part("forks")
    print(f"{len(forks)} branches off another session's opening, "
          f"{len({r.get('parent') for r in forks})} distinct parents")

builds = [r for r in rows if r["event"] == "build"]
loads = [r for r in rows if r["event"] == "load"]
part("openings")
if builds:
    for shelf in ("base", "deep"):
        shelf_rows = [r for r in builds if r.get("shelf") == shelf]
        ok_rows = [r for r in shelf_rows if r.get("ok")]
        secs = numbers(ok_rows, "secs")
        print(f"built {shelf:<4}  {len(ok_rows):>3} ok, {len(shelf_rows) - len(ok_rows):>2} failed"
              + (f", median {statistics.median(secs):.0f}s" if secs else ""))
if loads:
    for shelf in ("base", "deep"):
        shelf_rows = [r for r in loads if r.get("shelf") == shelf and r.get("ok")]
        secs = numbers(shelf_rows, "secs")
        mib = numbers(shelf_rows, "bytes")
        print(f"loaded {shelf:<4} {len(shelf_rows):>3} ok"
              + (f", median {statistics.median(secs):.1f}s" if secs else "")
              + (f", median {statistics.median(mib) / 2**20:.0f} MiB" if mib else ""))
used = Counter(r.get("key") for r in loads if r.get("ok"))
per = Counter((r.get("shelf"), r.get("key")) for r in builds if r.get("ok"))
waste = [(shelf, key, used.get(key, 0)) for (shelf, key) in per if used.get(key, 0) == 0]
if builds:
    print(f"built and never loaded: {len(waste)} of {len(per)}"
          + (": " + ", ".join(f"{s}/{k}" for s, k, _ in waste[:8]) if waste else ""))

wants = [r for r in rows if r["event"] == "want"]
if wants:
    part("wants")
    acts = Counter(r.get("action") for r in wants)
    print("  ".join(f"{k}={v}" for k, v in acts.most_common()))
    for act in ("built", "dropped"):
        ages = numbers([r for r in wants if r.get("action") == act], "age")
        if ages:
            print(f"age when {act:<8} median {statistics.median(ages):>6.0f}s, "
                  f"max {max(ages):>6.0f}s")

for event, label in (("park", "parked"), ("recall", "recalled"),
                     ("migrate", "migrated")):
    rows_e = [r for r in rows if r["event"] == event]
    if rows_e:
        ok_rows = [r for r in rows_e if r.get("ok", True)]
        secs = numbers(ok_rows, "secs")
        mib = numbers(ok_rows, "bytes")
        print(f"{label:<10} {len(rows_e):>3} ({len(rows_e) - len(ok_rows)} failed)"
              + (f", median {statistics.median(secs):.1f}s" if secs else "")
              + (f", median {statistics.median(mib) / 2**20:.0f} MiB" if mib else ""))

back = [r for r in rows if r["event"] == "backend"]
if back:
    part("backends said")
    kinds = defaultdict(float)
    for r in back:
        kinds[r.get("kind")] += 1
    ev_mib = sum(r.get("amount") or 0 for r in back if r.get("kind") == "evicted")
    line = "  ".join(f"{k}={int(v)}" for k, v in sorted(kinds.items()))
    print(line + f", evicted total {ev_mib / 1024:.1f} GiB")

usage = [r for r in rows if r["event"] == "usage"]
if usage:
    part("usage the client was told")
    # Two names for the same number: the anthropic splice writes `cache_read`
    # and the openai one `cached`. Reading only the first printed a heading
    # with nothing under it on a box whose traffic is all openai.
    read = numbers(usage, "cache_read") + numbers(usage, "cached")
    if read:
        print(f"{len(usage)} turns reported usage, {sum(read)} prompt tokens "
              f"read from the provider cache, {sum(1 for v in read if v)} turns of it")
    else:
        print(f"{len(usage)} turns reported usage, none of it cached")
