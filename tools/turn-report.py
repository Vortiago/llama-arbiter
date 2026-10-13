#!/usr/bin/env python3
"""What each turn took, read from the router's own event log.

The dashboard shows the present; this answers questions about the past from
run/cache-events-*.jsonl, per day, per build and per backend: the median, p90
and max of every number a `request` event carries, and how many events of each
kind there were.

    python3 tools/turn-report.py [--dir run]

No dependencies: it reads the JSON lines the router writes and computes its
percentiles with a sort.
"""
import argparse, json, os, pathlib, sys, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.environ.get("CACHE_LOG_DIR") or os.environ.get("RUN")
                or str(pathlib.Path(__file__).resolve().parent.parent / "run"))
a = ap.parse_args()

# The numbers a request event may carry. `first_token_s` is not written today:
# it is printed only where a row has it, so a later probe shows up without a
# change here.
FIELDS = ("took", "waited", "pre_ms", "tg", "first_token_s", "accept",
          "recall_s", "handoff_s", "gen_steps", "gen_draft_per_step_ms",
          "gen_decode_per_step_ms", "gen_sync_per_step_ms",
          "gen_spec_per_step_ms", "gen_emit_per_step_ms")


def percentile(values, fraction):
    """The value below this fraction of the sample, from a sort. Approximate
    nearest rank: a six-value sample has a real p90 and not one between two
    of them."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[rank]


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float) and value == int(value):
        return f"{value:.1f}"
    return f"{value:g}"


rows = []
for path in sorted(pathlib.Path(a.dir).glob("cache-events-*.jsonl")):
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        # One line may be a list or a scalar, whatever was written beside the
        # router's own. Only an object carries an event.
        if isinstance(row, dict):
            rows.append(row)

if not rows:
    print(f"no events in {a.dir}")
    sys.exit(0)

print(f"{len(rows)} events, "
      f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(min(r['ts'] for r in rows)))}"
      f" to {time.strftime('%Y-%m-%d %H:%M', time.localtime(max(r['ts'] for r in rows)))}")

print("\n== events ==")
for event, count in Counter(r.get("event") for r in rows).most_common():
    print(f"  {event or '?':<12} {count}")

# A failure is a copied file the backend refused, an opening that would not
# build, or a turn that started over. Each names its own reason.
fails = Counter()
for r in rows:
    if r.get("ok") is False:
        fails[r.get("event") + ": " + str(r.get("error") or "?")] += 1
    elif r.get("event") == "start_over":
        fails["start_over: " + str(r.get("reason") or "?")] += 1
if fails:
    print("\n== failures ==")
    for what, count in fails.most_common():
        print(f"  {count:>5}  {what}")

requests = [r for r in rows if r.get("event") == "request"]
if not requests:
    print("\nno request events")
    sys.exit(0)

# Day x generating backend, as the router recorded them. `backend`/`build`
# describe the reader and `serving`/`serving_build` the generator a carried
# turn moved to, so grouping by the generator files a carried turn under the
# backend that answered, not the one that read. A row from before either was
# written groups under "?".
groups = defaultdict(list)
for r in requests:
    day = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
    groups[(day, r.get("serving_build") or r.get("build") or "?",
            r.get("serving") or r.get("backend") or "?")].append(r)

print(f"\n== turns by day, generating backend and its build "
      f"({len(requests)} requests) ==")
for (day, build, backend), group in sorted(groups.items()):
    print(f"\n{day}  build {build}  {backend}  n={len(group)}")
    for field in FIELDS:
        values = [r[field] for r in group
                  if isinstance(r.get(field), (int, float))]
        if not values:
            continue
        print(f"  {field:<14} p50 {fmt(percentile(values, 0.5)):>9}"
              f"  p90 {fmt(percentile(values, 0.9)):>9}"
              f"  max {fmt(max(values)):>9}  n={len(values)}")
