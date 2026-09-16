#!/usr/bin/env python3
"""Prefill rate per backend, from what the backends themselves reported.

Reads every "prompt eval time" line in each backend log and reports the median
and the spread. Use it to compare two settings running side by side.

    python3 tools/prefill-rates.py [--least 500] [--dir run]
"""
import argparse, os, pathlib, re, statistics

LINE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.\d+ .*prompt eval time =\s*"
                  r"([\d.]+) ms /\s*(\d+) tokens")
BATCH = re.compile(r"n_batch\s*=\s*(\d+)")

def rates(path, least):
    out, batch = [], None
    for line in path.read_text(errors="replace").splitlines():
        if batch is None:
            found = BATCH.search(line)
            if found:
                batch = int(found.group(1))
        found = LINE.match(line)
        if found:
            ms, tokens = float(found.group(4)), int(found.group(5))
            if tokens >= least and ms > 0:
                out.append(tokens / (ms / 1000))
    return batch, out

ap = argparse.ArgumentParser()
ap.add_argument("--least", type=int, default=500,
                help="ignore reads shorter than this many tokens")
# Named, as cache-report.py names it: the logs need not be in the checkout.
ap.add_argument("--dir", default=os.environ.get("RUN")
                or str(pathlib.Path(__file__).resolve().parent.parent / "run"),
                help="where the backend logs are")
args = ap.parse_args()
RUN = pathlib.Path(args.dir)

print(f"{'backend':10} {'n_batch':>8} {'reads':>6} {'median':>8} {'p10':>7} "
      f"{'p90':>7}   tokens per second")
for path in sorted(RUN.glob("*.log")):
    if path.stem == "router":
        continue
    batch, seen = rates(path, args.least)
    if not seen:
        continue
    seen.sort()
    p10 = seen[len(seen) // 10]
    p90 = seen[min(len(seen) - 1, len(seen) * 9 // 10)]
    print(f"{path.stem:10} {batch or '?':>8} {len(seen):>6} "
          f"{statistics.median(seen):>8.1f} {p10:>7.1f} {p90:>7.1f}")
