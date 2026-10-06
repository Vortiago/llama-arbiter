# One-off tools

Run once, or not at all. `get-llama.sh` is the exception: it builds the
server the rest of this repository is a router for.

| file | purpose |
|---|---|
| `get-llama.sh` | Clone llama.cpp at the commit in `patches/llama-ref`, apply `patches/` and then `patches/cpu/`, build `llama-server`. Safe to re-run. |
| `export-cpu-patches.sh` | Write `patches/cpu/` again from a llama.cpp branch, one patch per commit. |
| `perf-ab.py` | Measure one llama.cpp build at a time for an A/B, and compare the results. |
| `setup-nvme.sh` | Partition and format the NVMe disk. Run once, as root. Erases the disk. |
| `numa-ab.sh`    | Compare NUMA settings. |
| `qwen.sh`       | The old non-MTP backend. Slower. Kept for reference. |
| `watch.sh`      | Follow every backend log at once, filtered to the timings. |
| `cache-report.py` | What the cache did, from `run/cache-events-*.jsonl`. |
| `prefill-rates.py` | Prefill tokens/s per backend, from the backends' own logs. |
