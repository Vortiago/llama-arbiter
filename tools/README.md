# One-off tools

Not needed to run the server.

| file | purpose |
|---|---|
| `setup-nvme.sh` | Partition and format the NVMe disk. Run once, as root. Erases the disk. |
| `numa-ab.sh`    | Compare NUMA settings. |
| `qwen.sh`       | The old non-MTP backend. Slower. Kept for reference. |
| `watch.sh`      | Follow every backend log at once, filtered to the timings. |
| `cache-report.py` | What the cache did, from `run/cache-events-*.jsonl`. |
| `prefill-rates.py` | Prefill tokens/s per backend, from the backends' own logs. |
