# One-off tools

Run once, or not at all. `get-llama.sh` is the exception: it builds the
server the rest of this repository is a router for.

| file | purpose |
|---|---|
| `get-llama.sh` | Clone llama.cpp at the commit in `patches/llama-ref`, apply `patches/`, `patches/cpu/core/` and the optional CPU sets that `CPU_OPTIONAL` names, build `llama-server`. Safe to re-run. |
| `export-cpu-patches.sh` | Write `patches/cpu/` again from a llama.cpp branch: core, and one directory for each optional set. |
| `perf-ab.py` | Measure one llama.cpp build at a time for an A/B, and compare the results. |
| `setup-nvme.sh` | Partition and format the NVMe disk. Run once, as root. Erases the disk. |
| `numa-ab.sh`    | Compare NUMA settings. |
| `qwen.sh`       | The old non-MTP backend. Slower. Kept for reference. |
| `watch.sh`      | Follow every backend log at once, filtered to the timings. |
| `cache-report.py` | What the cache did, from `run/cache-events-*.jsonl`. |
| `prefill-rates.py` | Prefill tokens/s per backend, from the backends' own logs. |

## The CPU speed patches

`get-llama.sh` applies the core CPU patches by default. To add optional sets,
name them in `CPU_OPTIONAL`, separated by spaces:

    CPU_OPTIONAL="I1 Q1" tools/get-llama.sh

The script applies the sets in name order, whatever order `CPU_OPTIONAL` gives.
A name with no directory in `patches/cpu/optional/` stops the script. The
script also stops when the checkout holds an optional set that `CPU_OPTIONAL`
does not name. To drop that set, run `git -C llama.cpp-mtp checkout -- .`, and
run the script again. No patch adds a file, so that removes every patch.
`LLAMA_REF` works as before.

To write `patches/cpu/` again, give `export-cpu-patches.sh` the branch and one
`-o <name>=<pattern>` for each optional set:

    LLAMA=<llama.cpp checkout> tools/export-cpu-patches.sh \
        -o 'I1=sigmoid|dsv4_hc_pre' -o 'Q1=top-k|argsort' perf/stack4

The pattern is an extended regular expression. It matches the subject of each
commit between the base and the branch. Each commit that no pattern matches
goes to `core/`. The commits that a pattern matches go to `optional/<name>/`,
numbered from 0001. A test commit goes with the patch it tests, so the pattern
must match its subject too. That is the command for the patches in this
repository.

The branch may hold an optional commit before a core one. The script then
replays the commits in memory with `git merge-tree`: core first, and each
optional set on top of core alone. It writes no ref and does not change the
llama.cpp checkout. Before it writes `patches/cpu/`, it checks four things:

- The base is `patches/llama-ref` plus the server patches. The default base is
  `perf/base`, and a second argument names another one.
- Core applies in order and gives the tree it was replayed to.
- Each combination of the optional sets applies on top of core, in name order.
- Core plus every optional set gives the tree of the branch.
