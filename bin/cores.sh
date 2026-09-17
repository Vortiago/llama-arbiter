#!/usr/bin/env bash
# cores_on_node <node> -- physical cores on one NUMA node.
#
# One thread per hyperthread puts two threads on one core, which is slower
# than one. Without lscpu this falls back to nproc: right on a single-node
# machine, too high on any other. Set THREADS there.
cores_on_node() {
  local count
  count=$(lscpu -p=Core,Node 2>/dev/null |
          awk -F, -v n="$1" '!/^#/ && $2 == n { seen[$1] = 1 } END { print length(seen) }')
  (( ${count:-0} > 0 )) && echo "$count" || nproc
}
