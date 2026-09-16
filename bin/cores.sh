#!/usr/bin/env bash
# cores_on_node <node> -- physical cores on one NUMA node.
#
# Counted rather than written down, so the tree runs anywhere. A thread per
# hyperthread would put two threads on one core, which is slower than one.
# Falls back to nproc: right on a single-node machine, too high on any other,
# so set THREADS where there is no lscpu.
cores_on_node() {
  local count
  count=$(lscpu -p=Core,Node 2>/dev/null |
          awk -F, -v n="$1" '!/^#/ && $2 == n { seen[$1] = 1 } END { print length(seen) }')
  (( ${count:-0} > 0 )) && echo "$count" || nproc
}
