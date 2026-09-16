#!/usr/bin/env bash
# Show what the backends are doing, live.  ./tools/watch.sh
# One log per backend instance. This used to name gpu.log and cpu.log, which
# nothing has written since the backends were split per socket, so it tailed
# four missing files and printed nothing at all.
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
RUN=${RUN:-$ROOT/run}
shopt -s nullglob
# From the backend table, not from a name pattern: the shipped default names
# its one instance `solo`, so globbing gpu*/cpu* found nothing on a fresh
# checkout and this exited 1 while the backend was up and logging.
source "$ROOT/bin/common.sh"
logs=()
while read -r name; do
  [[ -n $name && -f $RUN/$name.log ]] && logs+=("$RUN/$name.log")
done < <(backend_names)
(( ${#logs[@]} )) || { echo "no backend logs in $RUN"; exit 1; }
tail -f "${logs[@]}" | grep --line-buffered -E \
  "prompt processing|n_gen =|prompt eval time|eval time|draft acceptance|==>" |
  sed -u -e 's/^[0-9.]* I slot [a-z_]*: //' -e 's/==> //'
