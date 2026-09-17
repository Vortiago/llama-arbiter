#!/usr/bin/env bash
# Show what the backends are doing, live.  ./tools/watch.sh
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
RUN=${RUN:-$ROOT/run}
shopt -s nullglob
# Log names come from the backend table, not from a name pattern.
source "$ROOT/bin/common.sh"
logs=()
while read -r name; do
  [[ -n $name && -f $RUN/$name.log ]] && logs+=("$RUN/$name.log")
done < <(backend_names)
(( ${#logs[@]} )) || { echo "no backend logs in $RUN"; exit 1; }
tail -f "${logs[@]}" | grep --line-buffered -E \
  "prompt processing|n_gen =|prompt eval time|eval time|draft acceptance|==>" |
  sed -u -e 's/^[0-9.]* I slot [a-z_]*: //' -e 's/==> //'
