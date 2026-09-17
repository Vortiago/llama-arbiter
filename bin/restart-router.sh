#!/usr/bin/env bash
# Restart the router. The backends and their warm page cache stay up.
#
#   bin/restart-router.sh
#
# Do not relaunch the router by hand. It reads its backend table from
# config.local.sh through bin/common.sh. Without that it silently falls back
# to the single-backend default and looks healthy.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/bin/common.sh"
source "$ROOT/bin/run-lib.sh"
mkdir -p "$RUN"

pid=$(pid_on_port "$ROUTER_PORT")
[[ -z ${pid:-} ]] && pid=$(cat "$RUN/router.pid" 2>/dev/null || true)

if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null; then
  if ! ours "$pid"; then
    echo "pid $pid on :$ROUTER_PORT is not the router. Leaving it alone." >&2
    exit 1
  fi
  echo "stopping the router (it parks every live cache on the way out)..."
  kill "$pid"
  # 90 s, as in stop-all.sh: copying the caches out takes longer than an
  # exit. An early kill costs a full prompt re-read per conversation.
  for _ in $(seq 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "  it did not stop in 90s, forcing it" >&2
    ours "$pid" && kill -9 "$pid" 2>/dev/null
  fi
fi

echo "starting the router..."
start_router || exit 1
grep -a 'backend(s) from' "$RUN/router.log" | tail -1
