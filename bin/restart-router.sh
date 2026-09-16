#!/usr/bin/env bash
# Restart the router, leaving the backends and their warm page cache alone.
#
#   bin/restart-router.sh
#
# Use this rather than killing the pid and relaunching by hand. The router
# learns its backend table from config.local.sh, which it only sees through
# bin/common.sh, and one started without that silently falls back to the
# built-in single-backend default while looking perfectly healthy.
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
  # 90s, the same as stop-all.sh gives it: copying the caches out takes longer
  # than dying, and killing early is a whole prompt re-read per conversation.
  for _ in $(seq 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "  it did not stop in 90s, forcing it" >&2
    ours "$pid" && kill -9 "$pid" 2>/dev/null
  fi
fi

echo "starting the router..."
start_router || exit 1
grep -a 'backend(s) from' "$RUN/router.log" | tail -1
