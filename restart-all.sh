#!/usr/bin/env bash
# Restart everything with no window where nothing serves.
#
#   ./restart-all.sh
#
# One backend at a time. The router stops sending work to it, waits for the
# work already running, copies its caches to disk, restarts it and puts it
# back in service before the next one is touched. Requests queue in the
# router meanwhile, so a client sees a slow reply rather than an error. A
# backend that is not running is started instead.
#
# Run it whatever is running. start-all.sh refuses when anything is up,
# because it drops the page cache for every model file first: the weights
# then land on whichever node reads them, and each backend reads about
# 124 GiB. This keeps that cache, so a backend is back in a minute.
#
# So use stop-all.sh and start-all.sh when NUMA placement is the thing you
# need to change, and this the rest of the time.
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")" && pwd)}
source "$ROOT/bin/common.sh"
source "$ROOT/bin/run-lib.sh"
mkdir -p "$RUN"

# A backend is rolled by asking the router to drain it, so the router has to
# be up before any of them move. Starting it here is also the restart it
# needed, so it is not restarted again at the end.
router_rolled=0
if ! alive router; then
  echo "the router is not running, so it starts first"
  start_router || exit 1
  router_rolled=1
fi

# Stop at the first backend that does not come back. restart-backend.sh
# leaves a backend it could not start drained, and carrying on from there
# would drain the next one too, and the one after that.
while read -r name port script rest; do
  [[ -n $name ]] || continue
  echo
  if alive "$name"; then
    echo "=== $name ==="
    "$ROOT/bin/restart-backend.sh" "$name" || {
      echo "$name did not come back. Nothing else was touched." >&2; exit 1; }
  else
    echo "=== $name, which was not running ==="
    # shellcheck disable=SC2086  # rest is a list of VAR=value words, by design
    start_backend "$name" "$port" "$script" $rest || {
      echo "$name did not start. Nothing else was touched." >&2; exit 1; }
  fi
done < <(backend_rows)

# Last, so every drain above went through the router that held those pins.
# It stops in seconds, and it parks what is live on the way out.
if (( ! router_rolled )); then
  echo
  echo "=== router ==="
  "$ROOT/bin/restart-router.sh" || exit 1
fi

echo
echo "everything is back. dashboard: http://127.0.0.1:$ROUTER_PORT/router"
