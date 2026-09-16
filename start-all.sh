#!/usr/bin/env bash
# Start every backend in the table, then the router on :8090, which is the
# address to send requests to. The table is BACKENDS in bin/common.sh: one
# instance on :8080 unless config.local.sh says otherwise.
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")" && pwd)}
# For MODELS and MODELS2, which say where the page cache has to be dropped.
source "$ROOT/bin/common.sh"   # sets RUN, among the rest
mkdir -p "$RUN"
# keep_log, alive, wait_for and start_router, shared with restart-router.sh.
source "$ROOT/bin/run-lib.sh"

for name in $(backend_names) router; do
  alive "$name" && { echo "$name is already running. Run ./stop-all.sh first." >&2; exit 1; }
done

# The kernel places a page on first read and never moves it, so stale cache
# would keep the weights on the wrong node.
drop_cache() {
  python3 - "$@" <<'PYEOF'
import glob, os, sys
count = 0
for pattern in sys.argv[1:]:
    for path in glob.glob(pattern):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        finally:
            os.close(fd)
print(f"  dropped cache for {count} model files")
PYEOF
}

start() { # start <name> <port> <script> [VAR=value ...]
  local name=$1 port=$2 script=$3; shift 3
  echo "starting $name..."
  local began=$SECONDS
  keep_log "$name"
  # Through `env`, so a row's settings reach this backend and no other. As
  # prefix assignments on a shell function they stayed set for every later
  # call, which is a trap the table would otherwise walk into.
  env PORT="$port" "$@" nohup "$ROOT/bin/$script" > "$RUN/$name.log" 2>&1 &
  echo $! > "$RUN/$name.pid"
  wait_for "$port" "$name" 2400 || return 1
  # Most of this is reading about 124 GiB to warm the page cache.
  echo "    took $((SECONDS - began))s"
}

echo "dropping stale page cache..."
drop_cache "$MODELS/*/*.gguf" "$MODELS2/*/*.gguf"

# One at a time: each backend reads about 124 GiB to warm its page cache. The
# order is the table's, in bin/common.sh - the name says the type, the socket,
# and which instance on that socket, and only the first instance on a socket
# primes, because the second maps a file the first has already read in.
while read -r name port script rest; do
  [[ -n $name ]] || continue
  # shellcheck disable=SC2086  # rest is a list of VAR=value words, by design
  start "$name" "$port" "$script" $rest
done < <(backend_rows)

echo "starting router..."
start_router

# `|| true` on both: under `set -e` a missing command here killed the script
# after every backend and the router had started, so the addresses below were
# never printed and a successful start exited 127. Neither is needed to serve.
lan=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
tailnet=$(tailscale ip -4 2>/dev/null | head -1 || true)

echo
echo "ready. Send requests to:"
if [[ -n $tailnet ]]; then
  echo "  http://$tailnet:$ROUTER_PORT   (tailscale)"
fi
echo "  http://${lan:-127.0.0.1}:$ROUTER_PORT   (lan)"
echo "  dashboard: http://${tailnet:-${lan:-127.0.0.1}}:$ROUTER_PORT/router"
