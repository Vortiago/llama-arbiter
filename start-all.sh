#!/usr/bin/env bash
# Start every backend in BACKENDS (bin/common.sh), then the router on :8090.
# Send requests to the router.
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")" && pwd)}
source "$ROOT/bin/common.sh"
mkdir -p "$RUN"
source "$ROOT/bin/run-lib.sh"

for name in $(backend_names) router; do
  alive "$name" && { echo "$name is already running. Run ./stop-all.sh first." >&2; exit 1; }
done

# The kernel places a page on first read and never moves it. A stale cache
# keeps the weights on the wrong node.
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

echo "dropping stale page cache..."
drop_cache "$MODELS/*/*.gguf" "$MODELS2/*/*.gguf"

# One at a time, in table order. Each backend reads about 124 GiB to warm its
# page cache. A second instance on the same socket needs no prime: it maps the
# file the first one has already read.
while read -r name port script rest; do
  [[ -n $name ]] || continue
  # shellcheck disable=SC2086  # rest is a list of VAR=value words, by design
  start_backend "$name" "$port" "$script" $rest
done < <(backend_rows)

echo "starting router..."
start_router

# `|| true`: neither command is needed to serve, and under `set -e` a missing
# one must not end the script here.
lan=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
tailnet=$(tailscale ip -4 2>/dev/null | head -1 || true)

echo
echo "ready. Send requests to:"
if [[ -n $tailnet ]]; then
  echo "  http://$tailnet:$ROUTER_PORT   (tailscale)"
fi
echo "  http://${lan:-127.0.0.1}:$ROUTER_PORT   (lan)"
echo "  dashboard: http://${tailnet:-${lan:-127.0.0.1}}:$ROUTER_PORT/router"
