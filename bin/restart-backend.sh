#!/usr/bin/env bash
# Restart one backend without dropping the requests waiting for it.
#
#   bin/restart-backend.sh cpu
#
# The router stops sending it work, waits for running work to finish, and
# copies its caches to disk. Requests wait in the router. A client sees a slow
# reply, not an error.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/bin/common.sh"
source "$ROOT/bin/run-lib.sh"
ROUTER=${ROUTER:-http://127.0.0.1:${ROUTER_PORT:-8090}}

name=${1:-}
row=$(backend_row "$name")
if [[ -z $row ]]; then
  echo "usage: $0 $(backend_names | paste -sd'|' -)" >&2
  exit 2
fi
read -r _ port script rest <<<"$row"
# shellcheck disable=SC2206  # rest is a list of VAR=value words, by design
env=($rest)

# A setting in the environment wins over the table: `env` takes the last
# assignment of a name. A sweep restarts one backend with one knob moved.
for name_of in BATCH UBATCH TB SLOTS PRIME CACHE_RAM CPUSET CKPT_STEP CKPT_N PRIO N_CPU_MOE CPU_MOE_DRAFT; do
  [[ -n ${!name_of:-} ]] && env+=("$name_of=${!name_of}")
done
# THREADS_ENV, not THREADS: common.sh fills THREADS with the core count, which
# would override every THREADS in the table.
[[ -n $THREADS_ENV ]] && env+=("THREADS=$THREADS_ENV")

echo "draining $name (this waits for work already running)..."
# mktemp, not /tmp/drain.$$: curl -o follows a symlink planted at a guessable
# name, and the reply then lands wherever that link points.
answer=$(mktemp) || die "no temporary file"
trap 'rm -f "$answer"' EXIT
code=$(curl -s -o "$answer" -w '%{http_code}' \
  -H "X-Router-Key: ${ROUTER_CONTROL_KEY:-}" \
  -X POST "$ROUTER/router/drain/$name")
cat "$answer"; echo
if [[ $code != 200 ]]; then
  echo "not drained, leaving $name alone" >&2
  curl -s -H "X-Router-Key: ${ROUTER_CONTROL_KEY:-}" \
    -X POST "$ROUTER/router/resume/$name" >/dev/null
  exit 1
fi

# Ask the socket, not the pid file. The file goes stale when anyone starts a
# backend by hand.
pid=$(pid_on_port "$port")
[[ -z ${pid:-} ]] && pid=$(cat "$RUN/$name.pid" 2>/dev/null || true)
if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null && ! ours "$pid"; then
  echo "$name: pid $pid is something else now, leaving it alone" >&2
  curl -s -H "X-Router-Key: ${ROUTER_CONTROL_KEY:-}" \
    -X POST "$ROUTER/router/resume/$name" >/dev/null
  exit 1
fi
if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null; then
  echo "stopping $name..."
  kill "$pid"
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  # Check `ours` again before SIGKILL. Linux reuses pids while llama-server is
  # still releasing 124 GiB.
  kill -0 "$pid" 2>/dev/null && ours "$pid" && kill -9 "$pid" 2>/dev/null
fi

# Put the model pages on the node that will read them. The gpu backend's
# --preferred=0 puts a page it faults in the mirrored copy on node 0, so a full
# start-all places each copy first (place_copies). Do the same for the copy this
# backend reads. Set PLACE_COPIES=0 to skip.
if [[ ${PLACE_COPIES:-1} == 1 ]]; then
  read -ra place_nodes <<<"$(printf '%s\n' $rest | sed -n 's/^NODE=//p' | tail -1 | tr ',' ' ')"
  # A row with no NODE is no instruction to place anything. place_copies with
  # no node argument reads every node with a copy, which for one backend's
  # restart is both copies. A row that wants both says NODE=0,1.
  (( ${#place_nodes[@]} )) && place_copies "${place_nodes[@]}"
fi

echo "starting $name..."
keep_log "$name"
PORT=$port env "${env[@]}" nohup "$ROOT/bin/$script" > "$RUN/$name.log" 2>&1 &
started=$!
echo $started > "$RUN/$name.pid"
sleep 2
# Watch the process as well as the port. A backend that dies at load must not
# be waited on for forty minutes and then resumed.
up=0
for _ in $(seq 480); do
  # -f: a loading llama-server answers /health with 503, and plain -s
  # took that for up. A backend that then died at load was put in service.
  curl -sf -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$started" 2>/dev/null || break
  sleep 5
done
if [[ $up != 1 ]]; then
  echo "$name did not come up. See $RUN/$name.log" >&2
  echo "it is still drained, so nothing is being sent to it" >&2
  exit 1
fi

if [[ ${KEEP_DRAINED:-0} == 1 ]]; then
  echo "$name is up but still drained, resume it yourself"
  exit 0
fi

curl -s -H "X-Router-Key: ${ROUTER_CONTROL_KEY:-}" \
  -X POST "$ROUTER/router/resume/$name" >/dev/null
echo "$name is back in service"
