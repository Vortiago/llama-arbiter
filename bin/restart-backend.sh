#!/usr/bin/env bash
# Restart one backend without dropping the requests waiting for it.
#
#   bin/restart-backend.sh cpu
#
# The router stops sending it work, waits for what is running to finish, and
# copies its caches to disk. Requests wait in the router while it is away, so a
# client sees a slow reply rather than an error.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# For the backend table, RUN and config.local.sh, which common.sh sources first.
source "$ROOT/bin/common.sh"
# For `ours`, `keep_log` and `pid_on_port`, shared with the other stop paths.
source "$ROOT/bin/run-lib.sh"
# ROUTER names the whole address; ROUTER_PORT is what the launch scripts use.
ROUTER=${ROUTER:-http://127.0.0.1:${ROUTER_PORT:-8090}}

name=${1:-}
# From the one table in bin/common.sh, which start-all.sh starts from too. This
# was a case statement restating it, so a machine with other backends had them
# started by one script and refused by this one.
row=$(backend_row "$name")
if [[ -z $row ]]; then
  echo "usage: $0 $(backend_names | paste -sd'|' -)" >&2
  exit 2
fi
read -r _ port script rest <<<"$row"
# shellcheck disable=SC2206  # rest is a list of VAR=value words, by design
env=($rest)

# A setting in the environment wins over the table above, because `env` takes
# the last assignment of a name. That is how a sweep restarts one backend with
# a single knob moved.
for name_of in BATCH UBATCH TB SLOTS PRIME CACHE_RAM; do
  [[ -n ${!name_of:-} ]] && env+=("$name_of=${!name_of}")
done

echo "draining $name (this waits for work already running)..."
# mktemp, not /tmp/drain.$$: a pid is a 32k space anyone can watch, and curl -o
# follows a symlink already sitting there, so the router's reply lands on
# whatever that link points at and the rm takes the link away.
answer=$(mktemp) || die "no temporary file"
trap 'rm -f "$answer"' EXIT
code=$(curl -s -o "$answer" -w '%{http_code}' -X POST "$ROUTER/router/drain/$name")
cat "$answer"; echo
if [[ $code != 200 ]]; then
  echo "not drained, leaving $name alone" >&2
  curl -s -X POST "$ROUTER/router/resume/$name" >/dev/null
  exit 1
fi

# Ask the socket who is listening: a pid file goes stale the moment anyone
# starts a backend by hand, and killing a stale pid leaves the old one up.
pid=$(pid_on_port "$port")
[[ -z ${pid:-} ]] && pid=$(cat "$RUN/$name.pid" 2>/dev/null || true)
if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null && ! ours "$pid"; then
  echo "$name: pid $pid is something else now, leaving it alone" >&2
  curl -s -X POST "$ROUTER/router/resume/$name" >/dev/null
  exit 1
fi
if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null; then
  echo "stopping $name..."
  kill "$pid"
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  # Both checks again before SIGKILL, as stop-all.sh and restart-router.sh do.
  # The loop breaks the moment the backend exits, and Linux reuses pids while
  # llama-server is still releasing 124 GiB, so the pid here may be somebody
  # else's by now.
  kill -0 "$pid" 2>/dev/null && ours "$pid" && kill -9 "$pid" 2>/dev/null
fi

echo "starting $name..."
keep_log "$name"    # the last run's log is what a restart is asking about
PORT=$port env "${env[@]}" nohup "$ROOT/bin/$script" > "$RUN/$name.log" 2>&1 &
started=$!
echo $started > "$RUN/$name.pid"
sleep 2
# Watch the process as well as the port, the way start-all.sh's wait_for does.
# Without this a backend that died at load - a bad flag, an OOM - was waited on
# for forty minutes against a closed port and then resumed anyway, and the
# router started handing conversations to an instance that was not there.
up=0
for _ in $(seq 480); do
  curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { up=1; break; }
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

curl -s -X POST "$ROUTER/router/resume/$name" >/dev/null
echo "$name is back in service"
