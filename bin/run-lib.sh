#!/usr/bin/env bash
# Process control shared by the start and restart scripts. Source it after
# bin/common.sh.

RUN=${RUN:-$ROOT/run}
ROUTER_PORT=${ROUTER_PORT:-8090}

# Keep the last run's log. A restart is when it is needed, and the tools that
# read these logs read whole files.
keep_log() { [[ -f $RUN/$1.log ]] && mv -f "$RUN/$1.log" "$RUN/$1.log.prev"; return 0; }

alive() { [[ -f $RUN/$1.pid ]] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

# Is this pid still what we started? A pid file goes stale when anyone starts
# a server by hand, and Linux reuses pids.
ours() {
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
  [[ $cmd == *llama-server* || $cmd == *-m\ router* || $cmd == *qwen-mtp* ]]
}

# Who holds a port, from the kernel rather than a file.
pid_on_port() {
  ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1
}

wait_for() { # wait_for <port> <name> <seconds> [path]
  local port=$1 name=$2 limit=$3 path=${4:-/health} waited=0
  while (( waited < limit )); do
    alive "$name" || { echo "$name stopped. See $RUN/$name.log" >&2; return 1; }
    curl -sf "http://127.0.0.1:$port$path" >/dev/null 2>&1 && { echo "  $name ready on :$port"; return 0; }
    sleep 3; waited=$((waited + 3))
  done
  echo "$name did not start within ${limit}s" >&2; return 1
}

# The one place a backend starts. The trailing arguments are the VAR=value
# words from its row in the BACKENDS table.
start_backend() { # start_backend <name> <port> <script> [VAR=value ...]
  local name=$1 port=$2 script=$3; shift 3
  echo "starting $name..."
  local began=$SECONDS
  keep_log "$name"
  # `env`, not prefix assignments: a prefix assignment on a shell function
  # stays set for every later call.
  env PORT="$port" "$@" nohup "$ROOT/bin/$script" > "$RUN/$name.log" 2>&1 &
  echo $! > "$RUN/$name.pid"
  wait_for "$port" "$name" 2400 || return 1
  echo "    took $((SECONDS - began))s"
}

# The one place the router starts. Source common.sh first: it reads
# config.local.sh, which exports ROUTER_BACKENDS.
start_router() {
  keep_log router
  # -m router, not a file: bin/router is a package now. PYTHONPATH names the
  # directory it sits in.
  PYTHONPATH="$ROOT/bin" nohup python3 -m router --host "${ROUTER_HOST:-::}" \
        --port "$ROUTER_PORT" > "$RUN/router.log" 2>&1 &
  echo $! > "$RUN/router.pid"
  wait_for "$ROUTER_PORT" router 60 /router/json
}
