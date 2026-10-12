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
  # directory it sits in, in front of whatever the operator already set:
  # `python3 bin/router.py` inherited that, and overwriting it here took it
  # away from the router alone, while the backends kept it.
  PYTHONPATH="$ROOT/bin${PYTHONPATH:+:$PYTHONPATH}" \
        nohup python3 -m router --host "${ROUTER_HOST:-::}" \
        --port "$ROUTER_PORT" > "$RUN/router.log" 2>&1 &
  echo $! > "$RUN/router.pid"
  wait_for "$ROUTER_PORT" router 60 /router/json
}

# warm_moe_split <port>: make a backend pin its expert memory now, not on the
# first real turn.
#
# The prefill split (LLAMA_MOE_SPLIT) only engages once one prompt reads that
# many tokens or more, so this reads a prompt longer than the threshold. The
# first split of a fresh process registers the model's expert pages as pinned
# (cudaHostRegister through GGML_CUDA_REGISTER_HOST), which for this model takes
# about a minute; paying it here keeps it off the first turn a client sends.
# The slot is erased afterwards, so the router finds a backend with an empty
# slot. Best effort: if the call fails, the first prefill pins instead.
warm_moe_split() { # warm_moe_split <port>
  local port=$1 began=$SECONDS
  local body='{"prompt":"The router keeps every conversation warm, so the card computes the experts that many tokens route to while the sockets read and compute the rest. This line is deliberately long enough that the prefill engages the expert split, because the split only begins once one prompt reads at least as many tokens as the threshold names. The first split of a fresh process registers the host memory of the model as pinned once, and that registration takes about a minute for a model of this size, so paying it here at startup keeps it off the first real turn that a client sends.","n_predict":1,"cache_prompt":false,"temperature":0}'
  echo "warming the experts on :$port (one-time pin; this can take a minute)..."
  if ! curl -sf --max-time 300 -H 'Content-Type: application/json' -d "$body" \
       "http://127.0.0.1:$port/completion" >/dev/null; then
    echo "  :$port did not answer the warmup; the first prompt will pin instead" >&2
    return 0
  fi
  # Leave the slot as the router expects to find a freshly started backend.
  curl -sf --max-time 30 -X POST "http://127.0.0.1:$port/slots/0?action=erase" >/dev/null 2>&1 || true
  echo "    took $((SECONDS - began))s"
}

# place_copies [node ...]: read each model copy once, bound to its own node, so
# the kernel faults its pages there.
#
# A page lives on the node that faults it first, and the kernel never moves it.
# The gpu backend maps the other node's copy (LLAMA_NUMA_MIRROR) so its node-1
# threads read locally, but it runs with --preferred=0: a page it faults in that
# copy lands on node 0, not node 1. Reading the copy here first puts the pages
# where they belong; the backends' own prime then finds them. With no node
# argument every node with a copy is read. Set PLACE_COPIES=0 to skip.
place_copies() { # place_copies [node ...]
  command -v numactl >/dev/null || return 0
  local -a nodes=("$@")
  (( ${#nodes[@]} )) || nodes=(0 1)
  local node dir f pid
  local -a pids=()
  for node in "${nodes[@]}"; do
    [[ -d /sys/devices/system/node/node$node ]] || continue
    dir=$MODELS; [[ $node == 1 ]] && dir=$MODELS2
    [[ -d $dir ]] || continue
    [[ $node == 1 && $MODELS2 == $MODELS ]] && continue
    local -a keep=()
    shopt -s nullglob
    for f in "$dir"/*/*.gguf; do
      # shellcheck disable=SC2053  # PRIME_SKIP is a pattern, by design
      [[ $f == ${PRIME_SKIP:-'*00003-of-00006.gguf'} ]] || keep+=("$f")
    done
    shopt -u nullglob
    (( ${#keep[@]} )) || continue
    echo "placing $dir on node $node (${#keep[@]} files)..."
    # One reader per node, in parallel. The copies live in separate memory
    # banks, so the placement is the same; they may share the disk, in which
    # case the gain is only what a single reader leaves on the table.
    numactl --cpunodebind="$node" --membind="$node" -- \
      cat -- "${keep[@]}" >/dev/null &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
}
