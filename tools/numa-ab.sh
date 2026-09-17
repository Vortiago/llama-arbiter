#!/usr/bin/env bash
# Compare NUMA policies for llama-server decode speed.
# Each config is started, measured and fully stopped before the next begins.
set -uo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
[[ -f $ROOT/config.local.sh ]] && source "$ROOT/config.local.sh"

MODEL=${MODEL:-${MODELS:-$ROOT/models}/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
SERVER=${SERVER:-$ROOT/llama.cpp-mtp/build/bin/llama-server}
# Not 8080: that is the production backend's port, and stop_server kills
# whatever listens on PORT. tests/live/harness.py skips 8080-8083 and 8090
# for the same reason.
PORT=${PORT:-18080}
LOG_DIR=${LOG_DIR:-/tmp/numa-ab}
started=
LOAD_TIMEOUT=600
STOP_TIMEOUT=120

SERVER_FLAGS=(--model "$MODEL" --n-gpu-layers 99 --load-mode none --n-cpu-moe 46
              --numa numactl --ctx-size 32768 --flash-attn on --parallel 1
              --host 127.0.0.1 --port "$PORT")

# ignore_eos forces exactly n_predict tokens: decode is always measured.
# cache_prompt false makes every config redo the same prefill work.
REQUEST=$(python3 -c "import json; print(json.dumps(
  {'prompt': 'The quick brown fox jumps over the lazy dog. ' * 200,
   'n_predict': 200, 'ignore_eos': True, 'cache_prompt': False}))")

# Ask the socket who listens. A pattern match on the binary path hits every
# llama-server on the box, and the production backends run the same binary.
server_pid()     { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1; }
server_running() { local p; p=$(server_pid); [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
server_ready()   { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
port_busy()      { ss -ltn 2>/dev/null | grep -q ":$PORT "; }

# Signal only the server this script started: `started` is the pid the launch
# handed back. A cmdline check like `ours` in bin/run-lib.sh cannot tell this
# server from a production backend.
stop_server() {
  local pid deadline=$((SECONDS + STOP_TIMEOUT))
  pid=$(server_pid)
  if [[ -n $pid && ${started:-} != "$pid" ]]; then
    echo "port $PORT is held by pid $pid, which this script did not start." >&2
    echo "Set PORT to a free one, or stop that process yourself." >&2
    exit 1
  fi
  [[ -n $pid ]] && kill "$pid" 2>/dev/null
  while server_running && ((SECONDS < deadline)); do sleep 2; done
  pid=$(server_pid)
  [[ -n $pid && ${started:-} == "$pid" ]] && kill -9 "$pid" 2>/dev/null
  while port_busy && ((SECONDS < deadline + 30)); do sleep 2; done
  started=
}

# 0 = up and healthy, 1 = died or timed out
wait_for_server() {
  local deadline=$((SECONDS + LOAD_TIMEOUT))
  while ((SECONDS < deadline)); do
    server_ready && return 0
    server_running || return 1
    printf '.'; sleep 5
  done
  return 1
}

gpu_numa_node() {
  local bdf
  bdf=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1)
  bdf=$(tr 'A-Z' 'a-z' <<<"${bdf: -12}")
  cat "/sys/bus/pci/devices/$bdf/numa_node"
}

# stdin: /completion json -> "decode_tps decode_tokens prefill_tps prefill_tokens"
parse_timings() {
  python3 -c '
import json, sys
t = json.load(sys.stdin).get("timings") or {}
print(t.get("predicted_per_second", 0), t.get("predicted_n", 0),
      t.get("prompt_per_second", 0),    t.get("prompt_n", 0))'
}

measure() {  # measure <label> <threads> <numactl args...>
  local label=$1 threads=$2; shift 2

  stop_server
  printf '\n==> %s: %s --threads %s\n' "$label" "$*" "$threads"
  "$@" -- "$SERVER" "${SERVER_FLAGS[@]}" --threads "$threads" \
    >"$LOG_DIR/$label.log" 2>&1 &
  started=$!        # the only pid stop_server will ever signal

  if ! wait_for_server; then
    printf '%-20s FAILED, see %s/%s.log\n' "$label" "$LOG_DIR" "$label" | tee -a "$RESULTS"
    stop_server
    return
  fi

  local decode_tps decode_n prefill_tps prefill_n
  read -r decode_tps decode_n prefill_tps prefill_n < <(
    curl -s "http://127.0.0.1:$PORT/completion" \
         -H 'Content-Type: application/json' -d "$REQUEST" | parse_timings)

  printf '%-20s decode %6.2f tok/s (%s tok)   prefill %7.2f tok/s (%s tok)\n' \
    "$label" "$decode_tps" "$decode_n" "$prefill_tps" "$prefill_n" | tee -a "$RESULTS"
  stop_server
}

mkdir -p "$LOG_DIR"
RESULTS=$LOG_DIR/results.txt
: >"$RESULTS"

node=$(gpu_numa_node)
echo "gpu is on numa node $node"

measure A-interleave-t36 36 numactl --interleave=all
measure B-cpubind-t18    18 numactl --cpunodebind="$node"
measure C-onesocket-t18  18 numactl --cpunodebind="$node" --membind="$node"

printf '\n=== results ===\n'
cat "$RESULTS"
