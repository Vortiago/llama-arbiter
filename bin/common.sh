#!/usr/bin/env bash
# Shared settings for the launch scripts. A script sets MODEL, SERVER, ALIAS
# and ARGS, then calls launch. ARGS comes last: a flag there overrides a
# default here. LAYOUT.md has the measurements behind these choices.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Per-machine settings. Not tracked. See config.local.example.sh.
[[ -f $ROOT/config.local.sh ]] && source "$ROOT/config.local.sh"

# Everything a run writes: logs, pids, saved slots, cache events. The router
# reads RUN too, so config.local.sh must export it.
RUN=${RUN:-$ROOT/run}

# One copy of the weights per NUMA node, as separate files. Two processes that
# map the same file share one page cache. That cache stays on the node that
# faulted it first, and the other node reads every expert over the
# interconnect. A hard link or a reflink shares the same pages.
#
#   MODELS=/mnt/nvme/models MODELS2=/mnt/nvme/models-node1 ./start-all.sh
MODELS=${MODELS:-$ROOT/models}
MODELS2=${MODELS2:-$MODELS}

# Override these in config.local.sh for a model with other filenames.
MODEL_Q8=${MODEL_Q8:-$MODELS/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
MODEL_Q6=${MODEL_Q6:-$MODELS/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf}
DRAFT=${DRAFT:-$MODELS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}

MODEL2_Q8=${MODEL2_Q8:-$MODELS2/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
DRAFT2=${DRAFT2:-$MODELS2/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}

# --no-mmproj-offload keeps the image encoder in RAM. The gpu backend has under
# 1 GiB of VRAM spare at ctx 150000, and the encoder runs once per image.
# VISION=0 starts without it.
MMPROJ=${MMPROJ:-$MODELS/mmproj-F16.gguf}
MMPROJ2=${MMPROJ2:-$MODELS2/mmproj-F16.gguf}

SERVER_MAIN=${SERVER_MAIN:-$ROOT/qwen/llama.cpp/build/bin/llama-server}
SERVER_MTP=${SERVER_MTP:-$ROOT/llama.cpp-mtp/build/bin/llama-server}

PORT=${PORT:-8080}
NODE=${NODE:-0}        # 0 is the node with the GPU

# The backends this machine starts, one "name port script [VAR=value…]" a
# line. start-all.sh starts them in this order, and bin/restart-backend.sh
# restarts one by name. Set BACKENDS in config.local.sh for more than one, and
# ROUTER_BACKENDS to give the router the same set. The two must agree.
BACKENDS=${BACKENDS:-'
solo 8080 qwen-mtp.sh
'}

backend_rows() {
  printf '%s\n' "$BACKENDS" | sed -e 's/#.*//' -e '/^[[:space:]]*$/d'
}

backend_names() { backend_rows | awk '{print $1}'; }

backend_row() { backend_rows | awk -v n="$1" '$1 == n'; }

source "$ROOT/bin/cores.sh"
# THREADS_ENV keeps the caller's THREADS. A script that moves NODE after this
# (qwen-mtp-cpu.sh) recounts from it.
THREADS_ENV=${THREADS:-}
THREADS=${THREADS:-$(cores_on_node "$NODE")}

# --preferred, not --membind: spill to the other node instead of failing.
NUMACTL=(--cpunodebind="$NODE" --preferred="$NODE")

SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0)
MTP_ARGS=(--model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3)

# vision_args <mmproj-file> -- empty if the file is absent or VISION=0
vision_args() {
  [[ ${VISION:-1} == 1 && -f $1 ]] && printf '%s\n' --mmproj "$1" --no-mmproj-offload
}

# --agent gives any client shell and file access, and there is no API key.
# The backends listen on localhost. Only the router is public.
COMMON=(
  --numa numactl
  --parallel 1
  --flash-attn auto
  --jinja
  --agent
  --metrics            # the router reads these for its dashboard
  --cache-ram "${CACHE_RAM:-0}"
                       # llama.cpp's own RAM copy of idle conversations. Off:
                       # the router parks every slot to disk instead. Two days
                       # of logs: 5 hits in 394 reads on the cpu instances,
                       # against 39 s of "updating prompt cache" pauses. It
                       # costs its full size in anonymous memory.
  --slot-save-path "$RUN/slots/"   # the router reads the same directory, so
                       # RUN must agree.
  # A slot steps back to a checkpoint. This model's recurrent state cannot be
  # rewound without one at or below the point where a prompt stops matching.
  # At the default 8192, a 49,533 token prompt held four, the lowest at 17,271.
  # A turn sharing 17,270 tokens found none below it and re-read all 49,533.
  # At 2048 it re-reads at most 2048. One checkpoint costs 115 MiB plus
  # 1.9 KiB a token: 24 across a 50,000 token prompt is about 4 GiB.
  --checkpoint-min-step 2048
  --ctx-checkpoints 64   # room for a long prompt at 2048 apart without
                         # evicting the earliest one.
  --no-cache-idle-slots  # Otherwise llama.cpp copies every idle slot into its
                         # RAM cache when a task starts and, under --kv-unified,
                         # CLEARS the slot the router has just restored.
                         # Measured: a strictly extended opening re-read all
                         # 609 tokens on a kv-unified backend, 10 without it.
  -lv 4                # trace: checkpoint reuse shows in the log
  --host "${HOST:-127.0.0.1}"
  --port "$PORT"
)

die() { echo "${0##*/}: $*" >&2; exit 1; }

# Read the weights once, so the server maps pages already in memory. Under
# --numa, llama.cpp sets MADV_RANDOM and skips MAP_POPULATE, and never does
# this itself. PRIME_SKIP names shards not worth reading. The default is the
# shard with the 50 GiB per_layer_token_embd tensor, which --lazy-mode leaves
# on disk. Best effort: a missing DRAFT or oddly named shards must not stop a
# backend from starting.
PRIME_SKIP=${PRIME_SKIP:-'*00003-of-00006.gguf'}
prime() {
  local shard keep=() head=${MODEL%%-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf}
  local -A seen=()
  shopt -s nullglob
  for shard in "$head"*.gguf "$MODEL" "$DRAFT"; do
    [[ -f $shard && -z ${seen[$shard]:-} ]] || continue
    seen[$shard]=1
    # shellcheck disable=SC2053  # PRIME_SKIP is a pattern, by design
    [[ $shard == $PRIME_SKIP ]] || keep+=("$shard")
  done
  shopt -u nullglob
  (( ${#keep[@]} )) || { echo "${0##*/}: nothing to prime for $MODEL" >&2; return 0; }
  numactl "${NUMACTL[@]}" -- cat "${keep[@]}" > /dev/null 2>&1 || true
  return 0
}

launch() {
  mkdir -p "$RUN/slots"
  # Without this check the backend exits 127 in a log nobody reads yet.
  command -v numactl >/dev/null || die "numactl is not installed, and every backend is started through it"
  [[ -x $SERVER ]] || die "no llama-server at $SERVER (set SERVER_MTP in config.local.sh)"
  [[ -f $MODEL  ]] || die "no model at $MODEL (set MODELS, or MODEL_Q8, in config.local.sh)"

  # `if`, not `[[ ... ]] && prime`: under `set -e` a false `&&` list as the
  # last statement of a function returns 1 from it.
  if [[ ${PRIME:-1} == 1 ]]; then prime; fi

  exec numactl "${NUMACTL[@]}" -- "$SERVER" \
    --model "$MODEL" \
    --alias "$ALIAS" \
    --threads "$THREADS" \
    "${COMMON[@]}" \
    "${SAMPLING[@]}" \
    "${ARGS[@]}"
}
