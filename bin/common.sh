#!/usr/bin/env bash
# Shared settings for the launch scripts. A script sets MODEL, SERVER, ALIAS
# and ARGS, then calls launch. ARGS comes last, so a flag there overrides a
# default here. LAYOUT.md has the measurements behind these choices.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# What this machine differs in. Not tracked, so no box is written into the
# repository. See config.local.example.sh.
[[ -f $ROOT/config.local.sh ]] && source "$ROOT/config.local.sh"

# Everything a run writes: logs, pids, saved slots, cache events. The router
# reads RUN too, so a value set here has to be exported to reach it.
RUN=${RUN:-$ROOT/run}

# One copy per NUMA node, and they must be separate files: two processes that
# map the same file share one page cache, pinned to whichever node faulted it
# first, so the other node then reads every expert over the interconnect. A
# hard link or a reflink shares the same pages and defeats this.
#
#   MODELS=/mnt/nvme/models MODELS2=/mnt/nvme/models-node1 ./start-all.sh
MODELS=${MODELS:-$ROOT/models}
MODELS2=${MODELS2:-$MODELS}

# MODELS only moves the directory; a different model needs the filenames too,
# so every name below is overridable from config.local.sh.
MODEL_Q8=${MODEL_Q8:-$MODELS/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
MODEL_Q6=${MODEL_Q6:-$MODELS/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf}
DRAFT=${DRAFT:-$MODELS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}

MODEL2_Q8=${MODEL2_Q8:-$MODELS2/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
DRAFT2=${DRAFT2:-$MODELS2/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}

# --no-mmproj-offload keeps the image encoder in RAM: the gpu backend has under
# 1 GiB of VRAM spare at ctx 150000, and the encoder runs once per image.
# VISION=0 starts without it.
MMPROJ=${MMPROJ:-$MODELS/mmproj-F16.gguf}
MMPROJ2=${MMPROJ2:-$MODELS2/mmproj-F16.gguf}

# Where llama-server is. The default is a build under the checkout; set
# SERVER_MTP in config.local.sh for one kept anywhere else.
SERVER_MAIN=${SERVER_MAIN:-$ROOT/qwen/llama.cpp/build/bin/llama-server}
SERVER_MTP=${SERVER_MTP:-$ROOT/llama.cpp-mtp/build/bin/llama-server}

PORT=${PORT:-8080}
NODE=${NODE:-0}        # 0 is the node with the GPU

# The backends this machine starts: one row of "name port script [VAR=value…]"
# a line. start-all.sh starts them in this order, stop-all.sh stops them in
# reverse, and bin/restart-backend.sh restarts one by name - so the names,
# ports and per-backend settings are written once rather than in three scripts
# that had to be kept in step by hand.
#
# The default is one instance, which needs no configuring and works on any
# machine with the weights. Set BACKENDS in config.local.sh for more, and
# ROUTER_BACKENDS to tell the router about the same set: this says what to
# start, that says where to send a turn, and the two have to agree.
BACKENDS=${BACKENDS:-'
solo 8080 qwen-mtp.sh
'}

# The rows of BACKENDS, blank lines and # comments dropped.
backend_rows() {
  printf '%s\n' "$BACKENDS" | sed -e 's/#.*//' -e '/^[[:space:]]*$/d'
}

# The names, in start order.
backend_names() { backend_rows | awk '{print $1}'; }

# One row by name, or nothing.
backend_row() { backend_rows | awk -v n="$1" '$1 == n'; }

source "$ROOT/bin/cores.sh"
# Kept before the default fills it in, so a script that moves NODE after this
# (qwen-mtp-cpu.sh does) can count the right socket without overriding a
# THREADS the caller set on purpose.
THREADS_ENV=${THREADS:-}
THREADS=${THREADS:-$(cores_on_node "$NODE")}

# preferred, not membind: spilling to the other node beats failing.
NUMACTL=(--cpunodebind="$NODE" --preferred="$NODE")

SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0)
MTP_ARGS=(--model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3)

# vision_args <mmproj-file> -- empty if the file is absent or VISION=0
vision_args() {
  [[ ${VISION:-1} == 1 && -f $1 ]] && printf '%s\n' --mmproj "$1" --no-mmproj-offload
}

# --agent gives any client shell and file access, and there is no API key, so
# the backends listen on localhost. Only the router is public.
COMMON=(
  --numa numactl
  --parallel 1
  --flash-attn auto
  --jinja
  --agent
  --metrics            # the router reads these for its dashboard
  --cache-ram "${CACHE_RAM:-0}"
                       # llama.cpp's own RAM copy of an idle conversation. Off,
                       # because the router parks every slot to disk instead.
                       # Two days of logs: 5 hits in 394 reads across the cpu
                       # instances, against 39 s of "updating prompt cache"
                       # pauses. It costs its full size in anonymous memory,
                       # which is what a further prefill instance needs.
  --slot-save-path "$RUN/slots/"   # save/restore a slot's KV to a file. The
                       # router reads the same directory, so RUN has to agree.
  # A checkpoint is what lets a slot step back, and this model's recurrent
  # state cannot be rewound without one at or below the point where a prompt
  # stops matching. At the default 8192 a 49,533 token prompt held four, the
  # lowest at 17,271; a turn sharing 17,270 tokens found nothing below it and
  # re-read all 49,533. At 2048 it would have re-read 2048. One costs 115 MiB
  # plus 1.9 KiB a token, so 24 across a 50,000 token prompt is about 4 GiB.
  --checkpoint-min-step 2048
  --ctx-checkpoints 64   # room for a long prompt at 2048 apart without evicting
                         # the earliest, which an early divergence needs.
  --no-cache-idle-slots  # llama.cpp otherwise copies every idle slot into its
                         # RAM cache when a task starts and, under --kv-unified,
                         # CLEARS the slot, wiping a state the router has just
                         # restored. Measured: a strictly extended opening
                         # re-read all 609 tokens on a kv-unified backend and
                         # 10 on one without it.
  -lv 4                # trace, so checkpoint reuse shows up in the log
  --host "${HOST:-127.0.0.1}"
  --port "$PORT"
)

die() { echo "${0##*/}: $*" >&2; exit 1; }

# Read the weights once, so the server maps pages already in memory. llama.cpp
# sets MADV_RANDOM and skips MAP_POPULATE under --numa, so it never does this
# itself. PRIME_SKIP names shards not worth reading: the default is the shard
# holding the 50 GiB per_layer_token_embd tensor, which --lazy-mode leaves on
# disk.
#
# Every part of this is best effort: a model whose shards are named some other
# way, or a missing DRAFT, must not stop a backend from starting. Reading the
# weights is a speed-up, not a precondition.
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
  # Checked with the other two: without it a box that has no numactl exits 127
  # with "exec: numactl: not found" in a log nobody is looking at yet, while
  # start-all.sh reports only that the backend stopped.
  command -v numactl >/dev/null || die "numactl is not installed, and every backend is started through it"
  [[ -x $SERVER ]] || die "no llama-server at $SERVER (set SERVER_MTP in config.local.sh)"
  [[ -f $MODEL  ]] || die "no model at $MODEL (set MODELS, or MODEL_Q8, in config.local.sh)"

  # Not `[[ ... ]] && prime`: that list's status is the function's when PRIME
  # is 0, and under `set -e` a later statement is all that saved the exec.
  if [[ ${PRIME:-1} == 1 ]]; then prime; fi

  exec numactl "${NUMACTL[@]}" -- "$SERVER" \
    --model "$MODEL" \
    --alias "$ALIAS" \
    --threads "$THREADS" \
    "${COMMON[@]}" \
    "${SAMPLING[@]}" \
    "${ARGS[@]}"
}
