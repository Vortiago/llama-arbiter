#!/usr/bin/env bash
# Backend on node 1, CPU only. PORT=8081 SLOTS=1 ./qwen-mtp-cpu.sh
#
# Node 1 runs two of these, one slot each. A slot reading a long prompt holds
# up every other slot on its own instance, so a second slot does not buy a
# second prefill: it buys a queue behind the first. Both instances get every
# core on the node: when only one has work it uses the whole socket, and the
# scheduler splits it when both do.
set -euo pipefail
source "$(dirname "$0")/common.sh"

NODE=${NODE:-1}        # 1 is the socket without the GPU
NUMACTL=(--cpunodebind="$NODE" --preferred="$NODE")
# And the thread count with it: common.sh counted it against the default NODE,
# before the line above moved it. THREADS_ENV is what the caller asked for.
THREADS=${THREADS_ENV:-$(cores_on_node "$NODE")}

# Each node reads its own copy of the weights, so its page cache is local.
if [[ $NODE == 0 ]]; then
  MODEL=$MODEL_Q8; DRAFT_FILE=$DRAFT; MMPROJ_FILE=$MMPROJ
else
  MODEL=$MODEL2_Q8; DRAFT_FILE=$DRAFT2; MMPROJ_FILE=$MMPROJ2
fi
DRAFT=$DRAFT_FILE
MTP_ARGS=(--model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3)
SERVER=$SERVER_MTP
ALIAS=qwen3.8-flash-next-mtp-cpu

mapfile -t VISION_ARGS < <(vision_args "$MMPROJ_FILE")

# A state file records n_stream, and a restore refuses a file that disagrees.
# n_stream is 1 when the KV is unified, and otherwise the number of sequences.
# So one slot is already 1 and needs nothing; more than one has to be told.
# CTX matches the GPU backend, and has to: a conversation that outgrew the
# smaller backend could never move to it, and reading a full context takes hours.
if [[ ${SLOTS:-1} -gt 1 ]]; then
  KV_ARGS=(--kv-unified --kv-unified-per-slot "${CTX:-150000}")
else
  KV_ARGS=(--ctx-size "${CTX:-150000}")
fi

ARGS=(
  --device none        # keep the GPU free for the other backend
  "${MTP_ARGS[@]}"
  "${VISION_ARGS[@]}"
  # Quoted, like every other argument here: these four come straight from the
  # caller's environment through bin/restart-backend.sh, and an empty one would
  # otherwise leave its flag as the last word with no value after it.
  --parallel "${SLOTS:-1}"
  "${KV_ARGS[@]}"
  --batch-size "${BATCH:-512}"
  --ubatch-size "${UBATCH:-512}"
                       # The shape of the work: each expert sees UBATCH rows at
                       # a time. Two instances on one socket, aggregate
                       # tokens/s: 43.6 at 512 against 41.0 at 2048 reading
                       # 8192 tokens, 35.0 against 34.3 at 32768. Never worse
                       # at any length measured, drained or paired.
  --threads-batch "${TB:-$THREADS}"
                       # Prefill only. THREADS is the physical cores on the
                       # socket; the second hyperthread may or may not pay.
)
# No --poll: this build is GGML_OPENMP=ON, and that path never reads it.
# OMP_WAIT_POLICY and GOMP_SPINCOUNT are the equivalent, unmeasured here.

launch
