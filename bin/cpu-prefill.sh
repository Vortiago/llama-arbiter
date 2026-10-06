#!/usr/bin/env bash
# A prefill-only backend for a single-node CPU machine, one slot, pinned to
# CPUSET.  PORT=8080 CPUSET=0-31 THREADS=32 ./cpu-prefill.sh
#
# One slot, not two: a slot reading a long prompt stops every other slot on its
# instance, so a second slot buys a queue rather than a second reader. Two
# instances are two schedulers. On a 32-core EPYC 7502P, two readers at 32
# threads over every core read 73.05 tok/s aggregate, against 63.75 for two at
# 16 on disjoint halves: the scheduler lends an idle core, a static split
# cannot.
set -euo pipefail
source "$(dirname "$0")/common.sh"

SERVER=$SERVER_MTP
MODEL=$MODEL_Q8
ALIAS=qwen3.8-flash-next-mtp-prefill

mapfile -t VISION_ARGS < <(vision_args "$MMPROJ")
mapfile -t TOOLS_ARGS  < <(tools_args)

ARGS=(
  --device none
  "${MTP_ARGS[@]}"
  "${VISION_ARGS[@]}"
  ${TOOLS_ARGS[@]+"${TOOLS_ARGS[@]}"}
  --parallel 1
  --ctx-size "${CTX:-150000}"
  --cpu-range "${CPUSET:-0-31}" --cpu-strict 1
  --batch-size "${BATCH:-512}"
  --ubatch-size "${UBATCH:-512}"
                       # 512/512, not the 2048/512 that one server alone
                       # settled on. Two instances sharing a socket: 43.6
                       # tok/s aggregate at 512 against 41.0 at 2048 reading
                       # 8192 tokens, 35.0 against 34.3 at 32768.
  --threads-batch "${TB:-$THREADS}"
)

launch
