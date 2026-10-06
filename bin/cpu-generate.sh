#!/usr/bin/env bash
# A generate-only backend for a single-node CPU machine. It never reads a
# prompt, so it is always free to answer one that another instance has read.
#   PORT=8082 CPUSET=0-31 THREADS=16 SLOTS=2 ./cpu-generate.sh
#
# At two slots the KV is unified (kv_args in common.sh says why). That costs
# nothing on a 32-core EPYC: split 17.88 tok/s against unified 18.12 with both
# active. It costs 0.82x only at six.
#
# Unpinned and --prio 1: the prefill instances own the physical cores, and a
# reply is seconds of work against their tens of minutes, so it preempts
# briefly rather than being fenced into too few cores.
set -euo pipefail
source "$(dirname "$0")/common.sh"

SERVER=$SERVER_MTP
MODEL=$MODEL_Q8
ALIAS=qwen3.8-flash-next-mtp-generate

mapfile -t VISION_ARGS < <(vision_args "$MMPROJ")
mapfile -t TOOLS_ARGS  < <(tools_args)

SLOTS=${SLOTS:-2}
mapfile -t KV_ARGS < <(kv_args "$SLOTS")

ARGS=(
  --device none
  "${MTP_ARGS[@]}"
  "${VISION_ARGS[@]}"
  "${TOOLS_ARGS[@]}"
  --parallel "$SLOTS"
  "${KV_ARGS[@]}"
  --cpu-range "${CPUSET:-0-31}" --cpu-strict 0
  --prio "${PRIO:-1}"
  --batch-size "${BATCH:-512}"
  --ubatch-size "${UBATCH:-512}"
  --threads-batch "${TB:-$THREADS}"
)

launch
