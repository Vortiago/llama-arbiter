#!/usr/bin/env bash
# Run Qwen3.8-Flash-Next, GPU with the experts kept in RAM.  ./qwen.sh
set -euo pipefail
source "$(dirname "$0")/../bin/common.sh"

MODEL=$MODEL_Q6
SERVER=$SERVER_MAIN
ALIAS=qwen3.8-flash-next

CPU_MOE=46    # expert layers kept in RAM; 48 = all, leaving max VRAM for context

ARGS=(
  --n-cpu-moe "$CPU_MOE"
  --n-gpu-layers 99
  --ctx-size 130000
  --batch-size 2048
  --ubatch-size 512
)

launch
