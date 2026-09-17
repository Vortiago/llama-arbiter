#!/usr/bin/env bash
# Backend on node 0, with the GPU. One slot. ./qwen-mtp.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$MODEL_Q8
SERVER=$SERVER_MTP
ALIAS=qwen3.8-flash-next-mtp

mapfile -t VISION_ARGS < <(vision_args "$MMPROJ")

ARGS=(
  "${MTP_ARGS[@]}"
  "${VISION_ARGS[@]}"
  --n-gpu-layers "${NGL:-99}"
  --n-cpu-moe "${N_CPU_MOE:-48}"
                       # Attention and KV on the card, experts in RAM. VRAM is
                       # the limit, not RAM. Raise it for a smaller card.
  --ctx-size "${CTX:-150000}"
                       # Sized to this card's 16 GiB at f16 KV, about 36.5 KiB
                       # a token. Every backend must use the same CTX (see
                       # config.local.example.sh). docs/LAYOUT.md has the rest.
  --batch-size "${BATCH:-2048}"
  --ubatch-size "${UBATCH:-512}"
)

launch
