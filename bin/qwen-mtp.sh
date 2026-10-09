#!/usr/bin/env bash
# Backend on node 0, with the GPU. One slot. ./qwen-mtp.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Never copy the experts to the card. A batch of 32 tokens or more ran its
# experts there, which uploads nearly all 120 GB of them for each ubatch. On
# koishi's PCIe gen3 x8 the upload set the pace: 120-token prompts read at
# 9.8 tok/s and 2000-token ones at 30, against 68 and 112 with the experts on
# the CPU and the rest on the card. A generate step is 4 tokens and never
# uploaded. A card on a faster link may want this back at 32.
export GGML_OP_OFFLOAD_MIN_BATCH=${GGML_OP_OFFLOAD_MIN_BATCH:-1000000}

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
  --backend-sampling   # Sample on the card: +2 to +5% generate at temperature
                       # 1.0 on koishi, the same acceptance. llama.cpp turns it
                       # off for a request with a grammar, and the grammar
                       # readout of /v1/systemone came back the same.
  --ctx-size "${CTX:-150000}"
                       # Sized to this card's 16 GiB at f16 KV, about 36.5 KiB
                       # a token. Every backend must use the same CTX (see
                       # config.local.example.sh). docs/LAYOUT.md has the rest.
  --batch-size "${BATCH:-2048}"
  --ubatch-size "${UBATCH:-512}"
)

# The draft's experts on the card unless CPU_MOE_DRAFT=0. Keeping them on the
# CPU frees 2.5 GiB of VRAM (with them on the card the draft context did not fit
# at ctx 150000) and costs 4 to 5% of generate speed: 14.4/19.0/17.0 against
# 15.1/19.8/17.8 tok/s at ctx 130000, the same acceptance.
if [[ ${CPU_MOE_DRAFT:-1} == 1 ]]; then
  ARGS+=(--cpu-moe-draft)
fi

launch
