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
  --cpu-moe-draft      # The draft's experts too, as the model's are: 2.5 GiB
                       # of VRAM. The ggml-org draft adds a 644 MiB output
                       # head, and with its experts on the card the draft
                       # context did not fit at ctx 150000. It costs 4 to 5%
                       # of generate speed: 14.4/19.0/17.0 against
                       # 15.1/19.8/17.8 tok/s at ctx 130000 on koishi, the
                       # same acceptance, two alternating rounds.
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

launch
