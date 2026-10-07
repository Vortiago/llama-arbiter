#!/usr/bin/env bash
# Build tools/op-profile.cpp against a llama.cpp build and run it.
#
#   tools/op-profile.sh -m MODEL [llama flags...] -f PROMPT_FILE [-n TOKENS]
#
# LLAMA is the checkout (default: the one SERVER_MTP is in), BUILD its build
# directory (default: build). The binary lands in $RUN.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/bin/common.sh"
LLAMA=${LLAMA:-$(cd "$(dirname "$SERVER_MTP")/../.." && pwd)}
LIB=$LLAMA/${BUILD:-build}/bin
out=$RUN/op-profile
if [[ ! -x $out || $ROOT/tools/op-profile.cpp -nt $out ]]; then
  g++ -O2 -std=c++17 "$ROOT/tools/op-profile.cpp" -o "$out" \
    -I"$LLAMA/include" -I"$LLAMA/common" -I"$LLAMA/ggml/include" -I"$LLAMA/vendor" \
    -L"$LIB" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$LIB"
fi
exec "$out" "$@"
