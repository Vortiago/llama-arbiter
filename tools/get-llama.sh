#!/usr/bin/env bash
# Fetch llama.cpp, apply the patches in patches/, and build llama-server.
#
#   tools/get-llama.sh                 clone, patch, build
#   BUILD=0 tools/get-llama.sh         clone and patch, stop before cmake
#   LLAMA_REF=<commit> tools/...       pin upstream instead of its tip
#   CUDA=0 tools/get-llama.sh          build without CUDA
#   CMAKE_ARGS='-DGGML_HIPBLAS=ON' …   anything else cmake needs
#
# The build lands at llama.cpp-mtp/build/bin/llama-server, the default
# SERVER_MTP. Safe to run again: a patch already in the tree is skipped.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DIR=${DIR:-$ROOT/llama.cpp-mtp}
UPSTREAM=${UPSTREAM:-https://github.com/ggml-org/llama.cpp}

die() { echo "${0##*/}: $*" >&2; exit 1; }
say() { echo "==> $*"; }

command -v git >/dev/null || die "git is not installed"
[[ ${BUILD:-1} != 1 ]] || command -v cmake >/dev/null ||
  die "cmake is not installed. BUILD=0 clones and patches without it."

if [[ -d $DIR/.git ]]; then
  say "$DIR is already a checkout, leaving it as it is"
else
  say "cloning $UPSTREAM into $DIR"
  git clone "$UPSTREAM" "$DIR"
fi

cd "$DIR"
if [[ -n ${LLAMA_REF:-} ]]; then
  say "checking out $LLAMA_REF"
  git fetch --all --tags
  git checkout "$LLAMA_REF"
fi
say "upstream at $(git rev-parse --short HEAD), $(git log -1 --format=%ad --date=short)"

shopt -s nullglob
patches=("$ROOT"/patches/*.patch)
shopt -u nullglob
(( ${#patches[@]} )) || die "no patches in $ROOT/patches"

# The router does not work without these three. patches/README.md says why.
# The rest are worth having and are not worth stopping for.
REQUIRED=(slot-state-carries-checkpoints slots-report-the-prompt-size
          anthropic-pass-id-slot)

required() {
  local want name=${1%.patch}
  for want in "${REQUIRED[@]}"; do [[ $name == "$want" ]] && return 0; done
  return 1
}

# Reverse-check first: an already-applied patch is a re-run, not a failure.
# A patch that neither applies nor un-applies means upstream has moved.
#
# Every patch is tried, and the verdict comes at the end. Stopping at the first
# failure hid a worse one: the patches are applied in name order, so an
# optional patch that had gone stale ended the run before the two required ones
# after it were tried at all. The build then failed for a patch nobody needed.
applied=0 already=0
missing_required=() missing_optional=()
for patch in "${patches[@]}"; do
  name=${patch##*/}
  if git apply --reverse --check "$patch" 2>/dev/null; then
    already=$(( already + 1 )); echo "    $name - already applied"
  elif git apply "$patch" 2>/dev/null; then
    applied=$(( applied + 1 )); echo "    $name - applied"
  elif required "$name"; then
    missing_required+=("$name"); echo "    $name - DOES NOT APPLY (required)"
  else
    missing_optional+=("$name"); echo "    $name - does not apply (optional, skipped)"
  fi
done
say "$applied applied, $already already in the tree"

if (( ${#missing_optional[@]} )); then
  echo "    ${#missing_optional[@]} optional patch(es) skipped: ${missing_optional[*]}"
  echo "    The build goes on without them. patches/README.md says what each one"
  echo "    changes, and rebasing one is usually a context conflict, not a real one."
fi

if (( ${#missing_required[@]} )); then
  die "these required patches do not apply to $(git rev-parse --short HEAD):
      ${missing_required[*]}
    The router does not work without them. Upstream has moved underneath.
    Pin a commit that works with LLAMA_REF, or rebase the patch;
    patches/README.md says what each one changes and why. Everything that
    did apply stays applied - running this again picks up where it stopped."
fi

if [[ ${BUILD:-1} != 1 ]]; then
  say "BUILD=0, so stopping here. To build it:"
  echo "    cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server"
  exit 0
fi

# CUDA when nvidia-smi sees a card and CUDA is unset. Other accelerators go
# through CMAKE_ARGS. An `if`, not a `&&` chain: `set -e` exits on the failure
# of the last command in a chain.
if [[ -z ${CUDA:-} ]]; then
  CUDA=0
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    CUDA=1
  fi
fi
[[ $CUDA == 1 ]] && cuda=ON || cuda=OFF
read -ra extra <<<"${CMAKE_ARGS:-}"

say "configuring, GGML_CUDA=$cuda"
cmake -B build "-DGGML_CUDA=$cuda" ${extra[@]+"${extra[@]}"}
say "building llama-server - this takes a while"
cmake --build build -j --target llama-server

server=$DIR/build/bin/llama-server
[[ -x $server ]] || die "the build finished but there is no llama-server at $server"
say "built $server"
if [[ $DIR != "$ROOT/llama.cpp-mtp" ]]; then
  echo "    That is not where the launch scripts look, so put"
  echo "      export SERVER_MTP=$server"
  echo "    in config.local.sh."
fi
