#!/usr/bin/env bash
# Fetch llama.cpp, apply the patches in patches/, and build llama-server.
#
#   tools/get-llama.sh                 clone, patch, build
#   BUILD=0 tools/get-llama.sh         clone and patch, stop before cmake
#   LLAMA_REF=<commit> tools/...       pin upstream instead of taking its tip
#   CUDA=0 tools/get-llama.sh          build without CUDA, however the box looks
#   CMAKE_ARGS='-DGGML_HIPBLAS=ON' …   anything else cmake needs
#
# It lands at llama.cpp-mtp/build/bin/llama-server, which is what SERVER_MTP
# already defaults to, so a box with an nvidia card needs no config for it.
#
# Safe to run again: a patch already in the tree is skipped rather than failing
# the rest, so this is also how to re-apply them after moving upstream.
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

# One at a time, and reverse-checked first: an already-applied patch is a
# re-run, not a failure. Only a patch that will neither apply nor un-apply is
# a real problem, and then upstream has moved under it.
applied=0 already=0
for patch in "${patches[@]}"; do
  name=${patch##*/}
  if git apply --reverse --check "$patch" 2>/dev/null; then
    already=$(( already + 1 )); echo "    $name - already applied"
  elif git apply "$patch" 2>/dev/null; then
    applied=$(( applied + 1 )); echo "    $name - applied"
  else
    die "$name does not apply to $(git rev-parse --short HEAD).
    Upstream has moved under it. Pin a commit that works with LLAMA_REF, or
    rebase the patch; patches/README.md says what each one changes and why.
    The ones before it are applied and stay that way - running this again
    picks up where it stopped rather than doing them twice."
  fi
done
say "$applied applied, $already already in the tree"

if [[ ${BUILD:-1} != 1 ]]; then
  say "BUILD=0, so stopping here. To build it:"
  echo "    cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server"
  exit 0
fi

# CUDA when the box has a working nvidia driver and nothing says otherwise.
# Every other accelerator is somebody else's flag, which is what CMAKE_ARGS is
# for. Written as an if rather than a && chain because `set -e` takes the
# failure of the last command in a chain as the script failing.
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
