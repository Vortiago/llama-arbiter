#!/usr/bin/env bash
# Fetch llama.cpp, apply the server patches in patches/ and then the CPU speed
# patches in patches/cpu/, and build llama-server.
#
#   tools/get-llama.sh                 clone, patch, build
#   BUILD=0 tools/get-llama.sh         clone and patch, stop before cmake
#   LLAMA_REF=<commit> tools/...       another upstream commit than the pin
#   LLAMA_REF= tools/get-llama.sh      keep the commit the checkout is on
#   CUDA=0 tools/get-llama.sh          build without CUDA
#   CMAKE_ARGS='-DGGML_HIPBLAS=ON' …   anything else cmake needs
#
# The pin is the commit in patches/llama-ref, which the patches are made
# against. The build lands at llama.cpp-mtp/build/bin/llama-server, the
# default SERVER_MTP. Safe to run again: a patch already in the tree is
# skipped.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DIR=${DIR:-$ROOT/llama.cpp-mtp}
UPSTREAM=${UPSTREAM:-https://github.com/ggml-org/llama.cpp}

die() { echo "${0##*/}: $*" >&2; exit 1; }
say() { echo "==> $*"; }

command -v git >/dev/null || die "git is not installed"
[[ ${BUILD:-1} != 1 ]] || command -v cmake >/dev/null ||
  die "cmake is not installed. BUILD=0 clones and patches without it."

if [[ -e $DIR/.git ]]; then   # -e: a worktree has a .git file
  say "$DIR is already a checkout"
else
  say "cloning $UPSTREAM into $DIR"
  git clone "$UPSTREAM" "$DIR"
fi

LLAMA_REF=${LLAMA_REF-$(<"$ROOT/patches/llama-ref")}

# check_out <ref>: fetch only when the commit is not here yet. A checkout that
# holds changes stays where it is: they are patches made for its own commit.
check_out() {
  local ref=$1 want
  if ! want=$(git rev-parse -q --verify "$ref^{commit}") ||
     [[ ! $ref =~ ^[0-9a-f]{7,40}$ ]]; then
    say "fetching, to find $ref"
    git fetch --all --tags
    want=$(git rev-parse -q --verify "$ref^{commit}") ||
      die "no commit $ref in $DIR or its remotes. Add $UPSTREAM as a remote, or set DIR to a new directory."
  fi
  [[ $want == "$(git rev-parse HEAD)" ]] && return 0
  git diff --quiet HEAD ||
    die "$DIR holds changes on another commit than $ref. Set DIR to a new directory, or remove the changes first."
  say "checking out $ref"
  git checkout -q "$ref"
}

cd "$DIR"
if [[ -n $LLAMA_REF ]]; then check_out "$LLAMA_REF"; fi
say "upstream at $(git rev-parse --short HEAD), $(git log -1 --format=%ad --date=short)"

shopt -s nullglob
server_patches=("$ROOT"/patches/*.patch)
cpu_patches=("$ROOT"/patches/cpu/*.patch)
shopt -u nullglob
(( ${#server_patches[@]} )) || die "no patches in $ROOT/patches"

# The router does not work without these three. patches/README.md says why.
# The rest are worth having and are not worth stopping for.
REQUIRED=(slot-state-carries-checkpoints slots-report-the-prompt-size
          anthropic-pass-id-slot)

required() {
  local want name=${1%.patch}
  for want in "${REQUIRED[@]}"; do [[ $name == "$want" ]] && return 0; done
  return 1
}

# Written only by apply_alone and apply_series. The verdict below reads them.
applied=0 already=0
missing_required=() missing_optional=()

# apply_alone <patch>...: each patch on its own. Reverse-check first: an
# already-applied patch is a re-run, not a failure. A patch that neither
# applies nor un-applies means upstream has moved.
#
# Every patch is tried, and the verdict comes at the end. Stopping at the first
# failure hid a worse one: the patches are applied in name order, so an
# optional patch that had gone stale ended the run before the two required ones
# after it were tried at all. The build then failed for a patch nobody needed.
apply_alone() {
  local patch name
  for patch in "$@"; do
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
}

# series_held <patch>...: how many patches of the series, from the first, the
# tree already holds. A later patch rewrites lines an earlier one wrote, so an
# earlier one no longer reverses on its own. This peels the series off a
# scratch copy of the tree, last patch first, and leaves the tree as it is.
series_held() {
  local scratch index held=0 i
  local -a series=("$@") paths=()
  scratch=$(mktemp -d)
  index=$scratch/index
  mapfile -t paths < <(sed -n -e 's|^+++ b/||p' -e 's|^--- a/||p' "$@" | sort -u)
  GIT_INDEX_FILE=$index git read-tree HEAD
  GIT_INDEX_FILE=$index git update-index --add --remove -- "${paths[@]}"
  for (( i = ${#series[@]} - 1; i >= 0; i-- )); do
    if GIT_INDEX_FILE=$index git apply --cached --reverse "${series[i]}" 2>/dev/null; then
      held=$(( held + 1 ))
    fi
  done
  rm -rf "$scratch"
  echo "$held"
}

# apply_series <patch>...: in order, each on top of the one before it. All of
# them are optional. After the first that does not apply, the rest are not
# tried: each one is made on top of the patch that failed.
apply_series() {
  local held i=0 broken=0 patch name
  held=$(series_held "$@")
  for patch in "$@"; do
    name=cpu/${patch##*/}
    if (( i < held )); then
      already=$(( already + 1 )); echo "    $name - already applied"
    elif (( broken )); then
      missing_optional+=("$name"); echo "    $name - not tried, it follows one that does not apply"
    elif git apply "$patch" 2>/dev/null; then
      applied=$(( applied + 1 )); echo "    $name - applied"
    else
      broken=1
      missing_optional+=("$name"); echo "    $name - does not apply (optional, skipped)"
    fi
    i=$(( i + 1 ))
  done
}

# The CPU series changes the CPU backend, ggml.c and the tests. A CUDA build
# compiles the CPU backend too, so it takes the series as well.
apply_alone "${server_patches[@]}"
(( ${#cpu_patches[@]} == 0 )) || apply_series "${cpu_patches[@]}"
say "$applied applied, $already already in the tree"

if (( ${#missing_optional[@]} )); then
  echo "    ${#missing_optional[@]} optional patch(es) skipped: ${missing_optional[*]}"
  echo "    The build goes on without them. patches/README.md says what each one"
  echo "    changes, and rebasing one is usually a context conflict, not a real one."
  echo "    tools/export-cpu-patches.sh writes patches/cpu/ again from a rebased branch."
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
