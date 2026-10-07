#!/usr/bin/env bash
# Fetch llama.cpp, apply the server patches in patches/, the core CPU speed
# patches in patches/cpu/core/ and the optional sets that CPU_OPTIONAL names,
# and build llama-server.
#
#   tools/get-llama.sh                 clone, patch, build
#   CPU_OPTIONAL="I1 Q1" tools/...     also the optional CPU sets I1 and Q1
#   BUILD=0 tools/get-llama.sh         clone and patch, stop before cmake
#   LLAMA_REF=<commit> tools/...       another upstream commit than the pin
#   LLAMA_REF= tools/get-llama.sh      keep the commit the checkout is on
#   CUDA=0 tools/get-llama.sh          build without CUDA
#   CMAKE_ARGS='-DGGML_HIPBLAS=ON' …   anything else cmake needs
#
# The pin is the commit in patches/llama-ref, which the patches are made
# against. The build lands at llama.cpp-mtp/build/bin/llama-server, the
# default SERVER_MTP. Safe to run again: a patch already in the tree is
# skipped. A checkout that holds an optional set CPU_OPTIONAL does not name
# stops the script. To drop that set, run `git -C llama.cpp-mtp checkout -- .`
# and run the script again: no patch adds a file.
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
core_patches=("$ROOT"/patches/cpu/core/*.patch)
shopt -u nullglob
(( ${#server_patches[@]} )) || die "no patches in $ROOT/patches"

# The optional sets in name order, whatever order CPU_OPTIONAL gives:
# export-cpu-patches.sh checks every combination in that order.
read -ra wanted <<<"${CPU_OPTIONAL:-}"
optional_sets=()
(( ${#wanted[@]} == 0 )) || mapfile -t optional_sets < <(printf '%s\n' "${wanted[@]}" | sort -u)
shopt -s nullglob
known_sets=("$ROOT"/patches/cpu/optional/*/)
shopt -u nullglob
known_sets=("${known_sets[@]%/}")
known_sets=("${known_sets[@]##*/}")
for set in "${optional_sets[@]}"; do
  compgen -G "$ROOT/patches/cpu/optional/$set/*.patch" >/dev/null ||
    die "expected an optional CPU set in CPU_OPTIONAL, got $set. patches/cpu/optional/ holds: ${known_sets[*]:-none}"
done

# The router does not work without these three. patches/README.md says why.
# The rest are worth having and are not worth stopping for.
REQUIRED=(slot-state-carries-checkpoints slots-report-the-prompt-size
          anthropic-pass-id-slot)

required() {
  local want name=${1%.patch}
  for want in "${REQUIRED[@]}"; do [[ $name == "$want" ]] && return 0; done
  return 1
}

# Written only by apply_alone, apply_series and apply_cpu. The verdict below
# reads them.
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

# set_patches <set>: the patches of one optional set, in order.
set_patches() { printf '%s\n' "$ROOT"/patches/cpu/optional/"$1"/*.patch; }

# peel <index> <patch>...: reverses the patches off a scratch index, last
# first, and prints how many reversed. A later patch rewrites lines an earlier
# one wrote, so an earlier one reverses only once the later one is off.
peel() {
  local index=$1 held=0 i
  shift
  local -a series=("$@")
  for (( i = ${#series[@]} - 1; i >= 0; i-- )); do
    if GIT_INDEX_FILE=$index git apply --cached --reverse "${series[i]}" 2>/dev/null; then
      held=$(( held + 1 ))
    fi
  done
  echo "$held"
}

# Written only by count_held. apply_cpu reads them.
core_held=0
declare -A set_held=()

# count_held: how many patches of core, and of every optional set, from the
# first, the tree already holds. Each set is made on top of core alone, so the
# sets come off a scratch copy of the tree one by one, and core comes off
# last. The tree stays as it is.
count_held() {
  local scratch index set
  local -a all=("${core_patches[@]}") paths=() patches
  for set in "${known_sets[@]}"; do
    mapfile -t patches < <(set_patches "$set")
    all+=("${patches[@]}")
  done
  scratch=$(mktemp -d)
  index=$scratch/index
  mapfile -t paths < <(sed -n -e 's|^+++ b/||p' -e 's|^--- a/||p' "${all[@]}" | sort -u)
  GIT_INDEX_FILE=$index git read-tree HEAD
  GIT_INDEX_FILE=$index git update-index --add --remove -- "${paths[@]}"
  for set in "${known_sets[@]}"; do
    mapfile -t patches < <(set_patches "$set")
    set_held[$set]=$(peel "$index" "${patches[@]}")
  done
  core_held=$(peel "$index" "${core_patches[@]}")
  rm -rf "$scratch"
}

# apply_series <held> <patch>...: in order, each on top of the one before it.
# The first <held> are already in the tree. All of them are optional. After the
# first that does not apply, the rest are not tried: each one is made on top of
# the patch that failed. Returns 1 when the series is not whole.
apply_series() {
  local held=$1 i=0 broken=0 patch name
  shift
  for patch in "$@"; do
    name=${patch#"$ROOT"/patches/}
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
  (( ! broken ))
}

# apply_cpu: core, then each optional set on top of it. An optional set is made
# on top of core alone, so a set that does not apply leaves the next one to be
# tried, and a core that is not whole leaves every set untried.
apply_cpu() {
  local set
  local -a patches
  count_held
  for set in "${known_sets[@]}"; do
    (( ${set_held[$set]} == 0 )) || [[ " ${optional_sets[*]} " == *" $set "* ]] ||
      die "$DIR holds the optional CPU set $set, which CPU_OPTIONAL does not name.
    Name it, or remove every patch with: git -C $DIR checkout -- ."
  done
  if ! apply_series "$core_held" "${core_patches[@]}"; then
    for set in "${optional_sets[@]}"; do
      missing_optional+=("cpu/optional/$set"); echo "    cpu/optional/$set - not tried, core does not apply whole"
    done
    return 0
  fi
  for set in "${optional_sets[@]}"; do
    mapfile -t patches < <(set_patches "$set")
    apply_series "${set_held[$set]}" "${patches[@]}" || true
  done
}

# The CPU patches change the CPU backend, ggml.c and the tests. A CUDA build
# compiles the CPU backend too, so it takes them as well.
apply_alone "${server_patches[@]}"
(( ${#core_patches[@]} == 0 )) || apply_cpu
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
cmake -B build "-DGGML_CUDA=$cuda" "${extra[@]}"
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
