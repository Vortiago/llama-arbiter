#!/usr/bin/env bash
# Write patches/cpu/ again from a llama.cpp branch: one numbered patch for each
# commit between the base and the branch. tools/get-llama.sh applies them in
# that order, after the server patches.
#
#   tools/export-cpu-patches.sh perf/stack3
#   tools/export-cpu-patches.sh perf/stack3 perf/base
#   LLAMA=~/src/llama.cpp tools/export-cpu-patches.sh <branch> [<base>]
#
# The base is the branch that holds the commit in patches/llama-ref plus the
# server patches in patches/. The script checks that the base is exactly that,
# and that the new series gives the tree of the branch. It checks both in a
# scratch index, so the llama.cpp checkout is not touched.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-$ROOT/llama.cpp-mtp}
OUT=$ROOT/patches/cpu

die() { echo "${0##*/}: $*" >&2; exit 1; }
say() { echo "==> $*"; }
llama() { git -C "$LLAMA" "$@"; }

(( $# == 1 || $# == 2 )) || die "expected <branch> [<base>], got $# arguments"
BRANCH=$1
BASE=${2:-perf/base}
PIN=$(<"$ROOT/patches/llama-ref")

# The series carries the author of this repository, not the identity the
# llama.cpp commits were made under.
FROM=${PATCH_FROM:-$(git -C "$ROOT" log -1 --format='%an <%ae>')}

[[ -e $LLAMA/.git ]] || die "expected a llama.cpp checkout at $LLAMA, found none (set LLAMA)"
llama rev-parse -q --verify "$BRANCH^{commit}" >/dev/null || die "no branch $BRANCH in $LLAMA"
llama rev-parse -q --verify "$BASE^{commit}" >/dev/null || die "no base $BASE in $LLAMA"
llama cat-file -e "$PIN^{commit}" 2>/dev/null || die "no commit $PIN in $LLAMA (patches/llama-ref)"

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export GIT_INDEX_FILE=$scratch/index

# check_base: the base must be the pin plus the server patches, because that
# is the tree get-llama.sh builds before it applies the series.
check_base() {
  local patch
  llama read-tree "$PIN"
  for patch in "$ROOT"/patches/*.patch; do
    llama apply --cached "$patch" || die "${patch##*/} does not apply to $PIN"
  done
  [[ $(llama write-tree) == "$(llama rev-parse "$BASE^{tree}")" ]] ||
    die "$BASE is not $PIN plus patches/*.patch. Refresh the server patches, or name the right base."
}

# export_series <dir>: one file per commit. --no-numbered keeps the subject of
# a patch the same when a later export adds a commit to the series.
export_series() {
  local dir=$1 patch
  llama format-patch -q --no-signature --no-numbered -o "$dir" "$BASE..$BRANCH"
  for patch in "$dir"/*.patch; do
    awk -v from="$FROM" '!done && /^From: /{print "From: " from; done=1; next} {print}' \
      "$patch" > "$patch.tmp"
    mv "$patch.tmp" "$patch"
  done
}

# check_series <dir>: applied in order onto the base, the series must give the
# tree of the branch.
check_series() {
  local patch
  llama read-tree "$BASE"
  for patch in "$1"/*.patch; do
    llama apply --cached "$patch" || die "${patch##*/} does not apply in order"
  done
  [[ $(llama write-tree) == "$(llama rev-parse "$BRANCH^{tree}")" ]] ||
    die "the series does not give the tree of $BRANCH"
}

say "checking that $BASE is $(cut -c1-9 <<<"$PIN") plus the server patches"
check_base
mkdir "$scratch/series"
export_series "$scratch/series"
shopt -s nullglob
series=("$scratch"/series/*.patch)
(( ${#series[@]} )) || die "no commits between $BASE and $BRANCH"
say "checking that ${#series[@]} patches give the tree of $BRANCH"
check_series "$scratch/series"

mkdir -p "$OUT"
rm -f "$OUT"/*.patch
mv "${series[@]}" "$OUT"/
say "wrote ${#series[@]} patches to $OUT from $BRANCH ($(llama rev-parse --short "$BRANCH"))"
