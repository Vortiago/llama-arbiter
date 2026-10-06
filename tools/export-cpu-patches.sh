#!/usr/bin/env bash
# Write patches/cpu/ again from a llama.cpp branch, in two tiers: core/ holds
# every commit that no -o names, and optional/<name>/ holds the commits whose
# subject matches <name>'s pattern. tools/get-llama.sh applies core after the
# server patches, then the optional sets that CPU_OPTIONAL names.
#
#   tools/export-cpu-patches.sh perf/stack4
#   tools/export-cpu-patches.sh -o 'I1=sigmoid|dsv4_hc_pre' -o 'Q1=top-k|argsort' perf/stack4
#   LLAMA=~/src/llama.cpp tools/export-cpu-patches.sh [-o <name>=<ERE>]... <branch> [<base>]
#
# The pattern is an extended regular expression, matched against the subject
# of each commit between the base and the branch. A test commit goes with the
# patch it tests, so the pattern must match its subject too.
#
# The branch may hold an optional commit before a core one. The script then
# replays the commits in memory, core first and each optional set on top of
# core alone, with the 3-way merge of `git merge-tree`. It writes no ref and
# does not touch the llama.cpp checkout.
#
# The base is the branch that holds the commit in patches/llama-ref plus the
# server patches in patches/. The script checks that the base is exactly that,
# that core and every combination of the optional sets apply in order, and
# that core plus all of them gives the tree of the branch.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-$ROOT/llama.cpp-mtp}
OUT=$ROOT/patches/cpu

die() { echo "${0##*/}: $*" >&2; exit 1; }
say() { echo "==> $*"; }
llama() { git -C "$LLAMA" "$@"; }

# Written only by the option loop. names lists its keys in the order the
# optional sets apply in.
declare -A pattern=()
while getopts o: flag; do
  [[ $flag == o ]] || die "expected [-o <name>=<ERE>]... <branch> [<base>]"
  [[ $OPTARG =~ ^([A-Za-z0-9_-]+)=(.+)$ ]] ||
    die "expected -o <name>=<ERE>, got -o $OPTARG"
  pattern[${BASH_REMATCH[1]}]=${BASH_REMATCH[2]}
done
shift $(( OPTIND - 1 ))

(( $# == 1 || $# == 2 )) || die "expected [-o <name>=<ERE>]... <branch> [<base>], got $# arguments"
BRANCH=$1
BASE=${2:-perf/base}
PIN=$(<"$ROOT/patches/llama-ref")
names=()
(( ${#pattern[@]} == 0 )) || mapfile -t names < <(printf '%s\n' "${!pattern[@]}" | sort)

# The series carries the author of this repository, not the identity the
# llama.cpp commits were made under.
FROM=${PATCH_FROM:-$(git -C "$ROOT" log -1 --format='%an <%ae>')}

[[ -e $LLAMA/.git ]] || die "expected a llama.cpp checkout at $LLAMA, found none (set LLAMA)"
llama rev-parse -q --verify "$BRANCH^{commit}" >/dev/null || die "no branch $BRANCH in $LLAMA"
llama rev-parse -q --verify "$BASE^{commit}" >/dev/null || die "no base $BASE in $LLAMA"
llama cat-file -e "$PIN^{commit}" 2>/dev/null || die "no commit $PIN in $LLAMA (patches/llama-ref)"

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# indexed: a git command on a scratch index, so the checkout's index is not
# touched.
indexed() { GIT_INDEX_FILE=$scratch/index llama "$@"; }

# check_base: the base must be the pin plus the server patches, because that
# is the tree get-llama.sh builds before it applies the series.
check_base() {
  local patch
  indexed read-tree "$PIN"
  for patch in "$ROOT"/patches/*.patch; do
    indexed apply --cached "$patch" || die "${patch##*/} does not apply to $PIN"
  done
  [[ $(indexed write-tree) == "$(llama rev-parse "$BASE^{tree}")" ]] ||
    die "$BASE is not $PIN plus patches/*.patch. Refresh the server patches, or name the right base."
}

# tier_of <commit>: core, or the name of the one optional set whose pattern
# matches the subject.
tier_of() {
  local subject name tier=core
  subject=$(llama log -1 --format=%s "$1")
  for name in "${names[@]}"; do
    grep -qE -- "${pattern[$name]}" <<<"$subject" || continue
    [[ $tier == core ]] || die "\"$subject\" matches both $tier and $name"
    tier=$name
  done
  echo "$tier"
}

# replay <onto> <commit>: the commit cherry-picked onto <onto>, in memory, with
# its own message, author and dates. Prints the new commit. A commit already on
# <onto> is kept, so its patch keeps the sha on its first line.
replay() {
  local onto=$1 commit=$2 tree
  local -a who
  if [[ $(llama rev-parse "$commit^") == "$onto" ]]; then echo "$commit"; return; fi
  tree=$(llama merge-tree --write-tree --merge-base="$commit^" "$onto" "$commit") ||
    die "$(llama log -1 --format='%h %s' "$commit") does not replay onto the commits before it in its tier"
  mapfile -t who < <(llama log -1 --format='%an%n%ae%n%aD%n%cn%n%ce%n%cD' "$commit")
  llama log -1 --format=%B "$commit" |
    GIT_AUTHOR_NAME=${who[0]} GIT_AUTHOR_EMAIL=${who[1]} GIT_AUTHOR_DATE=${who[2]} \
    GIT_COMMITTER_NAME=${who[3]} GIT_COMMITTER_EMAIL=${who[4]} GIT_COMMITTER_DATE=${who[5]} \
    llama commit-tree "$tree" -p "$onto"
}

# export_series <from> <to> <dir>: one file per commit. --no-numbered keeps the
# subject of a patch the same when a later export adds a commit to the series.
export_series() {
  local patch
  mkdir -p "$3"
  llama format-patch -q --no-signature --no-numbered -o "$3" "$1..$2"
  for patch in "$3"/*.patch; do
    awk -v from="$FROM" '!done && /^From: /{print "From: " from; done=1; next} {print}' \
      "$patch" > "$patch.tmp"
    mv "$patch.tmp" "$patch"
  done
}

# apply_dirs <dir>...: the patches of each directory in order onto the index.
apply_dirs() {
  local dir patch
  for dir in "$@"; do
    for patch in "$dir"/*.patch; do
      indexed apply --cached "$patch" || die "${patch#"$scratch"/} does not apply in order"
    done
  done
}

# check_combinations: core gives the tree it was replayed to, every subset of
# the optional sets applies on top of it in name order, and core plus all of
# them gives the tree of the branch.
check_combinations() {
  local mask i last=$(( (1 << ${#names[@]}) - 1 ))
  local -a dirs
  for (( mask = 0; mask <= last; mask++ )); do
    dirs=("$scratch/core")
    for i in "${!names[@]}"; do
      if (( mask & (1 << i) )); then dirs+=("$scratch/optional/${names[i]}"); fi
    done
    indexed read-tree "$BASE"
    apply_dirs "${dirs[@]}"
    (( mask )) || [[ $(indexed write-tree) == "$(llama rev-parse "$core^{tree}")" ]] ||
      die "core does not give the tree it was replayed to"
  done
  [[ $(indexed write-tree) == "$(llama rev-parse "$BRANCH^{tree}")" ]] ||
    die "core plus ${names[*]:-nothing} does not give the tree of $BRANCH"
}

say "checking that $BASE is $(cut -c1-9 <<<"$PIN") plus the server patches"
check_base

mapfile -t commits < <(llama rev-list --reverse "$BASE..$BRANCH")
(( ${#commits[@]} )) || die "no commits between $BASE and $BRANCH"
declare -A tier=() count=([core]=0)
for commit in "${commits[@]}"; do
  tier[$commit]=$(tier_of "$commit")
  count[${tier[$commit]}]=$(( ${count[${tier[$commit]}]:-0} + 1 ))
done
(( count[core] )) || die "every commit between $BASE and $BRANCH matches an -o pattern, so core is empty"
for name in "${names[@]}"; do
  (( ${count[$name]:-0} )) || die "-o $name matches no commit between $BASE and $BRANCH"
done

core=$(llama rev-parse "$BASE")
for commit in "${commits[@]}"; do
  if [[ ${tier[$commit]} == core ]]; then core=$(replay "$core" "$commit"); fi
done
export_series "$BASE" "$core" "$scratch/core"
for name in "${names[@]}"; do
  tip=$core
  for commit in "${commits[@]}"; do
    if [[ ${tier[$commit]} == "$name" ]]; then tip=$(replay "$tip" "$commit"); fi
  done
  export_series "$core" "$tip" "$scratch/optional/$name"
done

say "checking core and every combination of the ${#names[@]} optional set(s) against $BRANCH"
check_combinations

rm -rf "$OUT/core" "$OUT/optional"
rm -f "$OUT"/*.patch
mkdir -p "$OUT"
mv "$scratch/core" "$OUT/"
[[ ! -d $scratch/optional ]] || mv "$scratch/optional" "$OUT/"
say "wrote ${count[core]} core patches to $OUT/core from $BRANCH ($(llama rev-parse --short "$BRANCH"))"
for name in "${names[@]}"; do
  echo "    and ${count[$name]} to optional/$name"
done
