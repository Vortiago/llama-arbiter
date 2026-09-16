#!/usr/bin/env bash
# Stop what start-all.sh started.
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")" && pwd)}
# Nothing below sources the launch config - a broken setting must not leave the
# box running - but RUN says where the pid files are, so take that one name from
# it, in a subshell that cannot affect anything else here.
RUN=${RUN:-$( (source "$ROOT/config.local.sh" >/dev/null 2>&1; echo "${RUN:-}") )}
RUN=${RUN:-$ROOT/run}
shopt -s nullglob

# Whatever left a pid file, not a written-down list: a backend started by hand,
# or one from a table this run no longer has, still has to be stopped.
others=()
for file in "$RUN"/*.pid; do
  name=${file##*/}; name=${name%.pid}
  [[ $name == router ]] || others+=("$name")
done

# `kill` from a pid file, but only after asking the kernel what that pid is.
# A file goes stale the moment anyone starts a backend by hand, and Linux
# reuses pids: without this a stale file had stop-all.sh SIGKILL something
# else of this user's and leave the real backend listening, so the next
# start-all.sh brought up a second one beside it.
ours() {
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
  [[ $cmd == *llama-server* || $cmd == *router.py* || $cmd == *qwen-mtp* ]]
}

# The router goes first, while the backends are still up to be read from: it
# copies every live cache to disk, so no conversation re-reads its whole prompt
# after a restart. Copying takes longer than dying, so it waits longer.
for name in router "${others[@]}"; do
  file=$RUN/$name.pid
  [[ -f $file ]] || continue
  pid=$(cat "$file")
  patience=30
  [[ $name == router ]] && patience=90
  if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && ours "$pid"; then
    echo "stopping $name"
    kill "$pid" 2>/dev/null
    # The launch scripts exec, so this pid is the server. It needs time to
    # release about 124 GiB.
    for _ in $(seq $patience); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && ours "$pid" && kill -9 "$pid" 2>/dev/null
  elif [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$name: pid $pid is something else now, leaving it alone" >&2
  fi
  rm -f "$file"
done
echo "stopped"
