#!/usr/bin/env bash
# Stop what start-all.sh started.
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")" && pwd)}
# Do not source the launch config here: a broken setting must not stop this
# script. Take only RUN from it, in a subshell.
RUN=${RUN:-$( (source "$ROOT/config.local.sh" >/dev/null 2>&1; echo "${RUN:-}") )}
RUN=${RUN:-$ROOT/run}
shopt -s nullglob

# Every pid file, not a fixed list. A backend started by hand, or from an
# older table, must stop too.
others=()
for file in "$RUN"/*.pid; do
  name=${file##*/}; name=${name%.pid}
  [[ $name == router ]] || others+=("$name")
done

# Check /proc before a kill. A pid file goes stale when anyone starts a backend
# by hand, and Linux reuses pids. A stale file would SIGKILL an unrelated
# process and leave the real backend listening.
ours() {
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
  [[ $cmd == *llama-server* || $cmd == *router.py* || $cmd == *qwen-mtp* ]]
}

# The router stops first, while the backends are still up. It copies every
# live cache to disk, and no conversation re-reads its prompt after a restart.
# The copy takes time. The router gets 90 s, a backend 30 s.
for name in router "${others[@]}"; do
  file=$RUN/$name.pid
  [[ -f $file ]] || continue
  pid=$(cat "$file")
  patience=30
  [[ $name == router ]] && patience=90
  if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && ours "$pid"; then
    echo "stopping $name"
    kill "$pid" 2>/dev/null
    # The launch scripts exec, and this pid is the server itself. It needs
    # time to release about 124 GiB.
    for _ in $(seq $patience); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && ours "$pid" && kill -9 "$pid" 2>/dev/null
  elif [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$name: pid $pid is something else now, leaving it alone" >&2
  fi
  rm -f "$file"
done
echo "stopped"
