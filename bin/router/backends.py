"""The table of backends to serve, and what each one may do."""

import json, os, re
from pathlib import Path

# A prefill is compute bound for tens of minutes. A generation is memory
# bound for seconds. A `generate`-only instance is a generator: turns
# migrate to it, `pref` lowest first. One slot per instance: a slot
# reading a long prompt blocks every other slot on it.
DEFAULT_BACKENDS = [
    {"name": "solo", "url": "http://127.0.0.1:8080", "pref": 0,
     "prefill": True, "generate": True, "node": 0},
]


def read_backend_table(env=None):
    """The backends to serve, and where the table came from.

    ROUTER_BACKENDS names a JSON file holding another table: the same fields.
    A table this cannot serve raises here, at startup, rather than at the
    first turn that needs the field. This is a function and not work done at
    import, so that importing this module cannot exit the process that did
    it.
    """
    env = os.environ if env is None else env
    whence = env.get("ROUTER_BACKENDS")
    if not whence:
        return [dict(be) for be in DEFAULT_BACKENDS], "the built-in default"

    table = json.loads(Path(whence).read_text())
    for be in table:
        short = {"name", "url", "pref", "prefill", "generate", "node"} - set(be)
        if short:
            hint = ("  (`reads: true` is now `prefill` and `generate`; "
                    "`reads: false` is `prefill: false, generate: true`)"
                    if "reads" in be else "")
            raise SystemExit(f"[router] backend {be.get('name', be)} has no "
                             f"{', '.join(sorted(short))}{hint}")
        if not (be["prefill"] or be["generate"]):
            raise SystemExit(f"[router] backend {be['name']} neither prefills "
                             f"nor generates, so nothing can be sent to it")
    for job in ("prefill", "generate"):
        if not any(be[job] for be in table):
            raise SystemExit(f"[router] no backend in {whence} can "
                             f"{job}, so no request could be served")
    # A turn leaves its reader only through the handoff.
    if env.get("HANDOFF") == "0" and not all(be["generate"] for be in table):
        raise SystemExit("[router] HANDOFF=0 keeps every turn on the backend "
                         "that read it, so every backend has to generate")
    return table, whence


def prefills(be):
    """May a new conversation have its prompt read on this backend."""
    return be.get("prefill", True)


def generates(be):
    """May a reply be generated on this backend."""
    return be.get("generate", True)


PLACE_RE = re.compile(r"(\d+)_(\d+)$")


def by_place(name):
    """Sort key from a backend name: the socket, then the instance on it.
    gpu0_0 sorts with cpu0_0. A name without a place sorts last."""
    found = PLACE_RE.search(name or "")
    if not found:
        return (9, 9, name or "")
    return (int(found.group(1)), int(found.group(2)), name)
