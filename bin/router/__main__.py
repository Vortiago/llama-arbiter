"""Read the environment once, wire the router, and serve."""

import argparse, os, signal, socket, sys, threading, time
from pathlib import Path
from .backends import read_backend_table
from .pool.pool import Pool
from .settings import Tuning
from .store.events import EventLog
from .store.files import Store
from .web.handler import Handler
from .web.server import Server, Stamped

def build(env=None):
    """Read the environment once and wire the router from it.

    Importing this module does none of this. It defines names, and nothing
    else: no thread starts, no directory is read, and a bad backend table
    fails here rather than at the import of whatever asked for it."""
    env = os.environ if env is None else env
    tuning = Tuning.from_env(env)
    store = Store(env.get("RUN")
                  # bin/router/__main__.py -> the checkout
                  or Path(__file__).resolve().parents[2] / "run",
                  env.get("BLOCK_DIR"))
    events = EventLog(env.get("CACHE_LOG_DIR") or store.run, on=tuning.cache_log)
    table, whence = read_backend_table(env)
    pool = Pool(table, store=store, tuning=tuning, events=events)
    return pool, whence


if __name__ == "__main__":
    sys.stdout = Stamped(sys.stdout)
    sys.stderr = Stamped(sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="::")   # "::" means IPv6 and IPv4
    args = parser.parse_args()
    Server.address_family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    POOL, BACKENDS_FROM = build()
    POOL.adopt()

    stopping = threading.Event()

    def shut_down(signum, frame):
        """Copy every live cache out before the backends go away. A cache
        only exists in a slot."""
        if stopping.is_set():
            sys.exit(1)           # a second signal means stop arguing
        stopping.set()
        print("[router] stopping, parking caches", flush=True)
        # The worker's copy first. park_all skips a record being copied, and
        # sys.exit kills the daemon worker mid-copy.
        began = time.time()
        if not POOL.drain_parks(POOL.tuning.park_all_timeout):
            print("[router] a copy on the worker did not land in time",
                  flush=True)
        parked = POOL.park_all(timeout=POOL.tuning.park_all_timeout,
                               budget=max(1.0, POOL.tuning.park_all_budget
                                          - (time.time() - began)))
        kept = POOL.save_pins()
        print(f"[router] parked {parked} conversation(s), wrote {kept} pin(s)",
              flush=True)
        # sys.exit drops what the daemon writer still holds.
        POOL.events.flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shut_down)
    signal.signal(signal.SIGINT, shut_down)
    print(f"[router] {len(POOL.backends)} backend(s) from {BACKENDS_FROM}: "
          f"{', '.join(b['name'] for b in POOL.backends)}", flush=True)
    print(f"[router] listening on {args.host}:{args.port}", flush=True)
    server = Server((args.host, args.port), Handler)
    server.pool = POOL
    server.serve_forever()
