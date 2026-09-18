"""The numbers this router was tuned to."""

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Tuning:
    """The numbers this router was tuned to, and where each one came from.

    docs/LAYOUT.md says which of these are properties of llama.cpp, which are
    properties of Linux, and which belong to one machine. A caller that wants
    another set builds another Tuning; nothing reads these from the module.
    """

    # -- pins and slots
    max_pins: int = 512               # conversations to remember
    pin_patience: float = 20.0        # seconds a conversation waits for the
                                      # backend that holds its cache before
                                      # taking a free one
    poll: float = 2.0                 # seconds between backend checks
    rate_window: float = 10.0         # seconds a per-slot rate is measured over
    stall_rate: float = 0.5           # tokens/s. Under this a generating slot
                                      # is stalled behind another slot's read.
                                      # Measured 0.02 to 0.06 against 6.3 solo.

    # -- what a turn may take
    forward_timeout: float = 7200.0   # longest a backend may take to answer
    read_timeout: float = 7200.0      # a read runs 30 to 40 minutes
    ping_every: float = 15.0          # seconds of quiet before a ping. Clients
                                      # drop a stream after 300 s of silence.
    post_timeout: float = 300.0       # a save waits for the slot to finish
    max_body: int = 256 * 1024 * 1024  # largest request body read into memory.
                                      # A turn at ctx 150000 is a few megabytes.
    chars_per_tok: float = 4.0
    reply_tokens: int = 1024          # room to reserve for the reply

    # -- stopping and draining
    drain_deadline: float = 1800.0    # seconds a drain waits for running work
    park_all_timeout: float = 75.0    # longest one save may take at shutdown.
                                      # Measured: median 11 s, p90 19 s, 77 of
                                      # 952 saves over 20 s.
    park_all_budget: float = 80.0     # wall-clock cap over all shutdown saves.
                                      # stop-all.sh gives the router 90 s.

    # -- the disk
    park_floor: int = 64 * 1024 * 1024  # a real state file is about 112 MiB.
                                      # A smaller file means the slot changed
                                      # hands.
    # Disk for the conversation copies, in bytes. A copy is about 115 MiB plus
    # 36.6 KiB a token: 0.2 to 5.4 GiB at ctx 150000. Size it to the disk RUN
    # is on. Too low displaces a copy still in use, which costs a full re-read.
    park_budget: int = 256 * 1024 ** 3
    # Disk for the saved openings, in bytes. One block is 0.6 to 3.7 GB. Least
    # recently used goes first. Size it to the disk the blocks are on.
    block_budget: int = 64 * 1024 ** 3
    mount_poll: float = 30.0          # seconds between disk usage checks

    # -- openings
    prefix_min_chars: int = 8000      # about 2000 tokens. A shorter cut is not
                                      # worth a file.
    system_min_chars: int = 2000      # the only cut two sessions share. Claude
                                      # Code sends 6,100 characters: a minute
                                      # to read, a fifth of a second to load.
    build_patience: float = 1800.0    # longest a request waits for another to
                                      # save the opening they share
    build_poll: float = 10.0          # seconds between builder passes
    idle_polls: int = 2               # polls a slot must look idle before the
                                      # builder reads into it. One poll can be
                                      # two seconds old.
    want_keep: int = 8                # openings noted as missing, not built yet
    # Deeper openings: a cut where two conversations diverge. Off by default:
    # over two days of real traffic it was built 0 times and loaded 0 times.
    # The detection still runs, and the `choice` event records how deep a fork
    # could have started. tools/cache-report.py reads it.
    deep_openings: bool = False

    # -- moving a turn
    # A restored slot needs its context checkpoints in the state file, which
    # needs patches/slot-state-carries-checkpoints.patch. Without the patch a
    # move is followed by a full re-read: turn this off.
    handoff: bool = True

    # -- what the dashboard is shown
    history_step: float = 10.0        # seconds per history bucket
    history_keep: int = 60            # buckets kept: ten minutes
    gpu_poll: float = 10.0            # seconds between nvidia-smi runs. Each
                                      # costs 50 to 100 ms, so it runs as a
                                      # child.
    recent_requests: int = 20         # requests the dashboard lists
    recent_files: int = 24            # slot files the dashboard lists. A turn
                                      # boundary can write three in one push.
    flow_log: int = 150               # stage transitions the flow view replays
    capture_keep: int = 24            # bodies kept per conversation

    # -- telemetry
    cache_log: bool = True            # one JSON line per cache decision

    @classmethod
    def from_env(cls, env=None):
        """The five settings a machine overrides. Everything else is measured,
        and changing it means changing this file."""
        env = os.environ if env is None else env
        return cls(
            park_budget=int(float(env.get("PARK_BUDGET_GB") or 256) * 1024 ** 3),
            block_budget=int(float(env.get("BLOCK_BUDGET_GB") or 64) * 1024 ** 3),
            handoff=env.get("HANDOFF", "1") == "1",
            deep_openings=env.get("DEEP_OPENINGS", "0") == "1",
            cache_log=env.get("CACHE_LOG", "1") == "1")
