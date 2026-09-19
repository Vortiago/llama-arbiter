"""Reading what a backend says about itself in its log."""

import re
from pathlib import Path
from ..sizing import VISION

EVICTED_RE = re.compile(r"removing oldest entry \(size = ([\d.]+) MiB\)")


SKIPPED_RE = re.compile(r"prompt state size ([\d.]+) MiB exceeds cache size limit")


STATE_RE = re.compile(r"cache state: (\d+) prompts, ([\d.]+) MiB "
                      r"\(limits: ([\d.]+) MiB")


# Printed once, at startup.
LIMIT_RE = re.compile(r"prompt cache is enabled, size limit: (\d+) MiB")


# A slot restored from a file has no checkpoint. The recurrent state cannot
# be rewound without one, so the backend re-reads the whole prompt.
REREAD_RE = re.compile(r"forcing full prompt re-processing")


# A checkpoint the slot stepped back to. Printed at -lv 4.
CHECKPOINT_RE = re.compile(r"restored context checkpoint \(pos_min = \d+, "
                           r"pos_max = \d+, n_tokens = (\d+)")


# Mapped less lazy is what must stay in the page cache. When it does not,
# prefill falls from 37 tokens a second to single digits. Shard sizes on disk
# say something else: one shard is almost all the lazy tensor.
MAPPED_RE = re.compile(r"CPU_Mapped model buffer size = +([\d.]+) MiB")


LAZY_RE = re.compile(r"add: tensor \S+ \(size = +([\d.]+) MiB\) lazy read enabled")


CONFIG_RE = re.compile(r"(n_ctx|n_batch|n_ubatch|kv_unified|n_slots)\s*=\s*"
                       r"'?([\w.]+)'?")


CONFIG_KEYS = ("n_ctx", "n_batch", "n_ubatch", "kv_unified", "n_slots")


# Printed once under "vision hparams". The word break matters: the
# tokenizer's n_merges is printed long before n_merge.
VISION_RE = re.compile(r"\b(patch_size|n_merge|image_min_pixels|image_max_pixels)"
                       r"\b\s*[:=]\s*(\d+)")


def read_config(lines):
    """The settings a backend started with, from its log. llama-server does
    not report them on /props. A restart appends, so the first of each wins."""
    found = {}
    for line in lines:
        hit = CONFIG_RE.search(line)
        if not hit:
            continue
        key, raw = hit.group(1), hit.group(2)
        if key in found or key not in CONFIG_KEYS:
            continue
        if raw in ("true", "false"):
            found[key] = raw == "true"
        else:
            try:
                found[key] = int(raw)
            except ValueError:
                found[key] = raw
    return found


def read_vision(lines):
    """The vision encoder's geometry from a backend's startup log, or None
    when any part of it is missing."""
    found = {}
    for line in lines:
        hit = VISION_RE.search(line)
        if hit and hit.group(1) not in found:
            found[hit.group(1)] = int(hit.group(2))
    if set(found) != set(VISION) or not all(found.values()):
        return None
    return found


def cache_event(line):
    """Classify one backend log line, or return None.

    Returns ("evicted", mib), ("skipped", mib), ("reread", 0.0),
    ("checkpoint", tokens), ("mapped", mib), ("lazy", mib), ("limit", mib) or
    ("state", (prompts, used_mib, limit_mib))."""
    if not line:
        return None
    found = LIMIT_RE.search(line)
    if found:
        return "limit", float(found.group(1))
    if REREAD_RE.search(line):
        return "reread", 0.0
    found = CHECKPOINT_RE.search(line)
    if found:
        return "checkpoint", float(found.group(1))
    found = MAPPED_RE.search(line)
    if found:
        return "mapped", float(found.group(1))
    found = LAZY_RE.search(line)
    if found:
        return "lazy", float(found.group(1))
    found = EVICTED_RE.search(line)
    if found:
        return "evicted", float(found.group(1))
    found = SKIPPED_RE.search(line)
    if found:
        return "skipped", float(found.group(1))
    found = STATE_RE.search(line)
    if found:
        return "state", (int(found.group(1)), float(found.group(2)),
                         float(found.group(3)))
    return None


class CacheWatch:
    """Follow one backend log and total what it says about the prompt cache.
    The totals cover the life of that backend. `sink(kind, value)` gets the
    per-request events."""

    SUNK = ("evicted", "skipped", "reread", "checkpoint")

    def __init__(self, path, sink=None):
        self.path = Path(path)
        self.sink = sink
        self.offset = 0
        self.inode = None
        # Bytes already in the log when this router started: in the totals,
        # not the sink. bin/restart-router.sh restarts the router with the
        # backends up, and replaying four logs of 3.7 to 9.8 MB wrote 473
        # stale events.
        try:
            self.replay_to = self.path.stat().st_size
        except OSError:
            self.replay_to = 0
        self.stats = {"evictions": 0, "evicted_mib": 0.0, "skipped": 0,
                      "rereads": 0, "checkpoints": 0,
                      "mapped_mib": 0.0, "lazy_mib": 0.0,
                      "prompts": 0, "used_mib": 0.0, "limit_mib": 0.0}

    def poll(self):
        """Read the lines added since the last call."""
        try:
            stat = self.path.stat()
            size, inode = stat.st_size, stat.st_ino
        except OSError:
            return
        # Smaller, or a different inode: a restarted backend writes a new
        # log. See read_settings.
        if size < self.offset or (self.inode is not None and inode != self.inode):
            self.offset = 0
            self.replay_to = 0        # a new log: none of it predates this run
            self.stats.update(evictions=0, evicted_mib=0.0, skipped=0, rereads=0,
                              checkpoints=0, mapped_mib=0.0, lazy_mib=0.0,
                              prompts=0, used_mib=0.0, limit_mib=0.0)
        self.inode = inode
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                fresh = handle.read()
                self.offset = handle.tell()
        except OSError:
            return
        # Block buffered: a partial line waits for the next poll.
        cut = fresh.rfind(b"\n") + 1
        if cut < len(fresh):
            self.offset -= len(fresh) - cut
            fresh = fresh[:cut]
        # Counted in bytes, like the offset. A decoded length is characters.
        at = self.offset - len(fresh)
        for line in fresh.splitlines(keepends=True):
            at += len(line)
            event = cache_event(line.decode("utf-8", "replace"))
            if not event:
                continue
            kind, value = event
            if kind in self.SUNK and self.sink and at > self.replay_to:
                self.sink(kind, value)
            if kind == "evicted":
                self.stats["evictions"] += 1
                self.stats["evicted_mib"] = round(self.stats["evicted_mib"] + value, 1)
            elif kind == "skipped":
                self.stats["skipped"] += 1
            elif kind == "reread":
                self.stats["rereads"] += 1
            elif kind == "checkpoint":
                self.stats["checkpoints"] += 1
            elif kind == "mapped":
                self.stats["mapped_mib"] += value
            elif kind == "lazy":
                self.stats["lazy_mib"] += value
            elif kind == "limit":
                self.stats["limit_mib"] = value
            else:
                prompts, used, limit = value
                self.stats.update(prompts=prompts, used_mib=used, limit_mib=limit)
