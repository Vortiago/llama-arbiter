#!/usr/bin/env python3
"""Send each turn to the backend that already holds its cache.

A turn on the wrong backend reads the whole prompt again, at about 25 tokens a
second. Every wait here is cheaper than that.

    ./router.py [--port 8090] [--host 0.0.0.0]

Open /router for the dashboard. /router/json is the same data.
docs/LAYOUT.md gives the measurements behind the defaults.
"""

import argparse, base64, hashlib, http.client, http.server, json, math, os
import queue, re, struct
import select, shutil
import signal, socket
import subprocess, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from collections import OrderedDict, deque
from pathlib import Path

# `prefill` and `generate` are separate settings because they are separate
# work: a prefill is compute bound for tens of minutes, a generation is memory
# bound for seconds. An instance that generates and does not prefill is a
# generator, and turns migrate to it. `pref` orders those, lowest first.
#
# One slot per instance: a slot reading a long prompt holds up every other slot
# on that instance, so a second slot buys a queue, not a second prefill.
BACKENDS = [
    {"name": "solo", "url": "http://127.0.0.1:8080", "pref": 0,
     "prefill": True, "generate": True, "node": 0},
]

# ROUTER_BACKENDS names a JSON file holding another table: the same fields.
BACKENDS_FROM = "the built-in default"
if os.environ.get("ROUTER_BACKENDS"):
    BACKENDS_FROM = os.environ["ROUTER_BACKENDS"]
    BACKENDS = json.loads(Path(BACKENDS_FROM).read_text())
    # Checked here, not where each field is read: a backend missing one would
    # otherwise start, serve, and fail on the turn that first needs it.
    for be in BACKENDS:
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
        if not any(be[job] for be in BACKENDS):
            raise SystemExit(f"[router] no backend in {BACKENDS_FROM} can "
                             f"{job}, so no request could be served")
    # A turn can only leave the instance that read it through the handoff, so
    # an instance that does not generate needs it on.
    if os.environ.get("HANDOFF") == "0" and not all(be["generate"]
                                                    for be in BACKENDS):
        raise SystemExit("[router] HANDOFF=0 keeps every turn on the backend "
                         "that read it, so every backend has to generate")

def prefills(be):
    """May a new conversation have its prompt read on this backend."""
    return be.get("prefill", True)


def generates(be):
    """May a reply be generated on this backend."""
    return be.get("generate", True)


MAX_PINS      = 512    # conversations to remember
FORWARD_TIMEOUT = 7200.0  # longest a backend may take to answer a request
READ_TIMEOUT    = 7200.0  # a read runs 30 to 40 minutes
PIN_PATIENCE  = 20.0   # how long a conversation waits for the backend that
                       # holds its cache before taking a free one instead
POLL          = 2.0    # seconds between backend checks
RATE_WINDOW   = 10.0   # seconds a per-slot rate is measured over
PING_EVERY       = 15.0   # seconds of quiet before a ping. Clients drop a
                          # stream after 300, and a read sends nothing.
POST_TIMEOUT     = 300.0  # a save waits for the slot to finish its turn
DRAIN_DEADLINE   = 1800.0 # how long a drain waits for work already running.
                          # Killing a read wastes the twenty minutes it has run.
PARK_ALL_TIMEOUT = 75.0   # longest one save may take on the way out. Measured
                          # median 11 s, p90 19 s; 77 of 952 saves exceeded 20.
PARK_ALL_BUDGET  = 80.0   # wall-clock cap over all of them. stop-all.sh gives
                          # the router 90 s, and the pin map is written last.
PARK_FLOOR       = 64 * 1024 * 1024   # a real state carries the recurrent
                          # state, about 112 MiB at any length. A smaller file
                          # means the slot had changed hands.
# Disk for the conversation copies, in bytes rather than files: a copy is
# about 115 MiB plus 36.6 KiB a token, so 0.2 to 5.4 GiB at ctx 150000. Size it
# to the disk RUN is on. Too low and a copy still in use is displaced, which
# costs a full re-read.
PARK_BUDGET = int(float(os.environ.get("PARK_BUDGET_GB") or 256) * 1024 ** 3)
# A restored slot is only usable if the state file carries its context
# checkpoints, which needs patches/slot-state-carries-checkpoints.patch.
# Without it a move is followed by a full re-read: set HANDOFF=0.
HANDOFF_ON = os.environ.get("HANDOFF", "1") == "1"
PREFIX_MIN_CHARS = 8000   # about 2000 tokens. A shorter cut is not worth a
                          # file: it only guesses what a later request shares.
BUILD_PATIENCE   = 1800.0 # longest a request waits for another to save the
                          # opening they share
SYSTEM_MIN_CHARS = 2000   # The system prompt is the only cut two sessions
                          # share, because anything deeper carries a first user
                          # message that differs. Claude Code sends 6,100
                          # characters: a minute to read, a fifth of a second
                          # to load from disk.
# Disk for the saved openings, in bytes rather than files: one block runs 0.6
# to 3.7 GB, so a file count means anything. Least recently used goes first.
# Size it to the disk BLOCK_DIR is on.
BLOCK_BUDGET = int(float(os.environ.get("BLOCK_BUDGET_GB") or 64) * 1024 ** 3)
# Deeper openings: a cut where two conversations diverge, read and saved so a
# branch of a session can start from it. Off by default - over two days of real
# traffic here it was built 0 times and loaded 0 times, against several GB of
# disk, and a fork is only visible while its parent still holds a slot.
#
# The machinery stays, and the detection still runs: the `choice` event records
# how deep a fork could have started and whether some conversation's own copy
# already held that cut. tools/cache-report.py reads it. Turn this on if it
# says forks are real here.
DEEP_OPENINGS = os.environ.get("DEEP_OPENINGS", "0") == "1"
WANT_KEEP        = 8      # openings noted as missing but not built yet. They
                          # cost nothing until built, and one dropped is an
                          # opening read again from cold.
IDLE_POLLS       = 2      # polls a slot must have looked idle for before the
                          # builder reads into it. One poll can be two seconds
                          # old, and a slot between two turns looks free in it.
BUILD_POLL       = 10.0   # seconds between builder passes
STALL_RATE = 0.5          # tokens/s. Under this a generating slot is stalled
                          # behind another slot's read, getting one step per
                          # prompt chunk. Measured 0.02 to 0.06 against 6.3 solo.
HISTORY_STEP = 10.0       # seconds per history bucket
HISTORY_KEEP = 60         # buckets kept: ten minutes of slot-time and load
GPU_POLL = 10.0           # seconds between nvidia-smi runs. Each costs 50 to
                          # 100 ms, so the poll collects a child rather than
                          # waiting for one.
RECENT_REQUESTS = 20      # requests the dashboard lists, with how each started
RECENT_FILES = 24         # slot files the dashboard lists. A turn boundary can
                          # write three inside one status push, so this holds
                          # several pushes of them rather than one.
FLOW_LOG = 150            # stage transitions the flow dashboard can replay, newest first
MOUNT_POLL = 30.0         # seconds between disk usage checks
CHARS_PER_TOK = 4.0
REPLY_TOKENS  = 1024   # room to reserve for the reply
MAX_BODY = 256 * 1024 * 1024   # largest request body read into memory. A turn
                       # at ctx 150000 is a few megabytes.

# These endpoints use a slot. The rest are cheap, so they skip the queue.
INFERENCE = {
    "/completion", "/completions", "/v1/completions",
    "/chat/completions", "/v1/chat/completions",
    "/infill", "/v1/messages", "/responses", "/v1/responses",
    "/embedding", "/embeddings", "/v1/embeddings",
}

# An allowlist, not a denylist. The backends run with --agent, which is shell
# and file access with no key, so a path this does not name must never be
# reachable from the public port. PASS_THROUGH adds to it, comma separated.
PASSED = {
    "/health", "/props", "/slots", "/models", "/v1/models",
    "/tokenize", "/detokenize", "/apply-template",
    "/v1/messages/count_tokens", "/v1/messages/apply-template",
}
PASSED |= {p.strip() for p in os.environ.get("PASS_THROUGH", "").split(",")
           if p.strip()}

DROP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "te", "trailers",
                "transfer-encoding", "upgrade", "content-length", "host"}


# What a backend prints under "vision hparams" when it loads an mmproj, and
# what to assume before one has. Pool.vision() reads the real thing.
VISION = {"patch_size": 16, "n_merge": 2,
          "image_min_pixels": 8192, "image_max_pixels": 4194304}
HEADER_B64 = 98304                        # base64 to decode looking for a size


def image_size(head):
    """Width and height from the front of an image file, or None.

    Only the header is needed, so the caller decodes a prefix rather than the
    whole picture."""
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", head[6:10])
    if head[:2] == b"BM" and len(head) >= 26:
        # biHeight is signed, and negative means the rows are stored top-down.
        # Read as-is it made every top-down bitmap unmeasurable, so a 240x120
        # screenshot was charged the 4096 tokens an unreadable header costs
        # instead of 32 - enough to push a request past the 413.
        wide, high = struct.unpack("<ii", head[18:26])
        return abs(wide), abs(high)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        if head[12:16] == b"VP8X":
            w = int.from_bytes(head[24:27], "little") + 1
            h = int.from_bytes(head[27:30], "little") + 1
            return w, h
        if head[12:16] == b"VP8L" and len(head) >= 25:
            bits = int.from_bytes(head[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if head[12:16] == b"VP8 " and head[23:26] == b"\x9d\x01\x2a":
            return struct.unpack("<HH", head[26:30])
    if head[:2] == b"\xff\xd8":           # jpeg: walk the markers to a frame
        i = 2
        while i + 9 < len(head):
            if head[i] != 0xFF:
                i += 1
                continue
            marker = head[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", head[i + 5:i + 9])
                return w, h
            i += 2 + int.from_bytes(head[i + 2:i + 4], "big")
    return None


def image_tokens(payload, vision=None):
    """What a backend charges for one base64 image.

    It resizes the picture to a multiple of patch_size * n_merge, holding the
    aspect ratio and the pixel range, then spends a token per aligned square.
    This is calc_size_preserved_ratio in mtmd-image.cpp followed by
    clip_n_output_tokens; a 240x120 png measured 32 tokens, which is what this
    returns. An unreadable header is charged the most an image can cost, so a
    format we cannot measure is never let in under its weight."""
    vision = vision or VISION
    align = vision["patch_size"] * vision["n_merge"]
    least, most = vision["image_min_pixels"], vision["image_max_pixels"]

    head = payload[:HEADER_B64]
    head = head[:len(head) // 4 * 4]              # whole base64 groups only
    try:
        size = image_size(base64.b64decode(head, validate=False))
    except Exception:
        size = None
    if not size or min(size) <= 0:
        return most // (align * align)

    w, h = size
    up   = lambda x: math.ceil(x / align) * align
    down = lambda x: math.floor(x / align) * align
    near = lambda x: math.floor(x / align + 0.5) * align    # c++ rounds half up
    w_bar, h_bar = max(align, near(w)), max(align, near(h))
    if w_bar * h_bar > most:
        beta = math.sqrt(w * h / most)
        w_bar, h_bar = max(align, down(w / beta)), max(align, down(h / beta))
    elif w_bar * h_bar < least:
        beta = math.sqrt(least / (w * h))
        w_bar, h_bar = max(align, up(w * beta)), max(align, up(h * beta))
    return (w_bar // align) * (h_bar // align)


def images_in(body):
    """Every base64 image in a request body, whichever api sent it."""
    def walk(node):
        if isinstance(node, list):
            for item in node:
                yield from walk(item)
            return
        if not isinstance(node, dict):
            return
        # anthropic: {"source": {"type": "base64", "media_type": "image/png"}}
        source = node.get("source")
        if isinstance(source, dict) and isinstance(source.get("data"), str):
            if str(source.get("media_type", "")).startswith("image/"):
                yield source["data"]
        # openai: {"image_url": {"url": "data:image/png;base64,..."}}. Taking
        # the url wherever it sits counts each picture once, however deep the
        # client nested it.
        for value in node.values():
            if isinstance(value, str) and value.startswith("data:image/"):
                _, _, payload = value.partition("base64,")
                if payload:
                    yield payload
            elif isinstance(value, (dict, list)):
                yield from walk(value)

    try:
        return list(walk(json.loads(body)))
    except Exception:
        return []


def request_cost(body, vision=None):
    """(tokens this request needs, pictures it carries, what they cost).

    Four characters make about one token, except in a picture. Base64
    is hundreds of times longer than what the vision encoder charges,
    so counting it as text refuses a screenshot that would have fit.

    All three come back together because the walk is the expensive
    part: it parses a body of several megabytes and decodes the header
    of every picture. Asking twice doubled that on the request path.

    The dashboard needs pictures counted apart from text, or it cannot
    tell a 4,000 token screenshot from 16,000 characters of prose."""
    text, charged, count = len(body), 0, 0
    for payload in images_in(body):
        text -= len(payload)
        charged += image_tokens(payload, vision)
        count += 1
    return int(max(0, text) / CHARS_PER_TOK) + charged + REPLY_TOKENS, count, charged


def token_estimate(body, vision=None):
    """Tokens this request needs. The first of request_cost's three."""
    return request_cost(body, vision)[0]


def conversation_id(body):
    """Identify a conversation by its opening, which does not change.

    The system messages and the first user message are the same on every turn.
    Anything after them grows, so including it would give every turn its own
    id and defeat the pinning."""
    try:
        req = json.loads(body)
    except Exception:
        return None

    messages = req.get("messages")
    if isinstance(messages, list):
        opening = []
        for message in messages:
            # A client may send anything. Without this a list of strings, or
            # one null, threw AttributeError here - before a status line had
            # been sent, so the client got a dropped connection rather than
            # the 400 a backend would have answered. Every other body reader
            # in this file guards the same way.
            if not isinstance(message, dict):
                break
            role = message.get("role")
            if role == "assistant":
                break                      # the reply. Everything after grows.
            text = message.get("content")
            if isinstance(text, list):     # multimodal message
                text = "".join(part.get("text", "") for part in text
                               if isinstance(part, dict))
            opening.append(f"{role}:{text}")
            if role == "user":
                break                      # first user message, and stop
        start = "\n".join(opening)
    elif isinstance(req.get("prompt"), str):
        start = req["prompt"]
    else:
        return None
    if not start:
        return None
    return hashlib.sha256(start.encode("utf-8", "replace")).hexdigest()


WEB = Path(__file__).with_name("web")          # the dashboard, a static app
# Everything a run writes. bin/common.sh sets RUN and exports nothing else
# about it, so a box that moves this has to export RUN for the router too.
RUN_DIR  = Path(os.environ.get("RUN")
                or Path(__file__).resolve().parent.parent / "run")
SLOT_DIR = RUN_DIR / "slots"
# System blocks go on the faster disk where there are two: each is read at the
# start of a new session, and BLOCK_BUDGET bounds what they take. A
# conversation's copy stays with the rest - written once, read at most once,
# and several gigabytes - under PARK_BUDGET. Two budgets because they can be
# two disks; beside the rest by default, and BLOCK_DIR moves them.
BLOCK_DIR = Path(os.environ.get("BLOCK_DIR") or RUN_DIR / "blocks")

# Debugging only, off unless CAPTURE names a directory. A prompt read again and
# again is a prompt whose opening changed, and nothing else here says what. A
# capture holds a whole conversation, so only the newest few are kept.
CAPTURE_DIR = Path(os.environ["CAPTURE"]) if os.environ.get("CAPTURE") else None
CAPTURE_KEEP = 24

# One JSON line per cache decision, in a dated file. The dashboard shows the
# present; improving the cache needs the past - which openings paid for
# themselves, which were never loaded, how long a want waited, whether a fork
# started from its parent's opening. Events go through a queue and a thread, so
# no request touches the disk for this. CACHE_LOG=0 turns it off.
CACHE_LOG_ON = os.environ.get("CACHE_LOG", "1") == "1"
CACHE_LOG_DIR = Path(os.environ.get("CACHE_LOG_DIR") or RUN_DIR)

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",  ".json": "application/json",
        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".map": "application/json"}



EVICTED_RE = re.compile(r"removing oldest entry \(size = ([\d.]+) MiB\)")
SKIPPED_RE = re.compile(r"prompt state size ([\d.]+) MiB exceeds cache size limit")
STATE_RE = re.compile(r"cache state: (\d+) prompts, ([\d.]+) MiB "
                      r"\(limits: ([\d.]+) MiB")
# The backend says its limit once, at startup. Without this the dashboard has
# nothing to show until the backend next touches its cache, which needs
# traffic: cpu0_0 showed a pre-restart 16384 MiB for an hour after coming back
# at 8192.
LIMIT_RE = re.compile(r"prompt cache is enabled, size limit: (\d+) MiB")
# A slot restored from a file has no checkpoint, and this model's recurrent
# state cannot be rewound without one, so the backend re-reads the whole prompt
# and says so. Each one is a request that took minutes longer than it needed.
REREAD_RE = re.compile(r"forcing full prompt re-processing")
# A checkpoint the slot actually stepped back to, at -lv 4. Each one is a turn
# that re-read only the tokens past this position instead of the whole prompt.
CHECKPOINT_RE = re.compile(r"restored context checkpoint \(pos_min = \d+, "
                           r"pos_max = \d+, n_tokens = (\d+)")
# What the backend mapped at startup, and the tensor it leaves on disk. Mapped
# less lazy is what must stay in the page cache; when it does not, prefill
# falls from 37 tokens a second to single digits. The shard sizes on disk say
# something else, since one shard is almost all the lazy tensor.
MAPPED_RE = re.compile(r"CPU_Mapped model buffer size = +([\d.]+) MiB")
LAZY_RE = re.compile(r"add: tensor \S+ \(size = +([\d.]+) MiB\) lazy read enabled")


CONFIG_RE = re.compile(r"(n_ctx|n_batch|n_ubatch|kv_unified|n_slots)\s*=\s*"
                       r"'?([\w.]+)'?")
CONFIG_KEYS = ("n_ctx", "n_batch", "n_ubatch", "kv_unified", "n_slots")

# The mmproj's geometry, printed once under "vision hparams". The word break
# matters: n_merges (the tokenizer's, in the hundreds of thousands) is printed
# long before n_merge, and would otherwise be read instead.
VISION_RE = re.compile(r"\b(patch_size|n_merge|image_min_pixels|image_max_pixels)"
                       r"\b\s*[:=]\s*(\d+)")


RATE_FLOOR = 1.0     # seconds. Below this a count is not a rate: just after a
                     # reset a token lands against no elapsed time and the
                     # division runs away.


def per_second(tokens, seconds):
    """A rate, or zero when there is not enough time to divide by."""
    return round(tokens / seconds, 1) if seconds and seconds >= RATE_FLOOR else 0


def read_config(lines):
    """The settings a backend was started with, from what it printed.

    llama-server does not report these on /props, so the only source is the log
    it writes once at startup. A restart appends to the same file, so the first
    of each wins."""
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
    """The vision encoder's geometry, from what a backend printed at startup.

    Returns None unless the whole of it is there, so a backend with no mmproj
    and a half-read log both say "nothing" rather than something wrong."""
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

    The counters run from the start of the log, which the backend truncates on
    each start, so they cover the life of that backend.

    `sink`, when given, is called as sink(kind, value) for the events worth a
    line of their own in the cache event log."""

    # The periodic and startup lines stay in the totals. Everything else
    # happened to one request, so it goes to the sink.
    SUNK = ("evicted", "skipped", "reread", "checkpoint")

    def __init__(self, path, sink=None):
        self.path = Path(path)
        self.sink = sink
        self.offset = 0
        self.inode = None
        # Bytes that were already in the log when this router started. They
        # belong in the totals, which cover the life of the backend, but not in
        # the sink: bin/restart-router.sh exists to restart the router with the
        # backends left up, and every restart wrote that backend's whole
        # history into today's cache-events file as if it had just happened.
        # Four logs of 3.7 to 9.8 MB replayed 473 of them.
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
        # Smaller, or a different file. The inode matters as much as the size:
        # restart-backend.sh and start-all.sh both move the old log aside and
        # the new backend creates its own, so watching the size alone missed
        # every restart whose new log passed the old offset before the next
        # poll - the counters kept the dead backend's totals and the new log's
        # opening lines, which carry the cache size limit, were skipped for
        # good. read_settings already reads the inode for the same reason.
        if size < self.offset or (self.inode is not None and inode != self.inode):
            self.offset = 0
            self.replay_to = 0        # a new log, so none of it predates this run
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
        # llama-server's log is block buffered, so a read can land mid line.
        # Taking the offset to the end anyway split that line across two polls
        # and neither half matched anything - and the cache size limit is
        # printed once, at startup, so losing that one is losing it for good.
        cut = fresh.rfind(b"\n") + 1
        if cut < len(fresh):
            self.offset -= len(fresh) - cut
            fresh = fresh[:cut]
        # Counted in bytes, like the offset: a decoded length is characters,
        # and one non-ascii line in the trace would put the two out of step.
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


class EventLog:
    """Append cache events to a dated JSONL file, one json object a line.

    Telemetry, not data: losing a line costs a number in a report, and nothing
    here may make a request wait or fail. A writer queues the row and returns,
    a thread empties the queue, and a full queue drops the newest event. The
    file is named by the day its rows fall in, so the log rotates itself."""

    def __init__(self, directory=None, on=CACHE_LOG_ON, name="cache-events",
                 maxsize=10000):
        self.on = on
        self.directory = Path(directory) if directory else CACHE_LOG_DIR
        self.name = name
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)
        self.handle = None
        self.day = None
        self.dropped = 0
        if self.on:
            threading.Thread(target=self._run, daemon=True).start()

    def write(self, event, **fields):
        """Queue one event. Never blocks, never raises."""
        if not self.on:
            return
        row = {"ts": round(time.time(), 3), "event": event}
        row.update({k: v for k, v in fields.items() if v is not None})
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            # Telemetry that costs a request has cost too much. Count the loss.
            self.dropped += 1

    def flush(self):
        """Wait until everything written so far has reached the file."""
        self.queue.join()

    def _run(self):
        while True:
            row = self.queue.get()
            try:
                self._emit(row)
            except Exception:
                # One bad row must not end the thread, or every event after it
                # is lost too.
                self.dropped += 1
            finally:
                self.queue.task_done()

    def _emit(self, row):
        day = time.strftime("%Y%m%d", time.localtime(row["ts"]))
        if day != self.day or self.handle is None:
            if self.handle:
                self.handle.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.handle = (self.directory / f"{self.name}-{day}.jsonl").open("a")
            self.day = day
        self.handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.handle.flush()


# The one log every cache hook writes to. Tests replace it with an EventLog on
# a temporary directory; the hooks look the name up at call time.
EVENTS = EventLog()


# Config files the dashboard offers for download. They are built per request
# so the address inside matches the one the reader reached the dashboard on.
CONFIG_FILES = {"opencode": "opencode.json", "claude": "settings.json"}

# The provider id in a generated OpenCode config. It names the machine, not the
# model: one client can reach two of these boxes, and this is how it tells them
# apart and keys its own settings. PROVIDER overrides the hostname.
PROVIDER = os.environ.get("PROVIDER") or socket.gethostname().split(".")[0] or "llama"


HOST_RE = re.compile(r"^(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$")


def host_only(host):
    """A Host header that is a host and a port, or None.

    The generated client configs carry this address, and a client keeps one
    for months. It arrives in a header anyone can set, so anything that is not
    plainly a name or an address with a port is refused rather than written
    into a file the reader will trust."""
    host = (host or "").strip()
    return host if HOST_RE.match(host) else None


def client_config(kind, host, model, n_ctx):
    """Build a client config for this router, or return None.

    The long timeouts are not padding. A backend reads at about 37 tokens a
    second on an empty box and sends nothing while it does, so a first turn on
    a full window is silent for hours. The divisor is the worst case, not the
    best: cpu0_0 read 114,354 tokens at 17.2 to 20.6 a second on 13 September
    with the other instances busy. At 25 the config was already short of a real
    cold read."""
    base = f"http://{host}"
    patience_ms = max(3600000, n_ctx // 15 * 1000)
    if kind == "opencode":
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": f"{PROVIDER}/{model}",
            # Title generation and other small jobs. Left unset it would reach
            # for a model this backend does not have.
            "small_model": f"{PROVIDER}/{model}",
            # This model is private to this machine, so nothing about a
            # conversation should leave it.
            "share": "disabled",
            "provider": {
                PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Qwen3.8 Flash Next ({PROVIDER})",
                    "options": {
                        "baseURL": f"{base}/v1",
                        # Name the session in every request, so the router
                        # does not have to guess it from the prompt.
                        "setCacheKey": True,
                        "timeout": patience_ms,
                        "headerTimeout": patience_ms,
                        "chunkTimeout": patience_ms,
                        "apiKey": "not-used-but-some-clients-require-one",
                    },
                    "models": {
                        model: {
                            "name": "Qwen3.8 Flash Next (MTP)",
                            "reasoning": True,
                            "attachment": True,
                            "modalities": {
                                "input": ["text", "image"],
                                "output": ["text"],
                            },
                            "interleaved": {"field": "reasoning_content"},
                            "limit": {"context": n_ctx, "output": 32768},
                        }
                    },
                }
            },
        }
    if kind == "claude":
        # The backends serve the anthropic messages api, so Claude Code needs
        # only the address. There is no key, but the client insists on one.
        return {
            "env": {
                "ANTHROPIC_BASE_URL": base,
                "ANTHROPIC_AUTH_TOKEN": "not-used-but-some-clients-require-one",
                "ANTHROPIC_MODEL": model,
                # Gateway discovery only keeps ids containing "claude" or
                # "anthropic", so this model has to be offered by hand.
                "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen3.8 Flash Next (MTP)",
                # Background work uses the haiku slot. Left alone it names a
                # model this backend does not have.
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_SMALL_FAST_MODEL": model,
                "API_TIMEOUT_MS": str(patience_ms),
                # Both watchdogs give up after five minutes of quiet by
                # default, which a prompt read outlasts easily.
                "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(patience_ms),
                "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": str(patience_ms),
                "API_FORCE_IDLE_TIMEOUT": "0",
                # Nothing here needs the internet, and a slow backend should
                # not wait on feature flags.
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                # The attribution block sits first in the system prompt and
                # carries a per-conversation fingerprint. Dropping it lets two
                # sessions share the system prompt, which is the difference
                # between reading it once and reading it every time.
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                # Pre-release body fields draw a 400 from a backend that does
                # not know them.
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                # Claude Code does not know this model, so it would assume a
                # 200k window and let a session outgrow the slot. State the
                # real size, so auto-compact runs at the right point.
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(n_ctx),
            }
        }
    return None


# The two shelves an opening can sit on, as the file names carry them. A
# conversation's own copy must not be able to look like one: adopt_files sorts
# the slot directory by these prefixes, so a key beginning with one put a
# private cache on the shared shelf and lost the conversation its restore.
SHELF_MARKS = ("base-", "deep-")


def copy_is_current(record):
    """True when this conversation's copy on disk is of the turn it last ran.

    A copy behind the slot is still a prefix, so it is worth keeping - but it
    is not worth keeping *instead of* a newer one. ensure_parked and park_all
    asked only whether a copy existed, so on a pool where nothing else writes
    one - every backend reading, which is the shipped default and what
    generator() returns None for - a conversation's copy froze at its first
    park and every later turn re-read everything after it."""
    return bool(record.get("parked")) and record.get("parked_turn") == record.get("turns")


def file_safe(key):
    """A conversation key that also works as a file name.

    llama.cpp refuses a colon, a path separator and a control character;
    each becomes a dash. A key that starts like an opening is moved out
    of the way, so a client cannot name a shelf. The key must be unique
    and stable, not readable."""
    keep = "-._"
    safe = "".join(c if c.isalnum() and c.isascii() or c in keep else "-"
                   for c in key).strip("-. ") or "conversation"
    # llama.cpp's fs_validate_filename refuses ".." anywhere in a name and
    # anything over 255 characters, and it is checked before the save is even
    # attempted (common/common.cpp). A key carrying either came back 400 on
    # every save and every restore, so that conversation was never parked and
    # re-read its whole prompt every turn - with nothing in the log but
    # "park failed". The tail is kept rather than the head because a key is
    # usually prefixed, not suffixed, and ".park" goes on the end.
    while ".." in safe:
        safe = safe.replace("..", ".")
    safe = safe[-200:].strip("-. ") or "conversation"
    return "c-" + safe if safe.startswith(SHELF_MARKS) else safe


def short_key(conv):
    """A conversation key, short enough to read and still telling apart.

    A subagent's key is its session with the agent joined on the end, so both
    ends have to show, or a session and every subagent under it read alike."""
    conv = conv or ""
    return conv[:8] if len(conv) <= 36 else f"{conv[:8]}/{conv[-6:]}"


def mark_shelf(mark):
    """The "base-" / "deep-" mark of an opening, or a want record, as a shelf.

    The marks keep their trailing dash because the builder passes them to file
    names."""
    mark = mark.get("mark") if isinstance(mark, dict) else mark
    return (mark or "deep-").rstrip("-")


def session_key(headers):
    """Name the conversation from the client's own headers, or return None.

    Claude Code sends its session id on every request, which beats guessing
    from the opening of the prompt. A subagent runs its own prompt, so it is a
    separate conversation."""
    lower = {str(name).lower(): value for name, value in dict(headers).items()}
    session = (lower.get("x-claude-code-session-id") or "").strip()
    if not session:
        return None
    agent = (lower.get("x-claude-code-agent-id") or "").strip()
    # The key becomes the name of a slot file, so it may only hold characters
    # a backend will accept in one. A colon is not one of them.
    return file_safe(f"{session}-{agent}") if agent else file_safe(session)


def client_kind(headers):
    """Which client sent this, by its user agent, or None.

    The two worth telling apart send their own name. Anything else keeps the
    first word of its agent string, so a new client shows up in the event log
    under its own name without a code change."""
    agent = (dict(headers).get("User-Agent") or "").lower()
    if "claude" in agent:
        return "claude-code"
    if "opencode" in agent:
        return "opencode"
    return agent.split("/")[0][:24] or None


# A keep-alive must be in the protocol the client speaks. A comment keeps an
# OpenAI stream alive, but is not data to an anthropic parser, which reports a
# stream that ended before any data arrived. That protocol has a ping event.
PING = b": ping\n\n"
ANTHROPIC_PING = b'event: ping\ndata: {"type": "ping"}\n\n'


def anthropic(path):
    """True for the endpoint that speaks the anthropic protocol."""
    return "/messages" in (path or "")


def ping_for(path):
    """The keep-alive this endpoint's client understands."""
    return ANTHROPIC_PING if anthropic(path) else PING


def sse_event(name, data):
    """One named SSE event, laid out as the backends lay theirs out."""
    return (f"event: {name}\ndata: ".encode()
            + json.dumps(data).encode() + b"\n\n")


def read_event(raw):
    """The name and the json of one SSE event, or (None, None).

    Anything else is left to the caller to pass on untouched: a comment, a
    part of an event, or a body that is not an event stream at all."""
    name, data = None, []
    for line in raw.split(b"\n"):
        if line.startswith(b"event:"):
            name = line[6:].strip().decode("utf-8", "replace")
        elif line.startswith(b"data:"):
            data.append(line[5:].strip())
    if name is None or not data:
        return None, None
    try:
        fields = json.loads(b"\n".join(data))
    except ValueError:
        return None, None
    return (name, fields) if isinstance(fields, dict) else (None, None)


def opening_event(path, body):
    """The event a stream of this protocol has to begin with, or nothing.

    An anthropic stream is a message being built. A ping may sit
    anywhere inside one, but a stream that has only pinged has begun no
    message, and the client reports a 502 that was never sent. The
    reply here comes half an hour after the stream opens, so this is
    all the client holds on to until then.

    The id is invented, because the router opens the stream before it
    has a backend. Nothing after message_start carries one to disagree.
    The usage counts are zero, and AnthropicSplice moves the real ones
    onto the closing message_delta.

    An OpenAI stream may open with a comment, so it needs no event."""
    if not anthropic(path):
        return b""
    try:
        model = json.loads(body).get("model")
    except Exception:
        model = None
    return sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_" + os.urandom(12).hex(), "type": "message",
                    "role": "assistant", "model": model or "unknown",
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}})


class AnthropicSplice:
    """Join a backend's stream onto one the router has already opened.

    The router sends message_start when the stream opens, so the backend's own
    must come out: two in one stream is not a message any parser will follow.
    The one thing it carries that no later event does is the prompt token
    count, which a client sizes its context window with, so that moves onto the
    closing message_delta.

    A socket delivers whatever arrived, not whole events, so a part event waits
    for the rest of it."""

    def __init__(self):
        self.rest = b""
        self.usage = None
        # What the client ends up seeing: prompt counts from message_start,
        # the generation count from message_delta. Nothing parses this stream
        # twice to find out what it was told.
        self.reported = {}

    def feed(self, chunk):
        """What to pass on for this piece of the backend's stream."""
        self.rest += chunk
        out = []
        while True:
            event, sep, rest = self.rest.partition(b"\n\n")
            if not sep:
                break
            self.rest = rest
            out.append(self._one(event + sep))
        return b"".join(out)

    def tail(self):
        """Whatever was left when the stream ended."""
        last, self.rest = self.rest, b""
        return last

    def _one(self, raw):
        name, data = read_event(raw)
        if name == "message_start":
            if self.usage is None:
                self.usage = (data.get("message") or {}).get("usage") or {}
                self.reported.update(self.usage)
            return b""
        if name == "message_delta" and self.usage:
            # The backend counts what it generated, so its own figures win.
            data["usage"] = merged = dict(self.usage, **(data.get("usage") or {}))
            self.reported.update(merged)
            self.usage = {}
            return sse_event(name, data)
        return raw


def wants_ping(content_type, content_length):
    """True when extra bytes can be inserted into this reply safely.

    Only a streamed event stream qualifies. A counted body has no room, and
    the extra bytes would corrupt anything else."""
    if content_length:
        return False
    return "text/event-stream" in (content_type or "")


def wants_usage(body):
    """True when an openai stream request asked for the usage chunk itself."""
    try:
        fields = json.loads(body)
    except Exception:
        return False
    opts = fields.get("stream_options") if isinstance(fields, dict) else None
    return bool(isinstance(opts, dict) and opts.get("include_usage"))


def with_usage(body):
    """A copy of the body with stream_options.include_usage set, or None.

    A streamed openai reply carries no usage unless the request asks for it,
    so the router asks on the client's behalf and takes the chunk back out
    unless the client wanted it. The answer arrives as its own final chunk with
    no choices (probe-usage.py). The client's reply is unchanged, and the
    router gets to read what its cache saved."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    opts = fields.get("stream_options")
    opts = dict(opts) if isinstance(opts, dict) else {}
    opts["include_usage"] = True
    fields["stream_options"] = opts
    try:
        return json.dumps(fields).encode()
    except (TypeError, ValueError):
        return None


class OaiUsageSplice:
    """Read the usage figures out of an openai stream as they pass.

    The chunk carrying them holds no choices, so it is easy to take out again
    when the client did not ask for it. The only bytes removed are the ones
    the router's own ask put there."""

    def __init__(self, strip=False):
        self.rest = b""
        self.strip = strip
        self.usage = {}

    def feed(self, chunk):
        self.rest += chunk
        out = []
        while True:
            event, sep, rest = self.rest.partition(b"\n\n")
            if not sep:
                break
            self.rest = rest
            out.append(self._one(event + sep))
        return b"".join(out)

    def tail(self):
        last, self.rest = self.rest, b""
        return last

    def _one(self, raw):
        line = raw.strip()
        if line.startswith(b"data: ") and b'"usage"' in line:
            try:
                obj = json.loads(line[6:])
            except ValueError:
                return raw
            if obj.get("choices") == [] and obj.get("usage"):
                self.usage = obj["usage"]
                if self.strip:
                    return b""
        return raw


def prompt_key(body):
    """Name the conversation from the body, or return None.

    OpenCode sends prompt_cache_key when setCacheKey is on, which serves the
    same purpose as Claude Code's session header."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    key = fields.get("prompt_cache_key")
    if not isinstance(key, str) or not key.strip():
        return None
    return file_safe(key.strip())     # the client chose it, so check it


def text_of(value):
    """The words in a system prompt, whether it is a string or a list of parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value
                       if isinstance(part, dict))
    return ""


def message_shape(message):
    """Everything about one message that a later request has to match.

    A cut name says "two requests open the same way", and the backend renders
    the whole message, not the words in it. So this hashes the message as it
    was sent, minus the keys a client varies without changing the prompt.

    Sorted keys, and separators without spaces, so the same message always
    gives the same bytes. Anything that will not serialise falls back to its
    text, which is what this read before."""
    try:
        return json.dumps(without_ignored(message),
                          sort_keys=True, separators=(",", ":"),
                          default=str).encode()
    except (TypeError, ValueError):
        return text_of(message.get("content")).encode("utf-8", "replace")


# Fields a client may set per request without changing what the template
# renders. `cache_control` is Anthropic's own cache hint and moves down the
# transcript every turn, so hashing it would give every turn a new opening.
IGNORED_KEYS = frozenset(("cache_control",))


def without_ignored(value):
    """The same body with IGNORED_KEYS dropped, however deep they sit.

    `cache_control` is a property of a content block, not of a message - the
    api puts it on a text, image or tool_result block - so dropping it from
    the message's own keys dropped nothing at all. The marker then moved down
    the transcript every turn and every cut name above it changed with it,
    which is exactly the churn IGNORED_KEYS exists to stop."""
    if isinstance(value, dict):
        return {k: without_ignored(v) for k, v in value.items()
                if k not in IGNORED_KEYS}
    if isinstance(value, list):
        return [without_ignored(v) for v in value]
    return value


def closes(message):
    """True when a template can end a prompt after this message.

    An assistant message that calls a tool is the model part way through its
    turn, and the template will not close one: "Cannot continue an assistant
    message that contains tool calls". An opening cut there renders nothing and
    is never saved. The tool result answering it is the next cut, and renders."""
    if message.get("role") != "assistant":
        return True
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    return not (isinstance(content, list)
                and any(isinstance(part, dict) and part.get("type") == "tool_use"
                        for part in content))


def prompt_cuts(body, least=PREFIX_MIN_CHARS):
    """Every point in this request that another request could share.

    A restored slot carries no checkpoints, so a block only helps when
    the next request extends it exactly. Each cut is therefore at a
    message boundary, and is hashed onto the one before it. Two requests
    that open alike share the name of every cut in that opening.

    Returns the cuts deepest last, the messages they cut, the system
    prompt when the request keeps it apart, and the tools it declares."""
    try:
        fields = json.loads(body)
    except Exception:
        return [], [], "", []
    if not isinstance(fields, dict):
        return [], [], "", []
    messages = fields.get("messages")
    if not isinstance(messages, list):
        return [], [], "", []

    system = text_of(fields.get("system"))     # /v1/messages keeps it apart
    tools = fields.get("tools")
    tools = tools if isinstance(tools, list) else []
    running = hashlib.sha256()
    size = 0
    cuts = []
    # The template renders the tools inside the system block, so they open the
    # prompt rather than follow it. For Claude Code they are most of it: 25
    # tools and 56,371 characters against 6,111 of system prompt, the same
    # every session. A client that gains or loses one sends a different
    # opening, which is what several shelves are for.
    written = json.dumps(tools, separators=(",", ":")) if tools else ""
    if system or written:
        # "replace", as conversation_id already does: json allows a lone
        # surrogate (\ud800), str.encode refuses one, and the throw came out of
        # _route before any status line had been sent - so the client got a
        # dropped connection rather than an answer.
        running.update(b"system\x00" + system.encode("utf-8", "replace"))
        running.update(b"tools\x00" + written.encode("utf-8", "replace"))
        size += len(system) + len(written)
        # Only a request that keeps the system prompt apart has anything to
        # render before its first message. An openai body carries its system
        # prompt as that first message, and tools alone are not a prompt: the
        # template refuses an empty one, so the opening is never saved. Its cut
        # is the one at the first message, which renders prompt and tools
        # together the way the template lays them out.
        if system and size >= SYSTEM_MIN_CHARS:
            cuts.append((-1, running.hexdigest()[:16]))   # before any message
    # An openai body's first message is the same opening as an anthropic body's
    # system field, so it earns a cut on the same terms. Otherwise a client
    # whose rules are shorter than a conversation shares nothing at all.
    lead = leading_system(messages)
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            break
        # The whole message, not just its words. text_of reads `text` parts
        # only, so a tool_use, a tool_result and an openai tool_calls list all
        # hashed to nothing: two agentic conversations sharing a system prompt
        # and a first user message produced the same cut names at every depth,
        # however different the tools they had called and the files they had
        # read. The router then loaded one session's saved block into the
        # other's slot, which diverges at the first tool call - a restored slot
        # has no checkpoint, so that is a full re-read, recorded as a hit.
        running.update(f"{message.get('role')}\x00".encode())
        running.update(message_shape(message))
        size += len(text_of(message.get("content")))
        bar = SYSTEM_MIN_CHARS if index < lead else least
        if size >= bar and closes(message):
            cuts.append((index, running.hexdigest()[:16]))
    return cuts, messages, system, tools


def deepest_shared(cuts, known):
    """The furthest cut in this request that something else also has."""
    for cut in reversed(cuts):
        if cut[1] in known:
            return cut
    return None


def common_prefix(first, second):
    """The text two renderings share. It ends where the messages differ."""
    limit = min(len(first), len(second))
    n = 0
    while n < limit and first[n] == second[n]:
        n += 1
    return first[:n]


def request_shape(body):
    """Describe a request without keeping its text.

    A backend that refuses a request usually objects to its shape, not its
    words. This records the shape and leaves the prompt out of it."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    messages = fields.get("messages")
    roles = [m.get("role") for m in messages if isinstance(m, dict)] \
        if isinstance(messages, list) else []
    return {"system": type(fields["system"]).__name__ if "system" in fields else None,
            "roles": roles,
            "tools": len(fields.get("tools") or [])}


SYSTEM_ROLES = ("system", "developer")


def leading_system(messages):
    """How many messages at the front of this list are the system prompt."""
    lead = 0
    while lead < len(messages) and isinstance(messages[lead], dict) \
            and messages[lead].get("role") in SYSTEM_ROLES:
        lead += 1
    return lead


def hoist_system(body):
    """Make a late system message one the template will take, where it is.

    This model's template refuses a system message that is not at the
    front, so an unchanged request cannot be served at all. Where the
    change lands decides how much of the prompt can be reused.

    Claude Code ends every turn with a token counter as a system
    message, and its value differs every request. Carried to the front
    it ends the shared prefix a few thousand tokens in, and the whole
    conversation behind it is read again.

    So the message keeps its index, and only its role changes. That
    leaves two user messages in a row, which the template accepts. A
    body already in order comes back unchanged."""
    try:
        fields = json.loads(body)
    except Exception:
        return body
    if not isinstance(fields, dict):
        return body
    messages = fields.get("messages")
    if not isinstance(messages, list):
        return body

    lead = leading_system(messages)
    later = [m for m in messages[lead:]
             if isinstance(m, dict) and m.get("role") in SYSTEM_ROLES]
    if not later:
        return body                     # already in order

    fields["messages"] = messages[:lead] + [
        dict(m, role="user")
        if isinstance(m, dict) and m.get("role") in SYSTEM_ROLES else m
        for m in messages[lead:]]
    return json.dumps(fields).encode()


PLACE_RE = re.compile(r"(\d+)_(\d+)$")


def by_place(name):
    """Sort key from a backend name: the socket, then the instance on it.

    A name is type, socket, instance, so gpu0_0 sits with cpu0_0 rather than
    with the other gpu names. A name without a place sorts last."""
    found = PLACE_RE.search(name or "")
    if not found:
        return (9, 9, name or "")
    return (int(found.group(1)), int(found.group(2)), name)


def _say(line):
    print(line, flush=True)


def wants_stream(body):
    """True when the client asked for a streamed reply."""
    try:
        fields = json.loads(body)
    except Exception:
        return False
    return isinstance(fields, dict) and bool(fields.get("stream"))


def template_route(path):
    """Where to ask this backend what a body renders to.

    The anthropic route converts the body first, as generation does, so a tool
    call renders instead of being refused."""
    return ("/v1/messages/apply-template" if (path or "").startswith("/v1/messages")
            else "/apply-template")


def read_only(body, slot=None):
    """The same request, asking for one token instead of an answer.

    Reading the prompt is the slow part, and it happens on a backend that
    reads. Asking for one token does that without settling where the answer
    comes from, so the router can choose once the read is done and it knows
    which backend is free."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    fields = dict(fields)
    fields["stream"] = False          # the answer is thrown away
    fields["verbose"] = True          # so the reply names the slot it used
    fields.pop("stream_options", None)
    # None at all. A generated token lands in the slot, so the slot would hold
    # the prompt and one more, while the request that follows carries the
    # prompt alone. A restored slot has no checkpoint to rewind to, so the
    # backend would re-read everything.
    if "n_predict" in fields:
        fields["n_predict"] = 0
    else:
        fields["max_tokens"] = 0
    # The responses api carries its own name for the same thing, and
    # llama.cpp's converter copies it over max_tokens unconditionally
    # (server_chat_convert_responses_to_chatcmpl). Left alone, this "read
    # only" pass generated the whole reply - once here and once again on the
    # forward - and the generated tokens landed in the slot, which is the one
    # thing the zero is for.
    if "max_output_tokens" in fields:
        fields["max_output_tokens"] = 0
    if slot is not None:
        fields["id_slot"] = slot      # say which slot, rather than ask after
    return fields


def http_post(url, path, payload, timeout=POST_TIMEOUT):
    """POST json and read the reply. Used for the slot save and restore."""
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as err:
        # A backend puts the reason in the body. Without it a refused save
        # reads as "400" and no more.
        raise OSError(f"{err.code} on {path}: {said(err)}") from None


def said(err):
    """The reason a backend gave, out of the body of its error reply."""
    try:
        body = json.loads(err.read().decode("utf-8", "replace"))
    except Exception:
        return err.reason
    trouble = body.get("error") if isinstance(body, dict) else None
    if isinstance(trouble, dict):
        return trouble.get("message") or err.reason
    return trouble or err.reason


class Gone(Exception):
    """The client stopped waiting, so what it asked for is no longer wanted."""


def http_post_wanted(url, path, payload, timeout, wanted, every=2.0):
    """POST to a backend, and stop if nobody is waiting for the answer.

    An abandoned read holds a slot for tens of minutes, and the client that gave
    up will send the turn again. Closing the connection cancels the task in
    llama.cpp and frees the slot.

    Use shutdown(), not close(): the reading thread holds the socket open through
    its file object, so close() alone never reaches the backend.

    Raises Gone when the client has left. Otherwise the same as http_post."""
    parts = urllib.parse.urlsplit(url)
    secure = parts.scheme == "https"
    opener = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    conn = opener(parts.hostname, parts.port or (443 if secure else 80),
                  timeout=timeout)
    prefix = parts.path.rstrip("/")
    got = {}

    def run():
        try:
            conn.request("POST", prefix + path, json.dumps(payload).encode(),
                         {"Content-Type": "application/json"})
            reply = conn.getresponse()
            body = reply.read()
            if reply.status >= 400:
                got["error"] = OSError(f"{reply.status} on {path}: "
                                       f"{said_in(body) or reply.reason}")
            else:
                got["answer"] = json.loads(body) if body else {}
        except Exception as err:
            got["error"] = err

    thread = threading.Thread(target=run, name="read", daemon=True)
    thread.start()
    while True:
        thread.join(every)
        if not thread.is_alive():
            break
        if not wanted():
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)   # the backend sees this
            except (OSError, AttributeError):
                pass                   # already gone, which is what we wanted
            conn.close()               # the backend cancels what it was doing
            thread.join(10)
            raise Gone("the client stopped waiting")
    conn.close()
    if "error" in got:
        raise got["error"]
    return got.get("answer") or {}


def said_in(body):
    """The reason inside a backend's error body, or None."""
    try:
        trouble = json.loads(body).get("error")
    except Exception:
        return None
    if isinstance(trouble, dict):
        return trouble.get("message")
    return trouble


def capture(conv, body):
    """Write one request body down, for comparing two turns offline.

    Kept per conversation. One busy client sends a turn a minute and a quiet
    one a turn an hour, so a single list of the newest bodies holds only the
    busy one, and the quiet one is what is being looked for."""
    if CAPTURE_DIR is None:
        return
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        tag = short_key(conv).replace("/", "_")
        # Nanoseconds, so two bodies of one conversation cannot share a name.
        # Fixed width, so the names sort by time.
        name = f"{time.time_ns()}-{tag}.json"
        (CAPTURE_DIR / name).write_bytes(body)
        old = sorted(CAPTURE_DIR.glob(f"*-{tag}.json"))[:-CAPTURE_KEEP]
        for spent in old:
            spent.unlink(missing_ok=True)
    except OSError as err:
        print(f"[router] could not write the capture: {err}", flush=True)


def how_started(warm, recalled, loaded):
    """Name what a request extended instead of reading.

    `warm`: its own cache was still in a slot on the backend that serves it.
    `recalled`: its own copy came back from disk. `loaded`: a saved opening
    was put in the slot for it. Otherwise it read its whole prompt."""
    if recalled:
        return "recalled"
    if loaded:
        return "saved prompt"
    if warm:
        return "warm slot"
    return "cold"


def disk_summary(pins, openings, opening_bytes, wants):
    """What the two slot directories hold against their budgets."""
    copies = [(conv, p) for conv, p in pins.items() if p.get("parked")]
    kinds = [shelf_of(name) for name in openings.values()]
    return {"copies": {"count": len(copies),
                       "bytes": sum(p.get("bytes") or 0 for _, p in copies),
                       "budget": PARK_BUDGET},
            # One budget over both shelves, so each count is what that kind
            # happens to be using and neither is a cap on its own.
            "openings": {"count": len(openings),
                         "bytes": sum(opening_bytes.values()),
                         "budget": BLOCK_BUDGET},
            "bases": {"count": kinds.count("base")},
            "deeps": {"count": kinds.count("deep")},
            "wants": {"count": len(wants), "keep": WANT_KEEP}}


class History:
    """Slot-seconds by phase, one bucket a minute, for the last few minutes.

    Fed from the poll. Between two polls a slot is taken to be in the phase the
    earlier one reported, and a bucket edge inside the gap splits it. This is
    what shows that a backend spent nine of the last ten minutes reading, which
    the live view cannot."""

    def __init__(self, keep=HISTORY_KEEP, step=HISTORY_STEP):
        self.keep, self.step = keep, step
        self.at = None
        self.since = None
        self.rows = {}        # backend name -> {"done": [...], "cur": {...}}
        self.state = {}       # backend name -> [(phase, tg_rate)] last seen
        # Machine load shares the buckets, so every graph has one time axis.
        # key -> {"done": [mean per bucket], "cur": [sum, count]}
        self.load = {}

    @staticmethod
    def _empty():
        return {"read": 0.0, "gen": 0.0, "stalled": 0.0, "secs": 0.0}

    def _row(self, name):
        return self.rows.setdefault(name, {"done": [], "cur": self._empty()})

    def _charge(self, dt):
        for name, slots in self.state.items():
            cur = self._row(name)["cur"]
            cur["secs"] += dt
            for phase, tg_rate in slots:
                if phase == "reading":
                    cur["read"] += dt
                elif phase == "generating":
                    # No rate yet is not evidence of a stall. The slot is
                    # generating either way, so the second is still counted -
                    # just not against the figure that says contention.
                    cur["gen" if tg_rate is None
                        else ("stalled" if tg_rate < STALL_RATE else "gen")] += dt

    def _roll(self):
        for row in self.rows.values():
            row["done"].append(row["cur"])
            del row["done"][:-self.keep]
            row["cur"] = self._empty()
        for row in self.load.values():
            total, count = row["cur"]
            row["done"].append(round(total / count, 1) if count else None)
            del row["done"][:-self.keep]
            row["cur"] = [0.0, 0]

    def push_load(self, gauges):
        """Add one sample of each gauge to the bucket in progress. Call after
        push, so the two share the roll."""
        for key, value in gauges.items():
            if value is None:
                continue
            row = self.load.setdefault(key, {"done": [], "cur": [0.0, 0]})
            row["cur"][0] += value
            row["cur"][1] += 1

    def push(self, backends, now):
        """Take one poll's worth of slot detail."""
        if self.at is None:
            self.since = self.at = now
        while self.at < now:
            edge = (math.floor(self.at / self.step) + 1) * self.step
            stop = min(now, edge)
            self._charge(stop - self.at)
            self.at = stop
            if stop == edge:
                self._roll()
        self.state = {be["name"]: [(s["phase"], s["tg_rate"])
                                   for s in be.get("slots_detail") or []]
                      for be in backends if be.get("up")}
        for name in self.state:
            self._row(name)

    def snapshot(self):
        def tidy(bucket):
            return {k: round(v, 1) for k, v in bucket.items()}
        return {"step": self.step, "keep": self.keep, "since": self.since,
                "backends": {name: {"done": [tidy(b) for b in row["done"]],
                                    "cur": tidy(row["cur"])}
                             for name, row in self.rows.items()},
                "load": {key: {"done": list(row["done"]),
                               "cur": (round(row["cur"][0] / row["cur"][1], 1)
                                       if row["cur"][1] else None)}
                         for key, row in self.load.items()}}


# ---------------------------------------------------------------- machine load
# The weights are mmap'd, so they live in the page cache. When free memory gets
# tight the kernel drops them and prefill falls from 37 tokens a second to
# single digits, which is why the dashboard shows page cache against the model
# size rather than only what is used.

def parse_cpulist(text):
    """'0-17,36-53' -> {0, 1, ..., 17, 36, ..., 53}."""
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def read_nodes(root=Path("/sys/devices/system/node")):
    """The NUMA nodes and their cpus, from sysfs."""
    nodes = []
    for path in sorted(root.glob("node[0-9]*")):
        try:
            cpus = parse_cpulist((path / "cpulist").read_text())
        except OSError:
            continue
        nodes.append({"id": int(path.name[4:]), "cpus": cpus, "path": path})
    return nodes


def cpu_times(text):
    """/proc/stat -> {cpu index: (busy, total)}, in jiffies."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        fields = line.split()
        values = [int(v) for v in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)   # idle + iowait
        out[int(fields[0][3:])] = (sum(values) - idle, sum(values))
    return out


def node_busy(before, after, cpus):
    """Per cent of a node's cpu time spent busy between two samples, or None."""
    busy = total = 0
    for n in cpus:
        if n in before and n in after:
            busy += after[n][0] - before[n][0]
            total += after[n][1] - before[n][1]
    return round(100.0 * busy / total, 1) if total > 0 else None


def node_meminfo(text):
    """A node's meminfo -> bytes: total, free, and cache (FilePages)."""
    want = {"MemTotal:": "total", "MemFree:": "free", "FilePages:": "cache"}
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "Node" and parts[2] in want:
            out[want[parts[2]]] = int(parts[3]) * 1024
    return out


def gpu_query(text):
    """One nvidia-smi line 'util, used, total' in MiB -> a dict, or None."""
    try:
        util, used, total = [float(x) for x in text.strip().split(",")]
    except ValueError:
        return None
    return {"util": util, "vram_used": int(used * 1024 * 1024),
            "vram_total": int(total * 1024 * 1024)}


GPU_CMD = ("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
           "--format=csv,noheader,nounits")


def resident_bytes(be):
    """What one backend needs in the page cache: the buffers it mapped, less
    the tensor it reads lazily from disk. Both from its startup log."""
    cache = be.get("cache") or {}
    mapped, lazy = cache.get("mapped_mib") or 0.0, cache.get("lazy_mib") or 0.0
    return int(max(0.0, mapped - lazy) * 1024 * 1024)


class Machine:
    """CPU, memory and GPU load, sampled beside the backend poll.

    /proc/stat and the node meminfo files are read every poll. nvidia-smi costs
    50 to 100 ms, so the poll starts it as a child and collects the answer on a
    later pass. Nothing here is on a request's path, and a missing command
    turns the gpu row off rather than raising."""

    def __init__(self, nodes=None, stat=Path("/proc/stat"), gpu_cmd=GPU_CMD):
        self.nodes = read_nodes() if nodes is None else nodes
        self.stat = stat
        self.gpu_cmd = list(gpu_cmd)
        self.prev = None          # the last /proc/stat sample
        self.cpu = {}             # node id -> busy per cent
        self.mem = {}             # node id -> {total, free, cache}
        self.gpu = None           # the last nvidia-smi answer
        self.gpu_at = None        # when nvidia-smi last started
        self.gpu_proc = None
        self.gpu_ok = True        # False once the command is found missing

    def sample(self, now):
        try:
            current = cpu_times(self.stat.read_text())
        except OSError:
            current = None
        if current and self.prev:
            for node in self.nodes:
                self.cpu[node["id"]] = node_busy(self.prev, current, node["cpus"])
        if current:
            self.prev = current
        for node in self.nodes:
            try:
                self.mem[node["id"]] = node_meminfo((node["path"] / "meminfo").read_text())
            except (OSError, TypeError):
                pass
        self._gpu(now)

    def _gpu(self, now):
        if not self.gpu_ok:
            return
        proc = self.gpu_proc
        if proc is not None:
            if proc.poll() is None:
                if now - (self.gpu_at or now) > 3 * GPU_POLL:
                    proc.kill()            # wedged. Try again next time.
                    self.gpu_proc = None
                return
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            if proc.stdout:
                proc.stdout.close()
            self.gpu = gpu_query(out) if proc.returncode == 0 else None
            self.gpu_proc = None
            return
        if self.gpu_at is not None and now - self.gpu_at < GPU_POLL:
            return
        self.gpu_at = now
        try:
            self.gpu_proc = subprocess.Popen(self.gpu_cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
        except (OSError, ValueError):
            self.gpu_ok, self.gpu = False, None

    def gauges(self):
        """The series the history keeps: one number per key, or None."""
        out = {}
        for node in self.nodes:
            key, mem = f"node{node['id']}", self.mem.get(node["id"]) or {}
            out[f"{key}.cpu"] = self.cpu.get(node["id"])
            out[f"{key}.cache"] = mem.get("cache")
            out[f"{key}.free"] = mem.get("free")
        gpu = self.gpu if self.gpu_ok else None
        out["gpu.util"] = gpu["util"] if gpu else None
        out["gpu.vram"] = gpu["vram_used"] if gpu else None
        return out

    def report(self, backends):
        """Current values for the dashboard. `resident_bytes` is the most any
        backend on the node must keep in the page cache: they map the same
        files, so the cache is shared."""
        nodes = []
        for node in self.nodes:
            here = [be for be in backends if be.get("node") == node["id"]]
            mem = self.mem.get(node["id"]) or {}
            nodes.append({"id": node["id"], "cpus": len(node["cpus"]),
                          "cpu": self.cpu.get(node["id"]),
                          "total": mem.get("total"), "free": mem.get("free"),
                          "cache": mem.get("cache"),
                          "resident_bytes": max([resident_bytes(be) for be in here], default=0),
                          "backends": [be["name"] for be in here]})
        return {"nodes": nodes, "gpu": self.gpu if self.gpu_ok else None}


def link_block(name):
    """Point the slot directory at a block on the faster disk.

    A backend takes a bare filename under its own --slot-save-path and rejects
    one with a directory in it, so a link is the only way to put a single file
    elsewhere. Writing through it lands on the other disk."""
    link = SLOT_DIR / name
    try:
        BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        print(f"[router] {BLOCK_DIR} is not usable, keeping blocks with the "
              f"rest: {err}", flush=True)
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    # Absolute: the kernel resolves a link's target against the link's own
    # directory, not the router's cwd, so a relative BLOCK_DIR would dangle.
    link.symlink_to(BLOCK_DIR.resolve() / name)


def file_size(name):
    """Bytes in one slot file, or 0 when it is gone."""
    try:
        return (SLOT_DIR / name).stat().st_size
    except OSError:
        return 0


def file_mtime(name):
    """When one slot file was last written, or 0 when it is gone."""
    try:
        return (SLOT_DIR / name).stat().st_mtime
    except OSError:
        return 0


def drop_file(name):
    """Delete a slot file, and whatever it points at. Never raises.

    Called from _route's finally and from _take, both before the turn
    ticket goes back. A throw here would claim the conversation for the
    life of the process, because claim_turn has no deadline. A file that
    will not go is worth a line in the log, not a conversation."""
    path = SLOT_DIR / name
    try:
        if path.is_symlink():
            path.readlink().unlink(missing_ok=True)
        path.unlink(missing_ok=True)
    except OSError as err:
        print(f"[router] could not delete {name}: {err}", flush=True)


def pins_file():
    """Where the pin map is kept between runs."""
    return SLOT_DIR / "pins.json"


def openings_file():
    """Where what the openings have earned is kept between runs."""
    return SLOT_DIR / "openings.json"


def read_rows(path):
    """The rows the last run wrote, or nothing if it wrote none we can use."""
    try:
        kept = json.loads(path.read_bytes())
    except Exception:
        return []                 # no file, or one we cannot trust
    return kept if isinstance(kept, list) else []


def write_rows(path, rows):
    """Write a bookkeeping file so half of one can never be read back.

    Beside the file and renamed over it, because write_text truncates first: a
    stop or a full disk part way through would leave a half file, which reads
    as empty, and adopt then deletes everything that file was vouching for."""
    spare = path.with_suffix(".json.new")
    try:
        spare.write_text(json.dumps(rows, indent=1))
        os.replace(spare, path)
        return True
    except OSError as err:
        print(f"[router] could not write {path.name}: {err}", flush=True)
        spare.unlink(missing_ok=True)
        return False


def opening_key(name):
    """The key inside a saved opening's file name, or None if it is not one."""
    for mark in SHELF_MARKS:
        if name.startswith(mark) and name.endswith(".park"):
            return name[len(mark):-len(".park")]
    return None


def shelf_of(name):
    """Which shelf a saved opening's file name puts it on."""
    return "base" if name.startswith("base-") else "deep"


def adopt_files(names, vouched=(), size=None):
    """Sort the files the last run left behind, oldest first.

    A saved opening is named after its own contents, so it is always
    good. A conversation's copy is good only if the pin file vouches for
    it, because nothing else says whose cache it is.

    The pin file is asked first. A conversation key comes from the
    client, so a name can be made to look like an opening."""
    size = size or file_size
    openings, bytes_ = OrderedDict(), {}
    parked, spent = [], []
    for name in names:
        if name in vouched:
            parked.append(name)
            continue
        key = opening_key(name)
        if key and (DEEP_OPENINGS or shelf_of(name) == "base"):
            openings[key] = name
            bytes_[key] = size(name)
        else:
            spent.append(name)      # a copy nothing vouches for, or a deeper
                                    # opening on a router that stopped making
                                    # them and is giving the disk back
    spent += trim_openings(openings, bytes_)
    return openings, bytes_, parked, spent


def trim_openings(openings, bytes_, keep=()):
    """Drop openings until they fit BLOCK_BUDGET, least useful first.

    Deeper cuts go before system prompts at any age: a system prompt
    serves every new session, a deeper cut reaches further but serves
    fewer. Within a kind the least recently used goes first.

    Returns the file names dropped. One is always kept, however big, or
    a budget under one block would delete the opening a request is about
    to load. `keep` names openings being built, which are not on disk yet."""
    order = sorted(openings, key=lambda k: shelf_of(openings[k]) == "base")
    dropped = []
    for key in order:
        if sum(bytes_.values()) <= BLOCK_BUDGET or len(openings) <= 1:
            break
        if key in keep:
            continue
        bytes_.pop(key, None)
        dropped.append(openings.pop(key))
    return dropped


class Flow:
    """Every turn in flight and the stages it walks, for the flow dashboard.

    A turn moves through `queued`, `prefill`, `generate-queue` and `generate`, then
    `done`. There is no scheduler to notice this from outside - each request's
    own thread makes every transition - so the threads say so where they
    already make them.

    The log outlives the live row: a turn can cross two stages between two
    pushes of the payload, and the animation replays the log rather than miss a
    stage it never saw. Held under Pool.cv."""

    def __init__(self):
        self.live = {}                        # conv -> its current stage and where
        self.log = deque(maxlen=FLOW_LOG)     # newest first, for the animation

    def note(self, conv, stage, backend=None, slot=None):
        """Move a turn to its next stage. Held under the lock."""
        if not conv:
            return
        now = time.time()
        row = self.live.get(conv)
        if stage == "done":
            if row is None:
                return
            self.log.appendleft(dict(row, stage="done", at=now,
                                     since=None, changed=None))
            del self.live[conv]
            return
        if (row and row["stage"] == stage
                and row["backend"] == backend and row["slot"] == slot):
            return
        entry = {"conv": short_key(conv), "stage": stage,
                 "backend": backend, "slot": slot}
        self.live[conv] = dict(entry, since=row["since"] if row else now,
                               changed=now)
        self.log.appendleft(dict(entry, at=now))

    def report(self):
        """What the dashboard animates. Held under the lock."""
        return {"live": list(self.live.values()), "log": list(self.log)}


class Pool:
    """Track free slots. Keep each conversation on one backend."""

    # Metrics that are not running totals, so a difference of two readings is
    # meaningless and since_reset leaves them as they stand. The first five are
    # llama-server's own `gauges` list (server-task.cpp); n_tokens_max sits in
    # its `counters` list but is built with std::max, so it is a maximum and
    # belongs here - that is the one the dashboard shows as "longest".
    GAUGES = ("prompt_tokens_seconds", "predicted_tokens_seconds",
              "requests_processing", "requests_deferred",
              "n_busy_slots_per_decode", "n_tokens_max")

    def __init__(self, backends, watch=True):
        self.cv = threading.Condition()
        self.backends = [dict(b, slots=1, n_ctx=0, busy=0, up=False, served=0, model="",
                              stats={}, slots_detail=[], slot_prev={}, misses=0,
                              cache={}, idle_runs={}, draining=False)
                         for b in backends]
        # Each backend reports prompt cache evictions in its own log, and
        # nowhere else.
        self.cache_watch = {be["name"]: CacheWatch(
            RUN_DIR / f"{be['name']}.log",
            sink=lambda kind, value, name=be["name"]: EVENTS.write(
                "backend", backend=name, kind=kind, amount=value))
            for be in self.backends}
        self.pins = OrderedDict()
        # conversation -> the wait ticket of the turn serving it now.
        #
        # A conversation is one pin, one slot and one copy on disk, all named
        # after it alone. Two turns at once move the same three: the second
        # takes a prefiller, clears the slot the first is reading in, and
        # writes over the first one's copy. The first then carries that copy to
        # generate, restores a prompt it never read, and starts from nothing.
        # Seen in production: one conversation recalled onto two backends
        # ninety seconds apart. So a turn holds its conversation and the next
        # one waits.
        self.turns = {}
        # Opening name -> the file holding it, least recently used first. The
        # file name says which kind it is: a system prompt on its own, which
        # every new session needs, or a cut where two conversations diverge.
        # One shelf, because they share a disk and BLOCK_BUDGET measures it.
        self.openings = OrderedDict()
        self.opening_bytes = {}        # the same keys, and what each takes
        # (backend, slot) -> the cuts that slot holds. A new request that opens
        # the same way can start from one of them.
        self.holds = {}
        # (backend, slot) -> the deepest message index that slot holds a cut
        # at, so the dashboard can say how far in another conversation could
        # start. -1 is the system prompt alone.
        self.holds_depth = {}
        # conversation -> what warm_prefix found and did, for the dashboard.
        # Without it, asking why an opening was not cut deeper means reading
        # the code and guessing which branch it stopped at.
        self.choices = OrderedDict()
        # conversation -> (parent, depth) for sessions branched off another's
        # opening, so a fork is logged once per depth rather than once a turn.
        self.forked = OrderedDict()
        # Counters at the last reset, per backend. A lifetime average carries
        # every run since the backend started, and after a change the figure
        # that matters is the one since the change.
        self.rates_from = {}
        self.rates_since = None
        # Opening name -> what it takes to read one, newest last. A request
        # that could have used an opening the router lacks writes it here, and
        # the builder reads it when a backend falls idle.
        self.wants = OrderedDict()
        # Openings being read now, so sessions starting together wait for the
        # first rather than each reading its own copy.
        self.building = {}
        self.waiting = 0          # requests with no free slot yet
        self.waiters = {}         # ticket -> the request waiting, for the dashboard
        # Turns read and parked, queued for a slot on the backend that
        # generates. They hold nothing while they wait, so they appear nowhere
        # else for the dashboard to count.
        self.to_generate = 0
        self.wait_seq = 0
        # Every turn in flight, at which stage, for the flow dashboard.
        self.flow = Flow()
        # The last few slot files written or read, so the dashboard shows the
        # disk side doing something.
        self.recent = deque(maxlen=RECENT_FILES)
        # The last few requests, with what each one started from.
        self.recent_requests = deque(maxlen=RECENT_REQUESTS)
        self.history = History()
        self.machine = Machine()
        self.loads = {}           # opening key -> times a request loaded it
        self.mounts = []          # disk usage, refreshed every MOUNT_POLL
        self.mounts_at = 0.0
        # Copies waiting to be written, and the one worker that writes them.
        # Started on the first park rather than here, so a Pool that never
        # parks - most of the test suite - never starts a thread.
        self.park_jobs = queue.Queue()
        self.parker = None
        if watch:
            threading.Thread(target=self._watch, daemon=True).start()
            threading.Thread(target=self._builder, daemon=True).start()

    def adopt(self, names=None, remove=drop_file):
        """Take over what the last run left in the slot directory."""
        if names is None:
            names = []
            for found in sorted(SLOT_DIR.glob("*.park"),
                                key=lambda f: f.lstat().st_mtime):
                if found.exists():
                    names.append(found.name)
                else:
                    found.unlink(missing_ok=True)   # a link with nothing behind
        # What the openings earned last run: the order they were in, least
        # recently used first, and how often each was loaded. Without it the
        # only order left is the file's mtime, which link_block sets when the
        # block is built and never touches again - so one loaded every morning
        # for a week sorts ahead of one nobody has ever asked for, and the
        # budget drops exactly the wrong one. An opening the file does not
        # name keeps its place among the others, by mtime, at the back.
        remembered = [row for row in read_rows(openings_file())
                      if isinstance(row, dict) and row.get("key")]
        was = {row["key"]: rank for rank, row in enumerate(remembered)}
        self.loads.update({row["key"]: row.get("loads") or 0
                           for row in remembered})
        names.sort(key=lambda n: was.get(opening_key(n), len(was)))
        kept = read_rows(pins_file())
        # Both fields, not just the file: read_rows already refuses a map it
        # cannot parse, and a row without a conv would then KeyError out of
        # adopt() in __main__ - before serve_forever, so the router never
        # starts and says only "KeyError: 'conv'".
        by_file = {row["file"]: row for row in kept
                   if isinstance(row, dict) and row.get("file") and row.get("conv")}
        openings, sizes, parked, spent = adopt_files(names, set(by_file))
        with self.cv:
            self.openings, self.opening_bytes = openings, sizes
            for name in parked:
                row = by_file[name]
                self.pins[row["conv"]] = {
                    # The slot went with the backend that held it, so this
                    # cache must be restored before it is served. Naming no
                    # live backend is what makes that happen.
                    "backend": "(before the restart)",
                    "slot": None, "tokens": row.get("tokens", 0),
                    "last": time.time(), "inflight": False,
                    "turns": row.get("turns", 1), "parked": name,
                    "bytes": row.get("bytes", 0),
                    # When it was written, from the pin map or from the file
                    # itself. Without it every copy that survived the restart
                    # reads as age zero, so the budget sweep cannot tell the
                    # oldest from the newest and the dashboard cannot say
                    # which one goes next.
                    "parked_at": row.get("parked_at") or file_mtime(name)}
        for name in spent:
            remove(name)
        self.save_openings()      # trimmed, so the file has to say so
        if openings or parked or spent:
            kinds = [shelf_of(name) for name in openings.values()]
            print(f"[router] kept {kinds.count('base')} system prompt(s), "
                  f"{kinds.count('deep')} deeper opening(s) and {len(parked)} "
                  f"conversation(s), dropped {len(spent)} stale file(s)",
                  flush=True)

    def note_file(self, did, name, be, slot, size=0):
        """Record one slot file written or read. Held under the lock."""
        self.recent.appendleft({"did": did, "name": name[:8], "at": time.time(),
                                "backend": be["name"], "slot": slot,
                                "bytes": size})

    def note_stage(self, conv, stage, backend=None, slot=None):
        """Move a turn along its stages, for the flow dashboard."""
        with self.cv:
            self.flow.note(conv, stage, backend, slot)

    def begin_wait(self, conv, tokens, images=0, image_tokens_=0):
        """Count a request as waiting until end_wait. Returns its ticket."""
        with self.cv:
            self.wait_seq += 1
            self.waiters[self.wait_seq] = {"conv": conv, "tokens": tokens,
                                           "images": images,
                                           "image_tokens": image_tokens_,
                                           "since": time.time()}
            self.waiting = len(self.waiters)
            return self.wait_seq

    def end_wait(self, ticket):
        with self.cv:
            self.waiters.pop(ticket, None)
            self.waiting = len(self.waiters)

    def claim_turn(self, conv, ticket, wanted=None):
        """Hold this conversation until finish_turn. One turn of it at a time.

        Returns True when the turn holds it. False means the client stopped
        waiting, and nothing is held.

        The wait has no deadline, for the reason acquire has none: the turn
        ahead ends on its own, and reading a prompt again costs more than any
        wait. A turn waiting here holds no backend, so nothing waits for it."""
        if not conv:
            return True
        began = None
        with self.cv:
            while conv in self.turns and self.turns[conv] != ticket:
                if wanted is not None and not wanted():
                    return False
                began = began or time.time()
                self.cv.wait(1.0)
            self.turns[conv] = ticket
            # Noted here rather than in begin_wait: a turn waiting for the one
            # ahead must not move that one's row on the flow dashboard, which
            # is keyed by conversation too.
            self.flow.note(conv, "queued")
        if began is not None:
            print(f"[router] {short_key(conv)} waited "
                  f"{time.time() - began:.0f}s for the turn ahead of it",
                  flush=True)
        return True

    def finish_turn(self, conv, ticket):
        """Let the next turn of this conversation start.

        The ticket must match: a turn that gave up waiting never held the
        conversation, and must not release one that does."""
        if not conv:
            return
        with self.cv:
            if self.turns.get(conv) == ticket:
                del self.turns[conv]
                self.cv.notify_all()

    def _waiting_detail(self, now):
        """Each waiter, and what it waits for. Held under the lock.

        A turn waiting for the turn ahead of it waits for neither a slot
        nor a pin. It says "turn".

        A pin is only worth naming when acquire would wait for it, and
        acquire waits only for a backend that prefills. Where one backend
        generates for the rest, every conversation is pinned to it
        afterwards, and naming that pin reads as a generator holding the
        box up while it sits idle. It is not: acquire drops such a pin at
        once and takes the first free prefiller."""
        takers = [be for be in self.backends
                  if be["up"] and not be.get("draining") and prefills(be)]
        largest = max([be["n_ctx"] for be in takers], default=0)
        rows = []
        for ticket, w in self.waiters.items():
            record = self.pins.get(w["conv"]) if w["conv"] else None
            pinned = record["backend"] if record else None
            held = self.turns.get(w["conv"])
            if held is not None and held != ticket:
                wants = "turn"
            elif pinned and any(be["name"] == pinned and be["up"]
                                and prefills(be) for be in self.backends):
                wants = "pinned"
            elif w["tokens"] > largest:
                wants = "big"
            else:
                wants = "prefill"
            # `tokens` is the ticket's, which carries REPLY_TOKENS of reply
            # room as well, because room is what a waiter needs. The dashboard
            # prints it as the size of the request and divides an image charge
            # into it, so that share is low by the reserve on a small request.
            # Taking it off here would make the number mean something different
            # from what begin_wait was handed; the honest fix is for the ticket
            # to carry the prompt and the room as two, which is a change to
            # begin_wait and note_request together.
            rows.append({"conv": short_key(w["conv"]), "since": w["since"],
                         "waited": round(now - w["since"], 1),
                         "tokens": w["tokens"], "wants": wants,
                         "images": w.get("images", 0),
                         "image_tokens": w.get("image_tokens", 0),
                         "backend": pinned if wants == "pinned" else None})
        rows.sort(key=lambda r: r["since"])
        return rows

    def note_request(self, conv, be, path, took, waited, started, tokens,
                     read_prompt_n=None, read_cache_n=None,
                     images=0, image_tokens_=0):
        """Record one finished request for the dashboard.

        `tokens` is this router's estimate, with the reply room already
        taken off by the caller: the dashboard measures a turn against what
        reading it cold would have cost, and 1,024 tokens nobody sent is 41
        seconds of reading nobody did.

        The two read counts come from the reply's own `timings`, and are
        None for a turn that never read."""
        with self.cv:
            self.recent_requests.appendleft({
                "conv": short_key(conv), "backend": be["name"], "path": path,
                "took": round(took, 1), "waited": round(waited, 1),
                "started": started, "tokens": tokens,
                "read": read_prompt_n, "reused": read_cache_n,
                "images": images, "image_tokens": image_tokens_,
                "at": time.time()})

    def _read_mounts(self, now):
        """Free space on the disks the slot files land on. Cheap, but not
        worth a statvfs per status call."""
        if now - self.mounts_at < MOUNT_POLL:
            return self.mounts
        rows = []
        seen = set()
        for path in (SLOT_DIR, BLOCK_DIR):
            try:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                usage = shutil.disk_usage(resolved)
                rows.append({"path": str(path), "total": usage.total,
                             "free": usage.free})
            except OSError:
                continue
        self.mounts, self.mounts_at = rows, now
        return rows

    def reset_rates(self, counters=None):
        """Start the averages again from now."""
        with self.cv:
            self.rates_from = {}
            for be in self.backends:
                if counters is True:
                    self.rates_from[be["name"]] = dict(be.get("counters") or {})
                elif counters is not None:
                    self.rates_from[be["name"]] = dict(counters)
            self.rates_since = time.time()
        return self.rates_since

    def since_reset(self, be, counters):
        """The counters as they read since the last reset.

        A counter that went backwards means the backend restarted and
        counts from zero again, so it is taken as it stands.

        GAUGES are left alone. The difference of two counters is a count,
        but the difference of two gauges is nothing. n_tokens_max is a
        running maximum and n_busy_slots_per_decode an average: subtracting
        them read 0 for the longest prompt served, and turned a busy
        average of 2.40 into 0.01. pp_total and tg_total multiply by that,
        so the button understated the box 320-fold."""
        # `is None`, not falsy: a backend that was down at the reset has an
        # empty baseline, which is a baseline. Read as "no reset" it kept
        # counting from process start while every other backend counted from
        # the reset, and pressing the button again stored {} again.
        was = self.rates_from.get(be["name"])
        if was is None:
            return counters
        out = {}
        for name, now in counters.items():
            if name in self.GAUGES:
                out[name] = now
                continue
            before = was.get(name, 0)
            out[name] = now - before if now >= before else now
        return out

    def read_settings(self, be):
        """A backend's startup settings, read from the log it wrote them to.

        Read again when the log is a different file. restart-backend.sh moves
        the old one aside and the new backend creates its own, so the inode
        changes; watching the size instead missed the restart whenever the new
        log passed the old offset before this next looked."""
        path = RUN_DIR / f"{be['name']}.log"
        try:
            stat = path.stat()
            size, ino = stat.st_size, stat.st_ino
        except OSError:
            return be.get("config") or {}
        if (be.get("config") and ino == be.get("config_ino")
                and size >= be.get("config_at", 0)):
            return be["config"]
        head = []
        try:
            with open(path, errors="replace") as handle:
                for _ in range(1500):   # loading prints hundreds of lines first
                    head.append(next(handle))
        except (OSError, StopIteration):
            pass
        be["config"] = read_config(head)
        be["vision"] = read_vision(head)
        be["config_at"] = size
        be["config_ino"] = ino
        return be["config"]

    def vision(self):
        """The geometry the vision encoder was built with.

        Whichever backend printed it. They load the same mmproj, and one that
        has not printed it has no vision at all. Read rather than held, so a
        restart onto a different mmproj is picked up."""
        for be in self.backends:
            self.read_settings(be)
            if be.get("vision"):
                return be["vision"]
        return VISION

    def _read_metrics(self, be):
        """Read the counters llama-server keeps, for the dashboard."""
        try:
            with urllib.request.urlopen(be["url"] + "/metrics", timeout=3) as r:
                text = r.read().decode()
        except Exception:
            return
        value = {}
        for line in text.splitlines():
            if line.startswith("#") or "{" in line:
                continue           # comment, or a metric with labels
            name, _, number = line.partition(" ")
            try:
                value[name.split(":", 1)[-1]] = float(number)
            except ValueError:
                pass

        be["counters"] = dict(value)      # raw, so a reset can mark this point
        value = self.since_reset(be, value)

        def rate(tokens, seconds):
            # Over the life of the backend. The *_tokens_seconds gauges are
            # rolling windows, so they read zero whenever it is idle.
            return per_second(value.get(tokens, 0), value.get(seconds, 0))

        # prompt_tokens_total counts processed tokens and excludes cached ones,
        # so the prompt is the sum of the two.
        processed = value.get("prompt_tokens_total", 0)
        cached = value.get("prompt_tokens_cached_total", 0)
        drafted = value.get("spec_decode_num_draft_tokens_total", 0)

        # tokens_predicted_seconds_total sums per-request time and concurrent
        # slots overlap, so these rates are per request. Times the average busy
        # slot count is what the backend delivers overall.
        busy_per_decode = value.get("n_busy_slots_per_decode", 1) or 1
        be["stats"] = {
            "busy_per_decode": round(busy_per_decode, 2),
            "pp_rate": rate("prompt_tokens_total", "prompt_seconds_total"),
            "tg_rate": rate("tokens_predicted_total", "tokens_predicted_seconds_total"),
            "accept": round(100 * value.get("spec_decode_num_accepted_tokens_total", 0)
                            / drafted, 1) if drafted else 0,
            "cached": round(100 * cached / (cached + processed), 1) if cached + processed else 0,
            "longest": int(value.get("n_tokens_max", 0)),
            "generated": int(value.get("tokens_predicted_total", 0)),
            # Reading against generating is the split a generator is chosen on;
            # reused against read is what the caches buy.
            "read_s": round(value.get("prompt_seconds_total", 0), 1),
            "gen_s": round(value.get("tokens_predicted_seconds_total", 0), 1),
            "prompt_tokens": int(processed),
            "cached_tokens": int(cached),
        }
        st = be["stats"]
        st["pp_total"] = round(st["pp_rate"] * busy_per_decode, 1)
        st["tg_total"] = round(st["tg_rate"] * busy_per_decode, 1)

    def _read_slots(self, be, raw=None):
        """Per-slot state, so a 3-slot backend is not a single average.

        `raw` is what /slots answered, so a test can hand it a reading instead
        of standing a backend up to produce one."""
        if raw is None:
            try:
                with urllib.request.urlopen(be["url"] + "/slots", timeout=3) as r:
                    raw = json.load(r)
            except Exception:
                return
        # /slots reports counters, not rates, and the backend average cannot
        # show one slot crawling while another runs free.
        now = time.time()
        previous = be.get("slot_prev") or {}
        current, detail = {}, []
        for slot in raw if isinstance(raw, list) else []:
            # llama.cpp sends this as a one-element array. Older builds sent a
            # bare object, so both are read.
            token = slot.get("next_token") or {}
            if isinstance(token, list):
                token = token[0] if token else {}
            cached = slot.get("n_prompt_tokens_cache", 0)
            sid = slot.get("id")
            task = slot.get("id_task")
            decoded = token.get("n_decoded", 0)
            processed = slot.get("n_prompt_tokens_processed", 0)

            # Measured over a window, not between polls: the counters are
            # integers, and a slot at 0.03 tokens/s does not move in two
            # seconds, so every delta would read zero. Hold an anchor and
            # recompute once the window can resolve a token.
            was = previous.get(sid) or {"task": None, "decoded": 0, "processed": 0,
                                        "done_d": 0.0, "done_p": 0.0, "since": now,
                                        "pp_rate": 0.0, "tg_rate": 0.0,
                                        # Whether a window has ever resolved
                                        # here. Until one has there is no rate,
                                        # which is not the same as a rate of
                                        # zero: a slot that has just started
                                        # generating would otherwise read as
                                        # one stalled at 0.00 for a whole
                                        # RATE_WINDOW, on the dashboard and in
                                        # the stalled seconds the history keeps.
                                        "measured": False}
            # A new task restarts the counters at zero, so the delta is the
            # whole count. Accumulating across that boundary is what lets a
            # slot serving short requests still report a rate.
            if was["task"] is None:
                grew_d = grew_p = 0        # first sight: take a baseline, count nothing
            elif was["task"] == task:
                grew_d = max(0, decoded - was["decoded"])
                grew_p = max(0, processed - was["processed"])
            else:
                grew_d, grew_p = decoded, processed
            done_d = was["done_d"] + grew_d
            done_p = was["done_p"] + grew_p

            gap = now - was["since"]
            measured = was["measured"]
            if gap >= RATE_WINDOW:
                pp_rate, tg_rate = done_p / gap, done_d / gap
                done_d = done_p = 0.0
                since = now
                measured = True
            else:
                pp_rate, tg_rate = was["pp_rate"], was["tg_rate"]
                since = was["since"]

            current[sid] = {"task": task, "decoded": decoded, "processed": processed,
                            "done_d": done_d, "done_p": done_p, "since": since,
                            "pp_rate": pp_rate, "tg_rate": tg_rate,
                            "measured": measured}

            # n_prompt_tokens_total is the prompt the task arrived with, added
            # by patches/slots-report-the-prompt-size.patch. What is left to
            # read is that, less what was reused and what has been read.
            #
            # n_prompt_tokens is not that number: it counts what the slot holds
            # now, growing while the prompt is read and again with every token
            # generated. Read that way, a slot 98% served from cache reported
            # "512 / 89,848 read". A backend without the patch has nothing
            # better, so it falls back to the old arithmetic.
            busy = bool(slot.get("is_processing"))
            whole = slot.get("n_prompt_tokens_total")
            if whole is None:
                whole = max(0, slot.get("n_prompt_tokens", 0) - decoded)
                to_read = max(0, whole - cached)
            else:
                to_read = max(0, whole - cached - processed)
            detail.append({
                "id": sid,
                "busy": busy,
                "phase": "idle" if not busy else ("generating" if decoded else "reading"),
                "prompt": to_read,
                "done": processed,
                "cached": cached,
                "decoded": decoded,
                # null, not 0.0, until a window has resolved: "no rate yet"
                # and "a rate of zero" mean opposite things to a reader.
                "pp_rate": round(pp_rate, 1) if measured else None,
                "tg_rate": round(tg_rate, 1) if measured else None,
            })
        be["slot_prev"] = current
        be["slots_detail"] = detail
        self._note_idle(be, detail)
        # What the backend delivers right now: the sum of its slots. /metrics
        # gives a lifetime average per request, which understates a busy
        # multi-slot backend by roughly its slot count.
        stats = be.setdefault("stats", {})
        stats["pp_live"] = round(sum(d["pp_rate"] or 0 for d in detail), 1)
        stats["tg_live"] = round(sum(d["tg_rate"] or 0 for d in detail), 1)

    @staticmethod
    def _note_idle(be, detail):
        """Count the polls in a row each slot has looked idle for.

        One poll cannot tell a free slot from one between two turns of the
        same conversation. Two polls apart can."""
        was = be.get("idle_runs") or {}
        be["idle_runs"] = {s["id"]: 0 if s["busy"] else was.get(s["id"], 0) + 1
                           for s in detail}

    def _watch(self):
        """Check each backend. Read its slot count, context size and counters."""
        while True:
            for be in self.backends:
                try:
                    with urllib.request.urlopen(be["url"] + "/props", timeout=3) as r:
                        props = json.load(r)
                    be["slots"] = int(props.get("total_slots") or 1)
                    be["n_ctx"] = int(props["default_generation_settings"]["n_ctx"])
                    be["model"] = props.get("model_alias") or ""
                    self._read_metrics(be)
                    self._read_slots(be)
                    self._read_cache(be)
                    up = True
                except Exception:
                    up = False
                # Two misses before declaring it down: a loaded box can miss a
                # 3 second deadline once, and marking it down re-pins every
                # conversation waiting on it.
                if up:
                    be["misses"] = 0
                else:
                    be["misses"] = be.get("misses", 0) + 1
                    up = be["up"] and be["misses"] < 2
                came_back = up and not be["up"] and be.get("seen")
                if up != be["up"]:
                    if up:
                        print(f"[router] {be['name']} is up: "
                              f"{be['slots']} slots, {be['n_ctx']} ctx each", flush=True)
                    else:
                        print(f"[router] {be['name']} is down", flush=True)
                with self.cv:
                    be["up"] = up
                    if up:
                        be["seen"] = True
                        # A backend that went away and came back is a new
                        # process with empty slots. Nothing said so before, so
                        # a conversation pinned here kept its slot number,
                        # recall saw "already here, and still in a slot" and
                        # refused, warm_prefix refused too because a copy
                        # existed - and the turn read its whole prompt into an
                        # empty slot with a good copy sitting on disk. That is
                        # the case restart-backend.sh exists to make invisible.
                        if came_back:
                            self.forget_slots(be["name"])
                        self.cv.notify_all()
            now = time.time()
            try:
                self.machine.sample(now)
            except Exception as err:          # load is a nicety. Never the poll.
                print(f"[router] machine sample failed: {err}", flush=True)
            with self.cv:
                self.history.push(self.backends, now)
                self.history.push_load(self.machine.gauges())
            time.sleep(POLL)

    def _read_cache(self, be):
        """Total the prompt cache events this backend has logged.

        An eviction is a conversation that lost its cached prefill. The count
        says whether a disk tier below this cache would pay for itself."""
        watch = self.cache_watch[be["name"]]
        before = watch.stats["evictions"]
        watch.poll()
        be["cache"] = dict(watch.stats)
        if watch.stats["evictions"] > before:
            print(f"[router] {be['name']} prompt cache: "
                  f"{watch.stats['evictions']} evictions, "
                  f"{watch.stats['evicted_mib']:.0f} MiB dropped, "
                  f"holding {watch.stats['prompts']} prompts "
                  f"in {watch.stats['used_mib']:.0f} of "
                  f"{watch.stats['limit_mib']:.0f} MiB", flush=True)

    def pick_slot(self, be, conv):
        """The slot this read should use on this backend.

        The router says which slot rather than reading it back, because a
        reply only names its slot on some paths.

        A conversation already holding a slot here keeps it. Otherwise it
        takes one no other request was handed. Judged by what the router
        handed out, not by the poll: the poll is two seconds old, so requests
        arriving together were all told the same slot."""
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if (record and record["backend"] == be["name"]
                    and record["slot"] is not None):
                # Written down as well as returned. `taken` below is built from
                # `using`, and this branch never set it, so a conversation that
                # kept its own warm slot was invisible to the next request: on
                # a backend with more than one slot, one arriving while this
                # turn ran could be handed the very slot it was reading in.
                record["using"] = record["slot"]
                return record["slot"]
            taken = {p.get("using") for name, p in self.pins.items()
                     if name != conv and p.get("inflight")
                     and p.get("backend") == be["name"]}
            detail = be.get("slots_detail") or []
            ids = [s["id"] for s in detail] or list(
                range(max(1, be.get("slots", 1))))
            working = {s["id"] for s in detail if s.get("busy")}
            # Free by both accounts first: nobody has been given it, and the
            # backend was not working on it when last asked. Then merely not
            # given out, because the poll is the older of the two.
            slot = next((i for i in ids if i not in taken and i not in working),
                        next((i for i in ids if i not in taken), ids[0]))
            if record is not None:
                record["using"] = slot
            return slot

    def _reading_rank(self, be):
        """Order backends for a prompt that has to be read somewhere.

        A socket already reading comes last. Reading is compute bound with
        local memory, so two reads on one socket share its cores and roughly
        halve each other, while a read on the other socket runs at full speed.
        A generating slot does not count: it competes for nothing a read
        needs."""
        def reads(backend):
            return sum(1 for slot in (backend.get("slots_detail") or [])
                       if slot.get("phase") == "reading")

        node = be.get("node")
        on_node = sum(reads(other) for other in self.backends
                      if other.get("node") == node)
        # Within a node, a quiet instance beats a second slot on a busy one:
        # llama.cpp lets the first reading slot take the whole batch.
        #
        # Then the opposite of pref, which says where a conversation would
        # rather generate. Reading takes that order backwards and leaves the
        # instances kept for generating free to do it.
        return (on_node, reads(be), -be["pref"], be["busy"])

    def _usable(self, be, tokens):
        """True if this backend is up, has a free slot, and is big enough."""
        return (be["up"] and not be.get("draining")
                and be["busy"] < be["slots"] and tokens <= be["n_ctx"])

    def drain(self, name, post, deadline=DRAIN_DEADLINE):
        """Take a backend out of service so it can be stopped and started.

        Requests wait in acquire rather than fail, so a restart under them is
        invisible however long it takes. What it does throw away is the caches
        in its slots, so they are copied out first."""
        be = next((b for b in self.backends if b["name"] == name), None)
        if be is None:
            return None
        with self.cv:
            be["draining"] = True
            self.cv.notify_all()      # waiters can pick the other backend now

        # Let work already running finish. Killing it throws away a prompt
        # that may have taken twenty minutes to read.
        stop = time.time() + deadline
        while True:
            with self.cv:
                quiet = be["busy"] <= 0
                if quiet or time.time() > stop:
                    break
                self.cv.wait(0.2)

        parked = self.park_all(post, only=name) if quiet else 0
        # What is still only in a slot after the pass. A save that timed out or
        # was refused leaves the pin unparked, and stopping the backend then
        # throws that cache away - so the caller has to be told, or
        # restart-backend.sh reads a 200 and kills it anyway.
        with self.cv:
            left = sum(1 for p in self.pins.values()
                       if p["backend"] == name and p["slot"] is not None
                       and not p["inflight"] and not copy_is_current(p))
        print(f"[router] {name} is drained: "
              f"{'quiet' if quiet else 'still busy'}, {parked} cache(s) parked"
              + (f", {left} still only in a slot" if left else ""), flush=True)
        return {"backend": name, "quiet": quiet, "parked": parked, "left": left}

    def resume(self, name):
        """Put a backend back in service."""
        be = next((b for b in self.backends if b["name"] == name), None)
        if be is None:
            return False
        with self.cv:
            be["draining"] = False
            self.cv.notify_all()
        print(f"[router] {name} is back in service", flush=True)
        return True

    def largest(self):
        """The largest prompt any backend will read.

        Prefillers only. A conversation pinned to a generator spills to a
        prefiller before it is served, so a generator's ctx is not a size
        this router can accept. Counting it let a turn past the 413 that
        acquire then waited on with no deadline, holding its turn ticket."""
        return max([be["n_ctx"] for be in self.backends
                    if be["up"] and prefills(be)], default=0)

    def acquire(self, conv, tokens, wanted=None):
        """Take a slot on the backend holding this conversation.

        A busy box is a queue, not a refusal: a taken slot means this turn
        starts later, never that it fails. So the wait has no deadline. It ends
        when a slot frees, when no backend could ever serve the request, or
        when `wanted` says the client has stopped waiting.

        A pin holds through the wait: reading a long prompt again costs far
        more, and a cache that has left its slot cannot be moved."""
        patience = time.time() + PIN_PATIENCE
        spill = False              # set once the pin is given up on

        while True:
            with self.cv:
                # Read the pin under the lock. Another thread may evict it.
                record = None if spill else self.pins.get(conv)
                pinned = record["backend"] if record else None
                target = next((b for b in self.backends if b["name"] == pinned), None)

                if target:
                    if prefills(target) and self._usable(target, tokens):
                        return self._take(target, conv, tokens)
                    # A turn can carry a whole file, and a fifth of them
                    # re-read everything, so a backend that does not read
                    # cannot serve one however recently it generated there.
                    if (not target["up"] or tokens > target["n_ctx"]
                            or not prefills(target)):
                        spill = True          # it can never take this request
                        target = None
                else:
                    spill = spill or pinned is not None   # pin names a gone backend

                if not target:
                    # With no pin the prompt must be read somewhere, so only a
                    # backend that reads will do. A pinned conversation skips
                    # this: its next turn extends a cache already there.
                    free = [b for b in self.backends
                            if prefills(b) and self._usable(b, tokens)]
                    if free:
                        return self._take(min(free, key=self._reading_rank),
                                          conv, tokens)

                # Nothing that reads is up, and this conversation has no cache
                # on one that is. Waiting cannot help.
                served_by = target is not None or any(
                    b["up"] and prefills(b) for b in self.backends)
                if not served_by:
                    return None
                # The only other thing that ends the wait is the client.
                if wanted is not None and not wanted():
                    return None

                # A pin is worth a short wait, not an idle backend.
                if target and time.time() > patience:
                    spill = True

                self.cv.wait(1.0)

    def _take(self, be, conv, tokens=0):
        be["busy"] += 1
        be["served"] += 1
        if conv:
            # Updated in place. Building a fresh record dropped every field
            # written elsewhere: `opening`, which note_holds writes at the end
            # of a turn and forget_stale_park reads at the start of the next,
            # and `parked_at`, which the PARK_BUDGET sweep sorts by - so a
            # conversation that parked and ran again sorted as age 0 and was
            # dropped first.
            record = self.pins.get(conv)
            if record is None:
                # A copy survives losing the slot, and the budget is spent
                # against its size, so the two travel together. Named here
                # because several readers index them directly.
                record = self.pins[conv] = {"parked": None, "bytes": 0}
            record.update(
                backend=be["name"],
                # A slot id only means something on its own backend.
                slot=record.get("slot") if record.get("backend") == be["name"] else None,
                tokens=tokens,
                last=time.time(),
                inflight=True,
                # A first turn still holds mostly system prompt.
                turns=record.get("turns", 0) + 1)
            self.pins.move_to_end(conv)
            while len(self.pins) > MAX_PINS:
                _, dropped = self.pins.popitem(last=False)
                if dropped.get("parked"):
                    drop_file(dropped["parked"])   # its copy is now orphaned
        return be

    def release(self, be, conv=None):
        """Give the backend back. The copy on disk stays where it is.

        The slot has moved past that copy, so the copy is behind. Behind is
        still a prefix, and a prefix is where every read starts. Deleting it
        left the only copy in a borrowed slot, which the next taker erases.

        recall refuses to restore over a slot that still holds the
        conversation, and the next save overwrites this file under the same
        name, so keeping it costs one copy rather than a pile."""
        with self.cv:
            be["busy"] -= 1
            record = self.pins.get(conv) if conv else None
            if record:
                record["inflight"] = False
                record["last"] = time.time()
            self.cv.notify_all()

    def forget_slots(self, name):
        """Forget which slot on this backend held what. Held under the lock.

        A restarted backend keeps its name and its port and loses every slot,
        so the router's slot numbers for it mean nothing afterwards. The pins
        stay - the conversation still ran there, and its copy on disk is still
        good - and clearing the slot is what lets recall put that copy back.

        A conversation mid-turn is left alone: it is talking to the process
        that is answering it, and its own thread owns that slot."""
        for record in self.pins.values():
            if record.get("backend") == name and not record.get("inflight"):
                record["slot"] = None
                record.pop("using", None)
        for key in [k for k in self.holds if k[0] == name]:
            del self.holds[key]
            self.holds_depth.pop(key, None)

    def holds_slot(self, conv):
        """True when the router knows which slot holds this conversation."""
        with self.cv:
            record = self.pins.get(conv)
            return bool(record) and record["slot"] is not None

    def note_slot(self, conv, slot):
        """Record which slot served this conversation. Needed to save it later."""
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if record:
                record["slot"] = slot

    def _builder(self):
        """Read one wanted opening while a backend is idle.

        Off the request path on purpose: an opening is tens of thousands of
        tokens, and nobody should wait behind it."""
        while True:
            time.sleep(BUILD_POLL)
            try:
                self.build_once(http_post)
            except Exception as err:
                print(f"[router] opening pass failed: {err}", flush=True)

    def _idle_slot(self, be):
        """A slot the builder may read into, or None.

        Stricter than _free_slot, which answers for a request that already has
        one. Three things must agree: the backend is up, the router's own count
        leaves room, and the poll has found the same slot idle more than once.
        The count is what acquire gates on, so honouring it keeps the builder
        out of the request path. The repeat keeps a two-second-old poll from
        handing out a slot that is working."""
        # A draining backend is about to be stopped. _usable already refuses
        # one for a request; the builder must refuse it too, or it reads an
        # opening into an instance that was just drained and parked.
        if not be["up"] or be.get("draining") or be["busy"] >= be["slots"]:
            return None
        runs = be.get("idle_runs") or {}
        for slot in be.get("slots_detail") or []:
            if not slot["busy"] and runs.get(slot["id"], 0) >= IDLE_POLLS:
                return slot["id"]
        return None

    def _free_slot(self, be):
        """A slot id on this backend that is not working, or None."""
        detail = be.get("slots_detail") or []
        if not detail:
            return 0                      # nothing reported yet, so slot 0
        for slot in detail:
            if not slot["busy"]:
                return slot["id"]
        return None

    def ensure_parked(self, be, skip_conv, post, remove=drop_file):
        """Copy every cache on this backend to disk before anything displaces it.

        Called before a request reaches the backend. A save reads a slot, so
        it only works while the cache is still in one: once llama.cpp pushes a
        cache out, only that backend can reach it again.

        Each conversation is tried once. A save that does not stick - the
        budget dropped it again, or any reason found later - must not come
        round again, or this loop never ends and the request never lands."""
        tried = set()
        while True:
            with self.cv:
                at_risk = [(conv, p) for conv, p in self.pins.items()
                           if p["backend"] == be["name"]
                           and conv != skip_conv
                           and conv not in tried
                           and not p["inflight"]
                           and p["slot"] is not None
                           and not copy_is_current(p)]
                if not at_risk:
                    return
                conv, record = min(at_risk, key=lambda item: item[1]["last"])
                record["inflight"] = True      # hold it still while it copies
                slot = record["slot"]
                tried.add(conv)

            self._save_park(conv, be, slot, post, remove)

    def _save_park(self, conv, be, slot, post, remove=drop_file):
        """Write one cache to disk. Mark the pin only if it really holds one."""
        name = conv + ".park"
        short = short_key(conv)
        kept = False
        written = 0
        # Two different failures. A short save means the slot holds somebody
        # else now, so the router no longer knows where this conversation is.
        # A save that fails outright says nothing about the slot: the cache is
        # still there, and forgetting it costs the next turn the whole prompt.
        lost = False
        refused = False
        began = time.time()
        # Which turn of this conversation this save belongs to. A save can take
        # POST_TIMEOUT - five minutes for a multi-gigabyte slot - and nothing
        # stops the conversation's own next turn starting inside that window:
        # claim_turn looks only at self.turns, which the previous turn emptied.
        # Clearing `inflight` afterwards then un-reserved a slot that was being
        # read, and pick_slot handed it to somebody else, which is the "two
        # turns move the same three things" failure the turn ticket exists to
        # stop. _take counts the turns, so comparing the count says whether the
        # record is still the one this save marked.
        with self.cv:
            record = self.pins.get(conv)
            turn = record.get("turns") if record else None
        try:
            answer = post(be["url"], f"/slots/{slot}?action=save",
                          {"filename": name}) or {}
            # Ask the disk, not the backend. PARK_BUDGET is spent against this
            # number, so it has to be the file. A build without
            # patches/slot-state-carries-checkpoints.patch counts the state and
            # not the checkpoint trailer - a 107 MB file came back as 49 MB -
            # and the cap would then be enforced against half the disk in use.
            # The backend's figure is the fallback for a stub that writes no
            # file. tests/live/test_llama_beliefs.py asserts the two agree.
            written = file_size(name) or (answer.get("n_written") or 0)
            kept = written >= PARK_FLOOR
            if not kept:
                lost = True
                print(f"[router] {short} was gone from {be['name']} slot {slot}, "
                      f"nothing to park", flush=True)
                remove(name)
        except Exception as err:
            refused = True
            print(f"[router] {short} park failed on {be['name']}: {err}", flush=True)

        with self.cv:
            record = self.pins.get(conv)
            if record:
                # Only if no turn of this conversation started while the save
                # ran. One that did owns `inflight` and `slot` now, and this
                # save's answer is about a slot it has moved on from. What the
                # file holds is still true, so the copy is still recorded.
                mine = record.get("turns") == turn
                if mine:
                    record["inflight"] = False
                if kept:
                    record["parked"] = name
                    record["bytes"] = written
                    record["parked_at"] = time.time()
                    # Which turn this copy holds, so the next pass can tell a
                    # copy of the slot as it stands from one several turns back.
                    record["parked_turn"] = turn
                    self.note_file("parked", conv, be, slot, written)
                elif not refused and mine:
                    record["parked"] = None
                    record["bytes"] = 0
                # A refused call keeps the copy it had. The backend said
                # nothing about the slot and nothing about the file, which is
                # still on disk and still a prefix of this conversation.
                # Clearing it orphaned the file - off the budget sweep and off
                # MAX_PINS eviction, eating the cap uncounted until a restart,
                # while the conversation re-read its whole prompt.
                if lost and mine:
                    # The slot holds someone else. Saying so stops the caller
                    # asking for the same empty copy again.
                    record["slot"] = None
            # Keep the newest copies that fit the budget, and the newest even
            # if it fills the budget alone. Newest means most recently written,
            # not pin order: by pin order the copy just written sits wherever
            # its conversation started, so a full budget drops it the moment it
            # lands, ensure_parked sees an unparked cache and writes it again.
            # cpu1_0 wrote the same 9.45 GiB copy 1,456 times between 04:44 and
            # 09:12 on 14 September, holding its slot throughout.
            held = sorted((c for c, p in self.pins.items() if p.get("parked")),
                          key=lambda c: self.pins[c].get("parked_at") or 0)
            spent, total = [], 0
            for age, name_held in enumerate(reversed(held)):
                older = self.pins[name_held]
                total += older.get("bytes") or 0
                if age and total > PARK_BUDGET:
                    spent.append(older["parked"])
                    older["parked"] = None
            self.cv.notify_all()
        for gone in spent:
            remove(gone)
        # The map is what vouches for these files on the next run, and adopt
        # deletes any copy it does not name. Written only from the signal
        # handler it was a shutdown's worth behind at every other moment, so a
        # crash, an OOM kill or a power cut threw away every copy this run had
        # made. A park is rare - 952 in three days - and this is a small file.
        if kept or spent:
            self.save_pins()
        EVENTS.write("park", conv=short, backend=be["name"], slot=slot,
                     ok=bool(kept), bytes=written if kept else 0,
                     secs=round(time.time() - began, 2))
        if kept:
            print(f"[router] parked {short} from {be['name']} slot {slot}", flush=True)
        return kept

    def recall(self, conv, be, slot, post):
        """Put a parked cache back, on the backend about to serve it.

        Returns True when the cache is now on that backend. Without it a
        conversation that lost its slot reads its whole prompt again."""
        with self.cv:
            record = self.pins.get(conv)
            if not record or not record.get("parked"):
                return False
            # Already here, and still in a slot. The slot has to be part of
            # the test: acquire re-pins to the serving backend before this
            # runs, so the name always matches and only the slot says whether
            # the cache survived. _take keeps the slot across a backend that
            # did not change, and clears it across one that did.
            if record["backend"] == be["name"] and record["slot"] is not None:
                return False
            target_slot = slot
            name = record["parked"]

        began = time.time()
        try:
            post(be["url"], f"/slots/{target_slot}?action=restore",
                 {"filename": name})
        except Exception as err:
            print(f"[router] {short_key(conv)} recall failed on {be['name']}: {err}",
                  flush=True)
            EVENTS.write("recall", conv=short_key(conv), backend=be["name"],
                         slot=target_slot, ok=False, error=str(err)[:120],
                         secs=round(time.time() - began, 2))
            return False

        with self.cv:
            record = self.pins.get(conv)
            if record:
                record["backend"] = be["name"]
                record["slot"] = target_slot
            note_bytes = record.get("bytes", 0) if record else 0
            self.note_file("recalled", conv, be, target_slot, note_bytes)
        EVENTS.write("recall", conv=short_key(conv), backend=be["name"],
                     slot=target_slot, ok=True, bytes=note_bytes,
                     secs=round(time.time() - began, 2))
        print(f"[router] recalled {short_key(conv)} onto {be['name']} slot {target_slot}",
              flush=True)
        return True

    def warm_prefix(self, conv, cuts, messages, system, tools, be, slot,
                    post, path):
        """Load the opening this request shares into a slot on this backend.

        Only one the router already has: that is a file read, and the request
        that follows extends the same slot, so it costs almost nothing. Reading
        one instead costs tens of thousands of tokens, so an opening this
        request could have used and the router lacks is written down for the
        builder and left there.

        Returns True when an opening was loaded."""
        with self.cv:
            if not cuts:
                return False
            record = self.pins.get(conv)
            if record and (record.get("parked") or record["slot"] is not None):
                return False       # it has a cache of its own, which is better
            saved = self.openings
            stored = deepest_shared(cuts, saved)

            base = cuts[0]
            # A key names its own contents, so a cut is on one shelf or on
            # neither: the first cut is a system prompt by construction.
            wanted = base[1] not in saved
            # Nobody has this opening. One request reads and saves it, and any
            # that start meanwhile wait, rather than read the same tokens.
            plan = None
            if stored:
                plan = ("load", stored[1], saved[stored[1]], slot)
            elif wanted:
                if base[1] in self.building:
                    plan = ("wait", base[1], None, None)
                else:
                    self.building[base[1]] = time.time()
                    plan = ("read", base[1], None, slot)
            # Every new session starts from a system prompt on its own, so it
            # is wanted whatever else is saved - unless it is already in hand,
            # where a want on top would have the builder read it twice.
            if wanted and plan is None:
                self.note_want(base, "base-", system, tools,
                               messages[:base[0] + 1], path)
            # Deeper only where a slot already holds more of the same opening,
            # and only past what is saved.
            seen = set().union(*self.holds.values()) if self.holds else set()
            shared = deepest_shared(cuts, seen)
            if (DEEP_OPENINGS and shared and shared[1] != base[1]
                    and shared[1] not in saved
                    and (stored is None or shared[0] > stored[0])):
                self.note_want(shared, "deep-", system, tools,
                               messages[:shared[0] + 1], path)

            # What a fork could have started from, had anything offered it.
            # Measurement only. `shared` above needs the parent to hold a slot,
            # and there are always more parked conversations than slots, so the
            # fork rate it shows is a floor. A parked copy is a saved state like
            # any other - recall and _load_prefix issue the same restore call -
            # so a hit here is a prompt read that need not have happened.
            copied, copied_from = None, None
            for other, other_pin in self.pins.items():
                if other == conv or not other_pin.get("parked"):
                    continue
                hit = deepest_shared(cuts, other_pin.get("holds") or ())
                if hit and hit[1] != base[1] and (copied is None
                                                  or hit[0] > copied[0]):
                    copied, copied_from = hit, other

            self.choices[conv] = {"cuts": len(cuts),
                                  "stored": stored[0] if stored else None,
                                  "shared": shared[0] if shared else None,
                                  "copied": copied[0] if copied else None,
                                  "held": len(set().union(*self.holds.values())
                                             if self.holds else set())}
            self.choices.move_to_end(conv)
            while len(self.choices) > RECENT_REQUESTS:
                self.choices.popitem(last=False)

            # A request sharing more than the opening everyone shares has
            # branched off somebody's session: a subagent given the parent's
            # transcript, or a second window on one conversation. The only sign
            # of the parent is a slot holding the deep cut, and the pins say
            # whose slot that is.
            if shared and shared[1] != base[1]:
                holder = next((c for c, p in self.pins.items()
                               if shared[1] in self.holds.get(
                                   (p["backend"], p["slot"]), ())), None)
                if holder and holder != conv \
                        and self.forked.get(conv) != (holder, shared[0]):
                    self.forked[conv] = (holder, shared[0])
                    # Assigning an existing key leaves it where it was, so
                    # without this a conversation that forked again still sat
                    # at the front and was the next one dropped - and the
                    # dedupe above then had nothing to compare against and
                    # wrote the same fork twice. choices, wants and pins all
                    # move_to_end at the same spot.
                    self.forked.move_to_end(conv)
                    while len(self.forked) > RECENT_REQUESTS:
                        self.forked.popitem(last=False)
                    EVENTS.write("fork", conv=short_key(conv),
                                 parent=short_key(holder), depth=shared[0],
                                 cuts=len(cuts))
            # The cut keys, not just how deep: without them nothing offline
            # can ask whether some copy already held the cut this request
            # wanted, which is the whole question deep openings exist for.
            EVENTS.write("choice", conv=short_key(conv),
                         base=short_key(base[1]),
                         stored=stored[0] if stored else None,
                         stored_key=short_key(stored[1]) if stored else None,
                         shelf=shelf_of(saved[stored[1]]) if stored else None,
                         shared=shared[0] if shared else None,
                         shared_key=short_key(shared[1]) if shared else None,
                         copied=copied[0] if copied else None,
                         copied_key=short_key(copied[1]) if copied else None,
                         copied_from=short_key(copied_from) if copied else None,
                         cuts_deep=cuts[-1][0] if cuts else None,
                         plan=plan[0] if plan else None)

            if plan is None:
                return False

        if plan[0] == "load":
            return self._load_prefix(plan[1], plan[2], be, plan[3], post)
        if plan[0] == "wait":
            return self._wait_for_opening(plan[1], be, slot, post)
        try:
            return self._read_prefix(base, messages, system, tools, be,
                                     plan[3], post, drop_file, "base-", path)
        finally:
            with self.cv:
                self.building.pop(plan[1], None)
                self.cv.notify_all()

    def _wait_for_opening(self, key, be, slot, post):
        """Wait for another request to save the opening, then use it.

        Reading it here too would be the same tokens twice: five sessions
        starting together read 92,000 tokens between them where 24,000 would
        have done, and the last finished after seventeen minutes. The wait is
        exactly as long as the read it replaces."""
        deadline = time.time() + BUILD_PATIENCE
        with self.cv:
            while key in self.building and time.time() < deadline:
                self.cv.wait(1.0)
            name = self.openings.get(key)
            if not name:
                return False        # it failed or timed out, so read it here
        return self._load_prefix(key, name, be, slot, post)

    def note_want(self, cut, mark, system, tools, head, path):
        """Write down an opening worth having, with what it takes to read it.

        The text has to be kept, because the request it came from is gone by
        the time the builder runs. Newest last, and a short list, so a prompt
        nobody sends any more cannot hold a place forever."""
        with self.cv:
            fresh = cut[1] not in self.wants
            self.wants[cut[1]] = {"cut": cut, "mark": mark, "system": system,
                                  "tools": tools, "head": list(head),
                                  "path": path, "at": time.time()}
            self.wants.move_to_end(cut[1])
            while len(self.wants) > WANT_KEEP:
                old, gone = self.wants.popitem(last=False)
                EVENTS.write("want", key=short_key(old), shelf=mark_shelf(gone),
                             action="dropped",
                             age=round(time.time() - gone.get("at", 0), 1))
            if fresh:
                EVENTS.write("want", key=short_key(cut[1]), shelf=mark_shelf(mark),
                             action="added")

    def build_once(self, post, remove=drop_file):
        """Read one wanted opening into an idle slot. Return its name, or None.

        One at a time, and only where the backend can spare a slot. The slot
        is held for the read, so a request cannot be admitted into it."""
        with self.cv:
            if not self.wants:
                return None
            # Reading an opening is reading a prompt, so the same rule holds:
            # only a backend that prefills may do it.
            idle = [(b, self._idle_slot(b)) for b in self.backends
                    if prefills(b)]
            idle = [pair for pair in idle if pair[1] is not None]
            if not idle:
                return None
            # The backend that keeps the most slots spare, so a single-slot
            # instance is not tied up for minutes.
            be, slot = max(idle, key=lambda p: p[0]["slots"] - p[0]["busy"])
            key, want = next(reversed(self.wants.items()))
            be["busy"] += 1

        # The builder is the second way into a slot, and the only one that did
        # not copy out what was there. A cpu slot holds a finished
        # conversation's only cache - the park runs on the generator - so reading
        # an opening over it lost that cache while the pin still said the slot
        # was held. IDLE_POLLS makes that rarer, not survivable.
        self.ensure_parked(be, None, post, remove)
        began = time.time()
        try:
            kept = self._read_prefix(want["cut"], want["head"], want["system"],
                                     want.get("tools") or [], be, slot, post,
                                     remove, want["mark"], want["path"])
        finally:
            with self.cv:
                be["busy"] -= 1
                # Dropped either way: a read that failed will fail again, and
                # the next request that misses this opening asks again.
                self.wants.pop(key, None)
                self.cv.notify_all()
        EVENTS.write("want", key=short_key(key), shelf=mark_shelf(want),
                     action="built", backend=be["name"], slot=slot,
                     ok=bool(kept), secs=round(time.time() - began, 1),
                     age=round(began - want.get("at", began), 1))
        return key if kept else None

    def park_all(self, post, timeout=POST_TIMEOUT, only=None, budget=None):
        """Copy live caches to disk, so a stop does not throw them away.

        `only` names one backend, for a drain. Without it every backend is
        copied, which is what a shutdown wants.

        A conversation mid-turn is skipped, because its slot is busy and
        the save would wait for the very turn the stop is ending.

        Each conversation is tried once. A refused save leaves the pin as
        it found it, so without this the same one is picked again and the
        loop never ends. A full disk at SIGTERM spun here instead of
        reaching save_pins."""
        parked = 0
        # `budget` is a wall clock across every backend, for the caller that
        # has one: the signal handler gets 90 s from stop-all.sh before it is
        # SIGKILLed, and save_pins runs after this. A drain passes none - it
        # has DRAIN_DEADLINE and nothing waiting on it, and cutting a save
        # short there is the whole cost this exists to avoid.
        stop = time.time() + budget if budget else None
        for be in self.backends:
            if not be["up"] or (only and be["name"] != only):
                continue
            tried = set()
            while True:
                with self.cv:
                    live = [(conv, record["slot"])
                            for conv, record in self.pins.items()
                            if record["backend"] == be["name"]
                            and conv not in tried
                            and record["slot"] is not None
                            and not record["inflight"]
                            and not copy_is_current(record)]
                    if not live:
                        break
                    # The per-save timeout is capped by what is left of the
                    # budget too, or a save started just inside it runs on well
                    # past the end. Asked with the work in hand, so an empty
                    # backend does not report running out of time.
                    each = timeout
                    if stop is not None:
                        each = min(timeout, stop - time.time())
                        if each <= 0:
                            print(f"[router] out of time with "
                                  f"{len(live)} cache(s) still in a slot on "
                                  f"{be['name']}", flush=True)
                            return parked
                    conv, slot = live[0]
                    self.pins[conv]["inflight"] = True   # hold it still
                    tried.add(conv)
                if self._save_park(conv, be, slot,
                                   lambda url, path, payload, secs=each:
                                   post(url, path, payload, secs)):
                    parked += 1
        return parked

    def save_openings(self):
        """Write down what each opening has earned, for the next run.

        The order is the shelf's own, least recently used first, which is the
        order the budget drops them in. `loads` is the count the dashboard
        shows, which reset to zero on every restart while the openings it
        counted stayed on disk."""
        with self.cv:
            rows = [{"key": key, "file": name, "loads": self.loads.get(key, 0)}
                    for key, name in self.openings.items()]
        write_rows(openings_file(), rows)
        return len(rows)

    def save_pins(self):
        """Write down whose cache each copy holds, for the next run."""
        with self.cv:
            kept = [{"conv": conv, "file": record["parked"],
                     "tokens": record.get("tokens", 0),
                     "bytes": record.get("bytes", 0),
                     "turns": record.get("turns", 1),
                     # The budget sweep orders by this, so it has to survive
                     # the restart with the file it describes.
                     "parked_at": record.get("parked_at")}
                    for conv, record in self.pins.items() if record.get("parked")]
        write_rows(pins_file(), kept)
        return len(kept)

    def hand_off(self, conv, source, tokens, post, remove=drop_file, wanted=None):
        """Move a conversation to the backend it generates on.

        The prefiller is released before the wait to generate, not after,
        so it takes the next prompt while this turn waits. Waiting the
        other way round left three prefillers idle for seven minutes on
        one reply.

        Between the save and the restore the conversation is parked and
        nowhere else, so a failure here leaves it parked, not lost.

        Returns the backend to generate on. That is `source` when nothing
        was carried. None means nobody is waiting any more, and the caller
        holds no backend."""
        if not HANDOFF_ON:
            return self._stay(source, "the handoff is turned off")
        target = self.generator(tokens)
        while target is None and not generates(source):
            # Nothing to carry this to, and the instance holding it is set not
            # to generate. Wait for a generator rather than break the setting -
            # the client leaving is what ends the wait, as everywhere else
            # here. This one waits holding a prefill slot rather than parked,
            # which is why it is the only wait that would rather not exist.
            if wanted is not None and not wanted():
                return None
            with self.cv:
                self.cv.wait(1.0)
            target = self.generator(tokens)
        if target is None or target is source:
            return self._stay(source, "nothing that generates can take it")
        with self.cv:
            record = self.pins.get(conv)
            slot = record["slot"] if record else None
        if slot is None:
            return self._stay(source, "its prompt is in no slot to carry")

        if not self._save_park(conv, source, slot, post, remove):
            return self._stay(source, "the slot had already changed hands")
        with self.cv:
            name = self.pins[conv]["parked"]
            written = self.pins[conv].get("bytes") or 0
            source["busy"] -= 1            # the reader takes the next prompt
            self.flow.note(conv, "generate-queue")
            self.cv.notify_all()

        while True:
            free = self._wait_to_generate(target, wanted)
            if free is not None:
                break
            if wanted is not None and not wanted():
                return None                # parked, and nobody to answer
            if generates(source):
                # The generator went away. Answering slowly beats not
                # answering, so take the prefiller back and generate where it
                # read.
                with self.cv:
                    source["busy"] += 1
                return source
            # `generate: false` says this instance does not generate, and a
            # convenient moment does not change that: an operator who set it
            # meant it, and the turn is parked on disk, which is the cheapest
            # place it could be waiting. So wait for a generator, the way
            # everything else here waits - the client leaving is what ends it.
            target = self.generator(tokens)
            if target is None:
                with self.cv:
                    self.cv.wait(1.0)

        try:
            post(target["url"], f"/slots/{free}?action=restore",
                 {"filename": name})
        except Exception as err:
            print(f"[router] {short_key(conv)} could not be carried to "
                  f"{target['name']}: {err}", flush=True)
            with self.cv:
                target["busy"] -= 1
                source["busy"] += 1        # it generates where it read instead
                self.cv.notify_all()
            return self._stay(source, f"{target['name']} refused the restore")

        with self.cv:
            record = self.pins.get(conv)
            if record:
                record["backend"] = target["name"]
                record["slot"] = free
                record["inflight"] = True
            self.note_file("moved", conv, target, free, written)
            self.flow.note(conv, "generate", target["name"], free)
            self.cv.notify_all()
        EVENTS.write("migrate", conv=short_key(conv), src=source["name"],
                     dst=target["name"], bytes=written)
        print(f"[router] {short_key(conv)} read on {source['name']}, "
              f"generates on {target['name']} slot {free}", flush=True)
        return target

    def _stay(self, source, why):
        """Generate where the prompt was read, because carrying it failed.

        Said out loud when the instance is one an operator set `generate` off
        on. The main path waits for a generator instead - the turn is parked on
        disk and costs nothing to hold there. These are the paths with nothing
        left to wait with: nothing was carried, so there is no copy for a
        generator to restore, and the alternative to breaking the preference is
        throwing away a prompt that took tens of minutes to read."""
        if not generates(source):
            print(f"[router] generating on {source['name']} though it is set "
                  f"not to: {why}", flush=True)
        return source

    def generator(self, tokens):
        """The backend turns migrate to once their prompt is read, or None.

        One that generates and does not prefill. Carrying a turn costs a
        save and a restore, so it is only worth reaching an instance outside
        the prefilling pool. Where every instance does both, a turn
        generates in the slot its prompt already sits in, and this is None.

        None also when the one configured is down, draining or too small."""
        with self.cv:
            for be in sorted(self.backends, key=lambda b: b["pref"]):
                if prefills(be) or not generates(be):
                    continue
                if be["up"] and not be.get("draining") and tokens <= be["n_ctx"]:
                    return be
            return None

    def _wait_to_generate(self, target, wanted):
        """Hold on until the generator has a slot. Returns the slot id, or None.

        A generation runs for seconds where a prefill runs for minutes, so this
        queue drains quickly even where there is one slot to generate in.

        None when the generator can no longer serve this turn, or when the
        client has stopped waiting."""
        with self.cv:
            self.to_generate += 1
        try:
            while True:
                with self.cv:
                    if not target["up"] or target.get("draining"):
                        return None
                    if target["busy"] < target["slots"]:
                        free = self._free_slot(target)
                        if free is not None:
                            target["busy"] += 1   # held against the request path
                            return free
                    if wanted is not None and not wanted():
                        return None
                    self.cv.wait(1.0)
        finally:
            with self.cv:
                self.to_generate -= 1

    def park_later(self, be, conv, post, ticket, remove=drop_file):
        """Copy a cache out of a backend that cannot read it again, on a worker.

        The copy runs to gigabytes and the client already has its reply, so
        the request thread does not wait for it. Two things go with the job.

        The slot is reserved before this returns. `inflight` is what stops
        pick_slot handing it to another conversation, wherever the copy runs.

        The turn ticket goes too, because the copy overwrites the same file
        the next turn would restore from. Holding the ticket is how that
        turn waits.

        Returns False when there is nothing to copy. The caller then still
        owns the ticket."""
        if prefills(be):
            return False              # it can be read again where it is
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if not record or record["slot"] is None:
                return False
            slot = record["slot"]
            record["inflight"] = True      # hold it still while it copies
            if self.parker is None:
                self.parker = threading.Thread(target=self._run_parks,
                                               name="park", daemon=True)
                self.parker.start()
        self.park_jobs.put((conv, be, slot, post, remove, ticket))
        return True

    def _run_parks(self):
        """Write the queued copies, ending each turn once its copy has landed.

        One worker rather than one thread a copy: two multi-gigabyte writes at
        once only divide the same disk between them, and the turn behind each
        waits either way."""
        while True:
            job = self.park_jobs.get()
            conv, be, slot, post, remove, ticket = job
            try:
                self._save_park(conv, be, slot, post, remove)
            except Exception as err:
                print(f"[router] {short_key(conv)} could not be put away: "
                      f"{err}", flush=True)
            finally:
                # Whatever happened above, and before task_done so a drain
                # cannot return while a ticket is still out. claim_turn has no
                # deadline, so a ticket that never comes back wedges this
                # conversation for as long as the router runs - which is worse
                # than any copy not written.
                self.finish_turn(conv, ticket)
                self.park_jobs.task_done()

    def drain_parks(self, timeout=30.0):
        """Wait for the queued copies to land. For tests and for shutdown."""
        end = time.time() + timeout
        while time.time() < end:
            if self.park_jobs.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self.park_jobs.unfinished_tasks == 0

    def park_partial(self, conv, be, slot, post, remove=drop_file):
        """Keep what an abandoned read got through, so the retry starts there.

        A prompt is read from the front, so a cancelled read leaves the slot
        holding the front of the turn the client will send again.

        Thrown away, a prompt too long for the client's patience can never
        be read at all: every attempt starts from the shared opening and
        gives up in the same place. Measured once at a 114,354 token turn,
        abandoned twice after an hour, two thirds read each time."""
        if not conv:
            return False
        with self.cv:
            record = self.pins.get(conv)
            if not record:
                return False
            if copy_is_current(record):
                return False               # already on disk for this turn, and
                                           # saving again from a slot it has
                                           # left would find nothing and delete
                                           # the copy. A copy from an earlier
                                           # turn is exactly what this exists
                                           # to get past: the slot holds the
                                           # front of the turn that was
                                           # abandoned, which is further on.
            record["inflight"] = True      # hold it still while it copies
        return self._save_park(conv, be, slot, post, remove)

    def note_holds(self, conv, be, cuts):
        """Record what a slot holds now, so a later request can start from it."""
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if not record or not cuts:
                return
            # Which opening this conversation was last left on. Its copy on
            # disk begins there, so the next turn can tell whether the copy is
            # worth restoring before it sends anything.
            record["opening"] = cuts[0][1]
            # Every cut, kept past the slot. The copy on disk holds all of
            # them, so this is what says how deep a fork of this conversation
            # could start - a question self.holds cannot answer, being wiped
            # the moment the slot goes.
            record["holds"] = {key for _, key in cuts}
            if record["slot"] is None:
                return
            self.holds[(be["name"], record["slot"])] = {key for _, key in cuts}
            self.holds_depth[(be["name"], record["slot"])] = max(i for i, _ in cuts)

    def forget_stale_park(self, conv, cuts, remove=drop_file):
        """Drop a copy whose opening the client has changed since.

        A copy is only useful as a prefix of the turn coming in, and every
        turn starts with the opening. Change one sentence of one tool
        description and every token after it differs.

        llama.cpp then restores the state, finds no checkpoint at or before
        where the prompts part, and reads everything again. Measured once at
        117,847 tokens, for prompts that parted at token 503.

        A conversation holding its own copy never looks for a shared opening,
        so it loses that too. Declaring the copy dead here puts it back on
        the path that loads one."""
        if not conv or not cuts:
            return False
        with self.cv:
            record = self.pins.get(conv)
            if not record or not record.get("parked"):
                return False
            was = record.get("opening")
            if was is None or was == cuts[0][1]:
                return False        # unchanged, or from before this was kept
            name = record["parked"]
            record["parked"] = None
            record["bytes"] = 0
            # Whatever is in the slot begins with the same dead opening.
            record["slot"] = None
        print(f"[router] {short_key(conv)} starts differently now, so its "
              f"copy is no use: reading from the opening instead", flush=True)
        remove(name)
        return True

    def _load_prefix(self, key, name, be, slot, post):
        """Put a saved opening back into a slot."""
        began = time.time()
        try:
            post(be["url"], f"/slots/{slot}?action=restore", {"filename": name})
        except Exception as err:
            print(f"[router] opening {key[:8]} failed to load on "
                  f"{be['name']}: {err}", flush=True)
            EVENTS.write("load", key=short_key(key), backend=be["name"],
                         slot=slot, ok=False, error=str(err)[:120])
            return False
        # Sized here rather than left at zero: this is the one file event that
        # reads the nvme, and the dashboard draws each one crossing in the time
        # its bytes take. Without a size the largest read on the page was drawn
        # as "0 MiB" in the floor of half a second.
        read = file_size(name)
        with self.cv:
            shelf = None
            if key in self.openings:
                self.openings.move_to_end(key)  # in use, so keep it longest
                shelf = shelf_of(self.openings[key])
            self.loads[key] = self.loads.get(key, 0) + 1
            self.note_file("loaded opening", key, be, slot, read)
            loads = self.loads[key]
        self.save_openings()      # a load is what earns an opening its place
        EVENTS.write("load", key=short_key(key), shelf=shelf,
                     backend=be["name"], slot=slot, ok=True,
                     bytes=read, secs=round(time.time() - began, 2),
                     loads=loads)
        print(f"[router] loaded opening {key[:8]} onto {be['name']} "
              f"slot {slot}", flush=True)
        return True

    def _read_prefix(self, cut, messages, system, tools, be, slot, post,
                     remove, mark, path):
        """Read one opening into a slot, then keep a copy of the slot."""
        index, key = cut
        name = f"{mark}{key}.park"
        began = time.time()
        link_block(name)           # so the save lands on the faster disk
        try:
            block = self._render_block(system, tools, messages[:index + 1],
                                       be, post, path)
            # The reply to a zero-token read carries the timings: tokens
            # processed and tokens the prompt cache skipped. The one place the
            # cost of an opening is measured rather than guessed.
            read = post(be["url"], "/completion",
                        {"prompt": block, "n_predict": 0, "cache_prompt": True,
                         "id_slot": slot}, timeout=READ_TIMEOUT) or {}
            answer = post(be["url"], f"/slots/{slot}?action=save",
                          {"filename": name}) or {}
        except Exception as err:
            print(f"[router] opening {key[:8]} failed to save on "
                  f"{be['name']}: {err}", flush=True)
            remove(name)           # take back the link made before the read
            EVENTS.write("build", key=short_key(key), shelf=mark_shelf(mark),
                         backend=be["name"], slot=slot, ok=False,
                         error=str(err)[:120],
                         secs=round(time.time() - began, 1))
            return False
        # Measured on the disk, as _save_park measures a conversation's copy:
        # an unpatched backend reports the state without the checkpoint
        # trailer, and judging an opening by that number would delete a good
        # one as "changed hands" and read it again on the next pass. The
        # backend's figure is the fallback for a stub that writes no file.
        written = file_size(name) or (answer.get("n_written") or 0)
        if written < PARK_FLOOR:
            remove(name)           # the slot had already changed hands
            EVENTS.write("build", key=short_key(key), shelf=mark_shelf(mark),
                         backend=be["name"], slot=slot, ok=False,
                         error="slot changed hands",
                         secs=round(time.time() - began, 1))
            return False

        with self.cv:
            self.openings[key] = name
            self.openings.move_to_end(key)      # just read, so the newest
            self.opening_bytes[key] = written
            self.note_file("kept opening", key, be, slot, written)
            # Whatever is left over goes, least recently used first. `building`
            # is spared: those have no file to count yet, and dropping one
            # would delete an opening a waiting request is about to load.
            dropped = trim_openings(self.openings, self.opening_bytes,
                                    keep=set(self.building))
        for extra in dropped:
            remove(extra)
        self.save_openings()
        timing = read.get("timings") or {}
        EVENTS.write("build", key=short_key(key), shelf=mark_shelf(mark),
                     backend=be["name"], slot=slot, ok=True,
                     secs=round(time.time() - began, 1),
                     bytes=written,
                     prompt_n=timing.get("prompt_n"),
                     cache_n=timing.get("cache_n"))
        print(f"[router] read and kept opening {key[:8]} on {be['name']} "
              f"slot {slot}", flush=True)
        return True

    @staticmethod
    def _render_block(system, tools, head, be, post, path):
        """One opening, as the backend's own template renders it.

        Two renderings that differ only in what follows the opening share
        exactly that opening, so there is no template to copy here.

        Each message goes through the route its own protocol uses.
        /apply-template reads openai-shaped messages and refuses anthropic
        tool_use and tool_result blocks, so an opening from /v1/messages is
        rendered by the anthropic route - the same conversion generation puts
        it through."""
        route = template_route(path)
        extra = {"tools": tools} if tools else {}
        # The anthropic route takes the system prompt in its own field, and
        # llama.cpp normalises it there before it renders - Claude Code opens
        # it with `x-anthropic-billing-header: ...cch=<hash>`, and the
        # converter rewrites that hash to fffff (server-chat.cpp
        # normalize_anthropic_billing_header). Sent as a message instead it
        # went through untouched, so the block this saved began cch=<hash>
        # where every real turn begins cch=fffff: the prompts part about
        # fifteen tokens in, a restored slot has no checkpoint below that, and
        # the whole prompt is read again - counted as a hit. The converter
        # pushes the system field on as the first message, so the list it
        # renders is the same one either way.
        opening = list(head)
        if system:
            if route.startswith("/v1/messages"):
                extra["system"] = system
            else:
                opening = [{"role": "system", "content": system}] + opening
        full = post(be["url"], route,
                    dict(extra,
                         messages=opening + [{"role": "user", "content": "x"}]))
        alone = post(be["url"], route, dict(extra, messages=opening))
        return common_prefix(full["prompt"], alone["prompt"])

    def status(self):
        """Report what the backends are really doing.

        `busy` counts what this router admitted, `active` what the backend
        says is processing. They differ when something else is talking to the
        backends, or after a router restart, and the dashboard wants the truth
        rather than this process's bookkeeping."""
        now = time.time()
        mounts = self._read_mounts(now)
        with self.cv:
            keys = ("name", "url", "model", "slots", "n_ctx", "busy", "up", "served",
                    "stats", "slots_detail", "cache", "draining")
            rows = []
            # Ordered by where they run, so one socket reads together.
            for be in sorted(self.backends, key=lambda b: by_place(b["name"])):
                row = {k: be[k] for k in keys}
                row["prefill"] = prefills(be)
                row["generate"] = generates(be)
                row["node"] = be.get("node")
                row["config"] = self.read_settings(be)
                detail = be.get("slots_detail") or []
                row["active"] = (sum(1 for s in detail if s["busy"]) if detail
                                 else be["busy"])
                rows.append(row)

            # Sizes from opening_bytes, not a stat each: this runs under the
            # lock on every push, and the shelves hold tens of files.
            def shelf(which, kind):
                return [{"name": key[:8], "file": name, "kind": kind,
                         "bytes": self.opening_bytes.get(key, 0),
                         "loads": self.loads.get(key, 0)}
                        for key, name in self.openings.items()
                        if shelf_of(name) == which]
            openings = {"bases": shelf("base", "system prompt"),
                        "deeps": shelf("deep", "shared history"),
                        "wants": [{"name": key[:8], "kind": want["mark"].rstrip("-")}
                                  for key, want in self.wants.items()]}
            # `backend` is the instance this conversation last ran on; `slot` is
            # set only while that instance's slot still holds the cache, and is
            # cleared the moment something displaces it. Without the slot the
            # dashboard cannot tell "its cache is in that slot" from "it ran
            # there once", and three copies naming one single-slot backend look
            # like three caches in one slot.
            # `parked_at` is what the PARK_BUDGET sweep orders by: it keeps the
            # newest copies and drops the oldest, so without it the dashboard
            # can only list the copies, never say which one goes next.
            copies = [{"name": p["parked"], "kind": "copy", "conv": short_key(conv),
                       "bytes": p.get("bytes") or 0, "backend": p["backend"],
                       "slot": p.get("slot"), "parked_at": p.get("parked_at")}
                      for conv, p in self.pins.items() if p.get("parked")]
            disk = disk_summary(self.pins, self.openings, self.opening_bytes,
                                self.wants)
            disk["files"] = openings["bases"] + openings["deeps"] + copies
            disk["mounts"] = mounts
            machine = self.machine.report(self.backends)
            held = [{"backend": name, "slot": slot, "cuts": len(keys),
                     "through": self.holds_depth.get((name, slot))}
                    for (name, slot), keys in sorted(self.holds.items()) if keys]
            return {"backends": rows,
                    "slots_hold": held,
                    "flow": self.flow.report(),
                    "rates_since": self.rates_since,
                    "cache_choices": [dict(self.choices[c], conv=short_key(c))
                                      for c in reversed(self.choices)],
                    "machine": machine,
                    "waiting": len(self.waiters),
                    "waiting_to_generate": self.to_generate,
                    "waiting_detail": self._waiting_detail(now),
                    "pinned_conversations": len(self.pins),
                    "saved_prompts": len(self.openings),
                    "recent_files": list(self.recent),
                    "recent_requests": list(self.recent_requests),
                    "history": self.history.snapshot(),
                    "openings": openings,
                    "disk": disk}


ON_THE_PAGE = ("lib", "views", "components")   # directories the browser needs
PAGE_FILES = ("index.html", "shell.js", "shell.css")


def on_the_page(rel):
    """True for a path the browser needs to draw the dashboard.

    The web directory also holds what built the page: node_modules, the
    checks, the package files. None of it belongs on the wire, and
    node_modules alone is 31 MB of someone else's code."""
    rel = (rel or "").strip("/")
    if not rel:
        return True                       # a directory, answered by its index
    head = rel.split("/", 1)[0]
    return head in ON_THE_PAGE or rel in PAGE_FILES


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen-router"

    def log_message(self, fmt, *args):
        pass                                   # the router prints its own line

    def do_GET(self):
        if self.path.rstrip("/") == "/router/json":
            return self._send(200, json.dumps(POOL.status(), indent=2).encode())
        if self.command == "GET" and self.path.split("?")[0] in ("", "/", "/router"):
            # The backends serve their own web ui at the root, which is not
            # what a reader wants here. /router needs the trailing slash so the
            # page's relative urls resolve under /router/ rather than the root.
            # Exact matches only: /router/ itself must reach the static handler.
            return self._redirect("/router/")
        if self.path.startswith("/router/"):
            rest = self.path[len("/router/"):].split("?")[0]
            if rest == "events":
                return self._events()
            # Take a backend out of service, or put it back, so it can be
            # restarted under the requests waiting for it.
            if rest.startswith("drain/") or rest.startswith("resume/"):
                return self._service(*rest.split("/", 1))
            if rest == "reset-rates":
                if self.command != "POST":
                    return self._error(405, "post to reset the rates")
                POOL.reset_rates(True)
                return self._send(200, json.dumps(
                    {"rates_since": POOL.rates_since}).encode())
            if rest.startswith("config/"):
                return self._config(rest[len("config/"):])
            return self._static(rest or "index.html")
        self._route()

    do_POST = do_DELETE = do_GET

    def _service(self, what, name):
        """Drain a backend, or put it back. POST only: both change the pool."""
        if self.command != "POST":
            return self._error(405, "post to drain or resume a backend")
        if what == "resume":
            done = POOL.resume(name)
            return (self._send(200, json.dumps({"backend": name, "serving": True}).encode())
                    if done else self._error(404, f"no backend called {name}"))
        report = POOL.drain(name, http_post)
        if report is None:
            return self._error(404, f"no backend called {name}")
        # A drain that gave up waiting parked nothing, and one whose saves
        # failed left caches only in slots. Either way say so, rather than let
        # a script stop a backend that still holds live caches.
        ok = report["quiet"] and not report["left"]
        return self._send(200 if ok else 409, json.dumps(report).encode())

    def _redirect(self, where):
        self.send_response(301)
        self.send_header("Location", where)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _config(self, kind):
        """Offer a ready client config, addressed to whatever host was used.

        The address has to come from the request - a config built with
        127.0.0.1 is no use to the laptop that downloaded it - but it is the
        client's own header, so it is checked before it goes into a file the
        reader will point a client at for months. A name, or an address, and a
        port: nothing that could carry a path, a scheme or credentials."""
        host = host_only(self.headers.get("Host"))
        if not host or kind not in CONFIG_FILES:
            return self._error(404, "no such config")
        with POOL.cv:
            up = [be for be in POOL.backends if be["up"]]
            model = next((be["model"] for be in up if be["model"]), "qwen")
            n_ctx = min([be["n_ctx"] for be in up if be["n_ctx"]], default=150000)
        config = client_config(kind, host, model, n_ctx)
        if config is None:
            return self._error(404, "no such config")
        payload = json.dumps(config, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # Make the browser save it instead of showing it.
        self.send_header("Content-Disposition",
                         f'attachment; filename="{CONFIG_FILES[kind]}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send(self, code, payload, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code, message, say=None):
        """Answer with an error, and write it down.

        An error that leaves no trace here is visible only to the client it
        was sent to, and cannot be diagnosed from this side."""
        (say or _say)(f"[router] {code} on {self.command} "
                      f"{self.path.split('?')[0]}: {message}")
        self._send(code, json.dumps({"error": {"message": message}}).encode())

    def _static(self, rel):
        """Serve the dashboard from bin/web. Files are read per request, so
        editing the page needs no restart."""
        if not on_the_page(rel):
            return self._error(404, f"no such file: {rel}")
        base = WEB.resolve()
        try:
            target = (base / rel).resolve()
            target.relative_to(base)              # refuse to escape the web dir
        except (ValueError, OSError):
            return self._error(403, "outside the web directory")
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            return self._error(404, f"no such file: {rel}")

        stat = target.stat()
        etag = f'"{int(stat.st_mtime)}-{stat.st_size}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _events(self):
        """Push the pool status, but only when it changed.

        The client does not poll: it opens one EventSource, which reconnects
        by itself, so a router restart heals without a reload."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        previous = None
        try:
            while True:
                payload = json.dumps(POOL.status())
                if payload != previous:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    previous = payload
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                   # the viewer closed the tab

    def _pool_props(self, path):
        """Answer /props and /slots for the whole pool.

        Forwarding these to one backend makes the router look like a one-slot
        server, and clients plan their concurrency around that."""
        live = [be for be in POOL.backends if be["up"]]
        if not live:
            return self._error(503, "no backend is up")

        if path == "/slots":
            slots = []
            for be in live:
                try:
                    with urllib.request.urlopen(be["url"] + "/slots", timeout=5) as r:
                        part = json.load(r)
                except Exception:
                    continue
                if isinstance(part, list):
                    for slot in part:
                        slot["backend"] = be["name"]
                        slots.append(slot)
            return self._send(200, json.dumps(slots).encode())

        try:
            with urllib.request.urlopen(live[0]["url"] + "/props", timeout=5) as r:
                props = json.load(r)
        except Exception as e:
            return self._error(502, str(e))
        props["total_slots"] = sum(be["slots"] for be in live)
        return self._send(200, json.dumps(props).encode())

    def _route(self):
        # A client's header, so it may be anything. int() on junk raised before
        # a status line had been sent, and BaseHTTPRequestHandler catches only
        # TimeoutError, so the client got a dropped connection and the log got
        # a traceback instead of a 400.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "Content-Length is not a number")
        if length < 0:
            return self._error(400, "Content-Length is negative")
        # Read before anything else looks at it, so a header alone could ask
        # this process to hold arbitrary memory - on the one port the design
        # says is public, on a box whose whole budget is the weights. The 413
        # below is a real limit but it runs after the body is in hand.
        if length > MAX_BODY:
            self.close_connection = True
            return self._error(413, f"body of {length} bytes; this router "
                                    f"reads at most {MAX_BODY}")
        # Nothing here decodes chunked, and `transfer-encoding` is dropped on
        # the way out. Refused rather than read as an empty body: that loses the
        # prompt, and leaves the unread chunks in the socket for the next
        # request on it to parse as a request line.
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            return self._error(411, "send the body with a Content-Length: "
                                    "this router does not read chunked requests")
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?")[0].rstrip("/") or "/"

        if self.command == "GET" and path in ("/props", "/slots"):
            return self._pool_props(path)

        # Both chat apis are POST only, and so is llama.cpp. Without this a
        # bodiless GET on one of these paths took a slot, parked every other
        # cache on the backend, restored gigabytes and loaded an opening -
        # all of it before _forward sent the backend a request it answers 404
        # to. From an unauthenticated public port.
        if path in INFERENCE and self.command != "POST":
            return self._error(405, f"post to {path}")
        if path not in INFERENCE:
            # Named, not "everything else". The backends run with --agent, so
            # anything this forwards is shell and file access handed to whoever
            # reached the router's public port. PASS_THROUGH widens it.
            if path not in PASSED:
                return self._error(404, f"this router does not serve {path}")
            be = next((b for b in POOL.backends if b["up"]), None)
            if not be:
                return self._error(503, "no backend is up")
            return self._forward(be, body)

        vision = POOL.vision()
        tokens, images, image_charge = request_cost(body, vision)
        # `tokens` is what has to fit, so it carries REPLY_TOKENS of room for
        # the answer. The prompt is what was sent, and it is what the dashboard
        # measures a finished turn against: 1,024 tokens nobody sent is 41
        # seconds of reading nobody did, on every turn with no exact split.
        prompt_tokens = max(0, tokens - REPLY_TOKENS)
        largest = POOL.largest()
        if largest and tokens > largest:
            return self._error(413, f"needs about {tokens} tokens. "
                                    f"The largest backend holds {largest}.")
        if not largest:
            return self._error(503, "no backend is up yet")

        # The client's own session id beats a guess from the prompt, and
        # covers /v1/messages, where the system prompt is a separate field
        # conversation_id never sees.
        #
        # `conv_source` says how it was recognised, so a report can tell a
        # client that sends its session id from one identified by a hash that
        # any edit to the opening silently changes. Decided here rather than
        # in a second expression over the same three calls: that one re-parsed
        # the body twice more to learn which branch had just won - on a ctx
        # 150000 turn, megabytes each time - and it carried the precedence
        # separately, so changing the order above would have left every hashed
        # conversation labelled "cache_key".
        conv, conv_source = session_key(self.headers), "header"
        if not conv:
            conv, conv_source = prompt_key(body), "cache_key"
        if not conv:
            conv, conv_source = conversation_id(body), "hash"
        if not conv:
            conv_source = "none"
        client = client_kind(self.headers)
        # Before anything is changed, so a capture holds what the client sent.
        capture(conv, body)
        # This model's template refuses a late system message.
        ordered = hoist_system(body)
        if ordered is not body:
            print(f"[router] a late system message became a user message "
                  f"for {path}", flush=True)
            EVENTS.write("start_over", conv=short_key(conv) if conv else None,
                         reason="late_system", client=client, path=path)
        body = ordered
        # Every point this request could share with another. One saved copy of
        # the deepest starts them both.
        cuts, messages, system, tools = prompt_cuts(body)
        # A full box is a queue, not a refusal. The wait for a slot is part of
        # the reply, so the stream and its keep-alive open before the slot is
        # asked for and run through the wait and the read alike.
        start = time.time()
        asked = read_only(body)
        opened = asked is not None and wants_stream(body)
        self.sending = threading.Lock()        # one writer at a time
        stop_ping = None
        if opened:
            self._open_stream(opening_event(path, body))
            stop_ping = self._ping_until()

        ticket = POOL.begin_wait(conv, tokens, images, image_charge)
        try:
            # The turn ahead in the same conversation holds the pin, the slot
            # and the copy this one needs. Waiting for it before asking for a
            # backend is what stops a second reader moving the same three.
            mine = POOL.claim_turn(conv, ticket, self._still_there)
            be = POOL.acquire(conv, tokens, self._still_there) if mine else None
        finally:
            POOL.end_wait(ticket)
        waited = time.time() - start
        if not be:
            # Either nothing that reads is up, or the client stopped waiting.
            # Only if this turn held the conversation: Flow is keyed by
            # conversation, so a turn that gave up in claim_turn saying "done"
            # deleted the live row of the turn actually running - a 40 minute
            # prefill vanished off the flow view and replayed a completion that
            # never happened.
            if mine:
                POOL.note_stage(conv, "done")
            POOL.finish_turn(conv, ticket)
            if stop_ping:
                stop_ping()
            if opened:
                return self._say_and_end("no backend can serve this request")
            return self._error(503, "no backend can serve this request")
        # Read here, then generate wherever is free. Which backend that is
        # cannot be known before the read, and the read takes minutes.
        serving = be
        left = False                           # the client gave up mid-read
        read_stats = {}
        recalled = loaded = warm = False
        slot = None
        # The backend is held from here, and the conversation with it. Every
        # way out of this block runs the same ending, because a claim left
        # behind stops the conversation for good. Nothing may sit between the
        # claim and the try for the same reason: the claim has no deadline, so
        # a throw before it would hold the conversation until a restart.
        try:
            # Its own cache is still in a slot here, so the turn extends it.
            warm = bool(conv) and POOL.holds_slot(conv)
            # One slot, decided once, because whatever is put there - this
            # conversation's copy, or the opening it shares - is put there for
            # the read below to extend. Asking once per step answered from a
            # two-second-old poll and landed the opening in a slot nothing
            # then read.
            slot = POOL.pick_slot(be, conv)
            POOL.note_stage(conv, "prefill", be["name"], slot)
            # Nothing reaches the backend until the caches on it are safe on
            # disk and this conversation's own cache is back where it will run.
            POOL.ensure_parked(be, conv, http_post)
            # A copy beginning with an opening the client no longer sends
            # cannot be extended. Dropping it lets the opening load instead.
            if POOL.forget_stale_park(conv, cuts):
                EVENTS.write("start_over", conv=short_key(conv) if conv else None,
                             reason="stale_copy", client=client, path=path)
            recalled = POOL.recall(conv, be, slot, http_post)
            loaded = (not recalled
                      and POOL.warm_prefix(conv, cuts, messages, system, tools,
                                           be, slot, http_post, path))
            if asked is not None:
                # Watch the client through the read: one nobody waits for
                # holds the slot its own retry needs. The reply's timings say
                # what the backend processed against its prompt cache, which
                # is the number every cache question comes down to.
                answer = http_post_wanted(be["url"], path, read_only(body, slot),
                                          READ_TIMEOUT, self._still_there)
                timing = (answer or {}).get("timings") or {}
                read_stats = {"read_prompt_n": timing.get("prompt_n"),
                              "read_cache_n": timing.get("cache_n")}
                POOL.note_slot(conv, slot)
                serving = POOL.hand_off(conv, be, tokens, http_post,
                                        wanted=self._still_there)
                if serving is None:
                    # The client left while it waited to generate. Its cache
                    # is parked, and no backend is held.
                    raise Gone("the client stopped waiting for a slot to generate in")
            if serving is be:
                # Nothing was carried, so the slot that read answers where it
                # stands. hand_off already noted a carried turn.
                POOL.note_stage(conv, "generate", be["name"], slot)
            if stop_ping:
                stop_ping()                    # waits for a ping in flight
            self._forward(serving, body, conv, opened=opened)
        except Gone:
            # Nobody to answer. The slot goes back below, and what the read
            # got through is parked so the retry starts there.
            print(f"[router] {short_key(conv)} left while {be['name']} was "
                  f"reading, {time.time() - start:.0f}s in ({self.went})",
                  flush=True)
            left = True
        except Exception as err:
            # Reading is how a request is served, not a step with a way round
            # it. If it fails, the request failed.
            if opened:
                # The reply already started, so there is no status left to
                # send. Say it in the stream and end it.
                self._say_and_end(f"{be['name']}: {err}")
            else:
                self._error(502, f"{be['name']}: {err}")
        finally:
            if stop_ping:
                stop_ping()
            parking = False
            try:
                if serving is not None:
                    POOL.note_holds(conv, serving, cuts)
                    POOL.release(serving, conv)
                # Both after the release, which drops the copy on disk because
                # a served turn leaves the slot ahead of it. These write the
                # copy that is not behind.
                if left:
                    POOL.park_partial(conv, be, slot, http_post)
                # A backend that does not read cannot serve the next turn, so
                # leave a copy for whichever one does. On a worker: the copy
                # runs to gigabytes and the client already has its reply, so
                # only the next turn of this conversation waits for it.
                if serving is not None:
                    parking = POOL.park_later(serving, conv, http_post, ticket)
            except Exception as err:
                # A copy not written costs the next turn its prompt. A turn
                # ticket not given back costs the conversation every turn
                # after it, because claim_turn has no deadline - so whatever
                # happens here, the lines below still run.
                print(f"[router] {short_key(conv)} could not be put away: "
                      f"{err}", flush=True)
            took = time.time() - start
            POOL.note_stage(conv, "done")
            # The worker ends the turn once the copy has landed, because the
            # next turn restores from the file it is writing. Only when there
            # is no copy to write does this thread still own the ticket.
            if not parking:
                POOL.finish_turn(conv, ticket)
            POOL.note_request(conv, be, path, took, waited,
                              how_started(warm, recalled, loaded), prompt_tokens,
                              images=images, image_tokens_=image_charge,
                              **read_stats)
            EVENTS.write("request", conv=short_key(conv) if conv else None,
                         source=conv_source, client=client, path=path,
                         est_tokens=tokens, n_cuts=len(cuts),
                         backend=be["name"],
                         started=how_started(warm, recalled, loaded),
                         took=round(took, 3), waited=round(waited, 3),
                         left=True if left else None,
                         **read_stats)
            print(f"[router] {self.command} {path} -> {be['name']} "
                  f"{took:.1f}s", flush=True)

    sending = None                             # set per request in _route
    went = "closed its end"                    # why the client stopped waiting

    def _still_there(self):
        """False once the client has closed its end.

        A queued request waits without a deadline, so the client leaving is
        the only thing that ends the wait. The body is already read, so
        nothing more arrives: ready to read, with nothing on it, means gone.

        `self.went` records which case it was, because a clean close and a
        broken socket send you to different logs.

        poll, not select: select refuses a descriptor at or above FD_SETSIZE,
        1024, and raises ValueError for a socket whose client is still there."""
        try:
            watch = select.poll()
            watch.register(self.connection, select.POLLIN)
            if not watch.poll(0):
                return True
            if self.connection.recv(1, socket.MSG_PEEK):
                return True
            self.went = "closed its end"
        except (OSError, ValueError) as err:
            # A closed socket has no descriptor left to poll - fileno() is -1,
            # which poll refuses - and an open one is never out of range. So
            # either of these means the same thing: nobody is there.
            self.went = f"{type(err).__name__}: {err}"
        return False

    def _open_stream(self, opening=b""):
        """Answer the client now, before the prompt is read.

        Reading runs for tens of minutes and sends nothing, and a client drops
        a stream that goes quiet, so the reply starts before the read and is
        kept alive through it.

        The stream opens with whatever its protocol opens with. A keep-alive is
        not that: it holds a stream open and starts nothing, and a client
        waiting for a message that never started leaves."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if opening:
            self._chunk(opening)

    def _chunk(self, data):
        """One frame of a chunked reply. The caller holds `sending`."""
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _ping_until(self):
        """Fill the silence with keep-alives. Returns the way to stop.

        Stopping joins the thread. A flag is not enough: the thread can be
        part way through a chunk frame when the next writer starts, and a frame
        carries its own length, so half of one makes the rest of the stream
        unreadable. Both writers take the same lock."""
        stop = threading.Event()
        began = time.time()
        beat = ping_for(self.path.split("?")[0])

        def run():
            while not stop.wait(PING_EVERY):
                with self.sending:
                    if stop.is_set():
                        return
                    try:
                        self._chunk(beat)
                    except Exception as err:
                        # Without this the keep-alive stops silently and the
                        # client times out with nothing said on either side.
                        print(f"[router] keep-alive stopped after "
                              f"{time.time() - began:.0f}s: {err}", flush=True)
                        return

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def done():
            stop.set()
            thread.join(5)
        return done

    def _say_and_end(self, message):
        """Put an error into a stream that has already started, and close it.

        The stream opened with a message_start, so a bare data line is
        nothing an anthropic parser can place. That protocol names an event
        for a stream that ends badly, and this ends in it.

        Under `sending`, like every other writer: the keep-alive thread can
        still be running here, and a ping inside this frame is the
        unreadable stream _ping_until's lock exists to prevent."""
        print(f"[router] {message}", flush=True)
        if anthropic(self.path.split("?")[0]):
            event = sse_event("error", {"type": "error",
                                        "error": {"type": "api_error",
                                                  "message": message}})
        else:
            event = b"data: " + json.dumps(
                {"error": {"message": message}}).encode() + b"\n\n"
        try:
            with (self.sending or threading.Lock()):
                self._chunk(event)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except Exception:
            pass                               # the client left

    def _forward(self, be, body, conv=None, opened=False):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in DROP_HEADERS}
        # llama-server 404s on a trailing slash, which clients joining a base
        # url ending in /v1 with /models produce.
        path, sep, query = self.path.partition("?")
        target = (path.rstrip("/") or "/") + sep + query
        # A streamed openai reply reports no usage unless asked, and usage is
        # what the cache is measured in. So the router asks, reads the figures
        # as they pass, and removes the chunk when the client did not ask. The
        # anthropic route reports usage unprompted and needs only the tee.
        data, oai_tee = body or None, None
        if opened and body and not anthropic(path):
            if wants_usage(body):
                oai_tee = OaiUsageSplice(strip=False)
            else:
                injected = with_usage(body)
                if injected is not None:
                    data, oai_tee = injected, OaiUsageSplice(strip=True)
        request = urllib.request.Request(be["url"] + target, data=data,
                                         headers=headers, method=self.command)
        try:
            upstream = urllib.request.urlopen(request, timeout=FORWARD_TIMEOUT)
        except urllib.error.HTTPError as e:
            upstream = e                       # pass the backend error through
        except Exception as e:
            # The same split every other error path here makes. With `opened`
            # the status line went out half an hour ago, so _error would write
            # a whole second HTTP response inside the chunked body in flight -
            # unparseable, and it poisons the keep-alive connection for the
            # request after it.
            if opened:
                return self._say_and_end(f"{be['name']}: {e}")
            return self._error(502, f"{be['name']}: {e}")

        if upstream.status >= 400:
            shape = request_shape(body)
            print(f"[router] {be['name']} refused {self.path.split('?')[0]} "
                  f"with {upstream.status}: {shape}", flush=True)
            if opened:
                # The reply started long ago, so there is no status left to
                # send, and a refusal body is not an event: passing it through
                # leaves a json object where an SSE frame should be.
                with upstream:
                    reason = said_in(upstream.read()) or upstream.reason
                return self._say_and_end(f"{be['name']}: {reason}")

        with upstream:
            # The reply may already have started, to give the client something
            # to hold while the prompt was read. Then it is chunked and the
            # headers are long gone.
            length = None if opened else upstream.headers.get("Content-Length")
            if not opened:
                self.send_response(upstream.status)
                for k, v in upstream.headers.items():
                    if k.lower() not in DROP_HEADERS:
                        self.send_header(k, v)
                if length:
                    self.send_header("Content-Length", length)
                else:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            # read1 returns what arrived. read waits for a full buffer, which
            # delays streamed tokens.
            read = getattr(upstream, "read1", upstream.read)

            # The backend sends nothing while it reads, which can be an hour,
            # and a client drops a stream quiet for five minutes.
            sending = self.sending or threading.Lock()
            done = threading.Event()
            last = [time.time()]

            def put(data):
                if length:
                    self.wfile.write(data)
                else:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
                last[0] = time.time()

            def keep_alive():
                while not done.wait(1.0):
                    if time.time() - last[0] < PING_EVERY:
                        continue
                    with sending:
                        if done.is_set():
                            return
                        try:
                            put(ping_for(self.path.split("?")[0]))
                        except Exception:
                            return             # client left. The reader sees it too.

            pinger = None
            streamed = wants_ping(upstream.headers.get("Content-Type"), length)
            if streamed:
                pinger = threading.Thread(target=keep_alive, daemon=True)
                pinger.start()

            # This stream began with a message_start of the router's own, so
            # the backend's cannot be let through as well.
            splice = (AnthropicSplice()
                      if opened and streamed and anthropic(path) else None)

            try:
                while True:
                    chunk = read(8192)
                    if not chunk:
                        break
                    if splice:
                        chunk = splice.feed(chunk)
                        if not chunk:
                            continue           # a part event, or the one dropped
                    elif oai_tee:
                        chunk = oai_tee.feed(chunk)
                        if not chunk:
                            continue           # the chunk the router asked for
                    with sending:
                        put(chunk)
                left = (splice.tail() if splice
                        else oai_tee.tail() if oai_tee else b"")
                if left:
                    with sending:
                        put(left)
                if splice and splice.reported:
                    # What the client was finally told. These events pass
                    # through here anyway, so it costs one dict read and
                    # answers "did the cache read show up" for real turns.
                    names = {"input_tokens": "input", "output_tokens": "output",
                             "cache_read_input_tokens": "cache_read",
                             "cache_creation_input_tokens": "cache_write"}
                    EVENTS.write("usage", backend=be["name"],
                                 conv=short_key(conv) if conv else None,
                                 path=self.path.split("?")[0],
                                 **{short: splice.reported[full]
                                    for full, short in names.items()
                                    if isinstance(splice.reported.get(full), int)})
                if oai_tee and oai_tee.usage:
                    details = oai_tee.usage.get("prompt_tokens_details") or {}
                    EVENTS.write("usage", backend=be["name"],
                                 conv=short_key(conv) if conv else None,
                                 path=self.path.split("?")[0],
                                 input=oai_tee.usage.get("prompt_tokens"),
                                 output=oai_tee.usage.get("completion_tokens"),
                                 cached=details.get("cached_tokens"))
                done.set()
                if pinger:
                    pinger.join(2)
                with sending:
                    if not length:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                           # client left. The caller frees the slot.
            except (OSError, http.client.HTTPException) as err:
                # The backend went away part way through the reply. The status
                # line and some of the body are already on the wire, so there
                # is no second reply to send: letting this reach _route's
                # `except Exception` wrote a whole HTTP response *inside* the
                # one in flight, which no client can parse and which poisons a
                # keep-alive connection. Say it in the stream when the stream
                # can carry it, and otherwise stop.
                print(f"[router] {be['name']} stopped mid-reply: {err}", flush=True)
                done.set()
                if pinger:
                    pinger.join(2)
                if not length:
                    self._say_and_end(f"{be['name']}: {err}")
                else:
                    self.close_connection = True   # the body is short of its count
            finally:
                done.set()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # Accept IPv4 on the IPv6 socket. Tailscale gives a machine both, and
        # macOS clients try the IPv6 one first.
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()

    def handle_error(self, request, address):
        """A client hanging up is not an error worth a page of traceback.

        Every client holds its connection open for a next request and drops it
        when it has none. Python's request loop is reading the next request
        line when that happens, so it raises and the default handler prints a
        traceback. A healthy session leaves several in the log, which are the
        first thing read when something really is wrong."""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError,
                                          ConnectionAbortedError)):
            return
        super().handle_error(request, address)


class Stamped:
    """Put the time in front of every line the router prints.

    The log is read hours later, when the question is what happened when.
    Every print in this file writes through this, so no call site has to
    remember."""

    def __init__(self, out):
        self.out = out
        self.fresh = True

    def write(self, text):
        for piece in text.splitlines(keepends=True):
            if self.fresh and piece.strip():
                self.out.write(time.strftime("%m-%d %H:%M:%S "))
            self.out.write(piece)
            self.fresh = piece.endswith("\n")

    def flush(self):
        self.out.flush()


if __name__ == "__main__":
    sys.stdout = Stamped(sys.stdout)
    sys.stderr = Stamped(sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="::")   # "::" means IPv6 and IPv4
    args = parser.parse_args()
    Server.address_family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    POOL = Pool(BACKENDS)
    POOL.adopt()

    stopping = threading.Event()

    def shut_down(signum, frame):
        """Copy every live cache out before the backends go away.

        A cache only exists in a slot. Stopping a backend throws it away, and
        every conversation then reads its whole prompt again."""
        if stopping.is_set():
            sys.exit(1)           # a second signal means stop arguing
        stopping.set()
        print("[router] stopping, parking caches", flush=True)
        # What the worker still holds, first. park_all skips a record that is
        # being copied, and the worker is a daemon thread that the sys.exit
        # below kills wherever it has got to - so without this a stop during a
        # copy loses that copy, and costs that conversation its whole prompt.
        # Free when the queue is empty, which it almost always is.
        began = time.time()
        if not POOL.drain_parks(PARK_ALL_TIMEOUT):
            print("[router] a copy on the worker did not land in time",
                  flush=True)
        parked = POOL.park_all(http_post, timeout=PARK_ALL_TIMEOUT,
                               budget=max(1.0, PARK_ALL_BUDGET
                                          - (time.time() - began)))
        kept = POOL.save_pins()
        print(f"[router] parked {parked} conversation(s), wrote {kept} pin(s)",
              flush=True)
        # The writer is a daemon thread, so sys.exit drops whatever is still
        # queued - and this is the largest batch of park rows the router ever
        # writes, the one tools/cache-report.py reads to say what a stop cost.
        EVENTS.flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shut_down)
    signal.signal(signal.SIGINT, shut_down)
    print(f"[router] {len(BACKENDS)} backend(s) from {BACKENDS_FROM}: "
          f"{', '.join(b['name'] for b in BACKENDS)}", flush=True)
    print(f"[router] listening on {args.host}:{args.port}", flush=True)
    Server((args.host, args.port), Handler).serve_forever()
