#!/usr/bin/env python3
"""Send each turn to the backend that already holds its cache.

A turn on the wrong backend reads the whole prompt again at about 25
tokens a second. Every wait here is cheaper than that.

    ./router.py [--port 8090] [--host 0.0.0.0]

/router is the dashboard. /router/json is the same data.
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

# A prefill is compute bound for tens of minutes. A generation is memory
# bound for seconds. A `generate`-only instance is a generator: turns
# migrate to it, `pref` lowest first. One slot per instance: a slot
# reading a long prompt blocks every other slot on it.
BACKENDS = [
    {"name": "solo", "url": "http://127.0.0.1:8080", "pref": 0,
     "prefill": True, "generate": True, "node": 0},
]

# ROUTER_BACKENDS names a JSON file holding another table: the same fields.
BACKENDS_FROM = "the built-in default"
if os.environ.get("ROUTER_BACKENDS"):
    BACKENDS_FROM = os.environ["ROUTER_BACKENDS"]
    BACKENDS = json.loads(Path(BACKENDS_FROM).read_text())
    # Checked at startup, not on the first turn that needs the field.
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
    # A turn leaves its reader only through the handoff.
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
PIN_PATIENCE  = 20.0   # seconds a conversation waits for the backend
                       # that holds its cache before taking a free one
POLL          = 2.0    # seconds between backend checks
RATE_WINDOW   = 10.0   # seconds a per-slot rate is measured over
PING_EVERY       = 15.0   # seconds of quiet before a ping. Clients
                          # drop a stream after 300 s of silence.
POST_TIMEOUT     = 300.0  # a save waits for the slot to finish its turn
DRAIN_DEADLINE   = 1800.0 # seconds a drain waits for running work
PARK_ALL_TIMEOUT = 75.0   # longest one save may take at shutdown.
                          # Measured: median 11 s, p90 19 s, 77 of 952
                          # saves over 20 s.
PARK_ALL_BUDGET  = 80.0   # wall-clock cap over all shutdown saves.
                          # stop-all.sh gives the router 90 s.
PARK_FLOOR       = 64 * 1024 * 1024   # a real state file is about 112 MiB.
                          # A smaller file means the slot changed hands.
# Disk for the conversation copies, in bytes. A copy is about 115 MiB plus
# 36.6 KiB a token: 0.2 to 5.4 GiB at ctx 150000. Size it to the disk RUN
# is on. Too low displaces a copy still in use, which costs a full re-read.
PARK_BUDGET = int(float(os.environ.get("PARK_BUDGET_GB") or 256) * 1024 ** 3)
# A prompt shorter than this is read again rather than copied to disk. A copy
# costs about the same whatever little it holds, and a prompt this short is
# back in seconds. Measured over 3,806 turns here: a floor halves the saves,
# 496 a day to 229, and costs 75 turns a day a re-read, median 2.4 s and
# worst 40 s. Prompt sizes are in two clumps - one-shot questions near 300
# tokens, sessions at 57,000 and up - so anything from 512 to 4096 does the
# same thing, and 8192 starts cutting into the sessions: the worst re-read
# there is 556 s. 0 turns the floor off and copies everything.
PARK_MIN_TOKENS = int(os.environ.get("PARK_MIN_TOKENS") or 1024)
# A restored slot needs its context checkpoints in the state file, which
# needs patches/slot-state-carries-checkpoints.patch. Without the patch a
# move is followed by a full re-read: set HANDOFF=0.
HANDOFF_ON = os.environ.get("HANDOFF", "1") == "1"
PREFIX_MIN_CHARS = 8000   # about 2000 tokens. A shorter cut is not
                          # worth a file.
BUILD_PATIENCE   = 1800.0 # longest a request waits for another to save the
                          # opening they share
SYSTEM_MIN_CHARS = 2000   # the only cut two sessions share. Claude Code
                          # sends 6,100 characters: a minute to read, a
                          # fifth of a second to load from disk.
# Disk for the saved openings, in bytes. One block is 0.6 to 3.7 GB. Least
# recently used goes first. Size it to the disk BLOCK_DIR is on.
BLOCK_BUDGET = int(float(os.environ.get("BLOCK_BUDGET_GB") or 64) * 1024 ** 3)
# Deeper openings: a cut where two conversations diverge. Off by default:
# over two days of real traffic it was built 0 times and loaded 0 times.
# The detection still runs, and the `choice` event records how deep a fork
# could have started. tools/cache-report.py reads it.
DEEP_OPENINGS = os.environ.get("DEEP_OPENINGS", "0") == "1"
WANT_KEEP        = 8      # openings noted as missing but not built yet
IDLE_POLLS       = 2      # polls a slot must look idle before the builder
                          # reads into it. One poll can be two seconds old.
BUILD_POLL       = 10.0   # seconds between builder passes
STALL_RATE = 0.5          # tokens/s. Under this a generating slot is
                          # stalled behind another slot's read. Measured
                          # 0.02 to 0.06 against 6.3 solo.
HISTORY_STEP = 10.0       # seconds per history bucket
HISTORY_KEEP = 60         # buckets kept: ten minutes
GPU_POLL = 10.0           # seconds between nvidia-smi runs. Each costs 50
                          # to 100 ms, so it runs as a child.
RECENT_REQUESTS = 20      # requests the dashboard lists
RECENT_FILES = 24         # slot files the dashboard lists. A turn
                          # boundary can write three in one status push.
FLOW_LOG = 150            # stage transitions the flow dashboard replays
MOUNT_POLL = 30.0         # seconds between disk usage checks
CHARS_PER_TOK = 4.0
REPLY_TOKENS  = 1024   # room to reserve for the reply
MAX_BODY = 256 * 1024 * 1024   # largest request body read into memory.
                       # A turn at ctx 150000 is a few megabytes.

# A typed question: one generated token, and the probabilities behind it.
# The path and the field names are TypeSafe's Jev, so a client written for
# that API reaches this router by changing the base URL.
SYSTEMONE = "/v1/systemone"
# The backend has never heard of SYSTEMONE. Every post for one goes here.
SYSTEMONE_UP = "/v1/chat/completions"
# One token an option. Verified against the production model: every one of
# these is a single token, with and without a leading space.
SYSTEMONE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Endpoints that use a slot.
INFERENCE = {
    "/completion", "/completions", "/v1/completions",
    "/chat/completions", "/v1/chat/completions",
    "/infill", "/v1/messages", "/responses", "/v1/responses",
    "/embedding", "/embeddings", "/v1/embeddings",
    SYSTEMONE,
}

# An allowlist. The backends run with --agent, which is shell and file
# access with no key. A path not named here must never be reachable from
# the public port. PASS_THROUGH adds to it, comma separated.
PASSED = {
    "/health", "/props", "/slots", "/models", "/v1/models",
    "/tokenize", "/detokenize", "/apply-template",
    "/v1/messages/count_tokens", "/v1/messages/apply-template",
}
PASSED |= {p.strip() for p in os.environ.get("PASS_THROUGH", "").split(",")
           if p.strip()}

DROP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "te", "trailers",
                "transfer-encoding", "upgrade", "content-length", "host"}


# Printed under "vision hparams" at load. Pool.vision() reads the real value.
VISION = {"patch_size": 16, "n_merge": 2,
          "image_min_pixels": 8192, "image_max_pixels": 4194304}
HEADER_B64 = 98304                        # base64 to decode looking for a size


def image_size(head):
    """Width and height from the front of an image file, or None."""
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", head[6:10])
    if head[:2] == b"BM" and len(head) >= 26:
        # biHeight is signed. Negative means top-down rows. Read unsigned, a
        # 240x120 top-down bitmap was charged 4096 tokens instead of 32.
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

    calc_size_preserved_ratio then clip_n_output_tokens (mtmd-image.cpp):
    one token per aligned square after the resize. A 240x120 png measured
    32 tokens. An unreadable header is charged the most an image can cost."""
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
    near = lambda x: math.floor(x / align + 0.5) * align    # c++ rounding
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
        # openai: {"image_url": {"url": "data:image/png;base64,..."}}
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

    Base64 is hundreds of times longer than what the vision encoder charges.
    One walk for all three: it parses megabytes and decodes every picture."""
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
    """Identify a conversation by its opening: the system messages and the
    first user message. Anything later grows every turn."""
    try:
        req = json.loads(body)
    except Exception:
        return None

    messages = req.get("messages")
    if isinstance(messages, list):
        opening = []
        for message in messages:
            # A non-dict here raised before any status line went out.
            if not isinstance(message, dict):
                break
            role = message.get("role")
            if role == "assistant":
                break                      # the reply
            text = message.get("content")
            if isinstance(text, list):     # multimodal message
                text = "".join(part.get("text", "") for part in text
                               if isinstance(part, dict))
            opening.append(f"{role}:{text}")
            if role == "user":
                break                      # the first user message
        start = "\n".join(opening)
    elif isinstance(req.get("prompt"), str):
        start = req["prompt"]
    else:
        return None
    if not start:
        return None
    return hashlib.sha256(start.encode("utf-8", "replace")).hexdigest()


WEB = Path(__file__).with_name("web")          # the dashboard, a static app
# Everything a run writes. RUN comes from bin/common.sh.
RUN_DIR  = Path(os.environ.get("RUN")
                or Path(__file__).resolve().parent.parent / "run")
SLOT_DIR = RUN_DIR / "slots"
# Openings go on the faster disk where there are two, under BLOCK_BUDGET.
# Conversation copies stay under RUN_DIR, under PARK_BUDGET.
BLOCK_DIR = Path(os.environ.get("BLOCK_DIR") or RUN_DIR / "blocks")

# Debugging only, on when CAPTURE names a directory.
CAPTURE_DIR = Path(os.environ["CAPTURE"]) if os.environ.get("CAPTURE") else None
CAPTURE_KEEP = 24

# One JSON line per cache decision, in a dated file. CACHE_LOG=0 turns it
# off.
CACHE_LOG_ON = os.environ.get("CACHE_LOG", "1") == "1"
CACHE_LOG_DIR = Path(os.environ.get("CACHE_LOG_DIR") or RUN_DIR)

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",  ".json": "application/json",
        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".map": "application/json"}



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


RATE_FLOOR = 1.0     # seconds. Under this a count is not a rate.


def per_second(tokens, seconds):
    """A rate, or zero when there is not enough time to divide by."""
    return round(tokens / seconds, 1) if seconds and seconds >= RATE_FLOOR else 0


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


class EventLog:
    """Append cache events to a dated JSONL file. Telemetry, not data: a
    thread writes, and a full queue drops the newest event."""

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
                # One bad row must not end the thread.
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


# Tests replace this with an EventLog on a temporary directory.
EVENTS = EventLog()


# Client configs the dashboard offers, built for the address the reader used.
CONFIG_FILES = {"opencode": "opencode.json", "claude": "settings.json"}

# Names the machine in an OpenCode config. PROVIDER overrides the hostname.
PROVIDER = os.environ.get("PROVIDER") or socket.gethostname().split(".")[0] or "llama"


HOST_RE = re.compile(r"^(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$")


def host_only(host):
    """A Host header that is only a host and a port, or None. It goes into
    a client config the reader keeps for months, and anyone can set it."""
    host = (host or "").strip()
    return host if HOST_RE.match(host) else None


def client_config(kind, host, model, n_ctx):
    """Build a client config for this router, or return None.

    The timeouts are measured: cpu0_0 read 114,354 tokens at 17.2 to 20.6
    tokens a second with the other instances busy, so the divisor is 15."""
    base = f"http://{host}"
    patience_ms = max(3600000, n_ctx // 15 * 1000)
    if kind == "opencode":
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": f"{PROVIDER}/{model}",
            # Title generation. Unset, it names a model this backend lacks.
            "small_model": f"{PROVIDER}/{model}",
            # The model is private to this machine.
            "share": "disabled",
            "provider": {
                PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Qwen3.8 Flash Next ({PROVIDER})",
                    "options": {
                        "baseURL": f"{base}/v1",
                        # Names the session in every request.
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
        # The client insists on a key. The backend ignores it.
        return {
            "env": {
                "ANTHROPIC_BASE_URL": base,
                "ANTHROPIC_AUTH_TOKEN": "not-used-but-some-clients-require-one",
                "ANTHROPIC_MODEL": model,
                # Gateway discovery keeps only ids that contain "claude" or
                # "anthropic".
                "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen3.8 Flash Next (MTP)",
                # Background work uses the haiku slot.
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_SMALL_FAST_MODEL": model,
                "API_TIMEOUT_MS": str(patience_ms),
                # Both watchdogs give up after five minutes of quiet.
                "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(patience_ms),
                "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": str(patience_ms),
                "API_FORCE_IDLE_TIMEOUT": "0",
                # Nothing here needs the internet.
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                # The attribution block carries a per-conversation
                # fingerprint. Without it two sessions share the prompt.
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                # Pre-release body fields draw a 400 from the backend.
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                # Claude Code assumes a 200k window for an unknown model.
                # The real size makes auto-compact run at the right point.
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(n_ctx),
            }
        }
    return None


# Opening file-name prefixes. A conversation key must not start with one.
SHELF_MARKS = ("base-", "deep-")


def copy_is_current(record):
    """True when the copy on disk is of the turn the conversation last ran.
    An older copy is a prefix, not a replacement for a newer one."""
    return bool(record.get("parked")) and record.get("parked_turn") == record.get("turns")


def last_used(record):
    """When this conversation last ran. PARK_BUDGET keeps the copies used
    most recently, so what goes is what nobody has come back to.

    Size is deliberately not in it. Measured over 131 real copies at a
    64 GiB budget: ranking a copy by the tokens it holds against its bytes
    kept 19 of them but only 7 of the 16 conversations in use that week,
    because the big copies of finished work outranked the small ones
    somebody was still typing into. Ranking by when each was last used kept
    18, and all 16."""
    return record.get("last") or 0


def worth_keeping(record):
    """Whether a copy of this conversation earns a place on disk.

    Only about storing one. A copy written to carry a turn to another
    instance is transport, and travels whatever it holds; so is the one that
    salvages a read the client gave up on, where the alternative is reading
    an hour of prompt again from nothing."""
    if not PARK_MIN_TOKENS:
        return True
    return (record.get("tokens") or 0) - REPLY_TOKENS >= PARK_MIN_TOKENS


def file_safe(key):
    """A conversation key that also works as a file name.

    llama.cpp's fs_validate_filename (common/common.cpp) refuses a colon, a
    path separator, a control character, ".." and a name over 255
    characters, with a 400 on every save and restore."""
    keep = "-._"
    safe = "".join(c if c.isalnum() and c.isascii() or c in keep else "-"
                   for c in key).strip("-. ") or "conversation"
    # The tail is kept: keys are prefixed, and ".park" goes on the end.
    while ".." in safe:
        safe = safe.replace("..", ".")
    safe = safe[-200:].strip("-. ") or "conversation"
    return "c-" + safe if safe.startswith(SHELF_MARKS) else safe


def short_key(conv):
    """A conversation key short enough to read. A subagent's key ends in
    its agent id, so both ends show."""
    conv = conv or ""
    return conv[:8] if len(conv) <= 36 else f"{conv[:8]}/{conv[-6:]}"


def mark_shelf(mark):
    """The shelf of a "base-" / "deep-" mark or a want record. Marks keep
    their dash for file names."""
    mark = mark.get("mark") if isinstance(mark, dict) else mark
    return (mark or "deep-").rstrip("-")


def session_key(headers):
    """Name the conversation from Claude Code's session headers, or None. A
    subagent runs its own prompt, so it is a separate conversation."""
    lower = {str(name).lower(): value for name, value in dict(headers).items()}
    session = (lower.get("x-claude-code-session-id") or "").strip()
    if not session:
        return None
    agent = (lower.get("x-claude-code-agent-id") or "").strip()
    # The key names a slot file. A colon is not allowed in one.
    return file_safe(f"{session}-{agent}") if agent else file_safe(session)


def client_kind(headers):
    """Which client sent this, by its user agent, or None."""
    agent = (dict(headers).get("User-Agent") or "").lower()
    if "claude" in agent:
        return "claude-code"
    if "opencode" in agent:
        return "opencode"
    return agent.split("/")[0][:24] or None


# A keep-alive must be in the client's protocol. A comment keeps an OpenAI
# stream alive. An anthropic parser reports a stream with only comments as
# ended before any data. That protocol has a ping event.
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
    """The name and the json of one SSE event, or (None, None)."""
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
    """The event a stream of this protocol must begin with, or nothing.

    An anthropic stream that has only pinged has begun no message, and the
    client reports a 502. The id is invented. AnthropicSplice moves the real
    usage onto the closing message_delta."""
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

    The backend's message_start must come out: two in one stream break every
    parser. Its prompt token count moves onto the closing message_delta."""

    def __init__(self):
        self.rest = b""
        self.usage = None
        # What the client ends up seeing.
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
            # The backend's own generation figures win.
            data["usage"] = merged = dict(self.usage, **(data.get("usage") or {}))
            self.reported.update(merged)
            self.usage = {}
            return sse_event(name, data)
        return raw


def wants_ping(content_type, content_length):
    """True when extra bytes can be inserted into this reply safely: only a
    streamed event stream."""
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
    """A copy of the body with stream_options.include_usage set, or None. A
    streamed openai reply carries no usage unless asked. The answer is a
    final chunk with no choices (probe-usage.py)."""
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
    """Read the usage figures out of an openai stream as they pass. The
    chunk carrying them has no choices, so it can be removed."""

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
    """Name the conversation from prompt_cache_key, or None. OpenCode sends
    it when setCacheKey is on."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    key = fields.get("prompt_cache_key")
    if not isinstance(key, str) or not key.strip():
        return None
    return file_safe(key.strip())     # the client chose it


def text_of(value):
    """The words in a system prompt, whether it is a string or a list of
    parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value
                       if isinstance(part, dict))
    return ""


def message_shape(message):
    """Everything about one message that a later request must match. The
    backend renders the whole message, not only its words. Sorted keys and
    compact separators, so the same message gives the same bytes."""
    try:
        return json.dumps(without_ignored(message),
                          sort_keys=True, separators=(",", ":"),
                          default=str).encode()
    except (TypeError, ValueError):
        return text_of(message.get("content")).encode("utf-8", "replace")


# Per-request fields that do not change the rendered prompt.
IGNORED_KEYS = frozenset(("cache_control",))


def without_ignored(value):
    """The same body with IGNORED_KEYS dropped, however deep they sit.
    `cache_control` sits on a content block, not on the message."""
    if isinstance(value, dict):
        return {k: without_ignored(v) for k, v in value.items()
                if k not in IGNORED_KEYS}
    if isinstance(value, list):
        return [without_ignored(v) for v in value]
    return value


def closes(message):
    """True when a template can end a prompt after this message. It refuses
    to end after an assistant tool call: "Cannot continue an assistant
    message that contains tool calls". The tool result that answers it is
    the next cut."""
    if message.get("role") != "assistant":
        return True
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    return not (isinstance(content, list)
                and any(isinstance(part, dict) and part.get("type") == "tool_use"
                        for part in content))


def prompt_cuts(body, least=PREFIX_MIN_CHARS):
    """Every point in this request that another request could share. Each
    cut is at a message boundary, hashed onto the cut before it. Returns the
    cuts deepest last, the messages, the system prompt when the request
    keeps it apart, and the tools it declares."""
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
    # The template renders the tools inside the system block. For Claude
    # Code they are most of it: 25 tools and 56,371 characters against 6,111
    # of system prompt.
    written = json.dumps(tools, separators=(",", ":")) if tools else ""
    if system or written:
        # "replace": json allows a lone surrogate, str.encode refuses one.
        running.update(b"system\x00" + system.encode("utf-8", "replace"))
        running.update(b"tools\x00" + written.encode("utf-8", "replace"))
        size += len(system) + len(written)
        # Tools alone are not a prompt: the template refuses an empty one.
        if system and size >= SYSTEM_MIN_CHARS:
            cuts.append((-1, running.hexdigest()[:16]))   # before any message
    # An openai body carries its system prompt as its first message.
    lead = leading_system(messages)
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            break
        # The whole message, not only its words: two agentic conversations
        # hashed by text_of got the same cut names at every depth.
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
    """Describe a request's shape without keeping its text."""
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
    """Turn a late system message into a user message, in place.

    The template refuses a system message that is not at the front. Claude
    Code ends every turn with a token counter as a system message. Moved to
    the front it would end the shared prefix a few thousand tokens in."""
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


class Refused(Exception):
    """A typed question this router will not guess at. The text reaches the
    client as a 400, so it says what to send instead."""


SYSTEMONE_RUBRIC = ("Answer the question about the text above with one letter.\n"
                    "The question gives a letter for every answer it takes.\n"
                    "Write that letter and nothing else.")


NOUL_YES = {"yes", "true", "1"}
NOUL_NO = {"no", "false", "0"}


def noul_criteria(criteria):
    """What yes and no mean for one noul question, or None.

    A noul answers yes or no whatever it is asked, so its criteria do not name
    the answers: they say what the two stand for. Jev writes them as
    `{"true": ..., "false": ...}`, which is what a client in the wild sends."""
    if not isinstance(criteria, dict) or len(criteria) != 2:
        return None
    said = {str(name).strip().lower(): means for name, means in criteria.items()}
    yes = next((said[name] for name in said if name in NOUL_YES), None)
    no = next((said[name] for name in said if name in NOUL_NO), None)
    if yes is None or no is None:
        yes, no = list(criteria.values())      # two of them, in the order given
    return {"yes": yes, "no": no}


def systemone_options(kind, criteria):
    """The answers one question takes, in the order they are lettered."""
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise Refused("a choice question needs criteria: an object of "
                          "answer name to what that answer means")
        return [str(key) for key in criteria]
    if kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise Refused("a score question needs criteria: a list of levels, "
                          "lowest first")
        return [str(level) for level in criteria]
    raise Refused(f"no question type called {kind!r}. The types are "
                  f"choice, score and noul")


def systemone_plan(raw):
    """Read a typed body into the questions to ask, one at a time."""
    try:
        fields = json.loads(raw)
    except Exception:
        raise Refused("the body is not json")
    if not isinstance(fields, dict):
        raise Refused("the body is not a json object")
    if fields.get("stream"):
        raise Refused(f"{SYSTEMONE} does not stream. One token has nothing to "
                      f"stream, so the answer arrives in one piece")
    state = fields.get("state")
    if state is None:
        state = ""
    elif not isinstance(state, str):
        # State is what the asking program holds, and a client in the wild
        # sends an object: an email, a request, a row. Render it once here, so
        # the prompt, the cuts and the conversation's name all see one text.
        state = json.dumps(state, indent=2, ensure_ascii=False)
    asked = fields.get("questions")
    if not isinstance(asked, dict) or not asked:
        raise Refused("questions is an object of one or more named questions")

    plan = []
    for name, question in asked.items():
        if not isinstance(question, dict):
            raise Refused(f"question {name!r} is not an object")
        kind = str(question.get("type") or "noul")
        options = systemone_options(kind, question.get("criteria"))
        if len(options) < 2:
            raise Refused(f"question {name!r} needs at least two answers to "
                          f"choose between, and has {len(options)}")
        if len(options) > len(SYSTEMONE_LETTERS):
            raise Refused(f"question {name!r} takes {len(options)} answers. "
                          f"One token carries {len(SYSTEMONE_LETTERS)} at most")
        said = question.get("criteria")
        plan.append({"name": name, "type": kind, "options": options,
                     "letters": SYSTEMONE_LETTERS[:len(options)],
                     "instructions": str(question.get("instructions") or ""),
                     "criteria": noul_criteria(said) if kind == "noul" else said})
    key = fields.get("prompt_cache_key")
    return {"model": fields.get("model") or "systemone",
            "key": key if isinstance(key, str) and key.strip() else None,
            "state": state, "questions": plan}


def systemone_says(question):
    """The message that asks one question and letters its answers."""
    criteria = question["criteria"]
    head = question["instructions"].strip()
    lines = [head, ""] if head else []
    for letter, option in zip(question["letters"], question["options"]):
        means = criteria.get(option) if isinstance(criteria, dict) else None
        lines.append(f"{letter} = {means or option}")
    lines += ["", "Answer with one letter.", "Answer:"]
    return "\n".join(lines)


def systemone_body(plan, question):
    """The chat body that asks one question about this plan's state."""
    messages = [{"role": "system", "content": SYSTEMONE_RUBRIC},
                {"role": "user", "content": plan["state"]},
                {"role": "user", "content": systemone_says(question)}]
    body = {"model": plan["model"], "messages": messages, "stream": False}
    if plan["key"]:
        body["prompt_cache_key"] = plan["key"]
    letters = question["letters"]
    body.update({
        # One token. tests/live belief 6: at one token the slot still holds
        # exactly the prompt, so a question leaves nothing behind it.
        "max_tokens": 1,
        "temperature": 0,        # -1 means greedy upstream, but field_num
                                 # clamps a soft limit, so -1 arrives as 0
        "logprobs": True,
        "top_logprobs": 2 * len(letters) + 8,
        # False, or the grammar and greedy sampling have already collapsed
        # the distribution and every answer comes back at 1.0.
        "post_sampling_probs": False,
        # patches/grammar-probs.patch: report every token the grammar allows,
        # and what they held of the distribution before it. Stock llama.cpp
        # ignores a field it does not know, and the two lines above still
        # answer - less exactly, because they see only the top of the list.
        "grammar_probs": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "grammar": "root ::= " + " | ".join(f'"{x}"' for x in letters)})
    return body


def systemone_read(reply, question):
    """One typed answer, from the one token the backend wrote.

    The probabilities are the raw softmax over the whole vocabulary: llama.cpp
    reports them before the grammar, so they are absolute and they cover words
    no answer letter stands for. Weight lands on " A" as well as on "A", and
    both mean the same answer. What is left after the letters are kept is
    scaled back up to one, and `mass` records how much was thrown away."""
    choices = (reply or {}).get("choices") or []
    first = choices[0] if choices else {}
    content = (first.get("logprobs") or {}).get("content") or []
    head = content[0] if content else {}
    letters = question["letters"]
    mass = {letter: 0.0 for letter in letters}
    for item in head.get("top_logprobs") or []:
        letter = (item.get("token") or "").strip()
        if letter in mass and isinstance(item.get("logprob"), (int, float)):
            mass[letter] += math.exp(item["logprob"])

    total = sum(mass.values())
    # A patched backend has already scaled those to sum to one over the
    # answers, and reports what they held before that scaling. It counts every
    # token the grammar allowed; the sum above counts only the ones that fit
    # in top_logprobs, and undercounts whenever an answer fell off the end.
    reported = head.get("grammar_mass")
    held = reported if isinstance(reported, (int, float)) and reported >= 0 else total
    if total <= 0:
        # The grammar let one letter through, but the model's own next token
        # was going to be something else entirely, so no letter was reported.
        # What the backend wrote is still the answer. `mass` stays at 0, which
        # is the reader's warning that the rest of this is one letter's word.
        wrote = ((first.get("message") or {}).get("content") or "").strip()
        if wrote not in mass:
            raise RuntimeError(f"no probabilities came back for question "
                               f"{question['name']!r}")
        mass[wrote] = 1.0

    probs = {option: weight / (total or 1.0) for option, weight
             in zip(question["options"], mass.values())}
    best = max(probs, key=lambda option: probs[option])
    answer = {"type": question["type"], "probabilities": probs,
              "confidence": probs[best], "mass": held}
    if question["type"] == "noul":
        answer["noul"] = probs[question["options"][0]]
    elif question["type"] == "score":
        # The expected level, not the likeliest one. A score of 1.6 says the
        # answer sits between the second and third level, which is what the
        # distribution says and what one letter cannot.
        answer["score"] = sum(rank * probs[option] for rank, option
                              in enumerate(question["options"]))
    else:
        answer["choice"] = best
    return answer


def systemone_answers(be, slot, plan, post):
    """Ask every question against the slot that already holds the state.

    `slot` is None when the turn was carried to another instance: the slot it
    landed in is that instance's to choose, and llama.cpp finds the state by
    prefix, exactly as it does for every turn the router forwards."""
    answers, wrote, steps = {}, 0, []
    for question in plan["questions"]:
        body = systemone_body(plan, question)
        if slot is not None:
            body["id_slot"] = slot
        reply = post(be["url"], SYSTEMONE_UP, body, READ_TIMEOUT)
        answers[question["name"]] = systemone_read(reply, question)
        usage = (reply or {}).get("usage") or {}
        wrote += int(usage.get("completion_tokens") or 0)
        # What this question cost on top of the state, by the backend's own
        # count. The first question extends what the read pass left; the ones
        # after it roll back to the end of the state and read their own words.
        timing = (reply or {}).get("timings") or {}
        steps.append({"question": question["name"],
                      "read": timing.get("prompt_n"),
                      "reused": timing.get("cache_n")})
    return answers, wrote, steps


PLACE_RE = re.compile(r"(\d+)_(\d+)$")


def by_place(name):
    """Sort key from a backend name: the socket, then the instance on it.
    gpu0_0 sorts with cpu0_0. A name without a place sorts last."""
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
    """Where to ask this backend what a body renders to. The anthropic route
    converts the body first, so a tool call renders."""
    return ("/v1/messages/apply-template" if (path or "").startswith("/v1/messages")
            else "/apply-template")


def read_only(body, slot=None):
    """The same request, asking for zero tokens. The read happens on a
    prefiller. The router chooses where to generate after the read."""
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
    # Zero, not one. A generated token lands in the slot, and a restored slot
    # has no checkpoint to rewind to, so the next request re-reads everything.
    if "n_predict" in fields:
        fields["n_predict"] = 0
    else:
        fields["max_tokens"] = 0
    # llama.cpp copies max_output_tokens over max_tokens unconditionally
    # (server_chat_convert_responses_to_chatcmpl).
    if "max_output_tokens" in fields:
        fields["max_output_tokens"] = 0
    if slot is not None:
        fields["id_slot"] = slot      # say which slot
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
        # A backend puts the reason in the body.
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
    """POST to a backend. Stop when nobody waits for the answer.

    Closing the connection cancels the task in llama.cpp and frees the slot.
    Use shutdown(), not close(): the reading thread holds the socket open
    through its file object, so close() alone never reaches the backend.
    Raises Gone when the client has left."""
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
                pass                   # already gone
            conn.close()               # the backend cancels the task
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
    """Write one request body down, for comparing two turns offline. Kept
    per conversation, so a busy client cannot crowd out a quiet one."""
    if CAPTURE_DIR is None:
        return
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        tag = short_key(conv).replace("/", "_")
        # Nanoseconds and fixed width: unique names that sort by time.
        name = f"{time.time_ns()}-{tag}.json"
        (CAPTURE_DIR / name).write_bytes(body)
        old = sorted(CAPTURE_DIR.glob(f"*-{tag}.json"))[:-CAPTURE_KEEP]
        for spent in old:
            spent.unlink(missing_ok=True)
    except OSError as err:
        print(f"[router] could not write the capture: {err}", flush=True)


def how_started(warm, recalled, loaded):
    """Name what a request extended instead of reading. `warm`: its own
    cache was in a slot. `recalled`: its own copy came back from disk.
    `loaded`: a saved opening was put in the slot."""
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
            # One budget over both shelves.
            "openings": {"count": len(openings),
                         "bytes": sum(opening_bytes.values()),
                         "budget": BLOCK_BUDGET},
            "bases": {"count": kinds.count("base")},
            "deeps": {"count": kinds.count("deep")},
            "wants": {"count": len(wants), "keep": WANT_KEEP}}


class History:
    """Slot-seconds by phase, per history bucket. Between two polls a slot
    stays in the phase the earlier poll reported. A bucket edge inside the
    gap splits it."""

    def __init__(self, keep=HISTORY_KEEP, step=HISTORY_STEP):
        self.keep, self.step = keep, step
        self.at = None
        self.since = None
        self.rows = {}        # backend name -> {"done": [...], "cur": {...}}
        self.state = {}       # backend name -> [(phase, tg_rate)] last seen
        # Machine load shares the buckets.
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
                    # No rate yet is not a stall.
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
        """Add one sample of each gauge to the current bucket. Call after
        push."""
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
# The weights are mmap'd and live in the page cache. When free memory gets
# tight the kernel drops them, and prefill falls from 37 tokens a second to
# single digits. The dashboard shows page cache against the model size.

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
        idle = values[3] + (values[4] if len(values) > 4 else 0)   # +iowait
        out[int(fields[0][3:])] = (sum(values) - idle, sum(values))
    return out


def node_busy(before, after, cpus):
    """Per cent of a node's cpu time spent busy between two samples, or
    None."""
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
    """Bytes one backend needs in the page cache: mapped less lazy, from its
    startup log."""
    cache = be.get("cache") or {}
    mapped, lazy = cache.get("mapped_mib") or 0.0, cache.get("lazy_mib") or 0.0
    return int(max(0.0, mapped - lazy) * 1024 * 1024)


class Machine:
    """CPU, memory and GPU load, sampled beside the backend poll. nvidia-smi
    costs 50 to 100 ms, so it runs as a child and is collected on a later
    pass. A missing command turns the gpu row off."""

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
        backend on the node needs: they map the same files."""
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
    """Point the slot directory at a block on the faster disk. A backend
    takes a bare filename under --slot-save-path and rejects a directory in
    it, so a link is the only way to put one file elsewhere."""
    link = SLOT_DIR / name
    try:
        BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        print(f"[router] {BLOCK_DIR} is not usable, keeping blocks with the "
              f"rest: {err}", flush=True)
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    # Absolute: the kernel resolves a relative target against the link's dir.
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
    """Delete a slot file, and whatever it points at. Never raises: it runs
    before the turn ticket goes back, and claim_turn has no deadline, so a
    throw here would hold the conversation for the life of the process."""
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
    """The rows the last run wrote, or [] when there are none to trust."""
    try:
        kept = json.loads(path.read_bytes())
    except Exception:
        return []                 # no file, or one we cannot trust
    return kept if isinstance(kept, list) else []


def write_rows(path, rows):
    """Write beside the file and rename over it, so half of one can never
    be read back. A half file reads as empty, and adopt then deletes every
    file it vouched for."""
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
    """Sort the files the last run left behind, oldest first. A saved
    opening is named after its contents. A conversation's copy is good only
    if the pin file vouches for it. The pin file is asked first, because a
    client can make a key look like an opening."""
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
            spent.append(name)      # unvouched copy, or a deep opening
                                    # with DEEP_OPENINGS off
    spent += trim_openings(openings, bytes_)
    return openings, bytes_, parked, spent


def trim_openings(openings, bytes_, keep=()):
    """Drop openings until they fit BLOCK_BUDGET, least useful first: deeper
    cuts before system prompts, then least recently used. One is always
    kept. `keep` names openings being built, which are not on disk yet.
    Returns the file names dropped."""
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
    Each request's own thread notes its transitions. The log outlives the
    live row, so the animation can replay a stage it never saw. Held under
    Pool.cv."""

    def __init__(self):
        self.live = {}                        # conv -> current stage and where
        self.log = deque(maxlen=FLOW_LOG)     # newest first, for the animation

    def note(self, conv, stage, backend=None, slot=None, kind=None):
        """Move a turn to its next stage. Held under the lock.

        `kind` is a word for the work, or None for an ordinary turn. This
        class does not read it. The turn keeps the word once one stage says
        it, so a stage noted from inside the pool does not drop it."""
        if not conv:
            return
        now = time.time()
        row = self.live.get(conv)
        kind = kind or (row or {}).get("kind")
        if stage == "done":
            if row is None:
                return
            self.log.appendleft(dict(row, stage="done", at=now,
                                     since=None, changed=None))
            del self.live[conv]
            return
        if (row and row["stage"] == stage and row["kind"] == kind
                and row["backend"] == backend and row["slot"] == slot):
            return
        entry = {"conv": short_key(conv), "stage": stage, "kind": kind,
                 "backend": backend, "slot": slot}
        self.live[conv] = dict(entry, since=row["since"] if row else now,
                               changed=now)
        self.log.appendleft(dict(entry, at=now))

    def report(self):
        """What the dashboard animates. Held under the lock."""
        return {"live": list(self.live.values()), "log": list(self.log)}


class Pool:
    """Track free slots. Keep each conversation on one backend."""

    # Not running totals, so since_reset leaves them alone. The first five
    # are llama-server's own `gauges` (server-task.cpp). n_tokens_max sits
    # in its `counters` list but is built with std::max.
    GAUGES = ("prompt_tokens_seconds", "predicted_tokens_seconds",
              "requests_processing", "requests_deferred",
              "n_busy_slots_per_decode", "n_tokens_max")

    def __init__(self, backends, watch=True):
        self.cv = threading.Condition()
        self.backends = [dict(b, slots=1, n_ctx=0, busy=0, up=False, served=0, model="",
                              stats={}, slots_detail=[], slot_prev={}, misses=0,
                              cache={}, idle_runs={}, draining=False)
                         for b in backends]
        # Each backend reports prompt cache evictions only in its own log.
        self.cache_watch = {be["name"]: CacheWatch(
            RUN_DIR / f"{be['name']}.log",
            sink=lambda kind, value, name=be["name"]: EVENTS.write(
                "backend", backend=name, kind=kind, amount=value))
            for be in self.backends}
        self.pins = OrderedDict()
        # conversation -> the wait ticket of the turn serving it now. A
        # conversation is one pin, one slot and one copy on disk. Two turns at
        # once corrupt all three: seen once as one conversation recalled onto
        # two backends ninety seconds apart.
        self.turns = {}
        # Opening key -> its file, least recently used first.
        self.openings = OrderedDict()
        self.opening_bytes = {}        # the same keys, and what each takes
        # (backend, slot) -> the cuts that slot holds.
        self.holds = {}
        # (backend, slot) -> deepest message index held. -1: system prompt.
        self.holds_depth = {}
        # conversation -> what warm_prefix found and did, for the dashboard.
        self.choices = OrderedDict()
        # conversation -> (parent, depth), so a fork is logged once a depth.
        self.forked = OrderedDict()
        # Counters at the last reset, per backend.
        self.rates_from = {}
        self.rates_since = None
        # Opening key -> what it takes to read one, newest last.
        self.wants = OrderedDict()
        # Openings being read now. A session that needs one waits for it.
        self.building = {}
        self.waiting = 0          # requests with no free slot yet
        self.waiters = {}         # ticket -> the waiting request
        # Turns read and parked, waiting for a generator slot.
        self.to_generate = 0
        self.wait_seq = 0
        self.flow = Flow()
        # The last few slot files written or read.
        self.recent = deque(maxlen=RECENT_FILES)
        # The last few requests, with what each one started from.
        self.recent_requests = deque(maxlen=RECENT_REQUESTS)
        self.history = History()
        self.machine = Machine()
        self.loads = {}           # opening key -> times a request loaded it
        self.mounts = []          # disk usage, refreshed every MOUNT_POLL
        self.mounts_at = 0.0
        # The park worker starts on the first park, so tests start no thread.
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
                    found.unlink(missing_ok=True)   # a dangling link
        # The order and load count each opening earned last run. Without it
        # the only order is the file mtime, which link_block sets once. An
        # opening the file does not name sorts at the back, by mtime.
        remembered = [row for row in read_rows(openings_file())
                      if isinstance(row, dict) and row.get("key")]
        was = {row["key"]: rank for rank, row in enumerate(remembered)}
        self.loads.update({row["key"]: row.get("loads") or 0
                           for row in remembered})
        names.sort(key=lambda n: was.get(opening_key(n), len(was)))
        kept = read_rows(pins_file())
        # Both fields: a row without a conv raised KeyError at startup.
        by_file = {row["file"]: row for row in kept
                   if isinstance(row, dict) and row.get("file") and row.get("conv")}
        openings, sizes, parked, spent = adopt_files(names, set(by_file))
        with self.cv:
            self.openings, self.opening_bytes = openings, sizes
            for name in parked:
                row = by_file[name]
                self.pins[row["conv"]] = {
                    # Not a live name, so recall restores the copy first.
                    "backend": "(before the restart)",
                    "slot": None, "tokens": row.get("tokens", 0),
                    # When it last ran, not now: the budget sweep drops what
                    # has gone longest without a turn, and `now` for every
                    # copy hides exactly that.
                    "last": row.get("last") or file_mtime(name),
                    "inflight": False,
                    "turns": row.get("turns", 1), "parked": name,
                    "bytes": row.get("bytes", 0),
                    # Without it every copy reads as age zero.
                    "parked_at": row.get("parked_at") or file_mtime(name)}
            # The budget is spent here too, not only where a copy is written.
            # Lowering it otherwise did nothing until the next turn parked,
            # and on a quiet router that is never: 251 GiB of copies sat
            # under a 64 GiB budget with one conversation in a slot.
            spent += self._trim_copies()
            # After the trim, not `parked`, which counts what the last run
            # left rather than what this one is keeping: the line read
            # "131 conversation(s)" on a start that kept 19.
            keeping = sum(1 for p in self.pins.values() if p.get("parked"))
        for name in spent:
            remove(name)
        self.save_openings()      # trimmed, so write it
        if spent:
            # Or the map still vouches for files this start has deleted,
            # until whenever the next turn happens to park.
            self.save_pins()
        if openings or keeping or spent:
            kinds = [shelf_of(name) for name in openings.values()]
            print(f"[router] kept {kinds.count('base')} system prompt(s), "
                  f"{kinds.count('deep')} deeper opening(s) and {keeping} "
                  f"conversation(s), dropped {len(spent)} stale file(s)",
                  flush=True)

    def note_file(self, did, name, be, slot, size=0):
        """Record one slot file written or read. Held under the lock."""
        self.recent.appendleft({"did": did, "name": name[:8], "at": time.time(),
                                "backend": be["name"], "slot": slot,
                                "bytes": size})

    def note_stage(self, conv, stage, backend=None, slot=None, kind=None):
        """Move a turn along its stages, for the flow dashboard."""
        with self.cv:
            self.flow.note(conv, stage, backend, slot, kind)

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

    def claim_turn(self, conv, ticket, wanted=None, kind=None):
        """Hold this conversation until finish_turn. One turn of it at a time.

        Returns True when the turn holds it. False means the client left and
        nothing is held. No deadline: a re-read costs more than any wait."""
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
            # Noted here, not in begin_wait: a waiting turn must not move the
            # row of the turn ahead, which is keyed by conversation too.
            # `kind` rides along, or a queue of typed questions shows as a
            # queue of unlabelled rows until each one starts reading.
            self.flow.note(conv, "queued", kind=kind)
        if began is not None:
            print(f"[router] {short_key(conv)} waited "
                  f"{time.time() - began:.0f}s for the turn ahead of it",
                  flush=True)
        return True

    def finish_turn(self, conv, ticket):
        """Let the next turn of this conversation start. The ticket must
        match: a turn that gave up waiting never held it."""
        if not conv:
            return
        with self.cv:
            if self.turns.get(conv) == ticket:
                del self.turns[conv]
                self.cv.notify_all()

    def _waiting_detail(self, now):
        """Each waiter, and what it waits for. Held under the lock. A pin is
        named only when acquire would wait for it: a pin to a generator is
        dropped at once."""
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
            # `tokens` includes REPLY_TOKENS, as begin_wait was handed it. The
            # fix is for the ticket to carry prompt and room as two numbers.
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
        """Record one finished request for the dashboard. `tokens` has the
        reply room taken off. The two read counts come from the reply's
        `timings`, None for a turn that never read."""
        with self.cv:
            self.recent_requests.appendleft({
                "conv": short_key(conv), "backend": be["name"], "path": path,
                "took": round(took, 1), "waited": round(waited, 1),
                "started": started, "tokens": tokens,
                "read": read_prompt_n, "reused": read_cache_n,
                "images": images, "image_tokens": image_tokens_,
                "at": time.time()})

    def _read_mounts(self, now):
        """Free space on the disks the slot files land on, refreshed every
        MOUNT_POLL."""
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

        A counter that went backwards means a restart, so it is taken as it
        stands. GAUGES are left alone: subtracting n_tokens_max read 0, and
        n_busy_slots_per_decode went from 2.40 to 0.01, 320-fold off."""
        # `is None`: a backend down at the reset has an empty baseline.
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
        """A backend's startup settings, from its log. Read again when the
        inode changes: restart-backend.sh moves the old log aside, and the
        size alone misses a restart whose new log passes the old offset."""
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
                for _ in range(1500):   # loading prints hundreds of lines
                    head.append(next(handle))
        except (OSError, StopIteration):
            pass
        be["config"] = read_config(head)
        be["vision"] = read_vision(head)
        be["config_at"] = size
        be["config_ino"] = ino
        return be["config"]

    def vision(self):
        """The vision encoder's geometry, from whichever backend printed it.
        Read each time, so a restart onto a different mmproj is picked up."""
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
            # Lifetime. The *_tokens_seconds gauges read zero when idle.
            return per_second(value.get(tokens, 0), value.get(seconds, 0))

        # prompt_tokens_total excludes cached tokens.
        processed = value.get("prompt_tokens_total", 0)
        cached = value.get("prompt_tokens_cached_total", 0)
        drafted = value.get("spec_decode_num_draft_tokens_total", 0)

        # tokens_predicted_seconds_total sums per-request time. Concurrent
        # slots overlap. These rates are per request.
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
            "read_s": round(value.get("prompt_seconds_total", 0), 1),
            "gen_s": round(value.get("tokens_predicted_seconds_total", 0), 1),
            "prompt_tokens": int(processed),
            "cached_tokens": int(cached),
        }
        st = be["stats"]
        st["pp_total"] = round(st["pp_rate"] * busy_per_decode, 1)
        st["tg_total"] = round(st["tg_rate"] * busy_per_decode, 1)

    def _read_slots(self, be, raw=None):
        """Per-slot state, so a 3-slot backend is not a single average. `raw`
        is what /slots answered, for tests."""
        if raw is None:
            try:
                with urllib.request.urlopen(be["url"] + "/slots", timeout=3) as r:
                    raw = json.load(r)
            except Exception:
                return
        # /slots reports counters, not rates.
        now = time.time()
        previous = be.get("slot_prev") or {}
        current, detail = {}, []
        for slot in raw if isinstance(raw, list) else []:
            # A one-element array. Older builds sent a bare object.
            token = slot.get("next_token") or {}
            if isinstance(token, list):
                token = token[0] if token else {}
            cached = slot.get("n_prompt_tokens_cache", 0)
            sid = slot.get("id")
            task = slot.get("id_task")
            decoded = token.get("n_decoded", 0)
            processed = slot.get("n_prompt_tokens_processed", 0)

            # Measured over RATE_WINDOW, not between polls: a slot at 0.03
            # tokens/s does not move in two seconds.
            was = previous.get(sid) or {"task": None, "decoded": 0, "processed": 0,
                                        "done_d": 0.0, "done_p": 0.0, "since": now,
                                        "pp_rate": 0.0, "tg_rate": 0.0,
                                        # False until a window has resolved.
                                        "measured": False}
            # A new task restarts the counters at zero.
            if was["task"] is None:
                grew_d = grew_p = 0        # first sight: take a baseline
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

            # n_prompt_tokens_total is the prompt the task arrived with, from
            # patches/slots-report-the-prompt-size.patch. n_prompt_tokens
            # grows while the prompt is read and with every token generated:
            # a slot 98% served from cache reported "512 / 89,848 read".
            # Without the patch the old arithmetic is the fallback.
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
                # null, not 0.0, until a window has resolved.
                "pp_rate": round(pp_rate, 1) if measured else None,
                "tg_rate": round(tg_rate, 1) if measured else None,
            })
        be["slot_prev"] = current
        be["slots_detail"] = detail
        self._note_idle(be, detail)
        # The sum of the slots, not /metrics' lifetime average.
        stats = be.setdefault("stats", {})
        stats["pp_live"] = round(sum(d["pp_rate"] or 0 for d in detail), 1)
        stats["tg_live"] = round(sum(d["tg_rate"] or 0 for d in detail), 1)

    @staticmethod
    def _note_idle(be, detail):
        """Count the polls in a row each slot has looked idle. One poll
        cannot tell a free slot from one between two turns."""
        was = be.get("idle_runs") or {}
        be["idle_runs"] = {s["id"]: 0 if s["busy"] else was.get(s["id"], 0) + 1
                           for s in detail}

    def _watch(self):
        """Check each backend. Read its slot count, context size and
        counters."""
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
                # Two misses before down: a loaded box can miss a 3 second
                # deadline once, and down re-pins every waiting conversation.
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
                        # A backend that came back has empty slots. Without
                        # this recall refused and the turn read its whole
                        # prompt with a good copy on disk.
                        if came_back:
                            self.forget_slots(be["name"])
                        self.cv.notify_all()
            now = time.time()
            try:
                self.machine.sample(now)
            except Exception as err:          # never stop the poll for load
                print(f"[router] machine sample failed: {err}", flush=True)
            with self.cv:
                self.history.push(self.backends, now)
                self.history.push_load(self.machine.gauges())
            time.sleep(POLL)

    def _read_cache(self, be):
        """Total the prompt cache events this backend has logged."""
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
        """The slot this read should use on this backend. The router chooses:
        a reply names its slot only on some paths. A conversation holding a
        slot here keeps it. Otherwise one no other request was handed: the
        poll is two seconds old, so requests arriving together would all be
        told the same slot."""
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if (record and record["backend"] == be["name"]
                    and record["slot"] is not None):
                # `taken` below is built from `using`. Without this a
                # second request could be handed this warm slot.
                record["using"] = record["slot"]
                return record["slot"]
            taken = {p.get("using") for name, p in self.pins.items()
                     if name != conv and p.get("inflight")
                     and p.get("backend") == be["name"]}
            detail = be.get("slots_detail") or []
            ids = [s["id"] for s in detail] or list(
                range(max(1, be.get("slots", 1))))
            working = {s["id"] for s in detail if s.get("busy")}
            # Free by both accounts first. Then merely not handed out: the
            # poll is the older of the two.
            slot = next((i for i in ids if i not in taken and i not in working),
                        next((i for i in ids if i not in taken), ids[0]))
            if record is not None:
                record["using"] = slot
            return slot

    def _reading_rank(self, be):
        """Order backends for a prompt that must be read somewhere. A socket
        already reading comes last: two reads on one socket roughly halve
        each other. A generating slot competes for nothing a read needs."""
        def reads(backend):
            return sum(1 for slot in (backend.get("slots_detail") or [])
                       if slot.get("phase") == "reading")

        node = be.get("node")
        on_node = sum(reads(other) for other in self.backends
                      if other.get("node") == node)
        # Within a node a quiet instance beats a second slot on a busy one:
        # llama.cpp lets the first reading slot take the whole batch. Then
        # the opposite of pref, which keeps the generating instances free.
        return (on_node, reads(be), -be["pref"], be["busy"])

    def _usable(self, be, tokens):
        """True if this backend is up, has a free slot, and is big enough."""
        return (be["up"] and not be.get("draining")
                and be["busy"] < be["slots"] and tokens <= be["n_ctx"])

    def drain(self, name, post, deadline=DRAIN_DEADLINE):
        """Take a backend out of service so it can be restarted. Requests wait
        in acquire rather than fail. The caches in its slots are copied out."""
        be = next((b for b in self.backends if b["name"] == name), None)
        if be is None:
            return None
        with self.cv:
            be["draining"] = True
            self.cv.notify_all()      # waiters can pick the other backend now

        # Killing running work throws away a read of up to twenty minutes.
        stop = time.time() + deadline
        while True:
            with self.cv:
                quiet = be["busy"] <= 0
                if quiet or time.time() > stop:
                    break
                self.cv.wait(0.2)

        parked = self.park_all(post, only=name) if quiet else 0
        # A save that timed out or was refused leaves a cache only in a slot.
        # The caller must know, or restart-backend.sh kills it anyway.
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
        """The largest prompt any prefiller will read. A generator's ctx does
        not count: a conversation pinned to one spills to a prefiller."""
        return max([be["n_ctx"] for be in self.backends
                    if be["up"] and prefills(be)], default=0)

    def acquire(self, conv, tokens, wanted=None):
        """Take a slot on the backend holding this conversation.

        A busy box is a queue, not a refusal, so the wait has no deadline. It
        ends when a slot frees, when no backend can serve the request, or when
        `wanted` says the client left. A pin holds for PIN_PATIENCE."""
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
                    # A fifth of turns re-read everything: prefillers only.
                    if (not target["up"] or tokens > target["n_ctx"]
                            or not prefills(target)):
                        spill = True          # it can never take this request
                        target = None
                else:
                    spill = spill or pinned is not None   # gone backend

                if not target:
                    free = [b for b in self.backends
                            if prefills(b) and self._usable(b, tokens)]
                    if free:
                        return self._take(min(free, key=self._reading_rank),
                                          conv, tokens)

                # Nothing that could serve this is up. Waiting cannot help.
                served_by = target is not None or any(
                    b["up"] and prefills(b) for b in self.backends)
                if not served_by:
                    return None
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
            # Updated in place. A fresh record dropped `opening` and
            # `parked_at`, which other paths write.
            record = self.pins.get(conv)
            if record is None:
                # Named here because several readers index them directly.
                record = self.pins[conv] = {"parked": None, "bytes": 0}
            record.update(
                backend=be["name"],
                # A slot id only means something on its own backend.
                slot=record.get("slot") if record.get("backend") == be["name"] else None,
                tokens=tokens,
                last=time.time(),
                inflight=True,
                turns=record.get("turns", 0) + 1)
            self.pins.move_to_end(conv)
            while len(self.pins) > MAX_PINS:
                _, dropped = self.pins.popitem(last=False)
                if dropped.get("parked"):
                    drop_file(dropped["parked"])   # its copy is now orphaned
        return be

    def release(self, be, conv=None):
        """Give the backend back. The copy on disk stays: it is behind the
        slot but still a prefix, and the next save overwrites it under the
        same name."""
        with self.cv:
            be["busy"] -= 1
            record = self.pins.get(conv) if conv else None
            if record:
                record["inflight"] = False
                record["last"] = time.time()
            self.cv.notify_all()

    def forget_slots(self, name):
        """Forget which slot on this backend held what. Held under the lock.
        A restarted backend loses every slot. The pins stay: the copies on
        disk are still good. A conversation mid-turn is left alone, because
        its own thread owns that slot."""
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
        """Record which slot served this conversation, for the save."""
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if record:
                record["slot"] = slot

    def _builder(self):
        """Read one wanted opening while a backend is idle, off the request
        path."""
        while True:
            time.sleep(BUILD_POLL)
            try:
                self.build_once(http_post)
            except Exception as err:
                print(f"[router] opening pass failed: {err}", flush=True)

    def _idle_slot(self, be):
        """A slot the builder may read into, or None. Stricter than
        _free_slot: the backend is up, the router's own count leaves room,
        and the poll has found the slot idle IDLE_POLLS times."""
        # A draining backend is about to be stopped.
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
        """Copy every cache on this backend to disk before a request lands. A
        save reads a slot, so it only works while the cache is still in one.
        Each conversation is tried once, or a save that does not stick loops
        forever."""
        tried = set()
        while True:
            with self.cv:
                at_risk = [(conv, p) for conv, p in self.pins.items()
                           if p["backend"] == be["name"]
                           and conv != skip_conv
                           and conv not in tried
                           and not p["inflight"]
                           and p["slot"] is not None
                           and not copy_is_current(p)
                           and worth_keeping(p)]
                if not at_risk:
                    return
                conv, record = min(at_risk, key=lambda item: item[1]["last"])
                record["inflight"] = True      # hold it still while it copies
                slot = record["slot"]
                tried.add(conv)

            self._save_park(conv, be, slot, post, remove)

    def _save_park(self, conv, be, slot, post, remove=drop_file):
        """Write one cache to disk. Mark the pin only if it holds one."""
        name = conv + ".park"
        short = short_key(conv)
        kept = False
        written = 0
        # A short save means the slot holds somebody else. A failed save says
        # nothing about the slot: the cache is still there.
        lost = False
        refused = False
        began = time.time()
        # Which turn this save belongs to. A save can take POST_TIMEOUT, and
        # the conversation's next turn can start inside that window. Clearing
        # `inflight` for the wrong turn un-reserves a slot being read.
        with self.cv:
            record = self.pins.get(conv)
            turn = record.get("turns") if record else None
        try:
            answer = post(be["url"], f"/slots/{slot}?action=save",
                          {"filename": name}) or {}
            # Ask the disk. Without
            # patches/slot-state-carries-checkpoints.patch the backend
            # reports the state without the checkpoint trailer: a 107 MB
            # file came back as 49 MB. Its figure is the fallback for a
            # stub. tests/live/test_llama_beliefs.py asserts the two agree.
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
                # Only if no later turn started while the save ran. A later
                # turn owns `inflight` and `slot` now.
                mine = record.get("turns") == turn
                if mine:
                    record["inflight"] = False
                if kept:
                    record["parked"] = name
                    record["bytes"] = written
                    record["parked_at"] = time.time()
                    # For copy_is_current.
                    record["parked_turn"] = turn
                    self.note_file("parked", conv, be, slot, written)
                elif not refused and mine:
                    record["parked"] = None
                    record["bytes"] = 0
                # A refused call keeps the copy it had: the file is still on
                # disk and still a prefix. Clearing it orphaned the file, off
                # the budget sweep and off MAX_PINS eviction.
                if lost and mine:
                    # The slot holds someone else.
                    record["slot"] = None
            spent = self._trim_copies(keep=conv)
            self.cv.notify_all()
        for gone in spent:
            remove(gone)
        # The map vouches for these files on the next run. Written from the
        # signal handler only, a crash or an OOM kill threw away every copy
        # this run made. A park is rare: 952 in three days.
        if kept or spent:
            self.save_pins()
        EVENTS.write("park", conv=short, backend=be["name"], slot=slot,
                     ok=bool(kept), bytes=written if kept else 0,
                     secs=round(time.time() - began, 2))
        if kept:
            print(f"[router] parked {short} from {be['name']} slot {slot}", flush=True)
        return kept

    def _trim_copies(self, keep=None):
        """Drop copies until they fit PARK_BUDGET, least recently used first.
        Returns the file names to delete. Held under the lock.

        `keep` names one copy that stays whatever its age: the one just
        written. Dropping that only has it written again, which is how
        cpu1_0 wrote the same 9.45 GiB copy 1,456 times in four and a half
        hours.

        Ordered by write time instead, a copy the migration had just
        rewritten looked fresh though nobody had asked for it, and a
        conversation somebody was working in was dropped for a question
        answered days ago."""
        held = sorted((c for c, p in self.pins.items() if p.get("parked")),
                      key=lambda c: last_used(self.pins[c]))
        held.reverse()                     # used most recently first
        if keep in held:
            held.remove(keep)
            held.insert(0, keep)
        spent, total = [], 0
        for name in held:
            record = self.pins[name]
            total += record.get("bytes") or 0
            if total > PARK_BUDGET and name != keep:
                spent.append(record["parked"])
                record["parked"] = None
        return spent

    def recall(self, conv, be, slot, post):
        """Put a parked cache back on the backend about to serve it. Returns
        True when the cache is now on that backend."""
        with self.cv:
            record = self.pins.get(conv)
            if not record or not record.get("parked"):
                return False
            # Already here, and still in a slot. acquire re-pins before this
            # runs, so only the slot says whether the cache survived.
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
        Only an opening the router already has: that is a file read. An
        opening the router lacks is written down for the builder. Returns
        True when an opening was loaded."""
        with self.cv:
            if not cuts:
                return False
            record = self.pins.get(conv)
            if record and (record.get("parked") or record["slot"] is not None):
                return False       # its own cache is better
            saved = self.openings
            stored = deepest_shared(cuts, saved)

            base = cuts[0]
            # The first cut is a system prompt by construction.
            wanted = base[1] not in saved
            # Nobody has this opening: one request reads it, the others wait.
            plan = None
            if stored:
                plan = ("load", stored[1], saved[stored[1]], slot)
            elif wanted:
                if base[1] in self.building:
                    plan = ("wait", base[1], None, None)
                else:
                    self.building[base[1]] = time.time()
                    plan = ("read", base[1], None, slot)
            # Every new session wants the base, unless it is being read now.
            if wanted and plan is None:
                self.note_want(base, "base-", system, tools,
                               messages[:base[0] + 1], path)
            # Deeper only past what is saved, where a slot holds it.
            seen = set().union(*self.holds.values()) if self.holds else set()
            shared = deepest_shared(cuts, seen)
            if (DEEP_OPENINGS and shared and shared[1] != base[1]
                    and shared[1] not in saved
                    and (stored is None or shared[0] > stored[0])):
                self.note_want(shared, "deep-", system, tools,
                               messages[:shared[0] + 1], path)

            # Measurement only: what a fork could have started from. `shared`
            # needs the parent to hold a slot, so the fork rate it shows is a
            # floor. A parked copy restores the same way an opening does.
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

            # A request sharing more than the base opening has branched off
            # somebody's session. The pins say whose slot holds the deep cut.
            if shared and shared[1] != base[1]:
                holder = next((c for c, p in self.pins.items()
                               if shared[1] in self.holds.get(
                                   (p["backend"], p["slot"]), ())), None)
                if holder and holder != conv \
                        and self.forked.get(conv) != (holder, shared[0]):
                    self.forked[conv] = (holder, shared[0])
                    # Assigning an existing key does not move it. Without
                    # this the dedupe above wrote the same fork twice.
                    self.forked.move_to_end(conv)
                    while len(self.forked) > RECENT_REQUESTS:
                        self.forked.popitem(last=False)
                    EVENTS.write("fork", conv=short_key(conv),
                                 parent=short_key(holder), depth=shared[0],
                                 cuts=len(cuts))
            # The cut keys, so an offline report can match them to copies.
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
        """Wait for another request to save the opening, then load it.
        Measured: five sessions starting together read 92,000 tokens where
        24,000 would do, and the last finished after seventeen minutes."""
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
        The request it came from is gone by the time the builder runs."""
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
        """Read one wanted opening into an idle slot. Return its key, or None.
        The slot is held for the read."""
        with self.cv:
            if not self.wants:
                return None
            idle = [(b, self._idle_slot(b)) for b in self.backends
                    if prefills(b)]
            idle = [pair for pair in idle if pair[1] is not None]
            if not idle:
                return None
            be, slot = max(idle, key=lambda p: p[0]["slots"] - p[0]["busy"])
            key, want = next(reversed(self.wants.items()))
            be["busy"] += 1

        # The builder is the second way into a slot. Reading an opening over
        # a finished conversation's only cache lost that cache while the pin
        # still said the slot was held. IDLE_POLLS makes that rarer only.
        self.ensure_parked(be, None, post, remove)
        began = time.time()
        try:
            kept = self._read_prefix(want["cut"], want["head"], want["system"],
                                     want.get("tools") or [], be, slot, post,
                                     remove, want["mark"], want["path"])
        finally:
            with self.cv:
                be["busy"] -= 1
                # Dropped either way. A failed read fails again.
                self.wants.pop(key, None)
                self.cv.notify_all()
        EVENTS.write("want", key=short_key(key), shelf=mark_shelf(want),
                     action="built", backend=be["name"], slot=slot,
                     ok=bool(kept), secs=round(time.time() - began, 1),
                     age=round(began - want.get("at", began), 1))
        return key if kept else None

    def park_all(self, post, timeout=POST_TIMEOUT, only=None, budget=None):
        """Copy live caches to disk, so a stop does not throw them away. `only`
        names one backend, for a drain. A conversation mid-turn is skipped:
        its slot is busy. Each conversation is tried once, or a refused save
        loops forever. A full disk at SIGTERM spun here."""
        parked = 0
        # `budget` is a wall clock across every backend: the signal handler
        # gets 90 s from stop-all.sh. A drain passes none.
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
                            and not copy_is_current(record)
                            and worth_keeping(record)]
                    if not live:
                        break
                    # The per-save timeout is capped by what is left of the
                    # budget, asked with work in hand.
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
        """Write down what each opening has earned, for the next run: the
        shelf order and the load counts."""
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
                     # What the budget sweep orders by, so it has to outlive
                     # the process: restored as `now`, every copy reads as
                     # freshly used and the first sweep after a restart has
                     # nothing to tell them apart by.
                     "last": record.get("last"),
                     # For adopt, which dates a file the last run vouched for.
                     "parked_at": record.get("parked_at")}
                    for conv, record in self.pins.items() if record.get("parked")]
        write_rows(pins_file(), kept)
        return len(kept)

    def hand_off(self, conv, source, tokens, post, remove=drop_file, wanted=None,
                 migrate=True):
        """Move a conversation to the backend it generates on.

        The prefiller is released before the wait to generate. The other
        order left three prefillers idle for seven minutes on one reply.
        Between the save and the restore the conversation is parked, so a
        failure leaves it parked, not lost. Returns the backend to generate
        on: `source` when nothing was carried, None when nobody waits.

        `migrate` false asks for a turn that is not worth carrying: a park and
        a recall of the whole slot, to write one token. A source that may not
        generate is carried anyway. Which instances answer is the operator's
        to say, not this turn's."""
        if not HANDOFF_ON:
            return self._stay(source, "the handoff is turned off")
        if not migrate and generates(source):
            return self._stay(source, "this turn is not worth carrying")
        target = self.generator(tokens)
        while target is None and not generates(source):
            # Nothing to carry this to, and the instance holding it does
            # not generate. Wait, holding a prefill slot.
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
                # The generator went away. A slow answer beats none.
                with self.cv:
                    source["busy"] += 1
                return source
            # `generate: false` is an operator's setting. The turn is parked
            # on disk, the cheapest place to wait. Wait for a generator.
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
        """Generate where the prompt was read, because carrying it failed. Said
        aloud when the instance is set not to generate: nothing was carried,
        so there is no copy for a generator to restore."""
        if not generates(source):
            print(f"[router] generating on {source['name']} though it is set "
                  f"not to: {why}", flush=True)
        return source

    def generator(self, tokens):
        """The backend turns migrate to after their prompt is read, or None:
        one that generates and does not prefill. Where every instance does
        both, a turn generates where it read. None also when the configured
        one is down, draining or too small."""
        with self.cv:
            for be in sorted(self.backends, key=lambda b: b["pref"]):
                if prefills(be) or not generates(be):
                    continue
                if be["up"] and not be.get("draining") and tokens <= be["n_ctx"]:
                    return be
            return None

    def _wait_to_generate(self, target, wanted):
        """Wait until the generator has a slot. Returns the slot id, or None
        when the generator cannot serve this turn or the client left."""
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
                            target["busy"] += 1   # counted like a request
                            return free
                    if wanted is not None and not wanted():
                        return None
                    self.cv.wait(1.0)
        finally:
            with self.cv:
                self.to_generate -= 1

    def park_later(self, be, conv, post, ticket, remove=drop_file):
        """Copy a cache out of a backend that cannot read it, on a worker: the
        copy runs to gigabytes and the client already has its reply.
        `inflight` reserves the slot before this returns. The turn ticket
        goes with the job, because the copy overwrites the file the next turn
        restores from. Returns False, with the ticket still the caller's,
        when there is nothing to copy."""
        if prefills(be):
            return False              # it can be read again here
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if not record or record["slot"] is None:
                return False
            if not worth_keeping(record):
                return False          # shorter to read again than to copy
            slot = record["slot"]
            record["inflight"] = True      # hold it still while it copies
            if self.parker is None:
                self.parker = threading.Thread(target=self._run_parks,
                                               name="park", daemon=True)
                self.parker.start()
        self.park_jobs.put((conv, be, slot, post, remove, ticket))
        return True

    def _run_parks(self):
        """Write the queued copies. One worker: two multi-gigabyte writes at
        once only divide the same disk."""
        while True:
            job = self.park_jobs.get()
            conv, be, slot, post, remove, ticket = job
            try:
                self._save_park(conv, be, slot, post, remove)
            except Exception as err:
                print(f"[router] {short_key(conv)} could not be put away: "
                      f"{err}", flush=True)
            finally:
                # Whatever happened above. claim_turn has no deadline, so a
                # lost ticket wedges the conversation until a restart.
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
        Thrown away, a prompt too long for the client's patience is never
        read: every attempt gives up in the same place. Measured once at a
        114,354 token turn, abandoned twice after an hour, two thirds read
        each time."""
        if not conv:
            return False
        with self.cv:
            record = self.pins.get(conv)
            if not record:
                return False
            if copy_is_current(record):
                return False               # already on disk for this turn.
                                           # A save from a slot it has left
                                           # would delete the copy.
            record["inflight"] = True      # hold it still while it copies
        return self._save_park(conv, be, slot, post, remove)

    def note_holds(self, conv, be, cuts):
        """Record what a slot holds now, for a later request to start from."""
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if not record or not cuts:
                return
            # The opening this turn left on, for forget_stale_park.
            record["opening"] = cuts[0][1]
            # Every cut, kept past the slot, for the fork measurement.
            record["holds"] = {key for _, key in cuts}
            if record["slot"] is None:
                return
            self.holds[(be["name"], record["slot"])] = {key for _, key in cuts}
            self.holds_depth[(be["name"], record["slot"])] = max(i for i, _ in cuts)

    def forget_stale_park(self, conv, cuts, remove=drop_file):
        """Drop a copy whose opening the client has changed since. llama.cpp
        restores the state, finds no checkpoint before the point where the
        prompts part, and reads everything again. Measured once at 117,847
        tokens for prompts that parted at token 503. Without the copy the
        conversation loads a shared opening instead."""
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
        # Sized: the dashboard draws each file event over its byte count.
        read = file_size(name)
        with self.cv:
            shelf = None
            if key in self.openings:
                self.openings.move_to_end(key)  # in use, so keep it longest
                shelf = shelf_of(self.openings[key])
            self.loads[key] = self.loads.get(key, 0) + 1
            self.note_file("loaded opening", key, be, slot, read)
            loads = self.loads[key]
        self.save_openings()      # a load earns an opening its place
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
            # The zero-token reply's timings: tokens processed and cached.
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
        # From the disk, as _save_park: unpatched backends under-report.
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
            # `building` has no file to count yet.
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
        """One opening, as the backend's own template renders it: what two
        renderings that differ only after the opening share. /apply-template
        refuses anthropic tool_use and tool_result blocks, so an opening from
        /v1/messages goes through the anthropic route."""
        route = template_route(path)
        extra = {"tools": tools} if tools else {}
        # The anthropic route takes the system prompt in its own field, where
        # llama.cpp normalises it: server-chat.cpp
        # normalize_anthropic_billing_header rewrites Claude Code's cch=<hash>
        # to cch=fffff. Sent as a message it went through untouched, and the
        # saved block parted from every real turn about fifteen tokens in.
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
        """Report what the backends are doing. `busy` is what this router
        admitted. `active` is what the backend says: they differ when
        something else talks to the backends, or after a router restart."""
        now = time.time()
        mounts = self._read_mounts(now)
        with self.cv:
            keys = ("name", "url", "model", "slots", "n_ctx", "busy", "up", "served",
                    "stats", "slots_detail", "cache", "draining")
            rows = []
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

            # Sizes from opening_bytes, not a stat each, under the lock.
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
            # `slot` is set only while the slot still holds the cache, or
            # three copies naming one single-slot backend look like three
            # caches in one slot. `used` is the PARK_BUDGET sweep order.
            copies = [{"name": p["parked"], "kind": "copy", "conv": short_key(conv),
                       "bytes": p.get("bytes") or 0, "backend": p["backend"],
                       "slot": p.get("slot"), "parked_at": p.get("parked_at"),
                       "used": last_used(p)}
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
    """True for a path the browser needs. The web directory also holds
    node_modules, 31 MB that must not go on the wire."""
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
            # The backends serve their own web ui at the root. /router needs
            # the trailing slash so relative urls resolve under /router/.
            # Exact matches only: /router/ must reach the static handler.
            return self._redirect("/router/")
        if self.path.startswith("/router/"):
            rest = self.path[len("/router/"):].split("?")[0]
            if rest == "events":
                return self._events()
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
        # A drain that gave up, or whose saves failed, left caches only in
        # slots. Say so, or a script stops a backend that holds live caches.
        ok = report["quiet"] and not report["left"]
        return self._send(200 if ok else 409, json.dumps(report).encode())

    def _redirect(self, where):
        self.send_response(301)
        self.send_header("Location", where)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _config(self, kind):
        """Offer a ready client config, addressed to the host the reader used.
        The Host header is checked: the reader keeps the file for months."""
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
        """Answer with an error, and log it."""
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
            target.relative_to(base)              # no escaping the web dir
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
        """Push the pool status when it changes. The client's EventSource
        reconnects by itself."""
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
        """Answer /props and /slots for the whole pool. One backend's answer
        makes the router look like a one-slot server."""
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
        # A client's header. int() on junk raised before a status line went
        # out, and BaseHTTPRequestHandler catches only TimeoutError.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "Content-Length is not a number")
        if length < 0:
            return self._error(400, "Content-Length is negative")
        # Checked before the body is read, or a header alone could ask
        # this process for arbitrary memory.
        if length > MAX_BODY:
            self.close_connection = True
            return self._error(413, f"body of {length} bytes; this router "
                                    f"reads at most {MAX_BODY}")
        # Nothing here decodes chunked. Read as an empty body, the unread
        # chunks become the next request line on this connection.
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            return self._error(411, "send the body with a Content-Length: "
                                    "this router does not read chunked requests")
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?")[0].rstrip("/") or "/"

        if self.command == "GET" and path in ("/props", "/slots"):
            return self._pool_props(path)

        # Both chat apis are POST only. A bodiless GET took a slot, parked
        # every cache on the backend and restored gigabytes before the
        # backend answered 404. From an unauthenticated public port.
        if path in INFERENCE and self.command != "POST":
            return self._error(405, f"post to {path}")
        if path not in INFERENCE:
            # An allowlist, see PASSED.
            if path not in PASSED:
                return self._error(404, f"this router does not serve {path}")
            be = next((b for b in POOL.backends if b["up"]), None)
            if not be:
                return self._error(503, "no backend is up")
            return self._forward(be, body)

        # A typed question becomes an ordinary chat body: the rubric and the
        # state, which every question shares. The questions extend it one at
        # a time, once the state is read. Everything below this line then
        # works on `messages`, as it does for any other turn.
        plan, sent = None, body
        if path == SYSTEMONE:
            try:
                plan = systemone_plan(body)
                # The read pass carries the first question, so that both
                # phases send the same prompt. Reading the state alone cost
                # the first question a full re-read on production: 351 tokens
                # of a state of 348. Why is an open question in
                # tests/live/README.md; that it costs is measured.
                body = json.dumps(
                    systemone_body(plan, plan["questions"][0])).encode()
            except Refused as err:
                return self._error(400, str(err))
        up_path = SYSTEMONE_UP if plan else path
        # The word the flow board puts on this turn's slot. A typed question
        # writes one token where a prompt writes a reply, so the board has to
        # tell them apart. None is an ordinary turn, which wears no label.
        work = "typed" if plan else None

        vision = POOL.vision()
        tokens, images, image_charge = request_cost(body, vision)
        # `tokens` carries REPLY_TOKENS of room. The dashboard measures a
        # turn against the prompt sent: 1,024 tokens nobody sent is 41
        # seconds of reading nobody did.
        prompt_tokens = max(0, tokens - REPLY_TOKENS)
        largest = POOL.largest()
        if largest and tokens > largest:
            return self._error(413, f"needs about {tokens} tokens. "
                                    f"The largest backend holds {largest}.")
        if not largest:
            return self._error(503, "no backend is up yet")

        # The session id beats a guess from the prompt, and covers
        # /v1/messages, where the system prompt is a separate field.
        # `conv_source` says how the conversation was recognised. Decided
        # here: a re-parse of a ctx 150000 turn is megabytes.
        conv, conv_source = session_key(self.headers), "header"
        if not conv:
            conv, conv_source = prompt_key(body), "cache_key"
        if not conv:
            conv, conv_source = conversation_id(body), "hash"
        if not conv:
            conv_source = "none"
        client = client_kind(self.headers)
        # Before anything is changed, so a capture holds what the client sent.
        # `sent` is the typed body on a systemone turn, and `body` everywhere
        # else: what went out is rebuilt from the plan, and what came in is not.
        capture(conv, sent)
        # This model's template refuses a late system message. A typed body is
        # built above, and always in order, so asking would parse the whole
        # state again to learn nothing.
        ordered = body if plan else hoist_system(body)
        if ordered is not body:
            print(f"[router] a late system message became a user message "
                  f"for {path}", flush=True)
            EVENTS.write("start_over", conv=short_key(conv) if conv else None,
                         reason="late_system", client=client, path=path)
        body = ordered
        cuts, messages, system, tools = prompt_cuts(body)
        # The stream and its keep-alive open before the slot is asked for.
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
            # The turn ahead holds the pin, slot and copy this one needs.
            mine = POOL.claim_turn(conv, ticket, self._still_there, work)
            be = POOL.acquire(conv, tokens, self._still_there) if mine else None
        finally:
            POOL.end_wait(ticket)
        waited = time.time() - start
        if not be:
            # `done` only if this turn held the conversation: Flow is keyed
            # by conversation, and a turn that gave up in claim_turn deleted
            # the live row of the turn running.
            if mine:
                POOL.note_stage(conv, "done")
            POOL.finish_turn(conv, ticket)
            if stop_ping:
                stop_ping()
            if opened:
                return self._say_and_end("no backend can serve this request")
            return self._error(503, "no backend can serve this request")
        # Read here, then generate wherever is free after the read.
        serving = be
        left = False                           # the client gave up mid-read
        read_stats = {}
        recalled = loaded = warm = False
        slot = None
        # The backend and the conversation are held from here. Every way out
        # runs the same ending: claim_turn has no deadline, so a claim left
        # behind stops the conversation for good.
        try:
            warm = bool(conv) and POOL.holds_slot(conv)
            # One slot, decided once, for the read below to extend.
            slot = POOL.pick_slot(be, conv)
            POOL.note_stage(conv, "prefill", be["name"], slot, work)
            # Nothing reaches the backend until the caches on it are on disk.
            POOL.ensure_parked(be, conv, http_post)
            # A copy with an opening the client no longer sends is no prefix.
            if POOL.forget_stale_park(conv, cuts):
                EVENTS.write("start_over", conv=short_key(conv) if conv else None,
                             reason="stale_copy", client=client, path=path)
            recalled = POOL.recall(conv, be, slot, http_post)
            loaded = (not recalled
                      and POOL.warm_prefix(conv, cuts, messages, system, tools,
                                           be, slot, http_post, up_path))
            if asked is not None:
                # Watch the client. The timings say what the cache saved.
                answer = http_post_wanted(be["url"], up_path, read_only(body, slot),
                                          READ_TIMEOUT, self._still_there)
                timing = (answer or {}).get("timings") or {}
                read_stats = {"read_prompt_n": timing.get("prompt_n"),
                              "read_cache_n": timing.get("cache_n")}
                POOL.note_slot(conv, slot)
                # A typed question writes one token, which is not worth a park
                # and a recall of the slot it was read into.
                serving = POOL.hand_off(conv, be, tokens, http_post,
                                        wanted=self._still_there,
                                        migrate=not plan)
                if serving is None:
                    # The cache is parked, and no backend is held.
                    raise Gone("the client stopped waiting for a slot to generate in")
            if serving is be:
                # Nothing was carried. hand_off already noted a carried turn.
                POOL.note_stage(conv, "generate", be["name"], slot, work)
            if stop_ping:
                stop_ping()                    # waits for a ping in flight
            if plan:
                self._systemone(serving, slot if serving is be else None,
                                plan, took_from=start, read_stats=read_stats)
            else:
                self._forward(serving, body, conv, opened=opened)
        except Gone:
            # Nobody to answer. What the read got through is parked below.
            print(f"[router] {short_key(conv)} left while {be['name']} was "
                  f"reading, {time.time() - start:.0f}s in ({self.went})",
                  flush=True)
            left = True
        except Exception as err:
            if opened:
                # The reply already started, so say it in the stream.
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
                # After the release: these write the copy that is not behind.
                if left:
                    POOL.park_partial(conv, be, slot, http_post)
                # A backend that does not read cannot serve the next turn, so
                # leave a copy for one that does, on a worker.
                if serving is not None:
                    parking = POOL.park_later(serving, conv, http_post, ticket)
            except Exception as err:
                # A ticket not given back costs the conversation every later
                # turn: claim_turn has no deadline. The lines below must run.
                print(f"[router] {short_key(conv)} could not be put away: "
                      f"{err}", flush=True)
            took = time.time() - start
            POOL.note_stage(conv, "done")
            # The worker ends the turn once the copy has landed.
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
        """False once the client has closed its end. The body is already read,
        so readable with nothing on it means gone. `self.went` records which
        case. poll, not select: select refuses a descriptor at or above
        FD_SETSIZE (1024) and raises ValueError for a live client."""
        try:
            watch = select.poll()
            watch.register(self.connection, select.POLLIN)
            if not watch.poll(0):
                return True
            if self.connection.recv(1, socket.MSG_PEEK):
                return True
            self.went = "closed its end"
        except (OSError, ValueError) as err:
            # fileno() is -1 on a closed socket, which poll refuses.
            self.went = f"{type(err).__name__}: {err}"
        return False

    def _open_stream(self, opening=b""):
        """Answer the client now, before the prompt is read: a read sends
        nothing for tens of minutes, and a client drops a quiet stream. The
        stream opens with its protocol's opening event, not a keep-alive."""
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
        """Fill the silence with keep-alives. Returns the way to stop, which
        joins the thread: a chunk frame carries its own length, so half a
        frame makes the rest of the stream unreadable. Both writers take the
        same lock."""
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
                        # Without this the keep-alive stops silently.
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
        An anthropic stream ends in its error event. Under `sending`: the
        keep-alive thread may still be running."""
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

    def _systemone(self, be, slot, plan, took_from, read_stats):
        """Ask every question against the slot that holds the state, and
        answer in one piece. _forward is no use here: it sends the client's
        own path upstream, and there is one reply a question to gather."""
        answers, wrote, steps = systemone_answers(be, slot, plan, http_post)
        read = read_stats.get("read_prompt_n") or 0
        reused = read_stats.get("read_cache_n") or 0
        self._send(200, json.dumps({
            "model": plan["model"],
            "answers": answers,
            "usage": {"input_tokens": read + reused, "output_tokens": wrote},
            "router": {"backend": be["name"], "read": read, "reused": reused,
                       "took": round(time.time() - took_from, 2),
                       "questions": steps},
        }).encode())

    def _forward(self, be, body, conv=None, opened=False):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in DROP_HEADERS}
        # llama-server 404s on a trailing slash, which a base url ending in
        # /v1 joined with /models produces.
        path, sep, query = self.path.partition("?")
        target = (path.rstrip("/") or "/") + sep + query
        # A streamed openai reply reports no usage unless asked. The router
        # asks, reads the figures and removes the chunk when the client did
        # not ask. The anthropic route reports usage unprompted.
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
            # With `opened` the status line went out long ago. A second HTTP
            # response inside the chunked body poisons the connection.
            if opened:
                return self._say_and_end(f"{be['name']}: {e}")
            return self._error(502, f"{be['name']}: {e}")

        if upstream.status >= 400:
            shape = request_shape(body)
            print(f"[router] {be['name']} refused {self.path.split('?')[0]} "
                  f"with {upstream.status}: {shape}", flush=True)
            if opened:
                # No status left to send, and a refusal body is not an event.
                with upstream:
                    reason = said_in(upstream.read()) or upstream.reason
                return self._say_and_end(f"{be['name']}: {reason}")

        with upstream:
            # With `opened` the headers went out long ago.
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

            # read1 returns what arrived. read waits for a full buffer.
            read = getattr(upstream, "read1", upstream.read)

            # The backend sends nothing while it reads, which can be an hour.
            # A client drops a stream quiet for five minutes.
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
                            return             # client left

            pinger = None
            streamed = wants_ping(upstream.headers.get("Content-Type"), length)
            if streamed:
                pinger = threading.Thread(target=keep_alive, daemon=True)
                pinger.start()

            # The stream began with the router's own message_start.
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
                            continue           # partial, or the dropped one
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
                pass                           # client left
            except (OSError, http.client.HTTPException) as err:
                # The backend went away mid-reply. The status line is on the
                # wire, so there is no second reply: reaching _route's handler
                # wrote a whole HTTP response inside the one in flight.
                print(f"[router] {be['name']} stopped mid-reply: {err}", flush=True)
                done.set()
                if pinger:
                    pinger.join(2)
                if not length:
                    self._say_and_end(f"{be['name']}: {err}")
                else:
                    self.close_connection = True   # body short of its count
            finally:
                done.set()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # Accept IPv4 on the IPv6 socket. Tailscale gives a machine both,
        # and macOS clients try IPv6 first.
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()

    def handle_error(self, request, address):
        """A client hanging up is not an error worth a traceback. Python's
        request loop is reading the next request line when a client drops
        its keep-alive connection."""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError,
                                          ConnectionAbortedError)):
            return
        super().handle_error(request, address)


class Stamped:
    """Put the time in front of every line the router prints. Every print
    in this file goes through this."""

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
        """Copy every live cache out before the backends go away. A cache
        only exists in a slot."""
        if stopping.is_set():
            sys.exit(1)           # a second signal means stop arguing
        stopping.set()
        print("[router] stopping, parking caches", flush=True)
        # The worker's copy first. park_all skips a record being copied, and
        # sys.exit kills the daemon worker mid-copy.
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
        # sys.exit drops what the daemon writer still holds.
        EVENTS.flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shut_down)
    signal.signal(signal.SIGINT, shut_down)
    print(f"[router] {len(BACKENDS)} backend(s) from {BACKENDS_FROM}: "
          f"{', '.join(b['name'] for b in BACKENDS)}", flush=True)
    print(f"[router] listening on {args.host}:{args.port}", flush=True)
    Server((args.host, args.port), Handler).serve_forever()
