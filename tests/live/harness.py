"""Start real llama-server instances, small enough to sit beside production.

The stub in tests/fake_backend.py was built from the beliefs the router holds,
so it agrees with the router whether or not llama.cpp does. These tests use the
real thing instead.

Everything here is deliberately tiny: a model of a few hundred megabytes, ctx
4096, four threads, no GPU, ports from 18080 up, its own directory under /tmp.
Production on 8080 to 8083 is never touched, and nothing is written into run/.

Set LIVE_E2E=1 to run the suite. Without it every test skips, so a stray
`unittest discover` over the project costs nothing.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
SERVER = PROJECT / "llama.cpp-mtp" / "build" / "bin" / "llama-server"

# Where the small models live. Nothing else on the machine uses this directory.
MODEL_DIR = Path(os.environ.get("LIVE_MODEL_DIR")
                 or Path.home() / ".cache" / "llama-arbiter" / "live-models")

# Ports well clear of production's 8080-8083 and the router's 8090.
FIRST_PORT = int(os.environ.get("LIVE_FIRST_PORT", "18080"))

# A server loads in a couple of seconds. The wait is long only so a loaded
# machine fails with a clear message instead of a flake.
BOOT_PATIENCE = float(os.environ.get("LIVE_BOOT_PATIENCE", "120"))

# Three shapes of model, because llama.cpp treats their KV caches differently
# and the router's beliefs are about the KV cache.
#
#   hybrid  attention layers and a recurrent state, like Qwen3-Next in
#           production. A recurrent state cannot be rewound.
#   plain   full attention and nothing else. Any prefix of what a slot holds
#           can be recovered by dropping cells.
#   swa     sliding-window attention. A slot only holds the last n_swa
#           positions of most layers.
MODELS = {
    "hybrid": "Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
    "plain":  "Qwen3-0.6B-Q4_K_M.gguf",
    "swa":    "gemma-3-270m-it-Q8_0.gguf",
}

# The model the suite uses unless a test asks for another. Production runs a
# hybrid recurrent model, so that is the default: a plain model would answer
# the rewind questions the way production does not.
DEFAULT_KIND = os.environ.get("LIVE_MODEL_KIND", "hybrid")


def model_path(kind=None):
    """The file for one shape of model, or None when it is not here."""
    name = MODELS[kind or DEFAULT_KIND]
    found = MODEL_DIR / name
    return found if found.exists() else None


def why_not(kind=None):
    """The reason this suite cannot run, or None when it can."""
    if os.environ.get("LIVE_E2E") != "1":
        return "set LIVE_E2E=1 to run the live suite (it starts llama-server)"
    if not SERVER.exists():
        return f"no llama-server at {SERVER}"
    if model_path(kind) is None:
        return (f"no {kind or DEFAULT_KIND} model at "
                f"{MODEL_DIR / MODELS[kind or DEFAULT_KIND]}; see tests/live/README.md")
    return None


def requires(kind=None):
    """Skip decorator for a test that needs a live server."""
    return unittest.skipIf(why_not(kind) is not None, why_not(kind) or "")


# ---------------------------------------------------------------------------
# talking to a backend
# ---------------------------------------------------------------------------

class HttpError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def post(url, path, payload, timeout=300):
    """POST json, return the parsed reply. Raises HttpError on 4xx and 5xx."""
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as err:
        raise HttpError(err.code, err.read().decode(errors="replace")) from None


def get(url, path, timeout=30):
    with urllib.request.urlopen(url + path, timeout=timeout) as reply:
        return json.load(reply)


def free_port(start=FIRST_PORT):
    """A port nothing is listening on, from `start` up. Never a production one."""
    for port in range(start, start + 200):
        if port in (8080, 8081, 8082, 8083, 8090):
            continue
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port in the test range")


class Server:
    """One llama-server, started for a test and stopped with it.

    The footprint is deliberately small: a few hundred megabytes of weights,
    4096 tokens of context and four threads. Production is using most of this
    machine, so a test instance has to be obviously harmless.
    """

    def __init__(self, root, name, kind=None, port=None, slots=2, ctx=4096,
                 unified=True, flash=None, threads=4, extra=(), env=None,
                 slot_dir=None, verbose=4):
        self.name = name
        self.root = Path(root)
        self.kind = kind or DEFAULT_KIND
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.slots = slots
        self.ctx = ctx
        self.slot_dir = Path(slot_dir) if slot_dir else self.root / "slots"
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / f"{name}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = None

        model = model_path(self.kind)
        if model is None:
            raise unittest.SkipTest(why_not(self.kind) or "no model")

        argv = [str(SERVER),
                "--model", str(model),
                "--alias", f"live-{self.kind}",
                "--host", "127.0.0.1", "--port", str(self.port),
                "--ctx-size", str(ctx),
                "--parallel", str(slots),
                "--threads", str(threads),
                # No GPU: the production instance that generates owns the card.
                "--n-gpu-layers", "0", "--device", "none",
                "--metrics",            # the router reads these
                "--no-webui",
                "--no-warmup",
                "-lv", str(verbose),    # trace, as production runs it
                "--slot-save-path", str(self.slot_dir) + os.sep]
        argv += ["--kv-unified"] if unified else ["--no-kv-unified"]
        if flash is not None:
            argv += ["--flash-attn", "on" if flash else "off"]
        argv += list(extra)
        self.argv = argv

        # CUDA_VISIBLE_DEVICES empty as well as --device none: two ways of
        # saying the same thing, because touching the GPU would be felt.
        run_env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        run_env.update(env or {})
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=self.log,
                                     start_new_session=True)

    # ---- lifecycle --------------------------------------------------------

    def wait_ready(self, patience=BOOT_PATIENCE):
        stop = time.time() + patience
        while time.time() < stop:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} died while starting, exit {self.proc.returncode}\n"
                    + self.tail(40))
            try:
                if get(self.url, "/health", timeout=2).get("status") == "ok":
                    return self
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(f"{self.name} never became ready\n" + self.tail(40))

    def stop(self):
        """Stop this instance. Safe to call twice, and never misses.

        The pid is the one Popen holds, so nothing here can match another
        process: no pattern, no pkill.
        """
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=20)
        if self.log and not self.log.closed:
            self.log.close()

    # ---- what the backend says --------------------------------------------

    def post(self, path, payload, timeout=300):
        return post(self.url, path, payload, timeout=timeout)

    def get(self, path, timeout=30):
        return get(self.url, path, timeout=timeout)

    def props(self):
        return self.get("/props")

    def slots_json(self):
        return self.get("/slots")

    def slot(self, id_slot):
        return next((s for s in self.slots_json() if s["id"] == id_slot), None)

    def held(self, id_slot):
        """How many tokens the backend says this slot holds. Ground truth."""
        row = self.slot(id_slot)
        return None if row is None else row.get("n_prompt_tokens")

    def prefill(self, prompt, id_slot=0, n_predict=0, timeout=300):
        """Read a prompt into a slot. Returns the backend's own timings.

        The interesting keys are `prompt_n`, the tokens it had to read, and
        `cache_n`, the tokens it did not.
        """
        reply = self.post("/completion",
                          {"prompt": prompt, "n_predict": n_predict,
                           "cache_prompt": True, "id_slot": id_slot},
                          timeout=timeout)
        return reply

    def save_slot(self, id_slot, filename, timeout=300):
        return self.post(f"/slots/{id_slot}?action=save", {"filename": filename},
                         timeout=timeout)

    def restore_slot(self, id_slot, filename, timeout=300):
        return self.post(f"/slots/{id_slot}?action=restore", {"filename": filename},
                         timeout=timeout)

    def metrics(self):
        with urllib.request.urlopen(self.url + "/metrics", timeout=10) as reply:
            text = reply.read().decode()
        out = {}
        for line in text.splitlines():
            if line.startswith("#") or "{" in line:
                continue
            key, _, number = line.partition(" ")
            try:
                out[key.split(":", 1)[-1]] = float(number)
            except ValueError:
                pass
        return out

    # ---- its log ----------------------------------------------------------

    def log_text(self):
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def tail(self, lines=20):
        return "\n".join(self.log_text().splitlines()[-lines:])

    def mark(self):
        """A bookmark in this backend's log, so a test can read only what
        happened after it.

        A byte offset, not a character count: the log grows through a test and
        reading the whole of a -lv 4 trace to measure its length, then reading
        it again to slice it, is the most expensive thing some of these tests
        do. CacheWatch.poll in the router keeps its place the same way."""
        try:
            return self.log_path.stat().st_size
        except OSError:
            return 0

    def since(self, mark):
        """Everything the log gained after that bookmark."""
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(mark)
                return handle.read().decode(errors="replace")
        except OSError:
            return ""


def tokens_read(reply):
    """Tokens this request had to read. The backend's own count."""
    return reply["timings"]["prompt_n"]


def tokens_reused(reply):
    """Tokens this request did not have to read, because a slot held them."""
    return reply["timings"]["cache_n"]


# "prompt eval time = 12.3 ms / 456 tokens", which is what the backend reports
# for every request. This is the ground truth a test asserts on: the router's
# own log would have agreed with the router all day.
EVAL_RE = re.compile(
    r"id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| prompt eval time =\s*"
    r"(?P<ms>[\d.]+) ms /\s*(?P<tokens>\d+) tokens")


def prompt_evals(text):
    """Every prompt eval the backend logged, in order.

    Each is {slot, task, ms, tokens}. `tokens` is what it actually had to
    read, so a cache that worked shows up as a small number here and a cache
    that did not shows up as the whole prompt."""
    return [{"slot": int(m["slot"]), "task": int(m["task"]),
             "ms": float(m["ms"]), "tokens": int(m["tokens"])}
            for m in EVAL_RE.finditer(text)]


def read_tokens(text):
    """Total tokens read in this stretch of a backend's log."""
    return sum(row["tokens"] for row in prompt_evals(text))


# A slot restored from a file has no context checkpoints in stock llama.cpp;
# patches/slot-state-carries-checkpoints.patch changes that. Which build is in
# llama.cpp-mtp/build is a property of this machine, so a test that turns on it
# asks the backend rather than assuming.
REREAD_LINE = "forcing full prompt re-processing"


def restored_slot_can_rewind(server, id_slot, prompt, filename="regime.bin"):
    """Measure which of the two builds this is, on this very backend.

    Reads `prompt` into a slot, saves it, restores it, and asks for a prompt
    one word shorter. Returns True when the restored slot rewound into a
    checkpoint carried in the file, False when the whole prompt was read
    again. Leaves the slot holding the shorter prompt.
    """
    server.prefill(prompt, id_slot=id_slot, n_predict=0)
    server.save_slot(id_slot, filename)
    server.restore_slot(id_slot, filename)
    before = server.mark()
    reply = server.prefill(prompt.rsplit(" ", 1)[0], id_slot=id_slot, n_predict=0)
    reread = REREAD_LINE in server.since(before)
    return reply["timings"]["cache_n"] > 0 and not reread


# A sentence of ordinary English, so a character count divides by about four
# to give tokens. A prompt of invented words tokenises at nearly one token a
# character, which makes every size in a test a surprise.
SENTENCE = ("The router keeps each conversation on one backend so that its "
            "next turn extends a cache instead of reading a prompt again. ")


def prose(chars, salt=""):
    """Roughly `chars` characters of filler, distinct per salt."""
    body = (salt + " " if salt else "") + SENTENCE * (chars // len(SENTENCE) + 1)
    return body[:max(len(salt) + 1, chars)]


def filler(words, word="token"):
    """A prompt of roughly `words` invented words. Tokenises at about four
    tokens a word, so it is the cheap way to make a long prompt."""
    return " ".join(f"{word}{n}" for n in range(words))


class LiveCase(unittest.TestCase):
    """A temporary root, and every server started here stopped at the end."""

    KIND = None          # None means the default, which is the hybrid model

    @classmethod
    def setUpClass(cls):
        reason = why_not(cls.KIND)
        if reason:
            raise unittest.SkipTest(reason)

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="live-e2e-", dir="/tmp"))
        self.servers = []
        self.addCleanup(self.stop_everything)

    def start(self, name, **kw):
        """One instance, sharing this test's slot directory and log directory.

        Every instance writes <name>.log into the same root, which is what the
        router's CacheWatch expects to find under RUN_DIR, and they share one
        --slot-save-path so a state saved on one can be restored on another.
        """
        kw.setdefault("kind", self.KIND)
        server = Server(self.root, name, **kw)
        self.servers.append(server)
        return server.wait_ready()

    def stop_everything(self):
        problems = []
        for server in self.servers:
            try:
                server.stop()
            except Exception as err:            # never leave one listening
                problems.append(f"{server.name}: {err}")
        if os.environ.get("LIVE_KEEP") != "1":
            shutil.rmtree(self.root, ignore_errors=True)
        else:
            print(f"[live] kept {self.root}", file=sys.stderr)
        if problems:
            raise AssertionError("a server would not stop: " + "; ".join(problems))
