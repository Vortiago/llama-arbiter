"""A stub llama-server on an ephemeral localhost port.

It serves every endpoint the router reaches for, and answers honestly: a slot
says it is processing while it really is, a save really writes a file into
the store it is given, and a restore really reads one back. Idle detection, park
recall all read this state, so a stub that lied would prove nothing.

The cache model is small but real. A slot holds the text its KV covers, a
request opening with that text skips those tokens, and a save and restore
carry the text from one slot to another.

A turn arrives twice: the router reads the prompt with a one-token request and
throws the answer away, then sends the real one wherever the slot ended up.
Both are chat calls, so each is recorded with which it was.

Every slow step is measured in milliseconds and can be set.
"""
import http.server
import json
import select
import socket
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import router

JSON = "application/json"


class Cancelled(Exception):
    """The client went away while the turn was being worked on."""

CHARS_PER_TOKEN = 4        # the same rough measure the router itself uses
REPLY = "ok<|im_end|>\n"   # what a turn leaves in the slot behind the prompt


def text_of(content):
    """The words in a message, whether it is a string or a list of parts."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def anthropic_messages(body):
    """The messages an anthropic body carries, the system prompt included.

    /v1/messages keeps the system prompt out of the list. The backend puts it
    back at the front before it renders, so the stub does too: otherwise two
    requests that differ only in their system prompt render alike here."""
    system = text_of(body.get("system"))
    lead = [{"role": "system", "content": system}] if system else []
    return lead + [m for m in (body.get("messages") or []) if isinstance(m, dict)]


def anthropic_stream(model, cached, read):
    """One turn as the anthropic protocol streams it.

    llama.cpp builds this in server-task.cpp: a message_start carrying the id
    and the prompt token count, one text block, then the message_delta and
    message_stop that close the message. A stub that sent something simpler
    could not show a stream that opens wrongly."""
    return [
        ("message_start",
         {"type": "message_start",
          "message": {"id": f"chatcmpl-{model}", "type": "message",
                      "role": "assistant", "content": [], "model": model,
                      "stop_reason": None, "stop_sequence": None,
                      "usage": {"cache_read_input_tokens": cached,
                                "input_tokens": read, "output_tokens": 0}}}),
        ("content_block_start",
         {"type": "content_block_start", "index": 0,
          "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta",
         {"type": "content_block_delta", "index": 0,
          "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta",
         {"type": "message_delta",
          "delta": {"stop_reason": "end_turn", "stop_sequence": None},
          "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def closes(message):
    """True when a template can end a prompt after this message.

    llama.cpp will not close an assistant message that calls a tool, because
    the model is part way through its turn. The router has to cut somewhere
    else, and this is where it finds out."""
    if message.get("role") != "assistant":
        return True
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    return not (isinstance(content, list)
                and any(isinstance(part, dict) and part.get("type") == "tool_use"
                        for part in content))


def render(messages):
    """These messages as a chat template would lay them out.

    The trailing generation prompt is what makes two renderings of the same
    opening differ, which is how the router finds where the opening ends."""
    parts = [f"<|im_start|>{m.get('role')}\n{text_of(m.get('content'))}"
             f"<|im_end|>\n" for m in messages if isinstance(m, dict)]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def tokens_in(text):
    return len(text) // CHARS_PER_TOKEN


def want_tokens(body):
    """How many tokens this request asks for, under either name, or None."""
    for name in ("n_predict", "max_tokens"):
        if isinstance(body.get(name), int):
            return body[name]
    return None


def is_probe(body):
    """True for the router's read pass, which is what read_only builds.

    No tokens at all, and a report of the slot that served it. No client asks
    for either, so nothing else can be mistaken for it."""
    return bool(body.get("verbose")) and want_tokens(body) == 0


def reusable(held, coming, checkpointed):
    """How much of `held` this request may reuse.

    A slot keeps context checkpoints only for text it read itself. A slot
    restored from a file has none, and llama.cpp must process at least one
    token to produce logits. So a restored slot can only be reused when the
    request extends it strictly: an exact match has nothing left to process,
    needs to step back one token, and there is no checkpoint to step back to.

    Measured on the real server: a restored slot given exactly its own prompt
    re-read all 601 tokens in 19.0 s; given one word more it read 10 in 0.56 s."""
    shared = shared_tokens(held, coming)
    if checkpointed:
        return shared
    return shared if shared == len(held) and len(coming) > len(held) else 0


def shared_tokens(held, coming):
    """The tokens this slot already holds for the request coming in.

    A slot reuses its KV only from the very start and only where the text
    matches, so this counts the common prefix and nothing else."""
    n, limit = 0, min(len(held), len(coming))
    while n < limit and held[n] == coming[n]:
        n += 1
    return n // CHARS_PER_TOKEN


class Slot:
    """One slot, and the prompt its KV covers."""

    def __init__(self, sid):
        self.id = sid
        self.busy = False
        self.task = -1
        self.held = ""
        self.n_prompt = 0
        self.processed = 0
        self.cached = 0
        self.decoded = 0


class FakeBackend:
    """A stub llama-server. Start it, hand its url to the Pool, stop it."""

    def __init__(self, name="be", slots=1, n_ctx=150000, model="fake-model",
                 busy_ms=20, save_ms=0, restore_ms=0, save_bytes=None,
                 queue_wait=20.0, gate_wait=30.0, store=None, park_floor=None):
        self.name = name
        self.n_ctx = n_ctx
        self.model = model
        self.busy_s = busy_ms / 1000.0
        self.save_s = save_ms / 1000.0
        self.restore_s = restore_ms / 1000.0
        # A real state carries the recurrent state whatever the length, so the
        # default is above the floor the router treats as a real cache. A test
        # that wants the "nothing to park" path sets it below.
        # The store it writes slot files into, and the size the router
        # treats as a real cache. Handed in, because this double has no
        # opinion about either.
        self.store = store
        park_floor = router.Tuning().park_floor if park_floor is None else park_floor
        self.save_bytes = (park_floor + 4096 if save_bytes is None
                           else save_bytes)
        self.queue_wait = queue_wait      # longest wait for a slot to free
        self.gate_wait = gate_wait        # longest wait for a held turn
        # A backend can refuse to move a state. Set by a test that wants the
        # path where the handoff gives up and the conversation stays put.
        self.fail_save = False
        self.fail_restore = False
        # A backend can turn a generate call down. Set by a test that wants
        # the path where the refusal lands in a stream already under way.
        self.refuse = False

        self.lock = threading.Lock()
        self.slots = [Slot(n) for n in range(slots)]
        self.gate = None                  # set by hold(), cleared by release()
        self.task_seq = 0

        # What a test reads afterwards.
        self.requests = []                # every call, in order
        self.chats = []
        self.completions = []
        self.saves = []
        self.restores = []
        self.erases = []
        self.templates = 0
        self.cancelled = 0                # turns whose client left
        self.counters = {"prompt_tokens_total": 0.0,
                         "prompt_tokens_cached_total": 0.0,
                         "prompt_seconds_total": 0.0,
                         "tokens_predicted_total": 0.0,
                         "tokens_predicted_seconds_total": 0.0,
                         "n_tokens_max": 0.0}

        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.server.backend = self
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name=f"stub-{name}", daemon=True)
        self.thread.start()

    # ---- what a test drives ----------------------------------------------

    @property
    def probes(self):
        """The read passes: one token asked for, the answer thrown away."""
        return [chat for chat in self.chats if chat["probe"]]

    @property
    def answers(self):
        """The calls that generated a reply for a client."""
        return [chat for chat in self.chats if not chat["probe"]]

    def hold(self):
        """Make every turn from now on wait until release() is called."""
        self.gate = threading.Event()

    def release(self):
        """Let the held turns finish, and stop holding new ones."""
        gate, self.gate = self.gate, None
        if gate is not None:
            gate.set()

    def stop(self):
        self.release()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    # ---- the slot machinery ----------------------------------------------

    def take_slot(self, wanted=None):
        """Hold a slot for a turn. Waits for one, as a real backend queues."""
        stop = time.time() + self.queue_wait
        while True:
            with self.lock:
                if wanted is None:
                    choices = self.slots
                elif 0 <= wanted < len(self.slots):
                    choices = [self.slots[wanted]]
                else:
                    return None
                for slot in choices:
                    if not slot.busy:
                        slot.busy = True
                        self.task_seq += 1
                        slot.task = self.task_seq
                        return slot
            if time.time() > stop:
                return None
            time.sleep(0.01)

    def wait_idle(self, sid):
        """Wait for a slot to stop working, as a save has to."""
        if not 0 <= sid < len(self.slots):
            raise IndexError(f"no slot {sid}")
        slot = self.slots[sid]
        stop = time.time() + self.queue_wait
        while slot.busy:
            if time.time() > stop:
                raise TimeoutError(f"slot {sid} is still working")
            time.sleep(0.01)
        return slot

    def _wait_out(self, gone=None):
        """Spend the time a turn takes, and stop if the client has left.

        llama.cpp polls the request socket while it works and cancels the task
        when the connection has ended. The stub polls the same way, because a
        router that closes its own end without the backend ever seeing it
        looks right here and leaves a real backend reading for nobody."""
        gate = self.gate
        stop = time.time() + (self.gate_wait if gate is not None
                              else self.busy_s)
        while time.time() < stop:
            if gone is not None and gone():
                with self.lock:
                    self.cancelled += 1
                raise Cancelled("the client left")
            if gate is not None:
                if gate.wait(0.02):
                    return
            else:
                # The step is only for the check. A turn still takes exactly
                # as long as it is set to, or every test that races one moves.
                time.sleep(min(0.02, max(0.0, stop - time.time())))
        if gate is not None:
            raise TimeoutError("the turn was never released")

    def run_turn(self, slot, prompt, predict=1, gone=None):
        """Read what the slot does not hold, then generate.

        A request that asks for no tokens generates none, and the slot is left
        holding the prompt exactly. That is what lets the request after it
        extend the slot rather than rewind into it."""
        started = time.time()
        with self.lock:
            slot.cached = shared_tokens(slot.held, prompt)
            slot.n_prompt = tokens_in(prompt)
            slot.processed = 0
            slot.decoded = 0
        try:
            try:
                self._wait_out(gone)
            except Cancelled:
                # A prompt is read from the front, so a cancelled read leaves
                # the slot holding the front of it. The stub stops halfway,
                # which is what tells a retry that starts from the abandoned
                # work apart from one that starts from nothing.
                with self.lock:
                    slot.held = prompt[:len(prompt) // 2]
                raise
            with self.lock:
                slot.processed = max(0, slot.n_prompt - slot.cached)
                asked = 1 if predict is None else predict
                slot.decoded = 0 if asked == 0 else 1
                # What is generated lands in the KV, so the next request sees
                # it. Ask for nothing and the slot holds the prompt alone.
                slot.held = prompt if asked == 0 else prompt + REPLY
                spent = time.time() - started
                count = self.counters
                # prompt_tokens_total counts what was read, not what was cached.
                count["prompt_tokens_total"] += slot.processed
                count["prompt_tokens_cached_total"] += slot.cached
                # A real backend spends seconds on a turn, not milliseconds,
                # and a rate needs enough time to divide by. Count the wall
                # time the stub actually took plus the second a turn is worth.
                count["prompt_seconds_total"] += spent + 1.0
                count["tokens_predicted_total"] += 1
                count["tokens_predicted_seconds_total"] += spent + 1.0
                count["n_tokens_max"] = max(count["n_tokens_max"], slot.n_prompt)
                return slot.cached, slot.processed
        finally:
            with self.lock:
                slot.busy = False

    # ---- the slot file ----------------------------------------------------

    def save(self, sid, filename):
        """Write this slot's cache to a file in the store's slot directory.

        The file is sparse: the size the router judges by is real, but the
        zeros behind the header cost no disk."""
        slot = self.wait_idle(sid)
        time.sleep(self.save_s)
        path = self.store.slots / filename
        header = json.dumps({"held": slot.held,
                             "tokens": tokens_in(slot.held)}).encode() + b"\n"
        with open(path, "wb") as handle:
            handle.write(header)
            if self.save_bytes > len(header):
                handle.truncate(self.save_bytes)
        written = max(self.save_bytes, len(header))
        with self.lock:
            self.saves.append(filename)
        return {"id_slot": sid, "filename": filename,
                "n_saved": tokens_in(slot.held), "n_written": written}

    def restore(self, sid, filename):
        """Read a file back into this slot."""
        path = self.store.slots / filename
        if not path.exists():
            raise FileNotFoundError(f"no such state file: {filename}")
        slot = self.wait_idle(sid)
        time.sleep(self.restore_s)
        with open(path, "rb") as handle:
            head = json.loads(handle.readline() or b"{}")
        with self.lock:
            slot.held = head.get("held", "")
            slot.cached = head.get("tokens", 0)
            slot.n_prompt = head.get("tokens", 0)
            slot.decoded = 0
            slot.processed = 0
            self.restores.append(filename)
        return {"id_slot": sid, "filename": filename,
                "n_restored": slot.cached, "n_read": path.stat().st_size}

    def erase(self, sid):
        slot = self.wait_idle(sid)
        with self.lock:
            gone = tokens_in(slot.held)
            slot.held = ""
            slot.cached = slot.n_prompt = slot.processed = slot.decoded = 0
            self.erases.append(sid)
        return {"id_slot": sid, "n_erased": gone}

    # ---- what the endpoints report ---------------------------------------

    def props(self):
        return {"total_slots": len(self.slots),
                "model_alias": self.model,
                "model_path": f"/models/{self.model}.gguf",
                "chat_template": "{# a stub #}",
                "build_info": "fake-0",
                "default_generation_settings": {
                    "id": 0, "n_ctx": self.n_ctx,
                    "params": {"n_predict": -1, "seed": 0}}}

    # A backend without patches/slots-report-the-prompt-size.patch answers
    # without n_prompt_tokens_total. Set by a test that wants the fallback.
    old_slots = False

    def slot_view(self):
        with self.lock:
            return [{"id": slot.id,
                     "id_task": slot.task,
                     "is_processing": slot.busy,
                     "n_ctx": self.n_ctx,
                     # What the slot holds: the prompt it has taken in so far
                     # plus everything generated since. It grows all through a
                     # turn, which is why the total beside it has to exist.
                     "n_prompt_tokens": slot.n_prompt + slot.decoded,
                     **({} if self.old_slots
                        else {"n_prompt_tokens_total": slot.n_prompt}),
                     "n_prompt_tokens_processed": slot.processed,
                     "n_prompt_tokens_cache": slot.cached,
                     # A list, as llama.cpp sends it (server-context.cpp,
                     # `res["next_token"] = json::array({...})`). It was a bare
                     # dict here, so the one line in the router that unwraps
                     # the real shape was never run offline: deleting it left
                     # all 464 tests green while production marked every
                     # backend down.
                     "next_token": [{"has_next_token": slot.busy,
                                     "n_decoded": slot.decoded,
                                     "n_remain": -1,
                                     "stopping_word": ""}]}
                    for slot in self.slots]

    def metrics(self):
        """The counters as Prometheus text, prefix and all.

        One line carries labels, because the router has to skip those."""
        with self.lock:
            count = dict(self.counters)
        busy = sum(1 for slot in self.slots if slot.busy)
        lines = ["# HELP llamacpp:prompt_tokens_total Prompt tokens processed.",
                 "# TYPE llamacpp:prompt_tokens_total counter"]
        for name, value in count.items():
            lines.append(f"llamacpp:{name} {value:.3f}")
        lines.append(f"llamacpp:n_busy_slots_per_decode {max(busy, 1):.3f}")
        lines.append("llamacpp:spec_decode_num_draft_tokens_total 0.000")
        lines.append("llamacpp:spec_decode_num_accepted_tokens_total 0.000")
        lines.append(f'llamacpp:kv_cache_usage_ratio{{slot="0"}} 0.100')
        return "\n".join(lines) + "\n"


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    backend = None


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-llama-server"

    def log_message(self, fmt, *args):
        pass                          # a test does not want a request log

    def do_GET(self):
        be = self.server.backend
        path, _, query = self.path.partition("?")
        path = path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        with be.lock:
            be.requests.append({"method": self.command, "path": path,
                                "query": f"?{query}" if query else "",
                                "at": time.time()})
        if path == "/v1/messages" and self._body(raw).get("stream"):
            return self._messages_stream(be, self._body(raw))
        try:
            code, payload, kind = self._answer(be, path, query, raw)
        except Cancelled:
            self.close_connection = True   # nobody is there to answer
            return
        except Exception as err:
            code = 500
            kind = JSON
            payload = json.dumps({"error": {"message": f"{err}"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_POST = do_DELETE = do_GET

    def _client_gone(self):
        """Whether the client has ended the connection, as llama.cpp asks.

        cpp-httplib polls the request socket and cancels the task when the
        other end has gone. A closed socket reads as ready with nothing on
        it."""
        try:
            ready, _, _ = select.select([self.connection], [], [], 0)
            return bool(ready) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    @staticmethod
    def _body(raw):
        try:
            return json.loads(raw or b"{}")
        except Exception:
            return {}

    def _answer(self, be, path, query, raw):
        if path == "/health":
            return 200, json.dumps({"status": "ok"}).encode(), JSON
        if path == "/props":
            return 200, json.dumps(be.props()).encode(), JSON
        if path == "/metrics":
            return 200, be.metrics().encode(), "text/plain; version=0.0.4"
        if path == "/slots" and self.command == "GET":
            return 200, json.dumps(be.slot_view()).encode(), JSON
        if path.startswith("/slots/"):
            return self._slot_action(be, path, query, self._body(raw))
        if path in ("/apply-template", "/v1/messages/apply-template"):
            with be.lock:
                be.templates += 1
            body = self._body(raw)
            messages = (anthropic_messages(body) if path.startswith("/v1/messages")
                        else body.get("messages") or [])
            # The template refuses these two, so the stub refuses them here.
            # A block the backend will never render has to fail in a test and
            # not once a client is waiting on it.
            if not messages:
                return 500, json.dumps(
                    {"error": {"message": "Jinja Exception: No messages "
                               "provided."}}).encode(), JSON
            if not closes(messages[-1]):
                return 400, json.dumps(
                    {"error": {"message": "Cannot continue an assistant "
                               "message that contains tool calls."}}).encode(), JSON
            return 200, json.dumps({"prompt": render(messages)}).encode(), JSON
        if path == "/v1/messages":
            return self._messages(be, self._body(raw))
        if path == "/completion":
            return self._completion(be, self._body(raw))
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self._chat(be, self._body(raw))
        return 404, json.dumps({"error": {"message": f"no {path}"}}).encode(), JSON

    # llama.cpp checks a slot file name before it does anything with it, and
    # a name it refuses comes back as 400. The stub has to refuse the same
    # names, or a name the router can never save under passes here and fails
    # only in production.
    ILLEGAL = set(':*?"<>|/\\') | {chr(c) for c in range(0x20)} | {chr(0x7f)}

    def _slot_action(self, be, path, query, body):
        sid = int(path.split("/")[2])
        action = dict(urllib.parse.parse_qsl(query)).get("action")
        name = body.get("filename") or ""
        # ".." and the 255 limit are the two rules fs_validate_filename adds
        # beyond the character set (common/common.cpp:830, :887). Without them
        # here a client-chosen key carrying either passed every offline test
        # and 400'd on every save in production.
        if action in ("save", "restore") and (
                not name or self.ILLEGAL & set(name)
                or ".." in name or len(name) > 255
                or name[0] == " " or name[-1] in " ."):
            return 400, json.dumps(
                {"error": {"message": "Invalid filename"}}).encode(), JSON
        if action == "save":
            if be.fail_save:
                return self._refused("the save failed")
            answer = be.save(sid, body["filename"])
        elif action == "restore":
            if be.fail_restore:
                return self._refused("the restore failed")
            answer = be.restore(sid, body["filename"])
        elif action == "erase":
            answer = be.erase(sid)
        else:
            return 400, json.dumps({"error": {"message": "no action"}}).encode(), JSON
        return 200, json.dumps(answer).encode(), JSON

    @staticmethod
    def _refused(why):
        """What a backend says when it cannot do what was asked."""
        return 500, json.dumps({"error": {"message": why}}).encode(), JSON

    def _completion(self, be, body):
        slot = be.take_slot(body.get("id_slot"))
        if slot is None:
            return 503, json.dumps({"error": {"message": "no slot"}}).encode(), JSON
        cached, read = be.run_turn(slot, body.get("prompt") or "",
                                   want_tokens(body), self._client_gone)
        with be.lock:
            be.completions.append({"slot": slot.id, "cached": cached,
                                   "read": read, "probe": is_probe(body)})
        reply = {"content": "ok", "stop": True, "id_slot": slot.id,
                 "tokens_evaluated": cached + read,
                 # cache_n as well as prompt_n: llama.cpp reports both
                 # (server-common.cpp), and the router reads both into the
                 # `read`/`reused` pair the dashboard and the cache report are
                 # built on. Without it every offline turn recorded None.
                 "timings": {"prompt_n": read, "cache_n": cached,
                             "predicted_n": 1}}
        if body.get("verbose"):
            reply["__verbose"] = {"id_slot": slot.id,
                                  "n_prompt_tokens_cache": cached}
        return 200, json.dumps(reply).encode(), JSON

    def _messages(self, be, body):
        """The anthropic endpoint, counted rather than streamed.

        The router's read pass comes here: it asks for no tokens and throws
        the answer away."""
        slot = be.take_slot(body.get("id_slot"))
        if slot is None:
            return 503, json.dumps({"error": {"message": "no slot"}}).encode(), JSON
        cached, read = self._turn(be, slot, body)
        reply = {"id": f"msg_{slot.task}", "type": "message", "role": "assistant",
                 "model": be.model, "content": [{"type": "text", "text": "ok"}],
                 "stop_reason": "end_turn", "stop_sequence": None,
                 "usage": {"cache_read_input_tokens": cached,
                           "input_tokens": read, "output_tokens": 1}}
        # No `timings` and no `__verbose`: to_json_anthropic emits neither
        # (see patches/anthropic-pass-id-slot.patch). The router's read pass
        # posts to the client's own path, so on this route it gets no read
        # counts at all - a real gap, and the stub has to show it rather than
        # invent numbers the real endpoint does not send.
        return 200, json.dumps(reply).encode(), JSON

    def _messages_stream(self, be, body):
        """The anthropic endpoint, streamed. Answers the client itself.

        A streamed reply has no length to send ahead of it, so it cannot go
        back through the counted path the other endpoints use."""
        if be.refuse:
            payload = json.dumps(
                {"error": {"message": "the template refused it"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", JSON)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return self.wfile.write(payload)
        slot = be.take_slot(body.get("id_slot"))
        if slot is None:
            payload = json.dumps({"error": {"message": "no slot"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", JSON)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return self.wfile.write(payload)
        cached, read = self._turn(be, slot, body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for name, data in anthropic_stream(be.model, cached, read):
                frame = f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(frame), frame))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True   # the router hung up, as it does when
                                           # its own client did

    def _turn(self, be, slot, body):
        """Run one anthropic turn and write it down beside the chat ones."""
        prompt = render(anthropic_messages(body))
        cached, read = be.run_turn(slot, prompt, want_tokens(body),
                                   self._client_gone)
        with be.lock:
            be.chats.append({"slot": slot.id, "cached": cached, "read": read,
                             "key": body.get("prompt_cache_key"),
                             "probe": is_probe(body)})
        return cached, read

    def _chat(self, be, body):
        slot = be.take_slot(body.get("id_slot"))
        if slot is None:
            return 503, json.dumps({"error": {"message": "no slot"}}).encode(), JSON
        prompt = render(body.get("messages") or [])
        cached, read = be.run_turn(slot, prompt, want_tokens(body),
                                   self._client_gone)
        with be.lock:
            be.chats.append({"slot": slot.id, "cached": cached, "read": read,
                             "key": body.get("prompt_cache_key"),
                             # A read pass asks for one token and for the slot
                             # id. A turn that answers a client asks for
                             # neither, so the two are never confused.
                             "probe": is_probe(body)})
        reply = {
            "id": f"chat-{slot.task}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": be.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": cached + read, "completion_tokens": 1,
                      "total_tokens": cached + read + 1},
            # llama.cpp attaches the timings to a chat reply as well as to a
            # /completion one (to_json_oaicompat_chat). The router's read pass
            # posts to the client's own path, so this is where `read` and
            # `reused` come from on the openai route - and with no timings here
            # both were None in every offline turn.
            "timings": {"prompt_n": read, "cache_n": cached, "predicted_n": 1},
        }
        if body.get("verbose"):
            # This is the only way the router learns which slot served a
            # conversation, and it cannot save a cache without knowing.
            reply["__verbose"] = {"id_slot": slot.id,
                                  "n_prompt_tokens_cache": cached}
        return 200, json.dumps(reply).encode(), JSON
