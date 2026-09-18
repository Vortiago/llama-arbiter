"""The public port: what it serves and what it refuses."""

import http.client, http.server, json, os, select, socket, threading, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
from ..identity import client_kind, conversation_id, prompt_key, session_key, short_key
from ..pool.pool import capture, how_started
from ..protocol.body import hoist_system, prompt_cuts, read_only, request_shape, wants_stream
from ..protocol.splice import AnthropicSplice, OaiUsageSplice, wants_usage, with_usage
from ..protocol.sse import _say, anthropic, opening_event, ping_for, sse_event, wants_ping
from ..sizing import request_cost
from ..transport import Gone, said_in
from .config import CONFIG_FILES, client_config, host_only

def passed_paths(env=None):
    """The paths a client may reach through the router, for this run.

    PASS_THROUGH adds to the fixed list. The backends run with --agent, which
    is shell and file access with no key, so a path not named here must never
    be reachable from the public port.
    """
    env = os.environ if env is None else env
    return PASSED | {p.strip() for p in env.get("PASS_THROUGH", "").split(",")
                     if p.strip()}


# Endpoints that use a slot.
INFERENCE = {
    "/completion", "/completions", "/v1/completions",
    "/chat/completions", "/v1/chat/completions",
    "/infill", "/v1/messages", "/responses", "/v1/responses",
    "/embedding", "/embeddings", "/v1/embeddings",
}


# An allowlist. The backends run with --agent, which is shell and file
# access with no key. A path not named here must never be reachable from
# the public port. PASS_THROUGH adds to it, comma separated.
PASSED = {
    "/health", "/props", "/slots", "/models", "/v1/models",
    "/tokenize", "/detokenize", "/apply-template",
    "/v1/messages/count_tokens", "/v1/messages/apply-template",
}


DROP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "te", "trailers",
                "transfer-encoding", "upgrade", "content-length", "host"}


# The dashboard, a static app. bin/router/web/handler.py -> bin/web:
# this package's own web/ is the port, that one is the page.
WEB = Path(__file__).resolve().parents[2] / "web"


MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",  ".json": "application/json",
        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".map": "application/json"}


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
            return self._send(200, json.dumps(self.server.pool.status(), indent=2).encode())
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
                self.server.pool.reset_rates(True)
                return self._send(200, json.dumps(
                    {"rates_since": self.server.pool.rates_since}).encode())
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
            done = self.server.pool.resume(name)
            return (self._send(200, json.dumps({"backend": name, "serving": True}).encode())
                    if done else self._error(404, f"no backend called {name}"))
        report = self.server.pool.drain(name)
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
        with self.server.pool.cv:
            up = [be for be in self.server.pool.backends if be["up"]]
            model = next((be["model"] for be in up if be["model"]), "qwen")
            n_ctx = min([be["n_ctx"] for be in up if be["n_ctx"]], default=150000)
        config = client_config(kind, host, model, n_ctx,
                               self.server.provider)
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
                payload = json.dumps(self.server.pool.status())
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
        live = [be for be in self.server.pool.backends if be["up"]]
        if not live:
            return self._error(503, "no backend is up")

        link = self.server.pool.link
        if path == "/slots":
            slots = []
            for be in live:
                part = link.slots(be, timeout=5)
                if isinstance(part, list):
                    for slot in part:
                        slot["backend"] = be["name"]
                        slots.append(slot)
            return self._send(200, json.dumps(slots).encode())

        props = link.props(live[0], timeout=5)
        if props is None:
            return self._error(502, f"{live[0]['name']} did not answer /props")
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
        if length > self.server.pool.tuning.max_body:
            self.close_connection = True
            return self._error(413, f"body of {length} bytes; this router "
                                    f"reads at most {self.server.pool.tuning.max_body}")
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
            # An allowlist, see passed_paths.
            if path not in self.server.passed:
                return self._error(404, f"this router does not serve {path}")
            be = next((b for b in self.server.pool.backends if b["up"]), None)
            if not be:
                return self._error(503, "no backend is up")
            return self._forward(be, body)

        vision = self.server.pool.vision()
        tokens, images, image_charge = request_cost(body, vision)
        # `tokens` carries reply_tokens of room. The dashboard measures a
        # turn against the prompt sent: 1,024 tokens nobody sent is 41
        # seconds of reading nobody did.
        prompt_tokens = max(0, tokens - self.server.pool.tuning.reply_tokens)
        largest = self.server.pool.largest()
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
        capture(self.server.capture_dir, conv, body)
        # This model's template refuses a late system message.
        ordered = hoist_system(body)
        if ordered is not body:
            print(f"[router] a late system message became a user message "
                  f"for {path}", flush=True)
            self.server.pool.events.write("start_over", conv=short_key(conv) if conv else None,
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

        ticket = self.server.pool.begin_wait(conv, tokens, images, image_charge)
        try:
            # The turn ahead holds the pin, slot and copy this one needs.
            mine = self.server.pool.claim_turn(conv, ticket, self._still_there)
            be = self.server.pool.acquire(conv, tokens, self._still_there) if mine else None
        finally:
            self.server.pool.end_wait(ticket)
        waited = time.time() - start
        if not be:
            # `done` only if this turn held the conversation: Flow is keyed
            # by conversation, and a turn that gave up in claim_turn deleted
            # the live row of the turn running.
            if mine:
                self.server.pool.note_stage(conv, "done")
            self.server.pool.finish_turn(conv, ticket)
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
            warm = bool(conv) and self.server.pool.holds_slot(conv)
            # One slot, decided once, for the read below to extend.
            slot = self.server.pool.pick_slot(be, conv)
            self.server.pool.note_stage(conv, "prefill", be["name"], slot)
            # Nothing reaches the backend until the caches on it are on disk.
            self.server.pool.ensure_parked(be, conv)
            # A copy with an opening the client no longer sends is no prefix.
            if self.server.pool.forget_stale_park(conv, cuts):
                self.server.pool.events.write("start_over", conv=short_key(conv) if conv else None,
                             reason="stale_copy", client=client, path=path)
            recalled = self.server.pool.recall(conv, be, slot)
            loaded = (not recalled
                      and self.server.pool.warm_prefix(conv, cuts, messages,
                                                       system, tools, be, slot,
                                                       path))
            if asked is not None:
                # Watch the client. The timings say what the cache saved.
                answer = self.server.pool.link.read(
                    be, path, read_only(body, slot),
                    self._still_there, self.server.pool.tuning.read_timeout)
                timing = (answer or {}).get("timings") or {}
                read_stats = {"read_prompt_n": timing.get("prompt_n"),
                              "read_cache_n": timing.get("cache_n")}
                self.server.pool.note_slot(conv, slot)
                serving = self.server.pool.hand_off(
                    conv, be, tokens, wanted=self._still_there)
                if serving is None:
                    # The cache is parked, and no backend is held.
                    raise Gone("the client stopped waiting for a slot to generate in")
            if serving is be:
                # Nothing was carried. hand_off already noted a carried turn.
                self.server.pool.note_stage(conv, "generate", be["name"], slot)
            if stop_ping:
                stop_ping()                    # waits for a ping in flight
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
                    self.server.pool.note_holds(conv, serving, cuts)
                    self.server.pool.release(serving, conv)
                # After the release: these write the copy that is not behind.
                if left:
                    self.server.pool.park_partial(conv, be, slot)
                # A backend that does not read cannot serve the next turn, so
                # leave a copy for one that does, on a worker.
                if serving is not None:
                    parking = self.server.pool.park_later(serving, conv, ticket)
            except Exception as err:
                # A ticket not given back costs the conversation every later
                # turn: claim_turn has no deadline. The lines below must run.
                print(f"[router] {short_key(conv)} could not be put away: "
                      f"{err}", flush=True)
            took = time.time() - start
            self.server.pool.note_stage(conv, "done")
            # The worker ends the turn once the copy has landed.
            if not parking:
                self.server.pool.finish_turn(conv, ticket)
            self.server.pool.note_request(conv, be, path, took, waited,
                              how_started(warm, recalled, loaded), prompt_tokens,
                              images=images, image_tokens_=image_charge,
                              **read_stats)
            self.server.pool.events.write("request", conv=short_key(conv) if conv else None,
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
            while not stop.wait(self.server.pool.tuning.ping_every):
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
        try:
            upstream = self.server.pool.link.open(
                be, target, data, headers, self.command,
                self.server.pool.tuning.forward_timeout)
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
                    if time.time() - last[0] < self.server.pool.tuning.ping_every:
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
                    self.server.pool.events.write("usage", backend=be["name"],
                                 conv=short_key(conv) if conv else None,
                                 path=self.path.split("?")[0],
                                 **{short: splice.reported[full]
                                    for full, short in names.items()
                                    if isinstance(splice.reported.get(full), int)})
                if oai_tee and oai_tee.usage:
                    details = oai_tee.usage.get("prompt_tokens_details") or {}
                    self.server.pool.events.write("usage", backend=be["name"],
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
