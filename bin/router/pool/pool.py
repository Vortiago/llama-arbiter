"""The slots, the pins, and everything a turn moves."""

import queue, threading, time
from collections import OrderedDict, deque
from ..backends import by_place, generates, prefills
from ..identity import copy_is_current, short_key
from ..protocol.body import common_prefix, deepest_shared, template_route
from ..settings import Tuning
from ..sizing import VISION
from ..store.backendlog import CacheWatch, read_config, read_vision
from ..store.events import EventLog
from ..store.files import adopt_files, opening_key, shelf_of, trim_openings
from ..transport import Gone
from ..backend.link import Link
from .turn import Turn
from .machine import Flow, History, Machine, per_second

def disk_summary(pins, openings, opening_bytes, tuning):
    """What the two slot directories hold against their budgets.

    `tuning` has no default. Both budgets below are the operator's to
    set, so a caller that dropped it drew one number while the sweeps
    enforced another."""
    copies = [(conv, p) for conv, p in pins.items() if p.get("parked")]
    kinds = [shelf_of(name) for name in openings.values()]
    return {"copies": {"count": len(copies),
                       "bytes": sum(p.get("bytes") or 0 for _, p in copies),
                       "budget": tuning.park_budget},
            # One budget over both shelves.
            "openings": {"count": len(openings),
                         "bytes": sum(opening_bytes.values()),
                         "budget": tuning.block_budget},
            "bases": {"count": kinds.count("base")},
            "deeps": {"count": kinds.count("deep")}}


class Pool:
    """Track free slots. Keep each conversation on one backend."""

    # Not running totals, so since_reset leaves them alone. The first five
    # are llama-server's own `gauges` (server-task.cpp). n_tokens_max sits
    # in its `counters` list but is built with std::max.
    GAUGES = ("prompt_tokens_seconds", "predicted_tokens_seconds",
              "requests_processing", "requests_deferred",
              "n_busy_slots_per_decode", "n_tokens_max")

    def __init__(self, backends, *, store, tuning=None, events=None,
                 link=None, capture_dir=None, watch=True):
        """`store` is where this run keeps its copies, `tuning` the numbers it
        was tuned to, `events` the log of what the cache decided. There is no
        default store on purpose: a Pool that made its own would make the one
        the checkout uses, and a test that forgot to pass one would write
        where a live router keeps its caches."""
        self.cv = threading.Condition()
        self.store = store
        # Where a turn writes down what the client sent, or None.
        # build() sets it from CAPTURE.
        self.capture_dir = capture_dir
        self.tuning = tuning or Tuning()
        # Off unless a caller hands in a live one, so that building a Pool
        # starts no writer thread.
        self.events = events or EventLog(on=False)
        # Every call to a backend goes through here. A test hands in a Link
        # that answers without a socket.
        self.link = link or Link(post_timeout=self.tuning.post_timeout)
        self.backends = [dict(b, slots=1, n_ctx=0, busy=0, up=False, served=0, model="",
                              stats={}, slots_detail=[], slot_prev={}, misses=0,
                              cache={}, draining=False,
                              saving=set())
                         for b in backends]
        # Each backend reports prompt cache evictions only in its own log.
        self.cache_watch = {be["name"]: CacheWatch(
            self.store.log(be["name"]),
            sink=lambda kind, value, name=be["name"]: self.events.write(
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
        # Openings being read now. A session that needs one waits for it.
        self.building = {}
        self.waiting = 0          # requests with no free slot yet
        self.waiters = {}         # ticket -> the waiting request
        # Turns read and parked, waiting for a generator slot.
        self.to_generate = 0
        self.wait_seq = 0
        self.flow = Flow(self.tuning.flow_log)
        # The last few slot files written or read.
        self.recent = deque(maxlen=self.tuning.recent_files)
        # The last few requests, with what each one started from.
        self.recent_requests = deque(maxlen=self.tuning.recent_requests)
        self.history = History(self.tuning.history_keep, self.tuning.history_step,
                               self.tuning.stall_rate)
        self.machine = Machine(gpu_poll=self.tuning.gpu_poll)
        self.loads = {}           # opening key -> times a request loaded it
        self.mounts = []          # disk usage, refreshed every mount_poll
        self.mounts_at = 0.0
        # The park worker starts on the first park, so tests start no thread.
        self.park_jobs = queue.Queue()
        self.parker = None
        if watch:
            threading.Thread(target=self._watch, daemon=True).start()

    def turn(self, ask, client):
        """Run one turn of one conversation. See pool/turn.py."""
        return Turn(self).run(ask, client)

    def adopt(self, names=None, remove=None):
        """Take over what the last run left in the slot directory."""
        remove = remove or self.store.drop
        if names is None:
            names = self.store.parked_names()
        # The order and load count each opening earned last run. Without it
        # the only order is the file mtime, which link_block sets once. An
        # opening the file does not name sorts at the back, by mtime.
        remembered = [row for row in self.store.read_openings()
                      if isinstance(row, dict) and row.get("key")]
        was = {row["key"]: rank for rank, row in enumerate(remembered)}
        self.loads.update({row["key"]: row.get("loads") or 0
                           for row in remembered})
        names.sort(key=lambda n: was.get(opening_key(n), len(was)))
        kept = self.store.read_pins()
        # Both fields: a row without a conv raised KeyError at startup.
        by_file = {row["file"]: row for row in kept
                   if isinstance(row, dict) and row.get("file") and row.get("conv")}
        openings, sizes, parked, spent = adopt_files(names, set(by_file),
                                                    store=self.store,
                                                    tuning=self.tuning)
        with self.cv:
            self.openings, self.opening_bytes = openings, sizes
            for name in parked:
                row = by_file[name]
                self.pins[row["conv"]] = {
                    # Not a live name, so recall restores the copy first.
                    "backend": "(before the restart)",
                    "slot": None, "tokens": row.get("tokens", 0),
                    "last": time.time(), "inflight": False,
                    "turns": row.get("turns", 1), "parked": name,
                    "bytes": row.get("bytes", 0),
                    # Without it every copy reads as age zero.
                    "parked_at": row.get("parked_at") or self.store.mtime(name)}
        for name in spent:
            remove(name)
        self.save_openings()      # trimmed, so write it
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

    def claim_turn(self, conv, ticket, alive=None):
        """Hold this conversation until finish_turn. One turn of it at a time.

        Returns True when the turn holds it. False means the client left and
        nothing is held. No deadline: a re-read costs more than any wait."""
        if not conv:
            return True
        began = None
        with self.cv:
            while conv in self.turns and self.turns[conv] != ticket:
                if alive is not None and not alive():
                    return False
                began = began or time.time()
                self.cv.wait(1.0)
            self.turns[conv] = ticket
            # Noted here, not in begin_wait: a waiting turn must not move the
            # row of the turn ahead, which is keyed by conversation too.
            self.flow.note(conv, "queued")
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
                waiting_on = "turn"
            elif pinned and any(be["name"] == pinned and be["up"]
                                and prefills(be) for be in self.backends):
                waiting_on = "pinned"
            elif w["tokens"] > largest:
                waiting_on = "big"
            else:
                waiting_on = "prefill"
            # `tokens` includes reply_tokens, as begin_wait was handed it. The
            # fix is for the ticket to carry prompt and room as two numbers.
            rows.append({"conv": short_key(w["conv"]), "since": w["since"],
                         "waited": round(now - w["since"], 1),
                         "tokens": w["tokens"], "waiting_on": waiting_on,
                         "images": w.get("images", 0),
                         "image_tokens": w.get("image_tokens", 0),
                         "backend": pinned if waiting_on == "pinned" else None})
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
        mount_poll."""
        if now - self.mounts_at < self.tuning.mount_poll:
            return self.mounts
        rows = self.store.disks()
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
        path = self.store.log(be["name"])
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
        text = self.link.metrics(be)
        if text is None:
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
            raw = self.link.slots(be)
            if raw is None:
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

            # Measured over rate_window, not between polls: a slot at 0.03
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
            if gap >= self.tuning.rate_window:
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
        # The sum of the slots, not /metrics' lifetime average.
        stats = be.setdefault("stats", {})
        stats["pp_live"] = round(sum(d["pp_rate"] or 0 for d in detail), 1)
        stats["tg_live"] = round(sum(d["tg_rate"] or 0 for d in detail), 1)


    def _watch(self):
        """Check each backend. Read its slot count, context size and
        counters."""
        while True:
            for be in self.backends:
                try:
                    props = self.link.props(be)
                    if props is None:
                        # Down, or too slow to answer. Same verdict either way,
                        # and the except below is where that verdict is made.
                        raise OSError("no answer from /props")
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
            time.sleep(self.tuning.poll)

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
            taken.update(be["saving"])
            # Free by both accounts first. Then merely not handed out: the
            # poll is the older of the two.
            # None rather than ids[0]: a slot a save is still reading must
            # not be handed out, or the copy lands under the wrong
            # conversation's name. acquire treats the backend as full.
            slot = next((i for i in ids if i not in taken and i not in working),
                        next((i for i in ids if i not in taken), None))
            if slot is not None and record is not None:
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
        """True if this backend is up, has a free slot, and is big enough.

        A slot a save is reading counts as busy. `busy` does not say so:
        the turn that filled it has its reply and has been released."""
        held = be["busy"] + len(be["saving"])
        return (be["up"] and not be.get("draining")
                and held < be["slots"] and tokens <= be["n_ctx"])

    def drain(self, name, deadline=None):
        """Take a backend out of service so it can be restarted. Requests wait
        in acquire rather than fail. The caches in its slots are copied out."""
        deadline = self.tuning.drain_deadline if deadline is None else deadline
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

        parked = self.park_all(only=name) if quiet else 0
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

    def acquire(self, conv, tokens, alive=None):
        """Take a slot on the backend holding this conversation.

        A busy box is a queue, not a refusal, so the wait has no deadline. It
        ends when a slot frees, when no backend can serve the request, or when
        `alive` says the client left. A pin holds for pin_patience."""
        patience = time.time() + self.tuning.pin_patience
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
                if alive is not None and not alive():
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
            while len(self.pins) > self.tuning.max_pins:
                _, dropped = self.pins.popitem(last=False)
                if dropped.get("parked"):
                    self.store.drop(dropped["parked"])   # its copy is orphaned
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

    def _free_slot(self, be):
        """A slot id on this backend that is not working, or None."""
        detail = be.get("slots_detail") or []
        if not detail:
            return 0                      # nothing reported yet, so slot 0
        for slot in detail:
            if not slot["busy"]:
                return slot["id"]
        return None

    def ensure_parked(self, be, skip_conv, remove=None):
        """Copy every cache on this backend to disk before a request lands. A
        save reads a slot, so it only works while the cache is still in one.
        Each conversation is tried once, or a save that does not stick loops
        forever."""
        remove = remove or self.store.drop
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

            self._save_park(conv, be, slot, remove)

    def _save_park(self, conv, be, slot, remove=None, timeout=None):
        """Write one cache to disk. Mark the pin only if it holds one."""
        remove = remove or self.store.drop
        with self.cv:
            be["saving"].add(slot)
        try:
            return self._park(conv, be, slot, remove, timeout)
        finally:
            with self.cv:
                be["saving"].discard(slot)
                self.cv.notify_all()

    def _park(self, conv, be, slot, remove, timeout):
        """The save itself. _save_park owns the claim on the slot."""
        name = conv + ".park"
        short = short_key(conv)
        kept = False
        written = 0
        # A short save means the slot holds somebody else. A failed save says
        # nothing about the slot: the cache is still there.
        lost = False
        refused = False
        began = time.time()
        # Which turn this save belongs to. A save can take post_timeout, and
        # the conversation's next turn can start inside that window. Clearing
        # `inflight` for the wrong turn un-reserves a slot being read.
        with self.cv:
            record = self.pins.get(conv)
            turn = record.get("turns") if record else None
        try:
            answer = self.link.save(be, slot, name, timeout) or {}
            # Ask the disk. Without
            # patches/slot-state-carries-checkpoints.patch the backend
            # reports the state without the checkpoint trailer: a 107 MB
            # file came back as 49 MB. Its figure is the fallback for a
            # stub. tests/live/test_llama_beliefs.py asserts the two agree.
            written = self.store.size(name) or (answer.get("n_written") or 0)
            kept = written >= self.tuning.park_floor
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
                # the budget sweep and off the max_pins eviction.
                if lost and mine:
                    # The slot holds someone else.
                    record["slot"] = None
            # Keep the newest copies that fit the budget, and the newest even
            # if it fills the budget alone. Newest by parked_at, not pin order:
            # by pin order a full budget dropped the copy just written, and
            # cpu1_0 wrote the same 9.45 GiB copy 1,456 times in 4.5 hours.
            held = sorted((c for c, p in self.pins.items() if p.get("parked")),
                          key=lambda c: self.pins[c].get("parked_at") or 0)
            spent, total = [], 0
            for age, name_held in enumerate(reversed(held)):
                older = self.pins[name_held]
                total += older.get("bytes") or 0
                if age and total > self.tuning.park_budget:
                    spent.append(older["parked"])
                    older["parked"] = None
            self.cv.notify_all()
        for gone in spent:
            remove(gone)
        # The map vouches for these files on the next run. Written from the
        # signal handler only, a crash or an OOM kill threw away every copy
        # this run made. A park is rare: 952 in three days.
        if kept or spent:
            self.save_pins()
        self.events.write("park", conv=short, backend=be["name"], slot=slot,
                     ok=bool(kept), bytes=written if kept else 0,
                     secs=round(time.time() - began, 2))
        if kept:
            print(f"[router] parked {short} from {be['name']} slot {slot}", flush=True)
        return kept

    def recall(self, conv, be, slot):
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
            self.link.restore(be, target_slot, name)
        except Exception as err:
            print(f"[router] {short_key(conv)} recall failed on {be['name']}: {err}",
                  flush=True)
            self.events.write("recall", conv=short_key(conv), backend=be["name"],
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
        self.events.write("recall", conv=short_key(conv), backend=be["name"],
                     slot=target_slot, ok=True, bytes=note_bytes,
                     secs=round(time.time() - began, 2))
        print(f"[router] recalled {short_key(conv)} onto {be['name']} slot {target_slot}",
              flush=True)
        return True

    def warm_prefix(self, conv, cuts, messages, system, tools, be, slot,
                    path, alive=None):
        """Load the opening this request shares into a slot on this backend,
        or read and save the one nobody has yet. Returns True when an
        opening was loaded."""
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
            unsaved = base[1] not in saved
            # Nobody has this opening: one request reads it, the others wait.
            plan = None
            if stored:
                plan = ("load", stored[1], saved[stored[1]], slot)
            elif unsaved:
                if base[1] in self.building:
                    plan = ("wait", base[1], None, None)
                else:
                    plan = ("read", base[1], None, slot)
            # Measurement only, both of these: how deep a fork could have
            # started. `shared` needs the parent to hold a slot, so the rate
            # it shows is a floor. A parked copy restores as an opening does.
            seen = set().union(*self.holds.values()) if self.holds else set()
            shared = deepest_shared(cuts, seen)
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
                                  "held": len(seen)}
            self.choices.move_to_end(conv)
            while len(self.choices) > self.tuning.recent_requests:
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
                    while len(self.forked) > self.tuning.recent_requests:
                        self.forked.popitem(last=False)
                    self.events.write("fork", conv=short_key(conv),
                                 parent=short_key(holder), depth=shared[0],
                                 cuts=len(cuts))
            # The cut keys, so an offline report can match them to copies.
            self.events.write("choice", conv=short_key(conv),
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
            # Next to the finally that pops it, so nothing that raises can
            # sit between. A key left behind makes every later conversation
            # with this system prompt wait build_patience for nothing.
            if plan[0] == "read":
                self.building[plan[1]] = time.time()

        if plan[0] == "load":
            return self._load_prefix(plan[1], plan[2], be, plan[3])
        if plan[0] == "wait":
            return self._wait_for_opening(plan[1], be, slot, alive)
        try:
            return self._read_prefix(base, messages, system, tools, be,
                                     plan[3], path)
        finally:
            with self.cv:
                self.building.pop(plan[1], None)
                self.cv.notify_all()

    def _wait_for_opening(self, key, be, slot, alive=None):
        """Wait for another request to save the opening, then load it.
        Measured: five sessions starting together read 92,000 tokens where
        24,000 would do, and the last finished after seventeen minutes."""
        deadline = time.time() + self.tuning.build_patience
        with self.cv:
            while key in self.building and time.time() < deadline:
                if alive is not None and not alive():
                    return False    # this wait holds a prefill slot
                self.cv.wait(1.0)
            name = self.openings.get(key)
            if not name:
                return False        # it failed or timed out, so read it here
        return self._load_prefix(key, name, be, slot)

    def park_all(self, timeout=None, only=None, budget=None):
        """Copy live caches to disk, so a stop does not throw them away. `only`
        names one backend, for a drain. A conversation mid-turn is skipped:
        its slot is busy. Each conversation is tried once, or a refused save
        loops forever. A full disk at SIGTERM spun here."""
        timeout = self.tuning.post_timeout if timeout is None else timeout
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
                            and not copy_is_current(record)]
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
                if self._save_park(conv, be, slot, timeout=each):
                    parked += 1
        return parked

    def save_openings(self):
        """Write down what each opening has earned, for the next run: the
        shelf order and the load counts."""
        with self.cv:
            rows = [{"key": key, "file": name, "loads": self.loads.get(key, 0)}
                    for key, name in self.openings.items()]
        self.store.write_openings(rows)
        return len(rows)

    def save_pins(self):
        """Write down whose cache each copy holds, for the next run."""
        with self.cv:
            kept = [{"conv": conv, "file": record["parked"],
                     "tokens": record.get("tokens", 0),
                     "bytes": record.get("bytes", 0),
                     "turns": record.get("turns", 1),
                     # The budget sweep orders by this.
                     "parked_at": record.get("parked_at")}
                    for conv, record in self.pins.items() if record.get("parked")]
        self.store.write_pins(kept)
        return len(kept)

    def hand_off(self, conv, source, tokens, remove=None, alive=None):
        """Move a conversation to the backend it generates on.

        The prefiller is released before the wait to generate. The other
        order left three prefillers idle for seven minutes on one reply.
        Between the save and the restore the conversation is parked, so a
        failure leaves it parked, not lost.

        Three ways out, and they say different things about who holds
        `source`:
          `source`  nothing was carried, and the caller still holds it
          None      the cache is parked and `source` is already released
          Gone      the client left before any of that. Nothing is parked,
                    the caller still holds `source`, and its ending is the
                    one that releases it and parks what the read got through
        """
        remove = remove or self.store.drop
        if not self.tuning.handoff:
            return self._stay(source, "the handoff is turned off")
        target = self.generator(tokens)
        while target is None and not generates(source):
            # Nothing to carry this to, and the instance holding it does
            # not generate. Wait, holding a prefill slot.
            if alive is not None and not alive():
                # The client leaving mid-turn, which the caller's ending
                # already finishes. The give-up below is not: past the save
                # the cache is on disk and only the reply is lost.
                raise Gone("while it waited for a slot to generate in")
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

        if not self._save_park(conv, source, slot, remove):
            return self._stay(source, "the slot had already changed hands")
        with self.cv:
            name = self.pins[conv]["parked"]
            written = self.pins[conv].get("bytes") or 0
            source["busy"] -= 1            # the reader takes the next prompt
            self.flow.note(conv, "generate-queue")
            self.cv.notify_all()

        while True:
            # None once the last generator went away.
            free = None if target is None else self._wait_to_generate(target, alive)
            if free is not None:
                break
            if alive is not None and not alive():
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
            self.link.restore(target, free, name)
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
        self.events.write("migrate", conv=short_key(conv), src=source["name"],
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

    def _wait_to_generate(self, target, alive):
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
                    if alive is not None and not alive():
                        return None
                    self.cv.wait(1.0)
        finally:
            with self.cv:
                self.to_generate -= 1

    def park_later(self, be, conv, ticket, remove=None):
        """Copy a cache out of a backend that cannot read it, on a worker: the
        copy runs to gigabytes and the client already has its reply.
        `inflight` reserves the slot before this returns. The turn ticket
        goes with the job, because the copy overwrites the file the next turn
        restores from. Returns False, with the ticket still the caller's,
        when there is nothing to copy."""
        remove = remove or self.store.drop
        if prefills(be):
            return False              # it can be read again here
        with self.cv:
            record = self.pins.get(conv) if conv else None
            if not record or record["slot"] is None:
                return False
            if record["inflight"]:
                return False               # a save is running: two of them
                                           # write the same file at once
            slot = record["slot"]
            record["inflight"] = True      # hold it still while it copies
            if self.parker is None:
                self.parker = threading.Thread(target=self._run_parks,
                                               name="park", daemon=True)
                self.parker.start()
        self.park_jobs.put((conv, be, slot, remove, ticket))
        return True

    def _run_parks(self):
        """Write the queued copies. One worker: two multi-gigabyte writes at
        once only divide the same disk."""
        while True:
            job = self.park_jobs.get()
            conv, be, slot, remove, ticket = job
            try:
                self._save_park(conv, be, slot, remove)
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

    def park_partial(self, conv, be, slot, remove=None):
        """Keep what an abandoned read got through, so the retry starts there.
        Thrown away, a prompt too long for the client's patience is never
        read: every attempt gives up in the same place. Measured once at a
        114,354 token turn, abandoned twice after an hour, two thirds read
        each time."""
        remove = remove or self.store.drop
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
            if record["inflight"]:
                return False               # a save is running: two of them
                                           # write the same file at once
            record["inflight"] = True      # hold it still while it copies
        return self._save_park(conv, be, slot, remove)

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

    def forget_stale_park(self, conv, cuts, remove=None):
        """Drop a copy whose opening the client has changed since. llama.cpp
        restores the state, finds no checkpoint before the point where the
        prompts part, and reads everything again. Measured once at 117,847
        tokens for prompts that parted at token 503. Without the copy the
        conversation loads a shared opening instead."""
        remove = remove or self.store.drop
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

    def _load_prefix(self, key, name, be, slot):
        """Put a saved opening back into a slot."""
        began = time.time()
        try:
            self.link.restore(be, slot, name)
        except Exception as err:
            print(f"[router] opening {key[:8]} failed to load on "
                  f"{be['name']}: {err}", flush=True)
            self.events.write("load", key=short_key(key), backend=be["name"],
                         slot=slot, ok=False, error=str(err)[:120])
            return False
        # Sized: the dashboard draws each file event over its byte count.
        read = self.store.size(name)
        with self.cv:
            shelf = None
            if key in self.openings:
                self.openings.move_to_end(key)  # in use, so keep it longest
                shelf = shelf_of(self.openings[key])
            self.loads[key] = self.loads.get(key, 0) + 1
            self.note_file("loaded opening", key, be, slot, read)
            loads = self.loads[key]
        self.save_openings()      # a load earns an opening its place
        self.events.write("load", key=short_key(key), shelf=shelf,
                     backend=be["name"], slot=slot, ok=True,
                     bytes=read, secs=round(time.time() - began, 2),
                     loads=loads)
        print(f"[router] loaded opening {key[:8]} onto {be['name']} "
              f"slot {slot}", flush=True)
        return True

    def _read_prefix(self, cut, messages, system, tools, be, slot, path):
        """Read one opening into a slot, then keep a copy of the slot."""
        index, key = cut
        name = f"base-{key}.park"
        began = time.time()
        self.store.link_block(name)   # so the save lands on the faster disk
        try:
            block = self._render_block(system, tools, messages[:index + 1],
                                       be, self.link, path)
            # The zero-token reply's timings: tokens processed and cached.
            read = self.link.prefill(be, block, slot,
                                     self.tuning.read_timeout) or {}
            answer = self.link.save(be, slot, name) or {}
        except Exception as err:
            print(f"[router] opening {key[:8]} failed to save on "
                  f"{be['name']}: {err}", flush=True)
            self.store.drop(name)  # take back the link made before the read
            self.events.write("build", key=short_key(key), shelf="base",
                         backend=be["name"], slot=slot, ok=False,
                         error=str(err)[:120],
                         secs=round(time.time() - began, 1))
            return False
        # From the disk, as _save_park: unpatched backends under-report.
        written = self.store.size(name) or (answer.get("n_written") or 0)
        if written < self.tuning.park_floor:
            self.store.drop(name)  # the slot had already changed hands
            self.events.write("build", key=short_key(key), shelf="base",
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
                                    keep=set(self.building),
                                    budget=self.tuning.block_budget)
        for extra in dropped:
            self.store.drop(extra)
        self.save_openings()
        timing = read.get("timings") or {}
        self.events.write("build", key=short_key(key), shelf="base",
                     backend=be["name"], slot=slot, ok=True,
                     secs=round(time.time() - began, 1),
                     bytes=written,
                     prompt_n=timing.get("prompt_n"),
                     cache_n=timing.get("cache_n"))
        print(f"[router] read and kept opening {key[:8]} on {be['name']} "
              f"slot {slot}", flush=True)
        return True

    @staticmethod
    def _render_block(system, tools, head, be, link, path):
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
        full = link.render(be, route,
                           dict(extra,
                                messages=opening + [{"role": "user",
                                                     "content": "x"}]))
        alone = link.render(be, route, dict(extra, messages=opening))
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
                        "deeps": shelf("deep", "shared history")}
            # `slot` is set only while the slot still holds the cache, or
            # three copies naming one single-slot backend look like three
            # caches in one slot. `parked_at` is the park_budget sweep order.
            copies = [{"name": p["parked"], "kind": "copy", "conv": short_key(conv),
                       "bytes": p.get("bytes") or 0, "backend": p["backend"],
                       "slot": p.get("slot"), "parked_at": p.get("parked_at")}
                      for conv, p in self.pins.items() if p.get("parked")]
            disk = disk_summary(self.pins, self.openings,
                                self.opening_bytes, self.tuning)
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
