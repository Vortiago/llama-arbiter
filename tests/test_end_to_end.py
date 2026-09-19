"""End-to-end tests for the router.

test_migration.py replaces every HTTP call with a stub. These drive the real
Pool, and the real Handler where that is the only way in, against stub backends
on real sockets, so the threads, the timing and the files are real.

Offline and deterministic: every slow step in the stub is measured in
milliseconds, so the file runs in a few seconds.
"""
import atexit
import http.client
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pathlib
from dataclasses import replace
import router

class SANDBOX:
    """What this test run is wired to.

    A Pool is handed its store, its tuning and its event log, so there is no
    module state to redirect and nothing to put back. One sandbox serves the
    whole file, under the temporary directory, and a case wanting its own
    builds another. Without this a case would reach the checkout's own
    run/slots, where a live router keeps parked copies worth hundreds of
    gigabytes and a stray pins.json tells adopt() to delete every copy it does
    not name.
    """

    store = router.Store(tempfile.mkdtemp(prefix="router-run-"))
    tuning = router.Tuning()
    events = router.EventLog(on=False)

# Nothing else deletes this. The path is read now rather than at exit, because
# a case may point SANDBOX.store somewhere else and put it back.
atexit.register(shutil.rmtree, SANDBOX.store.run, ignore_errors=True)


def make_pool(backends, **kw):
    """A Pool wired to the sandbox. A case that wants another store, tuning or
    event log passes it, and that one wins."""
    kw.setdefault("store", SANDBOX.store)
    kw.setdefault("tuning", SANDBOX.tuning)
    kw.setdefault("events", SANDBOX.events)
    return router.Pool(backends, **kw)
from fake_backend import FakeBackend

# A system prompt long enough that prompt_cuts names a cut in it. Below
# Tuning.prefix_min_chars the router writes down no opening at all.
LONG_SYSTEM = "You follow these rules. " * 400          # about 9600 characters

# How long a test waits on another thread. Generous, because it is reached
# only when the test is about to fail anyway.
PATIENCE = 10.0

# A read pass that outlasts a poll, as every real one does. A test that read
# faster than the router looks would prove nothing about production.
READ_MS = 150

POOL_LOOPS = ("_watch", "_builder")


class Bomb:
    """Stop a pool's daemon loops.

    The loops run `while True` and have no off switch. A SystemExit raised
    inside one ends that thread quietly, and `except Exception` does not catch
    it. Standing in for the condition variable stops _watch; standing in for
    build_once stops the builder."""

    def __enter__(self):
        raise SystemExit

    def __exit__(self, *rest):
        return False

    def __call__(self, *args, **kw):
        raise SystemExit


def pool_threads():
    """The pool loop threads that are alive now.

    A thread carries the name of its target, which is what tells one of these
    apart from a request thread."""
    return [t for t in threading.enumerate()
            if any(loop in t.name for loop in POOL_LOOPS) and t.is_alive()]


def wait_for(check, patience=PATIENCE, step=0.02):
    """Wait until check() is true. Return whether it became true."""
    stop = time.time() + patience
    while time.time() < stop:
        if check():
            return True
        time.sleep(step)
    return bool(check())


def sse_events(reply, most=None):
    """The events on an SSE stream, as (name, data) pairs.

    Reads until the stream ends, or until `most` events have arrived. A
    comment carries no event and no data, so it is passed over: it keeps a
    stream alive and says nothing to the protocol on top of it."""
    events, name, data = [], None, None
    for raw in iter(reply.readline, b""):
        line = raw.strip()
        if line.startswith(b"event:"):
            name = line[6:].strip().decode()
        elif line.startswith(b"data:"):
            try:
                data = json.loads(line[5:].strip())
            except ValueError:
                data = None
        elif not line and (name or data is not None):
            events.append((name, data))
            name, data = None, None
            if most and len(events) >= most:
                break
    return events


class EndToEnd(unittest.TestCase):
    """A temporary slot directory, stub backends, and a real Pool.

    Nothing here writes into the project's own run directory, and nothing is
    left running when a test ends."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="router-e2e-"))
        (self.root / "slots").mkdir()
        self.kept = {name: getattr(SANDBOX, name) for name in
                     ("store", "tuning")}
        # No <name>.log is written here on purpose. CacheWatch has to survive
        # a backend whose log it cannot find.
        SANDBOX.store = router.Store(self.root)
        # The stub answers at once, so polling fast is free. Every loop has to
        # come round quickly, because that is also how the cleanup stops it.
        # A pin worth 20 seconds in production is worth a fraction of one
        # here. handoff is on whatever the shipped default is, because these
        # tests are about the move.
        SANDBOX.tuning = replace(SANDBOX.tuning, poll=0.05, build_poll=0.05,
                                pin_patience=0.3, park_all_timeout=5.0,
                                handoff=True)

        self.stubs = []
        self.pools = []
        self.servers = []
        self.helpers = []
        self.addCleanup(self.stop_everything)

    # ---- setup helpers ----------------------------------------------------

    def stub(self, name, pref, prefill=True, generate=True, **kw):
        """Start one stub backend and return the spec the Pool takes."""
        kw.setdefault("store", SANDBOX.store)
        kw.setdefault("park_floor", SANDBOX.tuning.park_floor)
        backend = FakeBackend(name=name, **kw)
        self.stubs.append(backend)
        setattr(self, name, backend)
        return {"name": name, "url": backend.url, "pref": pref,
                "prefill": prefill, "generate": generate}

    def pool(self, specs, watch=True):
        """Build a Pool and wait until it has seen every backend."""
        made = make_pool(specs, watch=watch)
        self.pools.append(made)
        if watch:
            self.assertTrue(
                wait_for(lambda: all(b["up"] for b in made.backends)),
                "the pool never saw the stub backends come up")
        return made

    def serve(self, pool):
        """Put the real Handler in front of this pool. Returns its base url."""
        server = router.Server(("127.0.0.1", 0), router.Handler)
        # The Handler reads the pool off its own server, so two servers in one
        # process cannot take each other's.
        server.pool = pool
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}"

    # ---- driving a conversation ------------------------------------------

    @staticmethod
    def body(conv, text="hello", system="You are helpful.", tools=None):
        return json.dumps({
            "model": "fake-model",
            # The router names the conversation from this, exactly as it does
            # for OpenCode. No guessing from the prompt.
            "prompt_cache_key": conv,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": text}],
            **({"tools": tools} if tools else {}),
        }).encode()

    def turn(self, url, conv, text="hello", system="You are helpful.",
             tools=None, timeout=30):
        """One conversation turn, through the router."""
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=self.body(conv, text, system, tools),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)

    def start_turn(self, url, conv, **kw):
        """Start a turn in the background. Returns a box for its result."""
        box = {}

        def run():
            try:
                box["reply"] = self.turn(url, conv, **kw)
            except Exception as err:          # recorded, so a test can see it
                box["error"] = err

        thread = threading.Thread(target=run, name=f"turn-{conv}", daemon=True)
        self.helpers.append(thread)
        thread.start()
        return box

    def background(self, work, name="helper"):
        """Run something in a thread the cleanup will wait for."""
        box = {}

        def run():
            try:
                box["value"] = work()
            except Exception as err:
                box["error"] = err

        thread = threading.Thread(target=run, name=name, daemon=True)
        self.helpers.append(thread)
        thread.start()
        return box

    def backend(self, pool, name):
        return next(b for b in pool.backends if b["name"] == name)

    # ---- teardown ---------------------------------------------------------

    def stop_everything(self):
        for stub in self.stubs:
            stub.release()                # nothing may stay blocked on a gate
        stuck = []
        for helper in self.helpers:
            helper.join(PATIENCE)
            if helper.is_alive():
                stuck.append(helper.name)
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(PATIENCE)
        # A loop only meets its bomb when it comes round, so shorten the
        # wait whatever the test had set it to.
        for pool in self.pools:
            # The pool took its tuning when it was built, so shortening the
            # sandbox's reaches no running loop. Shorten each pool's own, and
            # do it before the bombs go in.
            pool.tuning = replace(pool.tuning, poll=0.02, build_poll=0.02)
            pool.build_once = Bomb()
            pool.cv = Bomb()
        alive = not wait_for(lambda: not pool_threads(), patience=5.0)
        for stub in self.stubs:
            stub.stop()
        for name, value in self.kept.items():
            setattr(SANDBOX, name, value)
        shutil.rmtree(self.root, ignore_errors=True)
        self.assertEqual(stuck, [], "a test thread never finished")
        self.assertFalse(alive, f"pool threads outlived the test: {pool_threads()}")


class NewConversationsAreNeverReadWhereReadsIsOff(EndToEnd):
    """A backend with `reads` off does not prefill, whatever its hardware.

    It does generate, which is the whole point of the handoff. So what stays
    off it is the prompt, not the conversation. The fixture calls it "gpu"
    because that is what this box happens to run there; the rule is the
    setting."""

    def setUp(self):
        super().setUp()
        self.trio = self.pool([self.stub("gpu", 0, prefill=False),
                               self.stub("cpu", 1),
                               self.stub("cpu2", 2)])
        url = self.serve(self.trio)
        for n in range(4):
            self.turn(url, f"fresh-{n}")

    def test_every_prompt_is_read_on_a_cpu_instance(self):
        # The gpu has to be up, or this proves only that it was down.
        self.assertTrue(self.backend(self.trio, "gpu")["up"])
        self.assertEqual(self.gpu.probes, [], "a prompt was read on the gpu")
        self.assertEqual(len(self.cpu.probes) + len(self.cpu2.probes), 4)

    def test_it_generates_without_ever_being_given_a_prompt(self):
        self.assertTrue(self.gpu.answers, "nothing generated there at all")
        for answer in self.gpu.answers:
            self.assertEqual(answer["read"], 0, "a prompt was read where "
                                                "`reads` is off")
            self.assertGreater(answer["cached"], 0, "it had nothing to extend")


class RecallFires(EndToEnd):
    """A parked cache goes back on whichever backend ends up serving it."""

    def test_a_parked_cache_is_restored_onto_the_backend_that_serves_it(self):
        pool = self.pool([self.stub("cpu", 1), self.stub("cpu2", 2)])
        url = self.serve(pool)

        self.turn(url, "early")
        # Nothing here generates anywhere but where it read: no backend in this
        # pool is a generating one, so the cache ends where the prompt was read.
        home = pool.pins["early"]["backend"]
        mine = self.cpu if home == "cpu" else self.cpu2
        other = self.cpu2 if home == "cpu" else self.cpu
        self.assertEqual(len(mine.answers), 1)

        # Somebody else takes early's slot, so its cache has to go to disk.
        # The other backend is out of service while that happens, so there is
        # nowhere else the filler could land.
        pool.drain(other.name, deadline=PATIENCE)
        mine.hold()
        self.start_turn(url, "filler")
        self.assertTrue(wait_for(lambda: self.backend(pool, home)["busy"] == 1),
                        "the filler never took early's backend")
        self.assertTrue(wait_for(lambda: pool.pins["early"].get("parked")),
                        "early's cache was never parked")
        self.assertEqual(pool.pins["early"]["parked"], "early.park")

        # early comes back. Its own backend is still held, so it takes the
        # other one and its cache has to follow it there.
        pool.resume(other.name)
        self.turn(url, "early")
        self.assertEqual(other.restores, ["early.park"])
        self.assertIn("early", [c["key"] for c in other.answers],
                      "early did not answer where its cache was restored")
        # The read pass is the one that would have read the prompt again, so
        # it is the one that has to find the cache waiting for it.
        self.assertGreater(other.probes[-1]["cached"], 0,
                           "the second backend served it without the cache")
        mine.release()


class ParkBeforeTheNewcomer(EndToEnd):
    """A cache is copied out before anything can displace it."""

    def test_the_resident_cache_is_copied_out_before_the_newcomer_is_admitted(self):
        pool = self.pool([self.stub("cpu", 1)])
        url = self.serve(pool)
        self.turn(url, "resident")
        self.turn(url, "newcomer")

        paths = [row["path"] + row["query"] for row in self.cpu.requests]
        save = paths.index("/slots/0?action=save")
        chats = [n for n, path in enumerate(paths) if path == "/v1/chat/completions"]
        # Two calls a turn now: the read pass, then the turn that answers.
        self.assertEqual(len(chats), 4)
        self.assertLess(chats[1], save, "the save came before the first turn")
        self.assertLess(save, chats[2], "the newcomer went in before the save")
        self.assertEqual(pool.pins["resident"]["parked"], "resident.park")
        self.assertTrue((SANDBOX.store.slots / "resident.park").exists())

    def test_nothing_is_parked_when_the_slot_had_already_changed_hands(self):
        """A short file means the slot holds someone else. It is not a cache."""
        pool = self.pool([self.stub("cpu", 1, save_bytes=1024)])
        url = self.serve(pool)
        self.turn(url, "resident")
        self.turn(url, "newcomer")

        self.assertIsNone(pool.pins["resident"]["parked"])
        self.assertIsNone(pool.pins["resident"]["slot"])
        self.assertFalse((SANDBOX.store.slots / "resident.park").exists())


class DrainUnderLoad(EndToEnd):
    """A drain waits for work already running, then copies the caches out."""

    def test_a_drain_waits_parks_and_sends_new_work_to_the_other_backend(self):
        pool = self.pool([self.stub("cpu", 1), self.stub("cpu2", 2)])
        url = self.serve(pool)

        # Reading takes pref backwards, so a prompt with no cache anywhere is
        # read on cpu2. That is the backend with work on it to drain.
        self.cpu2.hold()
        self.start_turn(url, "inflight")
        self.assertTrue(wait_for(lambda: self.backend(pool, "cpu2")["busy"] == 1),
                        "the turn never reached cpu2")

        report = self.background(lambda: pool.drain("cpu2",
                                                    deadline=PATIENCE),
                                 name="drain")
        time.sleep(0.4)
        self.assertEqual(report, {}, "the drain did not wait for the request")
        self.assertTrue(self.backend(pool, "cpu2")["draining"])

        # New work must not queue behind the drain.
        self.turn(url, "elsewhere")
        self.assertEqual(len(self.cpu.answers), 1)

        self.cpu2.release()
        self.assertTrue(wait_for(lambda: "value" in report),
                        "the drain never finished")
        # `left` is what is still only in a slot: a backend with any of those
        # must not be stopped, and the http answer is a 409 rather than a 200.
        self.assertEqual(report["value"], {"backend": "cpu2", "quiet": True,
                                           "parked": 1, "left": 0})
        self.assertEqual(self.cpu2.saves, ["inflight.park"])
        self.assertTrue((SANDBOX.store.slots / "inflight.park").exists())

    def test_resume_puts_the_backend_back_in_service(self):
        pool = self.pool([self.stub("cpu", 1), self.stub("cpu2", 2)])
        url = self.serve(pool)
        pool.drain("cpu", deadline=PATIENCE)
        self.turn(url, "while-drained")
        self.assertEqual(len(self.cpu2.answers), 1)

        # Back in service means work can land there again. Reading takes pref
        # backwards, so cpu2 goes out of service to leave only one answer.
        self.assertTrue(pool.resume("cpu"))
        pool.drain("cpu2", deadline=PATIENCE)
        self.turn(url, "after-resume")
        self.assertEqual(len(self.cpu.answers), 1)


class ShutdownRoundTrip(EndToEnd):
    """park_all, save_pins, and a fresh Pool that picks the copies up."""

    def setUp(self):
        super().setUp()
        self.spec = [self.stub("cpu", 1)]
        self.first = self.pool(self.spec)
        self.url = self.serve(self.first)
        self.turn(self.url, "survivor")
        # The reply reaches the client before the handler has finished with the
        # turn, and park_all reads the bookkeeping the handler is still
        # writing. Waiting for the pin to be free is waiting for the turn to be
        # over, which is what a shutdown would be doing anyway.
        self.assertTrue(
            wait_for(lambda: not self.first.pins["survivor"]["inflight"]),
            "the turn was never finished with")
        # A file the pin map does not vouch for. Nothing knows whose cache it
        # is, so the next run must throw it away.
        (SANDBOX.store.slots / "orphan.park").write_bytes(b"nobody claims this")
        self.parked = self.first.park_all()
        self.kept_pins = self.first.save_pins()

    def second_pool(self):
        """A fresh Pool over the same stubs, as a restart would build.

        The servers already running are pointed at it, because that is what a
        restart does: the port comes back in front of a new pool."""
        pool = self.pool(self.spec)
        pool.adopt()
        for server, _ in self.servers:
            server.pool = pool
        return pool

    def test_the_copies_survive_and_the_pin_map_names_them(self):
        self.assertEqual((self.parked, self.kept_pins), (1, 1))
        self.assertTrue((SANDBOX.store.slots / "survivor.park").exists())
        kept = json.loads((SANDBOX.store.slots / "pins.json").read_text())
        self.assertEqual([row["conv"] for row in kept], ["survivor"])
        self.assertEqual(kept[0]["file"], "survivor.park")

    def test_an_unvouched_copy_is_deleted_on_adoption(self):
        self.second_pool()
        self.assertFalse((SANDBOX.store.slots / "orphan.park").exists())
        self.assertTrue((SANDBOX.store.slots / "survivor.park").exists())

    def test_a_fresh_pool_restores_the_cache_on_the_next_turn(self):
        pool = self.second_pool()
        self.assertEqual(pool.pins["survivor"]["backend"], "(before the restart)")
        self.assertEqual(pool.pins["survivor"]["parked"], "survivor.park")

        before = len(self.cpu.answers)
        self.turn(self.url, "survivor")
        self.assertEqual(self.cpu.restores, ["survivor.park"])
        self.assertEqual(len(self.cpu.answers), before + 1)
        self.assertGreater(self.cpu.probes[-1]["cached"], 0,
                           "the turn after the restart read its prompt again")
        self.assertEqual(pool.pins["survivor"]["backend"], "cpu")


class TheFirstSessionSavesTheOpening(EndToEnd):
    """Nobody has this opening, so the request that needs it reads it.

    Those are tokens at the front of its own prompt, which it was going to
    read anyway, so the only cost is the copy. What it buys is that every
    session starting behind it loads the copy instead of reading the same
    tokens again."""

    def test_it_reads_the_opening_and_keeps_it(self):
        pool = self.pool([self.stub("cpu", 1, slots=2)])
        url = self.serve(pool)
        self.turn(url, "asker", system=LONG_SYSTEM)

        self.assertTrue(pool.openings, "the opening was not kept")
        name = next(iter(pool.openings.values()))
        self.assertTrue(name.startswith("base-") and name.endswith(".park"))
        self.assertEqual(pool.wants, {},
                         "it read the opening, so nothing is left to build")
        # Two renderings, then the read of the opening itself.
        self.assertEqual(self.cpu.templates, 2)
        self.assertIn(name, self.cpu.saves)
        self.assertTrue((SANDBOX.store.slots / name).exists())
        # A block goes on the faster disk, reached through a link.
        self.assertTrue((SANDBOX.store.slots / name).is_symlink())
        self.assertTrue((SANDBOX.store.blocks / name).exists())

    def test_the_next_session_loads_it_rather_than_reading_it(self):
        pool = self.pool([self.stub("cpu", 1, slots=2)])
        url = self.serve(pool)
        self.turn(url, "first", system=LONG_SYSTEM)
        name = next(iter(pool.openings.values()))
        before = self.cpu.templates

        self.turn(url, "second", system=LONG_SYSTEM)
        self.assertIn(name, self.cpu.restores, "it did not load the opening")
        # It may render a deeper opening, which is the builder's own work on a
        # cut two conversations now share. What it must not do is read this
        # one again.
        self.assertEqual([n for n in self.cpu.saves if n.startswith("base-")],
                         [name], "the opening was read a second time")


class AnOpenCodeSessionHasAnOpeningToo(EndToEnd):
    """An openai client keeps its system prompt in its first message.

    The router used to cut before that message anyway, which left an opening
    made of tools and no messages at all. The template refuses an empty
    prompt, so the block never saved, and because the opening a request wants
    is the first cut and no other, nothing else was ever tried: every OpenCode
    turn read its whole prompt from cold. One of them read for an hour before
    the client gave up on it."""

    TOOLS = [{"type": "function",
              "function": {"name": "read", "description": "D" * 2000,
                           "parameters": {"type": "object"}}}]

    def test_the_opening_is_kept(self):
        pool = self.pool([self.stub("cpu", 1, slots=2)])
        url = self.serve(pool)
        self.turn(url, "opencode", system=LONG_SYSTEM, tools=self.TOOLS)

        self.assertTrue(pool.openings, "the opening was not kept")
        name = next(iter(pool.openings.values()))
        self.assertIn(name, self.cpu.saves)
        self.assertTrue((SANDBOX.store.slots / name).exists())

    def test_the_next_session_loads_it(self):
        pool = self.pool([self.stub("cpu", 1, slots=2)])
        url = self.serve(pool)
        self.turn(url, "first", system=LONG_SYSTEM, tools=self.TOOLS)
        name = next(iter(pool.openings.values()))

        self.turn(url, "second", system=LONG_SYSTEM, tools=self.TOOLS)
        self.assertIn(name, self.cpu.restores, "it did not load the opening")
        self.assertEqual([n for n in self.cpu.saves if n.startswith("base-")],
                         [name], "the opening was read a second time")


class SessionsThatStartTogether(EndToEnd):
    """The stampede: several new sessions, one system prompt, nobody has it.

    Each would read its own copy of the same opening. Five real sessions did
    exactly that this afternoon and read 92,000 tokens between them where
    24,000 would have done, the last finishing after seventeen minutes. So the
    first reads it and saves it, and the rest wait for that and load it."""

    def test_the_opening_is_read_once_for_all_of_them(self):
        pool = self.pool([self.stub("cpu", 1, slots=4, busy_ms=READ_MS)])
        url = self.serve(pool)

        boxes = [self.start_turn(url, f"s{n}", system=LONG_SYSTEM,
                                 text=f"task number {n}") for n in range(4)]
        self.assertTrue(wait_for(lambda: all("reply" in b or "error" in b
                                             for b in boxes), patience=30),
                        f"a session never finished: {boxes}")
        for n, box in enumerate(boxes):
            self.assertNotIn("error", box, f"session {n}: {box.get('error')}")

        name = next(iter(pool.openings.values()))
        read = [c for c in self.cpu.completions if c["read"] > 0]
        self.assertEqual(self.cpu.saves.count(name), 1,
                         f"the opening was saved {self.cpu.saves.count(name)} "
                         f"times, so it was read that many times")
        self.assertGreaterEqual(self.cpu.restores.count(name), 1,
                                "no session loaded what the first one saved")

    def test_none_of_them_is_refused_or_left_waiting_for_ever(self):
        pool = self.pool([self.stub("cpu", 1, slots=2, busy_ms=READ_MS)])
        url = self.serve(pool)
        boxes = [self.start_turn(url, f"q{n}", system=LONG_SYSTEM,
                                 text=f"task {n}") for n in range(3)]
        self.assertTrue(wait_for(lambda: all("reply" in b or "error" in b
                                             for b in boxes), patience=40),
                        f"a session never finished: {boxes}")
        self.assertEqual([b for b in boxes if "error" in b], [])


class BuilderInTheBackground(EndToEnd):
    """The builder reads an opening only where a backend can spare a slot."""

    def test_it_builds_nothing_while_every_slot_is_busy(self):
        pool = self.pool([self.stub("cpu", 1, slots=1)])
        self.serve(pool)
        pool.wants["k1"] = {"cut": (-1, "k1"), "mark": "base-",
                            "system": LONG_SYSTEM, "head": [],
                            "path": "/v1/chat/completions"}
        for slot in self.backend(pool, "cpu")["slots_detail"]:
            slot["busy"] = True
        self.backend(pool, "cpu")["busy"] = 1

        time.sleep(0.5)               # several builder passes, all of them idle
        self.assertEqual(len(pool.wants), 1, "it built with no slot to spare")
        self.assertEqual(pool.openings, {})
        self.assertEqual(self.cpu.completions, [])


class TheDeliberateCost(EndToEnd):
    """Keeping an instance for generation costs a wait. This is that wait."""

    def test_a_new_conversation_waits_rather_than_prefill_where_reads_is_off(self):
        pool = self.pool([self.stub("gpu", 0, prefill=False),
                          self.stub("cpu", 1),
                          self.stub("cpu2", 2)])
        url = self.serve(pool)
        self.cpu.hold()
        self.cpu2.hold()
        # Which cpu takes which read follows the reading order, so wait for
        # both to be held rather than naming them.
        self.start_turn(url, "first")
        self.start_turn(url, "second")
        self.assertTrue(wait_for(
            lambda: self.backend(pool, "cpu")["busy"] == 1
            and self.backend(pool, "cpu2")["busy"] == 1),
            "both cpu instances never filled")

        box = self.start_turn(url, "third")
        self.assertTrue(wait_for(lambda: pool.waiting == 1),
                        "the third conversation never queued")
        time.sleep(0.5)
        self.assertEqual(box, {}, "the third conversation was served anyway")
        self.assertEqual(self.gpu.chats, [], "it went to the gpu")
        gpu = self.backend(pool, "gpu")
        self.assertTrue(gpu["up"])
        self.assertEqual(gpu["busy"], 0, "the gpu was idle the whole time")

        self.cpu.release()                # a cpu slot frees, and it goes there
        self.assertTrue(wait_for(lambda: "reply" in box),
                        "the waiting conversation was never served")
        # Both conversations were read on the cpu that freed. Where they
        # generate afterwards is the handoff's business, and the gpu is free
        # to take them now that the reading is done.
        self.assertEqual(self.gpu.probes, [], "a prompt was read on the gpu")
        # Which cpu read which follows the reading order; what matters is that
        # the third was read on a cpu once one freed, and never on the gpu.
        read_third = [be.name for be in (self.cpu, self.cpu2)
                      if "third" in [c["key"] for c in be.probes]]
        self.assertEqual(len(read_third), 1, "the third was not read on a cpu")
        self.cpu2.release()


class TheWatcherReadsTheBackend(EndToEnd):
    """Everything the pool knows about a backend comes from three endpoints.

    A stub that answered any of them in the wrong shape would fail quietly,
    because the watcher swallows the error and keeps the figures it had. So
    the shape is worth one test of its own."""

    def test_props_metrics_and_slots_all_arrive(self):
        pool = self.pool([self.stub("cpu", 1, slots=2)])
        url = self.serve(pool)
        self.turn(url, "counted")
        cpu = self.backend(pool, "cpu")
        self.assertEqual((cpu["slots"], cpu["n_ctx"], cpu["model"]),
                         (2, 150000, "fake-model"))
        self.assertEqual([s["id"] for s in cpu["slots_detail"]], [0, 1])
        self.assertTrue(wait_for(lambda: cpu["stats"].get("tg_rate")),
                        "nothing was parsed out of /metrics")
        self.assertTrue(wait_for(lambda: all(not s["busy"]
                                             for s in cpu["slots_detail"])),
                        "a slot still says it is processing")


class MissingBackendLog(EndToEnd):
    """CacheWatch follows a log the test directory does not have."""

    def test_a_backend_with_no_log_file_still_reports_its_cache(self):
        pool = self.pool([self.stub("cpu", 1)])
        self.assertFalse((SANDBOX.store.run / "cpu.log").exists())
        self.assertTrue(wait_for(lambda: self.backend(pool, "cpu")["cache"]))
        self.assertEqual(self.backend(pool, "cpu")["cache"]["evictions"], 0)
        self.assertEqual(self.backend(pool, "cpu")["up"], True)


class TheHandoff(EndToEnd):
    """One client request, two calls to the backends.

    The read pass asks for a single token on a backend that reads. The slot it
    leaves behind is carried to whichever backend should generate, and the real
    request extends it there, on whichever instance is configured to
    generate."""

    def setUp(self):
        super().setUp()
        self.duo = self.pool([self.stub("gpu", 0, prefill=False),
                              self.stub("cpu", 1)])
        self.url = self.serve(self.duo)
        self.turn(self.url, "talker")

    def test_it_is_read_on_the_cpu_and_generates_on_the_gpu(self):
        self.assertEqual([c["key"] for c in self.cpu.probes], ["talker"])
        self.assertEqual(self.cpu.answers, [], "the cpu generated the answer")
        self.assertEqual([c["key"] for c in self.gpu.answers], ["talker"])
        self.assertEqual(self.gpu.probes, [], "the gpu was given a prompt")
        self.assertEqual(self.duo.pins["talker"]["backend"], "gpu")
        # One turn for the client, whatever the router did behind it.
        self.assertTrue(wait_for(lambda: len(self.duo.recent_requests) == 1),
                        "the client's one request was not one request")

    def test_the_slot_is_carried_over_on_the_conversation_s_own_copy(self):
        """The copy that carries it is the copy it would have parked anyway.

        Handing the reader back before the wait means the conversation has to
        be somewhere while it waits, and on disk under its own name is a state
        the router already knows. So there is no separate carrier file: one
        save, one restore, and what is left on disk is its park."""
        self.assertEqual(self.cpu.saves, ["talker.park"])
        self.assertEqual(self.gpu.restores, ["talker.park"])
        self.assertFalse((SANDBOX.store.slots / "talker.kv").exists(),
                         "a carrier file was written after all")

    def test_the_gpu_is_given_no_prompt_to_read(self):
        answer = self.gpu.answers[0]
        self.assertEqual(answer["read"], 0, "the gpu read the prompt itself")
        self.assertGreater(answer["cached"], 0, "it extended nothing")
        # The state has to be in the slot before the request that extends it.
        paths = [row["path"] + row["query"] for row in self.gpu.requests]
        self.assertLess(paths.index("/slots/0?action=restore"),
                        paths.index("/v1/chat/completions"))


class ASubagentIsACaller_Too(EndToEnd):
    """A subagent turn has to save and restore like any other.

    Claude Code names a subagent with a second header, and the router joins
    the two into one conversation key. That key becomes the name of the slot
    file, so it has to be a name a backend will accept."""

    def setUp(self):
        super().setUp()
        self.duo = self.pool([self.stub("gpu", 0, prefill=False),
                              self.stub("cpu", 1)])
        self.url = self.serve(self.duo)
        self.agent_turn(self.url, "s-1", "a-1")

    def agent_turn(self, url, session, agent, timeout=30):
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps({"model": "fake-model",
                             "messages": [{"role": "system", "content": "s"},
                                          {"role": "user", "content": "hi"}]}
                            ).encode(),
            headers={"Content-Type": "application/json",
                     "x-claude-code-session-id": session,
                     "x-claude-code-agent-id": agent},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)

    def test_the_subagent_generates_on_the_gpu(self):
        self.assertEqual(self.gpu.answers and self.gpu.answers[0]["read"], 0,
                         "the subagent never reached the gpu")
        self.assertEqual(self.cpu.answers, [], "the cpu generated the answer")

    def test_its_slot_file_is_a_name_a_backend_accepts(self):
        self.assertEqual(len(self.cpu.saves), 1,
                         f"the save was refused: {self.cpu.saves}")
        self.assertEqual(self.cpu.saves, self.gpu.restores)


class TheGeneratorIsBusy(EndToEnd):
    """A turn queues to generate rather than generate where it prefilled.

    A prefiller that generates holds a slot that could be reading somebody
    else's prompt for the whole of it: minutes of their prefill spent to save
    seconds on this turn. So the turn waits, and the prefillers are left to
    read."""

    def test_it_waits_for_the_gpu_instead_of_answering_on_the_cpu(self):
        pool = self.pool([self.stub("gpu", 0, prefill=False),
                          self.stub("cpu", 1, busy_ms=READ_MS)])
        url = self.serve(pool)

        self.gpu.hold()               # the first conversation holds its one slot
        self.start_turn(url, "first")
        self.assertTrue(wait_for(lambda: self.backend(pool, "gpu")["busy"] == 1),
                        "the first conversation never reached the gpu")

        second = self.start_turn(url, "second")
        self.assertTrue(
            wait_for(lambda: "second" in [c["key"] for c in self.cpu.probes]),
            "the second prompt was never read")
        # Read, and now queued. A cpu answer here is the thing being ruled out,
        # so give it every chance to appear before saying it did not.
        self.assertFalse(
            wait_for(lambda: [c["key"] for c in self.cpu.answers] == ["second"],
                     patience=2.0),
            "it answered on the cpu instead of waiting for the gpu")

        self.gpu.release()            # the first turn ends, the slot frees
        self.assertTrue(wait_for(lambda: "reply" in second or "error" in second,
                                 patience=30), f"it never finished: {second}")
        self.assertNotIn("error", second, f"{second.get('error')}")
        self.assertIn("second", [c["key"] for c in self.gpu.answers],
                      "it did not generate on the gpu once the slot freed")
        self.assertEqual([c["key"] for c in self.cpu.answers], [],
                         "the cpu generated after all")
        self.assertEqual(pool.pins["second"]["backend"], "gpu")


class TheNextTurnComesBack(EndToEnd):
    """Nothing reads on the gpu, so a cache left there is a cache lost.

    It is copied out as the turn ends, and the next turn restores it onto a
    cpu instance and reads there. That round trip is the whole arrangement."""

    def test_the_cache_is_parked_and_the_next_turn_is_read_on_a_cpu(self):
        pool = self.pool([self.stub("gpu", 0, prefill=False),
                          self.stub("cpu", 1)])
        url = self.serve(pool)

        self.turn(url, "talker")
        # Wait for the gpu's own copy, not just for the pin to say parked: the
        # handoff parks it on the cpu on the way over, so the pin says parked
        # from the moment the turn leaves the reader.
        self.assertTrue(wait_for(lambda: self.gpu.saves == ["talker.park"]),
                        f"the gpu kept the only copy of the cache: {self.gpu.saves}")
        self.assertTrue(pool.pins["talker"].get("parked"))
        self.assertTrue((SANDBOX.store.slots / "talker.park").exists())

        self.turn(url, "talker")
        self.assertEqual(self.cpu.restores, ["talker.park"])
        self.assertEqual([c["key"] for c in self.cpu.probes],
                         ["talker", "talker"])
        self.assertEqual(self.gpu.probes, [], "the second turn read on the gpu")
        back = self.cpu.probes[-1]
        self.assertEqual(back["read"], 0, "it read the whole prompt again")
        self.assertGreater(back["cached"], 0, "the copy was never used")


class TheCarrierFails(EndToEnd):
    """A handoff that does not work costs the attempt and nothing else.

    The conversation generates where its prompt was read, which is what would
    have happened without the handoff at all."""

    def setUp(self):
        super().setUp()
        self.duo = self.pool([self.stub("gpu", 0, prefill=False),
                              self.stub("cpu", 1)])
        self.url = self.serve(self.duo)

    def check_it_stayed_on_the_cpu(self):
        self.assertEqual([c["key"] for c in self.cpu.answers], ["stuck"])
        self.assertEqual(self.gpu.chats, [], "the gpu served it anyway")
        self.assertEqual(self.duo.pins["stuck"]["backend"], "cpu")
        self.assertEqual(self.duo.pins["stuck"]["slot"], 0)
        # A held slot that nothing released would shut the backend out of
        # every later request, so the counts matter more than the failure.
        self.assertTrue(wait_for(lambda: [be["busy"] for be in self.duo.backends]
                                 == [0, 0]),
                        "a slot was counted busy after the turn ended")
        self.assertFalse((SANDBOX.store.slots / "stuck.kv").exists(),
                         "a carrier file was written after all")

    def test_a_failed_save_leaves_it_where_it_was_read(self):
        self.cpu.fail_save = True
        self.turn(self.url, "stuck")
        self.assertEqual(self.gpu.restores, [], "a state reached the gpu")
        self.check_it_stayed_on_the_cpu()

    def test_a_failed_restore_leaves_it_where_it_was_read(self):
        self.gpu.fail_restore = True
        self.turn(self.url, "stuck")
        self.assertEqual(self.cpu.saves, ["stuck.park"], "nothing was even saved")
        self.check_it_stayed_on_the_cpu()


class OneTurnOfAConversationAtATime(EndToEnd):
    """Two turns of one conversation move the same pin, slot and copy.

    Run at once, the second takes a reader of its own, clears the slot the
    first is reading in, and writes its own state over the first one's copy
    under the same name. The first then carries that copy to the gpu and
    reads its whole prompt again from nothing.

    So the second waits for the first, and starts from what it left."""

    def setUp(self):
        super().setUp()
        self.both = self.pool([self.stub("cpu0", 0, busy_ms=READ_MS),
                               self.stub("cpu1", 1, busy_ms=READ_MS)])
        self.url = self.serve(self.both)

    def busy(self):
        return sum(self.backend(self.both, name)["busy"]
                   for name in ("cpu0", "cpu1"))

    def test_the_second_turn_takes_no_reader_of_its_own(self):
        self.cpu0.hold()
        self.cpu1.hold()                       # whichever it lands on, it stops
        first = self.start_turn(self.url, "same")
        self.assertTrue(wait_for(lambda: self.busy() == 1),
                        "the first turn never took a reader")

        second = self.start_turn(self.url, "same")
        time.sleep(1.0)
        self.assertEqual(self.busy(), 1,
                         "two turns of one conversation held two readers")
        # The backends' own view, not the router's counter: one slot in use
        # between them.
        working = sum(1 for stub in (self.cpu0, self.cpu1)
                      for slot in stub.slots if slot.busy)
        self.assertEqual(working, 1, "the prompt was being read in two slots")

        self.cpu0.release()
        self.cpu1.release()
        self.assertTrue(wait_for(lambda: "reply" in first and "reply" in second,
                                 patience=20),
                        f"a turn never finished: {first} {second}")

    def test_another_conversation_is_not_made_to_wait(self):
        self.cpu0.hold()
        self.cpu1.hold()
        self.start_turn(self.url, "same")
        self.assertTrue(wait_for(lambda: self.busy() == 1),
                        "the first turn never took a reader")

        self.start_turn(self.url, "other")
        self.assertTrue(wait_for(lambda: self.busy() == 2),
                        "a second conversation was held up behind the first")
        self.cpu0.release()
        self.cpu1.release()


class AFullBoxMakesTheClientWait(EndToEnd):
    """A busy box is a queue, not a refusal.

    Every slot taken means this turn starts later, not that it fails. The
    client is told the reply has begun and is kept alive with the keep-alive
    its protocol defines until a slot frees."""

    def setUp(self):
        super().setUp()
        SANDBOX.tuning = replace(SANDBOX.tuning, ping_every=0.05)
        self.only = self.pool([self.stub("cpu", 0, busy_ms=READ_MS)])
        self.url = self.serve(self.only)

    def test_it_waits_for_the_slot_instead_of_refusing(self):
        self.cpu.hold()                        # the one slot is taken
        first = self.start_turn(self.url, "first")
        self.assertTrue(wait_for(lambda: self.backend(self.only, "cpu")["busy"] == 1),
                        "the first conversation never took the slot")

        second = self.start_turn(self.url, "second")
        time.sleep(6.0)          # longer than the deadline there used to be
        self.assertNotIn("error", second,
                         f"the waiting turn was refused: {second.get('error')}")

        self.cpu.release()
        self.assertTrue(wait_for(lambda: "reply" in second, patience=20),
                        f"the waiting turn never ran: {second}")
        self.assertIn("second", [c["key"] for c in self.cpu.probes])

    def test_the_waiting_client_is_sent_keep_alives(self):
        self.cpu.hold()
        self.start_turn(self.url, "first")
        self.assertTrue(wait_for(lambda: self.backend(self.only, "cpu")["busy"] == 1),
                        "the first conversation never took the slot")

        seen = self.background(lambda: self.stream_head(self.url, "second"),
                               name="waiter")
        self.assertTrue(wait_for(lambda: seen.get("value") or seen.get("error"),
                                 patience=20), "nothing came back at all")
        self.cpu.release()
        self.assertIn(b"ping", seen.get("value") or b"",
                      "a waiting client was sent no keep-alive")

    def stream_head(self, url, conv):
        """Open a streaming turn and read what arrives while it waits."""
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps({"model": "fake-model", "stream": True,
                             "prompt_cache_key": conv,
                             "messages": [{"role": "user", "content": "hi"}]}
                            ).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.read(64)


class WorkNobodyIsWaitingForIsStopped(EndToEnd):
    """A read whose client has gone is a slot held for nothing.

    A client that gives up sends the turn again, and the retry queues behind
    the read it replaced, so one abandoned turn costs the box two. Dropping
    the connection to the backend is also what tells it to cancel the task."""

    def setUp(self):
        super().setUp()
        self.only = self.pool([self.stub("cpu", 0, gate_wait=60)])
        self.url = self.serve(self.only)

    def leave_mid_read(self):
        """Start a turn, wait for the read, then give up as a client does."""
        self.cpu.hold()                    # the read will not finish by itself
        conn = http.client.HTTPConnection("127.0.0.1", self.port(self.url))
        conn.request("POST", "/v1/chat/completions", self.body("goer"),
                     {"Content-Type": "application/json"})
        self.assertTrue(wait_for(lambda: self.backend(self.only, "cpu")["busy"] == 1),
                        "the read never started")
        conn.close()

    def test_the_slot_comes_back_when_the_client_leaves(self):
        self.leave_mid_read()
        self.assertTrue(
            wait_for(lambda: self.backend(self.only, "cpu")["busy"] == 0,
                     patience=20),
            "the slot was still held for a client that had gone")
        self.cpu.release()

    def test_the_backend_is_told_to_stop(self):
        """The router's own books are not the thing that matters.

        Marking the slot free costs the backend nothing: it goes on reading a
        prompt nobody will read the answer to, and the retry queues behind it.
        On 13 September a backend read 114,354 tokens for forty minutes after
        the client had gone, because closing the connection left the socket
        open behind a thread still reading from it."""
        self.leave_mid_read()
        self.assertTrue(wait_for(lambda: self.cpu.cancelled == 1, patience=20),
                        "the backend never saw the client go and read on")
        self.cpu.release()

    def test_what_the_read_got_through_is_kept_for_the_retry(self):
        """A client that gives up sends the same turn again.

        Thrown away, a prompt this box cannot read inside the client's
        patience can never be read at all: every attempt starts from the
        shared opening and gives up in the same place. OpenCode sent a 114,354
        token turn twice on 13 September and gave up on both after an hour,
        two thirds of the way through."""
        self.leave_mid_read()
        self.assertTrue(wait_for(lambda: self.cpu.cancelled == 1, patience=20),
                        "the backend never saw the client go")
        self.assertTrue(
            wait_for(lambda: "goer.park" in self.cpu.saves, patience=20),
            f"the abandoned read was thrown away: {self.cpu.saves}")

        self.cpu.release()
        self.turn(self.url, "somebody-else")   # the one slot changes hands
        self.turn(self.url, "goer")            # the client sends it again
        self.assertIn("goer.park", self.cpu.restores,
                      "the retry did not start from the abandoned read")
        self.assertGreater(self.cpu.chats[-1]["cached"], 0,
                           "the retry read the whole prompt again")

    @staticmethod
    def port(url):
        return int(url.rsplit(":", 1)[1])


class TheClientIsNeverLeftInSilence(EndToEnd):
    """A client must get bytes while the prompt is being read.

    Reading runs for tens of minutes and the backend sends nothing during it.
    A client drops a stream that goes quiet, so the reply has to start before
    the reading does and be kept alive through it.

    The old test for this checked wants_ping(), a predicate. That kept
    answering correctly after the reading moved out of the streaming path, so
    nothing failed while every streamed request went silent for half an hour.
    This one asserts what the client actually receives."""

    def stream_until(self, url, key, seconds):
        """Open a streamed turn and report when the first byte arrives."""
        body = json.dumps({"model": "q", "stream": True, "max_tokens": 4,
                           "messages": [{"role": "user", "content": key}]})
        parts = urllib.parse.urlsplit(url)
        conn = http.client.HTTPConnection(parts.hostname, parts.port,
                                          timeout=seconds)
        started = time.time()
        conn.request("POST", "/v1/chat/completions", body,
                     {"content-type": "application/json",
                      "x-claude-code-session-id": key})
        reply = conn.getresponse()
        headers_at = time.time() - started
        first = reply.read(1)
        return headers_at, time.time() - started, bool(first), reply, conn

    def test_bytes_arrive_while_the_prompt_is_still_being_read(self):
        """The reply starts before the reading does, and is kept alive through
        it with the keep-alive that client's protocol defines."""
        was = SANDBOX.tuning
        SANDBOX.tuning = replace(was, ping_every=0.2)
        self.addCleanup(lambda: setattr(SANDBOX, "tuning", was))

        pool = self.pool([self.stub("cpu", 1)])
        url = self.serve(pool)
        self.cpu.hold()                  # the read pass never finishes

        headers_at, first_at, got, reply, conn = self.stream_until(url, "slow", 10)
        try:
            self.assertLess(headers_at, 2.0,
                            "the client waited for the whole read before a header")
            self.assertTrue(got, "the client got no bytes while the prompt was read")
            self.assertLess(first_at, 3.0,
                            "the client was left silent longer than a ping apart")
        finally:
            self.cpu.release()
            conn.close()

    def test_a_streamed_reply_says_so_from_the_start(self):
        pool = self.pool([self.stub("cpu", 1)])
        url = self.serve(pool)
        headers_at, _, _, reply, conn = self.stream_until(url, "quick", 10)
        try:
            self.assertEqual(reply.status, 200)
            self.assertEqual(reply.getheader("Content-Type"), "text/event-stream")
        finally:
            conn.close()


class AnAnthropicStreamIsAMessageFromTheStart(EndToEnd):
    """A stream that has only pinged has begun nothing.

    The router answers before the prompt is read, so on this hardware the
    first thing a client of /v1/messages sees comes half an hour before the
    reply does. ping is a valid event anywhere inside an anthropic stream and
    not a valid way to open one: the client waits for a message that never
    started, gives up, and reports a 502 the router never sent."""

    def setUp(self):
        super().setUp()
        was = SANDBOX.tuning
        SANDBOX.tuning = replace(was, ping_every=0.2)
        self.addCleanup(lambda: setattr(SANDBOX, "tuning", was))
        self.only = self.pool([self.stub("cpu", 0)])
        self.url = self.serve(self.only)

    def open_stream(self, key="agent", path="/v1/messages", timeout=30):
        """Start a streamed turn and hand back the reply, headers read."""
        body = json.dumps({"model": "qwen3.8-flash-next-mtp", "stream": True,
                           "max_tokens": 16, "system": LONG_SYSTEM,
                           "messages": [{"role": "user", "content": "hi"}]})
        parts = urllib.parse.urlsplit(self.url)
        conn = http.client.HTTPConnection(parts.hostname, parts.port,
                                          timeout=timeout)
        self.addCleanup(conn.close)
        conn.request("POST", path, body,
                     {"content-type": "application/json",
                      "x-claude-code-session-id": key})
        return conn.getresponse()

    def test_the_first_event_says_a_message_has_started(self):
        self.cpu.hold()                    # the read runs on, as a real one does
        reply = self.open_stream("early")
        self.addCleanup(self.cpu.release)
        name, data = sse_events(reply, most=1)[0]
        self.assertEqual(name, "message_start",
                         "the client was pinged before anything had begun")
        self.assertEqual(data["message"]["role"], "assistant")
        self.assertEqual(data["message"]["model"], "qwen3.8-flash-next-mtp")
        self.assertTrue(data["message"]["id"], "the message has no id")

    def test_the_client_is_sent_exactly_one_message_start(self):
        self.cpu.hold()
        reply = self.open_stream("whole")
        opening = sse_events(reply, most=1)
        time.sleep(0.5)                    # long enough for a ping or two
        self.cpu.release()
        names = [name for name, _ in opening + sse_events(reply)]
        self.assertEqual(names.count("message_start"), 1,
                         f"the backend's own was spliced in too: {names}")
        self.assertEqual(names[0], "message_start", names)
        self.assertEqual(names[-1], "message_stop", names)
        self.assertIn("ping", names, "nothing kept the stream alive")
        self.assertIn("content_block_delta", names, "no reply reached the client")

    def test_the_prompt_token_count_survives_the_join(self):
        """The dropped message_start is the only event carrying it, and a
        client sizes its context window with it."""
        reply = self.open_stream("counted")
        events = dict(sse_events(reply))
        usage = events["message_delta"]["usage"]
        self.assertGreater(usage.get("input_tokens", 0)
                           + usage.get("cache_read_input_tokens", 0), 0,
                           f"the prompt tokens went with the event: {usage}")
        self.assertEqual(usage["output_tokens"], 1)

    def test_an_openai_stream_still_opens_with_a_comment(self):
        """A comment is legal at the head of an OpenAI stream, so that path
        keeps the keep-alive it has."""
        self.cpu.hold()
        reply = self.open_stream("oai", path="/v1/chat/completions")
        self.addCleanup(self.cpu.release)
        self.assertEqual(reply.read(len(router.PING)), router.PING)


class AnIdleClientIsNotAGoneOne(EndToEnd):
    """A live client that sends nothing must never be judged to have left.

    alive aborts the read when it answers False, and a read is the
    only thing this box does slowly. A false positive here would kill every
    long read on it, so idleness has to be proven harmless rather than
    assumed to be."""

    def test_a_client_that_sits_quiet_through_several_polls_keeps_its_read(self):
        pool = self.pool([self.stub("cpu", 0, gate_wait=60)])
        url = self.serve(pool)
        self.cpu.hold()
        held = self.start_turn(url, "quiet")
        self.assertTrue(wait_for(lambda: self.backend(pool, "cpu")["busy"] == 1),
                        "the read never started")
        # http_post_wanted looks every two seconds. Sit still through three
        # looks, sending nothing, which is what a client does while it waits.
        time.sleep(6.5)
        self.assertEqual(self.backend(pool, "cpu")["busy"], 1,
                         "an idle client had its read taken away")
        self.cpu.release()
        self.assertTrue(wait_for(lambda: "reply" in held, patience=20),
                        f"the turn never finished: {held}")
        self.assertIn("quiet", [c["key"] for c in self.cpu.probes],
                      "the prompt was never read")


class ARefusalReachesAStreamAsAnEvent(EndToEnd):
    """A backend can turn the generate call down after the prompt was read.

    By then the stream has been open for half an hour and carries a message
    under way, so the refusal cannot be a status and its body is not an
    event. Passing it through raw leaves the client parsing a json object
    where an SSE frame should be."""

    def test_a_backend_refusal_arrives_as_an_error_event(self):
        pool = self.pool([self.stub("cpu", 0)])
        url = self.serve(pool)
        self.cpu.refuse = True
        parts = urllib.parse.urlsplit(url)
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
        self.addCleanup(conn.close)
        conn.request("POST", "/v1/messages", json.dumps(
            {"model": "q", "stream": True, "max_tokens": 16,
             "messages": [{"role": "user", "content": "hi"}]}),
            {"content-type": "application/json",
             "x-claude-code-session-id": "refused"})
        reply = conn.getresponse()
        self.assertEqual(reply.status, 200, "the stream had already opened")
        names = [name for name, _ in sse_events(reply)]
        self.assertEqual(names, ["message_start", "error"], names)

if __name__ == "__main__":
    unittest.main()
