"""The router, driving real llama-server instances.

tests/test_end_to_end.py runs the same Pool and Handler against a stub written
from the router's own beliefs, which can only confirm them. Here the backends
are real, and every assertion is on what one reports about itself: the tokens
in its own `prompt eval time = ... / N tokens` lines.

That is the difference these tests exist for. A test that read the router's own
log would have passed on the day four features were dead.

tests/live/README.md says how to run them.
"""

import json
import os
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from harness import REREAD_LINE, LiveCase, prompt_evals, prose, read_tokens

# The cache event log is on by default and writes into run/. These tests
# write nothing there, so it stays off here.
os.environ.setdefault("CACHE_LOG", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bin"))
import router                                                    # noqa: E402

POOL_LOOPS = ("_watch", "_builder")

# How long a test waits for another thread. Only reached when it is about to
# fail anyway, so it can afford to be generous.
PATIENCE = 90.0


class Bomb:
    """Stop a pool's daemon loops, exactly as tests/test_end_to_end.py does.

    The loops are `while True` with no off switch. A SystemExit raised inside
    one ends that thread quietly, and `except Exception` does not catch it."""

    def __enter__(self):
        raise SystemExit

    def __exit__(self, *rest):
        return False

    def __call__(self, *args, **kw):
        raise SystemExit


def pool_threads():
    return [t for t in threading.enumerate()
            if any(loop in t.name for loop in POOL_LOOPS) and t.is_alive()]


def wait_for(check, patience=PATIENCE, step=0.05):
    stop = time.time() + patience
    while time.time() < stop:
        if check():
            return True
        time.sleep(step)
    return bool(check())


class LiveRouter(LiveCase):
    """A real Pool and the real Handler over real backends.

    Nothing is written into the project's run/ directory and no production
    instance is touched: the pool only ever holds the backends this test
    started, on ports from 18080 up.
    """

    # Long enough that prompt_cuts names a cut in it. PREFIX_MIN_CHARS cannot
    # be lowered from a test: prompt_cuts takes it as a default argument, which
    # Python binds once at import, so patching the module attribute afterwards
    # changes nothing. The system prompt is sized for the real constant
    # instead, and the context is sized for the system prompt.
    SYSTEM_CHARS = 9000
    CTX = 8192

    # Router settings a test depends on are stated here and applied in setUp,
    # never read from the checkout. HANDOFF_ON in particular has changed
    # default twice while these tests were being written, and a test that
    # inherited it measured a different router each time.
    HANDOFF = False

    def setUp(self):
        super().setUp()
        self.kept = {name: getattr(router, name) for name in
                     ("STORE", "POLL", "BUILD_POLL", "PIN_PATIENCE",
                      "PARK_ALL_TIMEOUT", "PARK_FLOOR", "IDLE_POLLS",
                      "HANDOFF_ON")}
        self.had_pool = getattr(router, "POOL", None)
        router.HANDOFF_ON = self.HANDOFF
        # One store over this test's whole run directory. The backends share
        # its slot directory, because a backend takes only a bare filename
        # under its own --slot-save-path. Every backend also writes
        # <name>.log there, which is where CacheWatch looks. The cache
        # counters come off a real log for once. harness.Server builds the
        # slot and log paths the same way from the same root.
        router.STORE = router.Store(self.root)
        router.STORE.slots.mkdir(parents=True, exist_ok=True)
        router.POLL = 0.2
        router.BUILD_POLL = 0.3
        router.IDLE_POLLS = 1
        router.PIN_PATIENCE = 1.0       # worth 20 seconds in production
        router.PARK_ALL_TIMEOUT = 30.0
        # "A real state is at least this big". The figure in the router is for
        # the production model's fixed recurrent state; the test model's
        # states are smaller. An empty save is still under a kilobyte, so this
        # tells a real copy from an empty one just as well.
        router.PARK_FLOOR = 32 * 1024
        self.pools = []
        self.http = []
        self.helpers = []
        self.addCleanup(self.stop_router)

    # ---- building one ---------------------------------------------------

    # llama.cpp's own flags, chosen here rather than inherited from whatever
    # bin/common.sh happens to pass today. A test that cares about one of
    # these overrides it; the rest get an instance that keeps what it read,
    # which is what the router assumes and what production configures.
    KEEPS_ITS_SLOTS = ("--no-cache-idle-slots",)

    def start(self, name, **kw):
        kw.setdefault("ctx", self.CTX)
        kw.setdefault("extra", list(self.KEEPS_ITS_SLOTS))
        return super().start(name, **kw)

    @staticmethod
    def hush_builder(pool):
        """Stop the opening builder reading into these backends.

        It runs on its own thread and takes a slot when one is idle, so in a
        test with one slot an instance it lands in the middle of whatever is
        being measured. The loop still comes round, so the teardown can still
        stop it."""
        pool.build_once = lambda post, **kw: None
        return pool

    def pool(self, specs):
        made = router.Pool(specs, store=router.STORE, watch=True)
        self.pools.append(made)
        self.assertTrue(
            wait_for(lambda: all(b["up"] for b in made.backends)),
            "the pool never saw the live backends come up")
        return made

    def serve(self, pool):
        router.POOL = pool
        server = router.Server(("127.0.0.1", 0), router.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.http.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}"

    def spec(self, server, pref, prefill=True, generate=True, node=0):
        return {"name": server.name, "url": server.url, "pref": pref,
                "prefill": prefill, "generate": generate, "node": node}

    def backend(self, pool, name):
        return next(b for b in pool.backends if b["name"] == name)

    # ---- driving a conversation -----------------------------------------

    def turn(self, url, conv, messages, timeout=900, max_tokens=12,
             path="/v1/chat/completions"):
        body = json.dumps({"model": "live", "prompt_cache_key": conv,
                           "max_tokens": max_tokens, "stream": False,
                           "messages": messages}).encode()
        request = urllib.request.Request(
            url + path, data=body, headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)

    @staticmethod
    def said(reply):
        """What the model actually generated, to feed back as the next turn's
        assistant message."""
        return reply["choices"][0]["message"]["content"]

    def start_turn(self, url, conv, messages, **kw):
        box = {}

        def run():
            try:
                box["reply"] = self.turn(url, conv, messages, **kw)
            except Exception as err:
                box["error"] = err

        thread = threading.Thread(target=run, name=f"turn-{conv}", daemon=True)
        self.helpers.append(thread)
        thread.start()
        return box

    def opening(self, salt="live"):
        return [{"role": "system", "content": prose(self.SYSTEM_CHARS, salt)}]

    # ---- teardown --------------------------------------------------------

    def stop_router(self):
        stuck = []
        for helper in self.helpers:
            helper.join(PATIENCE)
            if helper.is_alive():
                stuck.append(helper.name)
        for server, thread in self.http:
            server.shutdown()
            server.server_close()
            thread.join(PATIENCE)
        router.POLL = router.BUILD_POLL = 0.02
        for pool in self.pools:
            pool.build_once = Bomb()
            pool.cv = Bomb()
        alive = not wait_for(lambda: not pool_threads(), patience=10.0)
        for name, value in self.kept.items():
            setattr(router, name, value)
        router.POOL = self.had_pool
        self.assertEqual(stuck, [], "a test thread never finished")
        self.assertFalse(alive, f"pool threads outlived the test: {pool_threads()}")


class WithTheHandOffOff(LiveRouter):
    """What a pool of two shapes does when nothing is carried.

    HANDOFF_ON is set here rather than read, so this says what the router does
    with the hand-off off whatever the checkout happens to default to.

    The consequence is worth stating plainly: an instance with `reads` off is
    never given a new conversation by `acquire`, and with the hand-off off
    nothing can arrive there by being carried either. It serves nothing.
    """

    HANDOFF = False

    def setUp(self):
        super().setUp()
        self.gen = self.start("gpu0_0", slots=1)
        self.read = self.start("cpu1_0", slots=2)
        self.pool_ = self.pool([self.spec(self.gen, 0, prefill=False, node=0),
                                self.spec(self.read, 1, prefill=True, node=1)])
        self.url = self.serve(self.pool_)
        self.messages = self.opening("handoff") + [{"role": "user", "content": "Hello."}]

    def test_the_prompt_is_read_on_the_instance_that_reads(self):
        before = self.read.mark()
        self.turn(self.url, "handoff", self.messages)
        self.assertGreater(read_tokens(self.read.since(before)), 500,
                           "the instance that reads did not read the prompt")

    def test_the_conversation_stays_where_it_read(self):
        self.turn(self.url, "handoff", self.messages)
        self.assertEqual(self.pool_.pins["handoff"]["backend"], "cpu1_0")
        self.assertEqual([row["did"] for row in self.pool_.recent
                          if row["did"] == "moved"], [])

    def test_an_instance_that_does_not_read_is_given_nothing_at_all(self):
        before = self.gen.mark()
        for n in range(3):
            self.turn(self.url, f"handoff-{n}", self.opening(f"h{n}")
                      + [{"role": "user", "content": "Hello."}])
        self.assertEqual(prompt_evals(self.gen.since(before)), [],
                         "something reached the instance that does not read")


class WithTheHandOffOn(LiveRouter):
    """Carrying a slot to the instance that should generate.

    HANDOFF_ON is set here rather than read. The hand-off saves the slot on
    the instance that read the prompt and restores it on the one that will
    generate, which is then handed the very same prompt. Whether that costs
    nothing or costs the whole prompt again depends on whether the state file
    carried the slot's context checkpoints, so this measures it rather than
    assuming either build.
    """

    HANDOFF = True

    def setUp(self):
        super().setUp()
        self.gen = self.start("gpu0_0", slots=1)
        self.read = self.start("cpu1_0", slots=2)
        self.pool_ = self.hush_builder(
            self.pool([self.spec(self.gen, 0, prefill=False, node=0),
                       self.spec(self.read, 1, prefill=True, node=1)]))
        self.url = self.serve(self.pool_)
        self.messages = self.opening("cost") + [{"role": "user", "content": "Hello."}]

    def test_the_slot_really_is_carried_across(self):
        self.turn(self.url, "cost", self.messages)
        self.assertIn("moved", [row["did"] for row in self.pool_.recent],
                      "no slot was carried, so there is nothing to measure")
        self.assertEqual(self.pool_.pins["cost"]["backend"], "gpu0_0")

    def test_what_the_instance_that_generates_then_has_to_read(self):
        """The number the decision to keep the hand-off turns on.

        A move costs a save and a restore of the whole state. It is worth it
        only if the instance that generates then reads almost nothing. If this
        says otherwise, the hand-off is paying a transfer for a full re-read.
        """
        before_read = self.read.mark()
        before_gen = self.gen.mark()
        self.turn(self.url, "cost", self.messages)
        read_here = read_tokens(self.read.since(before_read))
        read_there = read_tokens(self.gen.since(before_gen))
        self.assertGreater(read_here, 500, "nothing was read on the reader")
        rereading = REREAD_LINE in self.gen.since(before_gen)
        if rereading:
            self.assertGreaterEqual(
                read_there, read_here - 4,
                "the backend said it was re-reading everything and did not")
        else:
            self.assertLess(
                read_there, read_here // 8,
                f"the instance that generates read {read_there} of the "
                f"{read_here} tokens carried to it, without the backend "
                f"saying why")

class SavedOpeningsAreLoaded(LiveRouter):
    """One session's system prompt, read once and reused by the next.

    llama-server cannot share a prefix between two conversations, so the
    router keeps openings on disk. The claim is that a second session which
    opens the same way starts from the file instead of reading it again. The
    only honest measure is the second session's prompt eval.
    """

    # One instance, so a hand-off would be a no-op. Said out loud anyway.
    HANDOFF = False

    def setUp(self):
        super().setUp()
        self.be = self.start("cpu1_0", slots=2)
        self.pool_ = self.pool([self.spec(self.be, 0, prefill=True, node=0)])
        self.url = self.serve(self.pool_)
        self.head = self.opening("shared")

    def first_session(self):
        self.turn(self.url, "first", self.head + [{"role": "user", "content": "One."}])

    def test_the_opening_is_read_and_kept(self):
        self.first_session()
        self.assertTrue(wait_for(lambda: bool(self.pool_.openings), patience=180),
                        "no opening was kept")
        name = next(iter(self.pool_.openings.values()))
        kept = router.STORE.slots / name
        self.assertTrue(kept.exists())
        self.assertGreater(kept.stat().st_size, router.PARK_FLOOR)

    def test_a_second_session_starts_from_it_instead_of_reading_it(self):
        self.first_session()
        self.assertTrue(wait_for(lambda: bool(self.pool_.openings), patience=180),
                        "no opening was kept")
        before = self.be.mark()
        self.turn(self.url, "second", self.head + [{"role": "user", "content": "Two."}])
        self.assertIn("loaded opening", [row["did"] for row in self.pool_.recent],
                      "the second session did not load the opening")
        read_again = read_tokens(self.be.since(before))
        cold = self.SYSTEM_CHARS // 4          # roughly what reading it costs
        self.assertLess(read_again, cold // 8,
                        f"the second session read {read_again} tokens, which "
                        f"is most of the {cold} an opening costs to read")

    def test_an_opening_is_only_any_use_to_a_session_that_shares_it(self):
        """The control. A different system prompt has nothing to load."""
        self.first_session()
        self.assertTrue(wait_for(lambda: bool(self.pool_.openings), patience=180))
        before = self.be.mark()
        self.turn(self.url, "stranger",
                  self.opening("different") + [{"role": "user", "content": "Two."}])
        read_again = read_tokens(self.be.since(before))
        self.assertGreater(read_again, self.SYSTEM_CHARS // 8,
                           "a session with a different opening read nothing, "
                           "so the measurement above proves nothing")


class ParkedCachesComeBack(LiveRouter):
    """A cache copied out of a slot before it is lost, and restored elsewhere.

    Every cache on a backend is copied to disk before a new request reaches
    it, because a save can only read a slot that still holds the cache. When
    the conversation returns and its old backend is busy, it takes another
    one, and the copy has to follow it there.
    """

    # A park is about losing a slot, not about where a turn generates.
    # With the hand-off on, a turn could be carried to the other
    # instance and this would be measuring two things at once.
    HANDOFF = False

    def setUp(self):
        super().setUp()
        # One slot each, so a second conversation really does take the first
        # one's slot rather than sitting beside it.
        self.one = self.start("cpu1_0", slots=1)
        self.two = self.start("cpu1_1", slots=1)
        self.pool_ = self.hush_builder(
            self.pool([self.spec(self.one, 0, prefill=True, node=1),
                       self.spec(self.two, 1, prefill=True, node=1)]))
        self.url = self.serve(self.pool_)
        self.head = self.opening("parked")

    def server(self, name):
        return next(s for s in self.servers if s.name == name)

    def displace(self):
        """Run one turn, then fill its backend so the next turn goes elsewhere.

        Returns (the first turn's reply, the instance it ran on, the other)."""
        first = self.turn(self.url, "mine", self.head
                          + [{"role": "user", "content": "One."}])
        home = self.pool_.pins["mine"]["backend"]
        away = "cpu1_1" if home == "cpu1_0" else "cpu1_0"
        # Somebody else arrives on that backend and holds its only slot.
        box = self.start_turn(self.url, "theirs", self.opening("theirs")
                              + [{"role": "user", "content": "Hello."}],
                              max_tokens=200)
        self.assertTrue(wait_for(lambda: self.pool_.pins.get("theirs", {}).get("backend")
                                 == home or "error" in box),
                        "the newcomer never took the first one's backend")
        return first, home, away, box

    def test_the_newcomer_makes_the_router_copy_the_cache_out(self):
        first, home, away, box = self.displace()
        self.assertTrue(wait_for(lambda: bool(self.pool_.pins["mine"]["parked"])),
                        "the cache was never copied out")
        copy = router.STORE.slots / self.pool_.pins["mine"]["parked"]
        self.assertTrue(copy.exists())
        self.assertGreater(copy.stat().st_size, router.PARK_FLOOR)

    def test_the_next_turn_restores_it_on_the_backend_that_serves_it(self):
        first, home, away, box = self.displace()
        self.assertTrue(wait_for(lambda: bool(self.pool_.pins["mine"]["parked"])))
        elsewhere = self.server(away)
        before = elsewhere.mark()
        # The turn that follows has to extend the parked state exactly, so it
        # carries back what the model actually said, not a stand-in for it.
        self.turn(self.url, "mine", self.head
                  + [{"role": "user", "content": "One."},
                     {"role": "assistant", "content": self.said(first)},
                     {"role": "user", "content": "Two."}])
        self.assertIn("recalled", [row["did"] for row in self.pool_.recent],
                      "the copy was never restored")
        read_again = read_tokens(elsewhere.since(before))
        self.assertLess(read_again, self.SYSTEM_CHARS // 8,
                        f"the second turn read {read_again} tokens on the "
                        f"backend its own cache had just been restored onto")


class ASecondConversationCostsTheFirstItsCache(LiveRouter):
    """Not one of the router's stated beliefs. It is the one that was wrong.

    llama.cpp runs with --cache-idle-slots by default. When any task starts,
    every idle slot is copied into the server's own RAM prompt cache and --
    on an instance running --kv-unified, as the production cpu instances do --
    the slot is then CLEARED.

    The router believes a slot keeps what it read until something displaces
    it. On a kv-unified instance nothing has to displace it: another
    conversation starting a turn is enough.

    The way back is closed too. The RAM cache is only consulted when the
    server chooses the slot itself. A request that names a slot -- which is
    what the router does, so it knows which slot to save -- is given that slot
    as it stands, empty, and reads the whole prompt again.

    Both halves run here, so the difference is a measurement rather than an
    argument.
    """

    # One instance. The flag under test here is llama.cpp's, not the
    # router's, so the router's is pinned.
    HANDOFF = False

    def cost_of_a_neighbour(self, extra):
        """One instance, two conversations. `extra` is the llama.cpp flag
        under test, passed here rather than taken from bin/common.sh."""
        be = self.start("cpu1_0", slots=2, unified=True, extra=list(extra))
        pool = self.pool([self.spec(be, 0, prefill=True, node=0)])
        url = self.serve(pool)
        head = self.opening("neighbour")
        first = self.turn(url, "mine", head + [{"role": "user", "content": "One."}])
        # Somebody else starts a turn on the same instance.
        self.turn(url, "theirs", self.opening("other")
                  + [{"role": "user", "content": "Hello."}])
        before = be.mark()
        self.turn(url, "mine", head + [{"role": "user", "content": "One."},
                                       {"role": "assistant", "content": self.said(first)},
                                       {"role": "user", "content": "Two."}])
        return read_tokens(be.since(before))

    def test_with_llama_cpp_defaults_the_second_turn_reads_everything_again(self):
        read_again = self.cost_of_a_neighbour(extra=[])
        self.assertGreater(read_again, self.SYSTEM_CHARS // 8,
                           "the cache survived a neighbouring turn, so this "
                           "build no longer clears idle slots and production "
                           "no longer needs --no-cache-idle-slots")

    def test_with_no_cache_idle_slots_it_keeps_its_cache(self):
        read_again = self.cost_of_a_neighbour(extra=["--no-cache-idle-slots"])
        self.assertLess(read_again, self.SYSTEM_CHARS // 8,
                        f"the second turn still read {read_again} tokens with "
                        f"--no-cache-idle-slots, so something else is "
                        f"emptying the slot")


class DrainUnderLoad(LiveRouter):
    """Taking an instance out of service while requests are running."""

    # A drain is about slots, not about where a turn generates.
    HANDOFF = False

    def setUp(self):
        super().setUp()
        self.one = self.start("cpu1_0", slots=1)
        self.two = self.start("cpu1_1", slots=1)
        self.pool_ = self.hush_builder(
            self.pool([self.spec(self.one, 0, prefill=True, node=1),
                       self.spec(self.two, 1, prefill=True, node=1)]))
        self.url = self.serve(self.pool_)
        self.head = self.opening("drain")

    def test_a_drain_waits_for_the_work_already_running(self):
        box = self.start_turn(self.url, "running",
                              self.head + [{"role": "user", "content": "A long one."}],
                              max_tokens=200)
        self.assertTrue(wait_for(lambda: any(b["busy"] for b in self.pool_.backends)),
                        "the turn never took a slot")
        busy = next(b["name"] for b in self.pool_.backends if b["busy"])
        report = self.pool_.drain(busy, router.http_post, deadline=PATIENCE)
        self.assertTrue(report["quiet"], "the drain gave up on a running turn")
        self.assertNotIn("error", box, f"the turn failed: {box.get('error')}")
        self.assertIn("reply", box)
        self.pool_.resume(busy)

    def test_a_drained_instance_takes_nothing_new(self):
        self.pool_.drain("cpu1_0", router.http_post, deadline=PATIENCE)
        try:
            before = self.one.mark()
            for n in range(3):
                self.turn(self.url, f"after-{n}",
                          self.opening(f"a{n}") + [{"role": "user", "content": "Hi."}])
            self.assertEqual(prompt_evals(self.one.since(before)), [],
                             "a drained instance served a request")
        finally:
            self.pool_.resume("cpu1_0")

    def test_the_opening_builder_leaves_a_drained_instance_alone(self):
        """Two ways in, and both must refuse it.

        `_usable` refuses a draining backend, so no request reaches one. The
        builder asks `_idle_slot` instead, which once checked only that the
        backend was up and had a slot spare -- so a drain reported an instance
        quiet, parked its caches, and the builder then read a whole opening into
        it: work a restart was about to throw away, on an instance somebody was
        waiting to stop. `_idle_slot` now looks at `draining` as well.

        The want is noted here rather than left over from a turn. A turn reads
        its own base opening inline in warm_prefix, so it leaves nothing
        wanted, and the only want a turn can leave is a deep one, which needs
        DEEP_OPENINGS. tests/test_migration.py notes wants the same way."""
        self.pool_.note_want((0, "wanted-by-the-builder"), "base-",
                             self.head[0]["content"], [], self.head[:1],
                             "/v1/chat/completions")
        self.assertTrue(self.pool_.wants, "nothing was wanted, so nothing is proved")
        for name in ("cpu1_0", "cpu1_1"):
            self.pool_.drain(name, router.http_post, deadline=PATIENCE)
        try:
            before = [(s, s.mark()) for s in (self.one, self.two)]
            built = router.Pool.build_once(self.pool_, router.http_post)
            read = sum(read_tokens(s.since(m)) for s, m in before)
            self.assertIsNone(built, "an opening was read into a drained instance")
            self.assertEqual(read, 0, f"a drained instance read {read} tokens "
                                      f"for the opening builder")
        finally:
            for name in ("cpu1_0", "cpu1_1"):
                self.pool_.resume(name)

    def test_a_drain_copies_the_caches_out_before_the_instance_stops(self):
        self.turn(self.url, "living", self.head + [{"role": "user", "content": "Hello."}])
        home = self.pool_.pins["living"]["backend"]
        report = self.pool_.drain(home, router.http_post, deadline=PATIENCE)
        try:
            self.assertGreaterEqual(report["parked"], 1, "the drain parked nothing")
            copy = self.pool_.pins["living"]["parked"]
            self.assertTrue(copy and (router.STORE.slots / copy).exists())
        finally:
            self.pool_.resume(home)


if __name__ == "__main__":
    unittest.main()
