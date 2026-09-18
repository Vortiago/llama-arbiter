"""Tests for the cache event log.

The log is the record every cache improvement will be argued from, so the
rules it plays by are the behaviour under test: a line per event, nothing
that can block a request, no prompt text, and every hook writing what its
call site actually decided.
"""
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import pathlib
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


class QuietLink:
    """A link that answers every operation without a socket. `written` is what
    a save reports, because the router judges a real copy by that number."""

    def __init__(self, written=0):
        self.written = written

    def save(self, be, slot, name, timeout=None):
        return {"n_written": self.written}

    def restore(self, be, slot, name, timeout=None):
        return {}

    def prefill(self, be, block, slot, timeout=None):
        return {}

    def render(self, be, route, payload, timeout=None):
        return {"prompt": ""}


def make_pool(backends, **kw):
    """A Pool wired to the sandbox. A case that wants another store, tuning or
    event log passes it, and that one wins."""
    kw.setdefault("store", SANDBOX.store)
    kw.setdefault("tuning", SANDBOX.tuning)
    kw.setdefault("events", SANDBOX.events)
    return router.Pool(backends, **kw)


def rows_of(directory):
    """Every row in every log file here, oldest first."""
    rows = []
    for path in sorted(Path(directory).glob("cache-events-*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


class ALogLineHoldsOneEvent(unittest.TestCase):
    """The format the whole point rests on: json per line, no prompt text."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = router.EventLog(directory=self.dir, on=True)

    def test_a_written_event_lands_as_json_after_a_flush(self):
        self.log.write("choice", conv="abcd1234", plan="load", stored=12)
        self.log.flush()
        rows = rows_of(self.dir)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "choice")
        self.assertEqual(rows[0]["plan"], "load")
        self.assertIsInstance(rows[0]["ts"], float)

    def test_fields_that_are_none_are_left_out(self):
        self.log.write("load", key="k1", shelf=None, ok=True)
        self.log.flush()
        self.assertNotIn("shelf", rows_of(self.dir)[0])

    def test_a_log_that_is_off_writes_nothing(self):
        quiet = router.EventLog(directory=self.dir, on=False)
        quiet.write("choice", conv="x")
        quiet.flush()
        self.assertEqual(rows_of(self.dir), [])


class ARequestNeverWaitsForTheLog(unittest.TestCase):
    """Telemetry that can hold up a turn has cost too much."""

    def test_a_full_queue_drops_events_instead_of_blocking(self):
        # Built off, so no writer thread empties the queue, then switched on:
        # with the thread running it consumed the filler row before write() was
        # reached, and the row had no `ts` so _emit raised and counted the drop
        # itself. Both assertions then held even with a blocking put.
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        log = router.EventLog(directory=directory, on=False, maxsize=1)
        log.on = True
        log.queue.put({"ts": time.time(), "event": "holds the one place"})
        began = time.time()
        log.write("choice", conv="x")
        self.assertLess(time.time() - began, 0.05)
        self.assertEqual(log.dropped, 1)
        self.assertEqual(log.queue.qsize(), 1, "the queue took a second row")


class TheBackendSaysWhatHappened(unittest.TestCase):
    """The checkpoint line is the one number that proves the reuse is real."""

    def test_a_restored_checkpoint_is_an_event_with_its_position(self):
        line = ("12:00:00.123 I test: restored context checkpoint "
                "(pos_min = 110310, pos_max = 110310, n_tokens = 110311, "
                "n_past = 110311, size = 328.893 MiB)")
        self.assertEqual(router.cache_event(line), ("checkpoint", 110311.0))

    def test_a_watch_passes_sunk_events_and_counts_them(self):
        log_file = Path(tempfile.mkdtemp()) / "cpu.log"
        seen = []
        watch = router.CacheWatch(log_file,
                                  sink=lambda kind, value: seen.append(
                                      (kind, value)))
        log_file.write_text(
            "restored context checkpoint (pos_min = 9, pos_max = 9, "
            "n_tokens = 10, n_past = 10, size = 1.0 MiB)\n"
            "removing oldest entry (size = 3.0 MiB)\n")
        watch.poll()
        self.assertIn(("checkpoint", 10.0), seen)
        self.assertIn(("evicted", 3.0), seen)
        self.assertEqual(watch.stats["checkpoints"], 1)
        self.assertEqual(watch.stats["evictions"], 1)


class TheLogFollowsWhatThePoolDecided(unittest.TestCase):
    """Each hook must say what its call site knew, in the row it writes."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # Registered before anything else this setUp does, so a throw part way
        # through still restores the globals below and takes the directory
        # with it. The tearDown here restored the globals and left the
        # directory; CacheWatching in test_migration.py removes its own.
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = router.EventLog(directory=self.dir, on=True)
        self.old = SANDBOX.events
        SANDBOX.events = self.log
        # Linking and deleting slot files must not touch the real run dir.
        self.old_store = SANDBOX.store
        SANDBOX.store = router.Store(self.dir)
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu",
                                  "pref": 0}],
                                store=SANDBOX.store, watch=False)

    def tearDown(self):
        SANDBOX.events = self.old
        SANDBOX.store = self.old_store

    def test_an_unshared_deep_cut_is_written_as_a_fork_of_its_holder(self):
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.holds[("cpu", 0)] = {"k2"}
        self.pool.pins["parent"] = {"backend": "cpu", "slot": 0}
        self.pool._load_prefix = lambda *a, **k: True
        self.pool.link = QuietLink()
        self.pool.warm_prefix("child", [(0, "k1"), (3, "k2")],
                              [{"role": "user", "content": "x"}], "", [],
                              self.pool.backends[0], 0, "/v1/chat/completions")
        self.log.flush()
        forks = [r for r in rows_of(self.dir) if r["event"] == "fork"]
        self.assertEqual(len(forks), 1)
        self.assertEqual(forks[0]["parent"], "parent")
        self.assertEqual(forks[0]["depth"], 3)
        choice = [r for r in rows_of(self.dir) if r["event"] == "choice"][-1]
        self.assertEqual(choice["plan"], "load")
        self.assertEqual(choice["shelf"], "base")

    def test_a_want_says_added_while_it_waits_and_built_when_it_lands(self):
        self.pool.note_want((0, "k9"), "base-", "", [], [], "/completion")
        self.log.flush()
        added = [r for r in rows_of(self.dir)
                 if r["event"] == "want" and r["action"] == "added"][-1]
        self.assertEqual(added["shelf"], "base")
        self.assertEqual(self.pool.wants["k9"]["cut"][1], "k9")

        class Reads(QuietLink):
            def prefill(inner, be, block, slot, timeout=None):
                return {"timings": {"prompt_n": 4200, "cache_n": 0}}

        post = Reads(SANDBOX.tuning.park_floor + 1)
        self.pool._render_block = lambda *a, **k: "rendered"
        # The builder only reads into a slot the poll has found idle twice,
        # so the backend is given a slots_detail that says exactly that.
        self.pool.backends[0].update(
            up=True, busy=0, slots=1,
            slots_detail=[{"id": 0, "busy": False}],
            idle_runs={0: SANDBOX.tuning.idle_polls})
        self.pool.wants["k9"]["at"] = time.time() - 30
        self.pool.link = post
        self.pool.build_once(remove=lambda name: None)
        self.log.flush()
        built = [r for r in rows_of(self.dir)
                 if r["event"] == "want" and r["action"] == "built"][-1]
        self.assertTrue(built["ok"])
        self.assertGreaterEqual(built["age"], 30)
        build = [r for r in rows_of(self.dir) if r["event"] == "build"][-1]
        self.assertTrue(build["ok"])
        self.assertEqual(build["prompt_n"], 4200)

    def test_a_dropped_want_says_how_long_it_waited_unbuilt(self):
        for i in range(SANDBOX.tuning.want_keep + 1):
            self.pool.note_want((0, f"k{i}"), "deep-", "", [], [], "/completion")
        self.log.flush()
        dropped = [r for r in rows_of(self.dir)
                   if r["event"] == "want" and r["action"] == "dropped"]
        self.assertEqual(len(dropped), 1)

    def test_a_load_says_which_shelf_and_how_long_the_read_took(self):
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.link = QuietLink()
        self.assertTrue(self.pool._load_prefix("k1", "base-k1.park",
                                               self.pool.backends[0], 0))
        self.log.flush()
        load = [r for r in rows_of(self.dir) if r["event"] == "load"][-1]
        self.assertEqual(load["shelf"], "base")
        self.assertTrue(load["ok"])
        self.assertEqual(load["loads"], 1)

    def test_a_park_and_a_recall_record_the_copy_moving(self):
        self.pool.pins["conv1"] = {"backend": "cpu", "slot": 0,
                                   "inflight": True, "parked": None}
        self.pool.link = QuietLink(SANDBOX.tuning.park_floor + 1)
        self.assertTrue(self.pool._save_park("conv1", self.pool.backends[0],
                                             0, remove=lambda n: None))
        self.pool.pins["conv1"]["slot"] = None
        self.pool.recall("conv1", self.pool.backends[0], 1)
        self.log.flush()
        park = [r for r in rows_of(self.dir) if r["event"] == "park"][-1]
        recall = [r for r in rows_of(self.dir) if r["event"] == "recall"][-1]
        self.assertTrue(park["ok"])
        self.assertTrue(recall["ok"])
        self.assertEqual(recall["conv"], "conv1")


class AnAnthropicStreamReportsWhatItWasTold(unittest.TestCase):
    """The usage figures pass through the splice anyway; the log keeps them."""

    def test_the_counts_the_client_sees_are_the_counts_logged(self):
        splice = router.AnthropicSplice()
        splice.feed(router.sse_event("message_start", {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 90,
                                  "cache_read_input_tokens": 60}}}))
        splice.feed(router.sse_event("message_delta", {
            "type": "message_delta", "usage": {"output_tokens": 7}}))
        self.assertEqual(splice.reported,
                         {"input_tokens": 90,
                          "cache_read_input_tokens": 60,
                          "output_tokens": 7})


class TheStreamKeepsWhatItShould(unittest.TestCase):
    """The openai usage chunk is asked for by the router and belongs to the
    router, unless the client asked for it itself."""

    CONTENT = router.sse_event("message", {"choices": [{"delta": {"content": "hi"}}]})
    DONE = b"data: [DONE]\n\n"

    def usage_chunk(self):
        return ("data: " + json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 500,
                                      "completion_tokens": 3,
                                      "prompt_tokens_details":
                                          {"cached_tokens": 480}}})
                + "\n\n").encode()

    def test_the_usage_chunk_is_taken_out_and_read(self):
        tee = router.OaiUsageSplice(strip=True)
        out = tee.feed(self.CONTENT + self.usage_chunk() + self.DONE)
        out += tee.tail()
        self.assertEqual(out, self.CONTENT + self.DONE)
        self.assertEqual(tee.usage["prompt_tokens_details"]["cached_tokens"], 480)

    def test_a_client_that_asked_keeps_its_chunk(self):
        tee = router.OaiUsageSplice(strip=False)
        out = tee.feed(self.CONTENT + self.usage_chunk())
        out += tee.tail()
        self.assertIn(b'"usage"', out)
        self.assertEqual(tee.usage["prompt_tokens"], 500)

    def test_an_unasked_for_stream_passes_through_untouched(self):
        tee = router.OaiUsageSplice(strip=True)
        parts = self.CONTENT[:20], self.CONTENT[20:], self.DONE
        out = b"".join(tee.feed(p) for p in parts) + tee.tail()
        self.assertEqual(out, self.CONTENT + self.DONE)
        self.assertEqual(tee.usage, {})

    def test_the_ask_sets_the_option_and_nothing_else_changes(self):
        body = json.dumps({"model": "m", "messages": [], "stream": True}).encode()
        injected = json.loads(router.with_usage(body))
        self.assertTrue(injected["stream_options"]["include_usage"])
        self.assertEqual(injected["model"], "m")
        self.assertFalse(router.wants_usage(body))
        self.assertTrue(router.wants_usage(json.dumps(
            {"stream_options": {"include_usage": True}}).encode()))

    def test_a_body_that_is_not_json_is_left_alone(self):
        self.assertIsNone(router.with_usage(b"not json"))
        self.assertFalse(router.wants_usage(b"not json"))


class ARequestNamesItsClient(unittest.TestCase):
    def test_the_two_agents_are_named(self):
        self.assertEqual(router.client_kind({"User-Agent": "opencode/1.14"}),
                         "opencode")
        self.assertEqual(router.client_kind({"User-Agent": "claude-cli/2.0"}),
                         "claude-code")

    def test_anything_else_keeps_its_first_word(self):
        self.assertEqual(router.client_kind({"User-Agent": "curl/8.0"}), "curl")
        self.assertIsNone(router.client_kind({}))


if __name__ == "__main__":
    unittest.main()
