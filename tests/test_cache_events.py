"""Tests for the cache event log.

The log is the record every cache improvement will be argued from, so the
rules it plays by are the behaviour under test: a line per event, nothing
that can block a request, no prompt text, and every hook writing what its
call site actually decided.
"""
import atexit
import json
import shutil
import subprocess
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

# Nothing else deletes this. The path is read now rather than at exit, because
# a case may point SANDBOX.store somewhere else and put it back.
atexit.register(shutil.rmtree, SANDBOX.store.run, ignore_errors=True)


class QuietLink:
    """A link that answers every operation without a socket. `written` is what
    a save reports, because the router judges a real copy by that number."""

    def __init__(self, written=0):
        self.written = written

    def save(self, be, slot, name, timeout=None):
        return {"n_written": self.written}

    def restore(self, be, slot, name, timeout=None):
        return {}

    def prefill(self, be, block, slot, alive, timeout=None):
        return {}

    def render(self, be, route, payload, alive, timeout=None):
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
    """The checkpoint line is the one number that proves the reuse is real.
    It is counted, not written as an event: one line per restored slot was
    71,644 of the log over nine days."""

    def test_a_restored_checkpoint_is_an_event_with_its_position(self):
        line = ("12:00:00.123 I test: restored context checkpoint "
                "(pos_min = 110310, pos_max = 110310, n_tokens = 110311, "
                "n_past = 110311, size = 328.893 MiB)")
        self.assertEqual(router.cache_event(line), ("checkpoint", 110311.0))

    def test_a_checkpoint_is_counted_but_not_sunk(self):
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
        self.assertNotIn(("checkpoint", 10.0), seen)
        self.assertIn(("evicted", 3.0), seen)
        self.assertEqual(watch.stats["checkpoints"], 1)
        self.assertEqual(watch.stats["evictions"], 1)


class TheStartupFitAndACrashAreRead(unittest.TestCase):
    """Two lines that appear in no other file: what the backend projected it
    would need, and why it died."""

    def test_the_fit_line_is_memory_against_free_memory(self):
        line = ("0.03.586 I common_params_fit_impl: projected to use 13361 MiB "
                "of device memory vs. 15018 MiB of free device memory")
        self.assertEqual(router.cache_event(line), ("fit", (13361.0, 15018.0)))

    def test_a_crash_is_kept_as_the_last_error(self):
        self.assertEqual(router.cache_event("CUDA error: out of memory"),
                         ("error", "CUDA error: out of memory"))
        self.assertEqual(router.cache_event("failed to allocate a buffer"),
                         ("error", "failed to allocate a buffer"))

    def test_a_watch_keeps_the_fit_and_the_error(self):
        log_file = Path(tempfile.mkdtemp()) / "gpu.log"
        watch = router.CacheWatch(log_file)
        log_file.write_text(
            "projected to use 100 MiB of device memory vs. 40 MiB of free "
            "device memory\n"
            "CUDA error: out of memory\n")
        watch.poll()
        self.assertEqual((watch.stats["fit_used_mib"],
                          watch.stats["fit_free_mib"]), (100.0, 40.0))
        self.assertEqual(watch.stats["last_error"], "CUDA error: out of memory")


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
        # Each put back as a cleanup, beside the line that took it: a tearDown
        # does not run when setUp throws, so the two lines below used to leave
        # every later case in this file writing into a deleted directory.
        self.addCleanup(setattr, SANDBOX, "events", self.old)
        # Linking and deleting slot files must not touch the real run dir.
        self.old_store = SANDBOX.store
        SANDBOX.store = router.Store(self.dir)
        self.addCleanup(setattr, SANDBOX, "store", self.old_store)
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu",
                                  "pref": 0}],
                                store=SANDBOX.store, watch=False)

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

    def test_the_usage_chunk_carries_the_timings_and_the_fingerprint(self):
        chunk = ("data: " + json.dumps(
            {"choices": [], "system_fingerprint": "fp-9",
             "usage": {"prompt_tokens": 500, "completion_tokens": 3},
             "timings": {"prompt_n": 480, "predicted_n": 3,
                         "predicted_ms": 100.0}})
            + "\n\n").encode()
        tee = router.OaiUsageSplice(strip=True)
        tee.feed(chunk)
        self.assertEqual(tee.timings["predicted_n"], 3)
        self.assertEqual(tee.system_fingerprint, "fp-9")

    def test_the_fingerprint_is_kept_from_any_chunk(self):
        chunk = (b"data: " + json.dumps(
            {"choices": [{"delta": {"content": "hi"}}],
             "system_fingerprint": "fp-1"}).encode() + b"\n\n")
        tee = router.OaiUsageSplice(strip=True)
        tee.feed(chunk)
        self.assertEqual(tee.system_fingerprint, "fp-1")

    def test_a_non_streamed_reply_is_read_whole(self):
        body = json.dumps({"usage": {"prompt_tokens": 5},
                           "timings": {"predicted_n": 2},
                           "system_fingerprint": "fp-3"}).encode()
        usage, timings, fingerprint = router.read_oai(body)
        self.assertEqual(usage["prompt_tokens"], 5)
        self.assertEqual(timings["predicted_n"], 2)
        self.assertEqual(fingerprint, "fp-3")
        self.assertEqual(router.read_oai(b"not json"), ({}, {}, None))

    def test_a_timings_that_is_not_an_object_is_dropped(self):
        """A reply that sets timings to a list must not carry it into the
        turn record: `.get` on a list raises in the turn's finally."""
        body = json.dumps({"usage": {"prompt_tokens": 5},
                           "timings": [1, 2, 3]}).encode()
        usage, timings, _ = router.read_oai(body)
        self.assertEqual(usage["prompt_tokens"], 5)
        self.assertEqual(timings, {})

    def test_a_data_payload_that_is_not_an_object_is_passed_through(self):
        """A `data:` line that parses to a list has no .get. It is not an
        openai chunk, so it must reach the client unchanged."""
        tee = router.OaiUsageSplice(strip=True)
        chunk = b'data: ["usage", "noise"]\n\n'
        out = tee.feed(chunk) + tee.tail()
        self.assertEqual(out, chunk)
        self.assertEqual(tee.usage, {})

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


class TheTurnReportReadsTheLogBack(unittest.TestCase):
    """tools/turn-report.py groups the request events by day, build and
    backend. Its percentiles are computed offline, so a tiny fixture is
    enough."""

    def test_it_groups_by_build_and_backend_and_prints_percentiles(self):
        where = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, where, ignore_errors=True)
        rows = [
            {"ts": 1700000000.0, "event": "request", "build": "b1",
             "backend": "cpu", "took": 10.0, "waited": 1.0, "pre_ms": 5.0,
             "tg": 20.0, "accept": 50.0, "recall_s": 2.0, "handoff_s": 3.0},
            {"ts": 1700000100.0, "event": "request", "build": "b1",
             "backend": "cpu", "took": 30.0, "waited": 3.0, "pre_ms": 15.0,
             "tg": 40.0, "accept": 70.0, "recall_s": 4.0, "handoff_s": 5.0},
            {"ts": 1700000200.0, "event": "request", "build": "b2",
             "backend": "gpu", "took": 20.0, "first_token_s": 0.5},
            {"ts": 1700000300.0, "event": "park", "ok": False,
             "error": "backend said no"},
        ]
        (Path(where) / "cache-events-20231114.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
        tool = (Path(__file__).resolve().parent.parent
                / "tools" / "turn-report.py")
        out = subprocess.run([sys.executable, str(tool), "--dir", where],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("build b1", out)
        self.assertIn("cpu", out)
        self.assertIn("took", out)
        self.assertIn("p50", out)
        self.assertIn("first_token_s", out)     # present in one row only
        self.assertIn("park: backend said no", out)

    def run_tool(self, lines):
        where = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, where, ignore_errors=True)
        (Path(where) / "cache-events-20231114.jsonl").write_text(
            "\n".join(lines) + "\n")
        tool = (Path(__file__).resolve().parent.parent
                / "tools" / "turn-report.py")
        return subprocess.run([sys.executable, str(tool), "--dir", where],
                              capture_output=True, text=True, check=True).stdout

    def test_a_carried_turn_is_filed_under_its_generator(self):
        """`backend`/`build` describe the reader and `serving`/`serving_build`
        the generator, so a carried turn must not be filed under the reader
        with the generator's build."""
        out = self.run_tool([
            json.dumps({"ts": 1700000000.0, "event": "request",
                        "backend": "cpu", "build": "reader-1",
                        "serving": "gpu", "serving_build": "gen-9",
                        "took": 10.0, "waited": 1.0})])
        self.assertIn("build gen-9  gpu", out)
        self.assertNotIn("build reader-1", out)

    def test_a_non_dict_line_is_skipped(self):
        """A line that parses to a list or a scalar is not an event."""
        out = self.run_tool([
            "[1, 2, 3]",
            json.dumps({"ts": 1700000000.0, "event": "request",
                        "backend": "cpu", "build": "b1", "took": 10.0})])
        self.assertIn("build b1", out)


if __name__ == "__main__":
    unittest.main()
