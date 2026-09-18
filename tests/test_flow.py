"""Tests for the flow recorder behind the flow dashboard.

A turn walks queued -> prefill -> generate-queue -> generate -> done. The
recorder keeps one live row per conversation and a newest-first log, so an
animation can replay transitions that happened between two payload pushes.
"""
import os
import pathlib
import sys
import unittest

# The cache event log is on by default and writes into run/. A test run must
# not add lines a real run would read as its own, so the log stays off unless
# the test is about the log, which sets the variables itself. Read at import,
# so it has to be set before router is imported - and every test module has to
# set it, because whichever one discovery loads first decides for the process.
# This file was the one without it, and discovery only covered for it by
# sorting test_cache_events first. Run any other way - one module, an IDE, a
# -k filter - and the log came on: about 1,073 fixture rows reached
# run/cache-events-20260916.jsonl in two bursts, a third of that day's log,
# and tools/cache-report.py counted them as traffic (903 of them one failing
# park of a conversation called "a" on a backend called "cpu").
os.environ.setdefault("CACHE_LOG", "0")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bin"))

import tempfile
import router

# The same reasoning as CACHE_LOG above, for what this run writes. STORE is the
# default a Pool takes when it is handed none, so a case that forgets to give
# it one reads and writes inside the checkout's own run/slots - where a live
# router keeps conversation caches worth hundreds of gigabytes, and where a
# stray pins.json is adopt()'s instruction to delete every copy it does not
# name. One sandbox for the whole run, under the temporary directory; a case
# wanting its own builds another Store.
router.STORE = router.Store(tempfile.mkdtemp(prefix="router-run-"))


class ATurnWalksItsStages(unittest.TestCase):
    def setUp(self):
        self.flow = router.Flow()

    def test_each_stage_lands_in_live_and_log(self):
        self.flow.note("conv-a", "queued")
        self.flow.note("conv-a", "prefill", "cpu0_0", 2)
        live = self.flow.report()["live"]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["stage"], "prefill")
        self.assertEqual(live[0]["backend"], "cpu0_0")
        self.assertEqual(live[0]["slot"], 2)
        log = self.flow.report()["log"]
        self.assertEqual([row["stage"] for row in log], ["prefill", "queued"])

    def test_since_survives_the_walk_and_changed_follows_it(self):
        self.flow.note("conv-a", "queued")
        since = self.flow.report()["live"][0]["since"]
        self.flow.note("conv-a", "generate", "gpu0_0", 0)
        row = self.flow.report()["live"][0]
        self.assertEqual(row["since"], since)
        self.assertGreaterEqual(row["changed"], since)

    def test_a_note_of_the_same_place_moves_nothing(self):
        self.flow.note("conv-a", "prefill", "cpu0_0", 1)
        self.flow.note("conv-a", "prefill", "cpu0_0", 1)
        self.assertEqual(len(self.flow.report()["log"]), 1)

    def test_done_leaves_live_and_ends_the_log(self):
        self.flow.note("conv-a", "queued")
        self.flow.note("conv-a", "generate", "gpu0_0", 0)
        self.flow.note("conv-a", "done")
        report = self.flow.report()
        self.assertEqual(report["live"], [])
        self.assertEqual(report["log"][0]["stage"], "done")
        self.assertEqual(report["log"][0]["backend"], "gpu0_0")

    def test_done_without_a_turn_records_nothing(self):
        self.flow.note("conv-a", "done")
        self.assertEqual(self.flow.report(), {"live": [], "log": []})

    def test_no_conversation_no_row(self):
        self.flow.note("", "queued")
        self.assertEqual(self.flow.report(), {"live": [], "log": []})


class TwoTurnsOfOneConversation(unittest.TestCase):
    def test_the_next_turn_replaces_the_last_and_keeps_both_in_the_log(self):
        flow = router.Flow()
        flow.note("conv-a", "queued")
        flow.note("conv-a", "generate", "gpu0_0", 0)
        flow.note("conv-a", "done")
        flow.note("conv-a", "queued")
        live = flow.report()["live"]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["stage"], "queued")
        log = flow.report()["log"]
        self.assertEqual([row["stage"] for row in log],
                         ["queued", "done", "generate", "queued"])

    def test_the_log_is_bounded(self):
        flow = router.Flow()
        for i in range(router.FLOW_LOG + 40):
            flow.note(f"conv-{i}", "queued")
        self.assertEqual(len(flow.report()["log"]), router.FLOW_LOG)


if __name__ == "__main__":
    unittest.main()
