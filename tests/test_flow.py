"""Tests for the flow recorder behind the flow dashboard.

A turn walks queued -> prefill -> generate-queue -> generate -> done. The
recorder keeps one live row per conversation and a newest-first log, so an
animation can replay transitions that happened between two payload pushes.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bin"))

import tempfile
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


def make_pool(backends, **kw):
    """A Pool wired to the sandbox. A case that wants another store, tuning or
    event log passes it, and that one wins."""
    kw.setdefault("store", SANDBOX.store)
    kw.setdefault("tuning", SANDBOX.tuning)
    kw.setdefault("events", SANDBOX.events)
    return router.Pool(backends, **kw)


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
        for i in range(SANDBOX.tuning.flow_log + 40):
            flow.note(f"conv-{i}", "queued")
        self.assertEqual(len(flow.report()["log"]), SANDBOX.tuning.flow_log)


if __name__ == "__main__":
    unittest.main()
