"""The math of tools/expert-cache-sim.py.

The simulator decides whether a fixed set of hot experts on the card is worth
building, so its counts must be exact: the log format, the step grouping, the
two ways to spend a budget, and the hit and byte arithmetic.
"""
import importlib.util
import struct
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "expert_cache_sim", Path(__file__).resolve().parent.parent / "tools" / "expert-cache-sim.py")
sim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sim)


def log(records, t0_us=1_000_000):
    return sim.HEADER_MAGIC + struct.pack("<Q", t0_us) + b"".join(
        sim.encode_record(*r) for r in records)


class FormatTest(unittest.TestCase):
    def test_round_trip(self):
        data = log([(5, 3, 4, {1: 2, 9: 1}, 0), (7, 48, 1, {4095: 1, 2: 1}, 3)])
        t0, recs = sim.parse(data)
        self.assertEqual(t0, 1_000_000)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0], sim.Record(5, 3, 4, ((1, 2), (9, 1)), 0))
        self.assertEqual(recs[1].layer, 48)
        self.assertEqual(recs[1].picks, ((2, 1), (4095, 1)))
        self.assertEqual(recs[1].offload, 3)

    def test_count_bits(self):
        # rows - 1 in the top four bits
        raw = sim.encode_record(0, 0, 8, {5: 8}, 0)
        self.assertEqual(struct.unpack_from("<H", raw, 12)[0], 5 | (7 << 12))

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            sim.parse(b"NOTSTATS" + bytes(8))

    def test_truncated_tail_is_dropped(self):
        data = log([(1, 0, 1, {1: 1}, 0)])
        t0, recs = sim.parse(data + sim.encode_record(2, 1, 1, {2: 1}, 0)[:5])
        self.assertEqual(len(recs), 1)


class StepTest(unittest.TestCase):
    def test_steps_restart_at_a_lower_layer(self):
        recs = [sim.Record(i, l, 4, ((0, 1),), 0) for i, l in enumerate([0, 1, 2, 48, 48, 0, 1, 2, 48])]
        self.assertEqual(sim.assign_steps(recs, mtp_layer=48), [0, 0, 0, 0, 0, 1, 1, 1, 1])

    def test_split_half_by_step(self):
        steps = [0, 0, 1, 1, 2, 2, 3]
        train, test = sim.split_half(steps)
        self.assertEqual(train, {0, 1})
        self.assertEqual(test, {2, 3})


COUNTS = {0: {1: 5, 2: 3}, 1: {7: 4, 8: 1}}


class AllocationTest(unittest.TestCase):
    def test_slots_for_budget(self):
        self.assertEqual(sim.slots_for(1.7, 5.22), 325)
        self.assertEqual(sim.slots_for(0.5, 5.22), 95)

    def test_greedy_takes_the_global_top(self):
        self.assertEqual(sim.hot_greedy(COUNTS, 3), {0: {1, 2}, 1: {7}})

    def test_equal_splits_the_slots(self):
        self.assertEqual(sim.hot_equal(COUNTS, 2, [0, 1]), {0: {1}, 1: {7}})
        # the remainder goes to the lowest layers
        self.assertEqual(sim.hot_equal(COUNTS, 3, [0, 1]), {0: {1, 2}, 1: {7}})
        # a layer never fills more slots than it has seen experts
        self.assertEqual(sim.hot_equal(COUNTS, 8, [0, 1]), {0: {1, 2}, 1: {7, 8}})

    def test_read_counts_count_calls_not_rows(self):
        recs = [sim.Record(0, 0, 4, ((1, 3), (2, 1)), 0), sim.Record(1, 0, 4, ((1, 1),), 0)]
        self.assertEqual(sim.read_counts(recs), {0: {1: 2, 2: 1}})


class EvaluateTest(unittest.TestCase):
    def test_hits_and_bytes(self):
        recs = [sim.Record(0, 0, 4, ((1, 2), (3, 1)), 0), sim.Record(1, 1, 4, ((7, 1),), 0),
                sim.Record(2, 0, 4, ((1, 1), (2, 1)), 0), sim.Record(3, 1, 4, ((8, 3),), 0)]
        hot = {0: {1}, 1: {7}}
        r = sim.evaluate(recs, hot, n_steps=2, expert_bytes=5_222_400, gbps=94.0)
        self.assertAlmostEqual(r["pick_hit_rate"], 4 / 9)
        self.assertAlmostEqual(r["read_per_step"], 3.0)
        self.assertAlmostEqual(r["saved_per_step"], 1.5)
        self.assertAlmostEqual(r["bytes_before"], 3.0 * 5_222_400)
        self.assertAlmostEqual(r["bytes_after"], 1.5 * 5_222_400)
        self.assertAlmostEqual(r["ms_saved"], 1.5 * 5_222_400 / 94e9 * 1e3)

    def test_overlap(self):
        self.assertAlmostEqual(sim.overlap({0: {1, 2}}, {0: {2, 3}}), 0.5)
        self.assertAlmostEqual(sim.overlap({0: {1}}, {1: {1}}), 0.0)
        self.assertAlmostEqual(sim.overlap({}, {}), 1.0)


if __name__ == "__main__":
    unittest.main()
