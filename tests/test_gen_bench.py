"""The summary math of tools/gen-bench.py.

No server: each case gives the rows a run would write and checks the
numbers the summary prints from them.
"""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "gen_bench", Path(__file__).resolve().parent.parent / "tools" / "gen-bench.py")
gen_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen_bench)


def row(**over):
    base = {"label": "a", "prompt": "code", "temp": 0.0, "rep": 0,
            "gen_tps": 30.0, "prompt_tps": 100.0, "acceptance": 0.75,
            "predicted_n": 256, "predicted_ms": 1000.0}
    base.update(over)
    return base


class Median(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(gen_bench.median([3, 1, 2]), 2)

    def test_even_is_the_mean_of_the_middle_two(self):
        self.assertEqual(gen_bench.median([4, 1, 3, 2]), 2.5)

    def test_empty_is_none(self):
        self.assertIsNone(gen_bench.median([]))


class Acceptance(unittest.TestCase):
    def test_accepted_over_drafted(self):
        self.assertEqual(gen_bench.acceptance({"draft_n": 40, "draft_n_accepted": 30}), 0.75)

    def test_no_draft_is_none(self):
        self.assertIsNone(gen_bench.acceptance({}))
        self.assertIsNone(gen_bench.acceptance({"draft_n": 0, "draft_n_accepted": 0}))


class PerPromptSummary(unittest.TestCase):
    def test_median_and_spread_per_label_prompt_and_temp(self):
        rows = [row(gen_tps=30.0, acceptance=0.7), row(gen_tps=20.0, acceptance=0.8),
                row(gen_tps=25.0, acceptance=0.9), row(temp=1.0, gen_tps=10.0)]
        out = gen_bench.summarize(rows)
        greedy = out[("a", "code", 0.0)]
        self.assertEqual(greedy["n"], 3)
        self.assertEqual(greedy["gen_tps"], 25.0)
        self.assertEqual(greedy["gen_tps_min"], 20.0)
        self.assertEqual(greedy["gen_tps_max"], 30.0)
        # spread: (max - min) / median
        self.assertAlmostEqual(greedy["gen_tps_spread"], 0.4)
        self.assertAlmostEqual(greedy["acceptance"], 0.8)
        self.assertEqual(out[("a", "code", 1.0)]["n"], 1)

    def test_missing_acceptance_is_skipped(self):
        out = gen_bench.summarize([row(acceptance=None), row(acceptance=0.5)])
        self.assertEqual(out[("a", "code", 0.0)]["acceptance"], 0.5)


class StepBreakdown(unittest.TestCase):
    def test_pooled_over_runs_per_label_and_temp(self):
        # two runs: 10 + 30 steps, 40 steps in all
        r1 = row(gen_steps=10, gen_draft_ms=20.0, gen_decode_ms=60.0, gen_emit_ms=20.0,
                 predicted_n=21, predicted_ms=100.0)
        r2 = row(gen_steps=30, gen_draft_ms=60.0, gen_decode_ms=180.0, gen_emit_ms=60.0,
                 predicted_n=61, predicted_ms=300.0, prompt="prose")
        out = gen_bench.breakdown([r1, r2])
        b = out[("a", 0.0)]
        self.assertEqual(b["steps"], 40)
        self.assertAlmostEqual(b["parts"]["draft"]["ms_per_step"], 2.0)
        self.assertAlmostEqual(b["parts"]["decode"]["ms_per_step"], 6.0)
        self.assertAlmostEqual(b["parts"]["decode"]["pct"], 60.0)
        self.assertAlmostEqual(b["ms_per_step"], 10.0)
        # tokens after the first, per step: (20 + 60) / 40
        self.assertAlmostEqual(b["tokens_per_step"], 2.0)
        self.assertAlmostEqual(b["tps"], 200.0)
        # the parts against the server's own predicted_ms
        self.assertAlmostEqual(b["coverage"], 1.0)
        self.assertNotIn("gap", b["parts"])

    def test_rows_without_trace_are_left_out(self):
        self.assertEqual(gen_bench.breakdown([row()]), {})

    def test_decode_wait_is_a_share_of_decode_not_a_part(self):
        r = row(gen_steps=4, gen_decode_ms=8.0, gen_decode_wait_ms=2.0, gen_emit_ms=2.0,
                predicted_n=9, predicted_ms=10.0)
        b = gen_bench.breakdown([r])[("a", 0.0)]
        self.assertNotIn("decode_wait", b["parts"])
        self.assertAlmostEqual(b["decode_wait_ms_per_step"], 0.5)
        self.assertAlmostEqual(b["ms_per_step"], 2.5)


class ABOrder(unittest.TestCase):
    def test_order_alternates_per_rep(self):
        self.assertEqual(gen_bench.ab_order(["a", "b"], 0), ["a", "b"])
        self.assertEqual(gen_bench.ab_order(["a", "b"], 1), ["b", "a"])
        self.assertEqual(gen_bench.ab_order(["a"], 1), ["a"])


if __name__ == "__main__":
    unittest.main()
