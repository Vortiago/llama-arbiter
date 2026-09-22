"""What one poll of a backend says, parsed.

These need no pool, no thread and no socket: each case states what
llama-server answered and reads what it means.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import router


METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 4000
llamacpp:prompt_tokens_cached_total 6000
llamacpp:prompt_seconds_total 40
llamacpp:tokens_predicted_total 900
llamacpp:tokens_predicted_seconds_total 90
llamacpp:n_busy_slots_per_decode 2
llamacpp:n_tokens_max 89848
llamacpp:kv_cache_usage_ratio{slot="0"} 0.5
"""


def slot(**over):
    """One /slots row, with the fields a poll reads."""
    row = {"id": 0, "id_task": 7, "is_processing": True,
           "n_prompt_tokens_cache": 0, "n_prompt_tokens_processed": 0,
           "n_prompt_tokens_total": 1000, "next_token": {"n_decoded": 0}}
    row.update(over)
    return row


class TheCountersAreTheBareOnes(unittest.TestCase):
    """A metric with labels is one series per label, so it has no single
    value. Every counter this router reads is a bare one."""

    def test_a_comment_and_a_labelled_metric_are_skipped(self):
        got = router.counters(METRICS)
        self.assertNotIn("kv_cache_usage_ratio", got)
        self.assertEqual(got["prompt_tokens_total"], 4000)

    def test_the_prefix_llama_server_writes_is_dropped(self):
        self.assertIn("n_tokens_max", router.counters(METRICS))

    def test_a_line_that_is_not_a_number_is_passed_over(self):
        """`nan` and `inf` are numbers to float(), so they are not the case
        here. llama-server writes neither."""
        self.assertEqual(router.counters("a lots\nb 2\n"), {"b": 2.0})


class TheStatsAreWhatTheDashboardShows(unittest.TestCase):
    """The counters are lifetime totals. These are the figures a reader
    compares two backends with."""

    def setUp(self):
        self.stats = router.stats(router.counters(METRICS))

    def test_the_cache_share_counts_cached_against_the_whole_prompt(self):
        self.assertEqual(self.stats["cached"], 60.0)    # 6000 of 10000

    def test_a_rate_is_tokens_over_the_seconds_they_took(self):
        self.assertEqual(self.stats["pp_rate"], 100.0)  # 4000 over 40
        self.assertEqual(self.stats["tg_rate"], 10.0)   # 900 over 90

    def test_the_total_rate_counts_the_slots_that_decode_together(self):
        """A per-request rate times the slots busy per decode is what the box
        is doing, which is what a reader wants to compare."""
        self.assertEqual(self.stats["tg_total"], 20.0)  # 10.0 x 2

    def test_a_backend_that_has_served_nothing_reports_no_rate(self):
        got = router.stats(router.counters("prompt_tokens_total 5\n"))
        self.assertEqual((got["pp_rate"], got["cached"], got["accept"]),
                         (0, 0, 0))

    def test_a_second_of_work_is_too_little_to_divide_by(self):
        got = router.stats(router.counters(
            "prompt_tokens_total 12\nprompt_seconds_total 0.004\n"))
        self.assertEqual(got["pp_rate"], 0)


class ASlotRateNeedsAWindowToResolve(unittest.TestCase):
    """A slot at 0.03 tokens/s does not move between two polls two seconds
    apart, so a rate is measured over rate_window instead."""

    def test_the_first_sight_of_a_slot_reports_no_rate(self):
        _, detail = router.slot_state([slot()], {}, 10.0, 100.0)
        self.assertIsNone(detail[0]["pp_rate"])
        self.assertIsNone(detail[0]["tg_rate"])

    def test_a_rate_arrives_once_the_window_has_passed(self):
        sample, _ = router.slot_state([slot()], {}, 10.0, 100.0)
        sample, detail = router.slot_state(
            [slot(n_prompt_tokens_processed=500)], sample, 10.0, 110.0)
        self.assertEqual(detail[0]["pp_rate"], 50.0)   # 500 over 10 seconds

    def test_a_new_task_in_the_slot_starts_the_count_again(self):
        sample, _ = router.slot_state(
            [slot(n_prompt_tokens_processed=500)], {}, 10.0, 100.0)
        sample, _ = router.slot_state(
            [slot(id_task=8, n_prompt_tokens_processed=20)], sample, 10.0, 101.0)
        self.assertEqual(sample[0]["done_p"], 20)


class WhatIsLeftToReadIsWhatTheDashboardCountsDown(unittest.TestCase):
    """`prompt` is what a slot has still to read. Its two arithmetics differ:
    one needs patches/slots-report-the-prompt-size.patch."""

    def test_a_patched_backend_reports_the_prompt_it_arrived_with(self):
        _, detail = router.slot_state(
            [slot(n_prompt_tokens_total=1000, n_prompt_tokens_cache=600,
                  n_prompt_tokens_processed=100)], {}, 10.0, 100.0)
        self.assertEqual(detail[0]["prompt"], 300)

    def test_an_unpatched_backend_falls_back_to_what_it_does_report(self):
        """n_prompt_tokens grows with every token generated, so the generated
        ones come off before the cached ones."""
        row = slot(n_prompt_tokens_cache=600, next_token={"n_decoded": 50})
        row.pop("n_prompt_tokens_total")
        row["n_prompt_tokens"] = 1000
        _, detail = router.slot_state([row], {}, 10.0, 100.0)
        self.assertEqual(detail[0]["prompt"], 350)     # 1000 - 50 - 600

    def test_a_slot_says_which_of_the_three_things_it_is_doing(self):
        idle = router.slot_state([slot(is_processing=False)], {}, 10.0, 1.0)[1]
        reading = router.slot_state([slot()], {}, 10.0, 1.0)[1]
        generating = router.slot_state(
            [slot(next_token={"n_decoded": 3})], {}, 10.0, 1.0)[1]
        self.assertEqual([idle[0]["phase"], reading[0]["phase"],
                          generating[0]["phase"]],
                         ["idle", "reading", "generating"])

    def test_an_older_build_sends_the_next_token_as_an_array(self):
        _, detail = router.slot_state(
            [slot(next_token=[{"n_decoded": 4}])], {}, 10.0, 1.0)
        self.assertEqual(detail[0]["decoded"], 4)


if __name__ == "__main__":
    unittest.main()
