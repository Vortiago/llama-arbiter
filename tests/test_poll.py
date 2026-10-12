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


class HowFastABackendReads(unittest.TestCase):
    """The lifetime prompt rate, from the bare counters. The router ranks
    readers by it, so a reset of the dashboard's averages must not move it."""

    def test_tokens_read_over_the_seconds_spent_reading(self):
        self.assertEqual(router.read_rate(router.counters(METRICS)), 100.0)

    def test_nothing_before_a_second_of_reading(self):
        self.assertIsNone(router.read_rate({"prompt_tokens_total": 12,
                                            "prompt_seconds_total": 0.5}))

    def test_nothing_from_a_backend_that_has_not_answered(self):
        self.assertIsNone(router.read_rate({}))

    def test_nothing_from_a_few_short_prompts(self):
        """Per-request overhead dominates a short prompt. koishi's gpu read
        46 tokens of smoke tests in 1.5 s, 30 tokens/s, and ranked below the
        cpus (35) it reads three times as fast as; never chosen, it never
        read enough to be measured again."""
        self.assertIsNone(router.read_rate({"prompt_tokens_total": 46,
                                            "prompt_seconds_total": 1.5}))


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
        self.assertEqual(detail[0]["whole"], 1000)

    def test_an_unpatched_backend_falls_back_to_what_it_does_report(self):
        """n_prompt_tokens grows with every token generated, so the generated
        ones come off before the cached ones."""
        row = slot(n_prompt_tokens_cache=600, next_token={"n_decoded": 50})
        row.pop("n_prompt_tokens_total")
        row["n_prompt_tokens"] = 1000
        _, detail = router.slot_state([row], {}, 10.0, 100.0)
        self.assertEqual(detail[0]["prompt"], 350)     # 1000 - 50 - 600
        self.assertIsNone(detail[0]["whole"], "the fallback is not the prompt's size")

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


class TheDraftPositionsSayHowLongDraftsRun(unittest.TestCase):
    """One labelled counter: accepted draft tokens by position. Position i
    counts the steps that accepted at least i+1, so the counts fall off and
    their shape is how long a draft runs."""

    METRICS = """\
llamacpp:spec_decode_num_draft_tokens_total 1000
llamacpp:spec_decode_num_accepted_tokens_total 400
llamacpp:spec_decode_num_drafts_total 250
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 300
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 80
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 20
"""

    def test_the_labelled_series_is_read_by_name(self):
        self.assertEqual(router.accepted_positions(self.METRICS),
                         {0: 300.0, 1: 80.0, 2: 20.0})

    def test_the_bare_counters_still_skip_it(self):
        self.assertNotIn("spec_decode_num_accepted_tokens_per_pos_total",
                         router.counters(self.METRICS))

    def test_the_mean_accepted_length_and_the_position_shares(self):
        got = router.stats(router.counters(self.METRICS),
                           router.accepted_positions(self.METRICS))
        self.assertEqual(got["accept_mean"], 1.6)      # 400 accepted / 250 steps
        self.assertEqual(got["accept_shares"], [75.0, 20.0, 5.0])

    def test_the_mean_survives_a_reset_of_the_since_reset_counters(self):
        """/router/reset-rates zeroes the since-reset steps while the labelled
        positions stay lifetime. The mean must divide lifetime accepted by
        LIFETIME drafts, or it reads 400/1 after a reset instead of 400/250."""
        value = router.counters(self.METRICS)
        reset = dict(value, spec_decode_num_drafts_total=1)   # one step since
        got = router.stats(reset, router.accepted_positions(self.METRICS),
                           value)
        self.assertEqual(got["accept_mean"], 1.6)

    def test_no_positions_no_shape(self):
        got = router.stats(router.counters("prompt_tokens_total 5\n"))
        self.assertEqual((got["accept_mean"], got["accept_shares"]), (0, []))


if __name__ == "__main__":
    unittest.main()
