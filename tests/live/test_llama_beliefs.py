"""What the router believes about llama.cpp, checked against llama.cpp.

Every one of these is a belief the router acts on. The stub in
tests/fake_backend.py was written from the same beliefs, so it agrees with the
router whether or not the real server does. That is how four features stayed
dead in production with a green suite.

So nothing here asserts on what the router logged. It asserts on what the
backend reports: the tokens in its own `prompt eval time = ... / N tokens`
lines, the `cache_n` and `prompt_n` of its timings, the `n_prompt_tokens` of
its slots.

tests/live/README.md says how to run them. They need a model and a few
minutes, and `unittest discover -s tests` does not reach them.
"""

import json
import math
import threading
import time
import unittest

from harness import (REREAD_LINE, HttpError, LiveCase, filler, prompt_evals,
                     prose, read_tokens, tokens_read, tokens_reused)


class TheReadPass(LiveCase):
    """Belief 3: `n_predict: 0` prefills and leaves the slot holding the prompt.

    This one matters because of what it is for. A generated token lands in the
    slot, so a slot that generated holds prompt + 1, and the next request
    carries the prompt alone, which is a prefix. Rewinding into a prefix needs
    a checkpoint, and a restored slot has none. So the read pass must leave the
    slot holding exactly the prompt and nothing more.
    """

    def setUp(self):
        super().setUp()
        self.be = self.start("read", slots=2)

    def test_the_slot_holds_exactly_the_prompt(self):
        reply = self.be.prefill(filler(300), id_slot=0, n_predict=0)
        self.assertEqual(self.be.held(0), tokens_read(reply),
                         "the slot holds something other than the prompt")

    def test_it_still_returns_one_sampled_token(self):
        """The belief as written is not quite true, and the difference is real.

        `n_predict: 0` does sample a token and put it in the reply. What it
        does not do is feed that token back, so the slot never sees it. The
        router throws the answer away, so this costs nothing -- but a test
        that asserted "generates nothing" against the reply would fail."""
        reply = self.be.prefill(filler(300), id_slot=0, n_predict=0)
        self.assertEqual(reply["tokens_predicted"], 1)
        self.assertNotEqual(reply["content"], "",
                            "n_predict 0 returned no token at all")

    def test_max_tokens_zero_does_the_same_on_the_openai_endpoint(self):
        body = {"model": "live", "messages": [{"role": "user", "content": prose(2000)}],
                "max_tokens": 0, "stream": False, "id_slot": 1}
        reply = self.be.post("/v1/chat/completions", body)
        read = reply["timings"]["prompt_n"]
        self.assertGreater(read, 100, "nothing was read at all")
        self.assertEqual(self.be.held(1), read,
                         "the slot holds something other than the prompt")

    def test_max_tokens_zero_does_the_same_on_the_anthropic_endpoint(self):
        """The endpoint Claude Code uses. Its body goes through a whitelist,
        so nothing here can be assumed from the OpenAI path."""
        body = {"model": "live", "messages": [{"role": "user", "content": prose(2000)}],
                "max_tokens": 0, "stream": False, "id_slot": 1}
        reply = self.be.post("/v1/messages", body)
        usage = reply["usage"]
        held = self.be.held(1)
        self.assertEqual(held, usage["input_tokens"] + usage["cache_read_input_tokens"],
                         "the slot holds something other than the prompt")
        self.assertEqual(usage["output_tokens"], 1)


class NamingTheSlot(LiveCase):
    """Belief 4: `id_slot` is honoured on all three request paths.

    The router names the slot rather than reading it back out of the reply,
    because `/v1/messages` never emits it. On that endpoint the body is
    converted through a whitelist, and `id_slot` only survives because of
    patches/anthropic-pass-id-slot.patch. If the binary in llama.cpp-mtp/build
    were rebuilt from unpatched source, this is the test that would notice.
    """

    def setUp(self):
        super().setUp()
        self.be = self.start("named", slots=2)

    def landed_on(self, path, body, want):
        before = self.be.mark()
        self.be.post(path, body)
        evals = prompt_evals(self.be.since(before))
        self.assertTrue(evals, f"{path} read nothing, so it proves nothing")
        return {row["slot"] for row in evals}

    def test_completion_honours_id_slot(self):
        for want in (1, 0):
            body = {"prompt": filler(200, f"c{want}"), "n_predict": 0,
                    "cache_prompt": True, "id_slot": want}
            self.assertEqual(self.landed_on("/completion", body, want), {want})

    def test_chat_completions_honours_id_slot(self):
        for want in (1, 0):
            body = {"model": "live", "max_tokens": 0, "stream": False, "id_slot": want,
                    "messages": [{"role": "user", "content": prose(1200, f"oai{want}")}]}
            self.assertEqual(self.landed_on("/v1/chat/completions", body, want), {want})

    def test_the_anthropic_endpoint_honours_id_slot(self):
        """This is the one the local patch exists for."""
        for want in (1, 0):
            body = {"model": "live", "max_tokens": 0, "stream": False, "id_slot": want,
                    "messages": [{"role": "user", "content": prose(1200, f"ant{want}")}]}
            self.assertEqual(
                self.landed_on("/v1/messages", body, want), {want},
                "the anthropic endpoint dropped id_slot: is "
                "patches/anthropic-pass-id-slot.patch in this build?")

    def test_the_anthropic_reply_still_never_names_its_slot(self):
        """Why the patch was needed. `verbose` does not come back either."""
        reply = self.be.post("/v1/messages",
                             {"model": "live", "max_tokens": 1, "stream": False,
                              "verbose": True,
                              "messages": [{"role": "user", "content": "hello"}]})
        self.assertNotIn("id_slot", reply)
        self.assertNotIn("__verbose", reply)


class ExtendingASlot(LiveCase):
    """Beliefs 5 and 6: what a slot will extend, and what a token costs."""

    def setUp(self):
        super().setUp()
        self.be = self.start("extend", slots=1)
        self.prompt = filler(300)
        self.be.prefill(self.prompt, id_slot=0, n_predict=0)

    def test_an_exact_extension_reads_only_what_is_new(self):
        held = self.be.held(0)
        reply = self.be.prefill(self.prompt + " and a little more", id_slot=0)
        self.assertEqual(tokens_reused(reply), held,
                         "the shared part was read again")
        self.assertLess(tokens_read(reply), 10, "it read more than the new tail")

    def test_a_repeat_with_nothing_new_cannot_reuse_the_last_token(self):
        """The off-by-one that is easy to miss.

        llama.cpp must evaluate at least one token to have logits to sample
        from, so an identical prompt is knocked back by one. On a live slot a
        context checkpoint covers that; on a restored slot there is none, and
        the same one-token rewind costs the whole prompt."""
        held = self.be.held(0)
        reply = self.be.prefill(self.prompt, id_slot=0)
        self.assertLess(tokens_reused(reply), held,
                        "it reused every token, so it evaluated none")
        self.assertGreater(tokens_read(reply), 0)

    def test_a_generated_token_lands_in_the_slot(self):
        """All but the last one.

        A sampled token only enters the slot when it is fed back in to get the
        next one, so the final token of a reply is returned and never stored.
        `n_predict: 0` is the k = 1 case of this, which is why the read pass
        leaves the slot holding exactly the prompt."""
        held = self.be.held(0)
        reply = self.be.prefill(self.prompt + " tell me", id_slot=0, n_predict=6)
        self.assertGreater(reply["tokens_predicted"], 1,
                           "it stopped after one token, so there is nothing "
                           "to tell apart")
        self.assertEqual(self.be.held(0),
                         held + tokens_read(reply) + reply["tokens_predicted"] - 1,
                         "the slot did not grow by what it fed back")

    def test_and_makes_the_bare_prompt_a_prefix_again(self):
        """Which is why the read pass asks for no tokens at all."""
        self.be.prefill(self.prompt, id_slot=0, n_predict=6)
        grown = self.be.held(0)
        reply = self.be.prefill(self.prompt, id_slot=0)
        self.assertLess(tokens_reused(reply), grown,
                        "a slot holding prompt + reply extended the bare prompt")


class MovingAStateBetweenInstances(LiveCase):
    """Beliefs 1 and 2: a state saved on one instance, restored on another."""

    def setUp(self):
        super().setUp()
        self.source = self.start("source", slots=2)
        self.target = self.start("target", slots=2)
        self.prompt = filler(300)
        self.source.prefill(self.prompt, id_slot=0, n_predict=0)
        self.saved = self.source.save_slot(0, "move.bin")

    def test_a_saved_slot_restores_on_a_different_instance(self):
        answer = self.target.restore_slot(1, "move.bin")
        self.assertEqual(answer["n_restored"], self.saved["n_saved"])
        self.assertEqual(answer["n_read"], self.saved["n_written"])

    def test_the_restored_slot_serves_a_strict_extension_without_reading_it(self):
        self.target.restore_slot(1, "move.bin")
        reply = self.target.prefill(self.prompt + " and a little more", id_slot=1)
        self.assertEqual(tokens_reused(reply), self.saved["n_saved"],
                         "the restored state was not used")
        self.assertLess(tokens_read(reply), 10)

    def test_an_identical_prompt_needs_a_rewind_the_file_may_not_carry(self):
        """The belief as the router states it is too generous.

        A restored slot serves an *extension* for free either way. Serving the
        same prompt again is a different matter: with nothing new to process
        llama.cpp steps back one token, and whether that works depends on
        whether the state file carried the slot's context checkpoints.

        Stock llama.cpp writes only the KV and the tokens, so it does not, and
        the whole prompt is read again.
        `patches/slot-state-carries-checkpoints.patch` appends them, and then
        it costs a few tokens. Which build is in `llama.cpp-mtp/build` is a
        property of this machine, so the test asks rather than assumes."""
        self.target.restore_slot(1, "move.bin")
        before = self.target.mark()
        reply = self.target.prefill(self.prompt, id_slot=1)
        if REREAD_LINE in self.target.since(before):
            self.assertEqual(tokens_reused(reply), 0,
                             "it announced a full re-read and then reused "
                             "part of the slot anyway")
            self.assertGreaterEqual(tokens_read(reply), self.saved["n_saved"] - 1)
        else:
            self.assertGreater(tokens_reused(reply), 0,
                               "nothing was reused and nothing explained why")
            self.assertLess(tokens_read(reply), 50,
                            "the file carried checkpoints and it still read "
                            "most of the prompt")

    def test_a_rewind_on_a_restored_slot_needs_a_checkpoint(self):
        """Belief 2, in whichever of the two builds this is.

        The claim under test is not "a restored slot always re-reads". It is
        that a rewind needs a checkpoint, and that a restored slot only has
        one if the file carried it. Both builds agree on that; they disagree
        on whether the file carries it."""
        self.target.restore_slot(1, "move.bin")
        before = self.target.mark()
        shorter = self.prompt.rsplit(" ", 3)[0]
        reply = self.target.prefill(shorter, id_slot=1)
        if REREAD_LINE in self.target.since(before):
            self.assertEqual(tokens_reused(reply), 0,
                             "it said it was re-reading everything and did not")
        else:
            self.assertGreater(tokens_reused(reply), 0,
                               "it rewound without a checkpoint and without "
                               "re-reading, which cannot happen")

    def test_a_continuation_that_diverges_inside_the_last_message(self):
        """What a client costs itself by not echoing the model exactly.

        A parked cache holds the tokens the model generated. A next turn that
        sends back a different assistant message diverges inside it, which is
        a rewind past everything after the divergence -- the same question as
        above, with the answer the same way."""
        self.target.restore_slot(1, "move.bin")
        before = self.target.mark()
        head = self.prompt.rsplit(" ", 3)[0]
        reply = self.target.prefill(head + " quite something else entirely",
                                    id_slot=1)
        if REREAD_LINE in self.target.since(before):
            self.assertEqual(tokens_reused(reply), 0)
        else:
            self.assertGreater(tokens_reused(reply), 0)
            self.assertLess(tokens_read(reply), self.saved["n_saved"] // 2)

    def test_a_live_slot_rewinds_into_a_checkpoint_instead(self):
        """The contrast that makes the belief above mean something."""
        shorter = self.prompt.rsplit(" ", 3)[0]
        reply = self.source.prefill(shorter, id_slot=0)
        self.assertGreater(tokens_reused(reply), 0,
                           "a live slot could not rewind either, so the "
                           "restored one proves nothing about checkpoints")


class SavingASlot(LiveCase):
    """Belief 7: what `?action=save` reports, and when it waits."""

    def setUp(self):
        super().setUp()
        self.be = self.start("saving", slots=2)

    def test_a_save_reports_what_it_wrote(self):
        """`n_written` has to be the size of the file, and is not.

        The router stores `n_written` as a copy's size and spends
        `PARK_BUDGET` against the total, so an undercount is disk it thinks it
        still has. It also compares `n_written` against `PARK_FLOOR` to tell a
        real copy from an empty one, which an undercount only makes stricter.

        `patches/slot-state-carries-checkpoints.patch` appends the checkpoints
        after the state, and `res->n_bytes = nwrite` is set from
        `llama_state_seq_save_file` before that trailer is written. Measured
        here: a 49 MB figure for a 107 MB file, so the budget is out by more
        than a factor of two."""
        self.be.prefill(filler(300), id_slot=0, n_predict=0)
        answer = self.be.save_slot(0, "kept.bin")
        self.assertEqual(answer["n_saved"], self.be.held(0))
        self.assertGreater(answer["n_written"], 0)
        on_disk = (self.be.slot_dir / "kept.bin").stat().st_size
        self.assertEqual(on_disk, answer["n_written"],
                         f"the file is {on_disk} bytes and the save reported "
                         f"{answer['n_written']}: whatever is written after "
                         f"the state is not counted, and the router spends "
                         f"PARK_BUDGET against this number")

    def test_saving_a_slot_that_holds_nothing_still_succeeds(self):
        """Which is why the router has a floor on the size.

        A save of an empty slot is not an error. It writes a header and calls
        it done, so a router that only checked for an exception would believe
        it had parked a cache it had lost."""
        answer = self.be.save_slot(1, "empty.bin")
        self.assertEqual(answer["n_saved"], 0)
        self.assertLess(answer["n_written"], 64 * 1024,
                        "an empty save is big enough to look like a real one")

    def test_a_save_waits_for_a_working_slot_rather_than_failing(self):
        box = {}

        def work():
            box["reply"] = self.be.prefill(filler(200, "busy"), id_slot=0,
                                           n_predict=120)

        worker = threading.Thread(target=work, name="busy-slot")
        worker.start()
        try:
            self.assertTrue(self._wait(lambda: self._busy(0)),
                            "the slot never started working")
            answer = self.be.save_slot(0, "deferred.bin", timeout=300)
        finally:
            worker.join(300)
        self.assertFalse(worker.is_alive(), "the generation never finished")
        self.assertGreater(answer["n_saved"], 0, "the deferred save saved nothing")
        # It waited for the turn instead of failing or saving half of it.
        self.assertEqual(answer["n_saved"], self.be.held(0))

    def _busy(self, id_slot):
        row = self.be.slot(id_slot)
        return bool(row and row.get("is_processing"))

    @staticmethod
    def _wait(check, patience=60.0, step=0.05):
        stop = time.time() + patience
        while time.time() < stop:
            if check():
                return True
            time.sleep(step)
        return bool(check())


class TwoInstancesMustAgree(LiveCase):
    """Belief 8: what actually governs whether a state can move.

    The README says "both backends must agree on the KV layout, which is why
    the cpu backend runs with --kv-unified". That is one of four things, and
    the failure is the same opaque 400 for all of them: the reason is only in
    the target backend's log.
    """

    def setUp(self):
        super().setUp()
        self.source = self.start("origin", slots=2, unified=True, flash=True)
        self.prompt = filler(300)
        self.source.prefill(self.prompt, id_slot=0, n_predict=0)
        self.saved = self.source.save_slot(0, "shape.bin")

    def refuse(self, target, id_slot=0):
        """Try the restore. Returns the backend's own reason, or None."""
        before = target.mark()
        try:
            target.restore_slot(id_slot, "shape.bin")
            return None
        except HttpError as err:
            self.assertEqual(err.status, 400)
            log = target.since(before)
            reason = [line.split("] ")[-1].strip() for line in log.splitlines()
                      if " E " in line]
            self.assertTrue(reason, "it refused and said nothing in its log")
            return "\n".join(reason)

    def test_an_instance_of_the_same_shape_takes_it(self):
        target = self.start("same", slots=2, unified=True, flash=True)
        self.assertIsNone(self.refuse(target, 1))
        reply = target.prefill(self.prompt + " and a little more", id_slot=1)
        self.assertEqual(tokens_reused(reply), self.saved["n_saved"])

    def test_kv_unified_must_match(self):
        target = self.start("split", slots=2, unified=False, flash=True)
        self.assertIn("n_stream mismatch", self.refuse(target, 1) or "")

    def test_flash_attention_must_match(self):
        """Nothing in the README mentions this one, and `--flash-attn auto`
        resolves differently on a GPU instance and a CPU one."""
        target = self.start("noflash", slots=2, unified=True, flash=False)
        self.assertIn("incompatible V transposition", self.refuse(target, 1) or "")

    def test_the_model_must_match(self):
        target = self.start("other", slots=2, unified=True, flash=True, kind="plain")
        self.assertIn("mismatched layer count", self.refuse(target, 1) or "")

    def test_a_smaller_context_is_fine_while_the_state_fits(self):
        small = max(1024, self.saved["n_saved"] + 256)
        target = self.start("small", slots=1, ctx=small, unified=True, flash=True)
        self.assertIsNone(self.refuse(target, 0))

    def test_a_state_too_big_for_the_target_is_refused(self):
        target = self.start("tiny", slots=1, ctx=512, unified=True, flash=True)
        self.assertIn("failed to find", self.refuse(target, 0) or "")

    def test_a_slot_count_the_source_does_not_have_is_fine(self):
        target = self.start("wide", slots=4, unified=True, flash=True)
        self.assertIsNone(self.refuse(target, 3))


class IdleSlotsAreEmptiedOnAUnifiedBackend(LiveCase):
    """Not one of the eight. It is the one that was costing the most.

    llama.cpp runs with --cache-idle-slots by default. When any task starts,
    every idle slot is copied into the server's own RAM prompt cache and --
    when the instance runs --kv-unified -- the slot is then CLEARED.

    Production does not: bin/qwen-mtp-cpu.sh passes --kv-unified only when
    SLOTS is over 1, and docs/LAYOUT.md runs one slot each. So this half is
    what the flag would cost, not what it costs today; README.md in this
    directory says the same. The other half bites whatever the layout, and
    that is the one --no-cache-idle-slots is passed for.

    The router believes a slot keeps what it read until something displaces
    it. On a kv-unified instance nothing has to displace it: another
    conversation starting a turn is enough.

    Worse, the way back is closed. The RAM cache is only consulted when the
    server chooses the slot itself. A request that names a slot -- which is
    what the router now does, to know which slot to save -- is given that slot
    as it stands, empty, and reads the whole prompt again.
    """

    def held_after_a_task_elsewhere(self, **kw):
        be = self.start("idle-" + kw.get("tag", "x"),
                        slots=2, **{k: v for k, v in kw.items() if k != "tag"})
        mine = filler(300, "mine")
        be.prefill(mine, id_slot=0, n_predict=0)
        before = be.held(0)
        be.prefill(filler(200, "other"), id_slot=1, n_predict=0)
        return be, mine, before, be.held(0)

    def test_a_task_on_one_slot_empties_every_other_idle_slot(self):
        be, mine, before, after = self.held_after_a_task_elsewhere(
            tag="unified", unified=True)
        self.assertGreater(before, 0)
        self.assertEqual(after, 0,
                         "the idle slot kept its prompt, so this build no "
                         "longer clears idle slots")

    def test_and_a_request_that_names_that_slot_reads_everything_again(self):
        be, mine, before, after = self.held_after_a_task_elsewhere(
            tag="named", unified=True)
        reply = be.prefill(mine + " and a little more", id_slot=0)
        self.assertEqual(tokens_reused(reply), 0,
                         "naming the emptied slot still found the cache")
        self.assertGreaterEqual(tokens_read(reply), before)

    def test_while_the_same_request_without_a_slot_gets_it_back_for_free(self):
        """The same prompt, the same backend, the same moment. The only
        difference is that the router did not say which slot."""
        be, mine, before, after = self.held_after_a_task_elsewhere(
            tag="unnamed", unified=True)
        reply = be.post("/completion", {"prompt": mine + " and a little more",
                                        "n_predict": 0, "cache_prompt": True})
        self.assertEqual(tokens_reused(reply), before,
                         "the RAM prompt cache did not give it back")

    def test_no_cache_idle_slots_keeps_the_slot(self):
        """The flag production now passes."""
        be, mine, before, after = self.held_after_a_task_elsewhere(
            tag="off", unified=True, extra=["--no-cache-idle-slots"])
        self.assertEqual(after, before, "the idle slot was emptied anyway")
        reply = be.prefill(mine + " and a little more", id_slot=0)
        self.assertEqual(tokens_reused(reply), before)

    def test_an_instance_without_kv_unified_keeps_it_too(self):
        be, mine, before, after = self.held_after_a_task_elsewhere(
            tag="split", unified=False)
        self.assertEqual(after, before)

    def test_a_state_the_router_has_just_restored_is_cleared_the_same_way(self):
        """The worst case: the opening the builder spent minutes reading is
        loaded into a slot, another conversation starts a turn, and the
        opening is gone before the request that asked for it arrives."""
        be = self.start("wiped", slots=2, unified=True)
        opening = filler(300, "opening")
        be.prefill(opening, id_slot=0, n_predict=0)
        saved = be.save_slot(0, "opening.bin")
        be.restore_slot(1, "opening.bin")
        be.prefill(filler(200, "someone-else"), id_slot=0, n_predict=0)
        reply = be.prefill(opening + " and a little more", id_slot=1)
        self.assertEqual(tokens_reused(reply), 0,
                         "the restored opening survived, so this build differs")
        self.assertGreaterEqual(tokens_read(reply), saved["n_saved"])


class WhatSlotsReports(LiveCase):
    """The router reads /slots every POLL seconds and believes it."""

    def setUp(self):
        super().setUp()
        self.be = self.start("reporting", slots=2)

    def test_a_slot_that_never_ran_a_task_reports_no_size_at_all(self):
        """Not zero. The key is absent.

        A slot filled by `?action=restore` and not yet used is in exactly this
        state, so the dashboard shows an empty slot holding a restored cache.
        `.get("n_prompt_tokens", 0)` reads it as zero, which is wrong in the
        harmless direction; `slot["n_prompt_tokens"]` would raise."""
        self.be.prefill(filler(200), id_slot=0, n_predict=0)
        self.be.save_slot(0, "row.bin")
        self.be.restore_slot(1, "row.bin")
        row = self.be.slot(1)
        self.assertNotIn("n_prompt_tokens", row)
        self.assertIn("is_processing", row)
        self.assertFalse(row["is_processing"])

    def test_an_idle_slot_reports_a_size_but_zero_for_everything_else(self):
        """The counters belong to the task, and the task is over.

        `n_prompt_tokens` is what the slot holds now. `n_prompt_tokens_cache`
        and `n_prompt_tokens_processed` and `n_decoded` are all back to zero
        the moment the turn ends, so the router's
        `prompt = n_prompt_tokens - decoded - cached` reads a whole prompt as
        still to be read on a slot that is doing nothing. The phase is what
        saves the dashboard from showing it."""
        reply = self.be.prefill(filler(200), id_slot=0, n_predict=0)
        row = self.be.slot(0)
        self.assertEqual(row["n_prompt_tokens"], tokens_read(reply))
        self.assertEqual(row["n_prompt_tokens_cache"], 0)
        self.assertEqual(row["n_prompt_tokens_processed"], 0)
        self.assertEqual(row["next_token"][0]["n_decoded"], 0)
        self.assertFalse(row["is_processing"])


class ADivergenceAtAMessageBoundary(LiveCase):
    """The shape every agent client sends, and the one nothing here tested.

    Every other rewind test above uses one blob of text and chops words off
    its end. A blob has no message boundaries, so llama.cpp makes only the two
    checkpoints it always makes near the end of a prompt, and a divergence
    near the end always finds one below it. Those tests pass and say nothing
    about this.

    A chat is different. llama.cpp makes a checkpoint at the start of a user
    message, and an agent client rewrites a message in the middle of the
    conversation between turns. So the divergence lands on a boundary that has
    a checkpoint of its own, and whether the shared opening survives depends
    entirely on which side of that boundary the checkpoint sits.

    Production reads 52,804 tokens to avoid reusing 17,251 it already holds.
    This is that, in miniature."""

    # Small enough to be quick, spaced enough that a boundary in the middle
    # gets a checkpoint of its own. Production runs 2048 against prompts
    # twenty times this size.
    MIN_STEP = 128

    def setUp(self):
        super().setUp()
        self.be = self.start("chat", ctx=8192, slots=1,
                             extra=("--checkpoint-min-step", str(self.MIN_STEP),
                                    "--ctx-checkpoints", "64", "--jinja"))
        self.be.wait_ready()

    def turn(self, middle):
        """One chat whose middle user message is `middle`. Reads, never runs."""
        return {"model": "live", "max_tokens": 0, "stream": False,
                "messages": [
                    {"role": "system", "content": prose(6000, "rules")},
                    {"role": "user", "content": prose(1500, "first")},
                    {"role": "assistant", "content": prose(400, "reply one")},
                    {"role": "user", "content": middle},
                    {"role": "assistant", "content": prose(400, "reply two")},
                    {"role": "user", "content": prose(600, "last")},
                ]}

    def test_the_opening_before_the_changed_message_is_kept(self):
        first = self.be.post("/v1/chat/completions", self.turn(prose(1200, "middle")))
        held = read_tokens(self.be.log_text())
        self.assertTrue(held, "the first turn read nothing at all")

        before = self.be.mark()
        # The same conversation with one message rewritten, exactly as a client
        # does when it re-renders a tool result or a reminder.
        second = self.be.post("/v1/chat/completions",
                              self.turn(prose(1200, "rewritten")))
        trace = self.be.since(before)

        kept = tokens_reused(second)
        self.assertGreater(
            kept, 0,
            "everything before the rewritten message was read again. "
            f"checkpoints offered: {self.offered(trace)}, "
            f"asked for: {self.asked(trace)}")

    def test_a_checkpoint_below_the_boundary_exists_to_rewind_into(self):
        self.be.post("/v1/chat/completions", self.turn(prose(1200, "middle")))
        before = self.be.mark()
        self.be.post("/v1/chat/completions", self.turn(prose(1200, "rewritten")))
        trace = self.be.since(before)

        want = self.asked(trace)
        if want is None:
            self.skipTest("no rewind was attempted, so there is nothing to judge")
        offered = self.offered(trace)
        self.assertTrue(
            any(p < want for p in offered),
            f"the prompt stopped matching at {want} and the nearest checkpoint "
            f"below it does not exist; offered {sorted(offered)}. A checkpoint "
            f"at the boundary is one the divergence cannot use.")

    @staticmethod
    def offered(trace):
        import re
        return {int(m) for m in
                re.findall(r"checking checkpoint with \[(\d+),", trace)}

    @staticmethod
    def asked(trace):
        import re
        found = re.findall(r"checking checkpoint with \[\d+, \d+\] against (\d+)",
                           trace)
        return int(found[0]) if found else None


class ADivergenceAtTheFirstUserMessage(LiveCase):
    """The agent client's own shape: a long system prompt, then a turn.

    An agent sends a system prompt of tens of thousands of tokens and rewrites
    the first user message every turn, because that is where it puts the
    reminders that change. So the prompt stops matching exactly where the
    system prompt ends.

    Nothing can be checkpointed inside a system prompt: llama.cpp makes a
    checkpoint at the start of a user message, and there is no user message in
    there. The only checkpoint anywhere near is the one at the boundary
    itself. Whether the system prompt survives comes down to which side of the
    boundary that checkpoint sits on.

    Production: 17,251 tokens of system prompt shared, lowest checkpoint at
    17,252, all 52,804 tokens read again."""

    MIN_STEP = 128

    def setUp(self):
        super().setUp()
        self.be = self.start("first", ctx=8192, slots=1,
                             extra=("--checkpoint-min-step", str(self.MIN_STEP),
                                    "--ctx-checkpoints", "64", "--jinja"))
        self.be.wait_ready()

    def turn(self, opening):
        """A long system prompt with no boundary in it, then a rewritten turn."""
        return {"model": "live", "max_tokens": 0, "stream": False,
                "messages": [
                    {"role": "system", "content": prose(8000, "rules")},
                    {"role": "user", "content": opening},
                    {"role": "assistant", "content": prose(300, "reply")},
                    {"role": "user", "content": prose(500, "last")},
                ]}

    def test_the_system_prompt_survives_the_turn_being_rewritten(self):
        self.be.post("/v1/chat/completions", self.turn(prose(900, "monday")))
        before = self.be.mark()
        second = self.be.post("/v1/chat/completions", self.turn(prose(900, "tuesday")))
        trace = self.be.since(before)

        offered = ADivergenceAtAMessageBoundary.offered(trace)
        want = ADivergenceAtAMessageBoundary.asked(trace)
        self.assertGreater(
            tokens_reused(second), 0,
            f"the whole system prompt was read again. It stopped matching at "
            f"{want}; the checkpoints on offer were {sorted(offered)}, none at "
            f"or below it.")


class AskingForOneLetter(LiveCase):
    """What /v1/systemone rests on: one token, a grammar, and the numbers.

    The grammar decides what is written. It does **not** decide what is
    reported: before sampling, the probabilities are a plain softmax over the
    whole vocabulary, so they cover words no letter stands for and they may not
    mention the letters at all. The router keeps the letters and scales them
    back up, which is only meaningful when the prompt asked for a letter.
    """

    LETTERS = "ABC"

    def setUp(self):
        super().setUp()
        self.be = self.start("letter", slots=1)

    def body(self, messages, post_sampling=False, top=14):
        allowed = " | ".join(f'"{letter}"' for letter in self.LETTERS)
        return {"model": "live", "id_slot": 0, "stream": False,
                "messages": messages,
                "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": top,
                "post_sampling_probs": post_sampling,
                "grammar": "root ::= " + allowed}

    @staticmethod
    def bare():
        """A prompt with no hint that an answer is wanted."""
        return [{"role": "user", "content": prose(300, "notes")}]

    @classmethod
    def lettered(cls):
        """The shape the router sends: a rubric, a state, a lettered question."""
        return [
            {"role": "system", "content":
                "Answer the question about the text above with one letter.\n"
                "The question gives a letter for every answer it takes.\n"
                "Write that letter and nothing else."},
            {"role": "user", "content": "cpu1_0 read 150000 tokens in 94 minutes"},
            {"role": "user", "content":
                "What happened to this turn?\n\n"
                "A = it reused a cache\n"
                "B = it read from the start\n"
                "C = it never reached a backend\n\n"
                "Answer with one letter.\nAnswer:"},
        ]

    @staticmethod
    def offered(reply):
        """Every token the reply offered for the one position, as {token: p}.

        llama.cpp reports a logprob before sampling and a plain prob after it,
        so a test that read only one of the two would pass on either setting."""
        content = (reply["choices"][0].get("logprobs") or {}).get("content") or []
        rows = (content[0].get("top_logprobs") or []) if content else []
        found = {}
        for row in rows:
            p = row["prob"] if "prob" in row else math.exp(row["logprob"])
            token = row["token"].strip()
            found[token] = found.get(token, 0.0) + p
        return found

    def carried(self, reply):
        """How much of the reported mass fell on the answer letters."""
        found = self.offered(reply)
        return sum(p for token, p in found.items() if token in self.LETTERS), found

    def test_the_grammar_holds_what_is_written_to_one_letter(self):
        reply = self.be.post("/v1/chat/completions", self.body(self.bare()))
        self.assertIn(reply["choices"][0]["message"]["content"].strip(),
                      list(self.LETTERS))
        self.assertEqual(reply["usage"]["completion_tokens"], 1)

    def test_but_the_numbers_reported_ignore_it_entirely(self):
        """The belief that decides how the router reads an answer.

        The same request that is forced to write `A` reports the words it
        would rather have written, and the letters barely appear. So a
        confidence taken from these numbers means nothing unless the prompt
        itself asked for a letter."""
        on_letters, found = self.carried(
            self.be.post("/v1/chat/completions", self.body(self.bare())))
        self.assertGreater(len(found), 1, "only the written token was reported")
        self.assertLess(on_letters, 0.05,
                        f"the letters carried the prompt after all: {found}")

    def test_a_prompt_that_asks_for_a_letter_gets_letters(self):
        """And with the router's own prompt shape, they do carry it.

        Measured at 0.46 on the 0.5B model these tests run, against 0.0007 for
        the bare prompt above. The number is the model's, not the router's, so
        this asserts the difference rather than a level."""
        on_letters, found = self.carried(
            self.be.post("/v1/chat/completions", self.body(self.lettered())))
        self.assertGreater(on_letters, 0.1,
                           f"a prompt that asked for a letter got {found}")
        spread = [p for token, p in found.items() if token in self.LETTERS]
        self.assertGreaterEqual(len(spread), 2,
                                f"one letter took the whole answer: {found}")

    def test_more_room_in_the_report_does_not_find_more_letters(self):
        """Why top_logprobs stays small. The letters are near the top or
        nowhere: asking for 120 rather than 14 moved 0.0002."""
        small, _ = self.carried(
            self.be.post("/v1/chat/completions", self.body(self.lettered(), top=14)))
        large, _ = self.carried(
            self.be.post("/v1/chat/completions", self.body(self.lettered(), top=120)))
        self.assertAlmostEqual(small, large, places=2)

    def test_post_sampling_probs_reports_nothing_to_read(self):
        """The trap this endpoint had to avoid.

        After the sampling chain the grammar and greedy sampling have already
        chosen. On this route what comes back carries no probabilities at all,
        and on /completion it is the one token at 1.0. Either way a confidence
        read from here would be the same number every time."""
        reply = self.be.post("/v1/chat/completions",
                             self.body(self.lettered(), post_sampling=True))
        found = self.offered(reply)
        self.assertLessEqual(len(found), 1, f"a distribution survived: {found}")
        for p in found.values():
            self.assertAlmostEqual(p, 1.0, places=3)

    def test_one_token_leaves_the_slot_holding_the_prompt(self):
        """Belief 6 at k = 1, which is why a question costs the next turn
        nothing. A slot holding prompt + 1 makes the bare prompt a prefix
        again, and rewinding into a prefix needs a checkpoint."""
        reply = self.be.post("/v1/chat/completions", self.body(self.lettered()))
        self.assertEqual(self.be.held(0), reply["usage"]["prompt_tokens"],
                         "the slot holds something other than the prompt")



class ReadingWhatTheGrammarAllows(LiveCase):
    """patches/grammar-probs.patch: the readout AskingForOneLetter wanted.

    That class records what the stock readouts do. Before the sampler the
    probabilities cover the whole vocabulary and the grammar is not in them,
    so the answers may not appear at all; after it, the grammar and the
    sampler have already chosen. `grammar_probs` reports the tokens the
    grammar allows, and `grammar_mass` what they held before being scaled up.

    Skipped on a build without the patch: it is not one of the three the
    router requires, and llama.cpp ignores a field it does not know.
    """

    LETTERS = "ABC"

    def setUp(self):
        super().setUp()
        self.be = self.start("allowed", slots=1)

    def body(self, messages, **extra):
        allowed = " | ".join(f'"{letter}"' for letter in self.LETTERS)
        return {"model": "live", "id_slot": 0, "stream": False,
                "messages": messages, "max_tokens": 1, "temperature": 0,
                "grammar": "root ::= " + allowed, **extra}

    @staticmethod
    def lettered():
        return [{"role": "user", "content":
                 "Is this about caching?\n\nA = yes\nB = no\nC = unclear\n\n"
                 "Answer with one letter.\nAnswer:"}]

    def read(self, reply):
        """The first token's rows and its grammar_mass, or a skip."""
        content = (reply["choices"][0].get("logprobs") or {}).get("content") or []
        head = content[0] if content else {}
        mass = head.get("grammar_mass")
        if mass is None:
            raise unittest.SkipTest("this build has no grammar-probs patch")
        rows = {row["token"].strip(): math.exp(row["logprob"])
                for row in head.get("top_logprobs") or []}
        return rows, mass

    def test_it_reports_the_allowed_tokens_and_nothing_else(self):
        """The whole point. The stock readout returns whatever the model was
        going to write, which for a prompt like this is a word, not a letter."""
        rows, _ = self.read(self.be.post("/v1/chat/completions",
                                         self.body(self.lettered(),
                                                   grammar_probs=True)))
        self.assertEqual(sorted(rows), sorted(self.LETTERS))

    def test_the_answers_add_up_to_one(self):
        rows, _ = self.read(self.be.post("/v1/chat/completions",
                                         self.body(self.lettered(),
                                                   grammar_probs=True)))
        self.assertAlmostEqual(sum(rows.values()), 1.0, places=4)

    def test_it_needs_no_n_probs_at_all(self):
        """Three gates on n_probs stood between the readout and the reply, and
        a request that asks for none has to pass all three. The body here sets
        neither logprobs nor top_logprobs."""
        reply = self.be.post("/v1/chat/completions",
                             self.body(self.lettered(), grammar_probs=True))
        rows, _ = self.read(reply)
        self.assertEqual(len(rows), len(self.LETTERS))

    def test_the_mass_is_a_share_of_the_whole_distribution(self):
        """It is what the answers held before the scaling, so it is at most
        one, and it is not the scaled numbers over again."""
        _, mass = self.read(self.be.post("/v1/chat/completions",
                                         self.body(self.lettered(),
                                                   grammar_probs=True)))
        self.assertGreater(mass, 0.0)
        self.assertLessEqual(mass, 1.0)

    def test_prose_holds_less_of_it_than_a_question_does(self):
        """The reason the field exists. Both answers are a letter, because the
        grammar says so. Only the mass tells them apart."""
        _, asked = self.read(self.be.post("/v1/chat/completions",
                                          self.body(self.lettered(),
                                                    grammar_probs=True)))
        prose_body = self.body(
            [{"role": "user", "content": prose(300, "notes")}],
            grammar_probs=True)
        _, wandering = self.read(self.be.post("/v1/chat/completions", prose_body))
        self.assertGreater(asked, wandering * 10,
                           f"a question held {asked}, prose held {wandering}")

    def test_the_readout_happens_before_the_grammar_moves_on(self):
        """common_sampler_accept advances the grammar past the token just
        chosen. Read after it and the answer is what may follow the answer,
        which for a one-token grammar is the end of the string."""
        rows, _ = self.read(self.be.post("/v1/chat/completions",
                                         self.body(self.lettered(),
                                                   grammar_probs=True)))
        self.assertNotIn("", rows, "the end-of-string token was reported")
        for token in rows:
            self.assertIn(token, self.LETTERS)


if __name__ == "__main__":
    unittest.main()
