"""Tests for /v1/systemone, the typed question endpoint.

A typed question is an ordinary turn: the same pin, the same slot, the same
queue, the same park. Two things are its own. It generates where it was read,
because carrying a slot in order to write one token costs a park and a recall.
And it answers with a letter, which the router maps back to the answer's name.

These run against the stub backend and reuse test_end_to_end.py's fixture. The
stub lets the **last** letter win, so a router that never mapped a letter back
to its answer cannot pass.
"""
import json
import unittest
import urllib.error
import urllib.request

from test_end_to_end import EndToEnd, wait_for
from test_turn import TurnLink

import router

CHOICE = {"kind": {"type": "choice",
                   "instructions": "What happened to this turn?",
                   "criteria": {"warm": "it reused a cache",
                                "cold": "it read from the start",
                                "lost": "it never reached a backend"}}}


def typed(url, body):
    """The request one typed call makes."""
    return urllib.request.Request(
        url + router.SYSTEMONE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")


def typed_call(url, body, timeout=30):
    """One typed call through the router. Returns the parsed reply."""
    with urllib.request.urlopen(typed(url, body), timeout=timeout) as reply:
        return json.load(reply)


class TypedCall(EndToEnd):
    """One backend that reads and generates, which is the shipped default."""

    def setUp(self):
        super().setUp()
        self.one = self.pool([self.stub("solo", 0)])
        self.url = self.serve(self.one)

    def ask(self, questions, state="the turn read 150000 tokens", conv=None):
        """One typed call through the router. Returns the parsed reply."""
        body = {"model": "fake-model", "state": state, "questions": questions}
        if conv:
            body["prompt_cache_key"] = conv
        return typed_call(self.url, body)

    def refused(self, body):
        """The status and the message the router answers a bad body with."""
        try:
            with urllib.request.urlopen(typed(self.url, body), timeout=30):
                self.fail("the router accepted a body it should refuse")
        except urllib.error.HTTPError as err:
            said = json.loads(err.read()).get("error") or {}
            return err.code, said.get("message", "")


class TheThreeTypes(TypedCall):

    def test_a_choice_comes_back_named_rather_than_lettered(self):
        answer = self.ask(CHOICE)["answers"]["kind"]
        self.assertEqual(answer["type"], "choice")
        # The stub lets the last letter win. C is the third criterion.
        self.assertEqual(answer["choice"], "lost")
        self.assertEqual(sorted(answer["probabilities"]),
                         ["cold", "lost", "warm"])
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
        self.assertEqual(answer["confidence"], answer["probabilities"]["lost"])

    def test_a_token_the_grammar_forbade_is_dropped(self):
        """The stub offers a digit as well. Only the answers count."""
        answer = self.ask(CHOICE)["answers"]["kind"]
        self.assertNotIn("7", answer["probabilities"])
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)

    def test_mass_says_how_much_was_thrown_away(self):
        """The probabilities are the raw softmax, taken before the grammar.

        They cover tokens no answer stands for, and the router drops those.
        `mass` is what the answers held before it did, so a caller can tell an
        answer from a model that was writing something else entirely."""
        answer = self.ask(CHOICE)["answers"]["kind"]
        self.assertGreater(answer["mass"], 0)
        self.assertLess(answer["mass"], 1.0,
                        "the stub offers a token outside the grammar, so the "
                        "answers cannot hold all of the mass")

    def test_a_score_is_the_expected_level_not_the_likeliest(self):
        reply = self.ask({"urgency": {
            "type": "score", "instructions": "How urgent is it?",
            "criteria": ["routine", "watch", "act now"]}})
        answer = reply["answers"]["urgency"]
        self.assertEqual(answer["type"], "score")
        probs = answer["probabilities"]
        self.assertAlmostEqual(
            answer["score"],
            0 * probs["routine"] + 1 * probs["watch"] + 2 * probs["act now"])
        # Weight sits on the other levels too, so it is not a whole number.
        self.assertLess(answer["score"], 2.0)
        self.assertGreater(answer["score"], 1.0)

    def test_a_noul_says_what_its_two_answers_stand_for(self):
        """Jev's noul carries criteria for true and false, and a client in the
        wild sends them. The answer is still yes or no; the criteria only tell
        the model what the two mean, so they belong in the prompt."""
        plan = router.systemone_plan(json.dumps({"state": "x", "questions": {
            "ok": {"type": "noul", "instructions": "Is it healthy?",
                   "criteria": {"true": "every backend answered",
                                "false": "one of them is down"}}}}).encode())
        said = router.systemone_says(plan["questions"][0])
        self.assertIn("A = every backend answered", said)
        self.assertIn("B = one of them is down", said)

    def test_a_noul_without_criteria_letters_yes_and_no(self):
        plan = router.systemone_plan(json.dumps({"state": "x", "questions": {
            "ok": {"instructions": "Is it healthy?"}}}).encode())
        said = router.systemone_says(plan["questions"][0])
        self.assertIn("A = yes", said)
        self.assertIn("B = no", said)

    def test_a_noul_is_the_chance_of_yes(self):
        answer = self.ask({"ok": {"instructions": "Is it healthy?"}})
        answer = answer["answers"]["ok"]
        self.assertEqual(answer["type"], "noul")
        # A is yes and B is no, and the stub lets the last letter win.
        self.assertEqual(answer["noul"], answer["probabilities"]["yes"])
        self.assertLess(answer["noul"], 0.5)

    def test_the_reply_says_what_it_cost(self):
        reply = self.ask(CHOICE)
        self.assertEqual(reply["model"], "fake-model")
        self.assertEqual(reply["router"]["backend"], "solo")
        self.assertGreaterEqual(reply["router"]["took"], 0)
        self.assertEqual(reply["usage"]["output_tokens"], 1)
        self.assertEqual(reply["usage"]["input_tokens"],
                         reply["router"]["read"] + reply["router"]["reused"])


class AnOrdinaryTurn(TypedCall):
    """Everything the router does for a chat turn, it does for this one."""

    def test_it_is_pinned_under_the_name_the_client_chose(self):
        self.ask(CHOICE, conv="asker")
        self.assertIn("asker", self.one.pins)
        self.assertEqual(self.one.pins["asker"]["backend"], "solo")

    def test_the_dashboard_lists_it_under_its_own_path(self):
        self.ask(CHOICE, conv="asker")
        self.assertTrue(wait_for(lambda: self.one.recent_requests),
                        "the turn never reached recent_requests")
        self.assertEqual(self.one.recent_requests[0]["path"], router.SYSTEMONE)

    def test_every_stage_carries_the_label_the_dashboard_draws(self):
        """Including `queued`, which is noted before the turn holds anything.

        Without it a queue of typed questions shows as a queue of unlabelled
        rows, and a queue is exactly when a reader wants to know what is in
        it."""
        self.ask(CHOICE, conv="asker")
        log = self.one.flow.report()["log"]
        mine = [row for row in log if row["conv"] == "asker"]
        self.assertTrue(mine, "the turn left no trace in the flow log")
        stages = {row["stage"]: row["kind"] for row in mine}
        self.assertIn("queued", stages)
        for stage, kind in stages.items():
            self.assertEqual(kind, "typed", f"{stage} lost the label")

    def test_an_ordinary_turn_carries_no_label(self):
        self.turn(self.url, "talker")
        log = self.one.flow.report()["log"]
        for row in log:
            if row["conv"] == "talker":
                self.assertIsNone(row["kind"], f"{row['stage']} was labelled")

    def test_it_uses_one_slot(self):
        self.ask(CHOICE)
        slots = {chat["slot"] for chat in self.solo.chats}
        self.assertEqual(len(slots), 1, f"the turn used {len(slots)} slots")

    def test_a_question_names_the_slot_the_state_was_read_into(self):
        """It answers where it read, so it knows which slot that is."""
        self.ask(CHOICE)
        self.assertTrue(all(chat["named"] for chat in self.solo.answers),
                        "a question let the backend pick its own slot")


class SeveralQuestionsOneState(TypedCall):
    """The reason this endpoint belongs in this router and not elsewhere."""

    def setUp(self):
        super().setUp()
        self.reply = self.ask({
            "kind": CHOICE["kind"],
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["low", "high"]},
            "ok": {"instructions": "Is it healthy?"}}, conv="asker")

    def test_every_question_is_answered(self):
        self.assertEqual(sorted(self.reply["answers"]),
                         ["kind", "ok", "urgency"])

    def test_the_state_is_read_once(self):
        self.assertEqual(len(self.solo.probes), 1,
                         "the state was read more than once")
        self.assertEqual(len(self.solo.answers), 3,
                         "a question did not reach the backend")

    def test_each_question_extends_what_is_already_in_the_slot(self):
        for answer in self.solo.answers:
            self.assertGreater(answer["cached"], 0, "the state was read again")

    def test_the_read_pass_carries_the_first_question(self):
        """Both phases must send the same prompt, or the first question pays
        for the whole state twice.

        Reading the state alone and letting each question extend it is the
        obvious split, and on production it cost a full re-read: 351 tokens of
        a state of 348. Neither small model in tests/live reproduces that, so
        the cause is still open. This asserts the shape, which is the part the
        router controls."""
        probe = self.solo.probes[0]
        first = self.solo.answers[0]
        self.assertEqual(probe["read"] + probe["cached"],
                         first["read"] + first["cached"],
                         "the read pass and the first question sent different "
                         "prompts, so the first question re-read the state")

    def test_the_reply_says_what_each_question_cost(self):
        steps = self.reply["router"]["questions"]
        self.assertEqual([step["question"] for step in steps],
                         ["kind", "urgency", "ok"])
        for step in steps:
            self.assertGreater(step["reused"], 0,
                               f"{step['question']} read the state again")


class ItStaysWhereItRead(EndToEnd):
    """A gpu that only generates, and a cpu that only reads.

    Every ordinary turn migrates to the gpu. A typed question does not: one
    token is not worth a park and a recall of the whole slot."""

    def setUp(self):
        super().setUp()
        self.duo = self.pool([self.stub("gpu", 0, prefill=False),
                              self.stub("cpu", 1)])
        self.url = self.serve(self.duo)

    def ask(self, conv="asker"):
        return typed_call(self.url, {
            "model": "fake-model", "prompt_cache_key": conv,
            "state": "cpu1_0 read 150000 tokens",
            "questions": {"ok": {"instructions": "Healthy?"}}})

    def test_it_reads_and_answers_on_the_same_backend(self):
        reply = self.ask()
        self.assertEqual(reply["router"]["backend"], "cpu")
        self.assertEqual([c["key"] for c in self.cpu.probes], ["asker"])
        self.assertEqual(len(self.cpu.answers), 1,
                         "the question did not run where the state was read")
        self.assertEqual(self.gpu.chats, [], "the turn was handed to the gpu")

    def test_nothing_is_carried(self):
        self.ask()
        self.assertEqual(self.gpu.restores, [], "the slot was carried over")

    def test_an_ordinary_turn_still_migrates(self):
        """The same pool and the same router, on the chat endpoint."""
        self.turn(self.url, "talker")
        self.assertEqual([c["key"] for c in self.gpu.answers], ["talker"])


class UnlessTheReaderMayNotAnswer(EndToEnd):
    """A reader set `generate: false` is not overruled by this endpoint.

    Saving a park is worth having. It is not worth answering on an instance
    the operator said must only read."""

    def setUp(self):
        super().setUp()
        self.duo = self.pool([self.stub("gpu", 0, prefill=False),
                              self.stub("cpu", 1, generate=False)])
        self.url = self.serve(self.duo)

    def test_it_is_carried_to_the_generator_after_all(self):
        said = typed_call(self.url, {
            "model": "fake-model", "prompt_cache_key": "asker",
            "state": "cpu1_0 read 150000 tokens",
            "questions": {"ok": {"instructions": "Healthy?"}}})
        self.assertEqual(said["router"]["backend"], "gpu")
        self.assertEqual(len(self.cpu.probes), 1, "the cpu did not read it")
        self.assertEqual(self.cpu.answers, [], "the cpu answered anyway")
        self.assertEqual(len(self.gpu.answers), 1, "the gpu did not answer")

    def test_the_question_names_no_slot_on_a_backend_it_did_not_read_on(self):
        """The slot the state landed in is the target's to choose. Naming the
        reader's slot number there would name a slot holding something else."""
        self.test_it_is_carried_to_the_generator_after_all()
        self.assertFalse(any(chat["named"] for chat in self.gpu.answers),
                         "the question named a slot on the wrong backend")


class Refusals(TypedCall):
    """What the router will not guess at. Every one of these is a 400."""

    def test_a_body_without_questions(self):
        code, said = self.refused({"state": "x"})
        self.assertEqual(code, 400)
        self.assertIn("questions", said)

    def test_a_choice_without_criteria(self):
        code, said = self.refused({"questions": {"a": {"type": "choice"}}})
        self.assertEqual(code, 400)
        self.assertIn("criteria", said)

    def test_a_choice_with_only_one_answer(self):
        code, said = self.refused({"questions": {"a": {
            "type": "choice", "criteria": {"yes": "the only one"}}}})
        self.assertEqual(code, 400)
        self.assertIn("at least two", said)

    def test_a_type_that_does_not_exist(self):
        code, said = self.refused({"questions": {"a": {"type": "vibes"}}})
        self.assertEqual(code, 400)
        self.assertIn("choice, score and noul", said)

    def test_more_answers_than_one_token_carries(self):
        many = {str(n): f"answer {n}" for n in range(30)}
        code, said = self.refused(
            {"questions": {"a": {"type": "choice", "criteria": many}}})
        self.assertEqual(code, 400)
        self.assertIn("26", said)

    def test_a_request_to_stream(self):
        code, said = self.refused(
            {"stream": True, "questions": {"a": {"instructions": "?"}}})
        self.assertEqual(code, 400)
        self.assertIn("stream", said)

    def test_a_state_that_is_an_object_is_rendered_rather_than_refused(self):
        """State is what the asking program holds. A client in the wild sends
        an object, and the fields have to reach the prompt."""
        plan = router.systemone_plan(json.dumps({
            "state": {"from": "billing@example.com", "subject": "Invoice 41"},
            "questions": {"ok": {"instructions": "Is it real?"}}}).encode())
        self.assertIn("billing@example.com", plan["state"])
        self.assertIn("Invoice 41", plan["state"])

    def test_a_state_that_is_missing_is_empty(self):
        plan = router.systemone_plan(json.dumps({
            "questions": {"ok": {"instructions": "?"}}}).encode())
        self.assertEqual(plan["state"], "")

    def test_a_refusal_never_reaches_a_backend(self):
        """It is decided before the turn asks for a slot."""
        self.refused({"state": "x"})
        self.assertEqual(self.solo.chats, [])


class OneQuestionAtATime(unittest.TestCase):
    """A plan is a list, and the router answers it one question at a time
    against the slot that holds the state. Each one is a generation, bounded
    only by read_timeout, so the client has to be watched between them as it
    is during the read itself."""

    ANSWER = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": "A", "logprob": -0.1}]}]}}]}

    class Link(TurnLink):
        """TurnLink, plus the one thing the real call does between questions.

        Not pushed down into TurnLink: http_post_watched polls the client
        every two seconds, so a read that answers fast finishes even for a
        client that has gone, and a case there proves it."""

        def work(self, be, path, payload, alive, timeout=None):
            if not alive():
                raise router.Gone("the client stopped waiting")
            return super().work(be, path, payload, alive, timeout)

    def plan(self, how_many):
        body = {"model": "m", "state": "the text", "questions": {
            f"q{n}": {"type": "noul", "instructions": "Well?"}
            for n in range(how_many)}}
        return router.systemone_plan(json.dumps(body).encode())

    def test_it_stops_asking_once_the_client_has_gone(self):
        """The client leaves after the first question of five."""
        link = self.Link(reading=self.ANSWER)
        with self.assertRaises(router.Gone):
            router.answers(link, {"name": "cpu", "url": "http://cpu"}, 0,
                           self.plan(5), 30, lambda: not link.calls)
        self.assertEqual(len(link.calls), 1,
                         "it went on asking with nobody waiting")

    def test_it_asks_every_question_while_the_client_waits(self):
        link = self.Link(reading=self.ANSWER)
        said, _, _ = router.answers(
            link, {"name": "cpu", "url": "http://cpu"}, 0, self.plan(3), 30,
            lambda: True)
        self.assertEqual(len(link.calls), 3)
        self.assertEqual(sorted(said), ["q0", "q1", "q2"])


if __name__ == "__main__":
    unittest.main()
