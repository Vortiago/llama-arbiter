"""Tests for the migration policy: when to move a conversation's KV cache from
one backend to another. It is a pure function, so these need no server and no
network.
"""
import atexit
import base64
import io
import json
import os
import pathlib
import shutil
import socket
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import dataclasses
from dataclasses import replace
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


def make_pool(backends, **kw):
    """A Pool wired to the sandbox. A case that wants another store, tuning or
    event log passes it, and that one wins."""
    kw.setdefault("store", SANDBOX.store)
    kw.setdefault("tuning", SANDBOX.tuning)
    kw.setdefault("events", SANDBOX.events)
    return router.Pool(backends, **kw)


def backend(name, pref, slots=1, busy=0, n_ctx=150000, up=True):
    return {"name": name, "pref": pref, "slots": slots, "busy": busy,
            "n_ctx": n_ctx, "up": up}


def pin(backend_name, slot=0, tokens=SANDBOX.tuning.reply_tokens + 8192,
        last=0.0,
        inflight=False, parking=False, turns=1,
        moved=None, parked=None, parked_turn=None):
    # `tokens` clears PARK_MIN_TOKENS by default, or no test about parking
    # would park anything. A test about the floor names its own number.
    # A copy is of one turn. `parked_turn` defaults to the turn this pin is on,
    # so `parked=` alone means the copy is current; pass an earlier number for
    # a copy the slot has run past.
    return {"backend": backend_name, "slot": slot, "tokens": tokens,
            "last": last, "inflight": inflight, "moved": moved, "parked": parked,
            "turns": turns,
            "parked_turn": turns if parked and parked_turn is None else parked_turn}



class AParkedCopyKeepsItsSize(unittest.TestCase):
    """The size of a copy on disk is what the budget is spent against.

    A pin is rebuilt on every turn. The name of the copy survives that, so
    its size has to survive with it, or a 2 GB copy is counted as free."""

    def test_the_size_survives_the_next_turn(self):
        pool = make_pool([{"name": "cpu", "url": "http://x", "pref": 0}],
                           watch=False)
        pool.pins["c"] = {"backend": "cpu", "slot": 0, "parked": "c.park",
                          "bytes": 2_587_862_136, "turns": 1,
                          "inflight": False, "last": 0, "tokens": 10}
        pool._take(pool.backends[0], "c", tokens=20)
        self.assertEqual(pool.pins["c"]["parked"], "c.park")
        self.assertEqual(pool.pins["c"]["bytes"], 2_587_862_136)


class AnOpeningIsRenderedByItsOwnProtocol(unittest.TestCase):
    """A tool call written the anthropic way is not an openai message.

    /apply-template refuses tool_use and tool_result blocks, so an opening
    from /v1/messages has to be rendered by the route that converts them."""

    def test_the_anthropic_route_renders_an_anthropic_request(self):
        self.assertEqual(router.template_route("/v1/messages"),
                         "/v1/messages/apply-template")

    def test_the_openai_route_renders_an_openai_request(self):
        self.assertEqual(router.template_route("/v1/chat/completions"),
                         "/apply-template")

    def test_the_builder_asks_the_route_the_request_came_in_on(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           watch=False)
        asked = []

        class Rendering(FakeLink):
            def render(inner, be, route, payload, timeout=None):
                asked.append(route)
                return {"prompt": "rendered"}

        pool._render_block("rules", [], [{"role": "user", "content": "hi"}],
                           pool.backends[0], Rendering(), "/v1/messages")
        self.assertEqual(set(asked), {"/v1/messages/apply-template"})


class ARequestCanBeWrittenDown(unittest.TestCase):
    """A prompt that keeps re-reading is one whose start has changed.

    Nothing else in the router says what changed, so for debugging it can
    write the bodies down and let two turns be compared offline. Off unless a
    directory is named, because these hold the whole conversation."""

    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="capture-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.keep = SANDBOX.tuning.capture_keep

    def test_nothing_is_written_when_no_directory_is_named(self):
        router.capture(None, "conv", b'{"a":1}', self.keep)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_the_body_is_written_when_a_directory_is_named(self):
        router.capture(self.root, "abc", b'{"a":1}', self.keep)
        written = list(self.root.glob("*.json"))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].read_bytes(), b'{"a":1}')
        self.assertIn("abc", written[0].name)

    def test_only_the_newest_of_a_conversation_are_kept(self):
        for n in range(self.keep + 5):
            router.capture(self.root, "busy", b'{"n":%d}' % n, self.keep)
        self.assertEqual(len(list(self.root.glob("*.json"))), self.keep)

    def test_a_busy_conversation_does_not_push_out_a_quiet_one(self):
        """The quiet client is the one being looked for.

        OpenCode sends a turn an hour and Claude Code sends one a minute. Kept
        as one list, the hourly body was gone both times it was wanted."""
        router.capture(self.root, "quiet", b'{"quiet":1}', self.keep)
        for n in range(self.keep + 5):
            router.capture(self.root, "busy", b'{"n":%d}' % n, self.keep)
        self.assertEqual([p.read_bytes() for p in self.root.glob("*-quiet.json")],
                         [b'{"quiet":1}'])


class TheSystemPromptIsAlwaysWorthACut(unittest.TestCase):
    """The one cut every session of a client shares is its system prompt.

    Anything deeper carries the first user message, which differs from
    session to session, so if the system prompt does not get a cut of its own
    then nothing can ever be shared between two sessions."""

    def cuts(self, system, messages):
        body = json.dumps({"system": system, "messages": messages}).encode()
        return router.prompt_cuts(body)[0]

    def test_a_real_system_prompt_gets_its_own_cut(self):
        """Claude Code sends about 6,100 characters. At 30 tokens a second
        that is a minute of reading, against a fifth of a second to load the
        block, so it is worth keeping."""
        cuts = self.cuts("R" * 6107, [{"role": "user", "content": "hello"}])
        self.assertTrue(cuts, "no cut at all")
        self.assertEqual(cuts[0][0], -1,
                         "the first cut carries a message, so two sessions "
                         "cannot share it")

    def test_two_sessions_share_that_cut(self):
        a = self.cuts("R" * 6107, [{"role": "user", "content": "first task"}])
        b = self.cuts("R" * 6107, [{"role": "user", "content": "other task"}])
        self.assertEqual(a[0], b[0])

    def test_a_system_prompt_too_short_to_pay_for_itself_gets_none(self):
        cuts = self.cuts("short", [{"role": "user", "content": "hello"}])
        self.assertFalse([c for c in cuts if c[0] == -1])


class AClosedConnectionIsNotAFault(unittest.TestCase):
    """A client that hangs up is ordinary, and must not print a traceback.

    Every client keeps its connection open for the next request and drops it
    when it is done. Python answers that with a ConnectionResetError traceback
    from its own request loop, so a healthy session leaves several in the log.
    They are the first thing anyone reads when something is wrong, and they
    are about nothing."""

    def setUp(self):
        self.said = io.StringIO()
        self.kept, sys.stderr = sys.stderr, self.said
        self.addCleanup(lambda: setattr(sys, "stderr", self.kept))

    def server(self):
        made = router.Server.__new__(router.Server)
        return made

    def test_a_reset_by_the_client_says_nothing(self):
        try:
            raise ConnectionResetError(104, "Connection reset by peer")
        except ConnectionResetError:
            self.server().handle_error(None, ("127.0.0.1", 1))
        self.assertEqual(self.said.getvalue(), "")

    def test_a_broken_pipe_says_nothing_either(self):
        try:
            raise BrokenPipeError(32, "Broken pipe")
        except BrokenPipeError:
            self.server().handle_error(None, ("127.0.0.1", 1))
        self.assertEqual(self.said.getvalue(), "")

    def test_anything_else_is_still_reported(self):
        try:
            raise ValueError("this one is a fault")
        except ValueError:
            self.server().handle_error(None, ("127.0.0.1", 1))
        self.assertIn("this one is a fault", self.said.getvalue())


class OneSessionReadsTheOpeningForAll(unittest.TestCase):
    """Sessions that start together must not each read the same opening.

    Every new session of a client opens with the same system prompt. When
    several start at once none of them has it saved yet, so each reads its own
    copy of the same tokens. Five sessions this afternoon read 92,000 tokens
    between them where 24,000 would have done, and the last finished after
    seventeen minutes.

    So the first to arrive saves the opening and the rest wait for it."""

    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="stampede-"))
        self.kept = {n: getattr(SANDBOX, n) for n in
                     ("store", "tuning")}
        SANDBOX.store = router.Store(self.root)
        SANDBOX.tuning = replace(SANDBOX.tuning, build_patience=5.0)
        self.addCleanup(lambda: [setattr(SANDBOX, n, v)
                                 for n, v in self.kept.items()])
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu",
                                  "pref": 0}],
                                store=SANDBOX.store, watch=False)
        self.be = self.pool.backends[0]
        self.be.update(slots=4, slots_detail=[{"id": i, "busy": False}
                                              for i in range(4)])

    def body(self, opening, first):
        return json.dumps({"system": opening,
                           "messages": [{"role": "user", "content": first}]}
                          ).encode()

    def warm(self, conv, opening, first, link, slot=0):
        self.pool.link = link
        cuts, messages, system, tools = router.prompt_cuts(
            self.body(opening, first))
        return self.pool.warm_prefix(conv, cuts, messages, system, tools,
                                     self.be, slot, "/v1/messages")

    def test_only_one_of_them_reads_it(self):
        opening, reads, started = "R" * 40000, [], threading.Event()

        class OneReader(FakeLink):
            def prefill(inner, be, block, slot, timeout=None):
                reads.append(block)
                started.set()
                time.sleep(0.4)          # long enough for the others to queue
                return {}

            def save(inner, be, slot, name, timeout=None):
                return {"n_written": 10 ** 9}

            def render(inner, be, route, payload, timeout=None):
                return {"prompt": "rendered"}

        post = OneReader()
        threads = [threading.Thread(target=self.warm,
                                    args=(f"c{n}", opening, f"task {n}", post))
                   for n in range(4)]
        for t in threads:
            t.start()
            started.wait(2.0)            # the first one claims the build
        for t in threads:
            t.join(20)
        self.assertEqual(len(reads), 1,
                         f"the opening was read {len(reads)} times")

    def test_the_others_load_what_it_saved(self):
        opening, loaded = "R" * 40000, []

        class Loading(FakeLink):
            def restore(inner, be, slot, name, timeout=None):
                loaded.append(name)
                return {}

            def save(inner, be, slot, name, timeout=None):
                return {"n_written": 10 ** 9}

            def render(inner, be, route, payload, timeout=None):
                return {"prompt": "rendered"}

        post = Loading()
        self.warm("first", opening, "task one", post)      # builds it
        self.warm("second", opening, "task two", post)     # should load it
        self.assertEqual(len(loaded), 1, "the second one did not load it")

    def test_a_build_that_fails_does_not_strand_the_others(self):
        opening = "R" * 40000

        class ReadFails(FakeLink):
            def prefill(inner, be, block, slot, timeout=None):
                raise OSError("the read failed")

            def render(inner, be, route, payload, timeout=None):
                return {"prompt": "rendered"}

        post = ReadFails()
        self.assertFalse(self.warm("first", opening, "one", post))
        began = time.time()
        self.assertFalse(self.warm("second", opening, "two", post))
        self.assertLess(time.time() - began, SANDBOX.tuning.build_patience,
                        "the second one waited for a build that had failed")


class ASlotBeingSavedIsNotHandedOut(unittest.TestCase):
    """A save reads the slot it copies. A restore into that slot while the save
    runs writes one conversation's cache to disk under another's name, and the
    first conversation then re-reads its whole prompt.

    pick_slot used to work out which slots were taken and then fall through
    to ids[0] regardless, so on a one-slot backend it always answered 0."""

    def setUp(self):
        self.link = FakeLink(block=True, written=1 << 30)
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                              link=self.link, watch=False)
        self.be = self.pool.backends[0]
        self.be.update(up=True, n_ctx=150000, slots=1,
                       slots_detail=[{"id": 0, "busy": False}])

    def saving(self, conv):
        """Run one turn of `conv`, then leave its copy mid-save."""
        self.pool._take(self.be, conv, tokens=10)
        self.pool.pick_slot(self.be, conv)
        self.pool.release(self.be, conv)       # the client has its reply
        self.saver = threading.Thread(target=self.pool.park_partial,
                                      args=(conv, self.be, 0), daemon=True)
        self.saver.start()
        self.addCleanup(self.saver.join, 10)
        self.addCleanup(self.link.release)
        self.assertTrue(self.link.started.wait(10), "the save never began")

    def test_the_only_slot_is_refused_while_its_cache_is_saved(self):
        self.saving("first")
        self.assertIsNone(self.pool.pick_slot(self.be, "second"))

    def test_acquire_counts_the_slot_the_save_is_reading(self):
        """pick_slot answering None is the last line, not the plan. acquire
        already waits for a slot, already watches the client, and already
        ranks the other backends, so it is the one that has to know."""
        self.saving("first")
        self.assertIsNone(self.pool.acquire("second", 10, alive=lambda: False)[0])

    def test_the_save_is_claimed_by_save_park_and_not_its_callers(self):
        """Five callers reach _save_park. A claim each one has to remember is
        a claim one of them forgets."""
        self.saving("first")
        # As a set: `saving` counts the saves reading each slot, and this is
        # about which slots are claimed, not how many claims each one has.
        self.assertEqual(set(self.be["saving"]), {0})

    def test_the_claim_and_the_slot_come_back_once_the_save_lands(self):
        """_save_park discards the claim in a finally, before the thread
        ends, so the join is enough: no polling."""
        self.saving("first")
        self.link.release()
        self.saver.join(10)

        self.assertEqual(set(self.be["saving"]), set())
        self.assertEqual(self.pool.pick_slot(self.be, "second"), 0)


class OneFlagPerClaim(unittest.TestCase):
    """`inflight` says this turn owns the pin and its slot. `parking` says a
    copy of the record is being written. They were one flag, and the two
    readings disagree: a turn is over when its client has the reply, a park
    runs on past that."""

    def setUp(self):
        self.link = FakeLink(block=True, written=1 << 30)
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                              link=self.link, watch=False)
        self.be = self.pool.backends[0]
        self.be.update(up=True, n_ctx=150000, slots=2,
                       slots_detail=[{"id": 0, "busy": False},
                                     {"id": 1, "busy": False}])

    def test_a_turn_that_is_over_does_not_still_own_its_slot(self):
        """The park outlives the turn. Told by one flag, a record being
        copied out reads as a turn still holding the slot, and every caller
        that asks `is a turn using this` gets the wrong answer."""
        self.pool._take(self.be, "done", tokens=10)
        self.pool.pick_slot(self.be, "done")
        self.pool.release(self.be, "done")         # the client has its reply
        saver = threading.Thread(target=self.pool.park_partial,
                                 args=("done", self.be, 0), daemon=True)
        saver.start()
        self.addCleanup(saver.join, 10)
        self.addCleanup(self.link.release)
        self.assertTrue(self.link.started.wait(10), "the save never began")

        self.assertFalse(self.pool.pins["done"]["inflight"],
                         "a finished turn still reads as holding its slot")
        self.assertTrue(self.pool.pins["done"]["parking"],
                        "nothing says a copy is being written")
        # The slot being copied out is still refused, by the claim that
        # means what it says. The other one is free and is the answer.
        self.assertEqual(self.pool.pick_slot(self.be, "other"), 1)

    def test_a_turn_saving_over_its_own_slot_does_not_fill_the_backend(self):
        """Every turn calls ensure_parked, which copies the cache it is about
        to read over out of the slot it was just handed. Counted once as the
        turn's and once as the save's, that one slot filled two places for
        the length of the copy, and a backend with an idle slot refused work
        for eleven seconds at the median."""
        self.pool._take(self.be, "reader", tokens=10)
        self.assertEqual(self.pool.pick_slot(self.be, "reader"), 0)
        with self.pool.cv:
            self.be["saving"][0] += 1              # ensure_parked, on slot 0

        self.assertTrue(self.pool._usable(self.be, 10),
                        "slot 1 is idle, and the backend says it is full")

    def test_a_save_on_a_slot_no_turn_holds_still_counts(self):
        """The case the count exists for."""
        self.be["slots"] = 1
        with self.pool.cv:
            self.be["saving"][0] += 1

        self.assertFalse(self.pool._usable(self.be, 10))


class TwoRequestsNeverShareASlot(unittest.TestCase):
    """A slot holds one thing, so two requests cannot both be given it.

    The router used to answer this from the backend poll, which is two seconds
    old, so requests arriving together were all told the same slot. One put
    the opening there, the next overwrote it, and the first then read its whole
    prompt from a slot holding somebody else's. Claude Code's own session title
    request is enough to do it: 1,461 tokens landing on top of a 17,000 token
    opening."""

    def setUp(self):
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu",
                                  "pref": 0}], watch=False)
        self.be = self.pool.backends[0]
        self.be.update(slots=2, slots_detail=[{"id": 0, "busy": False},
                                              {"id": 1, "busy": False}])

    def taking(self, conv):
        self.pool._take(self.be, conv, tokens=10)
        return self.pool.pick_slot(self.be, conv)

    def test_two_conversations_are_given_different_slots(self):
        self.assertNotEqual(self.taking("one"), self.taking("two"))

    def test_a_stale_poll_does_not_hand_the_same_slot_out_twice(self):
        """Every slot still looks free, which is what the poll would say."""
        first = self.taking("one")
        for slot in self.be["slots_detail"]:
            slot["busy"] = False
        self.assertNotEqual(self.taking("two"), first)

    def test_a_conversation_keeps_the_slot_its_cache_is_in(self):
        self.pool._take(self.be, "keeper", tokens=10)
        self.pool.pins["keeper"]["slot"] = 1
        self.assertEqual(self.pool.pick_slot(self.be, "keeper"), 1)

    def test_a_slot_comes_back_when_the_request_ends(self):
        first = self.taking("one")
        self.pool.release(self.be, "one")
        self.assertEqual(self.taking("two"), first)


class ToolsBelongToTheOpening(unittest.TestCase):
    """A client's tools are the biggest part of what every session shares.

    Claude Code sends 25 of them, 56,371 characters, the same in every
    session, and the template renders them inside the system block. Left out
    of the opening, the opening is 1,466 tokens of a 17,375 token prompt and
    loading it saves nothing.

    They also change - a client gains a tool, or the user turns one off - so
    two tool sets have to be two openings, which is what several shelves are
    for."""

    TOOLS = [{"name": "read", "description": "read a file",
              "input_schema": {"type": "object"}},
             {"name": "write", "description": "write a file",
              "input_schema": {"type": "object"}}]

    def cuts(self, system, tools):
        body = json.dumps({"system": system, "tools": tools,
                           "messages": [{"role": "user", "content": "hi"}]})
        return router.prompt_cuts(body.encode())[0]

    def test_the_same_tools_give_the_same_opening(self):
        a = self.cuts("R" * 3000, self.TOOLS)
        b = self.cuts("R" * 3000, list(self.TOOLS))
        self.assertEqual(a[0], b[0])

    def test_a_changed_tool_set_is_a_different_opening(self):
        fewer = self.TOOLS[:1]
        self.assertNotEqual(self.cuts("R" * 3000, self.TOOLS)[0],
                            self.cuts("R" * 3000, fewer)[0])

    def test_a_changed_description_is_a_different_opening(self):
        other = [dict(self.TOOLS[0], description="read a file, carefully"),
                 self.TOOLS[1]]
        self.assertNotEqual(self.cuts("R" * 3000, self.TOOLS)[0],
                            self.cuts("R" * 3000, other)[0])

    def openai_cuts(self, system, tools):
        """The same request as an openai client sends it.

        There is no system field: the system prompt is the first message."""
        body = json.dumps({"tools": tools,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": "hi"}]})
        return router.prompt_cuts(body.encode())[0]

    def test_an_openai_opening_is_its_first_message(self):
        """Nothing can be rendered before that message.

        The tools on their own are not a prompt and the template refuses an
        empty one, so a cut before the messages saves nothing and the request
        reads its whole prompt from cold. The opening is the cut at the system
        message, which renders the rules and the tools together."""
        cuts = self.openai_cuts("R" * 3000, self.TOOLS)
        self.assertTrue(cuts, "an openai request shares no opening at all")
        self.assertEqual(cuts[0][0], 0)

    def test_the_tools_still_name_that_opening(self):
        """Two openai clients with different tools do not share an opening."""
        self.assertNotEqual(self.openai_cuts("R" * 3000, self.TOOLS)[0],
                            self.openai_cuts("R" * 3000, self.TOOLS[:1])[0])

    def test_the_opening_is_rendered_with_them(self):
        """Rendered without the tools it is not a prefix of the real prompt,
        because the template puts them inside the system block."""
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           watch=False)
        sent = []

        class Rendering(FakeLink):
            def render(inner, be, route, payload, timeout=None):
                sent.append(payload)
                return {"prompt": "rendered"}

        pool._render_block("rules", self.TOOLS,
                           [{"role": "user", "content": "hi"}],
                           pool.backends[0], Rendering(), "/v1/messages")
        self.assertTrue(sent, "it rendered nothing")
        for payload in sent:
            self.assertEqual(payload.get("tools"), self.TOOLS,
                             "the opening was rendered without the tools")


class ACutHasToBeSomewhereATemplateCanStop(unittest.TestCase):
    """An assistant message that calls a tool is the model mid-turn.

    The template will not close one: "Cannot continue an assistant message
    that contains tool calls". A cut there renders nothing, so the block never
    saves, and the whole conversation reads from cold instead. The tool result
    that answers it is the next cut, and that one renders."""

    LONG = "T" * SANDBOX.tuning.prefix_min_chars

    def cuts(self, messages):
        return router.prompt_cuts(json.dumps({"messages": messages}).encode())[0]

    def test_an_openai_tool_call_is_not_a_cut(self):
        cuts = self.cuts([
            {"role": "user", "content": self.LONG},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "the file"}])
        self.assertEqual([index for index, _ in cuts], [0, 2])

    def test_an_anthropic_tool_call_is_not_a_cut(self):
        cuts = self.cuts([
            {"role": "user", "content": self.LONG},
            {"role": "assistant",
             "content": [{"type": "tool_use", "id": "1", "name": "read",
                          "input": {}}]},
            {"role": "user",
             "content": [{"type": "tool_result", "tool_use_id": "1",
                          "content": "the file"}]}])
        self.assertEqual([index for index, _ in cuts], [0, 2])

    def test_an_assistant_that_only_talked_is_still_a_cut(self):
        cuts = self.cuts([
            {"role": "user", "content": self.LONG},
            {"role": "assistant", "content": "here you go"}])
        self.assertEqual([index for index, _ in cuts], [0, 1])


class AConversationKeyIsAFileName(unittest.TestCase):
    """Whatever names a conversation also names its slot file."""

    def test_a_colon_between_a_session_and_its_agent_becomes_a_dash(self):
        key = router.session_key({"x-claude-code-session-id": "abc",
                                  "x-claude-code-agent-id": "def"})
        self.assertEqual(key, "abc-def")

    def test_a_session_on_its_own_is_left_alone(self):
        self.assertEqual(
            router.session_key({"x-claude-code-session-id":
                                "44af7460-f95e-4430-830f-9441a4d0da20"}),
            "44af7460-f95e-4430-830f-9441a4d0da20")

    def test_no_client_key_can_name_a_path(self):
        self.assertEqual(router.file_safe("../../etc/passwd"),
                         "etc-passwd")

    def test_a_key_of_nothing_usable_still_names_something(self):
        self.assertEqual(router.file_safe("///"), "conversation")

    def test_it_obeys_the_two_rules_beyond_the_character_set(self):
        """llama.cpp refuses ".." anywhere and anything over 255 characters,
        before it looks at the slot. A key carrying either was never parked."""
        self.assertNotIn("..", router.file_safe("proj/../v2"))
        self.assertNotIn("..", router.file_safe("a...b"))
        self.assertLessEqual(len(router.file_safe("x" * 400)) + len(".park"), 255)

    def test_the_characters_llama_refuses_are_all_gone(self):
        safe = router.file_safe('a:b*c?d"e<f>g|h\\i/j')
        self.assertFalse(set(safe) & set(':*?"<>|/\\'), safe)

def png(width, height, weigh=0):
    """A real png of a given size. Only the header is read, so the pixels are
    blank; `weigh` pads the file out to the bulk a photograph would have."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    head = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pad = chunk(b"tEXt", b"pad\x00" + os.urandom(weigh)) if weigh else b""
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + pad
            + chunk(b"IDAT", zlib.compress(b"\x00" * height)) + chunk(b"IEND", b""))


def openai_body(*images, text="hello"):
    content = [{"type": "text", "text": text}]
    for raw in images:
        url = "data:image/png;base64," + base64.b64encode(raw).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    return json.dumps({"messages": [{"role": "user", "content": content}]}).encode()


class WhatAnImageCosts(unittest.TestCase):
    """A picture is charged by the area the vision encoder sees, not by the
    length of its base64. The two differ by a factor of a hundred or more, and
    counting the base64 refused a screenshot that fits with room to spare."""

    def most(self):
        align = router.VISION["patch_size"] * router.VISION["n_merge"]
        return router.VISION["image_max_pixels"] // (align * align)

    def test_a_picture_costs_what_the_backend_charged_for_it(self):
        """A 240x120 png measured n_tokens_batch = 32 on cpu0_0."""
        self.assertEqual(router.image_tokens(
            base64.b64encode(png(240, 120)).decode()), 32)

    def test_a_screenshot_costs_its_aligned_squares(self):
        self.assertEqual(router.image_tokens(
            base64.b64encode(png(1280, 800)).decode()), 40 * 25)

    def test_a_tiny_picture_is_pushed_up_to_the_smallest_the_encoder_takes(self):
        square = (router.VISION["patch_size"] * router.VISION["n_merge"]) ** 2
        self.assertGreaterEqual(
            router.image_tokens(base64.b64encode(png(8, 8)).decode()) * square,
            router.VISION["image_min_pixels"])

    def test_a_huge_picture_is_capped(self):
        self.assertLessEqual(
            router.image_tokens(base64.b64encode(png(8000, 8000)).decode()),
            self.most())

    def test_a_header_it_cannot_read_costs_the_most_a_picture_can(self):
        """Never let a format we cannot measure in under its real weight."""
        self.assertEqual(router.image_tokens(
            base64.b64encode(b"not a picture at all").decode()), self.most())

    def test_it_charges_by_the_geometry_it_is_given(self):
        """A backend on another mmproj has another patch size, and the same
        picture costs a different number of tokens there."""
        picture = base64.b64encode(png(240, 120)).decode()
        coarse = dict(router.VISION, patch_size=32)
        self.assertEqual(router.image_tokens(picture, coarse), 4 * 2)

    def test_a_screenshot_fits_where_its_base64_would_not(self):
        """Half a megabyte of png is 166,000 tokens of base64 and 1000 of
        picture. The largest backend holds 150,016, so counting the base64
        refused a screenshot that fits with room to spare."""
        body = openai_body(png(1280, 800, weigh=500000))
        self.assertGreater(len(body) / SANDBOX.tuning.chars_per_tok, 150016)
        self.assertLess(router.token_estimate(body), 150016)

    def test_each_picture_is_counted_once(self):
        """The url sits inside an image_url that the walk also descends into,
        so a careless walk charges twice for one picture."""
        body = openai_body(png(1280, 800), png(1280, 800))
        self.assertEqual(len(router.images_in(body)), 2)
        self.assertEqual(sum(router.image_tokens(p)
                             for p in router.images_in(body)), 2000)

    def test_it_reads_the_shape_claude_code_sends(self):
        body = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": base64.b64encode(
                                             png(240, 120)).decode()}}]}]}).encode()
        self.assertEqual(router.token_estimate(body) - SANDBOX.tuning.reply_tokens - 32,
                         int((len(body) - len(base64.b64encode(png(240, 120))))
                             / SANDBOX.tuning.chars_per_tok))

    def test_a_request_with_no_picture_is_still_its_length(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        self.assertEqual(router.token_estimate(body),
                         int(len(body) / SANDBOX.tuning.chars_per_tok) + SANDBOX.tuning.reply_tokens)

    def test_a_body_that_is_not_json_is_still_its_length(self):
        self.assertEqual(router.token_estimate(b"<not json>"),
                         int(10 / SANDBOX.tuning.chars_per_tok) + SANDBOX.tuning.reply_tokens)


class TheVisionGeometryComesFromTheBackend(unittest.TestCase):
    """Hard-coding the mmproj's numbers means they drift the day it changes.
    A backend prints them when it loads one, so they are read from there."""

    HPARAMS = [
        "print_info: n_merges              = 247587\n",
        "--- vision hparams ---\n",
        "load_hparams: image_size:         768\n",
        "load_hparams: patch_size:         16\n",
        "load_hparams: n_merge:            2\n",
        "load_hparams: image_min_pixels:   8192\n",
        "load_hparams: image_max_pixels:   4194304\n",
    ]

    def test_it_reads_what_the_backend_printed(self):
        self.assertEqual(router.read_vision(self.HPARAMS),
                         {"patch_size": 16, "n_merge": 2,
                          "image_min_pixels": 8192,
                          "image_max_pixels": 4194304})

    def test_the_tokenizer_merges_are_not_the_encoder_merges(self):
        """n_merges is printed first and is in the hundreds of thousands. It
        once read as n_merge, which made every picture cost nothing."""
        self.assertIsNone(router.read_vision(self.HPARAMS[:1]))
        self.assertEqual(router.read_vision(self.HPARAMS)["n_merge"], 2)

    def test_a_backend_with_no_mmproj_says_nothing(self):
        self.assertIsNone(router.read_vision(["nothing to see here\n"]))

    def test_half_a_log_says_nothing_rather_than_something_wrong(self):
        self.assertIsNone(router.read_vision(self.HPARAMS[:4]))


class FakeLink:
    """Stand in for the link to a backend. Records what was asked of it.

    A call is (operation, backend name, ...), so a test says which operation
    on which slot with which file rather than matching a URL. The router no
    longer builds those URLs; the Link does, and a test that asserted on one
    would be asserting on the thing the Link exists to own.

    `written` is what a save reports it wrote. The router spends park_floor
    and park_budget against that number, so a test about either says it here.

    `block` holds every call until release(), for the cases about a copy that
    is still being written: `started` says one has arrived and is waiting.
    """

    def __init__(self, fail_on=None, written=0, block=False):
        self.calls = []
        self.fail_on = fail_on          # an operation that should fail
        self.written = written
        self.started = threading.Event()
        self.go = threading.Event()
        if not block:
            self.go.set()

    def release(self):
        """Let every call through, the one waiting and any after it."""
        self.go.set()

    def ops(self):
        """Just the operation names, in order."""
        return [call[0] for call in self.calls]

    def files(self, op=None):
        """The file names this link was asked to save or restore. A prefill
        carries a block where those carry a name, so it is not one of these."""
        asked = (op,) if op else ("save", "restore")
        return [call[3] for call in self.calls
                if len(call) > 3 and call[0] in asked]

    def _note(self, op, be, *rest):
        self.calls.append((op, be["name"]) + rest)
        self.started.set()
        self.go.wait(10.0)
        if self.fail_on and self.fail_on == op:
            raise OSError("backend said no")
        return {"id_slot": 0, "n_saved": 3, "n_written": self.written}

    def save(self, be, slot, name, timeout=None):
        return self._note("save", be, slot, name)

    def restore(self, be, slot, name, timeout=None):
        return self._note("restore", be, slot, name)

    def prefill(self, be, block, slot, timeout=None):
        return self._note("prefill", be, slot, block)

    def render(self, be, route, payload, timeout=None):
        self._note("render", be, route)
        # Two renderings that share everything up to the opening. The router
        # keeps the common prefix, so the answer decides what a block is.
        messages = payload.get("messages") or []
        return {"prompt": "".join(str(m.get("content", "")) for m in messages)}

    def props(self, be, timeout=None):
        return self._note("props", be)

    def slots(self, be, timeout=None):
        self._note("slots", be)
        return []

    def metrics(self, be, timeout=None):
        self._note("metrics", be)
        return ""


def with_link(pool, link):
    """Give this pool the link and hand the pool back, so a call that used to
    take a poster stays one expression."""
    pool.link = link
    return pool


def linked(pool, link):
    """Give this pool the link, and hand the link back.

    A Pool is handed one link when it is built and uses it for every call, so
    a test that wants to watch the calls replaces it here rather than passing
    a poster to each method."""
    pool.link = link
    return link


class Bookkeeping(unittest.TestCase):
    """Pool keeps one record per conversation, and the policy reads it."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0},
             {"name": "cpu", "url": "http://cpu", "pref": 1}], watch=False)
        self.cpu = self.pool.backends[1]

    def test_taking_a_slot_records_the_conversation_as_in_flight(self):
        self.pool._take(self.cpu, "conv1", tokens=500)
        record = self.pool.pins["conv1"]
        self.assertEqual(record["backend"], "cpu")
        self.assertEqual(record["tokens"], 500)
        self.assertTrue(record["inflight"])
        self.assertIsNone(record["slot"])

    def test_releasing_marks_the_conversation_idle(self):
        self.pool._take(self.cpu, "conv1", tokens=500)
        self.pool.release(self.cpu, "conv1")
        self.assertFalse(self.pool.pins["conv1"]["inflight"])

    def test_noting_a_slot_records_where_the_conversation_landed(self):
        self.pool._take(self.cpu, "conv1", tokens=500)
        self.pool.note_slot("conv1", 2)
        self.assertEqual(self.pool.pins["conv1"]["slot"], 2)

    def test_noting_a_slot_for_an_unknown_conversation_is_harmless(self):
        self.pool.note_slot("never-seen", 1)     # must not raise



class CacheEvents(unittest.TestCase):
    """Read what the backend says about its prompt cache.

    These lines decide whether a disk tier is worth building: an eviction is a
    conversation that will have to prefill from scratch when it returns."""

    def test_reads_a_size_limit_eviction(self):
        line = ("5.01.002.003 W srv        update:  - cache size limit reached, "
                "removing oldest entry (size = 342.632 MiB)")
        self.assertEqual(router.cache_event(line), ("evicted", 342.632))

    def test_reads_a_token_limit_eviction(self):
        line = ("5.01.002.003 W srv        update:  - cache token limit (450048, "
                "est: 450048) reached, removing oldest entry (size = 226.077 MiB)")
        self.assertEqual(router.cache_event(line), ("evicted", 226.077))

    def test_reads_a_make_room_eviction(self):
        line = ("5.01.002.003 W srv        update:  - making room for prompt cache "
                "entry, removing oldest entry (size = 115.265 MiB)")
        self.assertEqual(router.cache_event(line), ("evicted", 115.265))

    def test_reads_a_prompt_too_big_to_cache(self):
        line = ("5.01.002.003 W srv        update:  - prompt state size 20480.000 MiB "
                "exceeds cache size limit 16384.000 MiB, skipping")
        self.assertEqual(router.cache_event(line), ("skipped", 20480.0))

    def test_reads_the_cache_state_line(self):
        line = ("5.01.002.003 I srv        update:  - cache state: 3 prompts, "
                "566.070 MiB (limits: 16384.000 MiB, 450048 tokens, 450048 est)")
        self.assertEqual(router.cache_event(line), ("state", (3, 566.070, 16384.0)))

    def test_reads_the_limit_the_backend_starts_with(self):
        line = ("0.11.107.926 I srv    load_model: prompt cache is enabled, "
                "size limit: 8192 MiB")
        self.assertEqual(router.cache_event(line), ("limit", 8192.0))

    def test_ignores_an_unrelated_line(self):
        self.assertIsNone(router.cache_event("5.01.002.003 I srv  log_server_r: done"))

    def test_ignores_an_empty_line(self):
        self.assertIsNone(router.cache_event(""))


EVICT = ("W srv update:  - cache size limit reached, removing oldest entry "
         "(size = 100.000 MiB)\n")
STATE = ("I srv update:  - cache state: 2 prompts, 500.000 MiB "
         "(limits: 16384.000 MiB, 450048 tokens, 450048 est)\n")
LIMIT = "I srv load_model: prompt cache is enabled, size limit: 8192 MiB\n"


class CacheWatching(unittest.TestCase):
    """Follow a backend log and total what it says about the cache."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / "backend.log"
        self.path.write_text("")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def watch(self):
        return router.CacheWatch(self.path)

    def test_counts_what_is_already_in_the_file(self):
        self.path.write_text(EVICT + EVICT)
        watch = self.watch()
        watch.poll()
        self.assertEqual(watch.stats["evictions"], 2)
        self.assertAlmostEqual(watch.stats["evicted_mib"], 200.0)

    def test_a_restart_clears_the_state_it_had_before(self):
        """The old size describes a cache that no longer exists.

        cpu0_0 came back at 8192 MiB and the dashboard showed the 16384 it had
        before, because a state line only arrives once traffic touches the
        cache."""
        self.path.write_text(STATE)
        watch = self.watch()
        watch.poll()
        self.path.write_text(LIMIT)            # shorter, so it restarted
        watch.poll()
        self.assertEqual(watch.stats["prompts"], 0)
        self.assertAlmostEqual(watch.stats["used_mib"], 0.0)
        self.assertAlmostEqual(watch.stats["limit_mib"], 8192.0)

    def test_keeps_the_latest_cache_state(self):
        self.path.write_text(STATE)
        watch = self.watch()
        watch.poll()
        self.assertEqual(watch.stats["prompts"], 2)
        self.assertAlmostEqual(watch.stats["used_mib"], 500.0)
        self.assertAlmostEqual(watch.stats["limit_mib"], 16384.0)

    def test_reads_only_the_new_lines_next_time(self):
        watch = self.watch()
        self.path.write_text(EVICT)
        watch.poll()
        with self.path.open("a") as handle:
            handle.write(EVICT)
        watch.poll()
        self.assertEqual(watch.stats["evictions"], 2)

    def test_starts_over_when_the_backend_restarts(self):
        """A restart truncates the log, so the old offset is past the end."""
        self.path.write_text(EVICT + EVICT)
        watch = self.watch()
        watch.poll()
        self.path.write_text(EVICT)          # shorter: the backend restarted
        watch.poll()
        self.assertEqual(watch.stats["evictions"], 1)

    def test_a_missing_file_is_not_an_error(self):
        watch = router.CacheWatch(Path(self.dir) / "gone.log")
        watch.poll()
        self.assertEqual(watch.stats["evictions"], 0)


class ClientConfig(unittest.TestCase):
    """Configs are built from the address the reader used, so a download works
    whether it came over tailscale, the lan, or a tunnel."""

    def opencode(self, host="example.test:8090"):
        return router.client_config("opencode", host, "qwen-test", 150000)

    def claude(self, host="example.test:8090"):
        return router.client_config("claude", host, "qwen-test", 150000)

    def test_opencode_points_at_the_host_that_was_used(self):
        config = self.opencode()
        url = config["provider"][router.default_provider()]["options"]["baseURL"]
        self.assertEqual(url, "http://example.test:8090/v1")

    def test_opencode_names_the_model_the_backend_reports(self):
        config = self.opencode()
        self.assertEqual(config["model"], f"{router.default_provider()}/qwen-test")
        self.assertIn("qwen-test", config["provider"][router.default_provider()]["models"])

    def test_opencode_carries_the_measured_context_limit(self):
        limit = (self.opencode()["provider"][router.default_provider()]
                 ["models"]["qwen-test"]["limit"])
        self.assertEqual(limit["context"], 150000)

    def test_opencode_keeps_reasoning_turned_on(self):
        model = self.opencode()["provider"][router.default_provider()]["models"]["qwen-test"]
        self.assertTrue(model["reasoning"])
        self.assertEqual(model["interleaved"]["field"], "reasoning_content")

    def test_opencode_offers_images(self):
        """attachment alone leaves modalities.input empty, and OpenCode then
        reads the model as text only."""
        model = self.opencode()["provider"][router.default_provider()]["models"]["qwen-test"]
        self.assertTrue(model["attachment"])
        self.assertIn("image", model["modalities"]["input"])

    def test_claude_points_at_the_router_root(self):
        """Claude Code speaks the messages api, which the backends also serve."""
        env = self.claude()["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://example.test:8090")

    def test_claude_names_the_model(self):
        env = self.claude()["env"]
        self.assertEqual(env["ANTHROPIC_MODEL"], "qwen-test")

    def test_claude_states_the_real_context_window(self):
        """Claude Code assumes 200k for a model it does not know, which is more
        than this backend holds. Auto-compact must use the real number."""
        env = self.claude()["env"]
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "150000")

    def test_the_context_window_follows_the_backend(self):
        env = router.client_config("claude", "h:1", "m", 96000)["env"]
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "96000")

    def test_claude_waits_longer_than_a_full_prompt_takes_to_read(self):
        """Reading a full window at 15 tokens a second is the worst case
        measured. Every timeout has to outlast it, or the client hangs up
        mid-prefill."""
        env = self.claude()["env"]
        floor_ms = 150000 // 15 * 1000
        for name in ("API_TIMEOUT_MS", "CLAUDE_STREAM_IDLE_TIMEOUT_MS",
                     "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS"):
            self.assertGreaterEqual(int(env[name]), floor_ms, name)

    def test_claude_turns_off_the_five_minute_idle_timeout(self):
        self.assertEqual(self.claude()["env"]["API_FORCE_IDLE_TIMEOUT"], "0")

    def test_claude_points_background_work_at_the_same_model(self):
        """Background work uses the haiku slot, which would otherwise name a
        model this backend does not have."""
        self.assertEqual(self.claude()["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"],
                         "qwen-test")

    def test_claude_keeps_traffic_off_the_internet(self):
        env = self.claude()["env"]
        for name in ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                     "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING"):
            self.assertEqual(env[name], "1", name)

    def test_timeouts_grow_with_the_context_window(self):
        small = router.client_config("claude", "h:1", "m", 32000)["env"]
        large = router.client_config("claude", "h:1", "m", 400000)["env"]
        self.assertLess(int(small["CLAUDE_STREAM_IDLE_TIMEOUT_MS"]),
                        int(large["CLAUDE_STREAM_IDLE_TIMEOUT_MS"]))

    def test_opencode_waits_as_long_as_claude_does(self):
        options = self.opencode()["provider"][router.default_provider()]["options"]
        floor_ms = 150000 // 15 * 1000
        for name in ("timeout", "headerTimeout", "chunkTimeout"):
            self.assertGreaterEqual(options[name], floor_ms, name)

    def test_opencode_keeps_conversations_off_the_internet(self):
        """The analogue of turning telemetry off in the other client."""
        self.assertEqual(self.opencode()["share"], "disabled")

    def test_opencode_uses_the_same_model_for_background_work(self):
        config = self.opencode()
        self.assertEqual(config["small_model"], config["model"])

    def test_opencode_sends_a_cache_key(self):
        """Off by default. On, it names the session in every request."""
        options = self.opencode()["provider"][router.default_provider()]["options"]
        self.assertTrue(options["setCacheKey"])

    def test_an_unknown_client_has_no_config(self):
        self.assertIsNone(router.client_config("emacs", "h:1", "m", 10))

    def test_a_host_with_no_port_still_works(self):
        config = self.opencode(host="somehost")
        self.assertEqual(
            config["provider"][router.default_provider()]["options"]["baseURL"],
            "http://somehost/v1")


class RequestShape(unittest.TestCase):
    """Describe a rejected request without keeping any of its text."""

    def test_lists_the_roles_in_order(self):
        body = json.dumps({"messages": [{"role": "system", "content": "a"},
                                        {"role": "user", "content": "b"},
                                        {"role": "system", "content": "c"}]})
        self.assertEqual(router.request_shape(body.encode())["roles"],
                         ["system", "user", "system"])

    def test_notes_a_top_level_system_field(self):
        body = json.dumps({"system": "be terse", "messages": []})
        self.assertEqual(router.request_shape(body.encode())["system"], "str")

    def test_notes_when_there_is_no_system_field(self):
        body = json.dumps({"messages": []})
        self.assertIsNone(router.request_shape(body.encode())["system"])

    def test_counts_tools(self):
        body = json.dumps({"messages": [], "tools": [{"name": "a"}, {"name": "b"}]})
        self.assertEqual(router.request_shape(body.encode())["tools"], 2)

    def test_keeps_no_message_text(self):
        body = json.dumps({"messages": [{"role": "user", "content": "SECRET"}]})
        self.assertNotIn("SECRET", json.dumps(router.request_shape(body.encode())))

    def test_a_body_it_cannot_parse_has_no_shape(self):
        self.assertIsNone(router.request_shape(b"not json"))


class HoistSystem(unittest.TestCase):
    """This model's template refuses a system message that is not at the front.

    Serving the request means changing it, and where the change lands decides
    whether the prompt can be cached. Claude Code sends a token counter as a
    system message at the end of every turn, and its value differs every time.
    Carried to the front, that one message ends the shared prefix a few
    thousand tokens in and the whole conversation behind it is read again.
    Left where it is, everything before it still matches."""

    def hoist(self, messages):
        out = router.hoist_system(json.dumps({"messages": messages}).encode())
        return json.loads(out)["messages"]

    def test_a_late_system_message_stays_where_it_was(self):
        got = self.hoist([{"role": "user", "content": "a"},
                          {"role": "system", "content": "rules"}])
        self.assertEqual([m["content"] for m in got], ["a", "rules"])

    def test_and_becomes_a_role_the_template_accepts(self):
        got = self.hoist([{"role": "user", "content": "a"},
                          {"role": "system", "content": "rules"}])
        self.assertNotIn("system", [m["role"] for m in got[1:]])

    def test_everything_before_it_is_untouched(self):
        """The point of the whole exercise: the prefix has to still match."""
        head = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"}]
        first = self.hoist(head + [{"role": "system", "content": "count 1"}])
        again = self.hoist(head + [{"role": "system", "content": "count 2"}])
        self.assertEqual(first[:len(head)], again[:len(head)])
        self.assertEqual(first[:len(head)], head)

    def test_a_leading_system_message_is_left_as_it_is(self):
        got = self.hoist([{"role": "system", "content": "rules"},
                          {"role": "user", "content": "a"},
                          {"role": "system", "content": "late"}])
        self.assertEqual(got[0], {"role": "system", "content": "rules"})

    def test_the_order_of_everything_is_kept(self):
        got = self.hoist([{"role": "user", "content": "a"},
                          {"role": "assistant", "content": "b"},
                          {"role": "system", "content": "rules"},
                          {"role": "user", "content": "c"}])
        self.assertEqual([m["content"] for m in got], ["a", "b", "rules", "c"])

    def test_treats_developer_like_system(self):
        got = self.hoist([{"role": "user", "content": "a"},
                          {"role": "developer", "content": "rules"}])
        self.assertNotIn("developer", [m["role"] for m in got])
        self.assertEqual([m["content"] for m in got], ["a", "rules"])

    def test_leaves_a_correct_body_byte_for_byte(self):
        body = json.dumps({"messages": [{"role": "system", "content": "s"},
                                        {"role": "user", "content": "u"}]}).encode()
        self.assertEqual(router.hoist_system(body), body)

    def test_leaves_a_body_with_no_system_message_alone(self):
        body = json.dumps({"messages": [{"role": "user", "content": "u"}]}).encode()
        self.assertEqual(router.hoist_system(body), body)

    def test_leaves_a_body_it_cannot_parse_alone(self):
        self.assertEqual(router.hoist_system(b"not json"), b"not json")


class SessionKey(unittest.TestCase):
    """Claude Code names its own session in a header, so the router does not
    have to guess a conversation from the text of the prompt."""

    def test_reads_the_session_id(self):
        self.assertEqual(router.session_key({"x-claude-code-session-id": "abc"}),
                         "abc")

    def test_header_names_are_case_insensitive(self):
        self.assertEqual(router.session_key({"X-Claude-Code-Session-Id": "abc"}),
                         "abc")

    def test_a_subagent_is_its_own_conversation(self):
        """A subagent has its own prompt, so it must not share a pin."""
        both = router.session_key({"x-claude-code-session-id": "abc",
                                   "x-claude-code-agent-id": "sub1"})
        self.assertNotEqual(both, router.session_key({"x-claude-code-session-id": "abc"}))
        self.assertIn("abc", both)
        self.assertIn("sub1", both)

    def test_no_header_means_no_key(self):
        self.assertIsNone(router.session_key({"content-type": "application/json"}))

    def test_an_empty_session_id_is_not_a_key(self):
        self.assertIsNone(router.session_key({"x-claude-code-session-id": ""}))


class ClaudeGatewaySettings(unittest.TestCase):
    """Settings that matter because this is not an Anthropic endpoint."""

    def env(self):
        return router.client_config("claude", "h:1", "qwen-test", 150000)["env"]

    def test_drops_the_attribution_block(self):
        """The block carries a per-conversation fingerprint at the very front
        of the system prompt, which stops two sessions sharing a prefix."""
        self.assertEqual(self.env()["CLAUDE_CODE_ATTRIBUTION_HEADER"], "0")

    def test_does_not_send_pre_release_body_fields(self):
        self.assertEqual(self.env()["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"], "1")

    def test_offers_the_model_in_the_picker(self):
        """Gateway discovery only keeps ids containing claude or anthropic, so
        this model would never appear on its own."""
        self.assertEqual(self.env()["ANTHROPIC_CUSTOM_MODEL_OPTION"], "qwen-test")


class PromptKey(unittest.TestCase):
    """OpenCode names its session in the body, the way Claude Code names its
    own in a header. Both beat guessing from the text of the prompt."""

    def test_reads_the_prompt_cache_key(self):
        body = json.dumps({"prompt_cache_key": "oc-123", "messages": []}).encode()
        self.assertEqual(router.prompt_key(body), "oc-123")

    def test_no_key_in_the_body_means_none(self):
        self.assertIsNone(router.prompt_key(json.dumps({"messages": []}).encode()))

    def test_an_empty_key_is_not_a_key(self):
        body = json.dumps({"prompt_cache_key": "  ", "messages": []}).encode()
        self.assertIsNone(router.prompt_key(body))

    def test_a_body_it_cannot_parse_has_no_key(self):
        self.assertIsNone(router.prompt_key(b"not json"))

    def test_a_key_that_is_not_text_is_ignored(self):
        body = json.dumps({"prompt_cache_key": 17, "messages": []}).encode()
        self.assertIsNone(router.prompt_key(body))


class WantsPing(unittest.TestCase):
    """A silent stream is aborted by the client after five minutes. The router
    fills the silence, but only where extra bytes are safe to insert."""

    def test_pings_a_streamed_event_stream(self):
        self.assertTrue(router.wants_ping("text/event-stream", None))

    def test_ignores_the_charset_on_the_content_type(self):
        self.assertTrue(router.wants_ping("text/event-stream; charset=utf-8", None))

    def test_does_not_ping_a_plain_json_reply(self):
        """Inserting bytes into a json body would corrupt it."""
        self.assertFalse(router.wants_ping("application/json", None))

    def test_does_not_ping_when_the_length_is_known(self):
        """A counted body has no room for extra bytes."""
        self.assertFalse(router.wants_ping("text/event-stream", "1024"))

    def test_a_missing_content_type_is_not_a_stream(self):
        self.assertFalse(router.wants_ping(None, None))


class PinIsAbsolute(unittest.TestCase):
    """A conversation waits for the backend holding its cache.

    Reading a long prompt again costs far more than any wait, and the router
    has no way to move a cache that is no longer in a slot. So the pin holds
    until the backend can never serve the request, or the deadline passes."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0},
             {"name": "cpu", "url": "http://cpu", "pref": 1}], watch=False)
        self.gpu, self.cpu = self.pool.backends
        self.gpu.update(up=True, slots=1, busy=1, n_ctx=150000)   # occupied
        self.cpu.update(up=True, slots=3, busy=0, n_ctx=150000)   # wide open
        self.pool.pins["conv1"] = pin("gpu", slot=0, tokens=5000)

    def take(self):
        got = {}
        thread = threading.Thread(
            target=lambda: got.update(be=self.pool.acquire("conv1", 5000)[0]))
        thread.daemon = True
        thread.start()
        return got, thread

    def free_the_pin(self):
        with self.pool.cv:
            self.gpu["busy"] = 0
            self.pool.cv.notify_all()

    def test_waits_for_its_own_backend_while_another_is_free(self):
        got, thread = self.take()
        thread.join(2.5)
        self.assertEqual(got, {}, "spilled to the other backend instead of waiting")
        self.free_the_pin()
        thread.join(3)
        self.assertEqual(got["be"]["name"], "gpu")

    def test_gives_up_on_the_pin_and_takes_a_free_backend(self):
        """A pin is worth a short wait, not an idle machine. After that the
        request goes wherever there is room."""
        original = SANDBOX.tuning
        # The pool took its tuning when it was built, so hand it the new one
        # as well. That is the dependency being explicit rather than global.
        SANDBOX.tuning = self.pool.tuning = replace(original, pin_patience=1.0)
        try:
            got, thread = self.take()
            thread.join(5)
            self.assertEqual(got["be"]["name"], "cpu")
        finally:
            SANDBOX.tuning = self.pool.tuning = original

    def test_spills_when_the_pinned_backend_is_down(self):
        self.gpu["up"] = False
        got, thread = self.take()
        thread.join(3)
        self.assertEqual(got["be"]["name"], "cpu")

    def test_spills_when_the_request_cannot_fit_the_pinned_backend(self):
        self.gpu["n_ctx"] = 1000                 # smaller than the request
        got, thread = self.take()
        thread.join(3)
        self.assertEqual(got["be"]["name"], "cpu")


class TheSuiteCannotTouchARunningRouter(unittest.TestCase):
    """The one case that is about the tests.

    STORE is the default store a Pool takes when it is handed none.
    A case that forgets to hand it one writes into the checkout's own
    run/slots. Two of the files there are instructions. pins.json says which
    parked copies to keep, and adopt() deletes every copy it does not name.
    A fixture pins.json is therefore a delete-everything order, against
    copies that cost twenty minutes each to rebuild. Guarded at the top of
    this module, and here so that removing the guard fails rather than goes
    quiet."""

    def test_the_slot_directory_is_not_the_one_a_router_uses(self):
        # The checkout, not bin/: router.__file__ is bin/router/__init__.py now
        # that the router is a package, so the parents this counts must match
        # __main__.build(), which takes parents[2] of bin/router/__main__.py.
        # Counted from the wrong depth this compared against bin/run, a
        # directory no router writes to, and could no longer fail.
        checkout = Path(router.__file__).resolve().parents[2] / "run"
        self.assertNotEqual(SANDBOX.store.run.resolve(), checkout,
                            "STORE points at a live router's files")
        # Both directories, as the guard this replaced covered both SLOT_DIR
        # and BLOCK_DIR. A store that kept its copies elsewhere but put its
        # openings in the live blocks directory passed the first line alone.
        self.assertNotEqual(SANDBOX.store.blocks.resolve(), checkout / "blocks",
                            "STORE points its openings at a live router's files")

    def test_the_files_that_are_instructions_land_in_the_sandbox(self):
        for path in (SANDBOX.store.slots / "pins.json",
                     SANDBOX.store.slots / "openings.json"):
            self.assertTrue(str(path).startswith(tempfile.gettempdir()), path)

    def test_the_tuned_numbers_are_not_module_globals(self):
        """The same reasoning as the directories, for the numbers.

        A test lowered POLL or PIN_PATIENCE by assignment, which reached the
        readers only while they shared this module. Three of them could not
        be reached at all, because they were default arguments that Python
        binds once at import: tests/live/test_router_live.py had to size a
        9,000 character system prompt around PREFIX_MIN_CHARS rather than
        lower it. A Tuning is passed in, and a frozen one cannot be edited by
        halves."""
        for name in ("POLL", "PIN_PATIENCE", "PARK_BUDGET", "BLOCK_BUDGET",
                     "HANDOFF_ON", "DEEP_OPENINGS", "PREFIX_MIN_CHARS",
                     "PING_EVERY", "MAX_BODY", "REPLY_TOKENS", "PARK_FLOOR"):
            self.assertFalse(
                hasattr(router, name),
                f"router.{name} is a module global again. A test can assign "
                f"it, and a reader that does not share this module will not "
                f"see the assignment.")

    def test_an_event_log_that_is_on_needs_a_directory(self):
        """The guard that replaces the CACHE_LOG preamble.

        This repository once lost 1,073 fixture rows into a live router's
        event log, and again 1,440 while this branch was being written: a
        commit changed EventLog's default to on=True while the module still
        built one at import, so the environment variable the preamble set no
        longer decided anything. An env var cannot guard a mechanism that
        moved. A log that is on and does not know where to write cannot."""
        with self.assertRaises(ValueError):
            router.EventLog()
        with self.assertRaises(ValueError):
            router.EventLog(on=True)

    def test_a_pool_writes_no_events_unless_it_is_given_a_log(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                         events=None, watch=False)
        self.assertFalse(pool.events.on)
        self.assertIsNone(pool.events.directory)

    def test_a_tuning_cannot_be_edited_by_halves(self):
        """Frozen on purpose. A half-changed Tuning is the global it replaced."""
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SANDBOX.tuning.poll = 0.01

    def test_the_directories_are_not_module_globals(self):
        """Store owns the directories, where no test can redirect them.

        While RUN_DIR, SLOT_DIR and BLOCK_DIR were module globals, a test
        redirected them by assignment. That works only while every reader
        lives in this one module. A reader in another module binds the name
        at import and never sees the redirect. The suite would have gone on
        passing while that reader wrote into the live run/slots."""
        for name in ("RUN_DIR", "SLOT_DIR", "BLOCK_DIR"):
            self.assertFalse(
                hasattr(router, name),
                f"router.{name} is a module global again. A test can redirect "
                f"it. A reader outside this module does not see the redirect, "
                f"and that is how a test comes to write into a live router's "
                f"slot directory.")


class SlotDirCase(unittest.TestCase):
    """A test case whose slot files land in a temporary directory.

    STORE is the default store a Pool takes when it is handed none.
    A class that leaves it alone reads and deletes inside the checkout's own
    run/slots. A running router keeps parked copies there. Several of them
    are symlinks into the block directory that Store.drop follows. A fixture
    named like a live file would take a real copy with it, and adopt() would
    read the live pins.json. setUp replaces STORE. The cleanup puts it
    back."""

    def setUp(self):
        super().setUp()
        self.slot_root = Path(tempfile.mkdtemp())
        was = SANDBOX.store
        SANDBOX.store = router.Store(self.slot_root)
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: setattr(SANDBOX, "store", was))
        self.addCleanup(shutil.rmtree, self.slot_root, True)

    def slot_dir(self):
        return SANDBOX.store.slots


class ParkIsReal(SlotDirCase):
    """A slot can change hands before the parker reaches it. The saved file
    then holds nothing, and must not be mistaken for a usable copy.

    Any real state for this model carries the recurrent state, which is a
    fixed ~112 MiB whatever the length. A small file means an empty slot."""

    def setUp(self):
        super().setUp()                    # a slot directory of its own
        self.pool = make_pool(
            [{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=1, n_ctx=150000)
        self.pool.pins["a"] = pin("cpu", slot=0, last=9.0)
        self.pool.pins["b"] = pin("cpu", slot=0, last=1.0)

    def park(self, written):
        return with_link(self.pool,
                         FakeLink(written=written))._save_park("b", self.cpu, 0)

    def test_keeps_a_copy_that_holds_real_state(self):
        self.assertTrue(self.park(120_000_000))
        self.assertEqual(self.pool.pins["b"]["parked"], "b.park")

    def test_refuses_a_copy_from_an_empty_slot(self):
        self.assertFalse(self.park(928))
        self.assertIsNone(self.pool.pins["b"]["parked"])


class ParkBeforeAdmitting(SlotDirCase):
    """Nothing reaches a backend until the caches already there are safe.

    The router is the only client, so it knows when a slot is about to be
    reused. Copying first is what makes the cache portable instead of lost."""

    def setUp(self):
        super().setUp()                    # a slot directory of its own
        self.pool = make_pool(
            [{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=1, n_ctx=150000)

    def saver(self, written=120_000_000):
        return FakeLink(written=written)

    def test_copies_the_resident_before_letting_a_newcomer_in(self):
        self.pool.pins["old"] = pin("cpu", slot=0, last=1.0)
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(self.pool.pins["old"]["parked"], "old.park")
        self.assertEqual(post.ops()[0], "save")

    def test_does_not_copy_the_conversation_being_admitted(self):
        self.pool.pins["new"] = pin("cpu", slot=0, last=1.0)
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(post.calls, [])

    def test_does_not_copy_again_what_is_already_on_disk(self):
        self.pool.pins["old"] = pin("cpu", slot=0, last=1.0, parked="old.park")
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(post.calls, [])

    def test_copies_again_a_slot_that_has_run_past_its_copy(self):
        """A copy from an earlier turn is a prefix, not the slot.

        Asking only whether a copy existed froze it at the first park: on a
        pool where every backend reads - the shipped default, and what
        generator() returns None for - nothing else writes one, so turn 10
        recalled turn 1 and read the nine turns between from cold."""
        self.pool.pins["old"] = pin("cpu", slot=0, last=1.0, turns=4,
                                    parked="old.park", parked_turn=1)
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(post.ops()[0], "save")
        self.assertEqual(self.pool.pins["old"]["parked_turn"], 4)

    def test_skips_a_conversation_that_is_working(self):
        self.pool.pins["busy"] = pin("cpu", slot=0, last=1.0, inflight=True)
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(post.calls, [])

    def test_stops_asking_when_the_cache_is_gone(self):
        """A slot that changed hands has nothing to give. Do not ask twice."""
        self.pool.pins["old"] = pin("cpu", slot=0, last=1.0)
        post = linked(self.pool, self.saver(written=900))
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(len(post.calls), 1)
        self.assertIsNone(self.pool.pins["old"]["slot"])

    def test_a_turn_keeps_the_copy_it_has_moved_past(self):
        """Behind is still a prefix, and a prefix is what a read starts from.

        The slot has gone past that copy, but a slot is borrowed: whatever
        takes it next erases the turn, and then the conversation has nothing
        at all and reads its whole prompt again. Keeping it costs one file the
        next save overwrites under the same name."""
        copy = self.slot_dir() / "old.park"
        copy.write_bytes(b"state")
        self.pool._take(self.cpu, "old", tokens=10)
        self.pool.pins["old"]["parked"] = "old.park"
        self.pool.release(self.cpu, "old")
        self.assertEqual(self.pool.pins["old"]["parked"], "old.park")
        self.assertTrue(copy.exists(), "the only copy on disk was deleted")

    def test_the_copy_is_not_restored_over_a_slot_that_still_holds_it(self):
        """Which is what made deleting it look necessary."""
        self.pool._take(self.cpu, "old", tokens=10)
        self.pool.pins["old"]["parked"] = "old.park"
        self.pool.pins["old"]["slot"] = 0
        self.pool.release(self.cpu, "old")
        self.assertFalse(
            with_link(self.pool, self.saver()).recall("old", self.cpu, 0))

    def test_a_turn_with_no_copy_deletes_nothing(self):
        stranger = self.slot_dir() / "someone-else.park"
        stranger.write_bytes(b"state")
        self.pool._take(self.cpu, "old", tokens=10)
        self.pool.release(self.cpu, "old")
        self.assertTrue(stranger.exists())


    def hold(self, name, size, last):
        record = pin("cpu", slot=0, last=last, parked=f"{name}.park")
        record["bytes"] = size
        self.pool.pins[name] = record

    def test_records_how_much_disk_a_copy_uses(self):
        self.pool.pins["old"] = pin("cpu", slot=0, last=1.0)
        with_link(self.pool,
                  self.saver(written=200_000_000)).ensure_parked(self.cpu, "new")
        self.assertEqual(self.pool.pins["old"]["bytes"], 200_000_000)

    def test_the_copy_just_written_is_the_one_that_stays(self):
        """Whatever it is worth, and whatever it cost.

        The copy written here is the larger of the two, so by worth alone it
        goes first and the budget drops it the moment it lands. The sweep
        counts it before anything else instead."""
        removed = []
        half = SANDBOX.tuning.park_budget // 2
        self.pool.pins["early"] = pin("cpu", slot=0, last=1.0, inflight=True)
        self.hold("later", half, last=2.0)
        with_link(self.pool, self.saver(written=half + 1))._save_park(
            "early", self.cpu, 0, remove=removed.append)
        self.assertEqual(self.pool.pins["early"]["parked"], "early.park")
        self.assertEqual(removed, ["later.park"])

    def test_a_copy_the_budget_drops_is_not_written_again(self):
        """One pass, one attempt each, whatever the budget does afterwards.

        cpu1_0 wrote the same 9.45 GiB copy 1,456 times over four and a half
        hours because the budget cleared the mark that says it is on disk."""
        self.pool.pins["stuck"] = pin("cpu", slot=0, last=1.0)
        self.hold("big", SANDBOX.tuning.park_budget, last=2.0)
        post = linked(self.pool, self.saver(written=200_000_000))
        self.pool.ensure_parked(self.cpu, "new")
        written = post.files("save")
        self.assertEqual(len(written), len(set(written)))
        self.assertEqual(written.count("stuck.park"), 1)

    def test_keeps_the_copies_that_fit_the_budget(self):
        """Two copies that earn the same are separated by nothing else, so
        the older one goes. The budget still has to stop somewhere."""
        removed = []
        half = SANDBOX.tuning.park_budget // 2
        self.hold("old", half, last=1.0)
        self.hold("mid", half, last=2.0)
        self.pool.pins["new"] = pin("cpu", slot=0, last=99.0, inflight=True)
        with_link(self.pool, self.saver(written=half))._save_park(
            "new", self.cpu, 0, remove=removed.append)
        self.assertEqual(removed, ["old.park"])
        self.assertIsNone(self.pool.pins["old"]["parked"])
        self.assertEqual(self.pool.pins["mid"]["parked"], "mid.park")

    def test_keeps_the_newest_even_when_it_spends_the_budget_alone(self):
        removed = []
        self.pool.pins["new"] = pin("cpu", slot=0, last=99.0, inflight=True)
        with_link(self.pool,
                  self.saver(written=SANDBOX.tuning.park_budget * 2))._save_park(
            "new", self.cpu, 0, remove=removed.append)
        self.assertEqual(removed, [])
        self.assertEqual(self.pool.pins["new"]["parked"], "new.park")

    def test_the_copy_nobody_has_come_back_to_goes_first(self):
        """Size does not decide it. A big copy of a conversation nobody has
        touched for days goes before a small one used a minute ago.

        Measured on 131 real copies at a 64 GiB budget: ordering on the
        tokens a copy holds against its bytes kept 19 of them and only 7 of
        the 16 conversations in use that week. Ordering on when each was last
        used kept 18, and all 16."""
        removed = []
        half = SANDBOX.tuning.park_budget // 2
        now = time.time()
        # The stale one is the SMALL one, so that any rule reading size keeps
        # it and drops the large copy somebody is still working in.
        self.hold("small_and_stale", half // 8, last=now - 3 * 86400)
        self.hold("big_and_fresh", half, last=now - 60)
        self.pool.pins["new"] = pin("cpu", slot=0, last=now, inflight=True)
        linked(self.pool, self.saver(written=half))
        self.pool._save_park("new", self.cpu, 0, remove=removed.append)
        self.assertEqual(removed, ["small_and_stale.park"])
        self.assertEqual(self.pool.pins["big_and_fresh"]["parked"],
                         "big_and_fresh.park")

    def test_a_copy_is_ranked_by_when_its_conversation_last_ran(self):
        self.assertGreater(router.last_used({"last": 200.0}),
                           router.last_used({"last": 100.0}))
        self.assertEqual(router.last_used({}), 0,
                         "a pin with no last turn sorts at the back")

    def test_a_copy_the_router_knows_nothing_about_sorts_at_the_back(self):
        """A pin adopted from a file an older run wrote carries no last turn,
        so it goes before anything this run has served."""
        self.assertEqual(router.last_used({}), 0)
        self.assertEqual(router.last_used({"last": None}), 0)

    def test_a_short_prompt_is_read_again_rather_than_copied(self):
        """A copy costs the same few hundred megabytes whatever it holds.
        Measured here, a recall under 1,024 tokens carried 36 of them."""
        short = pin("cpu", slot=0, last=1.0,
                    tokens=SANDBOX.tuning.park_min_tokens - 1
                    + SANDBOX.tuning.reply_tokens)
        self.pool.pins["short"] = short
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(post.calls, [], "a short prompt was copied to disk")
        self.assertFalse(short["inflight"], "the turn was left reserved")

    def test_a_long_prompt_still_is_copied(self):
        self.pool.pins["long"] = pin(
            "cpu", slot=0, last=1.0,
            tokens=SANDBOX.tuning.park_min_tokens
            + SANDBOX.tuning.reply_tokens)
        post = linked(self.pool, self.saver())
        self.pool.ensure_parked(self.cpu, "new")
        self.assertEqual(post.ops()[0], "save")

    def test_the_floor_is_read_in_prompt_tokens_not_the_estimate(self):
        """`tokens` on a pin carries REPLY_TOKENS of room for the reply. A
        floor spent against that number would be REPLY_TOKENS lower than it
        reads."""
        at_the_floor = {"tokens": SANDBOX.tuning.park_min_tokens
                        + SANDBOX.tuning.reply_tokens}
        self.assertTrue(router.worth_keeping(at_the_floor, SANDBOX.tuning))
        self.assertFalse(router.worth_keeping(
            {"tokens": at_the_floor["tokens"] - 1}, SANDBOX.tuning))

    def test_nothing_is_refused_when_the_floor_is_off(self):
        off = replace(SANDBOX.tuning, park_min_tokens=0)
        self.assertTrue(router.worth_keeping({"tokens": 0}, off))


class Recall(unittest.TestCase):
    """Bring a parked cache back on whichever backend has room."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0},
             {"name": "cpu", "url": "http://cpu", "pref": 1}], watch=False)
        self.gpu, self.cpu = self.pool.backends
        self.gpu.update(up=True, slots=1, n_ctx=150000)
        self.cpu.update(up=True, slots=3, n_ctx=150000,
                        slots_detail=[{"id": 0, "busy": True},
                                      {"id": 1, "busy": False},
                                      {"id": 2, "busy": False}])
        self.pool.pins["conv1"] = pin("gpu", slot=0, last=5.0, parked="conv1.park")

    def test_restores_the_parked_cache_on_the_new_backend(self):
        post = linked(self.pool, FakeLink())
        self.assertTrue(self.pool.recall("conv1", self.cpu, 1))
        op, name, slot, filename = post.calls[0]
        self.assertEqual((op, name, slot), ("restore", "cpu", 1))
        self.assertEqual(filename, "conv1.park")

    def test_repins_the_conversation_where_it_landed(self):
        with_link(self.pool, FakeLink()).recall("conv1", self.cpu, 1)
        record = self.pool.pins["conv1"]
        self.assertEqual(record["backend"], "cpu")
        self.assertEqual(record["slot"], 1)

    def test_does_nothing_when_the_backend_has_not_changed(self):
        post = linked(self.pool, FakeLink())
        self.assertFalse(self.pool.recall("conv1", self.gpu, 1))
        self.assertEqual(post.calls, [])

    def test_does_nothing_without_a_parked_copy(self):
        self.pool.pins["conv1"]["parked"] = None
        post = linked(self.pool, FakeLink())
        self.assertFalse(self.pool.recall("conv1", self.cpu, 1))
        self.assertEqual(post.calls, [])

    def test_a_failed_restore_leaves_the_pin_alone(self):
        with_link(self.pool,
                  FakeLink(fail_on="restore")).recall("conv1", self.cpu, 1)
        self.assertEqual(self.pool.pins["conv1"]["backend"], "gpu")


LONG = "You are a careful assistant. " * 400      # a client-sized system prompt


class TextOf(unittest.TestCase):
    """A system prompt arrives as a string or as a list of parts."""

    def test_reads_a_string(self):
        self.assertEqual(router.text_of("rules"), "rules")

    def test_joins_a_list_of_parts(self):
        self.assertEqual(router.text_of([{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]), "ab")

    def test_reads_nothing_from_anything_else(self):
        self.assertEqual(router.text_of(None), "")
        self.assertEqual(router.text_of(7), "")


class CommonPrefix(unittest.TestCase):
    """The text two template renderings share is the system block."""

    def test_returns_the_shared_head(self):
        self.assertEqual(router.common_prefix("abcX", "abcY"), "abc")

    def test_returns_all_of_a_string_the_other_starts_with(self):
        self.assertEqual(router.common_prefix("abc", "abcdef"), "abc")

    def test_returns_nothing_when_they_differ_at_once(self):
        self.assertEqual(router.common_prefix("abc", "xyz"), "")


class PrefixCase:
    """One backend with three slots, and a slot directory under /tmp.

    Slot 0 is working. The poll and the router's own busy count disagree
    about it, which is the state this code has to survive."""

    BLOCK = "<|im_start|>system\nrules<|im_end|>\n<|im_start|>"
    TALK = [{"role": "system", "content": "rules"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"}]
    CUTS = [(0, "k1"), (1, "k2"), (2, "k3")]

    def setUp(self):
        root = Path(tempfile.mkdtemp())
        (root / "slots").mkdir()
        self.was = SANDBOX.store
        SANDBOX.store = router.Store(root)
        self.addCleanup(shutil.rmtree, root)
        self.addCleanup(self.put_back)
        self.pool = make_pool(
            [{"name": "cpu", "url": "http://cpu", "pref": 0}],
            store=SANDBOX.store, watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=3, n_ctx=150000,
                        slots_detail=[{"id": 0, "busy": True},
                                      {"id": 1, "busy": False},
                                      {"id": 2, "busy": False}])
        self.pool.pins["new"] = pin("cpu", slot=None, inflight=True)

    def put_back(self):
        SANDBOX.store = self.was

    def talker(self, written=700_000_000, fail_on=None):
        block = self.BLOCK

        class Talk(FakeLink):
            def render(inner, be, route, payload, timeout=None):
                inner._note("render", be, route)
                # The longer rendering carries the extra user message, so the
                # two differ exactly where the opening ends.
                last = payload["messages"][-1]
                return {"prompt": block + ("user\nx"
                                           if last["content"] == "x"
                                           else "assistant\n")}

            def prefill(inner, be, blk, slot, timeout=None):
                inner._note("prefill", be, slot, blk)
                return {"tokens_evaluated": 2048}

            def save(inner, be, slot, name, timeout=None):
                inner._note("save", be, slot, name)
                return {"n_saved": 1, "n_written": written}

        return Talk(fail_on=fail_on)

    def warm(self, link, conv="new", cuts=None, system="", slot=1):
        """Slot 1 is the free one. The request path decides it once and hands
        it to everything that puts something in a slot."""
        self.pool.link = link
        return self.pool.warm_prefix(
            conv, self.CUTS[:1] if cuts is None else cuts, self.TALK,
            system, [], self.cpu, slot, "/v1/chat/completions")

    def paths(self, post):
        return post.ops()


class WarmPrefix(PrefixCase, unittest.TestCase):
    """What a request does about an opening.

    It loads one the router already has, which costs a file read. Where the
    opening it needs is most of its own prompt and nobody has it, it reads it
    and saves it: those are tokens this request was going to read anyway, so
    it pays only the save, and every session that starts behind it loads the
    result instead of reading the same tokens again."""

    def test_loads_the_deepest_saved_opening(self):
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.openings["k2"] = "deep-k2.park"
        post = linked(self.pool, self.talker())
        self.assertTrue(self.warm(post, cuts=self.CUTS))
        self.assertEqual(self.paths(post), ["restore"])
        self.assertEqual(post.files()[0], "deep-k2.park")

    def test_reads_nothing_when_the_opening_is_already_on_the_shelf(self):
        self.pool.openings["k1"] = "base-k1.park"
        post = linked(self.pool, self.talker())
        self.assertTrue(self.warm(post))
        self.assertEqual(self.paths(post), ["restore"])

    def test_a_load_is_written_down_where_the_next_run_will_see_it(self):
        """In memory it is lost on the restart that needs it most. Nothing
        else writes this file often enough: a build happens once per opening,
        so without a write here the count and the order are a week stale."""
        self.pool.openings["k1"] = "base-k1.park"
        self.assertTrue(self.warm(self.talker()))
        self.assertEqual(SANDBOX.store.read_openings(),
                         [{"key": "k1", "file": "base-k1.park", "loads": 1}])

    def test_reads_the_system_prompt_nobody_has_and_saves_it(self):
        """It is the front of this request's own prompt either way, so the
        only cost is the save, and the next session loads it."""
        post = linked(self.pool, self.talker())
        self.assertTrue(self.warm(post, system="rules"))
        self.assertEqual(self.paths(post),
                         ["render", "render", "prefill", "save"])
        self.assertIn("k1", self.pool.openings)

    def test_a_busy_machine_still_gets_the_opening_saved(self):
        """The request reads the opening on its own slot, so a machine busy
        elsewhere does not stop it."""
        for slot in self.cpu["slots_detail"]:
            slot["busy"] = True
        self.cpu["busy"] = 3
        post = linked(self.pool, self.talker())
        self.assertTrue(self.warm(post, system="rules"))
        self.assertIn("k1", self.pool.openings)

    def test_it_measures_the_deeper_cut_and_reads_only_the_opening(self):
        """A deeper cut was built 0 times and loaded 0 times in 51 hours of
        real traffic. It is still found, because it is what the choice event
        measures the fork question with, but nothing reads it."""
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.holds[("cpu", 0)] = {"k1", "k2"}
        post = linked(self.pool, self.talker())
        self.warm(post, cuts=self.CUTS)
        self.assertEqual(self.paths(post), ["restore"])
        self.assertEqual(self.pool.choices["new"]["shared"], 1,
                         "the fork is still seen, only not acted on")

    def test_it_measures_a_fork_against_a_parked_parent(self):
        """The one a slot cannot see. `shared` needs the parent resident, and
        this box has four slots against forty-four copies - so without this
        the logged fork rate is a floor of zero that means nothing."""
        self.pool.pins["parent"] = pin("cpu", slot=None, parked="parent.park")
        self.pool.pins["parent"]["holds"] = {"k1", "k2", "k3"}
        self.warm(self.talker(), cuts=self.CUTS)
        self.assertIsNone(self.pool.choices["new"]["shared"],
                          "no slot holds it, which is the point")
        self.assertEqual(self.pool.choices["new"]["copied"], 2)

    def test_a_conversation_is_not_measured_against_its_own_copy(self):
        self.pool.pins["new"]["parked"] = None
        self.pool.pins["new"]["holds"] = {"k1", "k2", "k3"}
        self.warm(self.talker(), cuts=self.CUTS)
        self.assertIsNone(self.pool.choices["new"]["copied"])

    def test_does_nothing_without_a_cut(self):
        post = linked(self.pool, self.talker())
        self.assertFalse(self.warm(post, cuts=[]))
        self.assertEqual(post.calls, [])

    def test_leaves_a_conversation_that_has_its_own_cache(self):
        self.pool.pins["new"]["parked"] = "new.park"
        post = linked(self.pool, self.talker())
        self.assertFalse(self.warm(post))
        self.assertEqual(post.calls, [])

    def test_leaves_a_conversation_that_already_holds_a_slot(self):
        self.pool.pins["new"]["slot"] = 2
        post = linked(self.pool, self.talker())
        self.assertFalse(self.warm(post))
        self.assertEqual(post.calls, [])

    def test_a_failed_load_is_not_fatal(self):
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.openings["k2"] = "deep-k2.park"
        self.assertFalse(self.warm(self.talker(fail_on="restore"),
                                   cuts=self.CUTS))

    def test_room_for_several_clients_and_projects(self):
        """Claude Code and OpenCode differ, and so does each project.

        Measured here: a system prompt block runs 0.58 to 3.68 GB, so the
        budget has to hold several of the big ones."""
        self.assertGreaterEqual(SANDBOX.tuning.block_budget, 4 * 4 * 1024 ** 3)

    def test_an_opening_in_use_is_not_the_next_one_dropped(self):
        """Loading an opening moves it to the end of the shelf, so the budget
        drops the one nobody has asked for rather than the one every session
        is starting from.

        The opening loaded has to be the one at the FRONT. Loading the one
        already last asserted the order it had been given, and passed with
        _load_prefix's move_to_end deleted."""
        self.pool.openings["k0"] = "base-k0.park"
        self.pool.openings["k1"] = "base-k1.park"
        self.warm(self.talker(), cuts=[(0, "k0")])
        self.assertEqual(list(self.pool.openings), ["k1", "k0"])


class SlotHistory(unittest.TestCase):
    """What a slot holds, so a later request can start from it."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        self.cpu = self.pool.backends[0]

    def test_records_the_cuts_a_slot_now_holds(self):
        self.pool.pins["a"] = pin("cpu", slot=2)
        self.pool.note_holds("a", self.cpu, [(0, "k1"), (1, "k2")])
        self.assertEqual(self.pool.holds[("cpu", 2)], {"k1", "k2"})

    def test_records_nothing_for_a_slot_it_does_not_know(self):
        self.pool.pins["a"] = pin("cpu", slot=None)
        self.pool.note_holds("a", self.cpu, [(0, "k1")])
        self.assertEqual(self.pool.holds, {})

    def test_a_later_request_replaces_what_the_slot_held(self):
        self.pool.pins["a"] = pin("cpu", slot=2)
        self.pool.note_holds("a", self.cpu, [(0, "k1")])
        self.pool.note_holds("a", self.cpu, [(0, "k9")])
        self.assertEqual(self.pool.holds[("cpu", 2)], {"k9"})


class AdoptFiles(unittest.TestCase):
    """Sort out what the last run left in the slot directory."""

    def sized(self, names, each=700_000_000):
        return router.adopt_files(names, size=lambda name: each,
                                  tuning=SANDBOX.tuning)

    def test_keeps_an_opening_for_a_system_prompt(self):
        openings, _, _, spent = self.sized(["base-k1.park"])
        self.assertEqual(openings, OrderedDict(k1="base-k1.park"))
        self.assertEqual(spent, [])

    def test_a_deeper_opening_is_given_back_to_the_disk(self):
        """Nothing reads one, so keeping them is 8 GB of a 92% full nvme held
        by four files that have never been loaded once."""
        openings, _, _, spent = self.sized(["base-k1.park", "deep-k2.park"])
        self.assertEqual(openings, OrderedDict(k1="base-k1.park"))
        self.assertEqual(spent, ["deep-k2.park"])

    def test_drops_a_copy_the_pin_file_does_not_vouch_for(self):
        openings, _, _, spent = self.sized(["abc123.park"])
        self.assertEqual(openings, OrderedDict())
        self.assertEqual(spent, ["abc123.park"])

    def test_it_measures_each_file_it_keeps(self):
        _, sizes, _, _ = self.sized(["base-k1.park", "base-k2.park"])
        self.assertEqual(sizes, {"k1": 700_000_000, "k2": 700_000_000})

    def test_what_will_not_fit_the_budget_does_not_come_back(self):
        """The budget outlives the run that wrote the files, so a restart
        under a smaller one drops what no longer fits - deeper cuts first."""
        names = ["base-b0.park", "deep-d0.park", "base-b1.park"]
        was = SANDBOX.tuning
        # room for two of the three
        SANDBOX.tuning = replace(was, block_budget=1_500_000_000)
        self.addCleanup(setattr, SANDBOX, "tuning", was)
        openings, sizes, _, spent = self.sized(names)
        self.assertEqual(spent, ["deep-d0.park"])
        self.assertEqual(list(openings), ["b0", "b1"])
        self.assertEqual(set(sizes), {"b0", "b1"})

    def test_the_pool_takes_them_over(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           watch=False)
        removed = []
        pool.adopt(["base-k1.park", "base-k2.park", "old.park"],
                   remove=removed.append)
        self.assertEqual(pool.openings,
                         OrderedDict(k1="base-k1.park", k2="base-k2.park"))
        self.assertEqual(removed, ["old.park"])


class BlockOnTheFasterDisk(unittest.TestCase):
    """A backend takes a bare filename under its own slot directory, so a link
    is the only way to keep one file on another disk."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.slots, self.blocks = self.root / "slots", self.root / "blocks"
        self.slots.mkdir()
        self.was = SANDBOX.store
        SANDBOX.store = router.Store(self.root)
        self.addCleanup(shutil.rmtree, self.root)
        self.addCleanup(self.put_back)

    def put_back(self):
        SANDBOX.store = self.was

    def test_the_slot_directory_points_at_the_other_disk(self):
        SANDBOX.store.link_block("base-k1.park")
        link = self.slots / "base-k1.park"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.readlink(), self.blocks / "base-k1.park")

    def test_a_write_through_the_link_lands_on_the_other_disk(self):
        SANDBOX.store.link_block("base-k1.park")
        (self.slots / "base-k1.park").write_bytes(b"state")
        self.assertEqual((self.blocks / "base-k1.park").read_bytes(), b"state")

    def test_linking_twice_is_harmless(self):
        SANDBOX.store.link_block("base-k1.park")
        SANDBOX.store.link_block("base-k1.park")
        self.assertTrue((self.slots / "base-k1.park").is_symlink())

    def test_dropping_a_block_removes_both_ends(self):
        SANDBOX.store.link_block("base-k1.park")
        (self.slots / "base-k1.park").write_bytes(b"state")
        SANDBOX.store.drop("base-k1.park")
        self.assertFalse((self.slots / "base-k1.park").is_symlink())
        self.assertFalse((self.blocks / "base-k1.park").exists())

    def test_dropping_a_conversation_copy_removes_only_it(self):
        (self.slots / "abc.park").write_bytes(b"state")
        SANDBOX.store.drop("abc.park")
        self.assertFalse((self.slots / "abc.park").exists())

    def test_a_link_with_nothing_behind_it_is_dropped(self):
        SANDBOX.store.link_block("base-k1.park")
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           watch=False)
        pool.adopt()
        self.assertEqual(pool.openings, OrderedDict())
        self.assertFalse((self.slots / "base-k1.park").is_symlink())


class PromptCuts(unittest.TestCase):
    """Every point in a request that another request could share.

    A restored slot cannot rewind, so a block has to be cut where messages end.
    Hashing each message onto the one before names every cut, and two requests
    that open the same way share the names of every cut in the opening."""

    def body(self, *messages, system=None):
        payload = {"messages": list(messages)}
        if system is not None:
            payload["system"] = system
        return json.dumps(payload).encode()

    def keys(self, *messages, **kw):
        cuts, _, _, _ = router.prompt_cuts(self.body(*messages, **kw))
        return [key for _, key in cuts]

    def tool_result(self, content):
        """One agentic turn as the router's own client sends it: the words
        sit under the block's own `content`, not at the top level."""
        return {"role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1",
                             "content": content}]}

    def test_an_agentic_conversation_names_its_cuts(self):
        """Every turn of the router's own client is a tool_result. Measured by
        the top level `text` alone each one counted zero, so no message ever
        reached the bar and a conversation of any length named no cut."""
        got = self.keys({"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        self.tool_result([{"type": "text", "text": LONG}]),
                        {"role": "assistant", "content": "ok"},
                        self.tool_result([{"type": "text", "text": LONG}]))
        self.assertEqual(len(got), 3, "no cut was named in a 24,000 "
                                      "character conversation")

    def test_the_call_an_assistant_makes_is_measured_too(self):
        """A tool_use carries the arguments under `input`, and the template
        writes them into the prompt. Measured by two key names it counted
        zero, so the assistant half of every agentic turn was invisible.

        The cut lands on the message that answers the call, not on the call:
        a template cannot end a prompt on one. What matters is that it lands
        at all."""
        call = {"role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Write",
                             "input": {"file_path": "/a.py", "content": LONG}}]}
        got = self.keys({"role": "user", "content": "hello"}, call,
                        {"role": "user", "content": "go on"})
        self.assertEqual(len(got), 1, "an assistant turn of 12,000 "
                                      "characters named no cut")

    def test_a_base64_image_is_not_counted_as_words(self):
        """It is hundreds of times longer than what the vision encoder
        charges, which is why request_cost leaves it out as well."""
        shot = [{"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": "A" * 40000}}]
        self.assertLess(router.content_size(shot), 200)

    def test_a_tool_result_that_carries_its_words_directly_counts_too(self):
        """The same block with a string where the list would be."""
        got = self.keys({"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        self.tool_result(LONG))
        self.assertEqual(len(got), 1)

    def test_names_a_cut_after_each_message(self):
        cuts = self.keys({"role": "system", "content": LONG},
                         {"role": "user", "content": LONG})
        self.assertEqual(len(cuts), 2)
        self.assertNotEqual(cuts[0], cuts[1])

    def test_two_requests_that_open_alike_share_their_cuts(self):
        head = [{"role": "system", "content": LONG},
                {"role": "user", "content": LONG}]
        one = self.keys(*head, {"role": "assistant", "content": "first"})
        two = self.keys(*head, {"role": "assistant", "content": "second"})
        self.assertEqual(one[:2], two[:2])
        self.assertNotEqual(one[2], two[2])

    def test_a_deeper_cut_comes_last(self):
        cuts, messages, _, _ = router.prompt_cuts(
            self.body({"role": "system", "content": LONG},
                      {"role": "user", "content": LONG}))
        self.assertEqual([index for index, _ in cuts], [0, 1])
        self.assertEqual(len(messages), 2)

    def test_ignores_an_opening_too_short_to_keep(self):
        self.assertEqual(self.keys({"role": "system", "content": "be brief"},
                                   {"role": "user", "content": "hi"}), [])

    def test_names_the_system_field_on_its_own(self):
        cuts, _, system, _ = router.prompt_cuts(
            self.body({"role": "user", "content": "hi"}, system=LONG))
        self.assertEqual(cuts[0][0], -1)      # before any message
        self.assertEqual(system, LONG)

    def test_ignores_a_body_that_is_not_a_request(self):
        self.assertEqual(router.prompt_cuts(b"not json"), ([], [], "", []))


class DeepestShared(unittest.TestCase):
    """Longest prefix match is one lookup per cut, deepest first."""

    CUTS = [(0, "a"), (1, "b"), (2, "c")]

    def test_takes_the_deepest_one_that_is_known(self):
        self.assertEqual(router.deepest_shared(self.CUTS, {"a", "b"}), (1, "b"))

    def test_takes_nothing_when_none_is_known(self):
        self.assertIsNone(router.deepest_shared(self.CUTS, {"z"}))

    def test_takes_nothing_from_an_empty_request(self):
        self.assertIsNone(router.deepest_shared([], {"a"}))


class ShutDownCleanly(unittest.TestCase):
    """Park every live conversation before the backends go away.

    A cache only exists in a slot. Stopping a backend throws it away, and the
    conversation reads its whole prompt again when it comes back."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "slots").mkdir()      # where a park file goes
        self.was = SANDBOX.store
        SANDBOX.store = router.Store(self.root)
        self.addCleanup(shutil.rmtree, self.root)
        self.addCleanup(self.put_back)
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0},
             {"name": "cpu", "url": "http://cpu", "pref": 1}],
            store=SANDBOX.store, watch=False)
        self.gpu, self.cpu = self.pool.backends
        self.gpu.update(up=True, slots=1, n_ctx=150000)
        self.cpu.update(up=True, slots=3, n_ctx=150000)

    def put_back(self):
        SANDBOX.store = self.was

    def saver(self, written=200_000_000):
        return FakeLink(written=written)

    def test_parks_every_live_conversation(self):
        self.pool.pins["a"] = pin("gpu", slot=0)
        self.pool.pins["b"] = pin("cpu", slot=2)
        post = linked(self.pool, self.saver())
        self.assertEqual(self.pool.park_all(), 2)
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")
        self.assertEqual(self.pool.pins["b"]["parked"], "b.park")

    def test_leaves_a_conversation_whose_slot_is_unknown(self):
        self.pool.pins["a"] = pin("gpu", slot=None)
        post = linked(self.pool, self.saver())
        self.assertEqual(self.pool.park_all(), 0)
        self.assertEqual(post.calls, [])

    def test_leaves_a_conversation_that_is_still_working(self):
        """Its slot is busy, so the save would wait for a turn we are ending."""
        self.pool.pins["a"] = pin("gpu", slot=0, inflight=True)
        post = linked(self.pool, self.saver())
        self.assertEqual(self.pool.park_all(), 0)
        self.assertEqual(post.calls, [])

    def test_writes_the_pins_beside_the_copies(self):
        self.pool.pins["a"] = pin("cpu", slot=1, tokens=4321)
        self.pool.holds[("cpu", 1)] = {"k1"}
        with_link(self.pool, self.saver()).park_all()
        self.pool.save_pins()
        kept = SANDBOX.store.read_pins()
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["conv"], "a")
        self.assertEqual(kept[0]["file"], "a.park")
        self.assertEqual(kept[0]["tokens"], 4321)

    def test_adopting_more_copies_than_the_budget_holds_trims_them(self):
        """The budget was spent only where a copy is written, so lowering it
        did nothing until the next turn parked - and on a quiet router that
        is never. Seen live: 251 GiB of copies under a 64 GiB budget, one
        conversation in a slot and nothing writing.

        Least recently used goes, as it does after a park."""
        removed = []
        third = SANDBOX.tuning.park_budget // 3
        rows = []
        for name, last in (("stale", 100.0), ("older", 200.0), ("newest", 300.0)):
            (SANDBOX.store.slots / f"{name}.park").write_bytes(b"x")
            rows.append({"conv": name, "file": f"{name}.park", "tokens": 9999,
                         "bytes": third + 1, "turns": 1, "last": last,
                         "parked_at": last})
        SANDBOX.store.write_pins(rows)

        fresh = make_pool([{"name": "cpu", "url": "http://cpu"}], watch=False)
        fresh.adopt([f"{n}.park" for n in ("stale", "older", "newest")],
                    remove=removed.append)
        self.assertEqual(removed, ["stale.park"],
                         "three copies of a third of the budget each, plus "
                         "one byte, do not fit")
        self.assertIsNone(fresh.pins["stale"]["parked"])
        self.assertEqual(fresh.pins["newest"]["parked"], "newest.park")

    def test_the_map_stops_vouching_for_what_the_trim_deleted(self):
        """Left alone, pins.json still named the copies this start had just
        removed, until whenever a turn next happened to park. A start that
        kept 19 of 131 left 112 rows pointing at nothing."""
        removed = []
        third = SANDBOX.tuning.park_budget // 3
        rows = []
        for name, last in (("stale", 100.0), ("older", 200.0), ("newest", 300.0)):
            (SANDBOX.store.slots / f"{name}.park").write_bytes(b"x")
            rows.append({"conv": name, "file": f"{name}.park", "tokens": 9999,
                         "bytes": third + 1, "turns": 1, "last": last,
                         "parked_at": last})
        SANDBOX.store.write_pins(rows)

        fresh = make_pool([{"name": "cpu", "url": "http://cpu"}], watch=False)
        fresh.adopt([f"{n}.park" for n in ("stale", "older", "newest")],
                    remove=removed.append)
        kept = {row["conv"] for row in SANDBOX.store.read_pins()}
        self.assertEqual(kept, {"older", "newest"})

    def test_when_a_conversation_last_ran_survives_the_restart(self):
        """The budget sweep drops whatever has gone longest without a turn,
        so that time has to outlive the process. Restored as `now` instead,
        every copy read as freshly used and the sweep had nothing to sort
        by on the first pass after a restart."""
        then = time.time() - 5 * 86400
        self.pool.pins["a"] = pin("cpu", slot=1, tokens=4321, last=then)
        linked(self.pool, self.saver())
        self.pool.park_all()
        self.pool.save_pins()
        self.assertAlmostEqual(SANDBOX.store.read_pins()[0]["last"], then,
                               places=3)

        fresh = make_pool([{"name": "cpu", "url": "http://cpu"}], watch=False)
        fresh.adopt(["a.park"])
        self.assertAlmostEqual(fresh.pins["a"]["last"], then, places=3,
                               msg="the copy came back looking newly used")

    def test_writes_nothing_for_a_conversation_with_no_copy(self):
        self.pool.pins["a"] = pin("cpu", slot=1, parked=None)
        self.pool.save_pins()
        self.assertEqual(SANDBOX.store.read_pins(), [])

    def test_reads_nothing_when_there_is_no_pin_file(self):
        self.assertEqual(router.Store(self.root / "empty").read_pins(), [])

    def test_reads_nothing_from_a_damaged_pin_file(self):
        """The branch that matters most. adopt() half believes a damaged pin
        file, and then deletes every copy the file does not name."""
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        (SANDBOX.store.slots / "pins.json").write_bytes(b"not json")
        self.assertEqual(SANDBOX.store.read_pins(), [])


class ComeBackAfterRestart(unittest.TestCase):
    """A copy the pin file vouches for is worth keeping."""

    def test_a_vouched_copy_is_kept(self):
        _, _, parked, spent = router.adopt_files(
            ["a.park"], vouched={"a.park"}, store=SANDBOX.store,
            tuning=SANDBOX.tuning)
        self.assertEqual(parked, ["a.park"])
        self.assertEqual(spent, [])

    def test_a_copy_nothing_vouches_for_is_dropped(self):
        _, _, parked, spent = router.adopt_files(["a.park"],
                                                 store=SANDBOX.store,
                                                 tuning=SANDBOX.tuning)
        self.assertEqual((parked, spent), ([], ["a.park"]))

    def test_a_vouched_copy_that_is_gone_cannot_be_kept(self):
        _, _, parked, spent = router.adopt_files(
            [], vouched={"a.park"}, store=SANDBOX.store,
            tuning=SANDBOX.tuning)
        self.assertEqual(parked, [])

    def test_the_pool_takes_back_its_pins(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        was = SANDBOX.store
        SANDBOX.store = router.Store(root)
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: setattr(SANDBOX, "store", was))
        (SANDBOX.store.slots / "pins.json").write_text(json.dumps(
            [{"conv": "a", "file": "a.park", "tokens": 99, "cuts": [[0, "k1"]]}]))
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           store=SANDBOX.store, watch=False)
        pool.adopt(["a.park"], remove=lambda name: None)
        self.assertEqual(pool.pins["a"]["parked"], "a.park")
        self.assertEqual(pool.pins["a"]["tokens"], 99)
        self.assertIsNone(pool.pins["a"]["slot"])

    def test_an_opening_keeps_what_it_earned(self):
        """A restart used to forget which openings were pulling their weight.

        The mtime is when the block was built and never changes, so without
        this the order after a restart is build order: one loaded every day
        for a week sorts ahead of one nobody has asked for, and the budget
        drops it first. The loads count the dashboard shows went to zero with
        it, while the openings it counted stayed on disk."""
        pool = self.restarted(
            openings=[{"key": "old", "file": "base-old.park", "loads": 0},
                      {"key": "busy", "file": "base-busy.park", "loads": 41}],
            names=["base-busy.park", "base-old.park"])   # built the other way
        self.assertEqual(list(pool.openings), ["old", "busy"])
        self.assertEqual(pool.loads["busy"], 41)

    def test_an_opening_it_was_never_told_about_keeps_its_place(self):
        """Last, by the mtime it came in with: nothing says it is unused, so
        dropping it before an opening known to be idle would be a guess."""
        pool = self.restarted(
            openings=[{"key": "known", "file": "base-known.park", "loads": 3}],
            names=["base-known.park", "base-new.park"])
        self.assertEqual(list(pool.openings), ["known", "new"])

    def test_a_shelf_file_it_cannot_parse_is_simply_ignored(self):
        pool = self.restarted(openings="not a list",
                              names=["base-a.park", "base-b.park"])
        self.assertEqual(list(pool.openings), ["a", "b"])
        self.assertEqual(pool.loads, {})

    def restarted(self, openings, names):
        """A pool adopting `names` with `openings` left by the last run."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        was = SANDBOX.store
        SANDBOX.store = router.Store(root)
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: setattr(SANDBOX, "store", was))
        (SANDBOX.store.slots / "openings.json").write_text(json.dumps(openings))
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           store=SANDBOX.store, watch=False)
        pool.adopt(list(names), remove=lambda name: None)
        return pool

    def test_it_writes_down_what_the_openings_earned(self):
        pool = self.restarted(openings=[], names=["base-a.park"])
        pool.loads["a"] = 7
        pool.save_openings()
        self.assertEqual(
            SANDBOX.store.read_openings(),
            [{"key": "a", "file": "base-a.park", "loads": 7}])

    def test_a_recovered_pin_names_no_live_backend(self):
        """Its slot is gone, so it must be restored before it is served."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        was = SANDBOX.store
        SANDBOX.store = router.Store(root)
        SANDBOX.store.slots.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: setattr(SANDBOX, "store", was))
        (SANDBOX.store.slots / "pins.json").write_text(json.dumps(
            [{"conv": "a", "file": "a.park", "tokens": 99, "cuts": []}]))
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           store=SANDBOX.store, watch=False)
        pool.adopt(["a.park"], remove=lambda name: None)
        self.assertNotEqual(pool.pins["a"]["backend"], "cpu")


class DrainABackend(unittest.TestCase):
    """Take a backend out of service without dropping anything.

    Requests wait in acquire rather than failing, so a backend can be stopped
    and started under them. What must not be lost is the caches in its slots."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.was = SANDBOX.store
        SANDBOX.store = router.Store(self.root)
        self.addCleanup(shutil.rmtree, self.root)
        self.addCleanup(lambda: setattr(SANDBOX, "store", self.was))
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0},
             {"name": "cpu", "url": "http://cpu", "pref": 1}],
            store=SANDBOX.store, watch=False)
        self.gpu, self.cpu = self.pool.backends
        self.gpu.update(up=True, slots=1, n_ctx=150000)
        self.cpu.update(up=True, slots=3, n_ctx=150000)

    def saver(self, written=200_000_000):
        return FakeLink(written=written)

    def test_a_draining_backend_takes_no_new_work(self):
        self.pool.drain("gpu")
        self.assertFalse(self.pool._usable(self.gpu, 1000))
        self.assertTrue(self.pool._usable(self.cpu, 1000))

    def test_work_goes_to_the_other_backend_while_it_drains(self):
        self.pool.drain("gpu")
        self.assertEqual(self.pool.acquire("a", 1000)[0]['name'], "cpu")

    def test_draining_parks_what_it_holds(self):
        self.pool.pins["a"] = pin("gpu", slot=0)
        post = linked(self.pool, self.saver())
        self.pool.drain("gpu")
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")
        self.assertEqual(post.ops()[0], "save")

    def test_draining_leaves_the_other_backend_alone(self):
        self.pool.pins["b"] = pin("cpu", slot=1)
        post = linked(self.pool, self.saver())
        self.pool.drain("gpu")
        self.assertIsNone(self.pool.pins["b"]["parked"])

    def test_resuming_puts_it_back_in_service(self):
        self.pool.drain("gpu")
        self.assertTrue(self.pool.resume("gpu"))
        self.assertTrue(self.pool._usable(self.gpu, 1000))

    def test_an_unknown_backend_cannot_be_drained(self):
        self.assertIsNone(self.pool.drain("nope"))
        self.assertFalse(self.pool.resume("nope"))

    def test_it_waits_for_work_in_flight_to_finish(self):
        """A request already running must be allowed to end."""
        self.pool._take(self.gpu, "a", tokens=10)          # gpu busy = 1
        done = threading.Event()

        def drain():
            self.pool.drain("gpu", deadline=5.0)
            done.set()

        threading.Thread(target=drain, daemon=True).start()
        self.assertFalse(done.wait(0.3))                   # still waiting
        self.pool.release(self.gpu, "a")
        self.assertTrue(done.wait(3.0))

    def test_it_gives_up_waiting_rather_than_hanging(self):
        self.pool._take(self.gpu, "a", tokens=10)
        report = self.pool.drain("gpu", deadline=0.2)
        self.assertFalse(report["quiet"])

    def test_the_dashboard_can_see_it(self):
        self.pool.drain("cpu")
        row = [b for b in self.pool.status()["backends"] if b["name"] == "cpu"][0]
        self.assertTrue(row["draining"])


class PrefillStaysOffABackendThatDoesNotRead(unittest.TestCase):
    """`reads: False` means no prompt is ever prefilled there.

    A machine turns it off where an instance is better at generating than at
    prefilling, so giving that instance a prompt to read spends the one thing
    it is better at. The fixture calls it "gpu" because that is what this box
    runs there; nothing in the rule is about the hardware."""

    def setUp(self):
        # _read_prefix links a file into the slot directory. setUp hands
        # this store to the pool, so the link cannot land in a running
        # router's own.
        root = Path(tempfile.mkdtemp())
        (root / "slots").mkdir()
        self.was = SANDBOX.store
        SANDBOX.store = router.Store(root)
        self.addCleanup(shutil.rmtree, root)
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True},
             {"name": "cpu", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True},
             {"name": "cpu2", "url": "http://cpu2", "pref": 2, "prefill": True, "generate": True}],
            store=SANDBOX.store, watch=False)
        self.gpu, self.cpu, self.cpu2 = self.pool.backends
        for be in self.pool.backends:
            be.update(up=True, slots=1, n_ctx=150000)
        self.addCleanup(lambda: setattr(SANDBOX, "store", self.was))

    def test_a_new_conversation_is_not_read_on_the_gpu(self):
        self.assertNotEqual(self.pool.acquire("a", 1000)[0]['name'], "gpu")

    def test_it_fills_the_reading_backends_in_order(self):
        """Backwards through pref, so the best place to generate is read on
        last and stays free to generate."""
        self.assertEqual(self.pool.acquire("a", 1000)[0]['name'], "cpu2")
        self.assertEqual(self.pool.acquire("b", 1000)[0]['name'], "cpu")

    def test_a_conversation_living_on_the_gpu_still_reads_elsewhere(self):
        """A later turn is not a few tokens. It can carry a whole file, and
        22% of turns re-read everything. So the cache comes back to a cpu to
        be read, and returns to the gpu only to generate."""
        self.pool.pins["a"] = pin("gpu", slot=0)
        self.assertNotEqual(self.pool.acquire("a", 1000)[0]['name'], "gpu")

    def test_it_waits_rather_than_reading_on_the_gpu(self):
        self.pool._take(self.cpu, "x", tokens=10)
        self.pool._take(self.cpu2, "y", tokens=10)
        got = {}

        def ask():
            got["be"] = self.pool.acquire("new", 1000)[0]

        threading.Thread(target=ask, daemon=True).start()
        time.sleep(0.4)
        self.assertNotIn("be", got)              # gpu is free and unused
        self.pool.release(self.cpu, "x")
        for _ in range(30):
            if "be" in got:
                break
            time.sleep(0.1)
        self.assertEqual(got["be"]["name"], "cpu")

    def test_it_does_not_wait_for_the_gpu_even_when_its_cache_is_there(self):
        """The gpu cannot read, so waiting for it would never help. The copy
        on disk is what gets the conversation back, on whichever cpu is free."""
        self.pool.pins["a"] = pin("gpu", slot=0, parked="a.park")
        self.assertNotEqual(self.pool.acquire("a", 1000)[0]['name'], "gpu")

    def test_it_gives_up_when_nothing_can_ever_serve_it(self):
        self.cpu["up"] = self.cpu2["up"] = False
        self.assertIsNone(self.pool.acquire("brand-new", 1000)[0])



class CacheRereads(unittest.TestCase):
    """A slot restored from a file has no checkpoint, so the backend reads the
    whole prompt again and says so. Each one costs minutes."""

    REREAD = ("438.34.317.915 I slot   operator(): id  0 | task 35055 | forcing full "
              "prompt re-processing due to lack of cache data (likely due to SWA "
              "or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)\n")

    def test_reads_the_reread_line(self):
        self.assertEqual(router.cache_event(self.REREAD), ("reread", 0.0))

    def test_the_watch_counts_them(self):
        d = tempfile.mkdtemp()
        try:
            path = Path(d) / "gpu.log"
            path.write_text(self.REREAD + EVICT + self.REREAD)
            watch = router.CacheWatch(path)
            watch.poll()
            self.assertEqual(watch.stats["rereads"], 2)
            self.assertEqual(watch.stats["evictions"], 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class HowARequestStarted(unittest.TestCase):
    """The dashboard's caching view names what each request extended."""

    def test_its_own_copy_back_from_disk(self):
        self.assertEqual(router.how_started(False, True, False), "recalled")

    def test_a_saved_opening(self):
        self.assertEqual(router.how_started(False, False, True), "saved prompt")

    def test_its_own_cache_still_in_a_slot(self):
        self.assertEqual(router.how_started(True, False, False), "warm slot")

    def test_nothing(self):
        self.assertEqual(router.how_started(False, False, False), "cold")


class ASlotRateIsNullUntilOneIsMeasured(unittest.TestCase):
    """A rate needs a window, and there is no rate before the first one.

    Reported as 0.0 that is indistinguishable from a slot crawling at zero,
    which is exactly what STALL_RATE tests for - so every turn read as stalled
    for its first RATE_WINDOW, on the dashboard and in the history."""

    def setUp(self):
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                                watch=False)
        self.be = self.pool.backends[0]

    @staticmethod
    def slots(decoded, processed=0, task=1):
        return [{"id": 0, "is_processing": True, "id_task": task,
                 "next_token": {"n_decoded": decoded},
                 "n_prompt_tokens_processed": processed,
                 "n_prompt_tokens_total": 1000, "n_prompt_tokens": 1000,
                 "prompt_n": 0}]

    def test_the_first_reading_reports_no_rate_at_all(self):
        self.pool._read_slots(self.be, self.slots(10))
        detail = self.be["slots_detail"][0]
        self.assertIsNone(detail["tg_rate"])
        self.assertIsNone(detail["pp_rate"])

    def test_a_rate_appears_once_the_window_has_resolved(self):
        self.pool._read_slots(self.be, self.slots(10))
        self.be["slot_prev"][0]["since"] -= SANDBOX.tuning.rate_window + 1
        self.pool._read_slots(self.be, self.slots(30))
        self.assertIsNotNone(self.be["slots_detail"][0]["tg_rate"])

    def test_a_backend_total_reads_an_unmeasured_slot_as_nothing(self):
        """tg_live sums the slots, and None does not add."""
        self.pool._read_slots(self.be, self.slots(10))
        self.assertEqual(self.be["stats"]["tg_live"], 0.0)


class SlotTimeHistory(unittest.TestCase):
    """Slot-seconds by phase, a bucket a minute, from the poll."""

    @staticmethod
    def polled(phase, tg_rate=0.0, up=True):
        return [{"name": "cpu", "up": up,
                 "slots_detail": [{"phase": phase, "tg_rate": tg_rate}]}]

    def test_the_gap_between_polls_belongs_to_the_earlier_phase(self):
        h = router.History(keep=10, step=60.0)
        h.push(self.polled("reading"), 100.0)
        h.push(self.polled("generating", 4.0), 110.0)
        h.push(self.polled("generating", 4.0), 115.0)
        cur = h.snapshot()["backends"]["cpu"]["cur"]
        self.assertEqual(cur, {"read": 10.0, "gen": 5.0, "stalled": 0.0, "secs": 15.0})

    def test_a_slow_generating_slot_counts_as_stalled(self):
        h = router.History(keep=10, step=60.0)
        h.push(self.polled("generating", 0.05), 100.0)
        h.push(self.polled("generating", 0.05), 102.0)
        cur = h.snapshot()["backends"]["cpu"]["cur"]
        self.assertEqual(cur["stalled"], 2.0)
        self.assertEqual(cur["gen"], 0.0)

    def test_a_rate_not_yet_measured_is_not_a_stall(self):
        """None is "no rate yet", which a slot carries for its first
        RATE_WINDOW of generating. Banked as a stall it put up to ten seconds
        of phantom contention into the graph on every turn."""
        h = router.History(keep=10, step=60.0)
        h.push(self.polled("generating", None), 100.0)
        h.push(self.polled("generating", None), 108.0)
        cur = h.snapshot()["backends"]["cpu"]["cur"]
        self.assertEqual(cur["stalled"], 0.0)
        self.assertEqual(cur["gen"], 8.0, "the slot was generating all the same")

    def test_a_minute_edge_splits_the_gap(self):
        h = router.History(keep=10, step=60.0)
        h.push(self.polled("reading"), 50.0)
        h.push(self.polled("reading"), 70.0)
        row = h.snapshot()["backends"]["cpu"]
        self.assertEqual(row["done"], [{"read": 10.0, "gen": 0.0, "stalled": 0.0, "secs": 10.0}])
        self.assertEqual(row["cur"]["read"], 10.0)

    def test_keeps_only_the_last_buckets(self):
        h = router.History(keep=3, step=60.0)
        h.push(self.polled("idle"), 0.0)
        h.push(self.polled("idle"), 60.0 * 7 + 1)
        self.assertEqual(len(h.snapshot()["backends"]["cpu"]["done"]), 3)

    def test_a_backend_that_is_down_is_not_charged(self):
        h = router.History(keep=10, step=60.0)
        h.push(self.polled("reading", up=False), 0.0)
        h.push(self.polled("reading", up=False), 10.0)
        self.assertEqual(h.snapshot()["backends"], {})


class DiskSummary(unittest.TestCase):
    def test_copies_are_counted_in_bytes_against_the_budget(self):
        pins = OrderedDict(a=pin("cpu", parked="a.park"), b=pin("cpu"), c=pin("gpu", parked="c.park"))
        pins["a"]["bytes"] = 100
        pins["c"]["bytes"] = 250
        openings = {"x": "base-x.park", "y": "deep-y.park"}
        got = router.disk_summary(pins, openings, {"x": 10, "y": 20},
                                  tuning=SANDBOX.tuning)
        self.assertEqual(got["copies"], {"count": 2, "bytes": 350, "budget": SANDBOX.tuning.park_budget})
        # One budget, and the two kinds counted under it rather than each
        # against a cap of its own.
        self.assertEqual(got["openings"],
                         {"count": 2, "bytes": 30, "budget": SANDBOX.tuning.block_budget})
        self.assertEqual(got["bases"], {"count": 1})
        self.assertEqual(got["deeps"], {"count": 1})


class WhoIsWaiting(unittest.TestCase):
    """The dashboard lists each waiter with what it waits for."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True},
             {"name": "cpu", "url": "http://cpu", "pref": 1}], watch=False)
        for be in self.pool.backends:
            be["up"] = True
            be["n_ctx"] = 1000

    def test_a_ticket_counts_until_it_ends(self):
        ticket = self.pool.begin_wait("abcdefghij", 500)
        self.assertEqual(self.pool.status()["waiting"], 1)
        self.assertEqual(self.pool.waiting, 1)
        self.pool.end_wait(ticket)
        self.assertEqual(self.pool.status()["waiting"], 0)
        self.assertEqual(self.pool.status()["waiting_detail"], [])

    def test_a_new_conversation_wants_a_reader(self):
        self.pool.begin_wait("abcdefghij", 500)
        row = self.pool.status()["waiting_detail"][0]
        self.assertEqual((row["conv"], row["tokens"], row["waiting_on"]), ("abcdefgh", 500, "prefill"))
        self.assertGreaterEqual(row["waited"], 0)

    def test_a_pinned_conversation_wants_its_backend(self):
        self.pool.pins["abc"] = pin("cpu")
        self.pool.begin_wait("abc", 500)
        row = self.pool.status()["waiting_detail"][0]
        self.assertEqual((row["waiting_on"], row["backend"]), ("pinned", "cpu"))

    def test_turns_queued_for_the_gpu_are_counted(self):
        """They hold no backend while they wait, so nothing else counts them.

        A reader is handed back before the wait, which is the point, and the
        cost is that a turn waiting to generate is invisible: it is in no
        slot, in no queue, and pinned to nothing."""
        self.assertEqual(self.pool.status()["waiting_to_generate"], 0)
        gpu = self.pool.backends[0]
        gpu.update(slots=1, busy=1,
                   slots_detail=[{"id": 0, "busy": True, "phase": "generating"}])
        seen = []
        started = threading.Event()

        def queue_for_it():
            started.set()
            self.pool._wait_to_generate(gpu, lambda: not len(seen))

        hand = threading.Thread(target=queue_for_it, daemon=True)
        hand.start()
        started.wait(5)
        for _ in range(200):
            if self.pool.status()["waiting_to_generate"] == 1:
                break
            time.sleep(0.01)
        self.assertEqual(self.pool.status()["waiting_to_generate"], 1)

        seen.append("the client left")     # ends the wait
        hand.join(5)
        self.assertEqual(self.pool.status()["waiting_to_generate"], 0,
                         "the count outlived the wait")

    def test_a_pin_on_a_backend_that_does_not_read_is_not_waited_for(self):
        """Every turn ends on the gpu, so every conversation is pinned there.

        acquire gives up a pin like that at once and takes the first reader
        free, so naming the gpu here reads as the gpu holding the box up while
        it sits idle. What this waits for is a reader."""
        self.pool.pins["abc"] = pin("gpu")
        self.pool.begin_wait("abc", 500)
        row = self.pool.status()["waiting_detail"][0]
        self.assertEqual((row["waiting_on"], row["backend"]), ("prefill", None))

    def test_too_big_for_any_reader(self):
        self.pool.begin_wait("abc", 5000)
        self.assertEqual(self.pool.status()["waiting_detail"][0]["waiting_on"], "big")

    def test_status_says_what_each_backend_may_do(self):
        can = {b["name"]: (b["prefill"], b["generate"])
               for b in self.pool.status()["backends"]}
        self.assertEqual(can, {"gpu": (False, True), "cpu": (True, True)})


class TheBackendTableIsCheckedAtStartup(unittest.TestCase):
    """A table that cannot serve is refused before the port opens.

    read_backend_table is where the check lives, and build() calls it before
    anything listens. So a misconfiguration is a message at startup rather
    than a turn that fails much later on a machine nobody is watching.

    This used to run in a subprocess, because the check was work done at
    import and there was no other way to reach it. It is a function now."""

    def loading(self, table):
        """Read this table the way the router does. Returns (code, output)."""
        room = Path(tempfile.mkdtemp(prefix="router-table-"))
        self.addCleanup(shutil.rmtree, room, ignore_errors=True)
        path = room / "backends.json"
        path.write_text(json.dumps(table))
        try:
            router.read_backend_table({"ROUTER_BACKENDS": str(path)})
        except SystemExit as stop:
            return 1, str(stop)
        return 0, ""

    @staticmethod
    def row(name, prefill=True, generate=True, pref=0):
        return {"name": name, "url": f"http://{name}", "pref": pref,
                "prefill": prefill, "generate": generate, "node": 0}

    def test_a_good_table_loads(self):
        code, said = self.loading([self.row("solo")])
        self.assertEqual(code, 0, said)

    def test_an_instance_that_can_do_neither_is_refused(self):
        code, said = self.loading([self.row("solo", prefill=False,
                                            generate=False)])
        self.assertNotEqual(code, 0)
        self.assertIn("neither prefills nor generates", said)

    def test_a_table_with_nothing_to_prefill_on_is_refused(self):
        """Every request starts with a prompt to read."""
        code, said = self.loading([self.row("gen", prefill=False)])
        self.assertNotEqual(code, 0)
        self.assertIn("can prefill", said)

    def test_a_table_with_nothing_to_generate_on_is_refused(self):
        code, said = self.loading([self.row("pre", generate=False)])
        self.assertNotEqual(code, 0)
        self.assertIn("can generate", said)

    def test_an_old_table_is_told_what_the_fields_became(self):
        """`reads` was one flag where there are now two, and a plain missing
        field message would not say which way round they go."""
        code, said = self.loading([{"name": "gpu", "url": "http://gpu",
                                    "pref": 0, "reads": False, "node": 0}])
        self.assertNotEqual(code, 0)
        self.assertIn("prefill: false, generate: true", said)


class OneTurnAtATime(unittest.TestCase):
    """A conversation is one pin, one slot and one copy, so one turn at a time.

    Two turns of it at once each moved the same three. The second took a
    reader, cleared the slot the first was reading in, and wrote its own
    state over the first one's copy under the same name. The first then
    carried that copy to the gpu and read its whole prompt again."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        for be in self.pool.backends:
            be["up"] = True
            be["n_ctx"] = 1000

    def claim(self, conv, alive=None):
        """Claim in a thread, and say when it got in. Returns (thread, got)."""
        got = threading.Event()
        ticket = self.pool.begin_wait(conv, 500)

        def run():
            if self.pool.claim_turn(conv, ticket, alive):
                got.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, got, ticket

    def test_the_second_turn_waits_for_the_first(self):
        first = self.pool.begin_wait("abc", 500)
        self.assertTrue(self.pool.claim_turn("abc", first))
        _, got, _ = self.claim("abc")
        self.assertFalse(got.wait(2))
        self.pool.finish_turn("abc", first)
        self.assertTrue(got.wait(5))

    def test_another_conversation_is_not_held_up(self):
        first = self.pool.begin_wait("abc", 500)
        self.pool.claim_turn("abc", first)
        _, got, _ = self.claim("xyz")
        self.assertTrue(got.wait(5))

    def test_a_request_with_no_conversation_is_never_held(self):
        self.assertTrue(self.pool.claim_turn(None, 1))
        self.assertTrue(self.pool.claim_turn(None, 2))
        self.assertEqual(self.pool.turns, {})

    def test_a_client_that_leaves_stops_waiting_and_holds_nothing(self):
        first = self.pool.begin_wait("abc", 500)
        self.pool.claim_turn("abc", first)
        here = [True]
        thread, got, second = self.claim("abc", lambda: here[0])
        here[0] = False
        thread.join(5)
        self.assertFalse(got.is_set())
        self.assertEqual(self.pool.turns.get("abc"), first)
        # It never held the conversation, so its ending must not free it.
        self.pool.finish_turn("abc", second)
        self.assertEqual(self.pool.turns.get("abc"), first)

    def test_finishing_lets_the_next_turn_hold_it(self):
        first = self.pool.begin_wait("abc", 500)
        self.pool.claim_turn("abc", first)
        self.pool.finish_turn("abc", first)
        self.assertEqual(self.pool.turns, {})
        second = self.pool.begin_wait("abc", 500)
        self.assertTrue(self.pool.claim_turn("abc", second))
        self.assertEqual(self.pool.turns, {"abc": second})

    def test_the_waiter_says_it_waits_for_a_turn(self):
        first = self.pool.begin_wait("abc", 500)
        self.pool.claim_turn("abc", first)
        self.claim("abc")
        for _ in range(50):
            rows = self.pool.status()["waiting_detail"]
            if len(rows) == 2:
                break
            time.sleep(0.05)
        waiting_on = [row["waiting_on"] for row in rows]
        self.assertEqual(waiting_on, ["prefill", "turn"])

    def test_the_turn_ahead_keeps_its_place_on_the_flow_board(self):
        """Both turns are one conversation, and the board has one row for it."""
        first = self.pool.begin_wait("abc", 500)
        self.pool.claim_turn("abc", first)
        self.pool.note_stage("abc", "generate", "cpu", 0)
        self.claim("abc")
        time.sleep(0.3)
        live = self.pool.flow.report()["live"]
        self.assertEqual([row["stage"] for row in live], ["generate"])


class RecentRequests(unittest.TestCase):
    def test_the_newest_comes_first_and_the_list_is_bounded(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        be = pool.backends[0]
        for i in range(SANDBOX.tuning.recent_requests + 5):
            pool.note_request(f"conv{i:04d}xx", be, "/v1/messages", 12.34, 1.0, "cold", 100)
        rows = pool.status()["recent_requests"]
        self.assertEqual(len(rows), SANDBOX.tuning.recent_requests)
        self.assertEqual(rows[0]["conv"], f"conv{SANDBOX.tuning.recent_requests + 4:04d}"[:8])
        self.assertEqual(rows[0]["started"], "cold")
        self.assertEqual(rows[0]["took"], 12.3)


class ACopyIsNoUseWhenTheOpeningChanges(unittest.TestCase):
    """The tools render into the block at the front of every prompt.

    Change one sentence of one tool description and every token after it
    differs, so a copy written before the change is a prefix of nothing.
    llama.cpp finds that out by restoring it, failing to match, and reading
    the whole prompt again - and because a conversation holding a copy never
    looks for an opening, it does not even get the block it still shares.

    OpenCode's bash tool went from "output exceeds 800 lines" to "2000 lines"
    on 13 September. The prompts parted at token 503, the earliest checkpoint
    was at 8,419, and 117,847 tokens were read from cold."""

    def setUp(self):
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                                watch=False)
        self.pool.backends[0].update(up=True, slots=1, n_ctx=150000)
        self.removed = []

    @staticmethod
    def cuts(opening):
        return [(-1, opening), (0, "deeper" + opening)]

    def park(self, opening):
        self.pool.pins["a"] = pin("cpu", slot=0, parked="a.park")
        self.pool.pins["a"]["bytes"] = 5_000_000
        self.pool.note_holds("a", self.pool.backends[0], self.cuts(opening))

    def drop(self, opening):
        return self.pool.forget_stale_park("a", self.cuts(opening),
                                           remove=self.removed.append)

    def test_the_copy_goes_when_the_opening_changed(self):
        self.park("tools-v1")
        self.assertTrue(self.drop("tools-v2"))
        self.assertIsNone(self.pool.pins["a"]["parked"])
        self.assertIsNone(self.pool.pins["a"]["slot"],
                          "the slot begins with the same dead opening")
        self.assertEqual(self.removed, ["a.park"])

    def test_the_copy_stays_when_the_opening_is_the_same(self):
        self.park("tools-v1")
        self.assertFalse(self.drop("tools-v1"))
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")
        self.assertEqual(self.removed, [])

    def test_a_copy_from_before_this_was_recorded_is_left_alone(self):
        """A pin adopted from disk names no opening. Guessing it is dead
        would throw away every copy the router restarted with."""
        self.pool.pins["a"] = pin("cpu", slot=0, parked="a.park")
        self.assertFalse(self.drop("tools-v2"))
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")

    def test_the_opening_is_loaded_once_the_dead_copy_is_gone(self):
        """warm_prefix returns early for a conversation that has a copy, so
        dropping the copy is what puts it back on the path that loads one."""
        self.park("tools-v1")
        self.pool.openings["tools-v2"] = "base-tools-v2.park"
        post = FakeLink(written=SANDBOX.tuning.park_floor + 1)

        self.drop("tools-v2")
        loaded = with_link(self.pool, post).warm_prefix(
            "a", self.cuts("tools-v2"), [], "sys", [],
            self.pool.backends[0], 0, "/v1/messages")
        self.assertTrue(loaded, "no opening was loaded for the changed tools")
        self.assertIn("restore", post.ops())


class ThePinRecordSurvivesTheNextTurn(unittest.TestCase):
    """_take runs at the start of every turn, and must not forget the record.

    It used to build a fresh dict, so only the fields it named itself lived
    through a turn. Two of the rest had readers. `opening` is written by
    note_holds at the end of a turn and read by forget_stale_park at the
    start of the next one, so it was always gone and that whole path was
    dead in production while its own unit tests passed - they set the pin by
    hand. `parked_at` is what the PARK_BUDGET sweep sorts by, so a copy whose
    conversation ran again sorted as age 0 and was dropped first, which is
    the failure the sweep's own comment says it fixed.

    AParkedCopyKeepsItsSize is this same bug, caught once already."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "cpu", "url": "http://cpu", "pref": 0},
             {"name": "gpu", "url": "http://gpu", "pref": 1}], watch=False)
        self.cpu, self.gpu = self.pool.backends
        for be in self.pool.backends:
            be.update(up=True, slots=1, n_ctx=150000)

    @staticmethod
    def cuts(opening):
        return [(-1, opening), (0, "deeper" + opening)]

    def turn(self, opening):
        """One whole turn, through the calls the request path makes."""
        self.pool._take(self.cpu, "a", tokens=1000)
        self.pool.note_slot("a", 0)
        self.pool.note_holds("a", self.cpu, self.cuts(opening))
        self.pool.release(self.cpu, "a")

    def test_a_changed_opening_is_noticed_on_the_turn_after_it(self):
        self.turn("tools-v1")
        self.pool.pins["a"]["parked"] = "a.park"
        self.pool._take(self.cpu, "a", tokens=1000)     # the next turn starts
        removed = []
        self.assertTrue(
            self.pool.forget_stale_park("a", self.cuts("tools-v2"),
                                        remove=removed.append),
            "the opening this conversation started from did not survive "
            "_take, so a changed one could never be noticed")
        self.assertEqual(removed, ["a.park"])

    def test_an_unchanged_opening_still_keeps_the_copy(self):
        self.turn("tools-v1")
        self.pool.pins["a"]["parked"] = "a.park"
        self.pool._take(self.cpu, "a", tokens=1000)
        removed = []
        self.assertFalse(self.pool.forget_stale_park("a", self.cuts("tools-v1"),
                                                     remove=removed.append))
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")

    def test_when_the_copy_was_written_survives_the_next_turn(self):
        self.turn("tools-v1")
        self.pool.pins["a"].update(parked="a.park", parked_at=1234.0)
        self.pool._take(self.cpu, "a", tokens=1000)
        self.assertEqual(self.pool.pins["a"].get("parked_at"), 1234.0,
                         "a copy whose conversation ran again sorted as "
                         "age 0 and the budget dropped it first")

    def test_a_first_turn_still_writes_a_whole_record(self):
        """Several readers index these fields directly."""
        self.pool._take(self.cpu, "fresh", tokens=10)
        record = self.pool.pins["fresh"]
        for field in ("backend", "slot", "tokens", "last", "inflight",
                      "turns", "parked", "bytes"):
            self.assertIn(field, record, field)

    def test_moving_backends_still_drops_the_slot(self):
        """A slot id only means something on its own backend."""
        self.pool._take(self.cpu, "a", tokens=10)
        self.pool.note_slot("a", 0)
        self.pool._take(self.gpu, "a", tokens=10)
        self.assertIsNone(self.pool.pins["a"]["slot"])

    def test_staying_on_one_backend_keeps_the_slot(self):
        self.pool._take(self.cpu, "a", tokens=10)
        self.pool.note_slot("a", 0)
        self.pool._take(self.cpu, "a", tokens=10)
        self.assertEqual(self.pool.pins["a"]["slot"], 0)


class ARefusedParkKeepsTheCopyItHad(unittest.TestCase):
    """Three outcomes, not two.

    A save that comes back short means the slot holds somebody else, so the
    copy is gone. A save the backend refuses says nothing about the slot and
    nothing about the file, which is still on disk and is still a prefix of
    this conversation. Forgetting it there left the file orphaned - off the
    PARK_BUDGET sweep and off MAX_PINS eviction, eating the cap uncounted
    until a restart - while the conversation re-read its whole prompt. One
    refused save during a backend restart was enough."""

    def setUp(self):
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                                watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=1, n_ctx=150000)
        self.pool.pins["a"] = pin("cpu", slot=0, parked="a.park")
        self.pool.pins["a"]["bytes"] = 5_000_000
        self.pool.pins["a"]["parked_at"] = 1234.0
        self.removed = []

    def park(self, post):
        return with_link(self.pool, post)._save_park(
            "a", self.cpu, 0, remove=self.removed.append)

    def test_a_refused_save_leaves_the_copy_named(self):
        self.assertFalse(self.park(FakeLink(fail_on="save")))
        record = self.pool.pins["a"]
        self.assertEqual(record["parked"], "a.park",
                         "the copy is still on disk and is still a prefix of "
                         "this conversation, but the pin forgot it")
        self.assertEqual(record["bytes"], 5_000_000,
                         "a copy the budget can no longer see is disk the "
                         "router thinks it still has")
        self.assertEqual(self.removed, [], "the file was not deleted either")

    def test_a_refused_save_leaves_the_slot_alone(self):
        """The backend said nothing about the slot, so the cache is still in
        it and the next turn can still extend it."""
        self.park(FakeLink(fail_on="save"))
        self.assertEqual(self.pool.pins["a"]["slot"], 0)

    def test_a_short_save_does_forget_the_copy(self):
        """That one really is gone: the slot holds somebody else."""
        self.assertFalse(self.park(FakeLink(written=1)))
        self.assertIsNone(self.pool.pins["a"]["parked"])
        self.assertIsNone(self.pool.pins["a"]["slot"])
        self.assertEqual(self.removed, ["a.park"])


class ACopyIsMeasuredOnTheDisk(unittest.TestCase):
    """PARK_BUDGET is spent against a copy's size, so the size has to be real.

    `n_written` counts the state and not whatever the build writes after it.
    patches/slot-state-carries-checkpoints.patch appends the checkpoints, and
    a server that does not report the trailer answers 49 MB for a 107 MB file
    - so the cap is enforced against half the disk in use. The file is the
    one place the answer cannot be wrong.
    tests/live/test_llama_beliefs.py asserts the two agree on a real one."""

    def setUp(self):
        root = Path(tempfile.mkdtemp())
        (root / "slots").mkdir()
        self.was = SANDBOX.store
        SANDBOX.store = router.Store(root)
        self.addCleanup(shutil.rmtree, root)
        self.addCleanup(lambda: setattr(SANDBOX, "store", self.was))
        self.pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                                store=SANDBOX.store, watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=1, n_ctx=150000)
        self.pool.pins["a"] = pin("cpu", slot=0)

    def test_the_size_on_disk_wins_over_what_the_backend_reported(self):
        real = SANDBOX.tuning.park_floor + 10_000_000
        (SANDBOX.store.slots / "a.park").write_bytes(b"\0" * real)
        # The backend undercounts by more than half, as an unpatched one does.
        with_link(self.pool,
                  FakeLink(written=real // 3))._save_park("a", self.cpu, 0)
        self.assertEqual(self.pool.pins["a"]["bytes"], real,
                         "the budget was spent against the backend's figure")

    def test_the_backend_figure_stands_in_when_there_is_no_file(self):
        """A stub writes no file, and so does a backend whose save landed
        somewhere this router cannot stat."""
        with_link(self.pool,
                  FakeLink(written=200_000_000))._save_park("a", self.cpu, 0)
        self.assertEqual(self.pool.pins["a"]["bytes"], 200_000_000)


class WhatIsLeftToRead(unittest.TestCase):
    """A slot says how much of its prompt is reused, read, and still to read.

    The three have to add up to the prompt. They did not: n_prompt_tokens
    counts what the slot holds, which grows as the prompt is consumed, so
    subtracting the counters from it gave what had already been read. A slot
    98% served from cache reported "512 / 89,848 read" and looked broken.
    patches/slots-report-the-prompt-size.patch adds the prompt's own size."""

    def pool(self):
        made = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                           watch=False)
        made.backends[0].update(up=True, slots=1, n_ctx=150000)
        return made

    @staticmethod
    def raw(**over):
        slot = {"id": 0, "id_task": 7, "is_processing": True, "n_ctx": 150000,
                "n_prompt_tokens_total": 89848, "n_prompt_tokens_cache": 88824,
                "n_prompt_tokens_processed": 512,
                "next_token": {"n_decoded": 0}}
        slot.update(over)
        # What the slot holds: the reused part plus what it has read so far.
        slot.setdefault("n_prompt_tokens",
                        slot["n_prompt_tokens_cache"]
                        + slot["n_prompt_tokens_processed"])
        return slot

    def detail(self, **over):
        pool = self.pool()
        pool._read_slots(pool.backends[0], [self.raw(**over)])
        return pool.backends[0]["slots_detail"][0]

    def test_the_three_parts_add_up_to_the_prompt(self):
        row = self.detail()
        self.assertEqual((row["cached"], row["done"], row["prompt"]),
                         (88824, 512, 512))
        self.assertEqual(row["cached"] + row["done"] + row["prompt"], 89848)

    def test_a_prompt_read_to_the_end_has_nothing_left(self):
        row = self.detail(n_prompt_tokens_processed=1024)
        self.assertEqual(row["prompt"], 0)

    def test_generated_tokens_do_not_grow_the_prompt(self):
        """n_prompt_tokens grows with every token generated. The total does
        not, so a long answer no longer inflates what the slot says it read."""
        row = self.detail(n_prompt_tokens_processed=1024,
                          n_prompt_tokens=89848 + 3309,
                          next_token={"n_decoded": 3309})
        self.assertEqual(row["cached"] + row["done"] + row["prompt"], 89848)
        self.assertEqual(row["phase"], "generating")

    def test_a_backend_without_the_patch_still_answers(self):
        """It has no better number to offer, so the old arithmetic stands."""
        slot = self.raw()
        del slot["n_prompt_tokens_total"]
        pool = self.pool()
        pool._read_slots(pool.backends[0], [slot])
        row = pool.backends[0]["slots_detail"][0]
        self.assertEqual(row["cached"], 88824)
        self.assertGreaterEqual(row["prompt"], 0)


class WhereToGenerate(unittest.TestCase):
    """Only the gpu generates, and a turn waits for it.

    It decodes about five times faster than a socket of cpu, and a generation
    started on a cpu holds a slot that could be prefilling for the whole of
    it: minutes of somebody else's read spent to save seconds on this turn.
    So there is nothing to choose between. The turn goes to the gpu, and if
    the gpu is busy it queues for it rather than settle for a cpu.

    The one answer that is not the gpu is no answer at all: the gpu down,
    draining, or too small for this prompt. Then the turn generates where its
    prompt was read, because the alternative is not answering."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True},
             {"name": "cpu", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True},
             {"name": "cpu0", "url": "http://cpu0", "pref": 2, "prefill": True, "generate": True},
             {"name": "cpu2", "url": "http://cpu2", "pref": 3, "prefill": True, "generate": True}],
            watch=False)
        self.gpu, self.cpu, self.cpu0, self.cpu2 = self.pool.backends
        for be in self.pool.backends:
            be.update(up=True, slots=1, n_ctx=150000)

    def fill(self, be):
        be["busy"] = be["slots"]

    def test_the_gpu(self):
        self.assertEqual(self.pool.generator(1000)["name"], "gpu")

    def test_the_gpu_even_when_every_cpu_is_idle_and_it_is_not(self):
        """Busy is a queue to join, not a reason to go elsewhere."""
        self.fill(self.gpu)
        self.assertEqual(self.pool.generator(1000)["name"], "gpu")

    def test_never_a_cpu(self):
        for be in self.pool.backends:
            self.fill(be)
        self.assertEqual(self.pool.generator(1000)["name"], "gpu")

    def test_nothing_when_the_gpu_is_down(self):
        self.gpu["up"] = False
        self.assertIsNone(self.pool.generator(1000))

    def test_nothing_when_the_gpu_is_draining(self):
        self.gpu["draining"] = True
        self.assertIsNone(self.pool.generator(1000))

    def test_nothing_when_the_prompt_outgrew_the_gpu(self):
        self.gpu["n_ctx"] = 4096
        self.assertIsNone(self.pool.generator(100000))

    def test_a_migration_target_generates_and_does_not_prefill(self):
        """Both halves, not one derived from the other: an instance that does
        both is not worth carrying a turn to."""
        self.assertFalse(self.gpu["prefill"])
        self.assertTrue(self.gpu["generate"])
        self.assertEqual(self.pool.generator(1000)["name"], "gpu")

    def test_nothing_when_the_only_generator_also_prefills(self):
        """A turn already sitting in a slot that can generate stays there."""
        self.gpu["prefill"] = True
        self.assertIsNone(self.pool.generator(1000))

    def test_nothing_when_the_candidate_does_not_generate(self):
        """prefill off and generate off is refused at startup, but a backend
        that only prefills is not a place to carry a turn to."""
        self.gpu["generate"] = False
        self.assertIsNone(self.pool.generator(1000))


class AnInstanceThatDoesNotGenerate(unittest.TestCase):
    """`prefill: true, generate: false` reads for the pool and hands every
    turn on. It could not be said at all while one flag meant both."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gen", "url": "http://gen", "pref": 0,
              "prefill": False, "generate": True},
             {"name": "pre", "url": "http://pre", "pref": 1,
              "prefill": True, "generate": False}], watch=False)
        self.gen, self.pre = self.pool.backends
        for be in self.pool.backends:
            be.update(up=True, slots=1, n_ctx=150000)

    def test_it_is_still_where_a_new_prompt_is_read(self):
        self.assertEqual(self.pool.acquire("a", 1000)[0]['name'], "pre")

    def test_it_waits_rather_than_generate_where_it_is_set_not_to(self):
        """The old fallback was "answering slowly beats not answering", which
        took the prefiller back. Where an operator said no, the turn waits: it
        is parked on disk, so waiting there costs nothing but time."""
        self.pool.pins["a"] = pin("pre", slot=0)
        self.gen.update(up=False)          # no generator to carry it to
        gave_up = []
        # `alive` is what ends the wait, so answer False on the second ask.
        def alive(asked=[]):
            asked.append(1)
            gave_up.append(len(asked))
            return len(asked) < 2
        with self.assertRaises(router.Gone,
                               msg="it generated on an instance set not to"):
            with_link(self.pool, FakeLink(written=200_000_000)).hand_off(
                "a", self.pre, 1000, alive=alive)
        self.assertTrue(gave_up, "it never waited at all")

    def test_it_generates_in_place_only_when_the_carry_never_happened(self):
        """Nothing was carried, so there is no copy for a generator to take
        and no prompt left to save. Breaking the preference beats throwing away
        a prompt that took tens of minutes - and it says so in the log."""
        self.pool.pins["a"] = pin("pre", slot=None)     # nothing to carry
        got = with_link(self.pool, FakeLink()).hand_off("a", self.pre, 1000)
        self.assertIs(got, self.pre)


class ParkAfterGenerating(unittest.TestCase):
    """A turn that generated on a backend which does not read leaves a copy.

    Nothing else can read there, so the next turn has to start somewhere else.
    Without a copy on disk it would read the whole prompt again."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True},
             {"name": "cpu", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True}],
            watch=False)
        self.gpu, self.cpu = self.pool.backends
        for be in self.pool.backends:
            be.update(up=True, slots=1, n_ctx=150000)

    def saver(self, written=200_000_000):
        return FakeLink(written=written)

    def park(self, be, conv, link, ticket="t"):
        """Queue the copy and wait for the worker, as one call."""
        self.pool.link = link
        took = self.pool.park_later(be, conv, ticket)
        self.assertTrue(self.pool.drain_parks(10.0), "the copy never landed")
        return took

    def test_parks_a_turn_that_generated_where_nothing_reads(self):
        self.pool.pins["a"] = pin("gpu", slot=0)
        post = linked(self.pool, self.saver())
        self.assertTrue(self.park(self.gpu, "a", post))
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")
        self.assertEqual(post.ops()[0], "save")

    def test_leaves_a_turn_that_generated_where_reading_happens(self):
        """It can be read again in place, so a copy would be wasted work."""
        self.pool.pins["a"] = pin("cpu", slot=0)
        post = linked(self.pool, self.saver())
        self.assertFalse(self.park(self.cpu, "a", post))
        self.assertEqual(post.calls, [])

    def test_does_nothing_without_a_known_slot(self):
        self.pool.pins["a"] = pin("gpu", slot=None)
        post = linked(self.pool, self.saver())
        self.assertFalse(self.park(self.gpu, "a", post))
        self.assertEqual(post.calls, [])

    def test_the_copy_comes_back_on_the_next_turn(self):
        """park then acquire then recall is the whole round trip."""
        self.pool.pins["a"] = pin("gpu", slot=0)
        self.park(self.gpu, "a", self.saver())
        be, _ = self.pool.acquire("a", 1000)
        self.assertEqual(be["name"], "cpu")
        post = linked(self.pool, FakeLink())
        self.assertTrue(self.pool.recall("a", be, 1))
        self.assertEqual(post.ops()[0], "restore")

    def test_a_copy_already_on_the_worker_is_not_started_again(self):
        """park_later marks the conversation before the request thread leaves.

        park_all skips a record that is being copied, so a stop arriving
        mid-copy does not start a second save of the same slot. It is also why
        the stop drains the worker first: skipped here and never written there
        would lose the copy altogether."""
        self.pool.pins["a"] = pin("gpu", slot=0)
        slow = FakeLink(written=200_000_000, block=True)
        self.assertTrue(
            with_link(self.pool, slow).park_later(self.gpu, "a", "t"))
        self.assertTrue(self.pool.pins["a"]["parking"],
                        "nothing says the worker has this record")
        self.assertEqual(
            with_link(self.pool, self.saver()).park_all(only="gpu"), 0,
                         "a copy already being written was started again")
        slow.release()
        self.assertTrue(self.pool.drain_parks(10.0))
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")

    def test_the_turn_is_not_over_until_the_copy_has_landed(self):
        """The copy overwrites the file the next turn restores from.

        Same name, in place - so a turn let go early would restore a half
        written file. The ticket travels with the job for that reason."""
        self.pool.pins["a"] = pin("gpu", slot=0)
        self.pool.claim_turn("a", "t")
        slow = FakeLink(written=200_000_000, block=True)
        self.assertTrue(
            with_link(self.pool, slow).park_later(self.gpu, "a", "t"))
        self.assertEqual(self.pool.turns.get("a"), "t")   # still held
        slow.release()
        self.assertTrue(self.pool.drain_parks(10.0))
        self.assertNotIn("a", self.pool.turns)            # and now given back

    def test_a_copy_that_throws_still_gives_the_turn_back(self):
        """claim_turn has no deadline, so a lost ticket wedges it for good.

        Worse than any copy not written, and the reason the worker ends the
        turn in a finally."""
        self.pool.pins["a"] = pin("gpu", slot=0)
        self.pool.claim_turn("a", "t")

        def explode(*args, **kwargs):
            raise RuntimeError("the disk is gone")

        self.pool._save_park = explode      # past its own error handling
        self.assertTrue(
            with_link(self.pool, self.saver()).park_later(self.gpu, "a", "t"))
        self.assertTrue(self.pool.drain_parks(10.0))
        self.assertNotIn("a", self.pool.turns)

    def test_one_copy_at_a_time(self):
        """Two multi-gigabyte writes at once only divide the same disk."""
        self.pool.pins["a"] = pin("gpu", slot=0)
        self.pool.pins["b"] = pin("gpu", slot=0)
        slow = FakeLink(written=200_000_000, block=True)
        self.pool.link = slow
        self.pool.park_later(self.gpu, "a", "ta")
        self.pool.park_later(self.gpu, "b", "tb")
        slow.started.wait(5.0)
        self.assertEqual(len(slow.calls), 1, "both copies ran at once")
        slow.release()
        self.assertTrue(self.pool.drain_parks(10.0))
        self.assertEqual(len(slow.calls), 2)


class MachineLoad(unittest.TestCase):
    """CPU by NUMA node, memory by node, and the gpu, read from the files and
    the one command that report them."""

    STAT = ("cpu  100 0 100 800 0 0 0 0 0 0\n"
            "cpu0 10 0 10 80 0 0 0 0 0 0\n"
            "cpu1 50 0 0 50 0 0 0 0 0 0\n"
            "cpu2 0 0 0 100 0 0 0 0 0 0\n")
    STAT_LATER = ("cpu  200 0 200 900 0 0 0 0 0 0\n"
                  "cpu0 60 0 10 130 0 0 0 0 0 0\n"      # 50 busy of 100
                  "cpu1 150 0 0 50 0 0 0 0 0 0\n"       # 100 busy of 100
                  "cpu2 0 0 0 200 0 0 0 0 0 0\n")       # idle
    MEMINFO = ("Node 0 MemTotal:       196670684 kB\n"
               "Node 0 MemFree:        22765556 kB\n"
               "Node 0 SwapCached:            0 kB\n"
               "Node 0 FilePages:      141527128 kB\n")

    def test_a_cpulist_expands_its_ranges(self):
        self.assertEqual(router.parse_cpulist("0-2,7,36-37\n"), {0, 1, 2, 7, 36, 37})

    def test_busy_share_between_two_samples_per_node(self):
        before, after = router.cpu_times(self.STAT), router.cpu_times(self.STAT_LATER)
        self.assertEqual(router.node_busy(before, after, {0, 1}), 75.0)
        self.assertEqual(router.node_busy(before, after, {2}), 0.0)
        self.assertIsNone(router.node_busy(before, before, {0}), "no time passed")

    def test_the_node_meminfo_in_bytes(self):
        self.assertEqual(router.node_meminfo(self.MEMINFO),
                         {"total": 196670684 * 1024, "free": 22765556 * 1024,
                          "cache": 141527128 * 1024})

    def test_the_nvidia_smi_line(self):
        self.assertEqual(router.gpu_query("0, 15909, 16376\n"),
                         {"util": 0.0, "vram_used": 15909 * 1024 ** 2,
                          "vram_total": 16376 * 1024 ** 2})
        self.assertIsNone(router.gpu_query("No devices were found"))

    def test_a_sample_reads_the_files_and_a_missing_command_turns_the_gpu_off(self):
        d = tempfile.mkdtemp()
        try:
            stat = Path(d) / "stat"
            node = Path(d) / "node0"
            node.mkdir()
            (node / "meminfo").write_text(self.MEMINFO)
            stat.write_text(self.STAT)
            machine = router.Machine(nodes=[{"id": 0, "cpus": {0, 1}, "path": node}],
                                     stat=stat, gpu_cmd=("/nonexistent/nvidia-smi",))
            machine.sample(0.0)
            stat.write_text(self.STAT_LATER)
            machine.sample(2.0)
            gauges = machine.gauges()
            self.assertEqual(gauges["node0.cpu"], 75.0)
            self.assertEqual(gauges["node0.cache"], 141527128 * 1024)
            self.assertIsNone(gauges["gpu.util"])
            self.assertFalse(machine.gpu_ok)
            cpu0 = {"name": "cpu0", "node": 0, "cache": {"mapped_mib": 182142.0, "lazy_mib": 51880.0}}
            report = machine.report([cpu0])
            self.assertEqual(report["nodes"][0]["resident_bytes"], 130262 * 1024 ** 2)
            self.assertEqual(report["nodes"][0]["backends"], ["cpu0"])
            self.assertIsNone(report["gpu"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_load_shares_the_history_buckets(self):
        h = router.History(keep=5, step=10.0)
        h.push([], 0.0)
        h.push_load({"node0.cpu": 20.0, "gpu.util": None})
        h.push([], 5.0)
        h.push_load({"node0.cpu": 40.0, "gpu.util": None})
        self.assertEqual(h.snapshot()["load"]["node0.cpu"]["cur"], 30.0)
        h.push([], 12.0)                       # crosses the edge at 10: the mean is banked
        h.push_load({"node0.cpu": 90.0, "gpu.util": 3.0})
        load = h.snapshot()["load"]
        self.assertEqual(load["node0.cpu"], {"done": [30.0], "cur": 90.0})
        self.assertNotIn("gpu.util", {k for k, v in load.items() if v["done"]})

    def test_status_reports_the_machine(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0, "node": 0}],
                           watch=False)
        machine = pool.status()["machine"]
        self.assertIn("nodes", machine)
        self.assertIn("gpu", machine)


class ReadOnly(unittest.TestCase):
    """The first of the two calls: read the prompt, do not answer."""

    def ask(self, **extra):
        body = {"model": "q", "max_tokens": 900, "stream": True,
                "messages": [{"role": "user", "content": "hi"}], **extra}
        return router.read_only(json.dumps(body).encode())

    def test_asks_for_no_tokens_at_all(self):
        """A generated token lands in the slot. The slot would then hold the
        prompt and one more, so the request that follows is a prefix of it and
        has to rewind. A restored slot carries no checkpoint to rewind to, so
        the backend re-reads everything instead."""
        self.assertEqual(self.ask()["max_tokens"], 0)

    def test_uses_the_field_the_request_already_has(self):
        """/completion counts tokens with n_predict, not max_tokens."""
        asked = router.read_only(json.dumps(
            {"prompt": "hello", "n_predict": 900, "stream": True}).encode())
        self.assertEqual(asked["n_predict"], 0)
        self.assertNotIn("max_tokens", asked)

    def test_does_not_stream_a_reply_it_throws_away(self):
        self.assertFalse(self.ask()["stream"])

    def test_asks_which_slot_served_it(self):
        self.assertTrue(self.ask()["verbose"])

    def test_leaves_the_prompt_alone(self):
        self.assertEqual(self.ask()["messages"],
                         [{"role": "user", "content": "hi"}])

    def test_ignores_a_body_that_is_not_a_request(self):
        self.assertIsNone(router.read_only(b"not json"))
        self.assertIsNone(router.read_only(b"[1,2]"))

    def test_naming_a_conversation_ignores_one_too(self):
        # A body that is not an object reached the public port and raised
        # before any status line went out.
        for body in (b"not json", b"[1,2]", b"null", b'"hi"', b"5", b"true"):
            self.assertIsNone(router.conversation_id(body), body)


class HandOff(unittest.TestCase):
    """Carry a conversation to the backend that should generate it.

    The prompt is read by the time this runs, so the slot holds everything but
    the answer. A save and a restore cost seconds; the gpu decodes five times
    faster."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True},
             {"name": "cpu", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True}],
            watch=False)
        self.gpu, self.cpu = self.pool.backends
        self.gpu.update(up=True, slots=1, n_ctx=150000,
                        slots_detail=[{"id": 0, "busy": False, "phase": "idle"}])
        self.cpu.update(up=True, slots=2, n_ctx=150000,
                        slots_detail=[{"id": 0, "busy": True, "phase": "reading"},
                                      {"id": 1, "busy": False, "phase": "idle"}])
        self.pool.pins["a"] = pin("cpu", slot=0, inflight=True)
        self.cpu["busy"] = 1
        self.removed = []
        # This class is about the move, so it turns it on whatever the shipped
        # default is. TheHandoffCanBeTurnedOff covers the default.
        self.was_on = SANDBOX.tuning
        SANDBOX.tuning = replace(self.was_on, handoff=True)
        self.pool.tuning = SANDBOX.tuning
        self.addCleanup(lambda: setattr(SANDBOX, "tuning", self.was_on))

    def mover(self, written=200_000_000, fail_on=None):
        return FakeLink(written=written, fail_on=fail_on)

    def go(self, post, alive=None):
        return with_link(self.pool, post).hand_off(
            "a", self.cpu, 1000, remove=self.removed.append, alive=alive)

    def test_saves_here_and_restores_there(self):
        post = linked(self.pool, self.mover())
        self.assertIs(self.go(post), self.gpu)
        (op1, be1, slot1, file1), (op2, be2, slot2, file2) = post.calls
        self.assertEqual((op1, be1, slot1), ("save", "cpu", 0))
        self.assertEqual((op2, be2), ("restore", "gpu"))
        self.assertEqual(file1, file2, "it restored something else")

    def test_the_pin_follows_the_conversation(self):
        self.go(self.mover())
        self.assertEqual(self.pool.pins["a"]["backend"], "gpu")
        self.assertEqual(self.pool.pins["a"]["slot"], 0)

    def test_a_short_prompt_is_carried_like_any_other(self):
        """park_min_tokens refuses to *store* a copy of a short prompt. This
        copy is transport: without it the turn cannot reach the instance that
        generates, and the prompt would be read there from nothing."""
        self.pool.pins["a"]["tokens"] = 1
        self.assertFalse(
            router.worth_keeping(self.pool.pins["a"], SANDBOX.tuning))
        post = self.mover()
        self.assertIs(self.go(post), self.gpu)
        self.assertEqual(post.ops()[0], "save")

    def test_the_slot_moves_from_one_backend_to_the_other(self):
        self.go(self.mover())
        self.assertEqual((self.cpu["busy"], self.gpu["busy"]), (0, 1))

    def test_the_copy_it_travels_on_is_its_own_park(self):
        """The reader is handed back before the wait, so the conversation has
        to be somewhere while it waits. On disk under its own name is a state
        the router already knows, so there is no separate carrier to spend."""
        post = linked(self.pool, self.mover())
        self.go(post)
        self.assertEqual(post.files(), ["a.park", "a.park"])
        self.assertEqual(self.removed, [], "its own copy was deleted")
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")

    def test_waits_for_the_gpu_rather_than_generating_on_the_cpu(self):
        """A cpu decodes five times slower and holds a slot that could be
        reading, so the queue for the gpu is the cheaper place to be."""
        self.gpu["busy"] = 1

        def free_it():
            time.sleep(0.2)
            with self.pool.cv:
                self.gpu["busy"] = 0
                self.pool.cv.notify_all()

        hand = threading.Thread(target=free_it, daemon=True)
        hand.start()
        self.assertIs(self.go(self.mover()), self.gpu)
        hand.join()

    def test_gives_up_the_wait_when_the_client_leaves(self):
        """Nobody is owed an answer, and the reader is already back.

        Nothing is held and nothing is lost: the conversation is on disk, so
        the turn the client sends again starts from what this one read."""
        self.gpu["busy"] = 1
        self.assertIsNone(self.go(self.mover(), alive=lambda: False))
        self.assertEqual(self.pool.pins["a"]["parked"], "a.park")
        self.assertEqual((self.cpu["busy"], self.gpu["busy"]), (0, 1))

    def test_the_reader_is_handed_back_before_the_wait_not_after(self):
        """The whole point. Three readers sat idle holding finished caches
        while the gpu spent seven minutes on one reply."""
        self.gpu["busy"] = 1
        freed = threading.Event()

        def watch():
            for _ in range(200):
                if self.cpu["busy"] == 0:
                    freed.set()
                    break
                time.sleep(0.01)
            with self.pool.cv:
                self.gpu["busy"] = 0
                self.pool.cv.notify_all()

        hand = threading.Thread(target=watch, daemon=True)
        hand.start()
        self.assertIs(self.go(self.mover()), self.gpu)
        hand.join()
        self.assertTrue(freed.is_set(),
                        "the reader was held through the wait for the gpu")

    def test_stays_when_the_gpu_cannot_serve_the_turn_at_all(self):
        """Down, draining or too small. Answering slowly beats not answering."""
        self.gpu["up"] = False
        post = linked(self.pool, self.mover())
        self.assertIs(self.go(post), self.cpu)
        self.assertEqual(post.calls, [])

    def test_stays_when_the_slot_is_unknown(self):
        self.pool.pins["a"]["slot"] = None
        post = linked(self.pool, self.mover())
        self.assertIs(self.go(post), self.cpu)
        self.assertEqual(post.calls, [])

    def test_stays_when_a_save_finds_nothing(self):
        self.assertIs(self.go(self.mover(written=900)), self.cpu)
        self.assertEqual(self.pool.pins["a"]["backend"], "cpu")
        self.assertEqual((self.cpu["busy"], self.gpu["busy"]), (1, 0))

    def test_stays_when_the_restore_fails(self):
        self.assertIs(self.go(self.mover(fail_on="restore")), self.cpu)
        self.assertEqual(self.pool.pins["a"]["backend"], "cpu")
        self.assertEqual((self.cpu["busy"], self.gpu["busy"]), (1, 0))

    def test_a_failed_save_leaves_nothing_behind(self):
        """And leaves the reader holding it. A save the backend refused says
        nothing about the slot: the cache is still sitting in it."""
        self.assertIs(self.go(self.mover(fail_on="save")), self.cpu)
        self.assertEqual(self.removed, [])
        self.assertIsNone(self.pool.pins["a"]["parked"])
        self.assertEqual(self.pool.pins["a"]["slot"], 0,
                         "it forgot a cache that never moved")
        self.assertEqual((self.cpu["busy"], self.gpu["busy"]), (1, 0))

    def test_it_shows_up_in_the_dashboard_feed(self):
        self.go(self.mover())
        self.assertEqual(self.pool.recent[0]["did"], "moved")


class WhatMustStayResident(unittest.TestCase):
    """The backend says what it mapped and what it reads lazily. The
    difference is what the page cache has to hold; the shard sizes on disk
    would say 175 GiB where 127 is the truth, and a mark drawn there would
    warn for ever."""

    MAPPED = "0.04.354.263 I load_tensors:   CPU_Mapped model buffer size = 47156.16 MiB\n"
    LAZY = ("0.04.307.094 I add: tensor per_layer_token_embd.weight (size = 51880 MiB) "
            "lazy read enabled\n")

    def test_reads_a_mapped_buffer_line(self):
        self.assertEqual(router.cache_event(self.MAPPED), ("mapped", 47156.16))

    def test_reads_the_lazy_tensor_line(self):
        self.assertEqual(router.cache_event(self.LAZY), ("lazy", 51880.0))

    def test_the_watch_sums_them_and_starts_over_on_restart(self):
        d = tempfile.mkdtemp()
        try:
            path = Path(d) / "cpu.log"
            path.write_text(self.LAZY + self.MAPPED + self.MAPPED)
            watch = router.CacheWatch(path)
            watch.poll()
            self.assertAlmostEqual(watch.stats["mapped_mib"], 94312.32, places=2)
            self.assertAlmostEqual(watch.stats["lazy_mib"], 51880.0)
            path.write_text(self.MAPPED)          # shorter: the backend restarted
            watch.poll()
            self.assertAlmostEqual(watch.stats["mapped_mib"], 47156.16, places=2)
            self.assertEqual(watch.stats["lazy_mib"], 0.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_resident_is_mapped_less_lazy(self):
        be = {"cache": {"mapped_mib": 182142.0, "lazy_mib": 51880.0}}
        self.assertEqual(router.resident_bytes(be), 130262 * 1024 ** 2)
        self.assertEqual(router.resident_bytes({"cache": {}}), 0, "nothing known yet")
        self.assertEqual(router.resident_bytes({}), 0)


class SpreadReadsAcrossNodes(unittest.TestCase):
    """Two reads on one socket share its cores. On two sockets they do not.

    Reading is compute bound and its memory is local to the socket, so a second
    read on a busy node roughly halves both. The same read on the other node
    runs at full speed."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True, "node": 0},
             {"name": "cpu", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True, "node": 1},
             {"name": "cpu0", "url": "http://cpu0", "pref": 2, "prefill": True, "generate": True, "node": 0},
             {"name": "cpu2", "url": "http://cpu2", "pref": 3, "prefill": True, "generate": True, "node": 1}],
            watch=False)
        self.gpu, self.cpu, self.cpu0, self.cpu2 = self.pool.backends
        self.gpu.update(up=True, slots=1, n_ctx=150000)
        self.cpu.update(up=True, slots=2, n_ctx=150000)
        self.cpu0.update(up=True, slots=2, n_ctx=150000)
        self.cpu2.update(up=True, slots=1, n_ctx=150000)

    def reading(self, be, how_many):
        be["slots_detail"] = [{"id": n, "busy": n < how_many,
                               "phase": "reading" if n < how_many else "idle"}
                              for n in range(be["slots"])]

    def test_reading_takes_the_opposite_order_to_generating(self):
        """pref says where to generate. A prompt goes to the last of those, so
        the instances kept for generating stay free to generate."""
        self.assertEqual(self.pool.acquire("a", 1000)[0]['name'], "cpu2")

    def test_the_second_read_crosses_to_the_other_node(self):
        """Not the other slot on node 1, which would share its cores."""
        self.reading(self.cpu2, 1)
        self.assertEqual(self.pool.acquire("b", 1000)[0]['name'], "cpu0")

    def test_the_generating_instance_is_read_on_last(self):
        self.reading(self.cpu2, 1)
        self.reading(self.cpu0, 1)
        self.assertEqual(self.pool.acquire("c", 1000)[0]['name'], "cpu")

    def test_a_node_reading_twice_loses_to_a_node_reading_once(self):
        self.reading(self.cpu, 2)          # node 1, two reads
        self.reading(self.cpu0, 1)         # node 0, one read
        self.assertEqual(self.pool.acquire("d", 1000)[0]['name'], "cpu0")

    def test_it_prefers_a_quiet_instance_over_a_second_slot_on_a_busy_one(self):
        """llama.cpp lets the first reading slot take the whole batch, so a
        second read on the same instance barely moves until the first ends."""
        self.reading(self.cpu, 1)          # node 1, on cpu
        self.reading(self.cpu0, 1)         # node 0, on cpu0
        self.assertEqual(self.pool.acquire("c", 1000)[0]['name'], "cpu2")

    def test_a_generating_slot_does_not_count_against_its_node(self):
        """Only reading contends for the cores a read needs. Node 1 keeps its
        turn while cpu generates, and loses it the moment cpu reads."""
        self.cpu["slots_detail"] = [{"id": 0, "busy": True, "phase": "generating"},
                                    {"id": 1, "busy": False, "phase": "idle"}]
        self.assertEqual(self.pool.acquire("e", 1000)[0]['name'], "cpu2")
        self.reading(self.cpu, 1)
        self.assertEqual(self.pool.acquire("f", 1000)[0]['name'], "cpu0")


class ReadingIsAllowedToTakeAsLongAsTheRequest(unittest.TestCase):
    """The read pass is part of a request, not a step with its own budget.

    A read on this machine runs for 30 to 40 minutes. Giving it less than the
    request is allowed means it times out as a matter of course, and the
    handoff quietly degrades to generating where the prompt was read, on
    exactly the long conversations it exists for."""

    def test_a_read_may_run_as_long_as_the_whole_request(self):
        self.assertGreaterEqual(SANDBOX.tuning.read_timeout, SANDBOX.tuning.forward_timeout)

    def test_a_queued_request_is_never_given_up_on(self):
        """A busy box makes a client wait, it does not refuse it. The wait ends
        when a slot frees or when the client leaves, and nothing else."""
        self.assertFalse(hasattr(router, "GIVE_UP"))


class NamedByPlace(unittest.TestCase):
    """A name says the type, the socket, and which instance on that socket.

    Sorting on the socket and the instance puts everything on one socket
    together, whatever type it is."""

    def key(self, name):
        return router.by_place(name)

    def test_one_socket_comes_before_the_other(self):
        self.assertLess(self.key("cpu0_0"), self.key("cpu1_0"))

    def test_the_gpu_sits_with_the_cpus_on_its_own_socket(self):
        order = sorted(["cpu1_0", "gpu0_0", "cpu1_1", "cpu0_0"], key=self.key)
        self.assertEqual(order, ["cpu0_0", "gpu0_0", "cpu1_0", "cpu1_1"])

    def test_instances_on_a_socket_keep_their_order(self):
        self.assertLess(self.key("cpu1_0"), self.key("cpu1_1"))

    def test_a_name_without_a_place_sorts_last(self):
        self.assertGreater(self.key("something"), self.key("cpu1_1"))


class WhichSlotReadIt(unittest.TestCase):
    """The router says which slot serves a read. It does not work it out.

    A reply only names its slot on some paths: the anthropic endpoint converts
    a body through a whitelist and drops anything not on it. Saying which slot
    works everywhere, and is a fact rather than an inference."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "cpu1_0", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True}],
            watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=2, n_ctx=150000,
                        slots_detail=[{"id": 0, "busy": True, "phase": "reading"},
                                      {"id": 1, "busy": False, "phase": "idle"}])

    def test_it_takes_a_slot_that_is_free(self):
        self.assertEqual(self.pool.pick_slot(self.cpu, "a"), 1)

    def test_it_keeps_the_slot_this_conversation_already_has(self):
        """Reading there extends the cache instead of starting again."""
        self.pool.pins["a"] = pin("cpu1_0", slot=0)
        self.assertEqual(self.pool.pick_slot(self.cpu, "a"), 0)

    def test_it_ignores_a_slot_held_on_another_backend(self):
        self.pool.pins["a"] = pin("elsewhere", slot=0)
        self.assertEqual(self.pool.pick_slot(self.cpu, "a"), 1)

    def test_it_falls_back_to_the_first_slot(self):
        """Nothing polled yet, or every slot busy. The backend queues it."""
        self.cpu["slots_detail"] = []
        self.assertEqual(self.pool.pick_slot(self.cpu, "a"), 0)

    def test_the_read_pass_names_no_slot_of_its_own(self):
        """Which slot to read into is the turn's to decide, and it adds the
        field itself. See TheReadNamesTheSlotTheTurnPicked."""
        asked = router.read_only(json.dumps(
            {"messages": [], "max_tokens": 900}).encode())
        self.assertNotIn("id_slot", asked)


class OnlyTheDashboardIsServed(unittest.TestCase):
    """The web directory holds the page and the things used to build it.

    The router serves that directory to any browser that reaches it, so only
    the page belongs on the wire. node_modules alone is 31 MB of someone
    else's code."""

    def test_the_page_and_its_parts_are_served(self):
        for path in ("index.html", "shell.js", "shell.css",
                     "lib/render.js", "views/registry.js",
                     "views/overview/overview.html", "components/x.js"):
            self.assertTrue(router.on_the_page(path), path)

    def test_the_toolchain_is_not(self):
        for path in ("package.json", "package-lock.json", "tsconfig.json",
                     "node_modules/typescript/package.json",
                     "node_modules/.package-lock.json",
                     "tools/check.mjs", "README.md"):
            self.assertFalse(router.on_the_page(path), path)

    def test_a_directory_asks_for_its_index(self):
        self.assertTrue(router.on_the_page(""))


class TheHandoffCanBeTurnedOff(unittest.TestCase):
    """Carrying a conversation is only worth it if the target can use it.

    A restored slot is not usable on every backend, so the move can cost a
    multi gigabyte transfer and still be followed by a full re-read. Until
    that is understood the move can be switched off, and a conversation
    generates where its prompt was read."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "gpu0_0", "url": "http://gpu", "pref": 0, "prefill": False, "generate": True},
             {"name": "cpu1_0", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True}],
            watch=False)
        self.gpu, self.cpu = self.pool.backends
        for be in self.pool.backends:
            be.update(up=True, slots=1, n_ctx=150000,
                      slots_detail=[{"id": 0, "busy": False, "phase": "idle"}])
        self.pool.pins["a"] = pin("cpu1_0", slot=0, inflight=True)
        self.was = SANDBOX.tuning

    def tearDown(self):
        SANDBOX.tuning = self.was

    def test_it_stays_put_when_the_move_is_off(self):
        self.pool.tuning = replace(SANDBOX.tuning, handoff=False)
        post = linked(self.pool, FakeLink())
        self.assertIs(self.pool.hand_off("a", self.cpu, 1000), self.cpu)
        self.assertEqual(post.calls, [], "it moved anyway")

    def test_it_moves_when_the_move_is_on(self):
        self.pool.tuning = replace(SANDBOX.tuning, handoff=True)

        post = linked(self.pool, FakeLink(written=200_000_000))
        self.assertIs(self.pool.hand_off("a", self.cpu, 1000), self.gpu)


class TheRightPingForTheProtocol(unittest.TestCase):
    """Each protocol has its own keep-alive, and a client only accepts its own.

    An SSE comment keeps an OpenAI stream alive. The anthropic protocol has a
    ping event instead, and a comment is not data to its parser: a client that
    sees only comments reports a stream that ended before any data arrived."""

    def test_the_anthropic_endpoint_gets_a_ping_event(self):
        beat = router.ping_for("/v1/messages")
        self.assertIn(b"event: ping", beat)
        self.assertIn(b'"type": "ping"', beat)
        self.assertTrue(beat.endswith(b"\n\n"))

    def test_the_openai_endpoints_get_a_comment(self):
        for path in ("/v1/chat/completions", "/v1/completions", "/completion"):
            self.assertTrue(router.ping_for(path).startswith(b":"), path)

    def test_a_ping_event_is_not_a_comment(self):
        self.assertNotEqual(router.ping_for("/v1/messages"),
                            router.ping_for("/v1/chat/completions"))


class AnAnthropicStreamOpensWithAMessage(unittest.TestCase):
    """The event a stream of this protocol has to begin with.

    ping may sit anywhere inside an anthropic stream, but a stream that has
    only pinged has begun no message and the client abandons it. The router
    opens the stream tens of minutes before the reply arrives, so what it
    opens with is all the client has to hold on to."""

    def opening(self, path="/v1/messages", **extra):
        body = json.dumps({"model": "qwen3.8-flash-next-mtp", "stream": True,
                           **extra}).encode()
        return router.opening_event(path, body)

    def test_the_anthropic_endpoint_opens_with_a_message_start(self):
        raw = self.opening()
        self.assertTrue(raw.startswith(b"event: message_start\n"))
        self.assertTrue(raw.endswith(b"\n\n"))
        name, data = router.read_event(raw)
        self.assertEqual(name, "message_start")
        self.assertEqual(data["type"], "message_start")

    def test_the_message_carries_what_the_protocol_asks_of_it(self):
        _, data = router.read_event(self.opening())
        message = data["message"]
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], [])
        self.assertIsNone(message["stop_reason"])
        self.assertIsNone(message["stop_sequence"])
        self.assertEqual(message["usage"], {"input_tokens": 0, "output_tokens": 0})

    def test_it_names_the_model_the_request_asked_for(self):
        _, data = router.read_event(self.opening())
        self.assertEqual(data["message"]["model"], "qwen3.8-flash-next-mtp")

    def test_every_message_gets_an_id_of_its_own(self):
        first, _ = router.read_event(self.opening()), None
        ids = {router.read_event(self.opening())[1]["message"]["id"]
               for _ in range(5)}
        self.assertEqual(len(ids), 5, "two messages were given the same id")
        for one in ids:
            self.assertTrue(one.startswith("msg_"), one)

    def test_the_openai_endpoints_open_with_nothing(self):
        """A comment is legal at the head of an OpenAI stream, so that one
        needs no opening event and must not be given one."""
        for path in ("/v1/chat/completions", "/v1/completions", "/completion"):
            self.assertEqual(self.opening(path), b"", path)

    def test_a_body_that_is_not_a_request_still_opens_a_stream(self):
        self.assertTrue(router.opening_event("/v1/messages", b"not json"))


class TheBackendsMessageStartIsTakenOut(unittest.TestCase):
    """Splice the backend's stream onto the one the router already opened.

    Two message_start events in a stream is not a message any parser will
    follow, so the backend's has to go. The only thing it carries that no
    later event does is the prompt token count, and a client sizes its
    context window with that, so it moves onto the message_delta."""

    START = (b'event: message_start\ndata: {"type": "message_start", "message": '
             b'{"id": "chatcmpl-x", "usage": {"input_tokens": 40, '
             b'"cache_read_input_tokens": 8, "output_tokens": 0}}}\n\n')
    DELTA = (b'event: content_block_delta\ndata: {"type": "content_block_delta", '
             b'"index": 0, "delta": {"type": "text_delta", "text": "hi"}}\n\n')
    END = (b'event: message_delta\ndata: {"type": "message_delta", "delta": '
           b'{"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}\n\n')
    STOP = b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    def through(self, splice, stream):
        return splice.feed(stream) + splice.tail()

    def test_the_first_message_start_is_dropped(self):
        splice = router.AnthropicSplice()
        out = self.through(splice, self.START + self.DELTA)
        self.assertNotIn(b"message_start", out)
        self.assertIn(b"content_block_delta", out)

    def test_everything_else_goes_through_byte_for_byte(self):
        splice = router.AnthropicSplice()
        self.assertEqual(self.through(splice, self.DELTA + self.STOP),
                         self.DELTA + self.STOP)

    def test_the_prompt_tokens_move_onto_the_message_delta(self):
        splice = router.AnthropicSplice()
        out = self.through(splice, self.START + self.END)
        _, data = router.read_event(out)
        self.assertEqual(data["usage"]["input_tokens"], 40)
        self.assertEqual(data["usage"]["cache_read_input_tokens"], 8)
        self.assertEqual(data["usage"]["output_tokens"], 3,
                         "the count of what was generated was overwritten")

    def test_a_part_event_waits_for_the_rest_of_itself(self):
        """A socket delivers whatever arrived, not whole events."""
        splice = router.AnthropicSplice()
        stream = self.START + self.DELTA + self.END + self.STOP
        out = b"".join(splice.feed(stream[n:n + 1]) for n in range(len(stream)))
        out += splice.tail()
        self.assertEqual(out.count(b"event: "), 3)
        self.assertNotIn(b"message_start", out)
        self.assertIn(b'"input_tokens": 40', out)

    def test_a_stream_with_no_message_start_is_left_alone(self):
        splice = router.AnthropicSplice()
        self.assertEqual(self.through(splice, self.END), self.END)

    def test_whatever_is_left_over_is_still_sent(self):
        """A backend that stops mid-event has still said something."""
        splice = router.AnthropicSplice()
        self.assertEqual(self.through(splice, b"data: half"), b"data: half")


class AnIdleClientIsStillThere(unittest.TestCase):
    """alive aborts a read when it answers False.

    A read is the only slow thing on this box, so a false positive would end
    every long one. A live client sends nothing for the whole read, and that
    silence must not read as a departure."""

    def peer(self):
        """A handler holding one end of a real socket, and the other end."""
        mine, theirs = socket.socketpair()
        self.addCleanup(mine.close)
        self.addCleanup(theirs.close)
        handler = router.Handler.__new__(router.Handler)
        handler.connection = mine
        return handler, theirs

    def test_a_client_that_sends_nothing_is_still_there(self):
        handler, theirs = self.peer()
        for _ in range(3):
            self.assertTrue(handler.alive())

    def test_a_client_that_closed_its_end_has_gone(self):
        handler, theirs = self.peer()
        theirs.close()
        self.assertFalse(handler.alive())

    def test_a_client_that_said_something_is_still_there(self):
        """A pipelined request arrives on the same socket. It is not a
        departure, and a byte waiting to be read must not read as one."""
        handler, theirs = self.peer()
        theirs.sendall(b"POST /v1/messages HTTP/1.1\r\n")
        self.assertTrue(handler.alive())

    def test_a_socket_that_is_gone_altogether_reads_as_gone(self):
        handler, theirs = self.peer()
        handler.connection.close()
        self.assertFalse(handler.alive())

    def test_a_client_on_a_high_descriptor_is_still_there(self):
        """select cannot be given a descriptor at or above FD_SETSIZE, 1024.

        It raises ValueError for one, on a socket that is open and whose
        client is waiting - and read as a departure that drops the request out
        from under them. Only reachable on a router holding that many
        descriptors at once, which is why it wants a test rather than luck."""
        handler, theirs = self.peer()
        high = 1500
        try:
            os.dup2(handler.connection.fileno(), high)
        except OSError as err:
            self.skipTest(f"no descriptor free at {high}: {err}")
        self.addCleanup(os.close, high)
        handler.connection = socket.socket(fileno=high)
        # detach, or closing this wrapper closes the descriptor addCleanup has
        self.addCleanup(handler.connection.detach)
        self.assertTrue(handler.alive())
        theirs.sendall(b"POST /v1/messages HTTP/1.1\r\n")
        self.assertTrue(handler.alive())


class AnErrorEndsTheStreamInItsOwnProtocol(unittest.TestCase):
    """A stream that has begun cannot be answered with a status.

    The router opens the stream before it knows whether it can serve the
    request at all, so whatever goes wrong after that has to be said inside
    the stream. A bare data line is nothing an anthropic parser can place,
    now that a message_start opened the stream in front of it."""

    def sent(self, path):
        wrote = []

        class Quiet(router.Handler):
            def __init__(inner):
                inner.path = path

            def _chunk(inner, data):
                wrote.append(data)

            @property
            def wfile(inner):
                return io.BytesIO()

        router.Handler._say_and_end(Quiet(), "cpu0_0: it went wrong")
        return b"".join(wrote)

    def test_the_anthropic_stream_ends_in_an_error_event(self):
        name, data = router.read_event(self.sent("/v1/messages"))
        self.assertEqual(name, "error")
        self.assertEqual(data["type"], "error")
        self.assertIn("it went wrong", data["error"]["message"])

    def test_the_openai_stream_keeps_the_line_it_had(self):
        raw = self.sent("/v1/chat/completions")
        self.assertTrue(raw.startswith(b"data: "), raw)
        self.assertIn(b"it went wrong", raw)


class EveryRefusalIsWrittenDown(unittest.TestCase):
    """An error sent to a client must leave a trace on this side of it.

    A 502 that logs nothing cannot be diagnosed from the router's own log, and
    the only person who can see it is the one it was sent to."""

    def test_an_error_says_what_it_was_and_where(self):
        said = []

        class Loud(router.Handler):
            def __init__(inner):
                inner.path = "/v1/messages"
                inner.command = "POST"

            def _send(inner, code, payload, content_type="application/json"):
                pass

        loud = Loud()
        router.Handler._error(loud, 502, "cpu1_1: it went wrong",
                              say=said.append)
        self.assertEqual(len(said), 1)
        self.assertIn("502", said[0])
        self.assertIn("cpu1_1: it went wrong", said[0])
        self.assertIn("/v1/messages", said[0])


class WhatTheCachesDecided(unittest.TestCase):
    """The router must say what each slot holds and what it did about it.

    Without this the only way to ask why an opening was not built is to read
    the code and guess which of three branches it stopped at."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "cpu1_1", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True}],
            watch=False)
        self.cpu = self.pool.backends[0]
        self.cpu.update(up=True, slots=1, n_ctx=150000,
                        slots_detail=[{"id": 0, "busy": False, "phase": "idle"}])

    def test_it_reports_what_each_slot_holds(self):
        self.pool.pins["a"] = pin("cpu1_1", slot=0)
        self.pool.note_holds("a", self.cpu, [(0, "k1"), (1, "k2"), (2, "k3")])
        held = self.pool.status()["slots_hold"]
        self.assertEqual(held, [{"backend": "cpu1_1", "slot": 0, "cuts": 3, "through": 2}])

    def test_it_reports_nothing_for_a_slot_that_holds_nothing(self):
        self.assertEqual(self.pool.status()["slots_hold"], [])

    def test_it_records_why_an_opening_was_not_cut_deeper(self):
        """The interesting case: a request shares only the system prompt."""
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.pins["new"] = pin("cpu1_1", slot=None, inflight=True)
        post = linked(self.pool, FakeLink())
        self.pool.warm_prefix("new", [(0, "k1"), (1, "k2")], [], "", [],
                              self.cpu, 1, "/v1/chat/completions")
        chose = self.pool.choices.get("new")
        self.assertEqual(chose["cuts"], 2)
        self.assertEqual(chose["stored"], 0, "it loaded the system prompt")
        self.assertIsNone(chose["shared"], "nothing else held a deeper cut")

    def test_it_records_a_deeper_cut_when_a_slot_holds_one(self):
        self.pool.openings["k1"] = "base-k1.park"
        self.pool.holds[("cpu1_1", 0)] = {"k1", "k2"}
        self.pool.pins["new"] = pin("cpu1_1", slot=None, inflight=True)
        # The plan here is "load", which restores through the link. Without a
        # stand-in that is a real POST to http://cpu, swallowed by
        # _load_prefix's except, so the case passed while asking a live
        # backend to overwrite one of its slots.
        linked(self.pool, FakeLink())
        self.pool.warm_prefix("new", [(0, "k1"), (1, "k2")], [{}, {}], "", [],
                              self.cpu, 1, "/v1/chat/completions")
        self.assertEqual(self.pool.choices["new"]["shared"], 1)


class WhatEachBackendWasStartedWith(unittest.TestCase):
    """The settings a backend runs with belong on the page.

    Two instances running different settings is how one is compared against
    the other, and it cannot be read off /props: llama-server does not report
    n_batch there. It prints it once at startup instead."""

    HEAD = """
0.04.100.012 I llama_context: n_ctx                 = 150016
0.04.100.014 I llama_context: n_batch               = 2048
0.04.100.015 I llama_context: n_ubatch              = 512
0.04.100.016 I llama_context: kv_unified            = false
0.09.779.535 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 150000
"""

    def test_it_reads_the_settings_out_of_the_log(self):
        seen = router.read_config(self.HEAD.splitlines())
        self.assertEqual(seen["n_batch"], 2048)
        self.assertEqual(seen["n_ubatch"], 512)
        self.assertEqual(seen["n_ctx"], 150016)
        self.assertEqual(seen["kv_unified"], False)

    def test_it_reports_nothing_it_did_not_find(self):
        self.assertEqual(router.read_config(["nothing here"]), {})

    def test_it_finds_a_setting_printed_late(self):
        """Loading prints hundreds of lines before the server settings."""
        lines = ["filler"] * 900 + ["I srv load_model: initializing, n_slots = 2"]
        self.assertEqual(router.read_config(lines)["n_slots"], 2)

    def test_it_takes_the_first_of_each(self):
        """A restart appends to the same file, so later runs must not win."""
        lines = self.HEAD.splitlines() + ["I llama_context: n_batch = 512"]
        self.assertEqual(router.read_config(lines)["n_batch"], 2048)

    def test_it_reads_again_when_the_log_is_a_new_file(self):
        """restart-backend.sh moves the log aside; the backend makes its own.

        The size cannot say a restart happened. A new log that has already
        grown past the offset the last read recorded is both larger and
        entirely different, which is how three backends went on being
        reported with the batch size they no longer ran."""
        with tempfile.TemporaryDirectory() as where:
            run = pathlib.Path(where)
            was, SANDBOX.store = SANDBOX.store, router.Store(run)
            try:
                log = run / "cpu.log"
                log.write_text(self.HEAD)
                # watch=False. Pool's second parameter is `watch`, not a
                # poster, and FakeLink() is truthy: this started the real
                # _watch and _builder threads and never stopped them, so any
                # run that put another module after this one failed in
                # test_end_to_end's "pool threads outlived the test" check.
                pool = make_pool([{"name": "cpu", "url": "http://cpu",
                                     "pref": 0}], watch=False)
                be = pool.backends[0]
                self.assertEqual(pool.read_settings(be)["n_batch"], 2048)

                log.rename(run / "cpu.log.prev")
                fresh = self.HEAD.replace("n_batch               = 2048",
                                          "n_batch               = 512")
                log.write_text(fresh + "filler\n" * 200)   # larger than before
                self.assertGreater(log.stat().st_size, len(self.HEAD))
                self.assertEqual(pool.read_settings(be)["n_batch"], 512)
            finally:
                SANDBOX.store = was


class RatesCanBeReset(unittest.TestCase):
    """A lifetime average carries every run since the backend started.

    After a change, the figure that matters is the one since the change. The
    counters are cumulative, so a baseline subtracted from them gives that."""

    def setUp(self):
        self.pool = make_pool(
            [{"name": "cpu1_1", "url": "http://cpu", "pref": 1, "prefill": True, "generate": True}],
            watch=False)
        self.cpu = self.pool.backends[0]

    def test_nothing_is_subtracted_before_a_reset(self):
        counters = {"prompt_tokens_total": 900.0, "prompt_seconds_total": 30.0}
        self.assertEqual(self.pool.since_reset(self.cpu, counters), counters)

    def test_a_reset_makes_the_counters_start_from_now(self):
        self.pool.reset_rates({"prompt_tokens_total": 900.0,
                               "prompt_seconds_total": 30.0})
        later = {"prompt_tokens_total": 1500.0, "prompt_seconds_total": 50.0}
        seen = self.pool.since_reset(self.cpu, later)
        self.assertEqual(seen["prompt_tokens_total"], 600.0)
        self.assertEqual(seen["prompt_seconds_total"], 20.0)

    def test_a_counter_that_went_backwards_starts_again(self):
        """The backend restarted, so its counters did too."""
        self.pool.reset_rates({"prompt_tokens_total": 900.0})
        seen = self.pool.since_reset(self.cpu, {"prompt_tokens_total": 10.0})
        self.assertEqual(seen["prompt_tokens_total"], 10.0)

    def test_it_says_when_the_count_started(self):
        self.assertIsNone(self.pool.rates_since)
        self.pool.reset_rates({})
        self.assertIsNotNone(self.pool.rates_since)


class ARateNeedsEnoughToDivideBy(unittest.TestCase):
    """Just after a reset the elapsed time is near zero.

    A token or two against a fraction of a second is not a rate, it is an
    artefact. Report nothing until there is enough to divide by."""

    def test_a_fraction_of_a_second_is_not_a_rate(self):
        self.assertEqual(router.per_second(12, 0.004), 0)

    def test_enough_time_gives_a_rate(self):
        self.assertEqual(router.per_second(120, 4.0), 30.0)

    def test_no_time_at_all_gives_nothing(self):
        self.assertEqual(router.per_second(5, 0), 0)


class HowFarInASlotHolds(unittest.TestCase):
    """The dashboard says how far in another conversation could start from
    what a slot holds, in messages. -1 is the system prompt alone."""

    def test_status_names_the_deepest_cut_each_slot_holds(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        cpu = pool.backends[0]
        pool._take(cpu, "conv1", tokens=500)
        pool.note_slot("conv1", 2)
        pool.note_holds("conv1", cpu, [(-1, "sys"), (0, "m0"), (3, "m3")])
        held = pool.status()["slots_hold"]
        self.assertEqual(held, [{"backend": "cpu", "slot": 2, "cuts": 3, "through": 3}])

    def test_a_slot_holding_only_the_system_prompt_says_so(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}], watch=False)
        cpu = pool.backends[0]
        pool._take(cpu, "conv1", tokens=500)
        pool.note_slot("conv1", 0)
        pool.note_holds("conv1", cpu, [(-1, "sys")])
        self.assertEqual(pool.status()["slots_hold"][0]["through"], -1)


class TheDiskReportDrawsTheBudgetTheSweepsEnforce(unittest.TestCase):
    """PARK_BUDGET_GB and BLOCK_BUDGET_GB are the operator's to set, and the
    sweeps spend against them. A report drawn from the built-in defaults
    instead shows a bar far over budget while the router evicts nothing."""

    def setUp(self):
        self.tuning = replace(SANDBOX.tuning,
                              park_budget=1000 * 1024 ** 3,
                              block_budget=500 * 1024 ** 3)

    def test_the_copies_budget_is_the_pools_own(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                         tuning=self.tuning, watch=False)
        self.assertEqual(pool.status()["disk"]["copies"]["budget"],
                         1000 * 1024 ** 3)

    def test_the_openings_budget_is_the_pools_own(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                         tuning=self.tuning, watch=False)
        self.assertEqual(pool.status()["disk"]["openings"]["budget"],
                         500 * 1024 ** 3)


if __name__ == "__main__":
    unittest.main()
