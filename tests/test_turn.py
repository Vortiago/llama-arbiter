"""Tests for one turn: the steps from a prompt to a reply.

A turn runs against a Client and a Link, so these need no socket. The Client
is whoever asked; the Link is the one way to a backend. Both are stood in for
here, which is what lets a whole turn run in this process.
"""
import atexit
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import router


class SANDBOX:
    """What this file's turns are wired to. See the same class in
    test_migration.py: a Pool is handed its store, so nothing here can reach
    the checkout's own run/slots."""

    store = router.Store(tempfile.mkdtemp(prefix="router-turn-"))
    tuning = router.Tuning()
    events = router.EventLog(on=False)

# Nothing else deletes this. The path is read now rather than at exit, because
# a case may point SANDBOX.store somewhere else and put it back.
atexit.register(shutil.rmtree, SANDBOX.store.run, ignore_errors=True)


class FakeClient:
    """Whoever asked, without a socket. Records what the turn told it.

    `alive` is what the router watches while a backend reads: a turn that
    outlives its client must stop and park what it got.
    """

    kind = "test"
    went = "closed its end"

    def __init__(self, alive=True):
        self._alive = alive
        self.asked = threading.Event()  # set when the router first asks
        self.opened = None             # the opening event, once a stream began
        self.relayed = []              # (backend name, conversation)
        self.failed = []               # (code, message)
        self.sent = None               # what relay was given to forward
        self.did = []                  # every operation, in order

    def alive(self):
        self.asked.set()
        return self._alive

    def open(self, opening):
        self.opened = opening
        self.did.append("open")

    def settle(self):
        self.did.append("settle")

    def relay(self, be, body, conv):
        self.relayed.append((be["name"], conv))
        self.did.append("relay")
        self.sent = body               # the body the backend was asked with

    def answer(self, payload):
        self.answered = payload
        self.did.append("answer")

    def fail(self, code, message):
        self.failed.append((code, message))
        self.did.append("fail")


class TurnLink:
    """The one way to a backend, answering without a socket.

    Records what was asked of it as (operation, backend name), so a test says
    which operation ran rather than matching a URL. `reading` is what a
    prefill pass answers: an exception here is raised instead, which is how a
    client that goes mid-read reaches the turn.
    """

    def __init__(self, reading=None, written=1 << 30):
        self.calls = []
        self.reading = reading
        self.written = written          # what a save reports it wrote

    def ops(self):
        """Just the operation names, in order."""
        return [call[0] for call in self.calls]

    def asked(self):
        """What the one read pass was sent."""
        return next(call[2] for call in self.calls if call[0] == "read")

    def save(self, be, slot, name, timeout=None):
        self.calls.append(("save", be["name"], name))
        return {"n_written": self.written}

    def restore(self, be, slot, name, timeout=None):
        self.calls.append(("restore", be["name"], name))
        return {"id_slot": slot, "n_restored": 3}

    def work(self, be, path, payload, alive, timeout=None):
        self.calls.append(("read", be["name"], payload))
        if isinstance(self.reading, BaseException):
            raise self.reading
        # llama-server reports what it read and what the cache saved it.
        return self.reading or {"timings": {"prompt_n": 12, "cache_n": 4}}


def make_pool(backends, **kw):
    """A Pool wired to the sandbox."""
    kw.setdefault("store", SANDBOX.store)
    kw.setdefault("tuning", SANDBOX.tuning)
    kw.setdefault("events", SANDBOX.events)
    kw.setdefault("link", TurnLink())
    kw.setdefault("watch", False)
    return router.Pool(backends, **kw)


def one_backend(n_ctx=150000, **kw):
    """A pool of one backend that is up. Pool.__init__ rebuilds every backend
    with n_ctx 0 and up False, which the watcher fills in on a live run."""
    pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}], **kw)
    pool.backends[0].update(up=True, n_ctx=n_ctx)
    return pool


def reader_only():
    """A pool whose one up backend reads but may not generate. Its generator
    is down, so a turn read here waits for one."""
    pool = make_pool([
        {"name": "cpu", "url": "http://cpu", "pref": 0, "generate": False},
        {"name": "gpu", "url": "http://gpu", "pref": 1, "prefill": False}])
    pool.backends[0].update(up=True, n_ctx=150000)
    pool.backends[1].update(n_ctx=150000)
    return pool


def with_system(rules, said="hello"):
    """A chat request that opens with a system prompt, so it has a cut for
    an opening to be parked at."""
    return json.dumps({"messages": [{"role": "system", "content": rules},
                                    {"role": "user", "content": said}]}).encode()


def parked_copy(name="c1.park"):
    """A pin whose cache is on disk and in no slot."""
    return {"backend": "cpu", "slot": None, "parked": name, "bytes": 1 << 20,
            "turns": 1, "parked_turn": 1, "inflight": False,
            "parking": False, "last": 0.0,
            "tokens": 1000, "moved": None}


def prompt(chars, **extra):
    """A chat request whose prompt is about this many characters long. `extra`
    adds top level fields, such as `stream=True`."""
    return json.dumps({**extra,
                       "messages": [{"role": "user",
                                     "content": "x" * chars}]}).encode()


class ATurnRefusesAPromptNoBackendCanHold(unittest.TestCase):
    """A prompt past the context window makes llama-server drop the whole
    slot it was read into. The router answers for itself instead, before any
    backend has seen the prompt."""

    def test_a_prompt_larger_than_the_largest_backend_is_refused(self):
        pool = one_backend(n_ctx=100)
        client = FakeClient()
        pool.turn(router.Ask("/v1/chat/completions", prompt(200_000), None),
                  client)
        self.assertEqual([code for code, _ in client.failed], [413])


class ATypedPlanIsSizedByItsLargestQuestion(unittest.TestCase):
    """A typed plan asks every question against the slot the read pass filled.

    The handler rebuilds the turn body from the first question, and the size
    check read only that. A later, longer question could then exceed n_ctx
    mid-plan: the backend drops the whole slot and every answer is lost. The
    plan has to be refused up front instead."""

    def test_a_later_long_question_refuses_the_plan_up_front(self):
        pool = one_backend(n_ctx=2000)
        raw = json.dumps({
            "state": "x" * 40,
            "questions": {
                "small": {"type": "noul", "instructions": "one word"},
                "big": {"type": "noul", "instructions": "y" * 20000},
            },
        }).encode()
        plan = router.systemone_plan(raw)
        body = router.systemone_body(plan, plan["questions"][0])
        client = FakeClient()

        pool.turn(router.Ask(router.SYSTEMONE, body, None, plan, body), client)

        self.assertEqual([code for code, _ in client.failed], [413])
        self.assertEqual(pool.link.ops(), [],
                         "the link saw work despite the refusal")


class ATurnNeedsABackendThatReads(unittest.TestCase):
    """The largest prompt any prefiller will read is 0 while none is up. A
    turn that went on from there would wait in acquire for a backend that
    may never come."""

    def test_a_turn_with_no_backend_up_is_refused(self):
        pool = one_backend()
        pool.backends[0]["up"] = False
        client = FakeClient()
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), None), client)
        self.assertEqual([code for code, _ in client.failed], [503])


class ATurnThatLosesItsClientTakesNoSlot(unittest.TestCase):
    """claim_turn has no deadline: it waits for the turn ahead of it. A
    client that goes while it waits must leave the pool as it found it, or a
    slot stays busy on a reply nobody is there to read."""

    def test_a_client_that_goes_while_it_queues_takes_no_slot(self):
        pool = one_backend()
        ahead = pool.begin_wait("conv", 10)
        pool.claim_turn("conv", ahead)          # the turn ahead holds it
        pool.end_wait(ahead)

        client = FakeClient(alive=False)
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "conv"),
                  client)

        self.assertEqual(pool.backends[0]["busy"], 0)
        self.assertEqual([code for code, _ in client.failed], [503])

    def test_the_turn_ahead_keeps_the_conversation(self):
        """finish_turn matches on the ticket, so a turn that gave up waiting
        must not end the turn that is running."""
        pool = one_backend()
        ahead = pool.begin_wait("conv", 10)
        pool.claim_turn("conv", ahead)
        pool.end_wait(ahead)

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "conv"),
                  FakeClient(alive=False))

        self.assertEqual(pool.turns.get("conv"), ahead)


class ATurnRunsToTheEndAndPutsEverythingBack(unittest.TestCase):
    """Every way out of a turn runs the same ending. claim_turn has no
    deadline, so a claim left behind stops the conversation for good, and a
    slot never released is one the pool stops offering."""

    def test_a_cold_turn_reads_on_the_backend_and_relays_the_reply(self):
        pool = one_backend()
        client = FakeClient()
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual(pool.link.ops(), ["read"])
        self.assertEqual(client.relayed, [("cpu", "c1")])

    def test_the_slot_and_the_conversation_are_given_back(self):
        pool = one_backend()
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertEqual(pool.backends[0]["busy"], 0)
        self.assertNotIn("c1", pool.turns)

    def test_the_turn_is_written_down(self):
        pool = one_backend()
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertEqual(len(pool.recent_requests), 1)
        self.assertEqual(pool.recent_requests[0]["started"], "cold")


class ATurnWritesDownWhatTheClientSent(unittest.TestCase):
    """A capture is for comparing two turns offline. It is written before
    anything is changed, so it holds what the client sent rather than what
    the router made of it."""

    def test_the_body_reaches_the_capture_directory(self):
        room = Path(tempfile.mkdtemp(prefix="router-capture-"))
        self.addCleanup(shutil.rmtree, room, ignore_errors=True)
        pool = one_backend(capture_dir=room)
        body = prompt(10)

        pool.turn(router.Ask("/v1/chat/completions", body, "c1"), FakeClient())

        self.assertEqual([p.read_bytes() for p in room.glob("*.json")], [body])


class ATurnHoistsALateSystemMessage(unittest.TestCase):
    """The template refuses a system message that is not at the front. Claude
    Code ends every turn with a token counter as a system message, so the
    router turns a late one into a user message before the backend sees it."""

    def test_a_late_system_message_does_not_reach_the_backend(self):
        pool = one_backend()
        late = json.dumps({"messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "tokens: 12"}]}).encode()
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", late, "c1"), client)

        roles = [m["role"] for m in json.loads(client.sent)["messages"]]
        self.assertEqual(roles, ["user", "user"])

    def test_a_prompt_already_in_order_is_passed_on_untouched(self):
        pool = one_backend()
        body = prompt(10)
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", body, "c1"), client)

        self.assertIs(client.sent, body)


class ACopyBeatsAnOpening(unittest.TestCase):
    """A conversation's own copy holds its whole prompt; an opening holds
    only the start of it. Loading an opening over a recalled cache would
    throw the longer one away."""

    def test_a_recalled_copy_stops_an_opening_loading_over_it(self):
        pool = one_backend()
        body = with_system("rules " * 500)
        cuts = router.prompt_cuts(body)[0]
        self.assertTrue(cuts, "the fixture needs a cut to share")
        pool.openings[cuts[0][1]] = "base-x.park"
        pool.pins["c1"] = parked_copy()

        pool.turn(router.Ask("/v1/chat/completions", body, "c1"), FakeClient())

        self.assertEqual(pool.link.ops(), ["restore", "read"])
        self.assertEqual(pool.recent_requests[0]["started"], "recalled")

    def test_a_turn_with_no_copy_loads_the_opening_it_shares(self):
        pool = one_backend()
        body = with_system("rules " * 500)
        cuts = router.prompt_cuts(body)[0]
        pool.openings[cuts[0][1]] = "base-x.park"

        pool.turn(router.Ask("/v1/chat/completions", body, "c1"), FakeClient())

        self.assertEqual(pool.link.ops(), ["restore", "read"])
        self.assertEqual(pool.recent_requests[0]["started"], "saved prompt")


class AWaitForAnOpeningWatchesTheClient(unittest.TestCase):
    """One turn reads a shared opening and the rest wait for it. That wait
    holds a prefill slot for up to build_patience, half an hour, so a client
    that has gone has to end it."""

    def test_a_client_that_has_gone_does_not_wait_out_build_patience(self):
        patience = 5.0
        pool = one_backend(tuning=replace(SANDBOX.tuning,
                                          build_patience=patience))
        body = with_system("You follow these rules. " * 400)
        base = router.prompt_cuts(body, pool.tuning)[0][0]
        pool.building[base[1]] = time.time()   # another turn is reading it

        began = time.monotonic()
        pool.turn(router.Ask("/v1/chat/completions", body, "c1"),
                  FakeClient(alive=False))
        waited = time.monotonic() - began

        self.assertLess(waited, patience / 2)


class AClientThatGoesMidReadLeavesItsCacheOnDisk(unittest.TestCase):
    """The read is the expensive half of a turn. What it got through is
    parked, so the next turn of that conversation starts from it instead of
    reading the same prompt again."""

    def test_what_the_read_got_through_is_parked(self):
        pool = one_backend(link=TurnLink(reading=router.Gone("the client left")))

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertEqual(pool.link.ops(), ["read", "save"])

    def test_a_client_that_left_is_told_nothing(self):
        pool = one_backend(link=TurnLink(reading=router.Gone("the client left")))
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual(client.failed, [])
        self.assertEqual(client.relayed, [])

    def test_the_slot_and_the_conversation_are_given_back(self):
        pool = one_backend(link=TurnLink(reading=router.Gone("the client left")))

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertEqual(pool.backends[0]["busy"], 0)
        self.assertNotIn("c1", pool.turns)


class ATurnWithNowhereToGenerateStopsAfterTheRead(unittest.TestCase):
    """A backend set not to generate holds the prompt it read while it waits
    for a generator. A client that goes during that wait must not leave the
    prefill slot held: the wait has no deadline."""

    def test_nothing_is_held_once_the_client_has_gone(self):
        pool = reader_only()
        client = FakeClient(alive=False)

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual(client.relayed, [])
        self.assertNotIn("c1", pool.turns)


class APrefillSlotIsNeverLeftHeld(unittest.TestCase):
    """acquire counts a slot the pool never gets back busy for the life of
    the process. hand_off gives up in two places: the later one releases the
    prefiller first and returns None, the earlier one raises Gone."""

    def test_giving_up_before_the_handoff_says_the_client_went(self):
        pool = reader_only()
        pool.acquire("c1", 10)             # the prefiller is held from here
        self.assertEqual(pool.backends[0]["busy"], 1)

        with self.assertRaises(router.Gone):
            pool.hand_off("c1", pool.backends[0], 10, alive=lambda: False)

    def test_a_turn_that_loses_its_client_before_the_handoff_holds_nothing(self):
        pool = reader_only()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient(alive=False))

        self.assertEqual(pool.backends[0]["busy"], 0)
        # `inflight` is the other half of holding a slot. Left set, the pin is
        # skipped by ensure_parked, park_all and the drain report, and no
        # other conversation can be given that slot.
        self.assertFalse(pool.pins["c1"]["inflight"])

    def test_the_read_it_finished_is_kept(self):
        """The read is the expensive half, and it completed. A turn that ends
        here has to leave it on disk, the same as a client that goes during
        the read itself."""
        pool = reader_only()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient(alive=False))

        self.assertEqual(pool.link.ops(), ["read", "save"])


class TheHandOffOutlivesItsGenerator(unittest.TestCase):
    """The wait for a generator slot is re-entered with whatever `generator`
    found, which is None once the last one has gone. A raise there is paid for
    twice: hand_off has released the prefiller, so the ending releases it
    again and the count falls below zero with no floor under it."""

    def test_no_generator_left_is_a_wait_and_not_a_raise(self):
        pool = make_pool([
            {"name": "cpu", "url": "http://cpu", "pref": 0, "generate": False},
            {"name": "gpu", "url": "http://gpu", "pref": 1, "prefill": False}])
        cpu, gpu = pool.backends
        cpu.update(up=True, n_ctx=150000, busy=1)   # it holds the read
        gpu.update(up=True, n_ctx=150000, busy=1)   # its one slot is taken
        pool.pins["c1"] = parked_copy()
        pool.pins["c1"]["slot"] = 0
        asked = []

        def alive():
            """The client, and the generator going away inside the wait."""
            asked.append(None)
            if len(asked) == 1:
                gpu["up"] = False
            return len(asked) < 3          # and then the client gives up

        self.assertIsNone(pool.hand_off("c1", cpu, 10, alive=alive))
        # Once, by hand_off. The turn's ending releases what it still holds,
        # and past a None it holds nothing.
        self.assertEqual(cpu["busy"], 0)


class ATurnWithNoFreeSlotWaits(unittest.TestCase):
    """A busy box is a queue, not a refusal.

    acquire answers the backend and the slot together, under one hold of the
    lock. Asked as two questions it could admit a turn and then find every
    slot taken, and the turn was refused 503 where it should have queued: a
    copy starting on the last free slot did exactly that."""

    def test_it_queues_and_is_served_once_the_slot_comes_back(self):
        pool = one_backend()
        pool.backends[0].update(slots=1,
                                slots_detail=[{"id": 0, "busy": False}])
        pool.pins["first"] = parked_copy()
        # `inflight` with `busy` at 0 is the window hand_off opens: the source
        # gives its count back when the prompt is parked, and the pin holds
        # the slot until the restore lands on the generator.
        pool.pins["first"].update(slot=0, using=0, inflight=True)
        client = FakeClient()

        turn = threading.Thread(
            target=pool.turn,
            args=(router.Ask("/v1/chat/completions", prompt(10), "c1"), client),
            daemon=True)
        turn.start()
        self.addCleanup(turn.join, 10)
        # acquire asks the client once per pass of its wait, and nothing else
        # asks here: the turn has looked for a slot and found none.
        self.assertTrue(client.asked.wait(10), "the turn never queued")
        self.assertEqual(client.failed, [], "it was refused instead")

        with pool.cv:                          # the turn ahead finishes
            pool.pins["first"]["inflight"] = False
            pool.cv.notify_all()
        turn.join(10)

        self.assertEqual(client.relayed, [("cpu", "c1")])
        self.assertEqual(pool.backends[0]["busy"], 0)


class ACacheIsWrittenToDiskOnce(unittest.TestCase):
    """Two saves of one conversation write the same file name from the same
    slot. The prefill slot is released before the partial copy is parked, so
    the turn taking that slot next meets the turn still on its way out."""

    def test_a_copy_already_being_written_is_not_written_twice(self):
        pool = one_backend()
        pool.pins["c1"] = parked_copy()
        pool.pins["c1"]["slot"] = 0
        pool.pins["c1"]["parked_turn"] = 0     # the copy is of an older turn
        pool.pins["c1"]["parking"] = True      # a copy is running on it now

        self.assertFalse(pool.park_partial("c1", pool.backends[0], 0))
        self.assertEqual(pool.link.ops(), [])

    def test_a_copy_on_the_worker_is_not_queued_over_a_running_save(self):
        """The same meeting, one step later. The turn's ending releases the
        backend, and park_all can take the record in
        the two lines before the copy is queued."""
        pool = one_backend()
        pool.backends[0]["prefill"] = False    # it cannot read the next turn
        pool.pins["c1"] = parked_copy()
        pool.pins["c1"]["slot"] = 0
        pool.pins["c1"]["parking"] = True      # a copy is running on it now

        self.assertFalse(pool.park_later(pool.backends[0], "c1", "t1"))
        self.assertTrue(pool.park_jobs.empty())
        self.assertIsNone(pool.parker)         # no worker was even started


class ASlotHoldsOneConversation(unittest.TestCase):
    """A turn that takes a slot overwrites what was in it. A pin that still
    claimed the slot was saved again before every later turn on that
    backend, with another conversation's cache in it: on koishi, 3861
    copies of 229 conversations in one day, 3.3 TB written."""

    def resident(self, pool):
        pool.pins["old"] = parked_copy("old.park")
        pool.pins["old"].update(slot=0, tokens=50_000)   # worth a copy

    def test_the_turn_that_takes_a_slot_ends_the_old_claim(self):
        pool = one_backend()
        self.resident(pool)
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertIsNone(pool.pins["old"]["slot"])

    def test_a_claim_that_ended_is_not_saved_again(self):
        pool = one_backend()
        self.resident(pool)
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())
        pool.pins["old"]["parked"] = None      # the budget took its copy
        pool.link.calls.clear()
        pool.turn(router.Ask("/v1/chat/completions", prompt(20), "c2"),
                  FakeClient())

        self.assertNotIn(("save", "cpu", "old.park"), pool.link.calls)


class AStreamSaysNothingAfterItsLastWord(unittest.TestCase):
    """A keep-alive chunk written after the stream's terminator stays on the
    connection, and the next request on that connection reads it as its own
    reply. So every way out stops the keep-alive before it says anything
    final."""

    def test_a_streamed_turn_opens_before_it_asks_for_a_slot(self):
        pool = one_backend()
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10, stream=True), "c1"),
                  client)

        self.assertIsNotNone(client.opened, "the stream never opened")
        self.assertEqual(client.did, ["open", "settle", "relay", "settle"])

    def test_a_backend_that_stops_mid_read_stops_the_keep_alive_first(self):
        pool = one_backend(link=TurnLink(reading=OSError("backend went away")))
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10, stream=True), "c1"),
                  client)

        self.assertEqual([code for code, _ in client.failed], [502])
        self.assertLess(client.did.index("settle"), client.did.index("fail"),
                        "the keep-alive was still running when the stream ended")


class TheReadNamesTheSlotTheTurnPicked(unittest.TestCase):
    """A reply names the slot it used only on some paths, so the router says
    which slot to read into rather than asking afterwards."""

    def test_the_read_carries_the_slot(self):
        pool = one_backend()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertEqual(pool.link.asked()["id_slot"], 0)

    def test_the_read_asks_for_no_tokens(self):
        """A generated token lands in the slot, and a restored slot has no
        checkpoint to rewind to, so the next turn would read it all again."""
        pool = one_backend()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertEqual(pool.link.asked()["max_tokens"], 0)
        self.assertIs(pool.link.asked()["stream"], False)


class AClaimIsNeverLeftBehind(unittest.TestCase):
    """claim_turn has no deadline, so a claim a turn does not give back stops
    that conversation for good. Every way out of a turn has to reach
    finish_turn, including a way out nobody planned."""

    def test_a_failure_while_taking_a_backend_gives_the_claim_back(self):
        pool = one_backend()

        def boom(*a, **kw):
            raise RuntimeError("the pool fell over")

        pool.acquire = boom
        with self.assertRaises(RuntimeError):
            pool.turn(router.Ask("/v1/chat/completions",
                                 prompt(10),
                                 "c1"), FakeClient())

        self.assertNotIn("c1", pool.turns,
                         "the conversation is claimed by a turn that is gone")

    def test_a_stream_already_open_is_stopped_on_the_way_out(self):
        """open() starts the keep-alive before the claim is taken, so a way
        out that skips settle() leaves the ping thread with nothing to stop
        it, and the flow board keeps a row the turn never finished."""
        pool = one_backend()

        def boom(*a, **kw):
            raise RuntimeError("the pool fell over")

        pool.acquire = boom
        client = FakeClient()
        with self.assertRaises(RuntimeError):
            pool.turn(router.Ask("/v1/chat/completions",
                                 prompt(10, stream=True), "c1"), client)

        self.assertIn("settle", client.did, "the keep-alive was left running")
        self.assertNotIn("c1", pool.flow.live,
                         "the board still shows a turn that is gone")


class ARequestEventCarriesTheNumbers(unittest.TestCase):
    """One decomposed line per turn: the reply's timings and the router's own
    stages. The event log is what an offline report is built from, so every
    field has a reader there."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="router-events-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = router.EventLog(directory=self.dir, on=True)

    def turn(self, **on_client):
        read = {"timings": {"prompt_n": 12, "cache_n": 4, "prompt_ms": 100.0,
                            "predicted_n": 0}}
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0,
                           "node": 1}], events=self.log, link=TurnLink(reading=read))
        pool.backends[0].update(up=True, n_ctx=150000)
        client = FakeClient()
        for name, value in on_client.items():
            setattr(client, name, value)
        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)
        self.log.flush()
        rows = []
        for path in Path(self.dir).glob("cache-events-*.jsonl"):
            rows += [json.loads(line) for line in path.read_text().splitlines()]
        return [r for r in rows if r["event"] == "request"][-1]

    def test_the_two_timings_are_decomposed(self):
        row = self.turn(timings={"prompt_n": 2, "cache_n": 8,
                                 "predicted_n": 40, "predicted_ms": 2000.0,
                                 "predicted_per_second": 20.0,
                                 "draft_n": 50, "draft_n_accepted": 25})
        self.assertEqual(row["gen_n"], 40)
        self.assertEqual(row["gen_ms"], 2000.0)
        self.assertEqual(row["tg"], 20.0)
        self.assertEqual(row["accept"], 50.0)      # 25 accepted of 50 drafts
        self.assertEqual(row["pre_ms"], 100.0)     # the read pass wrote the prompt
        self.assertEqual(row["ctx"], 56)           # 4 cached + 12 read + 40 written

    def test_the_router_side_fields_are_named(self):
        row = self.turn(timings={"predicted_n": 1},
                        system_fingerprint="fp-1")
        self.assertEqual(row["serving"], "cpu")       # the generator
        self.assertEqual(row["backend"], "cpu")       # the reader
        self.assertEqual(row["slot"], 0)
        self.assertEqual(row["node"], 1)              # the reader's node
        self.assertEqual(row["serving_node"], 1)      # the generator's node
        self.assertEqual(row["fingerprint"], "fp-1")
        # It never waited: the one slot was free when acquire asked, so no
        # reason is recorded. Classified after the take, every served request
        # with a conversation read "pinned".
        self.assertIsNone(row.get("waited_on"))
        self.assertEqual(row["readers_on_node"], 0)
        self.assertEqual(row["recall_s"], 0.0)


class TheWaitReasonIsSnapshotted(unittest.TestCase):
    """A turn that had no slot says why, and one served at once says nothing.

    The reason is taken while the turn is still in the queue. `_take` points
    the record at the backend that just took the request, so classified after
    acquire every served conversation reads `pinned` and `turn`/`big` are
    unreachable."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="router-wait-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = router.EventLog(directory=self.dir, on=True)

    def row(self):
        self.log.flush()
        rows = []
        for path in Path(self.dir).glob("cache-events-*.jsonl"):
            rows += [json.loads(line) for line in path.read_text().splitlines()]
        return [r for r in rows if r["event"] == "request"][-1]

    def test_a_turn_served_at_once_records_no_reason(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                         events=self.log)
        pool.backends[0].update(up=True, n_ctx=150000)

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"),
                  FakeClient())

        self.assertNotIn("waited_on", self.row())

    def test_a_turn_that_waits_for_its_pin_records_pinned(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                         events=self.log)
        be = pool.backends[0]
        be.update(up=True, n_ctx=150000, slots=1,
                  slots_detail=[{"id": 0, "busy": True}], busy=1)
        pool.pins["c1"] = parked_copy()          # c1 is pinned to cpu
        client = FakeClient()

        turn = threading.Thread(
            target=pool.turn,
            args=(router.Ask("/v1/chat/completions", prompt(10), "c1"), client),
            daemon=True)
        turn.start()
        self.addCleanup(turn.join, 10)
        self.assertTrue(client.asked.wait(10), "the turn never waited")

        with pool.cv:                            # the pinned slot frees
            be["busy"] = 0
            be["slots_detail"] = [{"id": 0, "busy": False}]
            pool.cv.notify_all()
        turn.join(10)

        self.assertEqual(client.relayed, [("cpu", "c1")])
        self.assertEqual(self.row()["waited_on"], "pinned")

    def test_a_turn_behind_another_of_its_conversation_records_turn(self):
        pool = make_pool([{"name": "cpu", "url": "http://cpu", "pref": 0}],
                         events=self.log)
        pool.backends[0].update(up=True, n_ctx=150000)
        ahead = pool.begin_wait("c1", 10)
        pool.claim_turn("c1", ahead)             # the turn ahead holds it
        pool.end_wait(ahead)
        client = FakeClient()

        turn = threading.Thread(
            target=pool.turn,
            args=(router.Ask("/v1/chat/completions", prompt(10), "c1"), client),
            daemon=True)
        turn.start()
        self.addCleanup(turn.join, 10)
        # claim_turn asks alive() once per pass of its wait.
        self.assertTrue(client.asked.wait(10), "the turn never queued")

        pool.finish_turn("c1", ahead)            # the turn ahead ends
        turn.join(10)

        self.assertEqual(client.relayed, [("cpu", "c1")])
        self.assertEqual(self.row()["waited_on"], "turn")

    def test_a_carried_turns_reason_is_the_wait_before_the_handoff(self):
        """It waits for its pin on the reader, then hand_off carries it to
        the generator. The reason reported is the wait before the carry."""
        pool = make_pool([
            {"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False},
            {"name": "cpu", "url": "http://cpu", "pref": 1}],
            events=self.log)
        gpu, cpu = pool.backends
        gpu.update(up=True, n_ctx=150000)
        cpu.update(up=True, n_ctx=150000, slots=1,
                   slots_detail=[{"id": 0, "busy": True}], busy=1)
        pool.pins["c1"] = parked_copy()          # c1 is pinned to cpu
        client = FakeClient()

        turn = threading.Thread(
            target=pool.turn,
            args=(router.Ask("/v1/chat/completions", prompt(10), "c1"), client),
            daemon=True)
        turn.start()
        self.addCleanup(turn.join, 10)
        self.assertTrue(client.asked.wait(10), "the turn never waited")

        with pool.cv:                            # the reader's slot frees
            cpu["busy"] = 0
            cpu["slots_detail"] = [{"id": 0, "busy": False}]
            pool.cv.notify_all()
        turn.join(10)

        row = self.row()
        self.assertEqual(row["backend"], "cpu")  # the reader
        self.assertEqual(row["serving"], "gpu")  # the generator it moved to
        self.assertEqual(row["waited_on"], "pinned")


class ANonDictTimingsObjectIsTolerated(unittest.TestCase):
    """A malformed upstream can send `timings` as a list. The turn decomposes
    it in its finally, so a non-dict there must not raise and lose the
    request record."""

    def test_timing_fields_drops_a_non_dict(self):
        got = router.timing_fields(["not", "an", "object"], "nope")
        self.assertIsNone(got["ctx"])
        self.assertIsNone(got["gen_n"])
        self.assertIsNone(got["pre_ms"])

    def test_a_reply_with_non_dict_timings_still_writes_the_record(self):
        pool = one_backend()
        client = FakeClient()
        client.timings = ["not", "an", "object"]

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual(len(pool.recent_requests), 1)

    def test_a_read_pass_with_non_dict_timings_still_serves(self):
        pool = one_backend(link=TurnLink(reading={"timings": ["oops"]}))
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual(client.relayed, [("cpu", "c1")])
        self.assertEqual(len(pool.recent_requests), 1)


class ACpuGeneratesOnlyTypedTurns(unittest.TestCase):
    """`generate` may name the paths a backend generates for.

    A cpu that reads every prompt but answers only /v1/systemone must not
    answer a chat turn read on it. A chat turn waits for a generator instead;
    with none, the handoff waits and the client's leaving ends it. A typed
    question still writes one token where it was read."""

    def duo(self, draining=False):
        pool = make_pool([
            {"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False},
            {"name": "cpu", "url": "http://cpu", "pref": 1,
             "generate": [router.SYSTEMONE]}])
        gpu, cpu = pool.backends
        gpu.update(up=True, n_ctx=150000, draining=draining)
        cpu.update(up=True, n_ctx=150000)
        return pool, gpu, cpu

    def test_generates_honors_a_list(self):
        be = {"generate": [router.SYSTEMONE]}
        self.assertTrue(router.generates(be, router.SYSTEMONE))
        self.assertFalse(router.generates(be, "/v1/chat/completions"))
        self.assertTrue(router.generates(be))         # it generates something
        self.assertTrue(router.generates({"generate": True},
                                         "/v1/chat/completions"))
        self.assertFalse(router.generates({"generate": False},
                                          "/v1/chat/completions"))

    def test_a_chat_turn_is_never_sent_to_the_cpu_to_generate(self):
        pool, gpu, cpu = self.duo()
        self.assertIs(pool.generator(10, cpu, "/v1/chat/completions"), gpu)

    def test_a_draining_gpu_leaves_no_generator_at_all(self):
        pool, gpu, cpu = self.duo(draining=True)
        self.assertIsNone(pool.generator(10, cpu, "/v1/chat/completions"))

    def test_a_chat_handoff_waits_rather_than_stay_on_the_cpu(self):
        pool, gpu, cpu = self.duo(draining=True)

        def alive():
            return False                  # the client gives up at once

        with self.assertRaises(router.Gone):
            pool.hand_off("c1", cpu, 10, path="/v1/chat/completions",
                          alive=alive)

    def test_a_chat_handoff_that_loses_its_generator_answers_nowhere(self):
        """The gpu is up when the carry starts, so the read is parked and the
        turn waits in `_wait_to_generate`. The generator goes away and then
        the client does. The turn must end with nothing to answer - not fall
        back to the cpu, which generates no chat path at all."""
        pool, gpu, cpu = self.duo()
        pool.pins["c1"] = parked_copy()
        pool.pins["c1"].update(backend="cpu", slot=0)
        cpu["busy"] = 1
        gpu["busy"] = 1                     # its one slot is taken
        asked = []

        def alive():
            asked.append(None)
            if len(asked) == 1:
                gpu["up"] = False           # the generator goes away
            return len(asked) < 3           # and then the client gives up

        self.assertIsNone(pool.hand_off("c1", cpu, 10,
                                        path="/v1/chat/completions",
                                        alive=alive))
        self.assertEqual(pool.generate_waits["gpu"], 0,
                         "the wait outlived the handoff")
        self.assertEqual(cpu["busy"], 0, "the cpu was left holding a slot")

    def test_a_first_loop_wait_is_counted_and_keeps_its_generator_free(self):
        """With no generator up at all, the cpu parks the read it took and
        gives the slot back, then waits for one to appear. That wait is
        counted against the generator it is headed for, so a read is kept off
        it: the wait shields its future generator, and the dashboard says
        so."""
        pool, gpu, cpu = self.duo()
        pool.pins["c1"] = parked_copy()
        pool.pins["c1"].update(backend="cpu", slot=0)
        cpu["busy"] = 1
        gpu["up"] = False                    # no generator up at all
        done = []

        def wait():
            try:
                pool.hand_off("c1", cpu, 10, path="/v1/chat/completions",
                              alive=lambda: not done)
            except router.Gone:
                pass

        hand = threading.Thread(target=wait, daemon=True)
        hand.start()
        for _ in range(200):
            if pool.status()["waiting_to_generate"] == 1:
                break
            time.sleep(0.01)
        self.assertEqual(pool.status()["waiting_to_generate"], 1,
                         "the wait was never counted")
        with pool.cv:
            owed = pool._owed(gpu, "new", 10, "/v1/chat/completions")
        self.assertTrue(owed, "a read would land on the generator it waits for")
        done.append(True)
        with pool.cv:
            pool.cv.notify_all()
        hand.join(10)
        self.assertEqual(pool.generate_waits["gpu"], 0,
                         "the count outlived the wait")

    def test_a_typed_turn_answers_where_it_was_read(self):
        pool, gpu, cpu = self.duo(draining=True)
        self.assertIs(pool.hand_off("c1", cpu, 10, migrate=False,
                                    path=router.SYSTEMONE), cpu)

    def test_a_wait_for_a_generator_gives_the_read_slot_back(self):
        """No generator up: a chat turn parks its read and gives the cpu slot
        back, so a typed turn that needs no gpu reads and answers on the same
        cpu while the chat waits. When one appears the chat is carried there,
        and the copy it parked is what travels - it is not read again."""
        pool, gpu, cpu = self.duo(draining=True)
        pool.pins["c1"] = parked_copy(name=None)      # not parked yet
        pool.pins["c1"].update(backend="cpu", slot=0, using=0, inflight=True)
        cpu["busy"] = 1
        got = []

        def wait():
            try:
                got.append(pool.hand_off("c1", cpu, 10, path="/v1/chat/completions",
                                         alive=lambda: not got))
            except router.Gone:
                got.append(None)

        hand = threading.Thread(target=wait, daemon=True)
        hand.start()
        for _ in range(300):
            if pool.pins["c1"].get("parked") and cpu["busy"] == 0:
                break
            time.sleep(0.01)
        self.assertEqual(cpu["busy"], 0, "the cpu slot was not given back")
        self.assertFalse(pool.pins["c1"]["inflight"],
                         "the pin still claims its read slot")
        self.assertIsNotNone(pool.pick_slot(cpu, "asker"),
                             "the freed cpu slot was not handed out")
        typed = pool.acquire("asker", 10, path=router.SYSTEMONE)
        self.assertTrue(typed, "a typed turn could not read on the cpu")
        self.assertIs(typed[0], cpu, "the typed turn did not take the cpu")

        gpu.update(up=True, draining=False)           # the generator appears
        with pool.cv:
            pool.cv.notify_all()
        hand.join(10)
        self.assertFalse(hand.is_alive(), "the handoff never finished")
        self.assertTrue(got and got[0] is gpu, "the chat did not reach the gpu")
        self.assertEqual(pool.pins["c1"]["backend"], "gpu")
        self.assertIn(("restore", "gpu", "c1.park"), pool.link.calls,
                      "the parked read was not restored onto the gpu")
        self.assertEqual(pool.generate_waits["gpu"], 0,
                         "the wait outlived the handoff")


class ATurnThatCannotBeCarriedReadsWhereItGenerates(unittest.TestCase):
    """A turn with no conversation has no pin, so no slot to carry. Read on
    a cpu that may not answer its path, it had nowhere to go and failed. It
    is read only where it may also be generated."""

    def pair(self):
        pool = make_pool([
            {"name": "cpu", "url": "http://cpu", "pref": 0,
             "generate": [router.SYSTEMONE]},
            {"name": "gpu", "url": "http://gpu", "pref": 1}])
        cpu, gpu = pool.backends
        cpu.update(up=True, n_ctx=150000)
        gpu.update(up=True, n_ctx=150000)
        return pool, cpu, gpu

    def test_a_chat_with_no_conversation_reads_on_the_gpu(self):
        pool, cpu, gpu = self.pair()
        got = pool.acquire(None, 10, path="/v1/chat/completions")
        self.assertIs(got[0], gpu, "it was read where it cannot be answered")

    def test_a_chat_with_a_conversation_still_reads_on_the_cpu(self):
        pool, cpu, gpu = self.pair()
        got = pool.acquire("c1", 10, path="/v1/chat/completions")
        self.assertIs(got[0], cpu)

    def test_a_typed_turn_with_no_conversation_reads_on_the_cpu(self):
        pool, cpu, gpu = self.pair()
        got = pool.acquire(None, 10, path=router.SYSTEMONE)
        self.assertIs(got[0], cpu)

    def test_with_the_gpu_draining_it_waits_and_never_takes_the_cpu(self):
        pool, cpu, gpu = self.pair()
        gpu["draining"] = True
        asked = []

        def alive():
            asked.append(None)
            return len(asked) < 2

        self.assertIsNone(pool.acquire(None, 10, alive,
                                       path="/v1/chat/completions"))
        self.assertEqual(cpu["busy"], 0, "it took the cpu")


class ATurnIsSizedByWhereItCanBeGenerated(unittest.TestCase):
    """A chat turn read on a cpu waits for a generator that can hold it. One
    larger than every generator of its path waited for one that never came.
    It is refused up front instead, before any read."""

    def pair(self):
        pool = make_pool([
            {"name": "gpu", "url": "http://gpu", "pref": 0, "prefill": False},
            {"name": "cpu", "url": "http://cpu", "pref": 1,
             "generate": [router.SYSTEMONE]}])
        gpu, cpu = pool.backends
        gpu.update(up=True, n_ctx=1000)
        cpu.update(up=True, n_ctx=150000)
        return pool, gpu, cpu

    def test_a_chat_larger_than_every_generator_is_refused(self):
        pool, gpu, cpu = self.pair()
        client = FakeClient(alive=False)
        pool.turn(router.Ask("/v1/chat/completions", prompt(20000), "c1"),
                  client)
        self.assertEqual([code for code, _ in client.failed], [413])
        self.assertEqual(pool.link.ops(), [], "it was read anyway")

    def test_a_generator_that_is_down_still_counts(self):
        """The turn waits for a generator that restarts, so its ctx counts."""
        pool, gpu, cpu = self.pair()
        gpu.update(up=False, n_ctx=150000)
        self.assertEqual(pool.largest("/v1/chat/completions"), 150000)

    def test_a_typed_turn_is_sized_by_the_cpu_that_answers_it(self):
        pool, gpu, cpu = self.pair()
        self.assertEqual(pool.largest(router.SYSTEMONE), 150000)


if __name__ == "__main__":
    unittest.main()
