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
        self.opened = None             # the opening event, once a stream began
        self.relayed = []              # (backend name, conversation)
        self.failed = []               # (code, message)
        self.sent = None               # what relay was given to forward
        self.did = []                  # every operation, in order

    def alive(self):
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

    def read(self, be, path, payload, alive, timeout):
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


def with_system(rules, said="hello"):
    """A chat request that opens with a system prompt, so it has a cut for
    an opening to be parked at."""
    return json.dumps({"messages": [{"role": "system", "content": rules},
                                    {"role": "user", "content": said}]}).encode()


def parked_copy(name="c1.park"):
    """A pin whose cache is on disk and in no slot."""
    return {"backend": "cpu", "slot": None, "parked": name, "bytes": 1 << 20,
            "turns": 1, "parked_turn": 1, "inflight": False, "last": 0.0,
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
        pool = one_backend()
        pool.backends[0]["generate"] = False
        client = FakeClient(alive=False)

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual(client.relayed, [])
        self.assertNotIn("c1", pool.turns)


class APrefillSlotIsNeverLeftHeld(unittest.TestCase):
    """acquire counts a slot the pool never gets back busy for the life of
    the process. hand_off gives up in two places: the later one releases the
    prefiller first and returns None, the earlier one raises Gone."""

    def test_giving_up_before_the_handoff_says_the_client_went(self):
        pool = one_backend()
        pool.backends[0]["generate"] = False
        pool.acquire("c1", 10)             # the prefiller is held from here
        self.assertEqual(pool.backends[0]["busy"], 1)

        with self.assertRaises(router.Gone):
            pool.hand_off("c1", pool.backends[0], 10, alive=lambda: False)

    def test_a_turn_that_loses_its_client_before_the_handoff_holds_nothing(self):
        pool = one_backend()
        pool.backends[0]["generate"] = False

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
        pool = one_backend()
        pool.backends[0]["generate"] = False

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


class ATurnWithNoFreeSlotIsRefused(unittest.TestCase):
    """acquire counts the slots a save is reading, so it normally waits. The
    window it cannot close is between its own answer and pick_slot: the turn
    ahead releases the backend, this one acquires it, and only then does the
    turn ahead start parking its cache out of the slot.

    Unslotted, the read goes to the backend with `id_slot: null`."""

    def test_the_turn_is_refused_rather_than_read_without_a_slot(self):
        pool = one_backend()
        pool.backends[0].update(slots=1,
                                slots_detail=[{"id": 0, "busy": False}])
        pool.pins["first"] = parked_copy()
        pool.pins["first"].update(slot=0, using=0, inflight=True)
        client = FakeClient()

        pool.turn(router.Ask("/v1/chat/completions", prompt(10), "c1"), client)

        self.assertEqual([code for code, _ in client.failed], [503])
        self.assertEqual(pool.link.ops(), [])
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
        pool.pins["c1"]["inflight"] = True     # a save is running on it now

        self.assertFalse(pool.park_partial("c1", pool.backends[0], 0))
        self.assertEqual(pool.link.ops(), [])

    def test_a_copy_on_the_worker_is_not_queued_over_a_running_save(self):
        """The same meeting, one step later. The turn's ending releases the
        backend, which clears `inflight`, and park_all can take the record in
        the two lines before the copy is queued."""
        pool = one_backend()
        pool.backends[0]["prefill"] = False    # it cannot read the next turn
        pool.pins["c1"] = parked_copy()
        pool.pins["c1"]["slot"] = 0
        pool.pins["c1"]["inflight"] = True     # a save is running on it now

        self.assertFalse(pool.park_later(pool.backends[0], "c1", "t1"))
        self.assertTrue(pool.park_jobs.empty())
        self.assertIsNone(pool.parker)         # no worker was even started


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


if __name__ == "__main__":
    unittest.main()
