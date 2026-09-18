"""Tests for one turn: the steps from a prompt to a reply.

A turn runs against a Client and a Link, so these need no socket. The Client
is whoever asked; the Link is the one way to a backend. Both are stood in for
here, which is what lets a whole turn run in this process.
"""
import json
import sys
import tempfile
import unittest
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
        self.settled = 0               # times the keep-alive was stopped
        self.relayed = []              # (backend name, conversation)
        self.failed = []               # (code, message)

    def alive(self):
        return self._alive

    def open(self, opening):
        self.opened = opening

    def settle(self):
        self.settled += 1

    def relay(self, be, body, conv):
        self.relayed.append((be["name"], conv))

    def fail(self, code, message):
        self.failed.append((code, message))


class TurnLink:
    """The one way to a backend, answering without a socket.

    Records what was asked of it as (operation, backend name), so a test says
    which operation ran rather than matching a URL. `reading` is what a
    prefill pass answers: an exception here is raised instead, which is how a
    client that goes mid-read reaches the turn.
    """

    def __init__(self, reading=None):
        self.calls = []
        self.reading = reading

    def ops(self):
        """Just the operation names, in order."""
        return [call[0] for call in self.calls]

    def read(self, be, path, payload, alive, timeout):
        self.calls.append(("read", be["name"]))
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


def prompt(chars):
    """A chat request whose prompt is about this many characters long."""
    return json.dumps({"messages": [{"role": "user",
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


if __name__ == "__main__":
    unittest.main()
