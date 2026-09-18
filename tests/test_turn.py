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
    """The one way to a backend, answering without a socket."""


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


if __name__ == "__main__":
    unittest.main()
