"""One turn: the steps from a prompt to the reply, and the seam back to
whoever asked.

A Turn talks to its client through five operations and two values. The
handler is the only real client; a test writes its own, which is what lets a
whole turn run with no socket.

    kind                 which client program asked, for the event log
    went                 why the client stopped waiting, once alive() is False
    alive()              False once the client has gone
    open(opening)        start the stream, with its protocol's opening event
    settle()             stop the keep-alive. Twice is safe
    relay(be, body, conv)  send this backend's reply to the client
    fail(code, message)  say the turn cannot be served
"""

from dataclasses import dataclass

from ..sizing import request_cost


@dataclass(frozen=True)
class Ask:
    """What one client sent for one turn, as the router reads it.

    `session` is the conversation a header named, which beats any guess from
    the prompt. The rest of what a turn needs it reads out of the body.
    """

    path: str
    body: bytes
    session: str | None = None


class Turn:
    """One prompt the client sends, and the reply it gets."""

    def __init__(self, pool):
        self.pool = pool

    def run(self, ask, client):
        pool = self.pool
        tokens, images, image_charge = request_cost(ask.body, pool.vision())
        largest = pool.largest()
        if largest and tokens > largest:
            return client.fail(413, f"needs about {tokens} tokens. "
                                    f"The largest backend holds {largest}.")
