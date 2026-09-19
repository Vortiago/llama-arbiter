"""One turn: the steps from a prompt to the reply, and the seam back to
whoever asked.

A Turn talks to its client through five operations and two values. The
handler is the only real client; a test writes its own, which is what lets a
whole turn run with no socket.

    kind                   which client program asked, for the event log
    went                   why the client stopped waiting, once alive() is False
    alive()                False once the client has gone
    open(opening)          start the stream, with its protocol's opening event
    settle()               stop the keep-alive. Twice is safe
    relay(be, body, conv)  send this backend's reply to the client
    fail(code, message)    say the turn cannot be served. Once a stream has
                           begun there is no status line left, so a client
                           that opened one may only be able to send `message`
"""

import time
from dataclasses import dataclass

from ..identity import conversation_id, prompt_key, short_key
from ..protocol.body import (hoist_system, prompt_cuts, read_only,
                             wants_stream)
from ..protocol.sse import opening_event
from ..sizing import request_cost
from ..transport import Gone


def capture(directory, conv, body, keep):
    """Write one request body down, for comparing two turns offline. Kept
    per conversation, so a busy client cannot crowd out a quiet one."""
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tag = short_key(conv).replace("/", "_")
        # Nanoseconds and fixed width: unique names that sort by time.
        name = f"{time.time_ns()}-{tag}.json"
        (directory / name).write_bytes(body)
        kept = sorted(directory.glob(f"*-{tag}.json"))
        # max, because a negative bound counts from the end: [:-11] of 13
        # files deletes two that were inside the keep.
        for spent in kept[:max(0, len(kept) - keep)]:
            spent.unlink(missing_ok=True)
    except OSError as err:
        print(f"[router] could not write the capture: {err}", flush=True)


def how_started(warm, recalled, loaded):
    """Name what a request extended instead of reading. `warm`: its own
    cache was in a slot. `recalled`: its own copy came back from disk.
    `loaded`: a saved opening was put in the slot."""
    if recalled:
        return "recalled"
    if loaded:
        return "saved prompt"
    if warm:
        return "warm slot"
    return "cold"


def name_conversation(ask):
    """Which conversation this turn belongs to, and how the router knew it.

    The conversation the client named beats a guess from the prompt, and
    covers /v1/messages, where the system prompt is a separate field. Decided
    once: a re-parse of a ctx 150000 turn is megabytes.
    """
    if ask.conv:
        return ask.conv, "header"
    named = prompt_key(ask.body)
    if named:
        return named, "cache_key"
    named = conversation_id(ask.body)
    return named, ("hash" if named else "none")


@dataclass(frozen=True)
class Ask:
    """What one client sent for one turn, as the router reads it.

    `conv` is the conversation a header named, which beats any guess from the
    prompt. The rest of what a turn needs it reads out of the body.
    """

    path: str
    body: bytes
    conv: str | None = None


class Turn:
    """One prompt the client sends, and the reply it gets."""

    def __init__(self, pool):
        self.pool = pool

    def run(self, ask, client):
        pool = self.pool
        tokens, images, image_charge = request_cost(ask.body, pool.vision(),
                                                    pool.tuning)
        # `tokens` carries reply_tokens of room. The dashboard measures a
        # turn against the prompt sent: 1,024 tokens nobody sent is 41
        # seconds of reading nobody did.
        prompt_tokens = max(0, tokens - pool.tuning.reply_tokens)
        largest = pool.largest()
        if largest and tokens > largest:
            return client.fail(413, f"needs about {tokens} tokens. "
                                    f"The largest backend holds {largest}.")
        if not largest:
            return client.fail(503, "no backend is up yet")

        conv, conv_source = name_conversation(ask)
        # `null` in the log, and not an empty string, for a turn whose
        # conversation the router could not name.
        short = short_key(conv) if conv else None
        capture(pool.capture_dir, conv, ask.body, pool.tuning.capture_keep)
        # This model's template refuses a late system message. The turn
        # holds both bodies from here: about 600 KB for a ctx 150000 turn,
        # against a box that holds the model itself.
        body = hoist_system(ask.body)
        if body is not ask.body:
            print(f"[router] a late system message became a user message "
                  f"for {ask.path}", flush=True)
            pool.events.write("start_over", conv=short, reason="late_system",
                              client=client.kind, path=ask.path)
        cuts, messages, system, tools = prompt_cuts(body, pool.tuning)
        # The stream and its keep-alive open before the slot is asked for.
        start = time.time()
        asked = read_only(body)
        if asked is not None and wants_stream(body):
            client.open(opening_event(ask.path, body))

        ticket = pool.begin_wait(conv, tokens, images, image_charge)
        try:
            # The turn ahead holds the pin, slot and copy this one needs.
            mine = pool.claim_turn(conv, ticket, client.alive)
            be = pool.acquire(conv, tokens, client.alive) if mine else None
        finally:
            pool.end_wait(ticket)
        waited = time.time() - start
        if not be:
            # `done` only if this turn held the conversation: Flow is keyed
            # by conversation, and a turn that gave up in claim_turn deleted
            # the live row of the turn running.
            if mine:
                pool.note_stage(conv, "done")
            pool.finish_turn(conv, ticket)
            client.settle()
            return client.fail(503, "no backend can serve this request")
        # Read here, then generate wherever is free after the read.
        serving = be
        left = False                           # the client gave up mid-read
        read_stats = {}
        recalled = loaded = warm = False
        slot = None
        # The backend and the conversation are held from here. Every way out
        # runs the same ending: claim_turn has no deadline, so a claim left
        # behind stops the conversation for good.
        try:
            warm = bool(conv) and pool.holds_slot(conv)
            # One slot, decided once, for the read below to extend.
            slot = pool.pick_slot(be, conv)
            pool.note_stage(conv, "prefill", be["name"], slot)
            # Nothing reaches the backend until the caches on it are on disk.
            pool.ensure_parked(be, conv)
            # A copy with an opening the client no longer sends is no prefix.
            if pool.forget_stale_park(conv, cuts):
                pool.events.write("start_over", conv=short, reason="stale_copy",
                                  client=client.kind, path=ask.path)
            recalled = pool.recall(conv, be, slot)
            loaded = (not recalled
                      and pool.warm_prefix(conv, cuts, messages, system, tools,
                                           be, slot, ask.path,
                                           wanted=client.alive))
            # warm_prefix was the last reader of these three, and each holds
            # a parsed copy of the prompt: about three times the bytes it
            # came from. The read below runs for tens of minutes.
            messages = system = tools = None
            if asked is not None:
                # Watch the client. The timings say what the cache saved.
                # `asked` is reused rather than read again: a second parse of
                # this body is megabytes.
                answer = pool.link.read(be, ask.path,
                                        dict(asked, id_slot=slot),
                                        client.alive, pool.tuning.read_timeout)
                timing = (answer or {}).get("timings") or {}
                read_stats = {"read_prompt_n": timing.get("prompt_n"),
                              "read_cache_n": timing.get("cache_n")}
                pool.note_slot(conv, slot)
                serving = pool.hand_off(conv, be, tokens, wanted=client.alive)
                if serving is None:
                    # The cache is parked, and no backend is held.
                    raise Gone("after its prompt was parked")
            if serving is be:
                # Nothing was carried. hand_off already noted a carried turn.
                pool.note_stage(conv, "generate", be["name"], slot)
            client.settle()                    # waits for a ping in flight
            client.relay(serving, body, conv)
        except Gone as gone:
            # Nobody to answer. What the read got through is parked below.
            # `gone` says which wait it gave up on: the read itself, or the
            # wait for a slot to generate in, which comes after the read.
            print(f"[router] {short} left {gone}, {time.time() - start:.0f}s "
                  f"into {be['name']} ({client.went})", flush=True)
            left = True
        except Exception as err:
            client.settle()                    # before the stream's last word
            client.fail(502, f"{be['name']}: {err}")
        finally:
            client.settle()
            parking = False
            try:
                if serving is not None:
                    pool.note_holds(conv, serving, cuts)
                    pool.release(serving, conv)
                # After the release: these write the copy that is not behind.
                if left:
                    pool.park_partial(conv, be, slot)
                # A backend that does not read cannot serve the next turn, so
                # leave a copy for one that does, on a worker.
                if serving is not None:
                    parking = pool.park_later(serving, conv, ticket)
            except Exception as err:
                # A ticket not given back costs the conversation every later
                # turn: claim_turn has no deadline. The lines below must run.
                print(f"[router] {short} could not be put away: {err}",
                      flush=True)
            took = time.time() - start
            started = how_started(warm, recalled, loaded)
            pool.note_stage(conv, "done")
            # The worker ends the turn once the copy has landed.
            if not parking:
                pool.finish_turn(conv, ticket)
            pool.note_request(conv, be, ask.path, took, waited, started,
                              prompt_tokens, images=images,
                              image_tokens_=image_charge, **read_stats)
            pool.events.write("request", conv=short, source=conv_source,
                              client=client.kind, path=ask.path,
                              est_tokens=tokens, n_cuts=len(cuts),
                              backend=be["name"], started=started,
                              took=round(took, 3), waited=round(waited, 3),
                              left=left or None, **read_stats)
            # Only POST reaches a turn: the handler refuses every other method
            # on an inference path.
            print(f"[router] POST {ask.path} -> {be['name']} "
                  f"{took:.1f}s", flush=True)
