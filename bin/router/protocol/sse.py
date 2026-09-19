"""Server-sent events, and the two chat protocols' shapes."""

import json, os

# A keep-alive must be in the client's protocol. A comment keeps an OpenAI
# stream alive. An anthropic parser reports a stream with only comments as
# ended before any data. That protocol has a ping event.
PING = b": ping\n\n"


ANTHROPIC_PING = b'event: ping\ndata: {"type": "ping"}\n\n'


def anthropic(path):
    """True for the endpoint that speaks the anthropic protocol."""
    return "/messages" in (path or "")


def ping_for(path):
    """The keep-alive this endpoint's client understands."""
    return ANTHROPIC_PING if anthropic(path) else PING


def sse_event(name, data):
    """One named SSE event, laid out as the backends lay theirs out."""
    return (f"event: {name}\ndata: ".encode()
            + json.dumps(data).encode() + b"\n\n")


def read_event(raw):
    """The name and the json of one SSE event, or (None, None)."""
    name, data = None, []
    for line in raw.split(b"\n"):
        if line.startswith(b"event:"):
            name = line[6:].strip().decode("utf-8", "replace")
        elif line.startswith(b"data:"):
            data.append(line[5:].strip())
    if name is None or not data:
        return None, None
    try:
        fields = json.loads(b"\n".join(data))
    except ValueError:
        return None, None
    return (name, fields) if isinstance(fields, dict) else (None, None)


def opening_event(path, body):
    """The event a stream of this protocol must begin with, or nothing.

    An anthropic stream that has only pinged has begun no message, and the
    client reports a 502. The id is invented. AnthropicSplice moves the real
    usage onto the closing message_delta."""
    if not anthropic(path):
        return b""
    try:
        model = json.loads(body).get("model")
    except Exception:
        model = None
    return sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_" + os.urandom(12).hex(), "type": "message",
                    "role": "assistant", "model": model or "unknown",
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}})


def wants_ping(content_type, content_length):
    """True when extra bytes can be inserted into this reply safely: only a
    streamed event stream."""
    if content_length:
        return False
    return "text/event-stream" in (content_type or "")


def _say(line):
    print(line, flush=True)
