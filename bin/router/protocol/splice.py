"""Rewriting a backend's stream on its way to the client."""

import json, re
from .sse import read_event, sse_event

# SSE ends an event with a blank line. sse.py reads \r\n as well, so the
# framer has to: on \n\n alone a CRLF stream is never split at all.
EVENT_END = re.compile(rb"\r?\n\r?\n")


class Splice:
    """A backend's stream, rewritten one whole event at a time.

    feed() holds back a partial event, so that half a `data:` line never
    reaches the client. What a whole one becomes is _one()'s business."""

    def __init__(self):
        self.rest = b""

    def feed(self, chunk):
        """What to pass on for this piece of the backend's stream."""
        self.rest += chunk
        out = []
        while True:
            end = EVENT_END.search(self.rest)
            if not end:
                break
            event, self.rest = self.rest[:end.end()], self.rest[end.end():]
            out.append(self._one(event))
        return b"".join(out)

    def tail(self):
        """Whatever was left when the stream ended."""
        last, self.rest = self.rest, b""
        return last


class AnthropicSplice(Splice):
    """Join a backend's stream onto one the router has already opened.

    The backend's message_start must come out: two in one stream break every
    parser. Its prompt token count moves onto the closing message_delta."""

    def __init__(self):
        super().__init__()
        self.usage = None
        # What the client ends up seeing.
        self.reported = {}

    def _one(self, raw):
        name, data = read_event(raw)
        if name == "message_start":
            if self.usage is None:
                self.usage = (data.get("message") or {}).get("usage") or {}
                self.reported.update(self.usage)
            return b""
        if name == "message_delta" and self.usage is not None:
            # The backend's own generation figures win.
            merged = dict(self.usage, **(data.get("usage") or {}))
            self.usage = {}
            # Only when there is something to say. An empty `usage` written
            # onto a delta that carried none is not what an anthropic parser
            # reads there: it wants output_tokens.
            if merged:
                data["usage"] = merged
                self.reported.update(merged)
                return sse_event(name, data)
        return raw


def wants_usage(body):
    """True when an openai stream request asked for the usage chunk itself."""
    try:
        fields = json.loads(body)
    except Exception:
        return False
    opts = fields.get("stream_options") if isinstance(fields, dict) else None
    return bool(isinstance(opts, dict) and opts.get("include_usage"))


def with_usage(body):
    """A copy of the body with stream_options.include_usage set, or None. A
    streamed openai reply carries no usage unless asked. The answer is a
    final chunk with no choices (probe-usage.py)."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    opts = fields.get("stream_options")
    opts = dict(opts) if isinstance(opts, dict) else {}
    opts["include_usage"] = True
    fields["stream_options"] = opts
    try:
        return json.dumps(fields).encode()
    except (TypeError, ValueError):
        return None


def read_oai(body):
    """The usage, timings and fingerprint in a whole (unstreamed) openai
    reply, or empty values when it is not one. A streamed reply is read by
    OaiUsageSplice; a non-streamed one has no events to frame. `usage` and
    `timings` are returned only when they are objects: a payload that sets
    one to a list or a string must not carry it into the turn record."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return {}, {}, None
    if not isinstance(obj, dict):
        return {}, {}, None
    usage, timings = obj.get("usage"), obj.get("timings")
    return (usage if isinstance(usage, dict) else {},
            timings if isinstance(timings, dict) else {},
            obj.get("system_fingerprint"))


class OaiUsageSplice(Splice):
    """Read the usage figures out of an openai stream as they pass. The
    chunk carrying them has no choices, so it can be removed.

    The final chunk also carries `timings`, the reply's own per-request
    numbers, and every chunk carries the backend's `system_fingerprint`. Both
    are kept for the turn record."""

    def __init__(self, strip=False):
        super().__init__()
        self.strip = strip
        self.usage = {}
        self.timings = {}
        self.system_fingerprint = None

    def _one(self, raw):
        line = raw.strip()
        if line.startswith(b"data:") and (b'"usage"' in line
                                          or b'"system_fingerprint"' in line):
            try:
                obj = json.loads(line[5:])
            except ValueError:
                return raw
            # A payload that parses to a list or a string has no .get. Pass
            # it through rather than raising inside the stream.
            if not isinstance(obj, dict):
                return raw
            if obj.get("system_fingerprint"):
                self.system_fingerprint = obj["system_fingerprint"]
            if obj.get("choices") == [] and obj.get("usage"):
                self.usage = obj["usage"]
                timings = obj.get("timings")
                if isinstance(timings, dict) and timings:
                    self.timings = timings
                if self.strip:
                    return b""
        return raw
