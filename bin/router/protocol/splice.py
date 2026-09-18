"""Rewriting a backend's stream on its way to the client."""

import json
from .sse import read_event, sse_event

class AnthropicSplice:
    """Join a backend's stream onto one the router has already opened.

    The backend's message_start must come out: two in one stream break every
    parser. Its prompt token count moves onto the closing message_delta."""

    def __init__(self):
        self.rest = b""
        self.usage = None
        # What the client ends up seeing.
        self.reported = {}

    def feed(self, chunk):
        """What to pass on for this piece of the backend's stream."""
        self.rest += chunk
        out = []
        while True:
            event, sep, rest = self.rest.partition(b"\n\n")
            if not sep:
                break
            self.rest = rest
            out.append(self._one(event + sep))
        return b"".join(out)

    def tail(self):
        """Whatever was left when the stream ended."""
        last, self.rest = self.rest, b""
        return last

    def _one(self, raw):
        name, data = read_event(raw)
        if name == "message_start":
            if self.usage is None:
                self.usage = (data.get("message") or {}).get("usage") or {}
                self.reported.update(self.usage)
            return b""
        if name == "message_delta" and self.usage:
            # The backend's own generation figures win.
            data["usage"] = merged = dict(self.usage, **(data.get("usage") or {}))
            self.reported.update(merged)
            self.usage = {}
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


class OaiUsageSplice:
    """Read the usage figures out of an openai stream as they pass. The
    chunk carrying them has no choices, so it can be removed."""

    def __init__(self, strip=False):
        self.rest = b""
        self.strip = strip
        self.usage = {}

    def feed(self, chunk):
        self.rest += chunk
        out = []
        while True:
            event, sep, rest = self.rest.partition(b"\n\n")
            if not sep:
                break
            self.rest = rest
            out.append(self._one(event + sep))
        return b"".join(out)

    def tail(self):
        last, self.rest = self.rest, b""
        return last

    def _one(self, raw):
        line = raw.strip()
        if line.startswith(b"data: ") and b'"usage"' in line:
            try:
                obj = json.loads(line[6:])
            except ValueError:
                return raw
            if obj.get("choices") == [] and obj.get("usage"):
                self.usage = obj["usage"]
                if self.strip:
                    return b""
        return raw
