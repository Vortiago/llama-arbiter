"""How the router reaches a backend.

Every call the router makes to a llama-server goes through here, so a test
substitutes one object rather than four transports. Pool is handed a Link when
it is built. The backend record is the argument rather than the receiver,
because a record is bookkeeping the Pool owns and a Link owns no state at all.
"""

import json
import urllib.error
import urllib.request

from ..transport import http_post, http_post_watched

# A read-only endpoint answers at once or the backend is in trouble. A slot
# file takes as long as the slot takes. Pool hands in Tuning.post_timeout.
LOOK_TIMEOUT = 3.0
POST_TIMEOUT = 300.0


class Link:
    """The production link: HTTP to a llama-server.

    Three kinds of call, by what the caller can do about each. A read-only
    endpoint answers None, because a backend that is down is a state the
    router expects rather than an error it reports. A slot file raises. Work
    raises OSError when the backend refused, and Gone when the client has
    left, and takes `alive` with no default, so a call site says whether
    anybody is waiting rather than getting the unwatched kind by saying
    nothing.
    """

    def __init__(self, look_timeout=LOOK_TIMEOUT, post_timeout=POST_TIMEOUT):
        self.look_timeout = look_timeout
        self.post_timeout = post_timeout

    # -- what a backend says about itself

    def props(self, be, timeout=None):
        return self._get(be["url"] + "/props", timeout)

    def slots(self, be, timeout=None):
        return self._get(be["url"] + "/slots", timeout)

    def metrics(self, be, timeout=None):
        """The text /metrics answers, not json: the caller parses it."""
        return self._get(be["url"] + "/metrics", timeout, raw=True)

    # -- the slot files. Not watched: a copy outlives the turn that asked
    # for it, and one abandoned half written is worse than one nobody reads.

    def save(self, be, slot, name, timeout=None):
        return self._post(be, f"/slots/{slot}?action=save",
                          {"filename": name}, timeout)

    def restore(self, be, slot, name, timeout=None):
        return self._post(be, f"/slots/{slot}?action=restore",
                          {"filename": name}, timeout)

    # -- work. One door, and `alive` is how it is held open.

    def work(self, be, path, payload, alive, timeout=None):
        """Ask a backend to do something, and stop when the client stops
        waiting. Raises Gone."""
        return http_post_watched(be["url"], path, payload,
                                 self.post_timeout if timeout is None else timeout,
                                 alive)

    def render(self, be, path, payload, alive, timeout=None):
        """What the backend's own template makes of these messages.

        `work` under a second name, so a double can answer the template
        without also answering the prefill that calls `work` too."""
        return self.work(be, path, payload, alive, timeout)

    def prefill(self, be, block, slot, alive, timeout=None):
        """Read a block into a slot and generate nothing. The reply's timings
        say how much was processed and how much came from the cache."""
        return self.work(be, "/completion",
                         {"prompt": block, "n_predict": 0,
                          "cache_prompt": True, "id_slot": slot},
                         alive, timeout)

    def open(self, be, path, body, headers, method, timeout):
        """The client's own request, passed through. The caller reads the
        response and closes it. An HTTPError is a reply the backend meant to
        send, so it comes back rather than raising."""
        request = urllib.request.Request(be["url"] + path, data=body,
                                         headers=headers, method=method)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as answered:
            return answered

    # -- one way in for each convention

    def _post(self, be, path, payload, timeout):
        """The slot files, which are read and written whole. Nothing watches
        them, so there is no `alive` here."""
        return http_post(be["url"], path, payload,
                         self.post_timeout if timeout is None else timeout)

    def _get(self, url, timeout, raw=False):
        # `is None`, not falsy: a caller that asks for no wait at all means 0.
        try:
            with urllib.request.urlopen(
                    url, timeout=self.look_timeout if timeout is None else timeout) as r:
                body = r.read()
                return body.decode(errors="replace") if raw else json.loads(body)
        except Exception:
            return None
