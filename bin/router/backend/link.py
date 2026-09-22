"""How the router reaches a backend.

Every call the router makes to a llama-server goes through here. Before this
there were four transports: bare urlopen for the three read-only endpoints,
http_post for the slot files, http_post_watched for the prefill probe, and a
urllib Request for the forward. Only http_post could be substituted, and it
was threaded through thirteen Pool methods as a `post` parameter. The other
three could be reached from a test only over a socket.

A Link carries all of them, and Pool is handed one when it is built. The
backend record is the argument rather than the receiver, because a record is
bookkeeping the Pool owns and a Link owns no state at all.
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

    Every method answers None when the backend cannot be reached, because a
    backend that is down is a state the router expects rather than an error
    it reports. The exception is `read`, which raises Gone when the client
    stopped waiting, and `open`, which hands back the live response for the
    caller to stream.
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

    # -- the slot files

    def save(self, be, slot, name, timeout=None):
        return http_post(be["url"], f"/slots/{slot}?action=save",
                         {"filename": name},
                         self.post_timeout if timeout is None else timeout)

    def restore(self, be, slot, name, timeout=None):
        return http_post(be["url"], f"/slots/{slot}?action=restore",
                         {"filename": name},
                         self.post_timeout if timeout is None else timeout)

    # -- work

    def render(self, be, route, payload, timeout=None):
        """What the backend's own template makes of these messages."""
        return http_post(be["url"], route, payload,
                         self.post_timeout if timeout is None else timeout)

    def prefill(self, be, block, slot, timeout=None):
        """Read a block into a slot and generate nothing. The reply's timings
        say how much was processed and how much came from the cache."""
        return http_post(be["url"], "/completion",
                         {"prompt": block, "n_predict": 0,
                          "cache_prompt": True, "id_slot": slot},
                         self.post_timeout if timeout is None else timeout)

    def read(self, be, path, payload, alive, timeout):
        """Read a prompt and stop when the client stops waiting. Raises Gone."""
        return http_post_watched(be["url"], path, payload, timeout, alive)

    def open(self, be, path, body, headers, method, timeout):
        """The client's own request, passed through. The caller reads the
        response and closes it; an HTTPError is a reply the backend meant to
        send, so it comes back rather than raising."""
        request = urllib.request.Request(be["url"] + path, data=body,
                                         headers=headers, method=method)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as answered:
            return answered

    # -- one way in for the three read-only endpoints

    def _get(self, url, timeout, raw=False):
        try:
            with urllib.request.urlopen(url, timeout=timeout or self.look_timeout) as r:
                body = r.read()
                return body.decode(errors="replace") if raw else json.loads(body)
        except Exception:
            return None
