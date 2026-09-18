"""Talking to a backend over HTTP."""

import http.client, http.server, json, socket, threading, urllib.error, urllib.parse, urllib.request

def http_post(url, path, payload, timeout=300.0):
    """POST json and read the reply. Used for the slot save and restore."""
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as err:
        # A backend puts the reason in the body.
        raise OSError(f"{err.code} on {path}: {said(err)}") from None


def said(err):
    """The reason a backend gave, out of the body of its error reply."""
    try:
        body = json.loads(err.read().decode("utf-8", "replace"))
    except Exception:
        return err.reason
    trouble = body.get("error") if isinstance(body, dict) else None
    if isinstance(trouble, dict):
        return trouble.get("message") or err.reason
    return trouble or err.reason


class Gone(Exception):
    """The client stopped waiting, so what it asked for is no longer wanted."""


def http_post_wanted(url, path, payload, timeout, wanted, every=2.0):
    """POST to a backend. Stop when nobody waits for the answer.

    Closing the connection cancels the task in llama.cpp and frees the slot.
    Use shutdown(), not close(): the reading thread holds the socket open
    through its file object, so close() alone never reaches the backend.
    Raises Gone when the client has left."""
    parts = urllib.parse.urlsplit(url)
    secure = parts.scheme == "https"
    opener = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    conn = opener(parts.hostname, parts.port or (443 if secure else 80),
                  timeout=timeout)
    prefix = parts.path.rstrip("/")
    got = {}

    def run():
        try:
            conn.request("POST", prefix + path, json.dumps(payload).encode(),
                         {"Content-Type": "application/json"})
            reply = conn.getresponse()
            body = reply.read()
            if reply.status >= 400:
                got["error"] = OSError(f"{reply.status} on {path}: "
                                       f"{said_in(body) or reply.reason}")
            else:
                got["answer"] = json.loads(body) if body else {}
        except Exception as err:
            got["error"] = err

    thread = threading.Thread(target=run, name="read", daemon=True)
    thread.start()
    while True:
        thread.join(every)
        if not thread.is_alive():
            break
        if not wanted():
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)   # the backend sees this
            except (OSError, AttributeError):
                pass                   # already gone
            conn.close()               # the backend cancels the task
            thread.join(10)
            raise Gone("the client stopped waiting")
    conn.close()
    if "error" in got:
        raise got["error"]
    return got.get("answer") or {}


def said_in(body):
    """The reason inside a backend's error body, or None."""
    try:
        trouble = json.loads(body).get("error")
    except Exception:
        return None
    if isinstance(trouble, dict):
        return trouble.get("message")
    return trouble
