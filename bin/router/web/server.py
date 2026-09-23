"""The HTTP server, and stamped output."""

import http.client, http.server, socket, sys, time
from .handler import PASSED

class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    # What a Handler reads. __main__ sets the pool and the pass-through
    # list; a test sets the pool and leaves the rest, and `provider` unset
    # means client_config names the machine itself. They are here rather than
    # in the module so that two servers in one process cannot share them by
    # accident.
    pool = None
    passed = PASSED
    provider = None

    def server_bind(self):
        # Accept IPv4 on the IPv6 socket. Tailscale gives a machine both,
        # and macOS clients try IPv6 first.
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()

    def handle_error(self, request, address):
        """A client hanging up is not an error worth a traceback. Python's
        request loop is reading the next request line when a client drops
        its keep-alive connection."""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError,
                                          ConnectionAbortedError)):
            return
        super().handle_error(request, address)


class Stamped:
    """Put the time in front of every line the router prints. Every print
    in this file goes through this."""

    def __init__(self, out):
        self.out = out
        self.fresh = True

    def write(self, text):
        for piece in text.splitlines(keepends=True):
            if self.fresh and piece.strip():
                self.out.write(time.strftime("%m-%d %H:%M:%S "))
            self.out.write(piece)
            self.fresh = piece.endswith("\n")

    def flush(self):
        self.out.flush()
