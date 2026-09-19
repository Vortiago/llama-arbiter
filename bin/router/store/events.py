"""The router's own log of what the cache decided."""

import json, queue, threading, time
from pathlib import Path

class EventLog:
    """Append cache events to a dated JSONL file. Telemetry, not data: a
    thread writes, and a full queue drops the newest event."""

    def __init__(self, directory=None, on=True, name="cache-events",
                 maxsize=10000):
        self.on = on
        self.directory = Path(directory) if directory else None
        if on and self.directory is None:
            raise ValueError("an event log that is on needs a directory")
        self.name = name
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)
        self.handle = None
        self.day = None
        self.dropped = 0
        if self.on:
            threading.Thread(target=self._run, daemon=True).start()

    def write(self, event, **fields):
        """Queue one event. Never blocks, never raises."""
        if not self.on:
            return
        row = {"ts": round(time.time(), 3), "event": event}
        row.update({k: v for k, v in fields.items() if v is not None})
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def flush(self):
        """Wait until everything written so far has reached the file."""
        self.queue.join()

    def _run(self):
        while True:
            row = self.queue.get()
            try:
                self._emit(row)
            except Exception:
                # One bad row must not end the thread.
                self.dropped += 1
            finally:
                self.queue.task_done()

    def _emit(self, row):
        day = time.strftime("%Y%m%d", time.localtime(row["ts"]))
        if day != self.day or self.handle is None:
            if self.handle:
                self.handle.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.handle = (self.directory / f"{self.name}-{day}.jsonl").open("a")
            self.day = day
        self.handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.handle.flush()
