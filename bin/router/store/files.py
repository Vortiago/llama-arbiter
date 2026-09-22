"""Everything one run keeps on disk."""

import json, os, shutil
from pathlib import Path
from collections import OrderedDict
from ..identity import SHELF_MARKS

class Store:
    """Everything one run keeps on disk: the parked copies, the openings,
    the two maps that say what they are, and the backend logs.

    A caller names a file. Where that file lives is this class's business.
    Nothing outside it reads a directory, so nothing outside it can be
    pointed at a live router's run/slots.
    """

    def __init__(self, run_dir, block_dir=None):
        self.run = Path(run_dir)
        # A backend writes here under --slot-save-path, so bin/common.sh
        # must name the same directory.
        self.slots = self.run / "slots"
        # Openings go on the faster disk where there are two, under
        # block_budget. Parked copies stay under the run directory, under
        # park_budget.
        self.blocks = Path(block_dir) if block_dir else self.run / "blocks"

    def __repr__(self):
        return f"Store({str(self.run)!r}, {str(self.blocks)!r})"

    def size(self, name):
        """Bytes in one slot file, or 0 when it is gone."""
        try:
            return (self.slots / name).stat().st_size
        except OSError:
            return 0

    def mtime(self, name):
        """When one slot file was last written, or 0 when it is gone."""
        try:
            return (self.slots / name).stat().st_mtime
        except OSError:
            return 0

    def drop(self, name):
        """Delete a slot file, and whatever it points at. Never raises: it
        runs before the turn ticket goes back, and claim_turn has no
        deadline, so a throw here would hold the conversation for the life
        of the process."""
        path = self.slots / name
        try:
            if path.is_symlink():
                path.readlink().unlink(missing_ok=True)
            path.unlink(missing_ok=True)
        except OSError as err:
            print(f"[router] could not delete {name}: {err}", flush=True)

    def link_block(self, name):
        """Point the slot directory at a block on the faster disk. A backend
        takes a bare filename under --slot-save-path and rejects a directory
        in it, so a link is the only way to put one file elsewhere."""
        link = self.slots / name
        try:
            self.slots.mkdir(parents=True, exist_ok=True)
            self.blocks.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            print(f"[router] cannot make {self.slots} and {self.blocks}, "
                  f"keeping blocks with the rest: {err}", flush=True)
            return
        if link.is_symlink() or link.exists():
            link.unlink()
        # Absolute: the kernel resolves a relative target against the link's
        # directory.
        link.symlink_to(self.blocks.resolve() / name)

    def parked_names(self):
        """Every parked copy the last run left, oldest first. Drops a link
        whose target is already gone."""
        if not self.slots.is_dir():
            return []
        names = []
        for found in sorted(self.slots.glob("*.park"),
                            key=lambda f: f.lstat().st_mtime):
            if found.exists():
                names.append(found.name)
            else:
                found.unlink(missing_ok=True)   # a dangling link
        return names

    def read_pins(self):
        """The pin rows the last run wrote."""
        return self._rows(self.slots / "pins.json")

    def write_pins(self, rows):
        return self._write(self.slots / "pins.json", rows)

    def read_openings(self):
        """What the openings earned last run."""
        return self._rows(self.slots / "openings.json")

    def write_openings(self, rows):
        return self._write(self.slots / "openings.json", rows)

    @staticmethod
    def _rows(path):
        """The rows the last run wrote, or [] when there are none to trust."""
        try:
            kept = json.loads(path.read_bytes())
        except Exception:
            return []             # no file, or one we cannot trust
        return kept if isinstance(kept, list) else []

    @staticmethod
    def _write(path, rows):
        """Write beside the file and rename over it, so half of one can never
        be read back. A half file reads as empty, and adopt then deletes
        every file it vouched for."""
        spare = path.with_suffix(".json.new")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            spare.write_text(json.dumps(rows, indent=1))
            os.replace(spare, path)
            return True
        except OSError as err:
            print(f"[router] could not write {path.name}: {err}", flush=True)
            spare.unlink(missing_ok=True)
            return False

    def log(self, name):
        """A backend's own log. The router reads it for the settings that
        backend started with. It also reads the cache lines that appear in no
        other file."""
        return self.run / f"{name}.log"

    def disks(self):
        """Free space on the disks the slot files land on. One row for each
        disk. Slots and blocks are often on the same one."""
        rows, seen = [], set()
        for path in (self.slots, self.blocks):
            try:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                usage = shutil.disk_usage(resolved)
                rows.append({"path": str(path), "total": usage.total,
                             "free": usage.free})
            except OSError:
                continue
        return rows


def opening_key(name):
    """The key inside a saved opening's file name, or None if it is not one."""
    for mark in SHELF_MARKS:
        if name.startswith(mark) and name.endswith(".park"):
            return name[len(mark):-len(".park")]
    return None


def shelf_of(name):
    """Which shelf a saved opening's file name puts it on."""
    return "base" if name.startswith("base-") else "deep"


def adopt_files(names, vouched=(), size=None, store=None, *, tuning):
    """Sort the files the last run left behind, oldest first. A saved
    opening is named after its contents. A conversation's copy is good only
    if the pin file vouches for it. The pin file is asked first, because a
    client can make a key look like an opening."""
    size = size or store.size
    openings, bytes_ = OrderedDict(), {}
    parked, spent = [], []
    for name in names:
        if name in vouched:
            parked.append(name)
            continue
        key = opening_key(name)
        if key and shelf_of(name) == "base":
            openings[key] = name
            bytes_[key] = size(name)
        else:
            spent.append(name)      # an unvouched copy, or a deep cut:
                                    # nothing reads one
    spent += trim_openings(openings, bytes_, budget=tuning.block_budget)
    return openings, bytes_, parked, spent


def trim_openings(openings, bytes_, keep=(), *, budget):
    """Drop openings until they fit the block budget, least useful first: deeper
    cuts before system prompts, then least recently used. One is always
    kept. `keep` names openings being built, which are not on disk yet.
    Returns the file names dropped."""
    order = sorted(openings, key=lambda k: shelf_of(openings[k]) == "base")
    dropped = []
    for key in order:
        if sum(bytes_.values()) <= budget or len(openings) <= 1:
            break
        if key in keep:
            continue
        bytes_.pop(key, None)
        dropped.append(openings.pop(key))
    return dropped
