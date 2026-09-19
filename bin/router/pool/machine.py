"""What the machine and the slots are doing, for the dashboard."""

import math, subprocess, time
from pathlib import Path
from collections import deque
from ..identity import short_key

RATE_FLOOR = 1.0     # seconds. Under this a count is not a rate.


def per_second(tokens, seconds):
    """A rate, or zero when there is not enough time to divide by."""
    return round(tokens / seconds, 1) if seconds and seconds >= RATE_FLOOR else 0


class History:
    """Slot-seconds by phase, per history bucket. Between two polls a slot
    stays in the phase the earlier poll reported. A bucket edge inside the
    gap splits it."""

    def __init__(self, keep=60, step=10.0, stall_rate=0.5):
        self.keep, self.step = keep, step
        self.stall_rate = stall_rate
        self.at = None
        self.since = None
        self.rows = {}        # backend name -> {"done": [...], "cur": {...}}
        self.state = {}       # backend name -> [(phase, tg_rate)] last seen
        # Machine load shares the buckets.
        # key -> {"done": [mean per bucket], "cur": [sum, count]}
        self.load = {}

    @staticmethod
    def _empty():
        return {"read": 0.0, "gen": 0.0, "stalled": 0.0, "secs": 0.0}

    def _row(self, name):
        return self.rows.setdefault(name, {"done": [], "cur": self._empty()})

    def _charge(self, dt):
        for name, slots in self.state.items():
            cur = self._row(name)["cur"]
            cur["secs"] += dt
            for phase, tg_rate in slots:
                if phase == "reading":
                    cur["read"] += dt
                elif phase == "generating":
                    # No rate yet is not a stall.
                    cur["gen" if tg_rate is None
                        else ("stalled" if tg_rate < self.stall_rate else "gen")] += dt

    def _roll(self):
        for row in self.rows.values():
            row["done"].append(row["cur"])
            del row["done"][:-self.keep]
            row["cur"] = self._empty()
        for row in self.load.values():
            total, count = row["cur"]
            row["done"].append(round(total / count, 1) if count else None)
            del row["done"][:-self.keep]
            row["cur"] = [0.0, 0]

    def push_load(self, gauges):
        """Add one sample of each gauge to the current bucket. Call after
        push."""
        for key, value in gauges.items():
            if value is None:
                continue
            row = self.load.setdefault(key, {"done": [], "cur": [0.0, 0]})
            row["cur"][0] += value
            row["cur"][1] += 1

    def push(self, backends, now):
        """Take one poll's worth of slot detail."""
        if self.at is None:
            self.since = self.at = now
        while self.at < now:
            edge = (math.floor(self.at / self.step) + 1) * self.step
            stop = min(now, edge)
            self._charge(stop - self.at)
            self.at = stop
            if stop == edge:
                self._roll()
        self.state = {be["name"]: [(s["phase"], s["tg_rate"])
                                   for s in be.get("slots_detail") or []]
                      for be in backends if be.get("up")}
        for name in self.state:
            self._row(name)

    def snapshot(self):
        def tidy(bucket):
            return {k: round(v, 1) for k, v in bucket.items()}
        return {"step": self.step, "keep": self.keep, "since": self.since,
                "backends": {name: {"done": [tidy(b) for b in row["done"]],
                                    "cur": tidy(row["cur"])}
                             for name, row in self.rows.items()},
                "load": {key: {"done": list(row["done"]),
                               "cur": (round(row["cur"][0] / row["cur"][1], 1)
                                       if row["cur"][1] else None)}
                         for key, row in self.load.items()}}


def parse_cpulist(text):
    """'0-17,36-53' -> {0, 1, ..., 17, 36, ..., 53}."""
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def read_nodes(root=Path("/sys/devices/system/node")):
    """The NUMA nodes and their cpus, from sysfs."""
    nodes = []
    for path in sorted(root.glob("node[0-9]*")):
        try:
            cpus = parse_cpulist((path / "cpulist").read_text())
        except OSError:
            continue
        nodes.append({"id": int(path.name[4:]), "cpus": cpus, "path": path})
    return nodes


def cpu_times(text):
    """/proc/stat -> {cpu index: (busy, total)}, in jiffies."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        fields = line.split()
        values = [int(v) for v in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)   # +iowait
        out[int(fields[0][3:])] = (sum(values) - idle, sum(values))
    return out


def node_busy(before, after, cpus):
    """Per cent of a node's cpu time spent busy between two samples, or
    None."""
    busy = total = 0
    for n in cpus:
        if n in before and n in after:
            busy += after[n][0] - before[n][0]
            total += after[n][1] - before[n][1]
    return round(100.0 * busy / total, 1) if total > 0 else None


def node_meminfo(text):
    """A node's meminfo -> bytes: total, free, and cache (FilePages)."""
    want = {"MemTotal:": "total", "MemFree:": "free", "FilePages:": "cache"}
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "Node" and parts[2] in want:
            out[want[parts[2]]] = int(parts[3]) * 1024
    return out


def gpu_query(text):
    """One nvidia-smi line 'util, used, total' in MiB -> a dict, or None."""
    try:
        util, used, total = [float(x) for x in text.strip().split(",")]
    except ValueError:
        return None
    return {"util": util, "vram_used": int(used * 1024 * 1024),
            "vram_total": int(total * 1024 * 1024)}


GPU_CMD = ("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
           "--format=csv,noheader,nounits")


def resident_bytes(be):
    """Bytes one backend needs in the page cache: mapped less lazy, from its
    startup log."""
    cache = be.get("cache") or {}
    mapped, lazy = cache.get("mapped_mib") or 0.0, cache.get("lazy_mib") or 0.0
    return int(max(0.0, mapped - lazy) * 1024 * 1024)


class Machine:
    """CPU, memory and GPU load, sampled beside the backend poll. nvidia-smi
    costs 50 to 100 ms, so it runs as a child and is collected on a later
    pass. A missing command turns the gpu row off."""

    def __init__(self, nodes=None, stat=Path("/proc/stat"), gpu_cmd=GPU_CMD,
                 gpu_poll=10.0):
        self.nodes = read_nodes() if nodes is None else nodes
        self.gpu_poll = gpu_poll
        self.stat = stat
        self.gpu_cmd = list(gpu_cmd)
        self.prev = None          # the last /proc/stat sample
        self.cpu = {}             # node id -> busy per cent
        self.mem = {}             # node id -> {total, free, cache}
        self.gpu = None           # the last nvidia-smi answer
        self.gpu_at = None        # when nvidia-smi last started
        self.gpu_proc = None
        self.gpu_ok = True        # False once the command is found missing

    def sample(self, now):
        try:
            current = cpu_times(self.stat.read_text())
        except OSError:
            current = None
        if current and self.prev:
            for node in self.nodes:
                self.cpu[node["id"]] = node_busy(self.prev, current, node["cpus"])
        if current:
            self.prev = current
        for node in self.nodes:
            try:
                self.mem[node["id"]] = node_meminfo((node["path"] / "meminfo").read_text())
            except (OSError, TypeError):
                pass
        self._gpu(now)

    def _gpu(self, now):
        if not self.gpu_ok:
            return
        proc = self.gpu_proc
        if proc is not None:
            if proc.poll() is None:
                if now - (self.gpu_at or now) > 3 * self.gpu_poll:
                    proc.kill()            # wedged. Try again next time.
                    self.gpu_proc = None
                return
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            if proc.stdout:
                proc.stdout.close()
            self.gpu = gpu_query(out) if proc.returncode == 0 else None
            self.gpu_proc = None
            return
        if self.gpu_at is not None and now - self.gpu_at < self.gpu_poll:
            return
        self.gpu_at = now
        try:
            self.gpu_proc = subprocess.Popen(self.gpu_cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
        except (OSError, ValueError):
            self.gpu_ok, self.gpu = False, None

    def gauges(self):
        """The series the history keeps: one number per key, or None."""
        out = {}
        for node in self.nodes:
            key, mem = f"node{node['id']}", self.mem.get(node["id"]) or {}
            out[f"{key}.cpu"] = self.cpu.get(node["id"])
            out[f"{key}.cache"] = mem.get("cache")
            out[f"{key}.free"] = mem.get("free")
        gpu = self.gpu if self.gpu_ok else None
        out["gpu.util"] = gpu["util"] if gpu else None
        out["gpu.vram"] = gpu["vram_used"] if gpu else None
        return out

    def report(self, backends):
        """Current values for the dashboard. `resident_bytes` is the most any
        backend on the node needs: they map the same files."""
        nodes = []
        for node in self.nodes:
            here = [be for be in backends if be.get("node") == node["id"]]
            mem = self.mem.get(node["id"]) or {}
            nodes.append({"id": node["id"], "cpus": len(node["cpus"]),
                          "cpu": self.cpu.get(node["id"]),
                          "total": mem.get("total"), "free": mem.get("free"),
                          "cache": mem.get("cache"),
                          "resident_bytes": max([resident_bytes(be) for be in here], default=0),
                          "backends": [be["name"] for be in here]})
        return {"nodes": nodes, "gpu": self.gpu if self.gpu_ok else None}


class Flow:
    """Every turn in flight and the stages it walks, for the flow dashboard.
    Each request's own thread notes its transitions. The log outlives the
    live row, so the animation can replay a stage it never saw. Held under
    Pool.cv."""

    def __init__(self, flow_log=150):
        self.live = {}                        # conv -> current stage and where
        self.log = deque(maxlen=flow_log)     # newest first, for the animation

    def note(self, conv, stage, backend=None, slot=None):
        """Move a turn to its next stage. Held under the lock."""
        if not conv:
            return
        now = time.time()
        row = self.live.get(conv)
        if stage == "done":
            if row is None:
                return
            self.log.appendleft(dict(row, stage="done", at=now,
                                     since=None, changed=None))
            del self.live[conv]
            return
        if (row and row["stage"] == stage
                and row["backend"] == backend and row["slot"] == slot):
            return
        entry = {"conv": short_key(conv), "stage": stage,
                 "backend": backend, "slot": slot}
        self.live[conv] = dict(entry, since=row["since"] if row else now,
                               changed=now)
        self.log.appendleft(dict(entry, at=now))

    def report(self):
        """What the dashboard animates. Held under the lock."""
        return {"live": list(self.live.values()), "log": list(self.log)}
