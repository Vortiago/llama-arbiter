// @ts-check
// The flow model: which slot is doing what and how fast, which copy a slot
// really holds, what each finished turn saved, and how the two stores fill.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  READ_RATE, MARK_CAP, slotKey, readRate, currentOf, markRate, stuck, historyOf,
  nodesOf, arrivalsOf, residency, transferOf, since, turnOf, tapeOf, skipped,
  shelvesOf, blocksOf, widest, loadOf, parkedOf,
} from "./flow-model.js";

/** @param {Partial<import("../status.js").Slot>} over */
const slot = (over) => ({ id: 0, busy: true, prompt: 0, done: 0, cached: 0, decoded: 0,
  pp_rate: 0, tg_rate: 0, phase: /** @type {const} */ ("idle"), ...over });
/** @param {Partial<import("../status.js").Backend>} over */
const backend = (over) => ({ name: "cpu1_1", up: true, slots: 1, busy: 1, served: 0, active: 1,
  n_ctx: 150016, ...over });
const never = () => 0;

test("a slot is the backend and the number it stands on", () => {
  assert.equal(slotKey({ backend: "cpu0_0", slot: 2 }), "cpu0_0:2");
  assert.equal(slotKey({ backend: null, slot: null }), "");
});

test("a reading slot's own rate of zero is silence, not a standstill", () => {
  // The live numbers that started this: cpu1_1 reading, its own pp_rate 0.0
  // because the router's ten second window had not closed, while the backend's
  // running average said 12.0. The page drew a healthy reader as a fault.
  const be = backend({ stats: { pp_rate: 12 } });
  const sl = slot({ phase: "reading", pp_rate: 0, done: 27384 });
  assert.equal(readRate(be, sl), 12, "the backend's average answers when the slot cannot");
  assert.equal(stuck(be, sl, 10), false, "ten seconds is not a standstill at 12 a second");
  assert.deepEqual(currentOf(be, sl), { tone: "read", rate: 12 });
});

test("the eta survives a slot whose own rate has not resolved", () => {
  // secondsLeft in status.js divides by slot.pp_rate, which is zero until the
  // router's ten second window closes - so the eta disappeared for most of
  // every read, which is exactly when it is worth having.
  const status = { backends: [backend({ stats: { pp_rate: 12 },
    slots_detail: [slot({ phase: "reading", pp_rate: 0, prompt: 24000, done: 1000 })] })] };
  const [n] = nodesOf(status, never);
  assert.equal(n.left, 2000, "24,000 still to read at the backend's 12 a second");
});

test("a reader is stuck only when its counter outlasts two batches", () => {
  const be = backend({ stats: { pp_rate: 12 }, config: { n_batch: 2048 } });
  const sl = slot({ phase: "reading", pp_rate: 0 });
  const twoBatches = (2 * 2048) / 12;                 // 341 seconds
  assert.equal(stuck(be, sl, twoBatches - 1), false);
  assert.equal(stuck(be, sl, twoBatches + 1), true);
});

test("a generating slot starved by a prefill beside it is stalled", () => {
  const be = backend({ name: "gpu0_0", prefill: false });
  assert.equal(stuck(be, slot({ phase: "generating", tg_rate: 0.05 }), 0), true);
  assert.equal(stuck(be, slot({ phase: "generating", tg_rate: 6.3 }), 0), false);
  assert.equal(stuck(be, slot({ phase: "idle" }), 99999), false, "an idle slot is not stuck");
});

test("the last ten minutes stack up the bucket without overflowing it", () => {
  // A bucket is ten slot-seconds for one slot, but a multi-slot backend can
  // spend more than the bucket is long - the bands scale to whichever is
  // larger so a stall can never push the stack past the top.
  const status = { history: { step: 10, keep: 60, since: 0, backends: { cpu: {
    done: [{ read: 10, gen: 0, stalled: 0, secs: 10 },
           { read: 4, gen: 4, stalled: 2, secs: 10 },
           { read: 0, gen: 0, stalled: 0, secs: 10 }],
    cur: { read: 3, gen: 0, stalled: 0, secs: 3 } } } } };
  const bars = historyOf(status, "cpu");
  assert.equal(bars.length, 4, "the bucket in progress counts too");
  assert.deepEqual(bars[0], { i: 0, read: 100, gen: 100, stalled: 100 });
  assert.deepEqual(bars[1], { i: 1, read: 40, gen: 80, stalled: 100 });
  assert.deepEqual(bars[2], { i: 2, read: 0, gen: 0, stalled: 0 }, "an idle bucket is empty");
  assert.equal(bars[3].read, 100, "a part-finished bucket is scaled to what it has had");
  const over = { history: { step: 10, keep: 60, since: 0, backends: { cpu: {
    done: [{ read: 12, gen: 8, stalled: 0, secs: 10 }], cur: null } } } };
  assert.equal(historyOf(over, "cpu")[0].gen, 100, "two slots busy never overflow the bar");
  assert.deepEqual(historyOf({}, "cpu"), [], "a router that sends no history draws none");
});

test("an idle slot pulls no current", () => {
  assert.deepEqual(currentOf(backend({}), slot({ phase: "idle" })), { tone: "none", rate: 0 });
});

test("one mark is one token until the eye cannot follow", () => {
  assert.deepEqual(markRate(11), { marks: 11, capped: false });
  assert.deepEqual(markRate(203.3), { marks: MARK_CAP, capped: true },
    "cpu0_0 burst at 203 a second; the wire says it is capped rather than lying");
});

test("a backend that is down stays on the page and says so", () => {
  // It vanished before: nodesOf skipped anything not up, so the one event an
  // ambient dashboard exists to show was the one it hid.
  const status = { backends: [backend({ name: "cpu1_0", up: false, slots_detail: [slot({ phase: "idle", busy: false })] })] };
  const [n] = nodesOf(status, never);
  assert.equal(n.phase, "down");
  assert.equal(n.stuck, false, "down is not stalled; it is absent");
  assert.deepEqual(currentOf(backend({ up: false }), slot({ phase: "reading", pp_rate: 9 })).rate, 9,
    "currentOf still answers for a slot; nodesOf is what silences a dead backend");
});

test("a turn in flight proves residency when the pin's slot is missing", () => {
  // A router that predates the `slot` field sends none, and then no connector
  // would ever be drawn. The live flow proves the same fact from the other side.
  const files = [{ name: "a", kind: "copy", conv: "one", backend: "cpu0_0", bytes: 1 }];
  const backends = [backend({ name: "cpu0_0" })];
  assert.equal(residency(files, backends).size, 0, "backend alone proves nothing");
  const live = [{ conv: "one", backend: "cpu0_0", slot: 0 }];
  assert.deepEqual([...residency(files, backends, live)], [["one", "cpu0_0:0"]]);
});

test("the readers come first and the generator last", () => {
  const status = { backends: [
    backend({ name: "gpu0_0", prefill: false, slots_detail: [slot({ phase: "generating", decoded: 40, tg_rate: 6 })] }),
    backend({ name: "cpu1_0", slots_detail: [slot({ phase: "idle", busy: false })] }),
  ] };
  const rows = nodesOf(status, never);
  assert.deepEqual(rows.map((r) => r.key), ["cpu1_0:0", "gpu0_0:0"]);
  assert.equal(rows[1].generator, true, "only prefill === false makes a generator");
});

test("a node carries the conversation the flow says is in its slot", () => {
  const status = {
    backends: [backend({ name: "cpu0_0", slots_detail: [slot({ phase: "reading", cached: 7821, done: 87836, prompt: 22452 })] })],
    flow: { live: [{ conv: "f37a52af/230358", stage: /** @type {const} */ ("prefill"), backend: "cpu0_0", slot: 0, since: 1, changed: 1 }], log: [] },
  };
  const [n] = nodesOf(status, never);
  assert.equal(n.conv, "f37a52af/230358");
  assert.deepEqual(n.bands, { total: 118109, reused: 7821, read: 87836, left: 22452 });
});

test("the arrivals say what each turn is waiting for", () => {
  const rows = arrivalsOf({ waiting_detail: [
    { conv: "a", since: 0, waited: 4695.6, tokens: 70989, wants: "turn", backend: null },
    { conv: "b", since: 0, waited: 3, tokens: 10, wants: "pinned", backend: "gpu0_0" },
  ] });
  assert.equal(rows[0].kind, "turn");
  assert.equal(rows[0].why, "behind its own turn", "a card in a stack gets four words, not a sentence");
  assert.equal(rows[1].why, "holding for gpu0_0");
});

test("two turns of one conversation are two arrivals, not one twice", () => {
  // `wants: "turn"` IS a turn queued behind another turn of the same
  // conversation, so the conversation alone cannot key the list: keyed by it,
  // the second row is built fresh every push and the first is never dropped.
  const rows = arrivalsOf({ waiting_detail: [
    { conv: "a", since: 10, waited: 4, tokens: 10, wants: "prefill", backend: null },
    { conv: "a", since: 20, waited: 2, tokens: 10, wants: "turn", backend: null },
  ] });
  assert.deepEqual(rows.map((r) => r.since), [10, 20]);
  assert.equal(new Set(rows.map((r) => `${r.conv}:${r.since}`)).size, 2);
});

test("a parked turn carries the file it is waiting behind", () => {
  // Between the park and the restore the conversation lives on disk and
  // nowhere else, so the queue can say how big each one is and how long it
  // has sat - a count alone says none of it.
  const status = {
    flow: { live: [
      { conv: "a/1", stage: /** @type {const} */ ("generate-queue"), backend: null, slot: null, since: 100, changed: 140 },
      { conv: "b/2", stage: /** @type {const} */ ("generate-queue"), backend: null, slot: null, since: 100, changed: 110 },
      { conv: "c/3", stage: /** @type {const} */ ("prefill"), backend: "cpu", slot: 0, since: 100, changed: 100 },
    ], log: [] },
    disk: { copies: { count: 2 }, bases: { count: 0 }, deeps: { count: 0 },
      files: [{ name: "x", kind: "copy", conv: "a/1", bytes: 9e9 }] },
  };
  const q = parkedOf(status, 200);
  assert.deepEqual(q.map((p) => p.conv), ["b/2", "a/1"], "longest wait first");
  assert.deepEqual(q.map((p) => p.waited), [90, 60]);
  assert.equal(q[0].bytes, 0, "a copy the payload has no size for reads as nothing, not as a guess");
  assert.equal(q[1].bytes, 9e9);
});

test("a request carrying pictures says so, and what they cost", () => {
  // token_estimate folds the image charge into one number; the router keeps
  // the two apart now because a 4,000 token screenshot and 16,000 characters
  // of text are not the same kind of work - the vision encoder runs in RAM on
  // whichever backend serves it, gpu included.
  const [a] = arrivalsOf({ waiting_detail: [{ conv: "a", since: 0, waited: 3, tokens: 91401,
    wants: "prefill", backend: null, images: 2, image_tokens: 4100 }] });
  assert.deepEqual([a.images, a.imageTokens], [2, 4100]);
  const [plain] = arrivalsOf({ waiting_detail: [{ conv: "b", since: 0, waited: 1, tokens: 10,
    wants: "prefill", backend: null }] });
  assert.deepEqual([plain.images, plain.imageTokens], [0, 0], "a router that sends neither reads as none");
  const t2 = turnOf({ conv: "c", backend: "b", path: "/x", took: 10, waited: 0,
    started: "cold", tokens: 5000, at: 1, images: 1, image_tokens: 300 });
  assert.deepEqual([t2.images, t2.imageTokens], [1, 300]);
});

test("only a slot number proves a copy is still in that slot", () => {
  // Three copies naming one single-slot backend: at most one is in it.
  const files = [
    { name: "a", kind: "copy", conv: "one", backend: "gpu0_0", slot: 0, bytes: 1 },
    { name: "b", kind: "copy", conv: "two", backend: "gpu0_0", slot: null, bytes: 1 },
    { name: "c", kind: "copy", conv: "three", backend: "gpu0_0", bytes: 1 },
    { name: "d", kind: "copy", conv: "four", backend: "(before the restart)", slot: 0, bytes: 1 },
  ];
  const held = residency(files, [backend({ name: "gpu0_0", prefill: false })]);
  assert.deepEqual([...held], [["one", "gpu0_0:0"]]);
});

test("every kind of file event moves something, and the openings move faster", () => {
  const ev = (did, bytes = 0) => ({ did, name: "n", at: 1, backend: "cpu0_0", slot: 0, bytes });
  assert.equal(transferOf(ev("parked")).down, true);
  assert.equal(transferOf(ev("kept opening")).down, true);
  assert.equal(transferOf(ev("recalled")).down, false);
  assert.equal(transferOf(ev("loaded opening")).down, false);
  // `moved` fires after the restore on the target, so it names the slot the
  // cache climbed INTO - an up, not a round trip.
  assert.equal(transferOf(ev("moved")).down, false);
  assert.equal(transferOf(ev("parked")).store, "copies");
  assert.equal(transferOf(ev("loaded opening")).store, "openings");
  const big = 2e9;
  assert.ok(transferOf(ev("loaded opening", big)).seconds < transferOf(ev("parked", big)).seconds,
    "the nvme reads an opening in a fraction of the time the sata disk takes a copy");
});

test("an eleven gigabyte park still crosses in a watchable time", () => {
  assert.ok(transferOf({ did: "parked", name: "n", at: 1, backend: "b", slot: 0, bytes: 11.6e9 }).seconds <= 4);
});

test("a log entry nobody has seen yet is read once, and never twice", () => {
  const log = [{ at: 5 }, { at: 3 }, { at: 1 }];       // newest first
  const first = since(log, 0);
  assert.deepEqual(first.entries.map((e) => e.at), [1, 3, 5], "oldest first, so they replay in order");
  assert.equal(first.lastAt, 5);
  assert.deepEqual(since(log, first.lastAt).entries, []);
  assert.deepEqual(since(undefined, 0), { entries: [], lastAt: 0 });
});

test("a turn is measured against what reading it cold would have cost", () => {
  const t = turnOf({ conv: "a", backend: "cpu1_1", path: "/x", took: 626, waited: 0,
    started: "recalled", tokens: 85034, at: 1 });
  assert.equal(t.mode, "recalled");
  assert.equal(t.whole, 85034);
  assert.equal(t.coldCost, 85034 / READ_RATE, "an hour of reading it did not do");
  assert.equal(t.saved, 85034 / READ_RATE - 626);
});

test("a turn that read its whole prompt saved nothing", () => {
  const t = turnOf({ conv: "a", backend: "cpu1_1", path: "/x", took: 4000, waited: 0,
    started: "cold", tokens: 85034, at: 1 });
  assert.equal(t.cold, true);
  assert.equal(t.saved, 0, "a cold turn cannot go faster than reading, so the bar fills its ghost");
});

test("a turn says its exact split once the router reports one", () => {
  const t = turnOf({ conv: "a", backend: "b", path: "/x", took: 10, waited: 0,
    started: "recalled", tokens: 0, at: 1, reused: 84172, read: 862 });
  assert.equal(t.whole, 85034, "the counts win over the estimate");
  assert.deepEqual([t.reused, t.read], [84172, 862]);
});

test("a conversation's turns are one group, and each turn keeps its own key", () => {
  // Ungrouped this was fourteen rows all naming the same conversation; the
  // group carries the name once and the saving for the lot.
  const r = (conv, took, at) => ({ conv, backend: "b", path: "/x", took, waited: 0,
    started: "recalled", tokens: 85034, at });
  const { groups } = tapeOf({ recent_requests: [r("a", 60, 3), r("b", 60, 2), r("a", 60, 1)] }, 14);
  assert.deepEqual(groups.map((g) => g.conv), ["a", "b"], "newest conversation first");
  assert.equal(groups[0].turns.length, 2, "both of a's turns, together");
  assert.equal(new Set(groups.flatMap((g) => g.turns.map((t) => t.key))).size, 3,
    "a key is still one turn, not one conversation");
  assert.equal(groups[0].saved, 2 * (85034 / READ_RATE - 60), "the group carries what the lot saved");
});

test("the tape is drawn against the longest thing on it", () => {
  const r = (took, tokens, at) => ({ conv: "c", backend: "b", path: "/x", took, waited: 0,
    started: "recalled", tokens, at });
  const { groups, scale, exact } = tapeOf({ recent_requests: [r(60, 85034, 2), r(10, 100, 1)] }, 14);
  assert.equal(scale, 85034 / READ_RATE, "the cold cost of the biggest prompt, not the longest turn");
  assert.equal(exact, false, "no router counts yet, so the split is an estimate");
  const half = tapeOf({ recent_requests: [
    { conv: "c", backend: "b", path: "/x", took: 60, waited: 0, started: "recalled",
      tokens: 85034, at: 2, reused: 84172, read: 862 },
    { conv: "c", backend: "b", path: "/x", took: 10, waited: 0, started: "recalled",
      tokens: 100, at: 1 },
  ] }, 14);
  assert.equal(half.exact, false,
    "one turn reporting counts does not make the one beside it measured");
  // half's groups, not the first call's: this read `groups`, which happens to
  // hold two turns of the same conversation as well, so the half-reported
  // payload's grouping was never checked at all.
  assert.equal(half.groups[0].turns.length, 2,
    "both turns of one conversation belong to one group, reported or not");
  assert.equal(groups[0].turns.length, 2);
});

test("the hours saved come from the backends' own counters", () => {
  const status = { backends: [
    backend({ name: "cpu0_0", stats: { prompt_tokens: 13797, cached_tokens: 2401191 } }),
    backend({ name: "gpu0_0", prefill: false, stats: { prompt_tokens: 999, cached_tokens: 999 } }),
  ] };
  const s = skipped(status);
  assert.equal(s.reused, 2401191, "only the backends that read; the gpu is carried a cache, it does not reuse one");
  assert.equal(s.hours, 2401191 / READ_RATE / 3600);
});

test("the two shelves are not one rack", () => {
  const shelves = shelvesOf({
    openings: { bases: [{ name: "a", kind: "system prompt" }], deeps: [] },
    disk: { copies: { count: 0 }, openings: { count: 1, bytes: 9, budget: 10 },
            bases: { count: 1 }, deeps: { count: 0 } },
  });
  // Kinds stay apart on the page even though one budget covers both: they are
  // dropped in a different order and serve different requests.
  assert.deepEqual(shelves.map((s) => [s.kind, s.files.length]), [["base", 1], ["deep", 0]]);
});

test("a copy is as wide as its real share of the budget", () => {
  const status = {
    backends: [backend({ name: "gpu0_0", prefill: false })],
    disk: { copies: { count: 2, bytes: 3, budget: 100 },
            bases: { count: 0 }, deeps: { count: 0 },
            files: [
              { name: "a", kind: "copy", conv: "one", backend: "gpu0_0", slot: 0, bytes: 25 },
              { name: "b", kind: "copy", conv: "two", backend: "(before the restart)", bytes: 50 },
            ] },
  };
  const { blocks, used, live, kept } = blocksOf(status, 900);
  assert.deepEqual(blocks.map((b) => b.share), [0.25, 0.5], "linear in bytes, because the budget counts bytes");
  assert.deepEqual(blocks.map((b) => b.state), ["live", "kept"]);
  assert.deepEqual(blocks.map((b) => b.doomed), [false, false], "nothing is past the budget yet");
  assert.equal(used, 0.75);
  assert.deepEqual([live, kept], [1, 1]);
  assert.match(blocks[1].where, /before the restart/);
});

test("the strip is ordered the way the budget sweeps, and marks what goes next", () => {
  // The sweep keeps the newest parked copies and drops the oldest, so newest
  // sits left and the tail of the filled run is what the next park sweeps away.
  const f = (conv, bytes, parked_at) => ({ name: conv, kind: "copy", conv, bytes, parked_at, backend: "x" });
  const status = { backends: [], disk: { copies: { count: 3, bytes: 3, budget: 100 },
    bases: { count: 0 }, deeps: { count: 0 },
    files: [f("old", 40, 10), f("new", 40, 30), f("mid", 40, 20)] } };
  const { blocks } = blocksOf(status, 900);
  assert.deepEqual(blocks.map((b) => b.conv), ["new", "mid", "old"], "newest parked first");
  assert.deepEqual(blocks.map((b) => b.doomed), [false, false, true],
    "40 + 40 fits in 100, the third does not, so the oldest is what goes");
  const huge = { ...status, disk: { ...status.disk,
    files: [f("only", 400, 30)] } };
  assert.deepEqual(blocksOf(huge, 900).blocks.map((b) => b.doomed), [false],
    "the newest copy is never swept however large - dropping what was just written is the bug that wrote one file 1,456 times");
});

test("a name only goes inside a block that can hold it", () => {
  // 27 copies over a 900px strip: at the real sizes only the largest few clear
  // 44px, and a narrower label would be a clipped one.
  assert.deepEqual([...widest([0.06, 0.004, 0.05], 900, 44, 6)], [0, 2]);
  assert.deepEqual([...widest([0.06, 0.05], 900, 44, 1)], [0], "never more than will fit");
  assert.deepEqual([...widest([0.001], 900, 44, 6)], [], "nothing wide enough is nothing labelled");
});

test("the stores say how full they are and what has gone unused", () => {
  const l = loadOf({
    openings: { bases: [{ name: "a", kind: "k", loads: 0 }, { name: "b", kind: "k", loads: 3 }], deeps: [] },
    disk: { copies: { count: 1, bytes: 5, budget: 10 }, bases: { count: 2 }, deeps: { count: 0 } },
  });
  assert.deepEqual([l.openings, l.unloaded], [2, 1],
    "loads is in-memory in the router, so a zero means not since it started");
  assert.deepEqual([l.bytes, l.budget], [5, 10]);
});
