// @ts-check
// The pure readers of the router payload: stall, time to finish, and the
// queue's shared reason. No DOM.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isStalled, secondsLeft, poolReason, nextFree, contendedRate, reuseShare, waitLabel,
         cacheShort, promptBands, cutDeeper, throughText, deeperAnswer } from "./status.js";

/** @param {Partial<import("./status.js").Slot>} over */
const slot = (over) => ({ id: 0, busy: true, prompt: 0, done: 0, cached: 0, decoded: 0,
  pp_rate: 0, tg_rate: 0, phase: /** @type {const} */ ("idle"), ...over });
/** @param {Partial<import("./status.js").Backend>} over */
const backend = (over) => ({ name: "cpu", up: true, slots: 1, busy: 0, served: 0, active: 0,
  n_ctx: 150000, ...over });

test("a generating slot under half a token a second is stalled", () => {
  assert.equal(isStalled(slot({ phase: "generating", tg_rate: 0.05 })), true);
  assert.equal(isStalled(slot({ phase: "generating", tg_rate: 4.05 })), false);
  assert.equal(isStalled(slot({ phase: "reading", tg_rate: 0 })), false);
});

test("a slot whose rate has not resolved yet is not stalled", () => {
  // null is "no rate measured", which a slot carries for its first
  // RATE_WINDOW of generating. Read as 0 it made every turn start stalled.
  assert.equal(isStalled(slot({ phase: "generating", tg_rate: null })), false);
  assert.equal(isStalled(slot({ phase: "generating", tg_rate: 0 })), true);
});

test("an unresolved rate is not the worst contended rate", () => {
  const be = backend({ slots_detail: [
    slot({ id: 0, phase: "reading", tg_rate: null }),
    slot({ id: 1, phase: "generating", tg_rate: null }),
    slot({ id: 2, phase: "generating", tg_rate: 3.5 }),
  ] });
  assert.equal(contendedRate(be), 3.5);
});

test("seconds left comes from what is still to read, not from the whole prompt", () => {
  // The live numbers that started this: cpu0_0 reading with 22,452 still to
  // read after 87,836 done. `done` is larger than `prompt`, which is only
  // possible because `prompt` is the remainder - so it must not be subtracted.
  assert.equal(secondsLeft(slot({ phase: "reading", prompt: 22452, done: 87836, pp_rate: 11 })), 22452 / 11);
  assert.equal(secondsLeft(slot({ phase: "reading", prompt: 2040, done: 21289, pp_rate: 55.5 })), 2040 / 55.5);
  assert.equal(secondsLeft(slot({ phase: "reading", prompt: 8192, done: 6144, pp_rate: 0 })), null, "rate not resolved");
  assert.equal(secondsLeft(slot({ phase: "reading", prompt: 0, done: 8192, pp_rate: 25 })), null, "nothing left to read");
  assert.equal(secondsLeft(slot({ phase: "generating", prompt: 100, done: 100, pp_rate: 1 })), null);
});

test("the queue reason is one phrase for the whole pool", () => {
  const busy = slot({ phase: "reading" });
  const full = { backends: [backend({ name: "cpu", slots: 1, slots_detail: [busy] })], waiting: 1 };
  assert.equal(poolReason(full), "every slot is busy");
  const free = { backends: [backend({ name: "cpu", slots_detail: [slot({ busy: false })] })] };
  assert.equal(poolReason(free), "a slot is free");
  const down = { backends: [backend({ name: "cpu", up: false })] };
  assert.equal(poolReason(down), "no backend that takes new conversations is up");
});

test("a backend that does not read is not counted, and is named in the reason", () => {
  const gpuFree = backend({ name: "gpu", prefill: false, slots_detail: [slot({ busy: false })] });
  const cpuFull = backend({ name: "cpu", slots_detail: [slot({ phase: "reading" })] });
  assert.equal(poolReason({ backends: [gpuFree, cpuFull] }), "every slot on a backend that prefills is busy");
  assert.equal(nextFree({ backends: [gpuFree, cpuFull] }), null, "the gpu's free slot takes nothing new");
});

test("a draining backend takes nothing new", () => {
  const draining = backend({ name: "cpu", draining: true, slots_detail: [slot({ busy: false })] });
  const other = backend({ name: "cpu2", slots_detail: [slot({ phase: "reading", prompt: 50, done: 50, pp_rate: 25 })] });
  assert.deepEqual(nextFree({ backends: [draining, other] }), { backend: "cpu2", slot: 0, seconds: 2 });
});

test("next free prefers an idle slot, else the read with the least left", () => {
  const cpu = backend({ name: "cpu", slots: 2, slots_detail: [
    slot({ id: 0, phase: "reading", prompt: 1000, done: 0, pp_rate: 10 }),
    slot({ id: 1, phase: "reading", prompt: 100, done: 900, pp_rate: 10 }),
  ] });
  assert.deepEqual(nextFree({ backends: [cpu] }), { backend: "cpu", slot: 1, seconds: 10 });
  cpu.slots_detail?.push(slot({ id: 2, busy: false }));
  assert.deepEqual(nextFree({ backends: [cpu] }), { backend: "cpu", slot: 2, seconds: 0 });
});

test("contended rate is the slowest generating slot with a reader beside it", () => {
  const cpu = backend({ name: "cpu", slots: 3, slots_detail: [
    slot({ id: 0, phase: "generating", tg_rate: 0.05 }),
    slot({ id: 1, phase: "reading", pp_rate: 55.5 }),
    slot({ id: 2, phase: "generating", tg_rate: 0.02 }),
  ] });
  assert.equal(contendedRate(cpu), 0.02);
  const alone = backend({ name: "gpu", slots_detail: [slot({ phase: "generating", tg_rate: 16.6 })] });
  assert.equal(contendedRate(alone), null, "nothing reads beside it");
});

test("reuse share pools the token counts, and is null until there are any", () => {
  assert.equal(reuseShare({ backends: [backend({ name: "cpu" })] }), null);
  const cpu = backend({ name: "cpu", stats: { prompt_tokens: 900, cached_tokens: 100 } });
  const cpu2 = backend({ name: "cpu2", stats: { prompt_tokens: 300, cached_tokens: 700 } });
  assert.equal(reuseShare({ backends: [cpu, cpu2] }), 40);
});

test("reuse share leaves out a backend that does not read", () => {
  // A conversation reaches the gpu with its prompt already in the slot, so the
  // gpu reuses nearly every token it sees and would carry the figure on its
  // own. What is being asked is how the prefilling is going.
  const cpu = backend({ name: "cpu", stats: { prompt_tokens: 900, cached_tokens: 100 } });
  const gpu = backend({ name: "gpu", prefill: false,
    stats: { prompt_tokens: 1, cached_tokens: 999_999 } });
  assert.equal(reuseShare({ backends: [cpu, gpu] }), 10);
  assert.equal(reuseShare({ backends: [gpu] }), null, "nothing that reads has reported");
});

test("a waiter's label is empty when it says what the queue already says", () => {
  const reads = { conv: "a", since: 0, waited: 1, tokens: 10, wants: /** @type {const} */ ("prefill"), backend: null };
  assert.equal(waitLabel(reads, "every slot on a backend that prefills is busy"), "");
  assert.equal(waitLabel(reads, "a slot is free"), "needs a backend that prefills");
  assert.equal(waitLabel({ ...reads, wants: "pinned", backend: "cpu" }, "a slot is free"), "holds for cpu, its cache is there");
  assert.equal(waitLabel({ ...reads, wants: "big" }, "a slot is free"), "too big for what is free");
  assert.equal(waitLabel({ ...reads, wants: "turn" }, "a slot is free"),
    "waits for the turn ahead of it in the same conversation");
});

test("the page cache warns only when it holds less than must stay resident", () => {
  const GIB = 1024 ** 3;
  const node = { id: 0, cpus: 36, cpu: 10, total: 187 * GIB, free: 21 * GIB, backends: [] };
  assert.equal(cacheShort({ ...node, cache: 135 * GIB, resident_bytes: 127 * GIB }), false, "135 GiB cached, 127 needed");
  assert.equal(cacheShort({ ...node, cache: 100 * GIB, resident_bytes: 127 * GIB }), true, "100 GiB cached, 127 needed");
  assert.equal(cacheShort({ ...node, cache: 127 * GIB, resident_bytes: 127 * GIB }), false, "exactly enough");
  assert.equal(cacheShort({ ...node, cache: 1 * GIB, resident_bytes: 0 }), false, "requirement unknown");
  assert.equal(cacheShort({ ...node, cache: null, resident_bytes: 127 * GIB }), false, "cache unknown");
});

test("the slot bar runs over the whole prompt: reused, read, and still to read", () => {
  // The three counters add up to the prompt: cached is what was reused, done
  // what has been read since, prompt what is left.
  const mid = slot({ phase: "reading", prompt: 512, done: 1536, cached: 16541 });
  assert.deepEqual(promptBands(mid), { total: 18589, reused: 16541, read: 1536, left: 512 });
  const cold = slot({ phase: "reading", prompt: 2040, done: 21289, cached: 0 });
  assert.deepEqual(promptBands(cold), { total: 23329, reused: 0, read: 21289, left: 2040 });
  const gen = slot({ phase: "generating", prompt: 0, done: 2048, cached: 16541, decoded: 40 });
  assert.deepEqual(promptBands(gen), { total: 18589, reused: 16541, read: 2048, left: 0 }, "a generating slot has read all of it");
});

test("a prompt almost entirely served from cache reads as almost done", () => {
  // The live numbers that started this: 88,824 reused of an 89,848 token
  // prompt, 512 of the remaining 1,024 read. The old arithmetic showed
  // "512 / 89,848 read" and looked like a cache that was not working.
  const nearly = slot({ phase: "reading", cached: 88824, done: 512, prompt: 512 });
  const bands = promptBands(nearly);
  assert.deepEqual(bands, { total: 89848, reused: 88824, read: 512, left: 512 });
  assert.equal(Math.round((100 * (bands.reused + bands.read)) / bands.total), 99);
});

test("no bar for a slot that cannot show its reuse", () => {
  // Counters reset to zero when a turn ends, and /slots keeps a stale prompt.
  assert.equal(promptBands(slot({ busy: false, phase: "idle", prompt: 23225, done: 0, cached: 0 })), null);
  // A slot that has never run a task reports no prompt at all.
  assert.equal(promptBands(slot({ busy: true, phase: "reading", prompt: 0, done: 0, cached: 0 })), null);
});

test("a deeper prompt is worth saving only when a slot holds more than is saved, past the system prompt", () => {
  assert.equal(cutDeeper({ conv: "a", cuts: 5, stored: 1, shared: 3, held: 4 }), true);
  assert.equal(cutDeeper({ conv: "a", cuts: 5, stored: 3, shared: 3, held: 4 }), false, "already saved that deep");
  assert.equal(cutDeeper({ conv: "a", cuts: 5, stored: null, shared: 0, held: 4 }), true, "through the first message, nothing saved");
  assert.equal(cutDeeper({ conv: "a", cuts: 5, stored: null, shared: -1, held: 4 }), false, "only the system prompt is shared");
  assert.equal(cutDeeper({ conv: "a", cuts: 1, stored: null, shared: null, held: 0 }), false, "no slot holds any of it");
});

test("a cut index reads as messages", () => {
  assert.equal(throughText(-1), "system prompt only");
  assert.equal(throughText(0), "through message 1");
  assert.equal(throughText(10), "through message 11");
  assert.equal(throughText(null), "-");
});

test("the answer to why nothing deeper is saved comes from the recent requests", () => {
  const only = { conv: "a", cuts: 3, stored: -1, shared: -1, held: 1 };
  assert.equal(deeperAnswer({ cache_choices: [only, only] }),
    "every conversation diverges right after the system prompt, so there is nothing deeper to start from");
  const deeper = { conv: "b", cuts: 6, stored: -1, shared: 4, held: 3 };
  assert.equal(deeperAnswer({ cache_choices: [only, deeper] }),
    "1 of 2 recent requests shared more than is saved");
  assert.equal(deeperAnswer({ cache_choices: [] }), "no request yet");
  assert.equal(deeperAnswer({}), "");
});

test("a conversation's own copy counts as somewhere deeper to start", () => {
  // A copy restores through the same call a saved opening does, so a fork
  // could start from one. Only a slot holding the cut was ever counted, and a
  // slot holds it only while the parent is resident - four of those against
  // dozens of copies, so the old answer was mostly measuring residency.
  const onDisk = { conv: "c", cuts: 6, stored: -1, shared: null, copied: 4, held: 0 };
  assert.equal(cutDeeper(onDisk), true);
  assert.equal(deeperAnswer({ cache_choices: [onDisk] }),
    "1 of 1 recent requests shared more than is saved, "
    + "1 of them with a conversation's own copy already on disk");
  // Nothing anywhere is still nothing.
  assert.equal(cutDeeper({ conv: "d", cuts: 3, stored: -1, shared: null, copied: null, held: 0 }),
    false);
});
