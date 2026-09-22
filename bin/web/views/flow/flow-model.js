// @ts-check
/**
 * The flow payload, read without a DOM. Pure, so the tests run in node.
 * Anything that needs a rect or a clock lives in index.js.
 */
import { backendsOf, slotsOf, slotKey, STALL_RATE, promptBands } from "../status.js";

/** @typedef {import("../status.js").Status} Status */
/** @typedef {import("../status.js").Backend} Backend */
/** @typedef {import("../status.js").Slot} Slot */
/** @typedef {import("../status.js").Waiter} Waiter */
/** @typedef {import("../status.js").DiskFile} DiskFile */
/** @typedef {import("../status.js").FileEvent} FileEvent */
/** @typedef {import("../status.js").Request} Request */

/** Prefill cost here in tokens a second, measured over a working day.
 * Every saving on the page is tokens divided by this. */
export const READ_RATE = 25;

/** Spacing of one mark on a wire, in px. A wire walks this far a second at
 * playbackRate 1, so the rate is marks a second. */
export const MARK_PERIOD = 15;

/** Marks a second past which the eye sees a wagon wheel. At 60Hz a wire
 * aliases past 30 marks a second. A capped wire says so. */
export const MARK_CAP = 24;

/** Every slot key on this page is built one way. status.js owns it: the
 * overview reads the same keys out of the same payload. */
export { slotKey };

/** The rate a reading slot moves at. A slot's own `pp_rate` is 0 until the
 * router's 10 second window closes, so 0 is silence, not a stall. Fall back
 * to the backend average, then to READ_RATE.
 * @param {Backend} be @param {Slot} sl @returns {number} */
export const readRate = (be, sl) => sl.pp_rate || be.stats?.pp_rate || READ_RATE;

/** The rate a generating slot moves at, or null when nothing knows yet.
 * `tg_rate` stays 0 for up to 10 seconds after the first token, which reads
 * as under STALL_RATE. Null, not a floor: a stall is what this number is read for.
 * @param {Backend} be @param {Slot} sl @returns {number | null} */
export const genRate = (be, sl) => sl.tg_rate || be.stats?.tg_rate || null;

/** What flows on the wire into or out of this slot, and how fast.
 * @param {Backend} be @param {Slot} sl
 * @returns {{ tone: "read" | "generate" | "none", rate: number }} */
export function currentOf(be, sl) {
  if (sl.phase === "reading") return { tone: "read", rate: readRate(be, sl) };
  if (sl.phase === "generating") return { tone: "generate", rate: genRate(be, sl) ?? 0 };
  return { tone: "none", rate: 0 };
}

/** Marks a second for a wire, and whether the cap had to hold it back.
 * @param {number} rate @returns {{ marks: number, capped: boolean }} */
export const markRate = (rate) => ({ marks: Math.min(rate, MARK_CAP), capped: rate > MARK_CAP });

/** A slot that has work and is not getting through it.
 * Generating: under STALL_RATE by genRate, not raw `tg_rate`, which is 0 for the first 10 seconds.
 * Reading: the `done` counter has stood still for longer than two batches take.
 * @param {Backend} be @param {Slot} sl @param {number} sinceStep seconds since
 *   `done` last moved, or 0 when it has never been seen to move */
export function stuck(be, sl, sinceStep) {
  if (sl.phase === "generating") {
    const rate = genRate(be, sl);
    return rate !== null && rate < STALL_RATE;
  }
  if (sl.phase !== "reading") return false;
  // bin/qwen-mtp-cpu.sh's default n_batch. 2048 is four times too patient.
  const batch = be.config?.n_batch || 512;
  return sinceStep > (2 * batch) / readRate(be, sl);
}

/** A word and a glyph per state. `--busy` and `--ok` are four units apart
 * under protanopia, so the glyph, not the hue, tells reading from generating. */
export const PHASE = {
  idle: { icon: "·", word: "idle" },
  reading: { icon: "📖", word: "reading" },
  generating: { icon: "⚡", word: "generating" },
  stuck: { icon: "⚠️", word: "not moving" },
  down: { icon: "✖", word: "down" },
};

/** Stands in for a backend that never answered. @type {Slot} */
const DOWN_SLOT = { id: 0, busy: false, prompt: 0, done: 0, cached: 0, decoded: 0,
  pp_rate: 0, tg_rate: 0, phase: "idle" };

/** @typedef {{ i: number, read: number, gen: number, stalled: number }} Bar one
 *  bucket of the history band, keyed by its index */

/** @typedef {{ key: string, backend: string, slot: number, generator: boolean,
 *              phase: string, stuck: boolean, bands: ReturnType<typeof promptBands>,
 *              tone: string, rate: number, marks: number, capped: boolean,
 *              decoded: number, ctx: number, nCtx: number, left: number | null,
 *              conv: string | null, kind: string | null, history: Bar[] }} SlotNode */

/** Every live slot, readers first and the generator last, with what it holds.
 * @param {Status} status @param {(key: string) => number} sinceStep
 * @returns {SlotNode[]} */
export function nodesOf(status, sinceStep) {
  const live = status.flow?.live || [];
  /** @type {SlotNode[]} */ const rows = [];
  for (const be of backendsOf(status)) {
    // One history per backend: every slot of a backend shares it.
    const history = historyOf(status, be.name);
    // A backend the router never polled has no slot detail. One stand-in slot
    // gives it a down card. Only while down: a live backend's slots are its own.
    const slots = slotsOf(be).length || be.up ? slotsOf(be) : [DOWN_SLOT];
    for (const sl of slots) {
      const key = slotKey({ backend: be.name, slot: sl.id });
      const { tone, rate } = be.up ? currentOf(be, sl) : { tone: "none", rate: 0 };
      const { marks, capped } = markRate(rate);
      const held = live.find((r) => r.backend === be.name && r.slot === sl.id);
      rows.push({
        key, backend: be.name, slot: sl.id, generator: be.prefill === false,
        phase: be.up ? sl.phase : "down",
        stuck: be.up && stuck(be, sl, sinceStep(key)),
        bands: promptBands(sl), tone, rate, marks, capped,
        decoded: sl.decoded || 0,
        // the prompt it holds plus what it generated, against its window
        ctx: (sl.cached || 0) + (sl.done || 0) + (sl.decoded || 0),
        nCtx: be.n_ctx || 0,
        // Not secondsLeft: that divides by the slot's own pp_rate, which is 0
        // until the window closes. Gated on be.up: a dead backend keeps the
        // slot detail of its last good poll.
        left: be.up && sl.phase === "reading" && sl.prompt > 0
          ? sl.prompt / readRate(be, sl) : null,
        history,
        conv: held ? held.conv : null,
        // The router's word for the work, so the card can say what kind of
        // turn this is. Null on an ordinary one.
        kind: held?.kind || null,
      });
    }
  }
  return rows.sort((a, b) => Number(a.generator) - Number(b.generator) || a.key.localeCompare(b.key));
}

/** @typedef {{ read: number, gen: number, stalled: number }} Band cumulative
 *  percentages up the bucket, so CSS builds one gradient from three stops */

/** The last ten minutes of a backend, one band per ten-second bucket.
 * The only part of this view with a memory.
 * @param {Status} status @param {string} backend @returns {Bar[]} */
export function historyOf(status, backend) {
  const h = status.history?.backends?.[backend];
  if (!h) return [];
  const buckets = [...(h.done || []), ...(h.cur ? [h.cur] : [])];
  return buckets.map((b, i) => {
    // A multi-slot backend can spend more slot-seconds than the bucket is long. Scale by the larger.
    const busy = (b.read || 0) + (b.gen || 0) + (b.stalled || 0);
    const span = Math.max(b.secs || 0, busy, 1e-9);
    const read = (100 * (b.read || 0)) / span;
    const gen = read + (100 * (b.gen || 0)) / span;
    return { i, read, gen, stalled: gen + (100 * (b.stalled || 0)) / span };
  });
}

/** @typedef {{ conv: string, since: number, waited: number, tokens: number,
 *              why: string, kind: "turn" | "pinned" | "other",
 *              images: number, imageTokens: number }} Arrival */

/** The turns with no slot yet, and what each waits for. `since` is part of the
 * key: a `wants: "turn"` waiter is queued behind another turn of the same conversation.
 * @param {Status} status @param {number} [aged] seconds since this payload
 *   arrived, measured locally @returns {Arrival[]} */
export function arrivalsOf(status, aged = 0) {
  return (status.waiting_detail || []).map((w) => ({
    conv: w.conv,
    since: w.since,
    // `waited` is stamped by the router. Age it by local elapsed time, never
    // by comparing this clock to the router's epoch: the skew is unknown.
    waited: w.waited + aged,
    tokens: w.tokens,
    why: w.wants === "turn" ? "behind its own turn"
      : w.wants === "pinned" ? `holding for ${w.backend || "its backend"}`
      : w.wants === "big" ? "too big for what is free"
      : "needs a reader",
    kind: w.wants === "turn" ? "turn" : w.wants === "pinned" ? "pinned" : "other",
    images: w.images || 0,
    imageTokens: w.image_tokens || 0,
  }));
}

/** @typedef {{ conv: string, waited: number, bytes: number }} Parked */

/** The turns parked on disk and waiting for the generator, longest wait first.
 * Each is a file: `hand_off` parks the cache, hands the reader back, then waits.
 * @param {Status} status @param {number} now epoch seconds @returns {Parked[]} */
export function parkedOf(status, now) {
  /** @type {Map<string, number>} */ const size = new Map();
  for (const f of status.disk?.files || []) {
    if (f.kind === "copy" && f.conv) size.set(f.conv, f.bytes || 0);
  }
  return (status.flow?.live || [])
    .filter((r) => r.stage === "generate-queue")
    .map((r) => ({
      conv: r.conv,
      waited: r.changed ? Math.max(0, now - r.changed) : 0,
      bytes: size.get(r.conv) ?? 0,
    }))
    .sort((a, b) => b.waited - a.waited);
}


/** Which slot provably holds each conversation's copy. A copy's `backend` only
 * says where it last ran; `slot` says the cache is still there. A live turn in
 * a slot proves the same fact and counts too.
 * @param {DiskFile[]} files @param {Backend[]} backends
 * @param {{conv: string, backend: string|null, slot: number|null}[]} [live]
 * @returns {Map<string, string>} conversation -> slot key */
export function residency(files, backends, live = []) {
  const up = new Set(backends.filter((b) => b.up).map((b) => b.name));
  /** @type {Map<string, string>} */ const bySlot = new Map();   // slot key -> conv
  /** @type {Map<string, string>} */ const held = new Map();     // conv -> slot key
  // One slot holds one cache, so a later claim evicts the earlier. The router
  // does not clear a displaced pin's `slot`, so two copies can name one slot.
  // Live rows come last: they are the better evidence.
  /** @param {string} conv @param {string} key */
  const claim = (conv, key) => {
    if (!key) return;
    const displaced = bySlot.get(key);
    if (displaced !== undefined && displaced !== conv) held.delete(displaced);
    const moved = held.get(conv);
    if (moved !== undefined && moved !== key) bySlot.delete(moved);
    bySlot.set(key, conv);
    held.set(conv, key);
  };
  for (const f of files) {
    if (f.kind !== "copy" || !f.conv || f.slot == null) continue;
    if (up.has(f.backend || "")) claim(f.conv, slotKey({ backend: f.backend || null, slot: f.slot }));
  }
  for (const r of live) {
    if (r.backend && r.slot != null && up.has(r.backend)) claim(r.conv, slotKey(r));
  }
  return held;
}

/** The inverse of `residency`. Its values are unique, so this is exact.
 * @param {Map<string, string>} held @returns {Map<string, string>} slot key -> conversation */
export const holderOf = (held) => new Map([...held].map(([conv, key]) => [key, conv]));

/** Bytes a second of the two disks. Openings sit on the NVMe. A conversation's
 * copy is written once and read at most once, so it stays on the SATA disk. */
const NVME = 2.3e9, SATA = 520e6;

/** @typedef {{ did: string, store: "openings" | "copies", down: boolean,
 *              key: string, name: string, bytes: number, seconds: number }} Transfer */

/** What one file event moves, which way, and how long it takes. `moved` fires
 * after the restore on the target, so it is an up, not a round trip. Only the
 * two `opening` verbs touch the NVMe.
 * @param {FileEvent} ev @returns {Transfer} */
export function transferOf(ev) {
  const opening = ev.did === "loaded opening" || ev.did === "kept opening";
  const down = ev.did === "parked" || ev.did === "kept opening";
  return {
    did: ev.did,
    store: opening ? "openings" : "copies",
    down,
    key: `${ev.backend}:${ev.slot}`,
    name: ev.name,
    bytes: ev.bytes || 0,
    seconds: Math.min(4, Math.max(0.5, (ev.bytes || 0) / (opening ? NVME : SATA))),
  };
}

/** The entries of a newest-first log newer than `lastAt`, oldest first, with the new stamp.
 * @template {{ at?: number }} T
 * @param {T[] | undefined} log @param {number} lastAt
 * @returns {{ entries: T[], lastAt: number }} */
export function since(log, lastAt) {
  const entries = [...(log || [])].reverse().filter((e) => (e.at ?? 0) > lastAt);
  return { entries, lastAt: entries.reduce((t, e) => Math.max(t, e.at ?? 0), lastAt) };
}

/** One word per `how_started` value from the router. */
export const MODE = {
  cold: "cold", recalled: "recalled", "saved prompt": "opening", "warm slot": "in slot",
};

/** @typedef {{ key: string, conv: string, mode: string, cold: boolean, took: number,
 *              coldCost: number, saved: number, whole: number,
 *              reused: number | null, read: number | null,
 *              images: number, imageTokens: number }} Turn */

/** One finished turn against what reading it cold would cost. `took` is
 * measured; `coldCost` is the whole prompt at READ_RATE. The gap is the saving.
 * @param {Request} r @returns {Turn} */
export function turnOf(r) {
  const reused = r.reused ?? null, read = r.read ?? null;
  const whole = reused !== null && read !== null ? reused + read : r.tokens || 0;
  const coldCost = whole / READ_RATE;
  return {
    key: `${r.conv}:${r.at}`,
    conv: r.conv,
    mode: MODE[/** @type {keyof typeof MODE} */ (r.started)] || r.started,
    cold: r.started === "cold",
    took: r.took,
    coldCost,
    saved: Math.max(0, coldCost - r.took),
    whole, reused, read,
    images: r.images || 0,
    imageTokens: r.image_tokens || 0,
  };
}

/** @typedef {{ conv: string, saved: number, turns: Turn[] }} Group */

/** The tape, grouped by conversation, newest group first.
 * @param {Status} status @param {number} keep
 * @returns {{ groups: Group[], scale: number, exact: boolean }} */
export function tapeOf(status, keep) {
  const rows = (status.recent_requests || []).slice(0, keep).map(turnOf);
  /** @type {Map<string, Group>} */ const by = new Map();
  for (const t of rows) {
    const g = by.get(t.conv);
    if (g) { g.turns.push(t); g.saved += t.saved; }
    else by.set(t.conv, { conv: t.conv, saved: t.saved, turns: [t] });
  }
  return {
    groups: [...by.values()],
    scale: Math.max(1, ...rows.map((t) => Math.max(t.took, t.coldCost))),
    // Every row, not any: the caveat is about the whole tape.
    exact: rows.length > 0 && rows.every((t) => t.reused !== null),
  };
}

/** Hours of reading that never happened, from the backends' lifetime counters.
 * @param {Status} status @returns {{ hours: number, reused: number, read: number }} */
export function skipped(status) {
  let read = 0, reused = 0;
  for (const be of backendsOf(status)) {
    if (be.prefill === false) continue;
    read += be.stats?.prompt_tokens || 0;
    reused += be.stats?.cached_tokens || 0;
  }
  return { hours: reused / READ_RATE / 3600, reused, read };
}

/** @typedef {{ title: string, what: string, files: DiskFile[],
 *              kind: "base" | "deep" }} Shelf */

/** The openings, one shelf per kind, biggest first. The two share one budget,
 * and a deeper cut is dropped before a system prompt. Size is the only thing
 * that tells one hash from another: 3.4 GiB costs eleven times what 357 MiB does to read.
 * @param {Status} status @returns {Shelf[]} */
export function shelvesOf(status) {
  const o = status.openings || { bases: [], deeps: [], wants: [] };
  /** @param {DiskFile[]} files */
  const bySize = (files) => files.slice().sort((a, b) => (b.bytes || 0) - (a.bytes || 0));
  return [
    { title: "system prompts", kind: "base", files: bySize(o.bases || []),
      what: "A system prompt on its own. Every brand new session can start from one." },
    { title: "deeper cuts", kind: "deep", files: bySize(o.deeps || []),
      what: "A cut where two conversations diverge. Reaches further, serves fewer, goes first." },
  ];
}

/** @typedef {{ conv: string, bytes: number, share: number, state: "live" | "kept" | "disk",
 *              where: string, label: boolean, doomed: boolean }} Block */

/** The copies as shares of PARK_BUDGET. The budget caps bytes, so width is
 * linear in bytes. Used most recently first, as the sweep orders them: the
 * right-hand end of the run goes next. A label goes only in a block wide
 * enough for it.
 * @param {Status} status @param {number} px strip width
 * @param {number} [minPx] pixels a name needs @param {number} [maxLabels]
 * @returns {{ blocks: Block[], used: number, count: number, live: number, kept: number }} */
export function blocksOf(status, px, minPx = 44, maxLabels = 6) {
  const files = (status.disk?.files || []).filter((f) => f.kind === "copy")
    .slice().sort((a, b) => (b.used ?? 0) - (a.used ?? 0));
  const budget = status.disk?.copies?.budget || 1;
  const held = residency(files, backendsOf(status), status.flow?.live || []);
  /** @type {Block[]} */
  const blocks = files.map((f) => {
    const conv = f.conv || "";
    const live = held.has(conv);
    // "(before the restart)" names a pin the router took back on startup.
    const kept = (f.backend || "").startsWith("(");
    return {
      conv, bytes: f.bytes || 0, share: (f.bytes || 0) / budget,
      state: live ? "live" : kept ? "kept" : "disk",
      where: live ? `in ${held.get(conv)}`
        : kept ? "kept from before the restart"
        : `on disk, last ran on ${f.backend}`,
      label: false,
      doomed: false,
    };
  });
  for (const i of widest(blocks.map((b) => b.share), px, minPx, maxLabels)) blocks[i].label = true;
  // Mark everything past the budget: the next park sweeps it. The newest is
  // never swept, however large: the router skips index 0. Otherwise one
  // 9.45 GiB file was written 1,456 times in four hours.
  let running = 0;
  blocks.forEach((b, i) => {
    running += b.share;
    b.doomed = i > 0 && running > 1;
  });
  return {
    blocks,
    used: blocks.reduce((t, b) => t + b.share, 0),
    count: blocks.length,
    live: blocks.filter((b) => b.state === "live").length,
    kept: blocks.filter((b) => b.state === "kept").length,
  };
}

/** The indexes of the widest marks that can hold a label at this size.
 * @param {number[]} shares @param {number} px @param {number} minPx
 * @param {number} max @returns {Set<number>} */
export function widest(shares, px, minPx, max) {
  return new Set(shares.map((s, i) => ({ s, i })).filter((e) => e.s * px >= minPx)
    .sort((a, b) => b.s - a.s).slice(0, max).map((e) => e.i));
}

/** How full each store is, for the captions. @param {Status} status */
export function loadOf(status) {
  const d = status.disk;
  const o = status.openings;
  const all = [...(o?.bases || []), ...(o?.deeps || [])];
  return {
    openings: all.length,
    unloaded: all.filter((f) => !f.loads).length,
    bytes: d?.copies?.bytes || 0,
    budget: d?.copies?.budget || 0,
  };
}
