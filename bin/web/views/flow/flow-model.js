// @ts-check
/**
 * The flow payload, read without a DOM: where each turn stands, what each slot
 * is doing and how fast, what the two disk stores hold, and what the caching
 * has saved. Pure, so the tests run in node.
 *
 * Everything here is a function of one payload. The view only places what
 * these say; anything that needs a rect or a clock lives in index.js.
 */
import { backendsOf, slotsOf, STALL_RATE, promptBands } from "../status.js";

/** @typedef {import("../status.js").Status} Status */
/** @typedef {import("../status.js").Backend} Backend */
/** @typedef {import("../status.js").Slot} Slot */
/** @typedef {import("../status.js").Waiter} Waiter */
/** @typedef {import("../status.js").DiskFile} DiskFile */
/** @typedef {import("../status.js").FileEvent} FileEvent */
/** @typedef {import("../status.js").Request} Request */

/** What a prompt costs to prefill here, in tokens a second, measured over a
 * working day. Every "what did this save" figure on the page is tokens
 * divided by this. */
export const READ_RATE = 25;

/** The spacing of one mark on a wire, in px. A wire walks exactly this far a
 * second at playbackRate 1, so the rate IS marks a second. The model decides
 * the rate; the stylesheet only draws it. */
export const MARK_PERIOD = 15;

/** Marks a second past which the eye reads a wagon wheel rather than a flow.
 * A mark covers one period a second at rate 1, so at 60Hz it aliases past half
 * the frame rate - 30 a second, whatever the period is - and this leaves
 * headroom. A capped wire says so rather than lying about its speed. */
export const MARK_CAP = 24;

/** The row a slot stands on. Every key on this page is built here, so a slot
 * named from a payload row and one named from a backend's own slots always
 * read alike. @param {{backend: string|null, slot: number|null}} r */
export const slotKey = (r) => (r.backend ? `${r.backend}:${r.slot ?? 0}` : "");

/** The rate a reading slot is really moving at.
 *
 * A slot's own `pp_rate` is 0.0 until the router's ten second window closes,
 * and a reader's counter steps a batch at a time, so a zero here is silence,
 * not a stall. The backend's own running average is the better answer, and the
 * measured machine rate is the floor.
 * @param {Backend} be @param {Slot} sl @returns {number} */
export const readRate = (be, sl) => sl.pp_rate || be.stats?.pp_rate || READ_RATE;

/** The rate a generating slot is really moving at, or null when nothing knows.
 *
 * The same zero as readRate's, from the same window: `tg_rate` stays 0 all
 * through the read, so it is still 0 for up to ten seconds after the first
 * token comes out. Taken at face value that is under STALL_RATE, so every turn
 * opened wearing "not moving" on a wire with no marks on it. There is no
 * machine floor to fall back on here - a stall is what this number is read for,
 * so inventing one would hide the thing it answers - hence null for silence.
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
 *
 * For a generating slot, under half a token a second because a prefill runs
 * beside it - against genRate, not the raw `tg_rate`, or a turn is stalled for
 * the first ten seconds of every generation it ever does. For a reading slot it
 * cannot be the rate at all, which reads zero whenever the window has not
 * closed; it is the counter standing still for longer than two batches take.
 * @param {Backend} be @param {Slot} sl @param {number} sinceStep seconds since
 *   `done` last moved, or 0 when it has never been seen to move */
export function stuck(be, sl, sinceStep) {
  if (sl.phase === "generating") {
    const rate = genRate(be, sl);
    return rate !== null && rate < STALL_RATE;
  }
  if (sl.phase !== "reading") return false;
  // bin/qwen-mtp-cpu.sh's default, for a backend whose log this router did not
  // write. 2048 here was four times too patient once the default moved.
  const batch = be.config?.n_batch || 512;
  return sinceStep > (2 * batch) / readRate(be, sl);
}

/** A word and a glyph per state. The glyph is not decoration: `--busy` and
 * `--ok` are four units apart under protanopia, so reading and generating
 * cannot be told apart by hue, and the mark carries it instead. */
export const PHASE = {
  idle: { icon: "·", word: "idle" },
  reading: { icon: "📖", word: "reading" },
  generating: { icon: "⚡", word: "generating" },
  stuck: { icon: "⚠️", word: "not moving" },
  down: { icon: "✖", word: "down" },
};

/** Stands in for a backend that never answered, so it has a card to be down on.
 *  @type {Slot} */
const DOWN_SLOT = { id: 0, busy: false, prompt: 0, done: 0, cached: 0, decoded: 0,
  pp_rate: 0, tg_rate: 0, phase: "idle" };

/** @typedef {{ i: number, read: number, gen: number, stalled: number }} Bar one
 *  bucket of the history band, with its index so a list can key on it */

/** @typedef {{ key: string, backend: string, slot: number, generator: boolean,
 *              phase: string, stuck: boolean, bands: ReturnType<typeof promptBands>,
 *              tone: string, rate: number, marks: number, capped: boolean,
 *              decoded: number, ctx: number, nCtx: number, left: number | null,
 *              conv: string | null, history: Bar[] }} SlotNode */

/** Every live slot, readers first and the generator last, with what it holds.
 * @param {Status} status @param {(key: string) => number} sinceStep
 * @returns {SlotNode[]} */
export function nodesOf(status, sinceStep) {
  const live = status.flow?.live || [];
  /** @type {SlotNode[]} */ const rows = [];
  for (const be of backendsOf(status)) {
    // One band per backend, not per slot: every slot of a backend shares its
    // history, and rebuilding sixty buckets a slot is work nobody reads.
    const history = historyOf(status, be.name);
    // A backend the router has never polled carries no slot detail at all -
    // it is seeded empty and only written after /slots answers - so a backend
    // that failed to come up drew nothing, which is the one case the down card
    // exists for. One stands in. Only while it is down: a live backend's slots
    // are its own, and inventing one would be inventing state.
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
        // what this slot's context is carrying: the prompt it holds plus what
        // it has generated on top, against the window it has to fit in.
        ctx: (sl.cached || 0) + (sl.done || 0) + (sl.decoded || 0),
        nCtx: be.n_ctx || 0,
        // Not status.js's secondsLeft: that divides by the slot's own pp_rate,
        // which is zero until the router's window closes, so the eta silently
        // vanished for most of every read. readRate has the honest answer.
        //
        // Gated on be.up like phase and stuck are: a backend that died mid-read
        // keeps the slot detail of its last good poll, and an eta counting down
        // beside "down" is a promise nothing is keeping.
        left: be.up && sl.phase === "reading" && sl.prompt > 0
          ? sl.prompt / readRate(be, sl) : null,
        history,
        conv: held ? held.conv : null,
      });
    }
  }
  return rows.sort((a, b) => Number(a.generator) - Number(b.generator) || a.key.localeCompare(b.key));
}

/** @typedef {{ read: number, gen: number, stalled: number }} Band cumulative
 *  percentages up the bucket, so CSS can build one gradient from three stops */

/** The last ten minutes of a backend, one band per ten-second bucket.
 *
 * `history` is the only part of the payload that remembers anything, and the
 * rest of this view is all present tense. A slot that has been reading for an
 * hour and one that started thirty seconds ago look identical without it, and
 * a stall that came and went leaves no trace at all.
 * @param {Status} status @param {string} backend @returns {Bar[]} */
export function historyOf(status, backend) {
  const h = status.history?.backends?.[backend];
  if (!h) return [];
  const buckets = [...(h.done || []), ...(h.cur ? [h.cur] : [])];
  return buckets.map((b, i) => {
    // A multi-slot backend can spend more slot-seconds than the bucket is
    // long, so the scale is whichever is larger - the bands never overflow.
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

/** The turns with no slot yet, and what each is waiting for.
 *
 * Not status.js's waitLabel: that writes a sentence, and a sentence per card
 * in a column three cards deep is a wall. The distinction is worth four words.
 *
 * `since` rides along because it is the only thing that tells two waiters
 * apart: the whole point of `wants: "turn"` is a turn queued behind another
 * turn of the SAME conversation, so the conversation alone is not a key.
 * @param {Status} status @param {number} [aged] seconds since this payload
 *   arrived, measured locally @returns {Arrival[]} */
export function arrivalsOf(status, aged = 0) {
  return (status.waiting_detail || []).map((w) => ({
    conv: w.conv,
    since: w.since,
    // The router stamps `waited` when it builds the payload, so between two
    // payloads it stands still. `aged` is seconds measured locally since this
    // one arrived, not a reading of any clock: `since` is the router's epoch,
    // and subtracting a browser's own clock from it is off by the skew
    // between two machines rather than by nothing.
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

/** The turns whose prompt is read and whose cache is on disk, waiting for the
 * generator, longest wait first.
 *
 * A count alone says nothing about a queue whose entries sit for minutes
 * holding gigabytes. Each of these is literally a file: `hand_off` parks the
 * cache, hands the reader back, and only then waits - so between the park and
 * the restore the conversation lives on disk and nowhere else.
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


/** Which slot provably holds each conversation's copy.
 *
 * A copy's `backend` is where the conversation last ran; only `slot` says the
 * cache is still sitting there, and the router clears it the moment something
 * displaces it. Three copies can name one single-slot backend and at most one
 * of them is in it, so a line drawn from `backend` alone claims three caches
 * in one slot.
 * Where a copy carries no `slot`, a turn the flow says is in a slot right now
 * proves the same fact from the other side, so it counts too.
 * @param {DiskFile[]} files @param {Backend[]} backends
 * @param {{conv: string, backend: string|null, slot: number|null}[]} [live]
 * @returns {Map<string, string>} conversation -> slot key */
export function residency(files, backends, live = []) {
  const up = new Set(backends.filter((b) => b.up).map((b) => b.name));
  /** @type {Map<string, string>} */ const bySlot = new Map();   // slot key -> conv
  /** @type {Map<string, string>} */ const held = new Map();     // conv -> slot key
  // One slot holds one cache, so a later claim evicts the earlier one. The
  // router does not clear a displaced pin's `slot` - it only clears its own
  // when the backend changes - so two copies really do name one slot after a
  // conversation hands it on, and without this both were drawn sitting in it.
  // The live rows come last because a turn running there now is the better
  // evidence than a copy that was written there once.
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

/** `residency` read from the slot's side. Its values are unique, so this is an
 * exact inverse - and one pass, where scanning the map per slot was not.
 * @param {Map<string, string>} held @returns {Map<string, string>} slot key -> conversation */
export const holderOf = (held) => new Map([...held].map(([conv, key]) => [key, conv]));

/** The two disks, and how each is read. Openings are symlinked onto the NVMe
 * because every new session starts from one; a conversation's copy is written
 * once and read at most once, so it stays on the roomier SATA disk. */
const NVME = 2.3e9, SATA = 520e6;

/** @typedef {{ did: string, store: "openings" | "copies", down: boolean,
 *              key: string, name: string, bytes: number, seconds: number }} Transfer */

/** What one slot-file event moves, which way, and how long it really takes.
 *
 * All five `did` values move something. `moved` fires after the restore on the
 * TARGET, so it names the slot the cache climbed into - an up, not a round
 * trip. Only the two `opening` verbs touch the nvme.
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

/** The entries of a newest-first log that are newer than the last look, oldest
 * first, with the stamp to remember. Serves `recent_files` and `flow.log`
 * alike - both are newest-first and carry `at`.
 * @template {{ at?: number }} T
 * @param {T[] | undefined} log @param {number} lastAt
 * @returns {{ entries: T[], lastAt: number }} */
export function since(log, lastAt) {
  const entries = [...(log || [])].reverse().filter((e) => (e.at ?? 0) > lastAt);
  return { entries, lastAt: entries.reduce((t, e) => Math.max(t, e.at ?? 0), lastAt) };
}

/** One word for where a turn's prompt came from, keyed by the router's own
 * strings (`how_started`). */
export const MODE = {
  cold: "cold", recalled: "recalled", "saved prompt": "opening", "warm slot": "in slot",
};

/** @typedef {{ key: string, conv: string, mode: string, cold: boolean, took: number,
 *              coldCost: number, saved: number, whole: number,
 *              reused: number | null, read: number | null,
 *              images: number, imageTokens: number }} Turn */

/** One finished turn, measured against what reading it cold would have cost.
 *
 * The bank asks what the caching saved, and tokens cannot answer it: every turn
 * is a near-identical wall of reuse. Time can. `took` is measured; `coldCost`
 * is the whole prompt at the rate the backends read, so the gap between them
 * is the saving.
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
 *
 * Ungrouped it was fourteen rows that all said the same conversation and very
 * nearly the same number - repetition reading as noise. One conversation's
 * turns belong together, and the group's total saving is a figure no single
 * row could carry.
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
    // Every row, not any row: one turn reporting counts does not make the
    // thirteen beside it measured, and the caveat is about the whole tape.
    exact: rows.length > 0 && rows.every((t) => t.reused !== null),
  };
}

/** Hours of reading that never happened, from the backends' own counters.
 * Lifetime and measured, so it stands whether or not a turn reports its split.
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

/** The openings, one shelf per kind, biggest first.
 *
 * The two share one budget but are not interchangeable: a deeper cut is
 * dropped before a system prompt, whatever their ages, because every brand new
 * session starts from a system prompt. Within a shelf, size is the only thing
 * that tells one 8-hex hash from another - a 3.4 GiB Claude Code prompt costs
 * eleven times what a 357 MiB one does to read - so the shelf is ordered by it
 * and nothing else. There are no empty places to draw: what fits depends on
 * what each one weighs, so a shelf is only ever as long as its files.
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

/** The copies, as shares of the budget they are capped by.
 *
 * Linear in bytes, never square-rooted: PARK_BUDGET caps bytes and not files,
 * so a width that does not mean bytes misstates the one thing the strip is
 * for. A name only goes inside a block that can hold it; the rest carry theirs
 * in the table twin.
 * The order is the budget's own: the sweep keeps the newest parked copies and
 * drops the oldest, so newest sits at the left and the right-hand end of the
 * filled run is what goes next. In payload order the strip was an inventory;
 * in this order it is a forecast.
 * @param {Status} status @param {number} px strip width
 * @param {number} [minPx] pixels a name needs @param {number} [maxLabels]
 * @returns {{ blocks: Block[], used: number, count: number, live: number, kept: number }} */
export function blocksOf(status, px, minPx = 44, maxLabels = 6) {
  const files = (status.disk?.files || []).filter((f) => f.kind === "copy")
    .slice().sort((a, b) => (b.parked_at ?? 0) - (a.parked_at ?? 0));
  const budget = status.disk?.copies?.budget || 1;
  const held = residency(files, backendsOf(status), status.flow?.live || []);
  /** @type {Block[]} */
  const blocks = files.map((f) => {
    const conv = f.conv || "";
    const live = held.has(conv);
    // "(before the restart)" is a pin the router took back on startup: it names
    // no live backend, so the next turn restores it wherever it lands.
    const kept = (f.backend || "").startsWith("(");
    return {
      conv, bytes: f.bytes || 0, share: (f.bytes || 0) / budget,
      state: live ? "live" : kept ? "kept" : "disk",
      where: live ? `in ${held.get(conv)}`
        : kept ? "kept from before the restart"
        : `on disk, last ran on ${f.backend}`,
      label: false,
      // the tail of the run is what the budget drops first
      doomed: false,
    };
  });
  for (const i of widest(blocks.map((b) => b.share), px, minPx, maxLabels)) blocks[i].label = true;
  // Walk from the newest and mark everything past the budget: those are what
  // the next park sweeps away. The newest is never swept, however large - the
  // router skips index 0 (`if age and total > PARK_BUDGET`), because dropping
  // the copy just written is what made cpu1_0 write the same 9.45 GiB file
  // 1,456 times in four hours.
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
