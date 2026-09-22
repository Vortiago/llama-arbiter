// @ts-check
/**
 * The router's status payload, and what the dashboard reads out of it.
 * Every function is pure, so status.test.mjs runs in node without a DOM.
 */

/** @typedef {{ id: number, busy: boolean, prompt: number, done: number, cached: number,
 *              decoded: number, pp_rate: number | null, tg_rate: number | null,
 *              phase: "idle"|"reading"|"generating" }} Slot */
/** @typedef {{ busy_per_decode?: number, pp_rate?: number, tg_rate?: number, accept?: number,
 *              cached?: number, longest?: number, generated?: number, pp_live?: number,
 *              tg_live?: number, read_s?: number, gen_s?: number, prompt_tokens?: number,
 *              cached_tokens?: number }} Stats */
/** @typedef {{ evictions?: number, evicted_mib?: number, skipped?: number, rereads?: number,
 *              prompts?: number, used_mib?: number, limit_mib?: number }} Cache */
/** @typedef {{ n_ctx?: number, n_batch?: number, n_ubatch?: number, n_slots?: number,
 *              kv_unified?: boolean }} Config */
/** @typedef {{ name: string, up: boolean, draining?: boolean,
 *              prefill?: boolean, generate?: boolean, slots: number,
 *              busy: number, served: number, active: number, n_ctx: number, model?: string,
 *              node?: number | null, config?: Config,
 *              stats?: Stats, slots_detail?: Slot[], cache?: Cache }} Backend */
/** @typedef {{ did: string, name: string, at: number, backend: string, slot: number,
 *              bytes: number }} FileEvent */
/** `image_tokens` is the vision encoder's charge for `images`, far below their base64 text cost.
 * @typedef {{ conv: string, since: number, waited: number, tokens: number,
 *              waiting_on: "prefill"|"pinned"|"big"|"turn", backend: string | null,
 *              images?: number, image_tokens?: number }} Waiter */
/** `reused` and `read` are the backend's `timings.cache_n` and `timings.prompt_n`.
 * Null for a turn that never reached the read path. Absent from an older router.
 * @typedef {{ conv: string, backend: string, path: string, took: number, waited: number,
 *              started: string, tokens: number, at: number,
 *              reused?: number | null, read?: number | null,
 *              images?: number, image_tokens?: number }} Request */
/** @typedef {{ done: (number | null)[], cur: number | null }} Series */
/** @typedef {{ step: number, keep: number, since: number | null,
 *              backends: Record<string, { done: Bucket[], cur: Bucket }>,
 *              load?: Record<string, Series> }} ServerHistory */
/** @typedef {{ id: number, cpus: number, cpu: number | null, total?: number | null,
 *              free?: number | null, cache?: number | null, resident_bytes: number,
 *              backends: string[] }} Node */
/** @typedef {{ util: number, vram_used: number, vram_total: number }} Gpu */
/** @typedef {{ nodes: Node[], gpu: Gpu | null }} Machine */
/** @typedef {{ backend: string, slot: number, cuts: number, through?: number | null }} SlotHold */
/** `shared`: how deep a slot holds this opening. `copied`: how deep another
 * conversation's disk copy does. Only `shared` is acted on today.
 * @typedef {{ conv: string, cuts: number, stored: number | null, shared: number | null,
 *             copied?: number | null, held: number }} CacheChoice */
/** @typedef {{ name: string, file?: string, kind: string, bytes?: number, loads?: number,
 *              conv?: string, backend?: string, slot?: number | null,
 *              parked_at?: number | null }} DiskFile */
/** @typedef {{ bases: DiskFile[], deeps: DiskFile[] }} Openings */
/** @typedef {{ count: number, bytes?: number, budget?: number, keep?: number }} Budget */
/** `openings` is one budget over both shelves. `bases` and `deeps` are usage of it, not caps.
 * @typedef {{ copies: Budget, openings: Budget, bases: Budget, deeps: Budget,
 *             files?: DiskFile[],
 *             mounts?: { path: string, total: number, free: number }[] }} Disk */
/** @typedef {"queued" | "prefill" | "generate-queue" | "generate" | "done"} Stage */
/** @typedef {{ conv: string, stage: Stage, backend: string | null, slot: number | null,
 *              since: number | null, changed: number | null, at?: number }} FlowRow */
/** @typedef {{ live: FlowRow[], log: FlowRow[] }} Flow */
/** @typedef {{ backends?: Backend[], waiting?: number, waiting_to_generate?: number,
 *              waiting_detail?: Waiter[], flow?: Flow,
 *              pinned_conversations?: number, saved_prompts?: number, recent_files?: FileEvent[],
 *              recent_requests?: Request[], history?: ServerHistory, machine?: Machine,
 *              slots_hold?: SlotHold[], cache_choices?: CacheChoice[], rates_since?: number | null,
 *              openings?: Openings, disk?: Disk }} Status */

/** A generating slot under this rate is stalled by a prompt read on the same backend.
 * Measured: 0.02 to 0.06 tokens/s against 6.3 solo. */
export const STALL_RATE = 0.5;

/** Ticks on a tape cover this many seconds. */
export const TAPE_SECONDS = 30;

/** @param {Slot} slot */
export const isStalled = (slot) =>
  slot.phase === "generating" && slot.tg_rate !== null
  && slot.tg_rate !== undefined && slot.tg_rate < STALL_RATE;

/** @param {Status} status */
export const backendsOf = (status) => status.backends || [];

/** @param {Backend} be */
export const slotsOf = (be) => be.slots_detail || [];

/** A slot on the same backend that is reading a prompt, or null.
 * @param {Backend} be @param {Slot} slot @returns {Slot | null} */
export function readerBeside(be, slot) {
  return slotsOf(be).find((s) => s !== slot && s.phase === "reading") || null;
}

/** Seconds until a reading slot finishes. Null until its rate resolves (10 second window).
 * `prompt` is what is still to read: Pool._read_slots sets it to
 * `whole - cached - processed`.
 * Do not subtract `done` from it. `done` routinely exceeds it, which leaves this always null.
 * @param {Slot} slot @returns {number | null} */
export function secondsLeft(slot) {
  if (slot.phase !== "reading" || !slot.pp_rate || slot.prompt <= 0) return null;
  return slot.prompt / slot.pp_rate;
}

/** True when this backend may take a new conversation. No `prefill` field means it does.
 * @param {Backend} be */
export const takesNew = (be) => be.up && !be.draining && be.prefill !== false;

/** What the whole queue is blocked on. Shown once above the list.
 * @param {Status} status @returns {string} */
export function poolReason(status) {
  const takers = backendsOf(status).filter(takesNew);
  if (!takers.length) return "no backend that takes new conversations is up";
  const full = takers.every((be) => slotsOf(be).length
    ? slotsOf(be).every((s) => s.busy)
    : be.busy >= be.slots);
  if (!full) return "a slot is free";
  // Ask the whole pool: `takers` passed takesNew, so `prefill === false` never holds there.
  return backendsOf(status).some((be) => be.prefill === false)
    ? "every slot on a backend that prefills is busy"
    : "every slot is busy";
}

/** The soonest slot a waiter could get: a free one, else the reading slot with
 * the least left. A generating slot has no known end.
 * @param {Status} status
 * @returns {{ backend: string, slot: number, seconds: number | null } | null} */
export function nextFree(status) {
  /** @type {{ backend: string, slot: number, seconds: number | null } | null} */
  let best = null;
  for (const be of backendsOf(status)) {
    if (!takesNew(be)) continue;
    for (const s of slotsOf(be)) {
      if (!s.busy) return { backend: be.name, slot: s.id, seconds: 0 };
      const left = secondsLeft(s);
      if (left !== null && (!best || best.seconds === null || left < best.seconds)) {
        best = { backend: be.name, slot: s.id, seconds: left };
      }
    }
  }
  return best;
}

/** The slowest generating slot with a reader beside it, or null.
 * @param {Backend} be @returns {number | null} */
export function contendedRate(be) {
  let worst = null;
  for (const s of slotsOf(be)) {
    if (s.phase !== "generating" || !readerBeside(be, s)) continue;
    // An unresolved rate is null, not zero. A zero would always be the worst.
    if (s.tg_rate == null) continue;
    if (worst === null || s.tg_rate < worst) worst = s.tg_rate;
  }
  return worst;
}

/** The share of prompt tokens the prefilling backends reused, from /metrics. Null until one reports.
 * A generator gets its prompt already read and sits at 99.9% reuse, so it is left out.
 * @param {Status} status @returns {number | null} */
export function reuseShare(status) {
  let read = 0, reused = 0;
  for (const be of backendsOf(status)) {
    if (be.prefill === false) continue;
    read += be.stats?.prompt_tokens || 0;
    reused += be.stats?.cached_tokens || 0;
  }
  return read + reused ? (100 * reused) / (read + reused) : null;
}

/** A waiter's own status, or empty when it matches the queue's shared reason.
 * @param {Waiter} w @param {string} reason */
export function waitLabel(w, reason) {
  if (w.waiting_on === "turn") return "waits for the turn ahead of it in the same conversation";
  if (w.waiting_on === "pinned") return `holds for ${w.backend || "its backend"}, its cache is there`;
  if (w.waiting_on === "big") return "too big for what is free";
  return reason.includes("busy") ? "" : "needs a backend that prefills";
}

/**
 * The whole prompt of a busy slot in three parts: `cached` (reused), `done` (read), `prompt` (left).
 * Null for an idle slot, whose counters reset when its turn ends, and for a slot that never ran.
 * @param {Slot} slot
 * @returns {{ total: number, reused: number, read: number, left: number } | null} */
export function promptBands(slot) {
  if (!slot.busy) return null;
  const reused = slot.cached || 0, left = slot.prompt || 0;
  const read = slot.done || 0;
  const total = reused + read + left;
  if (!total) return null;
  return { total, reused, read, left };
}

/** A cut is a point between messages where another conversation could pick up.
 * -1 is the system prompt alone, i is through message i + 1.
 * @param {number | null | undefined} i */
export function throughText(i) {
  if (i === null || i === undefined) return "-";
  return i < 0 ? "system prompt only" : `through message ${i + 1}`;
}

/** True when a slot (`shared`) or a disk copy (`copied`) holds more of this
 * opening than any saved file does.
 * @param {CacheChoice} c */
export function cutDeeper(c) {
  const held = Math.max(c.shared ?? -1, c.copied ?? -1);
  return held >= 0 && held > (c.stored ?? -1);
}

/** Why no deeper prompt is saved, from the recent requests.
 * @param {Status} status @returns {string} */
export function deeperAnswer(status) {
  const rows = status.cache_choices;
  if (!rows) return "";
  if (!rows.length) return "no request yet";
  const deeper = rows.filter(cutDeeper).length;
  if (!deeper) return "every conversation diverges right after the system prompt, so there is nothing deeper to start from";
  const fromCopy = rows.filter((c) => cutDeeper(c) && (c.copied ?? -1) > (c.shared ?? -1)).length;
  return `${deeper} of ${rows.length} recent requests shared more than is saved`
    + (fromCopy ? `, ${fromCopy} of them with a conversation's own copy already on disk` : "");
}

/** True when a node's page cache is smaller than its backends' resident weights.
 * Never true while the requirement is unknown. @param {Node} node */
export function cacheShort(node) {
  return node.resident_bytes > 0 && node.cache != null && node.cache < node.resident_bytes;
}

/** Slot-seconds by phase, for one bucket of history.
 * @typedef {{ read: number, gen: number, stalled: number, secs: number }} Bucket */
