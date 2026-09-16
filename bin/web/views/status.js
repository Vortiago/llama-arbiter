// @ts-check
/**
 * The router's status payload, and what the dashboard reads out of it.
 *
 * Everything here is a pure function of the payload, so it can be tested in
 * node without a DOM (status.test.mjs). The views only format what these say.
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
/** `images` is how many pictures the request carries and `image_tokens` what
 * the vision encoder charges for them, which is a fraction of what their
 * base64 would cost counted as text.
 * @typedef {{ conv: string, since: number, waited: number, tokens: number,
 *              wants: "prefill"|"pinned"|"big"|"turn", backend: string | null,
 *              images?: number, image_tokens?: number }} Waiter */
/** `reused` and `read` are the backend's own token counts for the read pass
 * (`timings.cache_n` and `timings.prompt_n`): what it skipped because a cache
 * already held it, against what it had to process. Null for a turn that never
 * reached the read path, and absent from a router that predates them.
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
/** `shared` is how deep a slot holds this opening, `copied` how deep another
 * conversation's copy on disk does. A copy is the same kind of saved state, so
 * either could serve a fork; only the first is acted on today.
 * @typedef {{ conv: string, cuts: number, stored: number | null, shared: number | null,
 *             copied?: number | null, held: number }} CacheChoice */
/** @typedef {{ name: string, file?: string, kind: string, bytes?: number, loads?: number,
 *              conv?: string, backend?: string, slot?: number | null,
 *              parked_at?: number | null }} DiskFile */
/** @typedef {{ bases: DiskFile[], deeps: DiskFile[], wants: { name: string, kind: string }[] }} Openings */
/** @typedef {{ count: number, bytes?: number, budget?: number, keep?: number }} Budget */
/** `openings` is one budget over both shelves; `bases` and `deeps` say what
 * each kind is using of it, and are not caps of their own.
 * @typedef {{ copies: Budget, openings: Budget, bases: Budget, deeps: Budget, wants: Budget,
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

/** A generating slot under this many tokens a second is stalled: another slot
 * on its backend is reading a prompt, and llama.cpp gives it one step per
 * prompt chunk. Measured: 0.02 to 0.06 tokens/s against 6.3 solo. */
export const STALL_RATE = 0.5;

/** Ticks on a tape cover this many seconds. */
export const TAPE_SECONDS = 30;

/** @param {Slot} slot */
export const isStalled = (slot) =>
  slot.phase === "generating" && slot.tg_rate !== null
  && slot.tg_rate !== undefined && slot.tg_rate < STALL_RATE;

/** The backends the payload has, never undefined. @param {Status} status */
export const backendsOf = (status) => status.backends || [];

/** The slots a backend reports, never undefined. @param {Backend} be */
export const slotsOf = (be) => be.slots_detail || [];

/** A slot on the same backend that is reading a prompt, or null.
 * @param {Backend} be @param {Slot} slot @returns {Slot | null} */
export function readerBeside(be, slot) {
  return slotsOf(be).find((s) => s !== slot && s.phase === "reading") || null;
}

/** Seconds until a reading slot finishes, at its current rate. Null when the
 * rate has not resolved yet: the router measures over a 10 second window.
 *
 * `prompt` is what is STILL TO READ, not the whole prompt - router.py sets it
 * to `whole - cached - processed`, and `promptBands` below reads it the same
 * way. Do not subtract `done` from it: that counts the read part twice, and
 * `done` routinely exceeds it, which leaves this and `nextFree` always null.
 * @param {Slot} slot @returns {number | null} */
export function secondsLeft(slot) {
  if (slot.phase !== "reading" || !slot.pp_rate || slot.prompt <= 0) return null;
  return slot.prompt / slot.pp_rate;
}

/** True when this backend may take a new conversation: up, not draining, and
 * one that prefills. A backend that does not say is assumed to do both, which
 * is what the router assumes too.
 * @param {Backend} be */
export const takesNew = (be) => be.up && !be.draining && be.prefill !== false;

/** What the queue as a whole is blocked on. One phrase, shared by every
 * waiter, so it is said once above the list rather than in every row.
 * @param {Status} status @returns {string} */
export function poolReason(status) {
  const takers = backendsOf(status).filter(takesNew);
  if (!takers.length) return "no backend that takes new conversations is up";
  const full = takers.every((be) => slotsOf(be).length
    ? slotsOf(be).every((s) => s.busy)
    : be.busy >= be.slots);
  if (!full) return "a slot is free";
  // `takers` is already filtered by takesNew, which requires prefill, so
  // asking it again could only ever be false. The pool is what to ask.
  return backendsOf(status).some((be) => be.prefill === false)
    ? "every slot on a backend that prefills is busy"
    : "every slot is busy";
}

/** The soonest slot a waiter could get: a free one, else the reading slot
 * with the least left. A generating slot has no known end.
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

/** The slowest generating slot that has a reader beside it right now, or null.
 * This is the contended rate the slots view shows moving.
 * @param {Backend} be @returns {number | null} */
export function contendedRate(be) {
  let worst = null;
  for (const s of slotsOf(be)) {
    if (s.phase !== "generating" || !readerBeside(be, s)) continue;
    // A slot whose rate has not resolved yet says nothing about contention,
    // and counting its null as a zero would make it the worst every time.
    if (s.tg_rate == null) continue;
    if (worst === null || s.tg_rate < worst) worst = s.tg_rate;
  }
  return worst;
}

/** The share of prompt tokens the backends that read reused instead of
 * reading, from the token counts in /metrics. Null until one reports them.
 *
 * Only the backends that prefill. A conversation is carried to a generator
 * with its prompt already in the slot, so that instance reuses very nearly
 * every token it is ever given: working properly it sits at 99.9% and drags
 * the pooled figure up with it whatever the prefilling instances are
 * managing. This number is about the prefilling, so it counts the
 * prefilling. A generator's own figures are on the caching page.
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

/** What a waiter's status says when it differs from the queue's shared
 * reason. Empty when it is the same thing. @param {Waiter} w @param {string} reason */
export function waitLabel(w, reason) {
  if (w.wants === "turn") return "waits for the turn ahead of it in the same conversation";
  if (w.wants === "pinned") return `holds for ${w.backend || "its backend"}, its cache is there`;
  if (w.wants === "big") return "too big for what is free";
  return reason.includes("busy") ? "" : "needs a backend that prefills";
}

/**
 * The whole prompt of a busy slot, in three parts that add up to it: what was
 * reused from a cache, what has been read since, and what is still to read.
 * `cached` is the reused part, `done` what has been read, and `prompt` what
 * is left, so the whole is all three. Null when there is nothing to show: an
 * idle slot, whose counters reset to zero the moment its turn ends, or a slot
 * that has never run a task, which reports no prompt at all.
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

/** A cut is a point between messages where another conversation could pick
 * up: -1 is the system prompt alone, i is through message i + 1.
 * @param {number | null | undefined} i */
export function throughText(i) {
  if (i === null || i === undefined) return "-";
  return i < 0 ? "system prompt only" : `through message ${i + 1}`;
}

/** A deeper start is there to be had when something holds more of this
 * request's opening than any saved file does, past the system prompt. A slot
 * holding it is `shared`; another conversation's copy on disk is `copied`, and
 * counts because a copy restores through the same call a saved opening does.
 * @param {CacheChoice} c */
export function cutDeeper(c) {
  const held = Math.max(c.shared ?? -1, c.copied ?? -1);
  return held >= 0 && held > (c.stored ?? -1);
}

/** The answer to "why is no deeper prompt ever saved", from the recent
 * requests. @param {Status} status @returns {string} */
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

/** True when a node's page cache holds less than its backends must keep
 * resident, so prefill will read weights from disk. Never true while the
 * requirement is unknown. @param {Node} node */
export function cacheShort(node) {
  return node.resident_bytes > 0 && node.cache != null && node.cache < node.resident_bytes;
}

/** Slot-seconds by phase, for one bucket of history. */
/** @typedef {{ read: number, gen: number, stalled: number, secs: number }} Bucket */
