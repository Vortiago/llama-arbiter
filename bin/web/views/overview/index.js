// @ts-check
/**
 * Right now: what every slot is doing, what the queue is blocked on, and the
 * last cache that moved. Rows are reconciled in place, so a bar can advance
 * and a tape can gain a tick between two payloads.
 */
import { loadTemplates, tpl, pick, mount } from "../../lib/templates.js";
import { renderRegion, reconcileList } from "../../lib/render.js";
import { num, time } from "../../lib/format.js";
import { subscribe } from "../feed.js";
import { isStalled, readerBeside, secondsLeft, poolReason, nextFree, backendsOf, slotsOf,
         reuseShare, waitLabel, promptBands, TAPE_SECONDS } from "../status.js";

/** @typedef {import("../status.js").Status} Status */
/** @typedef {import("../status.js").Backend} Backend */
/** @typedef {import("../status.js").Slot} Slot */
/** @typedef {import("../status.js").FileEvent} FileEvent */
/** @typedef {{ kind: "backend", be: Backend } | { kind: "slot", be: Backend, sl: Slot }} Row */

const MIB = 1024 * 1024;
const SVG = "http://www.w3.org/2000/svg";
const TAPE_WIDTH = 144;
const PHASE_LABEL = { idle: "idle", reading: "reading prompt", generating: "generating" };
/** What the router calls a file event, said plainly. @type {Record<string, string>} */
const DID = { "loaded opening": "loaded saved prompt", "kept opening": "saved prompt" };
const CLOCK = /** @type {Intl.DateTimeFormatOptions} */ ({ hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** @param {number} value */
const mib = (value) => `${num(Math.round(value / MIB))} MiB`;
/** @param {number} s */
const secs = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
/** @param {number | undefined} live */
const rate = (live) => (live ? `${live}` : "-");
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------- totals
/** The totals row's numbers. Derived once, so the region's sig and its render agree.
 * @param {Status} status */
function totalsOf(status) {
  const up = backendsOf(status).filter((b) => b.up);
  return {
    busy: up.reduce((t, b) => t + (slotsOf(b).length ? slotsOf(b).filter((s) => s.busy).length : b.active), 0),
    slots: up.reduce((t, b) => t + b.slots, 0),
    stalls: up.reduce((t, b) => t + slotsOf(b).filter(isStalled).length, 0),
    waiting: status.waiting || 0,
    toGenerate: status.waiting_to_generate || 0,
    reused: reuseShare(status),
  };
}

/** @param {Status} status @returns {DocumentFragment} */
function buildTotals(status) {
  const { busy, slots, stalls, waiting, toGenerate, reused } = totalsOf(status);
  const frag = new DocumentFragment();
  /** @param {string} value @param {string} label @param {boolean} warn @param {string} tip */
  const add = (value, label, warn, tip) => {
    const stat = tpl("tpl-stat");
    pick(stat, "value").textContent = value;
    pick(stat, "label").textContent = label;
    const root = stat.firstElementChild;
    if (root instanceof HTMLElement) { root.classList.toggle("warn", warn); root.title = tip; }
    frag.appendChild(stat);
  };
  add(`${busy} / ${slots}`, "slots busy", false, "");
  add(String(waiting), "waiting", waiting > 0, "Requests with no slot yet.");
  add(String(toGenerate), "queued to generate", toGenerate > 1,
      "Turns whose prompt is read and parked, waiting for a slot on the "
      + "instance that generates. They hold no backend while they wait, so a "
      + "prefiller is free to take the next prompt.");
  add(String(stalls), "slots stalled", stalls > 0,
      "Generating under 0.5 tokens a second while another slot on the same backend reads.");
  if (reused !== null) {
    add(`${reused.toFixed(0)}%`, "prompt tokens reused", false,
        "The instances that prefill, since they started, from /metrics: tokens "
        + "reused against tokens read. An instance that only generates is left "
        + "out - a prompt reaches it already read, so it reuses nearly all of "
        + "them - and is on the caching page.");
  }
  return frag;
}

/** One row per waiter, with only what differs. @param {Status} status @returns {DocumentFragment} */
function buildWaiting(status) {
  const frag = new DocumentFragment();
  const reason = poolReason(status);
  for (const w of status.waiting_detail || []) {
    const row = tpl("tpl-wait-row");
    pick(row, "conv").textContent = w.conv || "?";
    const waited = pick(row, "waited");
    waited.textContent = secs(w.waited);
    waited.classList.toggle("warn", w.waited > 60);
    pick(row, "tokens").textContent = num(w.tokens);
    pick(row, "status").textContent = waitLabel(w, reason);
    frag.appendChild(row);
  }
  return frag;
}

/** One line for the whole queue: how many, why, and the soonest slot. @param {Status} status */
function waitingText(status) {
  const waiting = status.waiting || 0;
  if (!waiting) return "nobody";
  const next = nextFree(status);
  const when = !next ? "no known end"
    : !next.seconds ? `${next.backend} slot ${next.slot}, free`
    : `${next.backend} slot ${next.slot} in ~${secs(next.seconds)}`;
  return `${waiting} · ${poolReason(status)} · next: ${when}`;
}

// ---------------------------------------------------------------- rows
/** @param {Status} status @returns {Row[]} */
function rowsOf(status) {
  /** @type {Row[]} */ const rows = [];
  for (const be of backendsOf(status)) {
    rows.push({ kind: "backend", be });
    for (const sl of slotsOf(be)) rows.push({ kind: "slot", be, sl });
  }
  return rows;
}

/** @param {Row} row */
const keyOf = (row) => (row.kind === "backend" ? `b:${row.be.name}` : `s:${row.be.name}:${row.sl.id}`);

/** @param {Row} row @returns {Element} */
function create(row) {
  const el = (row.kind === "backend" ? tpl("tpl-backend") : tpl("tpl-slot")).firstElementChild;
  if (!el) throw new Error("empty row template");
  update(el, row);
  return el;
}

/** @param {Element} el @param {Row} row */
function update(el, row) {
  if (row.kind === "backend") updateBackend(el, row.be);
  else updateSlot(el, row.be, row.sl);
}

/** @param {Element} el @param {Backend} be */
function updateBackend(el, be) {
  pick(el, "name").textContent = be.name;
  if (el instanceof HTMLElement) el.title = be.model || "";
  const flag = pick(el, "flag");
  flag.hidden = be.prefill !== false;
  flag.textContent = "does not read";
  flag.title = "New conversations go to a backend that prefills. This one only generates.";
  const state = pick(el, "state");
  const st = !be.up ? "down" : be.draining ? "draining" : "up";
  state.textContent = st;
  state.className = `phase ${st}`;
  state.title = st === "draining" ? "Taking no new requests. Its caches go to disk once its slots finish." : "";
  const active = slotsOf(be).length ? slotsOf(be).filter((s) => s.busy).length : be.active;
  pick(el, "slots").textContent = `${active} / ${be.slots}`;
  pick(el, "prompt").textContent = rate(be.stats?.pp_live);
  pick(el, "output").textContent = rate(be.stats?.tg_live);
}

/** Restart a one-shot CSS animation on `el`. @param {Element} el @param {string} cls */
function restart(el, cls) {
  el.classList.remove(cls);
  el.getBoundingClientRect();   // flush, so adding the class again starts it again
  el.classList.add(cls);
}

/** @param {Element} el @param {Backend} be @param {Slot} sl */
function updateSlot(el, be, sl) {
  const stalled = isStalled(sl);
  const shown = stalled ? "stalled" : sl.phase;
  pick(el, "id").textContent = `slot ${sl.id}`;

  const badge = pick(el, "phase");
  badge.textContent = stalled ? "stalled" : PHASE_LABEL[sl.phase] || sl.phase;
  badge.className = `phase ${shown}`;
  badge.title = stalled ? "Generating under 0.5 tokens a second: another slot on this backend is reading."
    : sl.phase === "reading" ? "Sends nothing to the client until the prompt is read." : "";
  if (el instanceof HTMLElement) {
    if (el.dataset.phase && el.dataset.phase !== shown && !reduced()) restart(el, "changed");
    el.dataset.phase = shown;
  }

  // One bar for the whole prompt: the reused band is calm, the read band is active.
  const bands = promptBands(sl);
  const reused = pick(el, "reused"), progress = pick(el, "progress");
  const pct = (/** @type {number} */ n) => (bands ? `${((100 * n) / bands.total).toFixed(1)}%` : "0");
  reused.style.width = pct(bands ? bands.reused : 0);
  progress.style.width = pct(bands ? bands.read : 0);   // sits beside the reused band
  progress.className = sl.phase === "reading" ? "reading" : stalled ? "stalled" : "";

  pick(el, "prompt").textContent = sl.phase === "reading" && sl.pp_rate ? `${sl.pp_rate}` : "";
  // Blank until a rate resolves, never "null".
  pick(el, "output").textContent =
    sl.phase === "generating" && sl.tg_rate != null ? `${sl.tg_rate}` : "";

  const detail = pick(el, "detail"), stall = pick(el, "stall"), eta = pick(el, "eta");
  if (sl.phase === "reading") {
    detail.textContent = !bands ? "starting"
      : `${bands.reused ? `${num(bands.reused)} reused · ` : ""}`
        + `${num(bands.reused + bands.read)} / ${num(bands.total)} done`;
    const left = secondsLeft(sl);
    eta.textContent = left !== null ? `~${secs(left)} left` : sl.prompt ? "rate not resolved yet" : "";
    stall.textContent = "";
  } else if (sl.phase === "generating") {
    detail.textContent = `${bands && bands.reused ? `${num(bands.reused)} reused · ` : ""}${num(sl.decoded)} tokens out`;
    eta.textContent = "";
    const reader = readerBeside(be, sl);
    stall.textContent = stalled && sl.tg_rate != null
      ? `stalled · ${sl.tg_rate.toFixed(2)} /s${reader ? ` while slot ${reader.id} reads` : ""}` : "";
  } else {
    detail.textContent = stall.textContent = eta.textContent = "";
  }
  tape(el, sl);
}

// ---------------------------------------------------------------- the token tape
/** @typedef {{ phase: string, decoded: number, done: number, at: number, ticks: number[] }} Tape */
/** @type {WeakMap<Element, Tape>} */
const tapes = new WeakMap();

/** Place one tick by its age. @param {Element} line @param {number} at @param {number} now */
function place(line, at, now) {
  const x = (TAPE_WIDTH * (1 - (now - at) / TAPE_SECONDS)).toFixed(1);
  line.setAttribute("x1", x);
  line.setAttribute("x2", x);
}

/** Redraw a row's ticks for the current moment. @param {Element} el @param {Tape} t @param {number} now */
function drawTape(el, t, now) {
  t.ticks = t.ticks.filter((at) => at > now - TAPE_SECONDS);
  reconcileList(pick(el, "ticks"), t.ticks, (at) => at.toFixed(3), (at) => {
    const line = document.createElementNS(SVG, "line");
    line.setAttribute("class", t.phase === "reading" ? "pp" : "tg");
    line.setAttribute("y1", "3");
    line.setAttribute("y2", "13");
    place(line, at, now);
    return line;
  }, (line, at) => place(line, at, now));
}

/** Add a tick per token landed since the last payload, spread across the gap.
 * @param {Element} el @param {Slot} sl */
function tape(el, sl) {
  const now = Date.now() / 1000;
  let t = tapes.get(el);
  if (!t) { t = { phase: sl.phase, decoded: sl.decoded, done: sl.done, at: now, ticks: [] }; tapes.set(el, t); }
  if (t.phase !== sl.phase) { t.ticks = []; t.phase = sl.phase; t.decoded = sl.decoded; t.done = sl.done; }
  let landed = 0;
  if (sl.phase === "generating") { landed = Math.max(0, sl.decoded - t.decoded); t.decoded = sl.decoded; }
  else if (sl.phase === "reading") {
    landed = Math.max(0, Math.floor(sl.done / 100) - Math.floor(t.done / 100));
    t.done = sl.done;
  }
  for (let k = 1; k <= landed; k++) t.ticks.push(t.at + ((now - t.at) * k) / landed);
  t.at = now;
  drawTape(el, t, now);
}

/** Ticks age even when no payload arrives: slide them each second. @param {Element} rows */
function slide(rows) {
  const now = Date.now() / 1000;
  for (const el of rows.children) {
    const t = tapes.get(el);
    if (t) drawTape(el, t, now);
  }
}

// ---------------------------------------------------------------- the lane
/** Newest file event seen, so a remount does not replay it. @type {number | null} */
let lastFileAt = null;

/** @param {FileEvent} ev */
const describe = (ev) => `${DID[ev.did] || ev.did} ${ev.name}${ev.bytes ? ` · ${mib(ev.bytes)}` : ""}`;

/** Cross the lane once, for as long as the copy took. The SATA disk writes at
 * 520 MB/s. A load has no size.
 * @param {FileEvent} ev @param {HTMLElement} lane @param {AbortSignal} signal */
function transfer(ev, lane, signal) {
  const toDisk = ev.did === "parked" || ev.did.startsWith("kept");
  const where = `${ev.backend} slot ${ev.slot}`;
  pick(lane, "laneFrom").textContent = toDisk ? where : "disk";
  pick(lane, "laneTo").textContent = toDisk ? "disk" : where;
  const packet = pick(lane, "packet");
  packet.textContent = describe(ev);
  if (reduced()) return;
  lane.classList.remove("idle");
  const seconds = ev.bytes ? Math.min(4, Math.max(0.6, ev.bytes / 520e6)) : 0.6;
  const dx = Math.max(40, lane.clientWidth - 2 * 6.5 * 16 - packet.offsetWidth - 8);
  const run = packet.animate(
    [{ transform: "translate(0, -50%)", opacity: 1 }, { transform: `translate(${dx}px, -50%)`, opacity: 1 }],
    { duration: seconds * 1000, easing: "linear", fill: "forwards" });
  const settle = () => {
    if (signal.aborted) return;
    packet.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 600, fill: "forwards" })
      .finished.then(() => { if (!signal.aborted) lane.classList.add("idle"); }, () => {});
  };
  run.finished.then(settle, () => {});
  signal.addEventListener("abort", () => run.cancel(), { once: true });
}

/** @param {Status} status @param {HTMLElement} lane @param {HTMLElement} lastMove @param {AbortSignal} signal */
function seeFiles(status, lane, lastMove, signal) {
  const newest = (status.recent_files || [])[0];
  lastMove.textContent = newest ? `last: ${describe(newest)} · ${time(newest.at * 1000, CLOCK)}` : "nothing yet";
  if (!newest) return;
  if (lastFileAt !== null && newest.at > lastFileAt) transfer(newest, lane, signal);
  lastFileAt = Math.max(lastFileAt ?? 0, newest.at);
}

export default {
  id: "overview",

  /** @param {HTMLElement} container @param {unknown} _data
   *  @param {{ loadCSS: Function, every: Function, signal: AbortSignal }} helpers */
  async mount(container, _data, { loadCSS, every, signal }) {
    loadCSS(import.meta.url, "../shared.css", signal);
    loadCSS(import.meta.url, "./style.css", signal);
    await loadTemplates(new URL("./overview.html", import.meta.url).href,
                        { signal });
    // Stop a mount cancelled mid-fetch. subscribe() and every() register teardown
    // on `signal`, which never fires again once aborted, so a dead view would leak them.
    if (signal.aborted) throw new DOMException("mount cancelled", "AbortError");

    mount(container, tpl("tpl-overview"));
    const root = /** @type {HTMLElement} */ (container.querySelector(".overview"));
    const totals = pick(root, "totals");
    const waitingWhy = pick(root, "waitingWhy");
    const waiting = pick(root, "waiting");
    const rows = pick(root, "rows");
    const lane = pick(root, "lane");
    const lastMove = pick(root, "lastMove");

    subscribe(
      /** @param {Status} status */
      (status) => {
        // Sign each region with the slice it renders, never the whole payload:
        // the history buckets move every second.
        const whyText = waitingText(status);
        renderRegion(totals, () => buildTotals(status),
                     { sig: `t${JSON.stringify(totalsOf(status))}` });
        renderRegion(waitingWhy, () => document.createTextNode(whyText), { sig: `w${whyText}` });
        renderRegion(waiting, () => buildWaiting(status),
                     { sig: `q${poolReason(status)}|${JSON.stringify(status.waiting_detail)}` });
        // In place, not a swap: a bar that keeps its node can animate to its new width.
        reconcileList(rows, rowsOf(status), keyOf, create, update);
        seeFiles(status, lane, lastMove, signal);
      },
      signal,
    );
    every(() => slide(rows), 1000, signal);
  },

  unmount() {},
};
