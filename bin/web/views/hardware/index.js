// @ts-check
/**
 * Hardware: is each backend doing what it is good at?
 *
 * An instance is configured to prefill, to generate, or to do both, and the
 * question this view answers is whether each is spending its time on the one
 * it is good at. Three blocks: what every slot is doing now, the rates per
 * backend, and ten minutes of slot-time by phase.
 */
import { loadTemplates, tpl, pick, mount } from "../../lib/templates.js";
import { renderRegion } from "../../lib/render.js";
import { num, time } from "../../lib/format.js";
import { subscribe } from "../feed.js";
import { isStalled, contendedRate, cacheShort, backendsOf, slotsOf } from "../status.js";

/** @typedef {import("../status.js").Status} Status */
/** @typedef {import("../status.js").Backend} Backend */
/** @typedef {import("../status.js").Bucket} Bucket */
/** @typedef {import("../status.js").Series} Series */
/** @typedef {import("../status.js").Node} Node */

const GIB = 1024 ** 3, MIB = 1024 ** 2;
/** @param {number} b */
const gib = (b) => `${(b / GIB).toFixed(b < 10 * GIB ? 1 : 0)} GiB`;

const SVG = "http://www.w3.org/2000/svg";

/** @param {string} name @param {Record<string, string | number>} attrs */
function el(name, attrs) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

// ---------------------------------------------------------------- machine
const SW = 240, SH = 44, SP = 4;

/** Draw a series as an area with a line, on a 0..max scale, the last value
 * marked. Nulls break the line. @param {Element} g @param {(number | null)[]} values
 * @param {number} max @param {number} cols */
function spark(g, values, max, cols) {
  const x = (/** @type {number} */ i) => SP + ((SW - 2 * SP) * i) / Math.max(1, cols - 1);
  const y = (/** @type {number} */ v) => SP + (SH - 2 * SP) * (1 - Math.min(v, max) / max);
  for (const gy of [0, 0.5, 1]) g.appendChild(el("line", { class: "grid", x1: SP, x2: SW - SP, y1: y(gy * max), y2: y(gy * max) }));
  const offset = cols - values.length;   // right-align: the newest is the last column
  let line = "", area = "", open = false;
  /** @type {{ px: string, py: string } | null} */ let last = null;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null || v === undefined) {
      if (open) { area += ` L${x(i - 1 + offset).toFixed(1)},${y(0).toFixed(1)} Z`; open = false; }
      continue;
    }
    const px = x(i + offset).toFixed(1), py = y(v).toFixed(1);
    if (!open) { line += ` M${px},${py}`; area += ` M${px},${y(0).toFixed(1)} L${px},${py}`; open = true; }
    else { line += ` L${px},${py}`; area += ` L${px},${py}`; }
    last = { px, py };
  }
  if (open && last) area += ` L${last.px},${y(0).toFixed(1)} Z`;
  if (area) g.appendChild(el("path", { class: "area", d: area.trim() }));
  if (line) g.appendChild(el("path", { class: "line", d: line.trim() }));
  if (last) g.appendChild(el("circle", { class: "dot", cx: last.px, cy: last.py, r: 3 }));
}

/** A load series as one array, oldest first, the bucket in progress last.
 * @param {Status} status @param {string} key @returns {(number | null)[]} */
function series(status, key) {
  const row = status.history?.load?.[key];
  return row ? [...row.done, row.cur] : [];
}

/** @param {Status} status @returns {DocumentFragment} */
function buildMachine(status) {
  const frag = new DocumentFragment();
  const m = status.machine;
  if (!m) return frag;
  const cols = (status.history?.keep || 60) + 1;
  for (const node of m.nodes) {
    const row = tpl("tpl-hw-node");
    pick(row, "name").textContent = `node ${node.id}`;
    pick(row, "runs").textContent = node.backends.length ? node.backends.join(", ") : `${node.cpus} cpus`;
    spark(pick(row, "cpuPlot"), series(status, `node${node.id}.cpu`), 100, cols);
    pick(row, "cpuNow").textContent = node.cpu === null || node.cpu === undefined ? "-" : `${node.cpu.toFixed(0)}%`;
    const total = node.total || 0, cache = node.cache || 0, free = node.free || 0;
    const pct = (/** @type {number} */ b) => (total ? `${((100 * b) / total).toFixed(1)}%` : "0");
    pick(row, "cache").style.width = pct(cache);
    pick(row, "free").style.width = pct(free);
    const mark = pick(row, "mark");
    const short = cacheShort(node);
    mark.hidden = !node.resident_bytes || !total;
    mark.style.left = pct(Math.min(node.resident_bytes, total));
    mark.classList.toggle("short", short);
    const text = pick(row, "memText");
    text.textContent = total
      ? `page cache ${gib(cache)} · free ${gib(free)}${node.resident_bytes ? ` · model in memory ${gib(node.resident_bytes)}` : ""}`
      : "-";
    text.classList.toggle("warn", short);
    frag.appendChild(row);
  }
  if (m.gpu) {
    const row = tpl("tpl-gpu");
    spark(pick(row, "utilPlot"), series(status, "gpu.util"), 100, cols);
    pick(row, "utilNow").textContent = `${m.gpu.util.toFixed(0)}%`;
    pick(row, "vram").style.width = m.gpu.vram_total ? `${((100 * m.gpu.vram_used) / m.gpu.vram_total).toFixed(1)}%` : "0";
    pick(row, "vramText").textContent = `${Math.round(m.gpu.vram_used / MIB).toLocaleString()} of ${Math.round(m.gpu.vram_total / MIB).toLocaleString()} MiB`;
    frag.appendChild(row);
  }
  return frag;
}

/** @param {Status} status */
function machineNote(status) {
  // Telemetry, not prefill: "reading" everywhere else on this page means
  // reading a prompt, and this is about nvidia-smi having nothing to say.
  return status.machine?.gpu ? "" : "no GPU telemetry";
}

// ---------------------------------------------------------------- now
/** @param {Status} status @returns {DocumentFragment} */
function buildSplits(status) {
  const frag = new DocumentFragment();
  for (const be of backendsOf(status)) {
    const row = tpl("tpl-split");
    pick(row, "name").textContent = be.name;
    const slots = slotsOf(be);
    const read = slots.filter((s) => s.phase === "reading").length;
    const stalled = slots.filter(isStalled).length;
    const gen = slots.filter((s) => s.phase === "generating").length - stalled;
    const pct = (/** @type {number} */ n) => `${be.slots ? (100 * n) / be.slots : 0}%`;
    pick(row, "read").style.width = pct(read);
    pick(row, "gen").style.width = pct(gen);
    pick(row, "stalled").style.width = pct(stalled);
    const parts = [];
    if (read) parts.push(`${read} reading`);
    if (gen) parts.push(`${gen} generating`);
    if (stalled) parts.push(`${stalled} stalled`);
    pick(row, "text").textContent = !be.up ? "down" : parts.length ? parts.join(" · ") : "idle";
    frag.appendChild(row);
  }
  return frag;
}

// ---------------------------------------------------------------- rates
/** @param {Status} status @returns {DocumentFragment} */
function buildHeads(status) {
  const frag = new DocumentFragment();
  // The row's first cell names the row, not a backend. renderRegion replaces
  // every child, so the corner has to be rebuilt here or each name lands one
  // column left of its own figures.
  frag.appendChild(document.createElement("th"));
  for (const be of backendsOf(status)) {
    const th = tpl("tpl-head");
    pick(th, "name").textContent = be.name;
    frag.appendChild(th);
  }
  return frag;
}

/** @typedef {{ value: number | null, text: string, tip: string, cls: string }} Cell */

/** @param {Status} status @returns {DocumentFragment} */
function buildRates(status) {
  const bes = backendsOf(status);
  /** @type {Array<{ label: string, tip: string, cls: string, cells: Cell[] }>} */
  const rows = [
    { label: "reading", cls: "reading", tip: "Prompt tokens a second, averaged over every request since the backend started.",
      cells: bes.map((be) => number(be.stats?.pp_rate)) },
    { label: "generating", cls: "", tip: "Tokens a second, averaged over every request since the backend started.",
      cells: bes.map((be) => number(be.stats?.tg_rate)) },
    { label: "contended", cls: "stalled", tip: "Generating while another slot on the same backend reads a prompt.",
      cells: bes.map((be) => {
        const worst = contendedRate(be);
        if (worst !== null) return { value: worst, text: worst.toFixed(2), tip: "", cls: "stalled" };
        return { value: null, text: "n/a", cls: "", tip: be.slots <= 1
          ? "One slot: nothing on it can starve it."
          : "No slot is generating beside a read right now." };
      }) },
  ];
  const frag = new DocumentFragment();
  for (const row of rows) {
    const tr = tpl("tpl-rate-row");
    const label = pick(tr, "label");
    label.textContent = row.label;
    label.title = row.tip;
    const top = Math.max(0, ...row.cells.map((c) => c.value ?? 0));
    const host = tr.firstElementChild;
    for (const cell of row.cells) {
      const td = tpl("tpl-rate-cell");
      const value = pick(td, "value");
      value.textContent = cell.text;
      value.title = cell.tip;
      value.classList.toggle("dim", cell.value === null);
      const fill = pick(td, "fill");
      fill.style.width = cell.value === null || !top ? "0" : `${Math.max(1, (100 * cell.value) / top)}%`;
      fill.className = cell.cls || row.cls;
      host?.appendChild(td);
    }
    frag.appendChild(tr);
  }
  return frag;
}

/** One row per setting, one column per backend. @param {Status} status @returns {DocumentFragment} */
function buildSettings(status) {
  const bes = backendsOf(status);
  /** @type {Array<{ label: string, tip: string, of: (be: Backend) => string }>} */
  const rows = [
    { label: "node", tip: "The NUMA node it is bound to.", of: (be) => (be.node === null || be.node === undefined ? "-" : `${be.node}`) },
    { label: "slots", tip: "Conversations it serves at once.", of: (be) => `${be.config?.n_slots ?? be.slots}` },
    { label: "context", tip: "Tokens each slot can hold.", of: (be) => (be.config?.n_ctx ? num(be.config.n_ctx) : "-") },
    { label: "batch", tip: "Prompt tokens per step while reading. A bigger batch reads faster and holds up the other slots longer.", of: (be) => (be.config?.n_batch ? num(be.config.n_batch) : "-") },
    { label: "micro-batch", tip: "Tokens per forward pass inside a batch.", of: (be) => (be.config?.n_ubatch ? num(be.config.n_ubatch) : "-") },
    { label: "shared kv", tip: "Slots share one cache layout, which a copy between backends needs.", of: (be) => (be.config?.kv_unified === undefined ? "-" : be.config.kv_unified ? "yes" : "no") },
  ];
  const frag = new DocumentFragment();
  for (const row of rows) {
    const tr = tpl("tpl-rate-row");
    const label = pick(tr, "label");
    label.textContent = row.label;
    label.title = row.tip;
    const host = tr.firstElementChild;
    for (const be of bes) {
      const td = tpl("tpl-setting-cell");
      pick(td, "value").textContent = row.of(be);
      host?.appendChild(td);
    }
    frag.appendChild(tr);
  }
  return frag;
}

/** @param {Status} status */
function ratesWindow(status) {
  return status.rates_since ? `since ${time(status.rates_since * 1000)}` : "since each backend started";
}

/** @param {number | undefined} v @returns {Cell} */
function number(v) {
  return v ? { value: v, text: `${v}`, tip: "", cls: "" } : { value: null, text: "-", tip: "No request finished yet.", cls: "" };
}

// ---------------------------------------------------------------- the last ten minutes
const W = 640, H = 150, L = 34, R = 8, T = 10, B = 20, GAP = 2;

/** @param {Backend} be @param {import("../status.js").ServerHistory} server
 * @returns {DocumentFragment} */
function buildChart(be, server) {
  const fig = tpl("tpl-chart");
  pick(fig, "name").textContent = be.name;
  pick(fig, "slots").textContent = `${be.slots} slot${be.slots === 1 ? "" : "s"}`;
  const plot = pick(fig, "plot");
  const step = server.step;
  const cap = Math.max(1, be.slots) * step;
  const COLS = server.keep + 1;
  const y = (/** @type {number} */ v) => T + (H - T - B) * (1 - v / cap);
  for (const g of [0, 0.5, 1]) {
    plot.appendChild(el("line", { class: "grid", x1: L, y1: y(g * cap), x2: W - R, y2: y(g * cap) }));
    const label = el("text", { x: L - 4, y: y(g * cap) + 4, "text-anchor": "end" });
    label.textContent = `${g * be.slots}`;
    plot.appendChild(label);
  }
  // A backend the router has never seen up has no row, and nothing to draw.
  const { done, cur } = server.backends[be.name] ?? { done: [], cur: null };
  const cw = (W - L - R - GAP * (COLS - 1)) / COLS;
  for (let i = 0; i < COLS; i++) {
    const x = L + i * (cw + GAP);
    /** @type {Bucket | undefined} */
    const b = i === COLS - 1 ? cur : done[done.length - (COLS - 1) + i];
    if (b) {
      const span = Math.min(step, b.secs) / step;   // a partial bucket is a shorter column
      const idle = Math.max(0, cap * span - b.read - b.gen - b.stalled);
      let base = 0;
      const seg = (/** @type {number} */ v, /** @type {Record<string, string>} */ attrs) => {
        if (v <= 0) return;
        const top = y(base + v), bottom = y(base);
        base += v;
        plot.appendChild(el("rect", { x: x.toFixed(1), y: top.toFixed(1), width: cw.toFixed(1),
          height: Math.max(0, bottom - top - 2).toFixed(1), rx: 2, ...attrs }));
      };
      seg(b.gen, { class: "gen" });
      seg(b.stalled, { fill: "url(#hatch-stall)" });
      seg(b.read, { fill: "url(#hatch-read)" });
      seg(idle, { class: "idle" });
    }
    const tick = i === COLS - 1 ? "now" : i === 0 ? "-10 min" : i === Math.floor((COLS - 1) / 2) ? "-5" : "";
    if (tick) {
      const label = el("text", { x: (x + cw / 2).toFixed(1), y: H - 5, "text-anchor": "middle" });
      label.textContent = tick;
      plot.appendChild(label);
    }
  }
  const unit = el("text", { x: L, y: T - 2, "font-size": 10 });
  unit.textContent = "slots busy";
  plot.appendChild(unit);
  return fig;
}

/** @param {Status} status @returns {DocumentFragment} */
function buildCharts(status) {
  const frag = new DocumentFragment();
  if (!status.history) return frag;
  for (const be of backendsOf(status)) frag.appendChild(buildChart(be, status.history));
  return frag;
}

/** @param {Status} status */
function sinceText(status) {
  const at = status.history?.since;
  return at ? `since ${time(at * 1000)}` : "";
}

export default {
  id: "hardware",

  /** @param {HTMLElement} container @param {unknown} _data
   *  @param {{ loadCSS: Function, every: Function, signal: AbortSignal }} helpers */
  async mount(container, _data, { loadCSS, signal }) {
    loadCSS(import.meta.url, "../shared.css", signal);
    loadCSS(import.meta.url, "./style.css", signal);
    await loadTemplates(new URL("./hardware.html", import.meta.url).href,
                        { signal });
    // The shell aborts this controller when the reader clicks another
    // view. Without the check a mount cancelled mid-fetch carried on and
    // painted over whatever mounted after it - and worse, subscribe() and
    // every() register their teardown on `signal`, which never fires again
    // once it has aborted, so the dead view kept its SSE subscriber and its
    // interval for the life of the tab. flow/index.js has had this guard.
    if (signal.aborted) throw new DOMException("mount cancelled", "AbortError");

    mount(container, tpl("tpl-hardware"));
    const root = /** @type {HTMLElement} */ (container.querySelector(".hardware"));
    const splits = pick(root, "splits");
    const heads = pick(root, "heads");
    const rates = pick(root, "rates");
    const ratesSince = pick(root, "ratesSince");
    const settingHeads = pick(root, "settingHeads");
    const settings = pick(root, "settings");
    pick(root, "resetRates").addEventListener("click", () => {
      fetch("/router/reset-rates", { method: "POST", signal }).catch((err) => window.reportError(err));
    }, { signal });
    const charts = pick(root, "charts");
    const since = pick(root, "since");
    const machine = pick(root, "machine");
    const machineNoteEl = pick(root, "machineNote");

    subscribe(
      /** @param {Status} status */
      (status) => {
        // Signed off the slice each region renders, which is what the two
        // `names` sigs below already did. The rest signed the whole payload,
        // and the payload carries the history buckets: one of those moves
        // every second, so those sigs never matched and the settings table -
        // which changes only when a backend restarts - was rebuilt once a
        // second for the life of the tab.
        const bes = backendsOf(status);
        const names = bes.map((b) => b.name).join(",");
        const noteText = machineNote(status);
        const windowText = ratesWindow(status);
        renderRegion(machine, () => buildMachine(status),
                     { sig: `l${status.history?.keep}|${JSON.stringify(status.machine)}` });
        renderRegion(machineNoteEl, () => document.createTextNode(noteText), { sig: `n${noteText}` });
        renderRegion(splits, () => buildSplits(status),
                     { sig: `s${JSON.stringify(bes.map((b) => [b.name, b.slots_detail]))}` });
        renderRegion(heads, () => buildHeads(status), { sig: `h${names}` });
        renderRegion(rates, () => buildRates(status),
                     { sig: `r${JSON.stringify(bes.map((b) => [b.name, b.stats]))}` });
        renderRegion(ratesSince, () => document.createTextNode(windowText), { sig: `w${windowText}` });
        renderRegion(settingHeads, () => buildHeads(status), { sig: `g${names}` });
        renderRegion(settings, () => buildSettings(status),
                     { sig: `c${JSON.stringify(bes.map((b) => [b.name, b.node, b.slots, b.config]))}` });
        // The bucket in progress moves with the clock, not just the payload.
        renderRegion(charts, () => buildCharts(status), { force: true });
        since.textContent = sinceText(status);
      },
      signal,
    );
  },

  unmount() {},
};
