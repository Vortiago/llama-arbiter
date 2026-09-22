// @ts-check
/**
 * Flow: a prompt crosses the top, the two stores hold the floor, and the bank
 * on the right says what the caching saved.
 *
 * Everything that moves is a measurement. Wires run at the slot's real rate, a
 * transfer crosses in the time its bytes take, and the tape lands a bar when a
 * turn finishes. Nothing animates on a timer that is not tied to the payload.
 *
 * `--busy` and `--ok` are four units apart under protanopia, so every slot
 * states its phase in a glyph and a word. This file writes only widths,
 * transforms, `strokeDashoffset` and one custom property. Colour stays in CSS.
 */
import { loadTemplates, tpl, pick, slot, mount, loadCSS, every } from "../../lib/templates.js";
import { reconcileList } from "../../lib/render.js";
import { num } from "../../lib/format.js";
import { subscribe } from "../feed.js";
import {
  MARK_PERIOD, READ_RATE, PHASE, nodesOf, arrivalsOf, residency, holderOf, transferOf,
  since, tapeOf, skipped, shelvesOf, blocksOf, loadOf, parkedOf,
} from "./flow-model.js";
import { backendsOf, reuseShare, workLabel } from "../status.js";

/** @typedef {import("../status.js").Status} Status */
/** @typedef {import("./flow-model.js").SlotNode} SlotNode */

/** Turns kept on the tape. */
const TAPE = 14;
/** How long a hero figure takes to count to a new value. */
const COUNT_MS = 900;
/** Sorts last, so the unused budget sits at the right of the strip. */
const FREE = "zzz-free";

const GIB = 1024 ** 3, MIB = 1024 ** 2;
/** Binary units: PARK_BUDGET is 256 GiB. `lib/format.js` labels GB and is vendored.
 *  @param {number} n */
const fmtBytes = (n) => (n >= GIB ? `${(n / GIB).toFixed(n < 10 * GIB ? 1 : 0)} GiB` : `${Math.round(n / MIB)} MiB`);

/** @param {number} s */
const secs = (s) => (s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Resolve once a stylesheet has applied, or the mount is abandoned.
 * @param {HTMLLinkElement} link @param {AbortSignal} signal @returns {Promise<void>} */
function styled(link, signal) {
  if (link.sheet) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => resolve();
    link.addEventListener("load", done, { once: true });
    link.addEventListener("error", done, { once: true });
    signal.addEventListener("abort", done, { once: true });
  });
}

/** @param {string} cls @param {string} text */
function spanOf(cls, text) {
  const s = document.createElement("span");
  s.className = cls;
  s.textContent = text;
  return s;
}

export default {
  id: "flow",

  /** @param {HTMLElement} container @param {unknown} _data
   *  @param {{ signal: AbortSignal }} helpers */
  async mount(container, _data, { signal }) {
    // Not shared.css: this view uses none of it, and `.lane` collided at equal specificity.
    const link = loadCSS(import.meta.url, "./style.css", signal);
    // Wait for the sheet: startViewTransition snapshots when the callback
    // resolves, so an unstyled capture is a white flash, and rects measured
    // before the sheet lands are wrong. Fetch the markup in parallel.
    await Promise.all([
      styled(link, signal),
      loadTemplates(new URL("./flow.html", import.meta.url).href),
    ]);
    if (signal.aborted) throw new DOMException("mount cancelled", "AbortError");

    mount(container, tpl("tpl-flow"));
    const root = /** @type {HTMLElement} */ (container.querySelector(".flowview"));
    const board = pick(root, "board"), stage = pick(root, "stage"), wires = pick(root, "wires");
    const heroEl = pick(root, "hero"), estEl = pick(root, "estimated");
    // Both queues wear the same gate class, so pick each by name.
    const arrivingGate = pick(root, "arrivingGate"), parkedGate = pick(root, "parkedGate");
    const hosts = {
      arrivals: pick(root, "arrivals"), readers: pick(root, "readers"),
      generators: pick(root, "generators"), tape: pick(root, "tape"),
      shelves: pick(root, "shelves"), strip: pick(root, "strip"),
      copiesTwin: pick(root, "copiesTwin"), openingsTwin: pick(root, "openingsTwin"),
    };
    root.style.setProperty("--f-period", `${MARK_PERIOD}px`);

    /** Every animation this view starts. Abort cancels them all: an infinite
     *  animation on a detached path keeps ticking. @type {Set<Animation>} */
    const anims = new Set();
    /** Paused when the tab went away. Only these resume on return. @type {Set<Animation>} */
    const parked = new Set();
    /** @type {Map<string, Animation>} */ const marching = new Map();
    /** When each slot's `done` last moved. A reader's rate reads 0 until the
     *  window closes, so a still counter is the only evidence of a stall.
     *  @type {Map<string, { done: number, phase: string, at: number }>} */
    const steps = new Map();
    /** @type {Status | null} */ let latest = null;
    /** performance.now() when `latest` arrived. Waits age by local elapsed time, never by comparing clocks. */
    let latestAt = 0;
    /** The nodes the last paint drew, for a resize to re-place. @type {SlotNode[]} */ let shown = [];
    /** @type {Status | null} */ let pending = null;
    /** @type {number | null} */ let lastFile = null;
    /** False until the first payload, whose file log is history, not a backlog
     *  to fly. A flag, not a stamp: an empty first log leaves `since` echoing the seed. */
    let seenFiles = false;
    /** @type {Set<string>} */ let onTape = new Set();
    /** What the hero last drew, or null before the first paint. @type {number | null} */
    let heroShown = null;
    /** The count-up in flight, so the next one can cancel it. @type {number | null} */
    let counting = null;
    let laidOut = false;
    /** The elements the pointer lit, held rather than re-queried: `data-pair`
     *  is rewritten on every paint. */
    let lit = "";
    /** @type {Element[]} */ let litEls = [];

    // ------------------------------------------------------------ geometry
    /** Overlay-relative box of an element. @param {Element} el */
    const box = (el) => {
      const b = wires.getBoundingClientRect(), r = el.getBoundingClientRect();
      return { x: r.left - b.left, y: r.top - b.top, w: r.width, h: r.height };
    };
    /** @typedef {{ x: number, y: number, w: number, h: number }} Box */
    /** A run between two stations. `lane` spreads the origins down the source
     *  box, so wires leaving one queue do not stack.
     *  @param {Box} a @param {Box} b @param {number} [lane] @param {number} [lanes] */
    const across = (a, b, lane = 0, lanes = 1) => {
      const y1 = lanes > 1 ? a.y + (a.h * (lane + 1)) / (lanes + 1) : a.y + a.h / 2;
      const x1 = a.x + a.w, x2 = b.x, y2 = b.y + b.h / 2;
      const dx = Math.min(Math.abs(x2 - x1) / 2, 70);
      return `M${x1} ${y1} C${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`;
    };
    /** @param {Box} a @param {Box} b */
    const down = (a, b) => {
      const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y;
      const dy = Math.min(Math.abs(y2 - y1) / 2, 70);
      return `M${x1} ${y1} C${x1} ${y1 + dy} ${x2} ${y2 - dy} ${x2} ${y2}`;
    };

    // --------------------------------------------------------------- nodes
    /** @param {SlotNode} n */
    const makeNode = (n) => {
      const el = /** @type {HTMLElement} */ (tpl("tpl-node").firstElementChild);
      fillNode(el, n);
      return el;
    };
    /** @param {Element} el @param {SlotNode} n */
    function fillNode(el, n) {
      const node = /** @type {HTMLElement} */ (el);
      node.dataset.phase = n.phase;
      node.dataset.key = n.key;
      node.toggleAttribute("data-stuck", n.stuck);
      const state = n.stuck ? PHASE.stuck : PHASE[/** @type {keyof typeof PHASE} */ (n.phase)] ?? PHASE.idle;
      slot(node, { name: n.backend, icon: state.icon, phase: state.word });
      const conv = pick(node, "conv");
      conv.hidden = !n.conv;
      conv.textContent = n.conv ? n.conv.split("/")[0] : "";

      // A kind of work is not a phase: the card keeps its reading or
      // generating colour and wears the label beside it.
      const work = workLabel(n.kind);
      const kind = pick(node, "kind");
      kind.hidden = !work;
      if (work) {
        slot(node, { kindIcon: work.icon, kindWord: work.word });
        kind.title = work.why;
      }
      // Deleted, not blanked: `[data-kind]` would match an empty attribute.
      if (n.kind) node.dataset.kind = n.kind;
      else delete node.dataset.kind;

      const b = n.bands;
      const reused = pick(node, "reused"), read = pick(node, "read");
      if (n.generator) {
        // A generator fills a context window, so its band is context used, not prompt read.
        const share = n.nCtx ? (100 * n.ctx) / n.nCtx : 0;
        reused.style.width = `${share}%`;
        read.style.width = "0%";
        slot(node, { done: `${num(n.decoded)} out`, eta: n.nCtx ? `${Math.round(share)}% of context` : "" });
      } else {
        reused.style.width = b ? `${(100 * b.reused) / b.total}%` : "0%";
        read.style.width = b ? `${(100 * b.read) / b.total}%` : "0%";
        slot(node, { done: b ? `${num(b.reused)} reused of ${num(b.total)}` : "" });
      }
      slot(node, { rate: n.phase === "idle" || n.phase === "down" ? "" : `${n.rate.toFixed(1)} tok/s` });
      // Colour stays in CSS; JS only says how far up each band reaches.
      reconcileList(pick(node, "hist"), n.history, (b) => String(b.i),
        (b) => { const el = document.createElement("i"); band(el, b); return el; }, band);
      if (!n.generator) slot(node, { eta: n.left !== null ? `~${secs(n.left)} left` : "" });
    }

    /** One key per waiting ticket, not per conversation: a turn queued behind
     *  its own conversation is the commonest waiter.
     *  @param {import("./flow-model.js").Arrival} a */
    const waiterKey = (a) => `${a.conv}:${a.since}`;
    /** @param {import("./flow-model.js").Arrival} a */
    const makeWaiter = (a) => {
      const el = /** @type {HTMLElement} */ (tpl("tpl-waiter").firstElementChild);
      fillWaiter(el, a);
      return el;
    };
    /** Mark a request that carries pictures, and what they cost against its prompt.
     *  @param {HTMLElement} el @param {number} n @param {number} charged @param {number} whole */
    function pics(el, n, charged, whole) {
      el.hidden = !n;
      if (!n) return;
      el.textContent = `🖼 ${n}`;
      const share = whole ? Math.round((100 * charged) / whole) : 0;
      el.title = `${n} image${n === 1 ? "" : "s"}, ${num(charged)} tokens`
        + (share ? ` - ${share}% of this prompt` : "")
        + ". The vision encoder runs in RAM on whichever instance serves it, one with a GPU included.";
    }

    /** @param {Element} el @param {import("./flow-model.js").Parked} q */
    function fillParked(el, q) {
      slot(el, { size: fmtBytes(q.bytes), conv: q.conv.split("/")[0], waited: secs(q.waited) });
      /** @type {HTMLElement} */ (el).title =
        `${q.conv} - ${fmtBytes(q.bytes)} on disk, waiting ${secs(q.waited)} for the generator`;
    }
    /** @param {import("./flow-model.js").Parked} q */
    const mkParked = (q) => {
      const el = /** @type {HTMLElement} */ (tpl("tpl-parked").firstElementChild);
      fillParked(el, q);
      return el;
    };

    /** @param {Element} el @param {import("./flow-model.js").Bar} b */
    function band(el, b) {
      const s = /** @type {HTMLElement} */ (el).style;
      // Prefixed: a custom property inherits, and plain `--r` is shell.css's border radius.
      s.setProperty("--f-read", `${b.read}%`);
      s.setProperty("--f-gen", `${b.gen}%`);
      s.setProperty("--f-stalled", `${b.stalled}%`);
    }

    /** @param {Element} el @param {import("./flow-model.js").Arrival} a */
    function fillWaiter(el, a) {
      /** @type {HTMLElement} */ (el).dataset.kind = a.kind;
      slot(el, {
        conv: a.conv.split("/")[0], waited: secs(a.waited),
        why: a.why, tokens: num(a.tokens),
      });
      pics(pick(el, "waitPics"), a.images, a.imageTokens, a.tokens);
    }

    /** @param {SlotNode[]} nodes */
    function trackSteps(nodes) {
      const now = Date.now();
      for (const n of nodes) {
        const done = n.bands ? n.bands.read : 0;
        const seen = steps.get(n.key);
        // The phase is part of the mark: an idle slot and a read in its first
        // batch both stand at done 0.
        if (!seen || seen.done !== done || seen.phase !== n.phase) {
          steps.set(n.key, { done, phase: n.phase, at: now });
        }
      }
      for (const key of [...steps.keys()]) {
        if (!nodes.some((n) => n.key === key)) steps.delete(key);
      }
    }

    // ---------------------------------------------------------------- bank
    /** Count a figure up rather than snapping it.
     *  @param {HTMLElement} node @param {number} value @param {string} unit @param {string} label */
    function countTo(node, value, unit, label) {
      /** @param {number} v */
      const show = (v) => {
        node.replaceChildren(document.createTextNode(v.toFixed(1))); // static-render
        node.append(spanOf("unit", unit), spanOf("lbl", label));
      };
      // One count at a time, from the figure on screen. Two chains on one
      // node make the digits jump backwards.
      if (counting !== null) { cancelAnimationFrame(counting); counting = null; }
      const from = heroShown;
      heroShown = value;
      // null, not zero: before the first paint there is nothing to count from. Zero is a real figure.
      if (from === null || Math.abs(value - from) < 0.05 || reducedMotion()) { show(value); return; }
      const t0 = performance.now();
      /** @param {number} now */
      const step = (now) => {
        if (signal.aborted) return;
        const k = Math.min(1, (now - t0) / COUNT_MS);
        const at = from + (value - from) * (1 - Math.pow(1 - k, 3));
        show(at);
        // what the screen holds, so an interrupted count hands on a shown figure
        heroShown = at;
        counting = k < 1 ? requestAnimationFrame(step) : null;
        if (k >= 1) heroShown = value;
      };
      counting = requestAnimationFrame(step);
    }

    /** @param {Status} status */
    function bank(status) {
      const s = skipped(status);
      countTo(heroEl, s.hours, "h", "of reading skipped");
      heroEl.title = s.reused
        ? `${num(s.reused)} prompt tokens came out of a cache instead of being read at ${READ_RATE} a second`
        : "nothing read yet";

      const share = reuseShare(status);
      pick(root, "shareFill").style.width = `${share ?? 0}%`;
      // The counters come through since_reset, so a reset walks this figure back.
      const span = status.rates_since ? "since the counters were reset" : "since each backend started";
      slot(root, {
        shareNote: share === null ? "" : `${share.toFixed(1)}% reused, ${span}`,
      });

      const { groups, scale, exact } = tapeOf(status, TAPE);
      pick(root, "noTurns").hidden = groups.length > 0;
      estEl.hidden = exact || !groups.length;
      // The router sends null counts for a turn that never reached the read
      // path. One such turn makes the whole tape an estimate.
      estEl.title = "At least one of these turns never ran a read pass, so the backend "
        + "reported no split for it and its bar stands on this router's estimate "
        + "of the prompt instead.";
      const keys = new Set(groups.flatMap((g) => g.turns.map((t) => t.key)));
      /** @param {Element} el @param {import("./flow-model.js").Turn} t */
      const fillTurn = (el, t) => {
        // A row lands once, when the turn it stands for really finished.
        el.classList.toggle("fresh", onTape.size > 0 && !onTape.has(t.key));
        slot(el, { mode: t.mode, took2: secs(t.took) });
        pics(pick(el, "turnPics"), t.images, t.imageTokens, t.whole);
        // The hollow between the two bars is the saving.
        pick(el, "ghost").style.width = `${(100 * t.coldCost) / scale}%`;
        pick(el, "took").style.width = `${(100 * t.took) / scale}%`;
        /** @type {HTMLElement} */ (el).title = `${num(t.whole)} tokens, took ${secs(t.took)}, `
          + `reading it cold would have taken ${secs(t.coldCost)}`;
      };
      /** @param {import("./flow-model.js").Turn} t */
      const mkTurn = (t) => {
        const el = /** @type {HTMLElement} */ (tpl("tpl-turn").firstElementChild);
        fillTurn(el, t);
        return el;
      };
      /** @param {Element} el @param {import("./flow-model.js").Group} g */
      const fillGroup = (el, g) => {
        slot(el, {
          conv: g.conv.split("/")[0],
          saved: g.saved > 60 ? `saved ${secs(g.saved)}` : `${g.turns.length} turns`,
        });
        reconcileList(pick(el, "turns"), g.turns, (t) => t.key, mkTurn, fillTurn);
      };
      reconcileList(hosts.tape, groups, (g) => g.conv, (g) => {
        const el = /** @type {HTMLElement} */ (tpl("tpl-group").firstElementChild);
        fillGroup(el, g);
        return el;
      }, fillGroup);
      onTape = keys;
    }

    // -------------------------------------------------------------- stores
    /** The accessible twin of a store: every value the marks encode, as text.
     *  @param {HTMLElement} host @param {string[]} head @param {string[][]} rows */
    function twin(host, head, rows) {
      if (!host.firstElementChild) mount(host, tpl("tpl-twin"));
      const details = /** @type {HTMLElement} */ (host.firstElementChild);
      slot(details, { summary: "as a table" });
      /** Hide, never blank, a cell nobody filled: an empty `th scope="col"` is
       *  a column a screen reader announces on every row.
       *  @param {Element} el @param {string[]} cells */
      const cellsInto = (el, cells) => {
        ["c0", "c1", "c2", "c3"].forEach((name, i) => {
          const cell = /** @type {HTMLElement} */ (pick(el, name));
          cell.hidden = i >= cells.length;
          cell.textContent = cells[i] ?? "";
        });
      };
      /** @param {string} id @returns {(cells: string[]) => Element} */
      const rowOf = (id) => (cells) => {
        const el = /** @type {HTMLElement} */ (tpl(id).firstElementChild);
        cellsInto(el, cells);
        return el;
      };
      // The head row is `th scope="col"`, not a body row.
      reconcileList(pick(details, "head"), [head], () => "h", rowOf("tpl-headrow"), cellsInto);
      reconcileList(pick(details, "body"), rows, (r) => r[0], rowOf("tpl-cellrow"), cellsInto);
    }

    /** @param {Status} status */
    function stores(status) {
      const load = loadOf(status);
      const shelves = shelvesOf(status);
      slot(root, {
        // `openings` first: with none saved both counts are zero, and "none loaded" would be wrong.
        openingsCap: !load.openings ? ""
          : load.unloaded === load.openings ? "none loaded since restart"
          : load.unloaded ? `${load.unloaded} unloaded` : "",
      });

      /** @param {Element} el @param {import("./flow-model.js").Shelf} s */
      const fillShelf = (el, s) => {
          // A count, not "n of a cap": the shelves share one budget in bytes.
          slot(el, { title: s.title, of: `${s.files.length}` });
          /** @type {HTMLElement} */ (el).title = s.what;
          const biggest = Math.max(1, ...s.files.map((f) => f.bytes || 0));
          const cells = s.files.map((f) =>
            ({ id: `${s.kind}:${f.name}`, f, share: (f.bytes || 0) / biggest }));
          /** @param {Element} cell @param {{ id: string, f: typeof s.files[0] | null, share: number }} c */
          const fillCell = (cell, c) => {
              const node = /** @type {HTMLElement} */ (cell);
              node.toggleAttribute("data-free", !c.f);
              node.dataset.name = c.f ? c.f.name : "";
              node.style.setProperty("--w", String(0.55 + 0.45 * c.share));
              // `loads` lives in router memory and resets on a restart, so only a non-zero count is said.
              slot(node, {
                size: c.f ? fmtBytes(c.f.bytes || 0) : "",
                name: c.f ? c.f.name : "",
                loads: c.f && c.f.loads ? `loaded ${c.f.loads}x` : "",
              });
              node.title = c.f
                ? `${c.f.kind} ${c.f.name} - ${fmtBytes(c.f.bytes || 0)}, `
                  + (c.f.loads
                    ? `loaded ${c.f.loads} times since the router started`
                    : "not loaded since the router started; the count resets on a restart, the file does not")
                : "room for one more";
          };
          reconcileList(pick(el, "rack"), cells, (c) => c.id, (c) => {
            const cell = /** @type {HTMLElement} */ (tpl("tpl-cell").firstElementChild);
            fillCell(cell, c);
            return cell;
          }, fillCell);
      };
      reconcileList(hosts.shelves, shelves, (s) => s.kind, (s) => {
        const el = /** @type {HTMLElement} */ (tpl("tpl-shelf").firstElementChild);
        fillShelf(el, s);
        return el;
      }, fillShelf);

      const width = hosts.strip.getBoundingClientRect().width || 900;
      const { blocks, used, count, live } = blocksOf(status, width);
      slot(root, {
        copiesCap: `${fmtBytes(load.bytes)} of ${fmtBytes(load.budget)}, ${live} of ${count} in a slot`,
      });
      const strip = [
        ...blocks.map((b) => ({ id: b.conv, ...b })),
        { id: FREE, conv: "", bytes: 0, share: Math.max(0, 1 - used),
          state: /** @type {const} */ ("free"), where: "", label: false, doomed: false },
      ];
      /** @param {Element} el @param {typeof strip[0]} b */
      const fillBlock = (el, b) => {
          const node = /** @type {HTMLElement} */ (el);
          // Basis 0, never auto: with auto a labelled block is its label wide
          // before its share is added, so widths stop meaning bytes.
          node.style.flex = `${b.share * 100} 0 0%`;
          node.dataset.state = b.state;
          node.toggleAttribute("data-free", b.state === "free");
          node.toggleAttribute("data-doomed", Boolean(b.doomed));
          if (b.conv) node.dataset.conv = b.conv;
          // The block carries `data-pair` like the slot does in layout(), or the highlight runs one way only.
          node.dataset.pair = b.state === "live" ? b.conv : "";
          slot(node, { label: b.label ? b.conv.split("/")[0] : "" });
          node.title = b.conv ? `${b.conv} - ${fmtBytes(b.bytes)}, ${b.where}` : "";
      };
      reconcileList(hosts.strip, strip, (b) => b.id, (b) => {
        const el = /** @type {HTMLElement} */ (tpl("tpl-block").firstElementChild);
        fillBlock(el, b);
        return el;
      }, fillBlock);

      twin(hosts.copiesTwin, ["conversation", "where", "size"],
        blocks.map((b) => [b.conv, b.where, fmtBytes(b.bytes)]));
      twin(hosts.openingsTwin, ["opening", "kind", "size", "loads"],
        shelves.flatMap((s) => s.files.map(
          (f) => [f.name, f.kind, fmtBytes(f.bytes || 0), String(f.loads || 0)])));
    }

    // --------------------------------------------------------------- wires
    /** One infinite animation a wire, retimed rather than restarted. One period
     *  a second at playbackRate 1, so the rate is marks a second. Rate 0 freezes the marks.
     *  @param {Element} path @param {string} id @param {number} marks */
    function march(path, id, marks) {
      let anim = marching.get(id);
      if (!anim) {
        anim = path.animate([{ strokeDashoffset: 0 }, { strokeDashoffset: -MARK_PERIOD }],
          { duration: 1000, iterations: Infinity, easing: "linear" });
        marching.set(id, anim);
        anims.add(anim);
      }
      const rate = reducedMotion() ? 0 : marks;
      anim.updatePlaybackRate(rate);
      if (document.hidden) anim.pause();
    }

    /** @param {Status} status @param {SlotNode[]} nodes */
    function layout(status, nodes) {
      if (signal.aborted || document.hidden) return;
      laidOut = true;
      // Wires leave the gates, not the lists: an empty list has a zero-height box.
      const arrivals = box(arrivingGate), gate = box(parkedGate);
      const bankBox = box(heroEl);
      const holder = holderOf(
        residency(status.disk?.files || [], backendsOf(status), status.flow?.live || []));
      /** @type {{ id: string, d: string, tone: string, marks: number, capped: boolean, conv: string }[]} */
      const paths = [];
      /** @param {string} id @param {string} d @param {string} tone
       *  @param {number} marks @param {boolean} capped @param {string} [conv] */
      const wire = (id, d, tone, marks, capped, conv = "") => {
        paths.push({ id: `rail-${id}`, d, tone: tone === "hold" ? "hold" : "rail", marks: 0, capped: false, conv });
        if (marks > 0) paths.push({ id: `flow-${id}`, d, tone, marks, capped, conv: "" });
      };

      const readers = nodes.filter((n) => !n.generator);
      for (const n of nodes) {
        const lane = readers.indexOf(n);
        const host = hosts[n.generator ? "generators" : "readers"];
        const el = host.querySelector(`[data-key="${n.key}"]`);
        if (!el) continue;
        const nb = box(el);
        if (n.generator) {
          wire(`gq-${n.key}`, across(gate, nb), "none", 0, false);
          wire(`out-${n.key}`, across(nb, bankBox), n.tone, n.marks, n.capped);
        } else {
          wire(`in-${n.key}`, across(arrivals, nb, lane, readers.length), n.tone, n.marks, n.capped);
          wire(`out-${n.key}`, across(nb, gate), "none", 0, false);
        }
        const conv = holder.get(n.key) || "";
        const node = /** @type {HTMLElement} */ (el);
        node.toggleAttribute("data-holds", Boolean(conv));
        node.dataset.pair = conv;
        // Draw a hold line only where the pin's own slot number proves it.
        if (conv) {
          const blk = hosts.strip.querySelector(`[data-conv="${conv}"]`);
          if (blk) wire(`hold-${conv}`, down(nb, box(blk)), "hold", 0, false, conv);
        }
      }

      /** @param {Element} el @param {typeof paths[0]} p */
      const fillPath = (el, p) => {
          el.setAttribute("d", p.d);
          el.setAttribute("data-tone", p.tone);
          el.classList.toggle("march", p.marks > 0);
          el.classList.toggle("capped", p.capped);
          if (p.conv) el.setAttribute("data-pair", p.conv);
        if (p.marks > 0) march(el, p.id, p.marks);
      };
      reconcileList(wires, paths, (p) => p.id, (p) => {
        const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
        fillPath(el, p);
        return el;
      }, fillPath);
      for (const [id, anim] of [...marching]) {
        if (!paths.some((p) => p.id === id)) { anim.cancel(); anims.delete(anim); marching.delete(id); }
      }
      transfers(status);
    }

    /** A copy crossing between a slot and its store, in the time its bytes take.
     *  @param {Status} status */
    function transfers(status) {
      const { entries, lastAt } = since(status.recent_files, lastFile ?? 0);
      lastFile = lastAt;
      // First paint: the log is history. A flag, because `since` echoes the seed for an empty log.
      if (!seenFiles) { seenFiles = true; return; }
      if (document.hidden || reducedMotion()) return;
      for (const ev of entries) {
        const t = transferOf(ev);
        const from = stage.querySelector(`[data-key="${t.key}"]`);
        if (!from) continue;
        // `note_file` records name[:8]. An opening is its 8-character key. A
        // copy's name is the head of a conversation key, so match by prefix.
        const to = t.store === "copies"
          ? hosts.strip.querySelector(`[data-conv^="${t.name}"]`) || hosts.strip
          : hosts.shelves.querySelector(`[data-name="${t.name}"]`) || hosts.shelves;
        fly(box(from), box(to), t.down, t.store === "copies" ? "💾" : "📚",
            fmtBytes(t.bytes), t.seconds);
      }
    }

    /** @param {Box} a @param {Box} b @param {boolean} downward
     *  @param {string} icon @param {string} size @param {number} seconds */
    function fly(a, b, downward, icon, size, seconds) {
      const frag = tpl("tpl-packet");
      slot(frag, { icon, size });
      const tag = /** @type {HTMLElement} */ (frag.firstElementChild);
      board.append(tag);
      /** @param {number} x @param {number} y */
      const at = (x, y) => `translate(${x - 26}px, ${y - 27}px)`;
      const p0 = downward ? { x: a.x + a.w / 2, y: a.y + a.h } : { x: b.x + b.w / 2, y: b.y };
      const p1 = downward ? { x: b.x + b.w / 2, y: b.y } : { x: a.x + a.w / 2, y: a.y + a.h };
      const anim = tag.animate([
        { transform: `${at(p0.x, p0.y)} scale(.6)`, opacity: 0 },
        { transform: `${at(p0.x, p0.y)} scale(1)`, opacity: 1, offset: .14 },
        { transform: `${at(p1.x, p1.y)} scale(1)`, opacity: 1, offset: .86 },
        { transform: `${at(p1.x, p1.y)} scale(.6)`, opacity: 0 },
      ], { duration: seconds * 1000, easing: "cubic-bezier(.4,0,.3,1)" });
      anims.add(anim);
      const gone = () => { tag.remove(); anims.delete(anim); };
      anim.finished.then(gone, gone);
    }

    // --------------------------------------------------------------- paint
    /** @param {Status} status */
    function paint(status) {
      latest = status;
      latestAt = performance.now();
      /** @param {string} key */
      const sinceStep = (key) => {
        const seen = steps.get(key);
        return seen ? (Date.now() - seen.at) / 1000 : 0;
      };
      const nodes = nodesOf(status, sinceStep);
      trackSteps(nodes);

      const waiting = arrivalsOf(status);
      slot(root, { waiting: String(waiting.length) });
      reconcileList(hosts.arrivals, waiting, waiterKey, makeWaiter, fillWaiter);
      reconcileList(hosts.readers, nodes.filter((n) => !n.generator), (n) => n.key, makeNode, fillNode);
      reconcileList(hosts.generators, nodes.filter((n) => n.generator), (n) => n.key, makeNode, fillNode);

      const parked = parkedOf(status, Date.now() / 1000);
      slot(root, { queued: String(parked.length || status.waiting_to_generate || 0) });
      reconcileList(pick(root, "parked"), parked, (q) => q.conv, mkParked, fillParked);

      bank(status);
      stores(status);
      shown = nodes;
      requestAnimationFrame(() => layout(status, nodes));
    }

    // ------------------------------------------------------------ the page
    /** Hovering a copy or the slot holding it lights both and the line between them. */
    root.addEventListener("pointerover", (e) => {
      const target = e.target instanceof Element ? e.target.closest("[data-pair]") : null;
      const conv = target ? target.getAttribute("data-pair") || "" : "";
      if (conv === lit) return;
      for (const el of litEls) el.classList.remove("paired", "lit");
      litEls = [];
      lit = conv;
      if (conv) {
        litEls = [...root.querySelectorAll(`[data-pair="${conv}"]`)];
        for (const el of litEls) el.classList.add(el instanceof SVGElement ? "lit" : "paired");
      }
    }, { signal });

    const observer = new ResizeObserver(() => {
      if (latest && shown.length && !document.hidden) layout(latest, shown);
    });
    observer.observe(stage);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        for (const a of anims) if (a.playState === "running") { parked.add(a); a.pause(); }
        return;
      }
      for (const a of parked) a.play();
      parked.clear();
      // paint does not run while the tab is away, so restart the step clock
      // from now. Otherwise every reader comes back "not moving".
      const back = Date.now();
      for (const seen of steps.values()) seen.at = back;
      // No backlog stampede on return: skip to now rather than replaying.
      if (pending) {
        lastFile = pending.recent_files?.[0]?.at ?? lastFile;
        paint(pending);
        pending = null;
      }
    }, { signal });

    signal.addEventListener("abort", () => {
      observer.disconnect();
      for (const a of anims) a.cancel();
      anims.clear();
      marching.clear();
    }, { once: true });

    subscribe((status) => {
      // A hidden tab still gets every push. Skip the DOM work until it is visible.
      if (document.hidden) { pending = status; latest = status; latestAt = performance.now(); return; }
      paint(status);
    }, signal);

    // Age the waits between pushes by local elapsed time. Never compare this
    // clock to the router's.
    every(() => {
      if (document.hidden || !latest || !laidOut) return;
      const aged = (performance.now() - latestAt) / 1000;
      const waiting = arrivalsOf(latest, aged);
      slot(root, { waiting: String(waiting.length) });
      reconcileList(hosts.arrivals, waiting, waiterKey, makeWaiter, fillWaiter);
      const parked = parkedOf(latest, Date.now() / 1000);
      reconcileList(pick(root, "parked"), parked, (q) => q.conv, mkParked, fillParked);
    }, 1000, signal);
  },

  unmount() {},
};
