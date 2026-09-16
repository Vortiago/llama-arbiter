// @ts-check
/**
 * Disk: what run/slots holds, and whether it is under control.
 *
 * Every number here is a count or size against a limit, because the limit is
 * the point. The whole view waits for a router that emits `disk`.
 */
import { loadTemplates, tpl, pick, mount } from "../../lib/templates.js";
import { renderRegion } from "../../lib/render.js";
import { num } from "../../lib/format.js";
import { subscribe } from "../feed.js";
import { backendsOf } from "../status.js";

/** @typedef {import("../status.js").Status} Status */
/** @typedef {import("../status.js").Disk} Disk */

const MIB = 1024 * 1024, GIB = 1024 ** 3;
/** @param {number} value */
const mib = (value) => `${num(Math.round(value / MIB))} MiB`;
/** @param {number} value */
const gib = (value) => `${num(value / GIB, { maximumFractionDigits: value < 10 * GIB ? 1 : 0 })} GiB`;

/** @param {Disk} disk @returns {DocumentFragment} */
function buildBudgets(disk) {
  const frag = new DocumentFragment();
  /** @param {string} label @param {number} share @param {string} text */
  const add = (label, share, text) => {
    const row = tpl("tpl-budget");
    pick(row, "label").textContent = label;
    pick(row, "fill").style.width = `${Math.min(100, Math.max(0.5, 100 * share)).toFixed(1)}%`;
    pick(row, "text").textContent = text;
    frag.appendChild(row);
  };
  const c = disk.copies;
  if (c.budget) add("conversation copies", (c.bytes || 0) / c.budget,
                    `${mib(c.bytes || 0)} of ${gib(c.budget)}, ${c.count} file${c.count === 1 ? "" : "s"}`);
  // One bar, because it is one budget on one disk. The two counts say what
  // each kind is using of it, which is what "24 of 24" used to say wrongly:
  // the shelves have no cap of their own to be full against.
  const o = disk.openings;
  if (o?.budget) add("saved openings", (o.bytes || 0) / o.budget,
                     `${mib(o.bytes || 0)} of ${gib(o.budget)}, `
                     + `${disk.bases.count} system prompt${disk.bases.count === 1 ? "" : "s"}`
                     + ` and ${disk.deeps.count} deeper`);
  return frag;
}

/** @param {Disk} disk @param {Status} status @returns {DocumentFragment} */
function buildFiles(disk, status) {
  const frag = new DocumentFragment();
  const files = disk.files || [];
  const live = new Set(backendsOf(status).map((b) => b.name));
  if (files.length) frag.appendChild(tpl("tpl-disk-head"));
  for (const f of files) {
    const row = tpl("tpl-disk-file");
    pick(row, "name").textContent = f.file || f.name;
    pick(row, "kind").textContent = f.kind;
    // A copy names the backend that held it. One kept from a previous run
    // names none that is live now, and that is a fact, not a gap.
    const where = f.backend && live.has(f.backend) ? `on ${f.backend}` : "from the last run";
    pick(row, "who").textContent = f.kind === "copy" ? `${f.conv || ""} · ${where}` : "";
    pick(row, "size").textContent = f.bytes ? mib(f.bytes) : "-";
    pick(row, "loads").textContent = f.kind === "copy" ? "" : `${f.loads || 0}×`;
    frag.appendChild(row);
  }
  if (!files.length) {
    const row = tpl("tpl-disk-file");
    pick(row, "name").textContent = "empty";
    frag.appendChild(row);
  }
  return frag;
}

/** @param {Disk} disk @returns {DocumentFragment} */
function buildMounts(disk) {
  const frag = new DocumentFragment();
  for (const m of disk.mounts || []) {
    const row = tpl("tpl-mount");
    pick(row, "path").textContent = m.path;
    const used = m.total ? (m.total - m.free) / m.total : 0;
    pick(row, "fill").style.width = `${(100 * used).toFixed(1)}%`;
    pick(row, "text").textContent = `${gib(m.free)} free of ${gib(m.total)}`;
    frag.appendChild(row);
  }
  return frag;
}

export default {
  id: "disk",

  /** @param {HTMLElement} container @param {unknown} _data
   *  @param {{ loadCSS: Function, every: Function, signal: AbortSignal }} helpers */
  async mount(container, _data, { loadCSS, signal }) {
    loadCSS(import.meta.url, "../shared.css", signal);
    loadCSS(import.meta.url, "./style.css", signal);
    await loadTemplates(new URL("./disk.html", import.meta.url).href,
                        { signal });
    // The shell aborts this controller when the reader clicks another
    // view. Without the check a mount cancelled mid-fetch carried on and
    // painted over whatever mounted after it - and worse, subscribe() and
    // every() register their teardown on `signal`, which never fires again
    // once it has aborted, so the dead view kept its SSE subscriber and its
    // interval for the life of the tab. flow/index.js has had this guard.
    if (signal.aborted) throw new DOMException("mount cancelled", "AbortError");

    mount(container, tpl("tpl-disk"));
    const root = /** @type {HTMLElement} */ (container.querySelector(".disk"));
    const budgets = pick(root, "budgets");
    const files = pick(root, "files");
    const mounts = pick(root, "mounts");

    subscribe(
      /** @param {Status} status */
      (status) => {
        const disk = status.disk;
        if (!disk) return;          // the payload always carries it; this narrows the type
        // The slice each region renders, not the whole payload: signing with
        // the payload meant signing the history buckets too, and one of those
        // moves every second, so nothing here could ever skip.
        const live = backendsOf(status).map((b) => b.name).join(",");
        renderRegion(budgets, () => buildBudgets(disk),
                     { sig: `b${JSON.stringify([disk.copies, disk.openings, disk.bases, disk.deeps])}` });
        // buildFiles asks which backends are live, to tell a copy held now
        // from one kept since the last run, so the names are part of the sig.
        renderRegion(files, () => buildFiles(disk, status),
                     { sig: `f${live}|${JSON.stringify(disk.files)}` });
        renderRegion(mounts, () => buildMounts(disk),
                     { sig: `m${JSON.stringify(disk.mounts)}` });
      },
      signal,
    );
  },

  unmount() {},
};
