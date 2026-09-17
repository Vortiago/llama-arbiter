// @ts-check
/**
 * Caching: is it working? The answer is a rate: the share of prompt tokens
 * each backend reused instead of reading again.
 */
import { loadTemplates, tpl, pick, mount } from "../../lib/templates.js";
import { renderRegion } from "../../lib/render.js";
import { num, time } from "../../lib/format.js";
import { subscribe } from "../feed.js";
import { backendsOf, cutDeeper, throughText, deeperAnswer } from "../status.js";

/** @typedef {import("../status.js").Status} Status */

const MIB = 1024 * 1024;
/** What the router calls a file event, said plainly. @type {Record<string, string>} */
const DID = { "loaded opening": "loaded saved prompt", "kept opening": "saved prompt" };
const CLOCK = /** @type {Intl.DateTimeFormatOptions} */ ({ hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** @param {Status} status @returns {DocumentFragment} */
function buildReuse(status) {
  const frag = new DocumentFragment();
  for (const be of backendsOf(status)) {
    const row = tpl("tpl-reuse");
    pick(row, "name").textContent = be.name;
    const st = be.stats || {};
    const known = Boolean(st.pp_rate || st.generated);
    const share = known ? st.cached || 0 : 0;
    pick(row, "reused").style.width = `${share}%`;
    pick(row, "read").style.width = known ? `${(100 - share).toFixed(1)}%` : "0";
    const text = pick(row, "text");
    text.textContent = known ? `${share}% reused` : "no request yet";
    text.classList.toggle("dim", !known);
    frag.appendChild(row);
  }
  return frag;
}

/** @param {Status} status @returns {DocumentFragment} */
function buildRam(status) {
  const frag = new DocumentFragment();
  for (const be of backendsOf(status)) {
    const c = be.cache;
    if (!c || !c.limit_mib) continue;
    const row = tpl("tpl-ram");
    pick(row, "name").textContent = be.name;
    pick(row, "fill").style.width = `${Math.min(100, (100 * (c.used_mib || 0)) / c.limit_mib).toFixed(1)}%`;
    const prompts = c.prompts || 0;
    pick(row, "held").textContent =
      `${num(Math.round(c.used_mib || 0))} of ${num(c.limit_mib)} MiB, ${prompts} conversation${prompts === 1 ? "" : "s"}`;
    const lost = [];
    if (c.evictions) lost.push(`${c.evictions} evicted, ${num(Math.round(c.evicted_mib || 0))} MiB`);
    if (c.skipped) lost.push(`${c.skipped} too big to cache`);
    if (c.rereads) lost.push(`${c.rereads} full re-read${c.rereads === 1 ? "" : "s"}`);
    pick(row, "lost").textContent = lost.join(", ");
    frag.appendChild(row);
  }
  return frag;
}

/** @param {Status} status */
function ramText(status) {
  const evicted = backendsOf(status).reduce((t, be) => t + (be.cache?.evictions || 0), 0);
  return evicted ? `${evicted} evicted` : "nothing evicted";
}

/** What each slot holds that a second conversation could start from.
 * @param {Status} status @returns {DocumentFragment} */
function buildHolds(status) {
  const frag = new DocumentFragment();
  const held = status.slots_hold;
  const say = (/** @type {string} */ where, /** @type {string} */ text) => {
    const row = tpl("tpl-hold");
    pick(row, "where").textContent = where;
    pick(row, "through").textContent = text;
    frag.appendChild(row);
  };
  if (!held) say("", "after the next router restart");
  else if (!held.length) say("", "no slot holds anything another conversation could start from");
  for (const h of held || []) {
    say(`${h.backend} slot ${h.slot}`, h.through === undefined ? `${h.cuts} starting point${h.cuts === 1 ? "" : "s"}` : throughText(h.through));
  }
  return frag;
}

/** @param {Status} status @returns {DocumentFragment} */
function buildChoices(status) {
  const frag = new DocumentFragment();
  const rows = status.cache_choices || [];
  if (!rows.length) return frag;
  frag.appendChild(tpl("tpl-choice-head"));
  for (const c of rows) {
    const row = tpl("tpl-choice");
    pick(row, "conv").textContent = c.conv;
    pick(row, "cuts").textContent = String(c.cuts);
    pick(row, "stored").textContent = throughText(c.stored);
    pick(row, "shared").textContent = throughText(c.shared);
    pick(row, "worth").hidden = !cutDeeper(c);
    frag.appendChild(row);
  }
  return frag;
}

/** @param {number} s */
const secs = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

/** @param {Status} status @returns {DocumentFragment} */
function buildRequests(status) {
  const frag = new DocumentFragment();
  const rows = status.recent_requests || [];
  if (!rows.length) {
    const row = tpl("tpl-request");
    pick(row, "conv").textContent = "none yet";
    frag.appendChild(row);
    return frag;
  }
  frag.appendChild(tpl("tpl-request-head"));
  for (const r of rows) {
    const row = tpl("tpl-request");
    pick(row, "conv").textContent = r.conv || "?";
    pick(row, "backend").textContent = r.backend;
    const started = pick(row, "started");
    started.textContent = r.started;
    started.classList.toggle("warn", r.started === "cold");
    pick(row, "tokens").textContent = num(r.tokens);
    pick(row, "waited").textContent = r.waited >= 1 ? secs(r.waited) : "";
    pick(row, "took").textContent = secs(r.took);
    frag.appendChild(row);
  }
  return frag;
}

/** @param {Status} status @returns {DocumentFragment} */
function buildFiles(status) {
  const frag = new DocumentFragment();
  const events = status.recent_files || [];
  for (const ev of events) {
    const row = tpl("tpl-file");
    pick(row, "did").textContent = DID[ev.did] || ev.did;
    pick(row, "where").textContent = `${ev.backend} ${ev.slot}`;
    pick(row, "name").textContent = ev.name;
    pick(row, "size").textContent = ev.bytes ? `${num(Math.round(ev.bytes / MIB))} MiB` : "";
    pick(row, "when").textContent = time(ev.at * 1000, CLOCK);
    frag.appendChild(row);
  }
  if (!events.length) {
    const row = tpl("tpl-file");
    pick(row, "did").textContent = "none yet";
    frag.appendChild(row);
  }
  return frag;
}

export default {
  id: "caching",

  /** @param {HTMLElement} container @param {unknown} _data
   *  @param {{ loadCSS: Function, every: Function, signal: AbortSignal }} helpers */
  async mount(container, _data, { loadCSS, signal }) {
    loadCSS(import.meta.url, "../shared.css", signal);
    loadCSS(import.meta.url, "./style.css", signal);
    await loadTemplates(new URL("./caching.html", import.meta.url).href,
                        { signal });
    // Stop a mount cancelled mid-fetch. subscribe() registers teardown on
    // `signal`, which never fires again once aborted, so a dead view would leak it.
    if (signal.aborted) throw new DOMException("mount cancelled", "AbortError");

    mount(container, tpl("tpl-caching"));
    const root = /** @type {HTMLElement} */ (container.querySelector(".caching"));
    const reuse = pick(root, "reuse");
    const ram = pick(root, "ram");
    const ramWhy = pick(root, "ramWhy");
    const requests = pick(root, "requests");
    const choices = pick(root, "choices");
    const holds = pick(root, "holds");
    const deeperWhy = pick(root, "deeperWhy");
    const files = pick(root, "files");

    subscribe(
      /** @param {Status} status */
      (status) => {
        // Sign each region with the slice it renders, never the whole payload:
        // the history buckets move every second.
        const bes = backendsOf(status);
        const reuseSig = bes.map((be) => {
          const st = be.stats || {};
          return `${be.name}:${Boolean(st.pp_rate || st.generated)}:${st.cached || 0}`;
        }).join("|");
        // A one-sentence region signs with the sentence.
        const ramWhyText = ramText(status);
        const deeperText = deeperAnswer(status);
        renderRegion(reuse, () => buildReuse(status), { sig: `u${reuseSig}` });
        renderRegion(ram, () => buildRam(status),
                     { sig: `c${JSON.stringify(bes.map((be) => [be.name, be.cache]))}` });
        renderRegion(ramWhy, () => document.createTextNode(ramWhyText), { sig: `e${ramWhyText}` });
        renderRegion(deeperWhy, () => document.createTextNode(deeperText), { sig: `d${deeperText}` });
        renderRegion(holds, () => buildHolds(status),
                     { sig: `h${JSON.stringify(status.slots_hold)}` });
        renderRegion(choices, () => buildChoices(status),
                     { sig: `k${JSON.stringify(status.cache_choices)}` });
        renderRegion(requests, () => buildRequests(status),
                     { sig: `r${JSON.stringify(status.recent_requests)}` });
        renderRegion(files, () => buildFiles(status),
                     { sig: `f${JSON.stringify(status.recent_files)}` });
      },
      signal,
    );
  },

  unmount() {},
};
