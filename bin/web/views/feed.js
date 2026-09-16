// @ts-check
/**
 * One live feed for the whole page.
 *
 * The stream is opened by liveSSE like any view would, but the controller
 * that owns it lives for the page, not for a view: one EventSource serves
 * every view, and the latest payload is replayed to a view that mounts
 * between pushes. Views subscribe with their own signal and are dropped when
 * it aborts, so nothing outlives a view except this one shared subscription.
 */
import { liveSSE } from "../lib/live.js";

/** @typedef {import("./status.js").Status} Status */
/** @typedef {(status: Status) => void} Subscriber */

/** @type {Set<Subscriber>} */
const subscribers = new Set();
// The parsed payload only. `raw` was passed to every subscriber and read by
// none, and keeping it here held a second, ~110 KB copy of each push - the
// json text as well as the object - for the life of the tab.
/** @type {Status | null} */
let last = null;
let opened = false;
const life = new AbortController();     // liveSSE needs a signal; never aborted.

/** Brighten the header dot for one frame. It marks a payload that changed,
 * and settles: the server pushes only on change. */
function blip() {
  const dot = document.getElementById("live");
  if (!dot) return;
  dot.classList.add("on");
  setTimeout(() => dot.classList.remove("on"), 80);
}

function open() {
  if (opened) return;
  opened = true;
  liveSSE(
    "/router/events",
    /** @param {Status} status */
    (status) => {
      last = status;
      blip();
      for (const fn of subscribers) {
        try { fn(status); } catch (err) { window.reportError(err); }
      }
    },
    life.signal,
  );
}

/** Hear every payload until `signal` aborts. The latest one is replayed at
 * once, so a view that mounts between pushes is not blank until the next.
 * @param {Subscriber} fn @param {AbortSignal} signal */
export function subscribe(fn, signal) {
  subscribers.add(fn);
  signal.addEventListener("abort", () => subscribers.delete(fn), { once: true });
  open();
  if (last) fn(last);
}
