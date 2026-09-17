// @ts-check
/**
 * One live feed for the whole page: one EventSource serves every view.
 * The latest payload is replayed to a view that mounts between pushes.
 */
import { liveSSE } from "../lib/live.js";

/** @typedef {import("./status.js").Status} Status */
/** @typedef {(status: Status) => void} Subscriber */

/** @type {Set<Subscriber>} */
const subscribers = new Set();
// The parsed payload only. Keeping the raw json too holds a second ~110 KB copy per push.
/** @type {Status | null} */
let last = null;
let opened = false;
const life = new AbortController();     // liveSSE needs a signal; never aborted.

/** Brighten the header dot for one frame. The server pushes only on change. */
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

/** Hear every payload until `signal` aborts. The latest one is replayed at once.
 * @param {Subscriber} fn @param {AbortSignal} signal */
export function subscribe(fn, signal) {
  subscribers.add(fn);
  signal.addEventListener("abort", () => subscribers.delete(fn), { once: true });
  open();
  if (last) fn(last);
}
