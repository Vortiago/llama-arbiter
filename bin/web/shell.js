// @ts-check
// The app shell. It owns three things:
//   1. routing     - location.hash ("#/<view-id>") is the source of truth;
//   2. lifecycle   - one AbortController per mount, aborted on switch;
//   3. transitions - swaps run inside document.startViewTransition when available.
//
// A mount() that throws aborts its controller, resets currentView to null so
// the nav link retries, and paints a textContent-only fallback. The error
// still reaches window.reportError. A throw because ITS OWN signal aborted is
// normal shutdown, not a failure. chrome.js wireErrorBar has the matching filter.

import { views } from "./views/registry.js";
import { loadCSS, every } from "./lib/templates.js";
import { withTransition } from "./lib/render.js";
import { wireTheme, wireErrorBar } from "./lib/chrome.js";

const stage = /** @type {HTMLElement} */ (document.getElementById("stage"));

/** @typedef {{ id: string, mount(container: HTMLElement, data: unknown, helpers: Helpers): void | Promise<void>, unmount(): void }} View */
/** @typedef {{ loadCSS: typeof loadCSS, every: typeof every, signal: AbortSignal }} Helpers */

/** @type {View | null} */ let currentView = null;
/** @type {AbortController | null} */ let currentController = null;
// Gates the document.title rewrite: the tab keeps index.html's <title> until the reader navigates.
let hasSwitchedOnce = false;
// Captured at boot, before a switch overwrites document.title.
const APP_NAME = document.title;

function viewIdFromHash() {
  const id = location.hash.replace(/^#\/?/, "");
  return views.some((v) => v.id === id) ? id : views[0].id;
}

/** Move focus to the stage after a swap, so screen readers announce from the top. */
function focusStage() {
  stage.tabIndex = -1;
  stage.focus({ preventScroll: false });
}

/** textContent-only fallback for a mount() that threw. The retry link is the
 * hash route for the same id: with currentView null, switchView(id) no longer
 * returns early. @param {string} id @param {unknown} err */
function renderFallback(id, err) {
  const wrap = document.createElement("div");
  wrap.dataset.slot = "viewError";
  const headline = document.createElement("p");
  headline.textContent = "This view failed to load.";
  const detail = document.createElement("p");
  detail.textContent = String(/** @type {{ message?: string }} */ (err)?.message ?? err);
  const retry = document.createElement("a");
  retry.href = `#/${id}`;
  retry.textContent = "Retry";
  retry.dataset.slot = "retryLink";
  wrap.append(headline, detail, retry);
  stage.replaceChildren(wrap);
}

/** @param {string} id */
async function switchView(id) {
  if (currentView?.id === id) return;
  const entry = views.find((v) => v.id === id);
  if (!entry) return;
  const view = /** @type {View} */ ((await entry.load()).default);

  const swap = async () => {
    currentController?.abort();
    currentView?.unmount();
    stage.replaceChildren();
    currentView = view;
    const controller = new AbortController();
    currentController = controller;
    try {
      await view.mount(stage, null, {
        loadCSS,
        every,
        signal: controller.signal,
      });
    } catch (err) {
      // Normal shutdown. Check THIS controller's signal, not only the error's
      // name: a view can throw an unrelated AbortError of its own.
      if (/** @type {{ name?: string }} */ (err)?.name === "AbortError" && controller.signal.aborted) return;
      window.reportError(err); // always surfaced, even if superseded below
      if (controller !== currentController) return; // a newer swap owns the stage
      controller.abort(); // release whatever the partial mount opened
      try { view.unmount(); } catch { /* half-mounted teardown is best-effort */ }
      currentView = null; // nav link becomes the retry
      hasSwitchedOnce = true; // a failed switch counts too
      renderFallback(id, err);
      focusStage();
      return;
    }
    syncNav(id);
    // Only on real switches: the boot view keeps index.html's own title.
    if (hasSwitchedOnce) document.title = entry.title ? `${entry.title} · ${APP_NAME}` : APP_NAME;
    hasSwitchedOnce = true;
    focusStage();
  };
  // startViewTransition awaits the async callback before it animates. Nothing
  // awaits switchView, so the plain-swap fallback is equivalent.
  withTransition(swap);
}

/** Mark the active nav link. Expects header links shaped href="#/<id>".
 * @param {string} id */
function syncNav(id) {
  for (const a of document.querySelectorAll('a[href^="#/"]')) {
    a.toggleAttribute("aria-current", a.getAttribute("href") === `#/${id}`);
  }
}

// Page chrome: theme toggle and error bar (lib/chrome.js).
wireTheme();
wireErrorBar();

window.addEventListener("hashchange", () => switchView(viewIdFromHash()));
switchView(viewIdFromHash());
