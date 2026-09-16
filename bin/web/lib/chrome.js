// @ts-check
// Canonical page-chrome wiring for the vanilla-web conventions (see SKILL.md).
// Copy into <app>/web/lib/chrome.js; extend, don't fork. Identity: the two
// pieces of chrome every page wires — the theme toggle and the error bar —
// shared by the app shell (shell.js) and the standalone component preview
// harness (preview.js), so the two pages can't drift. Both look up well-known
// ids in the shell markup (`#theme`, `#errbar`).
//
// This module imports nothing from templates.js or render.js, and nothing
// there imports this — components and defineComponent import ONLY
// templates.js, never this file.

/** Wire the `<button id="theme">` light/dark/auto cycle. light-dark() tokens
 * follow the root's color-scheme, so a manual override is one property; "auto"
 * clears it and defers to the OS. Choice persists per page under `storageKey`.
 * @param {string} [storageKey] */
export function wireTheme(storageKey = "theme") {
  const btn = document.getElementById("theme");
  const themes = ["auto", "light", "dark"];
  let current = localStorage.getItem(storageKey) || "auto";
  const apply = () => {
    document.documentElement.style.colorScheme = current === "auto" ? "" : current;
    if (btn) btn.textContent = current;
  };
  apply();
  btn?.addEventListener("click", () => {
    // ?? is for adopters compiling under noUncheckedIndexedAccess — the
    // modulo keeps the index in range, but their tsc can't see that.
    current = themes[(themes.indexOf(current) + 1) % themes.length] ?? "auto";
    localStorage.setItem(storageKey, current);
    apply();
  });
}

/** Surface listener exceptions and unhandled rejections (which vanish silently
 * by default): logs, and fills `<output id="errbar">` when present. `AbortError`
 * is filtered at both hooks: a cancelled fetch/mount from routine navigation is
 * a lifecycle event, not a failure, so it's `console.debug`'d instead of painted
 * red.
 *
 * Nothing is sent to the server. The upstream toolkit beaconed a copy to
 * `/api/client-errors`, on the assumption that an unrouted path 404s. The
 * router here has no such route, and its fallback forwards every unrouted
 * request — user agent and all — to a llama-server backend, so the beacon was
 * not free and not silent. */
export function wireErrorBar() {
  const errbar = document.getElementById("errbar");
  /** @param {unknown} reason */
  const isAbort = (reason) => reason instanceof DOMException && reason.name === "AbortError";
  /** @param {unknown} msg */
  const show = (msg) => {
    console.error(msg);
    if (errbar) {
      errbar.textContent = String(msg);
      errbar.hidden = false;
    }
  };
  window.addEventListener("error", (e) => {
    if (isAbort(e.error)) { console.debug(e.error); return; }
    show(e.message);
  });
  window.addEventListener("unhandledrejection", (e) => {
    if (isAbort(e.reason)) { console.debug(e.reason); return; }
    show(`unhandled: ${e.reason}`);
  });
}
