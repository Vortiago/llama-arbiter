# bin/web — the router dashboard

Static, served by `router.py` under `/router/`. `index.html` boots `shell.js`,
which mounts one view from `views/registry.js`; `views/feed.js` opens the single
`/router/events` SSE stream every view reads.

    node tools/check.mjs      # the gate. Run before committing.

## Vendored

`lib/templates.js`, `lib/render.js`, `lib/format.js`, `lib/live.js` and `tools/`
are copied verbatim from the vanilla-web toolkit. Their unused exports are
library surface, not dead code — extend, don't fork.

`SKILL.md`, `docs/adr/*` and bare `(#nn)` in those files name the toolkit's own
documents and issues, not anything here.

Two deliberate forks a re-sync must not undo:

- `lib/chrome.js` — `wireErrorBar` no longer beacons to `/api/client-errors`.
  That path is unrouted here, and `router.py` forwards anything unrouted to a
  backend rather than 404ing it.
- `shell.js` — upstream issue references dropped; they point at the toolkit's
  tracker.
