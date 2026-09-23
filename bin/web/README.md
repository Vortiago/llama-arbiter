# bin/web — the router dashboard

The dashboard is a static site. `bin/router/web/handler.py` serves it under
`/router/`.

It starts in three steps:

1. `index.html` loads `shell.js`.
2. `shell.js` mounts one view. It takes that view from `views/registry.js`.
3. `views/feed.js` opens the `/router/events` SSE stream.

There is one stream. Every view reads from that stream.

## Checks

Run the gate before you commit:

    npm install               # once, for the typescript type gate
    node tools/check.mjs

## Vendored files

These files are unchanged copies from the vanilla-web toolkit:

- `lib/templates.js`
- `lib/render.js`
- `lib/format.js`
- `lib/live.js`
- every file in `tools/`

Extend these files. Do not fork them. Some of their exports are never called
here. Those exports are library surface. They are not dead code.

These files also name `SKILL.md`, `docs/adr/*`, and issue numbers such as
`(#42)`. Those names belong to the toolkit. This repository has no file and no
issue that matches them.

## Two changes to the copies

Two vendored files are changed on purpose. A later copy from the toolkit must
not undo either change.

- `lib/chrome.js`: `wireErrorBar` no longer sends errors to
  `/api/client-errors`. No route here serves that path. The router forwards an
  unrouted path to a backend, instead of answering 404.
- `shell.js`: the upstream issue references are removed. They point at the
  toolkit's issue tracker.
