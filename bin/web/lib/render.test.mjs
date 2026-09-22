// @ts-check
// reconcileList keeps a host's children in step with a list of items, keyed.
// These guard what happens when two items carry the SAME key, which a caller
// can produce without meaning to: the flow strip keys copies by a short form
// of the conversation id, and two conversations whose ids agree in the first
// eight characters key the same.
import { test } from "node:test";
import assert from "node:assert/strict";
import { reconcileList } from "./render.js";

/** The smallest DOM reconcileList works against: ordered children, insertBefore
 * and remove. No moveBefore, so it takes the insertBefore path. */
function host() {
  /** @type {any} */
  const el = {
    kids: /** @type {any[]} */ ([]),
    get children() { return [...el.kids]; },
    get firstElementChild() { return el.kids[0] ?? null; },
    insertBefore(node, ref) {
      const from = el.kids.indexOf(node);
      if (from >= 0) el.kids.splice(from, 1);
      const at = ref ? el.kids.indexOf(ref) : el.kids.length;
      el.kids.splice(at < 0 ? el.kids.length : at, 0, node);
      node.parentNode = el;
      return node;
    },
  };
  return el;
}

/** @param {any} el @param {string} name */
const made = (el, name) => ({
  name, parentNode: null,
  get nextElementSibling() {
    const at = el.kids.indexOf(this);
    return at >= 0 ? el.kids[at + 1] ?? null : null;
  },
  remove() {
    const at = el.kids.indexOf(this);
    if (at >= 0) el.kids.splice(at, 1);
    this.parentNode = null;
  },
});

/** Run one pass of the same items over a host. @param {any} el @param {string[]} keys */
function pass(el, keys) {
  reconcileList(el, keys.map((k, i) => ({ k, i })), (it) => it.k,
    (it) => /** @type {any} */ (el.insertBefore(made(el, it.k), null)),
    () => {});
  return el.kids.length;
}

test("a host holds one node an item, and repeating the pass adds none", () => {
  const el = host();
  assert.equal(pass(el, ["a", "b", "c"]), 3);
  assert.equal(pass(el, ["a", "b", "c"]), 3, "a second pass created nodes again");
});

test("two items with the same key do not grow the host on every pass", () => {
  // The leak this guards: `prev` is a Map, so four children keyed "dup"
  // collapse to one entry. The item loop reuses that one and CREATES the
  // other three, and the final sweep cannot drop last pass's orphans because
  // they were never in `prev`. On an SSE feed that is a handful of nodes a
  // tick, for as long as the page is open - measured at 900+ children.
  const el = host();
  const keys = ["dup", "dup", "dup", "other"];
  assert.equal(pass(el, keys), 4, "first pass");
  assert.equal(pass(el, keys), 4, "second pass leaked");
  assert.equal(pass(el, keys), 4, "third pass leaked");
});

test("a key that goes away takes its node with it", () => {
  const el = host();
  pass(el, ["a", "b", "c"]);
  assert.equal(pass(el, ["a", "c"]), 2);
  assert.deepEqual(el.kids.map((/** @type {any} */ n) => n.name), ["a", "c"]);
});
