// @ts-check
// The shape guard: a 200 from a proxy in front of a backend carries a body
// this view must refuse to paint, and an error carries the router's own
// message. replyShape decides which of the two a reply is, so the view only
// ever mounts something that has an answer and a cost in it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { replyShape } from "./index.js";

const answer = { type: "noul", probabilities: { yes: 0.7, no: 0.3 },
                 confidence: 0.7, mass: 0.9 };
const router = { backend: "cpu", read: 12, reused: 4, took: 0.4 };

test("a whole reply is its answer and its cost", () => {
  const shaped = replyShape(true, 200, { answers: { it: answer }, router });
  assert.equal(shaped.note, undefined);
  assert.equal(shaped.answer, answer);
  assert.equal(shaped.cost, router);
});

test("a 200 with a foreign body is a note, not a mount", () => {
  // A proxy answers 200 with whatever it likes. Mounting said.answers.it out
  // of it threw before any note was shown.
  for (const foreign of [null, [], "ok", {}, { answers: {} },
                         { answers: { it: answer } }]) {
    assert.equal(replyShape(true, 200, foreign).note,
                 "the reply had no answer in it");
  }
});

test("an error carries the router's message", () => {
  assert.equal(replyShape(false, 502,
                          { error: { message: "no backend is up" } }).note,
               "no backend is up");
});

test("an error with a body this view cannot read names the status", () => {
  assert.equal(replyShape(false, 502, null).note, "the router said 502");
  assert.equal(replyShape(false, 401, "not json").note,
               "the router said 401");
});
