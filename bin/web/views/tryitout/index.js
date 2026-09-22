// @ts-check
/**
 * Try it out: ask the model one typed question and see the probabilities.
 *
 * The only view driven by a click rather than by the event stream. It posts to
 * /v1/systemone and paints what comes back. The state stays in its box between
 * questions, which is the point: the second question about one state does not
 * read it again.
 */
import { loadTemplates, tpl, pick, mount, withPending } from "../../lib/templates.js";

/** @typedef {{type: string, probabilities: Record<string, number>,
 *             confidence: number, mass: number, choice?: string,
 *             score?: number, noul?: number}} Answer */

/** What the criteria box holds, and what it is for, per question type. */
const CRITERIA = {
  noul: { label: "", placeholder: "" },
  choice: {
    label: "One answer a line, as name = what it means. Up to 26.",
    placeholder: "warm = it reused a cache\ncold = it read from the start\nlost = it never reached a backend",
  },
  score: {
    label: "One level a line, lowest first. The score is the expected level, so it lands between them.",
    placeholder: "routine\nwatch\nact now",
  },
};

/** The lines of a box, trimmed, with the blank ones dropped.
 *  @param {string} text @returns {string[]} */
const lines = (text) => text.split("\n").map((l) => l.trim()).filter(Boolean);

/** The question, in the shape /v1/systemone takes.
 *  @param {string} kind @param {string} instructions @param {string} criteria */
function asked(kind, instructions, criteria) {
  /** @type {{type: string, instructions: string, criteria?: unknown}} */
  const question = { type: kind, instructions };
  if (kind === "choice") {
    /** @type {Record<string, string>} */
    const named = {};
    for (const line of lines(criteria)) {
      const cut = line.indexOf("=");
      const name = (cut < 0 ? line : line.slice(0, cut)).trim();
      if (name) named[name] = (cut < 0 ? line : line.slice(cut + 1)).trim() || name;
    }
    question.criteria = named;
  } else if (kind === "score") {
    question.criteria = lines(criteria);
  }
  return question;
}

/** The line that says what the answer was.
 *  @param {Answer} answer @returns {[string, string]} */
function headline(answer) {
  const pct = (/** @type {number} */ p) => `${(100 * p).toFixed(1)}%`;
  if (answer.type === "score") {
    const levels = Object.keys(answer.probabilities);
    const at = Math.min(levels.length - 1, Math.round(answer.score ?? 0));
    return [`${(answer.score ?? 0).toFixed(2)} of ${levels.length - 1}`,
            `nearest level: ${levels[at]}`];
  }
  if (answer.type === "noul") {
    const yes = answer.noul ?? 0;
    return [yes >= 0.5 ? "yes" : "no", `${pct(yes)} yes`];
  }
  return [answer.choice ?? "", `${pct(answer.confidence)} of the mass`];
}

/** @param {Answer} answer @param {{backend: string, took: number,
 *  read: number, reused: number}} cost @returns {DocumentFragment} */
function buildAnswer(answer, cost) {
  const frag = tpl("tpl-tryitout-answer");
  const [first, second] = headline(answer);
  pick(frag, "headline").textContent = first;
  pick(frag, "second").textContent = ` · ${second}`;

  const probs = pick(frag, "probs");
  const rows = new DocumentFragment();
  const ranked = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
  for (const [name, p] of ranked) {
    const row = tpl("tpl-tryitout-prob");
    pick(row, "name").textContent = name;
    pick(row, "fill").style.width = `${Math.max(0.5, 100 * p).toFixed(1)}%`;
    pick(row, "p").textContent = p.toFixed(3);
    rows.appendChild(row);
  }
  mount(probs, rows);

  const whole = cost.read + cost.reused;
  const share = whole ? Math.round((100 * cost.reused) / whole) : 0;
  pick(frag, "cost").textContent =
    `${cost.backend} · ${cost.took.toFixed(1)} s · `
    + `${cost.reused} of ${whole} prompt tokens reused (${share}%)`;

  // How much of the model's own next token the answers held. Low means it
  // wanted to write something else, and the bars above are what was left.
  const held = Math.min(1, Math.max(0, answer.mass ?? 0));
  const warn = pick(frag, "mass");
  warn.textContent = held < 0.1
    ? `Careful: the answers held ${(100 * held).toFixed(1)}% of what the model `
      + `was going to write. It was not answering the question.`
    : `The answers held ${Math.round(100 * held)}% of what the model was `
      + `going to write.`;
  warn.classList.toggle("warn", held < 0.1);
  return frag;
}

/** @param {string} text @returns {DocumentFragment} */
function buildNote(text) {
  const frag = tpl("tpl-tryitout-empty");
  pick(frag, "text").textContent = text;
  return frag;
}

export default {
  id: "tryitout",

  /** @param {HTMLElement} container @param {unknown} _data
   *  @param {{ loadCSS: Function, every: Function, signal: AbortSignal }} helpers */
  async mount(container, _data, { loadCSS, signal }) {
    loadCSS(import.meta.url, "../shared.css", signal);
    loadCSS(import.meta.url, "./style.css", signal);
    await loadTemplates(new URL("./tryitout.html", import.meta.url).href,
                        { signal });
    if (signal.aborted) throw new DOMException("mount cancelled", "AbortError");

    mount(container, tpl("tpl-tryitout"));
    const root = /** @type {HTMLElement} */ (container.querySelector(".tryitout"));
    const state = /** @type {HTMLTextAreaElement} */ (pick(root, "state"));
    const question = /** @type {HTMLInputElement} */ (pick(root, "question"));
    const kind = /** @type {HTMLSelectElement} */ (pick(root, "type"));
    const criteria = /** @type {HTMLTextAreaElement} */ (pick(root, "criteria"));
    const criteriaLabel = pick(root, "criteriaLabel");
    const answersBox = pick(root, "answersBox");
    const button = /** @type {HTMLButtonElement} */ (pick(root, "ask"));
    const result = pick(root, "result");

    mount(result, buildNote("Nothing asked yet."));

    const showCriteria = () => {
      const how = CRITERIA[/** @type {keyof typeof CRITERIA} */ (kind.value)]
                  ?? CRITERIA.noul;
      answersBox.hidden = !how.label;
      criteriaLabel.textContent = how.label;
      criteria.placeholder = how.placeholder;
    };
    showCriteria();
    kind.addEventListener("change", showCriteria, { signal });

    const ask = async () => {
      const wanted = question.value.trim();
      if (!wanted) {
        mount(result, buildNote("Type a question first."));
        question.focus();
        return;
      }
      const body = {
        state: state.value,
        questions: { it: asked(kind.value, wanted, criteria.value) },
      };
      button.disabled = true;
      try {
        const reply = await withPending(result, fetch("/v1/systemone", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal,
        }));
        const said = await reply.json();
        if (!reply.ok) {
          mount(result, buildNote(said?.error?.message ?? `the router said ${reply.status}`));
          return;
        }
        mount(result, buildAnswer(said.answers.it, said.router));
      } catch (err) {
        if (signal.aborted) return;        // the view was left, not a failure
        mount(result, buildNote(String(err)));
      } finally {
        button.disabled = false;
      }
    };

    button.addEventListener("click", () => { void ask(); }, { signal });
    question.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); void ask(); }
    }, { signal });
  },

  unmount() {},
};
