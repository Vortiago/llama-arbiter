"""A typed question: one generated token, and the probabilities behind it.

The path and the field names are TypeSafe's Jev, so a client written for that
API reaches this router by changing the base URL. Everything here is given
what the client sent and gives back a value. The one call to a backend is
`answers`, which takes the link.
"""

import json
import math

# The backend has never heard of the router's path. Every post for one
# goes here.
SYSTEMONE = "/v1/systemone"
SYSTEMONE_UP = "/v1/chat/completions"
# One token an option. Verified against the production model: every one of
# these is a single token, with and without a leading space.
SYSTEMONE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Refused(Exception):
    """A typed question this router will not guess at. The text reaches the
    client as a 400, so it says what to send instead."""


SYSTEMONE_RUBRIC = ("Answer the question about the text above with one letter.\n"
                    "The question gives a letter for every answer it takes.\n"
                    "Write that letter and nothing else.")


NOUL_YES = {"yes", "true", "1"}
NOUL_NO = {"no", "false", "0"}


def noul_criteria(criteria):
    """What yes and no mean for one noul question, or None.

    A noul answers yes or no whatever it is asked, so its criteria do not name
    the answers: they say what the two stand for. Jev writes them as
    `{"true": ..., "false": ...}`, which is what a client in the wild sends."""
    if not isinstance(criteria, dict) or len(criteria) != 2:
        return None
    said = {str(name).strip().lower(): means for name, means in criteria.items()}
    yes = next((said[name] for name in said if name in NOUL_YES), None)
    no = next((said[name] for name in said if name in NOUL_NO), None)
    if yes is None or no is None:
        yes, no = list(criteria.values())      # two of them, in the order given
    return {"yes": yes, "no": no}


def systemone_options(kind, criteria):
    """The answers one question takes, in the order they are lettered."""
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise Refused("a choice question needs criteria: an object of "
                          "answer name to what that answer means")
        return [str(key) for key in criteria]
    if kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise Refused("a score question needs criteria: a list of levels, "
                          "lowest first")
        return [str(level) for level in criteria]
    raise Refused(f"no question type called {kind!r}. The types are "
                  f"choice, score and noul")


def systemone_plan(raw):
    """Read a typed body into the questions to ask, one at a time."""
    try:
        fields = json.loads(raw)
    except Exception:
        raise Refused("the body is not json")
    if not isinstance(fields, dict):
        raise Refused("the body is not a json object")
    if fields.get("stream"):
        raise Refused(f"{SYSTEMONE} does not stream. One token has nothing to "
                      f"stream, so the answer arrives in one piece")
    state = fields.get("state")
    if state is None:
        state = ""
    elif not isinstance(state, str):
        # State is what the asking program holds, and a client in the wild
        # sends an object: an email, a request, a row. Render it once here, so
        # the prompt, the cuts and the conversation's name all see one text.
        state = json.dumps(state, indent=2, ensure_ascii=False)
    asked = fields.get("questions")
    if not isinstance(asked, dict) or not asked:
        raise Refused("questions is an object of one or more named questions")

    plan = []
    for name, question in asked.items():
        if not isinstance(question, dict):
            raise Refused(f"question {name!r} is not an object")
        kind = str(question.get("type") or "noul")
        options = systemone_options(kind, question.get("criteria"))
        if len(options) < 2:
            raise Refused(f"question {name!r} needs at least two answers to "
                          f"choose between, and has {len(options)}")
        if len(options) > len(SYSTEMONE_LETTERS):
            raise Refused(f"question {name!r} takes {len(options)} answers. "
                          f"One token carries {len(SYSTEMONE_LETTERS)} at most")
        said = question.get("criteria")
        plan.append({"name": name, "type": kind, "options": options,
                     "letters": SYSTEMONE_LETTERS[:len(options)],
                     "instructions": str(question.get("instructions") or ""),
                     "criteria": noul_criteria(said) if kind == "noul" else said})
    key = fields.get("prompt_cache_key")
    return {"model": fields.get("model") or "systemone",
            "key": key if isinstance(key, str) and key.strip() else None,
            "state": state, "questions": plan}


def systemone_says(question):
    """The message that asks one question and letters its answers."""
    criteria = question["criteria"]
    head = question["instructions"].strip()
    lines = [head, ""] if head else []
    for letter, option in zip(question["letters"], question["options"]):
        means = criteria.get(option) if isinstance(criteria, dict) else None
        lines.append(f"{letter} = {means or option}")
    lines += ["", "Answer with one letter.", "Answer:"]
    return "\n".join(lines)


def systemone_body(plan, question):
    """The chat body that asks one question about this plan's state."""
    messages = [{"role": "system", "content": SYSTEMONE_RUBRIC},
                {"role": "user", "content": plan["state"]},
                {"role": "user", "content": systemone_says(question)}]
    body = {"model": plan["model"], "messages": messages, "stream": False}
    if plan["key"]:
        body["prompt_cache_key"] = plan["key"]
    letters = question["letters"]
    body.update({
        # One token. tests/live belief 6: at one token the slot still holds
        # exactly the prompt, so a question leaves nothing behind it.
        "max_tokens": 1,
        "temperature": 0,        # -1 means greedy upstream, but field_num
                                 # clamps a soft limit, so -1 arrives as 0
        "logprobs": True,
        "top_logprobs": 2 * len(letters) + 8,
        # False, or the grammar and greedy sampling have already collapsed
        # the distribution and every answer comes back at 1.0.
        "post_sampling_probs": False,
        # patches/grammar-probs.patch: report every token the grammar allows,
        # and what they held of the distribution before it. Stock llama.cpp
        # ignores a field it does not know, and the two lines above still
        # answer - less exactly, because they see only the top of the list.
        "grammar_probs": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "grammar": "root ::= " + " | ".join(f'"{x}"' for x in letters)})
    return body


def systemone_read(reply, question):
    """One typed answer, from the one token the backend wrote.

    The probabilities are the raw softmax over the whole vocabulary: llama.cpp
    reports them before the grammar, so they are absolute and they cover words
    no answer letter stands for. Weight lands on " A" as well as on "A", and
    both mean the same answer. What is left after the letters are kept is
    scaled back up to one, and `mass` records how much was thrown away."""
    choices = (reply or {}).get("choices") or []
    first = choices[0] if choices else {}
    content = (first.get("logprobs") or {}).get("content") or []
    head = content[0] if content else {}
    letters = question["letters"]
    mass = {letter: 0.0 for letter in letters}
    for item in head.get("top_logprobs") or []:
        letter = (item.get("token") or "").strip()
        if letter in mass and isinstance(item.get("logprob"), (int, float)):
            mass[letter] += math.exp(item["logprob"])

    total = sum(mass.values())
    # A patched backend has already scaled those to sum to one over the
    # answers, and reports what they held before that scaling. It counts every
    # token the grammar allowed; the sum above counts only the ones that fit
    # in top_logprobs, and undercounts whenever an answer fell off the end.
    reported = head.get("grammar_mass")
    held = reported if isinstance(reported, (int, float)) and reported >= 0 else total
    if total <= 0:
        # The grammar let one letter through, but the model's own next token
        # was going to be something else entirely, so no letter was reported.
        # What the backend wrote is still the answer. `mass` stays at 0, which
        # is the reader's warning that the rest of this is one letter's word.
        wrote = ((first.get("message") or {}).get("content") or "").strip()
        if wrote not in mass:
            raise RuntimeError(f"no probabilities came back for question "
                               f"{question['name']!r}")
        mass[wrote] = 1.0

    probs = {option: weight / (total or 1.0) for option, weight
             in zip(question["options"], mass.values())}
    best = max(probs, key=lambda option: probs[option])
    answer = {"type": question["type"], "probabilities": probs,
              "confidence": probs[best], "mass": held}
    if question["type"] == "noul":
        answer["noul"] = probs[question["options"][0]]
    elif question["type"] == "score":
        # The expected level, not the likeliest one. A score of 1.6 says the
        # answer sits between the second and third level, which is what the
        # distribution says and what one letter cannot.
        answer["score"] = sum(rank * probs[option] for rank, option
                              in enumerate(question["options"]))
    else:
        answer["choice"] = best
    return answer




def answers(link, be, slot, plan, timeout, alive):
    """Ask every question against the slot that already holds the state.

    `slot` is None when the turn was carried to another instance: the slot it
    landed in is that instance's to choose, and llama.cpp finds the state by
    prefix, exactly as it does for every turn the router forwards.

    Through `read`, the one call that watches the client: a plan is a list,
    each question is a generation bounded only by read_timeout, and a client
    that leaves holds the slot and the claim for the whole of it otherwise.
    Raises Gone, which the turn's ending already knows how to finish.
    """
    said, wrote, steps = {}, 0, []
    for question in plan["questions"]:
        body = systemone_body(plan, question)
        if slot is not None:
            body["id_slot"] = slot
        reply = link.read(be, SYSTEMONE_UP, body, alive, timeout)
        said[question["name"]] = systemone_read(reply, question)
        usage = (reply or {}).get("usage") or {}
        wrote += int(usage.get("completion_tokens") or 0)
        # What this question cost on top of the state, by the backend's own
        # count. The first question extends what the read pass left; the ones
        # after it roll back to the end of the state and read their own words.
        timing = (reply or {}).get("timings") or {}
        steps.append({"question": question["name"],
                      "read": timing.get("prompt_n"),
                      "reused": timing.get("cache_n")})
    return said, wrote, steps
