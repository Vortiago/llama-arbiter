"""What one poll of a backend says about itself, as numbers.

Every function here is given what llama-server answered and gives back a
value. Nothing opens a socket, reads a clock or writes to a backend record,
so a case states a payload as a literal and reads the answer. Pool does the
reading and holds what carries between polls.
"""

import re

RATE_FLOOR = 1.0     # seconds. Under this a count is not a rate.
# A read rate needs this many tokens too: per-request overhead dominates a
# few short prompts, and a reader ranked by such a figure is never chosen to
# read enough to be measured again.
READ_TOKENS_FLOOR = 2048

# The one labelled counter this router reads. Every other metric it wants is
# bare: position i counts the verification steps that accepted at least i+1
# draft tokens, so the positions are a survival curve.
POS_RE = re.compile(r'accepted_tokens_per_pos_total\{position="(\d+)"\}\s+'
                    r'([\d.eE+-]+)')


def per_second(tokens, seconds):
    """A rate, or zero when there is not enough time to divide by."""
    return round(tokens / seconds, 1) if seconds and seconds >= RATE_FLOOR else 0


def counters(text):
    """The /metrics text as numbers. A line with labels is skipped: every
    counter this router reads is a bare one, and accepted_positions reads the
    one labelled series by name."""
    value = {}
    for line in text.splitlines():
        if line.startswith("#") or "{" in line:
            continue           # comment, or a metric with labels
        name, _, number = line.partition(" ")
        try:
            value[name.split(":", 1)[-1]] = float(number)
        except ValueError:
            pass
    return value


def accepted_positions(text):
    """Accepted draft tokens at each position, from the labelled series. A
    position present at i means the verification step accepted at least i+1
    draft tokens, so the counts fall off with position."""
    out = {}
    for line in text.splitlines():
        if "accepted_tokens_per_pos_total" not in line:
            continue
        hit = POS_RE.search(line)
        if hit:
            try:
                out[int(hit.group(1))] = float(hit.group(2))
            except ValueError:
                pass
    return out


def read_rate(value):
    """Tokens a second this backend has read prompts at over its life, from
    the raw counters, or None before it has read for RATE_FLOOR seconds and
    READ_TOKENS_FLOOR tokens."""
    seconds = value.get("prompt_seconds_total", 0)
    tokens = value.get("prompt_tokens_total", 0)
    if seconds < RATE_FLOOR or tokens < READ_TOKENS_FLOOR:
        return None
    return tokens / seconds


def stats(value, per_pos=None, lifetime=None):
    """What the dashboard shows, from counters taken since the last reset.
    `per_pos` is accepted_positions' labelled series, lifetime: the shares
    are a shape, not a rate, so they are not taken since a reset. `lifetime`
    is the raw lifetime counters; `value` is since-reset. `accept_mean`
    divides lifetime accepted tokens by LIFETIME drafts, so a reset (which
    zeroes the since-reset steps while the labelled series stays lifetime)
    cannot inflate it. Without `lifetime` the two are the same reading."""
    def rate(tokens, seconds):
        # Lifetime. The *_tokens_seconds gauges read zero when idle.
        return per_second(value.get(tokens, 0), value.get(seconds, 0))

    # prompt_tokens_total excludes cached tokens.
    processed = value.get("prompt_tokens_total", 0)
    cached = value.get("prompt_tokens_cached_total", 0)
    drafted = value.get("spec_decode_num_draft_tokens_total", 0)
    # tokens_predicted_seconds_total sums per-request time. Concurrent slots
    # overlap. These rates are per request.
    busy_per_decode = value.get("n_busy_slots_per_decode", 1) or 1
    pp_rate = rate("prompt_tokens_total", "prompt_seconds_total")
    tg_rate = rate("tokens_predicted_total", "tokens_predicted_seconds_total")
    positions = sorted(per_pos or {})
    accepted = sum(per_pos.values()) if per_pos else 0
    # Lifetime drafts, beside the lifetime accepted above. The since-reset
    # `value` would make this lifetime/since-reset after a reset.
    lifetime_drafts = (lifetime if lifetime is not None
                       else value).get("spec_decode_num_drafts_total", 0)
    return {
        "busy_per_decode": round(busy_per_decode, 2),
        "pp_rate": pp_rate,
        "tg_rate": tg_rate,
        "accept": round(100 * value.get("spec_decode_num_accepted_tokens_total", 0)
                        / drafted, 1) if drafted else 0,
        # Accepted draft tokens a verification step, and the share of them
        # that fell at each position. The first is the mean accepted length.
        "accept_mean": round(accepted / lifetime_drafts, 2) if lifetime_drafts else 0,
        "accept_shares": [round(100.0 * per_pos[p] / accepted, 1)
                          for p in positions] if accepted else [],
        "cached": round(100 * cached / (cached + processed), 1) if cached + processed else 0,
        "longest": int(value.get("n_tokens_max", 0)),
        "generated": int(value.get("tokens_predicted_total", 0)),
        "read_s": round(value.get("prompt_seconds_total", 0), 1),
        "gen_s": round(value.get("tokens_predicted_seconds_total", 0), 1),
        "prompt_tokens": int(processed),
        "cached_tokens": int(cached),
        "pp_total": round(pp_rate * busy_per_decode, 1),
        "tg_total": round(tg_rate * busy_per_decode, 1),
    }


def slot_state(raw, previous, rate_window, now):
    """Per-slot state, so a 3-slot backend is not a single average.

    `previous` is what this returned last poll, and `raw` is what /slots
    answered. Returns (sample, detail): the sample to hand back next poll,
    and what the dashboard reads.
    """
    # /slots reports counters, not rates.
    current, detail = {}, []
    for slot in raw if isinstance(raw, list) else []:
        # A one-element array. Older builds sent a bare object.
        token = slot.get("next_token") or {}
        if isinstance(token, list):
            token = token[0] if token else {}
        cached = slot.get("n_prompt_tokens_cache", 0)
        sid = slot.get("id")
        task = slot.get("id_task")
        decoded = token.get("n_decoded", 0)
        processed = slot.get("n_prompt_tokens_processed", 0)

        # Measured over rate_window, not between polls: a slot at 0.03
        # tokens/s does not move in two seconds.
        was = previous.get(sid) or {"task": None, "decoded": 0, "processed": 0,
                                    "done_d": 0.0, "done_p": 0.0, "since": now,
                                    "pp_rate": 0.0, "tg_rate": 0.0,
                                    # False until a window has resolved.
                                    "measured": False}
        # A new task restarts the counters at zero.
        if was["task"] is None:
            grew_d = grew_p = 0        # first sight: take a baseline
        elif was["task"] == task:
            grew_d = max(0, decoded - was["decoded"])
            grew_p = max(0, processed - was["processed"])
        else:
            grew_d, grew_p = decoded, processed
        done_d = was["done_d"] + grew_d
        done_p = was["done_p"] + grew_p

        gap = now - was["since"]
        measured = was["measured"]
        if gap >= rate_window:
            pp_rate, tg_rate = done_p / gap, done_d / gap
            done_d = done_p = 0.0
            since = now
            measured = True
        else:
            pp_rate, tg_rate = was["pp_rate"], was["tg_rate"]
            since = was["since"]

        current[sid] = {"task": task, "decoded": decoded, "processed": processed,
                        "done_d": done_d, "done_p": done_p, "since": since,
                        "pp_rate": pp_rate, "tg_rate": tg_rate,
                        "measured": measured}

        # n_prompt_tokens_total is the prompt the task arrived with, from
        # patches/slots-report-the-prompt-size.patch. n_prompt_tokens grows
        # while the prompt is read and with every token generated: a slot 98%
        # served from cache reported "512 / 89,848 read".
        busy = bool(slot.get("is_processing"))
        whole = slot.get("n_prompt_tokens_total")
        if whole is None:
            # Without the patch, the old arithmetic is the fallback. It is not
            # the prompt's size: it is only what the slot holds, and it grows
            # while the prompt is read. Left unnamed so the bar does not pin
            # its track to a number that moves.
            whole = max(0, slot.get("n_prompt_tokens", 0) - decoded)
            to_read = max(0, whole - cached)
            known = None
        else:
            to_read = max(0, whole - cached - processed)
            known = whole
        detail.append({
            "id": sid,
            "busy": busy,
            "phase": "idle" if not busy else ("generating" if decoded else "reading"),
            "prompt": to_read,
            "done": processed,
            "cached": cached,
            "decoded": decoded,
            # The prompt the task arrived with, when the backend reports it
            # (patches/slots-report-the-prompt-size.patch). The bar pins its
            # track to this; absent, it falls back to cached + done + left.
            "whole": known,
            # null, not 0.0, until a window has resolved.
            "pp_rate": round(pp_rate, 1) if measured else None,
            "tg_rate": round(tg_rate, 1) if measured else None,
        })
    return current, detail
