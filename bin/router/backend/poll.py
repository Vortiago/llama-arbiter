"""What one poll of a backend says about itself, as numbers.

Every function here is given what llama-server answered and gives back a
value. Nothing opens a socket, reads a clock or writes to a backend record,
so a case states a payload as a literal and reads the answer. Pool does the
reading and holds what carries between polls.
"""

RATE_FLOOR = 1.0     # seconds. Under this a count is not a rate.


def per_second(tokens, seconds):
    """A rate, or zero when there is not enough time to divide by."""
    return round(tokens / seconds, 1) if seconds and seconds >= RATE_FLOOR else 0


def counters(text):
    """The /metrics text as numbers. A line with labels is skipped: every
    counter this router reads is a bare one."""
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


def stats(value):
    """What the dashboard shows, from counters taken since the last reset."""
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
    return {
        "busy_per_decode": round(busy_per_decode, 2),
        "pp_rate": pp_rate,
        "tg_rate": tg_rate,
        "accept": round(100 * value.get("spec_decode_num_accepted_tokens_total", 0)
                        / drafted, 1) if drafted else 0,
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
            # Without the patch, the old arithmetic is the fallback.
            whole = max(0, slot.get("n_prompt_tokens", 0) - decoded)
            to_read = max(0, whole - cached)
        else:
            to_read = max(0, whole - cached - processed)
        detail.append({
            "id": sid,
            "busy": busy,
            "phase": "idle" if not busy else ("generating" if decoded else "reading"),
            "prompt": to_read,
            "done": processed,
            "cached": cached,
            "decoded": decoded,
            # null, not 0.0, until a window has resolved.
            "pp_rate": round(pp_rate, 1) if measured else None,
            "tg_rate": round(tg_rate, 1) if measured else None,
        })
    return current, detail
