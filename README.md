# llama-arbiter

A router that keeps a conversation's prompt cache warm.

llama.cpp reads a long prompt in about an hour and continues it in seconds. That
saving lives in one slot on one server. So the router sends every turn back to
the slot holding its cache, copies caches to disk before anything can displace
them, and keeps the system-prompt openings every new session would otherwise
read for itself. Measured over one afternoon: the gpu backend read 80 prompt
tokens and reused 2,035,350.

    ./start-all.sh      start the backends and the router
    ./stop-all.sh       stop them

    http://<host>:8090/v1            openai
    http://<host>:8090/v1/messages   anthropic
    http://<host>:8090/router        dashboard

## What it needs

**A patched llama.cpp.** Three of the patches in `patches/` are load-bearing,
not optional: `slot-state-carries-checkpoints` is what lets a restored slot be
extended without re-reading, `slots-report-the-prompt-size` is what the
dashboard's "still to read" is computed from, and `anthropic-pass-id-slot` is
how the router names a slot on `/v1/messages`. Build a checkout beside this one
and point `SERVER_MTP` at it:

    git clone https://github.com/ggml-org/llama.cpp llama.cpp-mtp
    cd llama.cpp-mtp
    for p in ../patches/*.patch; do git apply "$p"; done
    cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server

The launch scripts also pass `--agent`, `--no-cache-idle-slots`,
`--ctx-checkpoints`, `--checkpoint-min-step`, `--n-cpu-moe` and `--spec-type
draft-mtp`, so a build old enough to lack those will not start.

**Weights**, wherever you keep them, named by `MODELS` and `MODEL_Q8` in
`config.local.sh`. This box runs Qwen3-Next at Q8 with an MTP draft model and an
F16 mmproj; nothing here is specific to it beyond those names.

## Running it elsewhere

Copy `config.local.example.sh` to `config.local.sh` and set what differs: where
llama-server is, where the weights are and what they are called, how much
context a slot gets and how the model is split across the card, threads per
backend, which disk holds the system blocks and where a run writes. It is not
tracked, and every launch script reads it first.

`CTX` is the one to check first: the default 150000 is sized to 16 GiB of VRAM
at f16 KV, and every backend has to use the same number, because a conversation
that outgrew one could never move back to it.

Out of the box it runs **one** `llama-server` on :8080 that both reads and
generates. That needs no configuring and is what a single instance wants.

For more than one, set two tables and keep them in step:

- `BACKENDS` in `config.local.sh` — what `start-all.sh` starts, `stop-all.sh`
  stops and `bin/restart-backend.sh` restarts. One `name port script [VAR=…]` a
  line.
- `ROUTER_BACKENDS` — a JSON file saying where the router sends a turn. Same
  set, plus what each instance may do: `prefill`, `generate`, and `pref` (where
  a turn would rather generate, lowest first).

Prefilling and generating are separate work — one is compute bound for tens of
minutes, the other memory bound for seconds — so they are separate settings,
and an instance can do either or both:

| `prefill` | `generate` | what it is |
|---|---|---|
| true | true | reads and answers in the same slot. One instance wants this. |
| false | true | a **generator**: turns migrate here once their prompt is read. |
| true | false | reads for the pool and hands every turn on, never spending a slot on a reply. |

None of that is tied to hardware. This box turns `prefill` off on the instance
holding the GPU, which measured no faster at prefilling than a cpu socket but
five times faster at generating; a machine whose card is also its fastest
prefiller would leave both on. A table with nothing that prefills, nothing that
generates, or an instance that does neither is refused at startup rather than
at the turn that first needs it. The router prints which table it loaded, and
from where.

The backends run with `--agent`, which is shell and file access with no key, so
they listen on localhost and the router serves only the two chat apis and a
short list of endpoints a client asks about itself. `PASS_THROUGH` widens that
list. Do not put the backends themselves on a public address.

## Tests

    python3 -m unittest discover -s tests -p "test_*.py"

    cd bin/web && npm install       # once: typescript, for the type gate
    node tools/check.mjs

`tests/live/` starts real `llama-server` processes instead of a double, and
asserts only on what a backend reports about itself. Its README says why, and
lists what it found.

## Worth knowing

- A conversation is one pin, one slot and one copy on disk. One turn of it runs
  at a time and the next waits, because two turns move the same three things.
- A read sends nothing for tens of minutes, so the router holds the stream open
  with whatever keep-alive that protocol has: a comment for OpenAI, a `ping`
  event for anthropic.
- Conversations are named from `x-claude-code-session-id` or
  `prompt_cache_key`, else by hashing the prompt's opening.
- `SIGTERM` parks every live cache first. `stop-all.sh` stops the router while
  the backends are still up to be read from.
- Cache decisions go to `run/cache-events-*.jsonl`. Read them with
  `tools/cache-report.py`; `CACHE_LOG=0` turns them off.
- Restart one backend with `bin/restart-backend.sh <name>`, or the router with
  `bin/restart-router.sh`. Both read `config.local.sh` on the way; a server
  started by hand does not, and falls back to the built-in defaults.
- Slow? Read `run/*.log`, or `tools/watch.sh`. A backend re-reading its weights
  from disk sits in D state at a tenth of its speed.

The dashboard serves Claude Code and OpenCode configs under `client setup`,
built with the address you reached it on.

`bin/router.py` is the whole router and its docstrings are the reference.
`docs/LAYOUT.md` says what the defaults above are based on, and which of them
are properties of llama.cpp rather than of the machine they were measured on.
