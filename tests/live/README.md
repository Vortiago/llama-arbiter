# Tests against a real llama-server

The offline suite passed on a day when four features were broken in production.
Every fault was a wrong belief about llama.cpp. `tests/fake_backend.py` was
written from those same beliefs, and a test double built from a belief cannot
disprove that belief.

These tests therefore start real `llama-server` processes with a small model.
They assert only on what a backend reports about itself: the token counts in
its own `prompt eval time = ... / N tokens` lines, `cache_n`, `prompt_n` and
`n_prompt_tokens`.

    cd tests/live
    LIVE_E2E=1 python3 -m unittest discover -s . -t . -v   # about 7 minutes

This directory has no `__init__.py`, so `discover -s tests` does not reach it.
Without `LIVE_E2E=1` every test skips. Four variables change the run:
`LIVE_MODEL_DIR`, `LIVE_MODEL_KIND`, `LIVE_FIRST_PORT` and `LIVE_KEEP`.

## The models

The three models total 900 MB. They live under
`~/.cache/llama-arbiter/live-models`, unless `LIVE_MODEL_DIR` says otherwise.

| kind | file | |
|---|---|---|
| hybrid | `Falcon-H1-0.5B-Instruct-Q4_K_M.gguf` | default |
| plain | `Qwen3-0.6B-Q4_K_M.gguf` | |
| swa | `gemma-3-270m-it-Q8_0.gguf` | |

The output quality of these models does not matter. The shape of their memory
does. The default is hybrid because production is hybrid: attention, plus a
recurrent state that cannot be rewound. Every belief below depends on that
shape.

A plain attention model answers beliefs 1, 2 and 5 the opposite way. Any prefix
of a full attention cache can be recovered by dropping cells. Run these tests
against a plain model and they pass while proving nothing about the system they
were written for.

To fetch the models:

    DIR=${LIVE_MODEL_DIR:-~/.cache/llama-arbiter/live-models}
    mkdir -p "$DIR" && cd "$DIR"
    curl -sSLO https://huggingface.co/unsloth/Falcon-H1-0.5B-Instruct-GGUF/resolve/main/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf
    curl -sSLO https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
    curl -sSLO https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf

## The eight beliefs

Measured with the hybrid model. Every figure is the backend's own.

| # | belief | verdict |
| --- | --- | --- |
| 1 | A saved slot restores elsewhere and serves without a re-read. | **Half true.** A strict extension costs nothing. The same prompt again costs everything, because llama.cpp steps back one token, and that step needs a checkpoint. 1891 of 1891 tokens were read again. |
| 2 | A restored slot carries no checkpoints, so any rewind reads everything again. | **Was true. Fixed by `patches/slot-state-carries-checkpoints.patch`.** Stock llama.cpp logs `forcing full prompt re-processing`. The same rewind on a live slot cost 4 tokens. These tests assume neither build; they read that log line. |
| 3 | `n_predict: 0` leaves the slot holding exactly the prompt. | **True where it matters.** The reply carries one sampled token. Nothing feeds that token back, so it never enters the slot. |
| 4 | `id_slot` is honoured on all three endpoints. | **True.** `/v1/messages` honours it only with `patches/anthropic-pass-id-slot.patch`. That reply still never names its slot, which is why the patch is needed. |
| 5 | A prompt that exactly extends a slot skips the shared part. | **True, with one off-by-one.** A repeat that adds nothing cannot skip everything: one token must be evaluated to get logits. The log says `n_past was set to N-1`. |
| 6 | A generated token lands in the slot. | **True for all but the last.** `n_predict: k` leaves the prompt plus k-1 tokens. The part that matters holds: a slot that has generated is no longer the bare prompt. |
| 7 | `?action=save` reports `n_written`, and defers while it works. | **True, once the patch is applied.** An empty slot saves under a kilobyte, which is what `PARK_FLOOR` exists to catch. Before the patch, `n_written` was taken before the checkpoint trailer: 49,214,256 for a file of 107,387,528 bytes. Stock llama.cpp writes no trailer, so this passes either way. It catches a half-patched build. |
| 8 | Two instances must agree on KV layout. | **True, and it is four things.** They must agree on `--kv-unified` (`n_stream mismatch`), on flash attention (`incompatible V transposition`), and on the model (`mismatched layer count`). Context size and slot count need not match, if the state fits: 1491 tokens moved onto 2048 and onto 4 slots, and were refused by 512. Every failure returns the same opaque 400, and the reason appears only in the target's log. |

## The belief that was not on the list

llama.cpp defaults to `--cache-idle-slots`. When any task starts, the server
copies every idle slot into its RAM cache. Under `--kv-unified` it then
**clears** that slot.

Production does not run `--kv-unified`, so the clearing does not apply here.
Measured both ways: a strictly extended opening read 609 tokens again on a
kv-unified backend, and 10 without it.

The other half applies whatever the layout. The server consults its RAM cache
only when it picks the slot itself. A request that names a slot gets that slot
as it stands, and the router must name a slot to know which one to save:

    named   id_slot=0   read 1491 tokens, reused    0
    unnamed             read    4 tokens, reused 1487

`--no-cache-idle-slots` stops this, and production passes that flag.
