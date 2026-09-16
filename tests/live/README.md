# Tests against a real llama-server

The offline suite was green on a day four features were dead in production.
Every fault was a wrong belief about llama.cpp, and `tests/fake_backend.py` was
written from the same beliefs — a double built from a belief cannot disprove it.

So these start real `llama-server` processes with a small model, and assert only
on what a backend reports about itself: the tokens in its own `prompt eval time
= ... / N tokens` lines, `cache_n`, `prompt_n`, `n_prompt_tokens`.

    cd tests/live
    LIVE_E2E=1 python3 -m unittest discover -s . -t . -v   # ~7 minutes

No `__init__.py`, so `discover -s tests` walks past; without `LIVE_E2E=1` every
test skips. Knobs: `LIVE_MODEL_DIR`, `LIVE_MODEL_KIND`, `LIVE_FIRST_PORT`,
`LIVE_KEEP`.

## The models

900 MB, under `~/.cache/llama-arbiter/live-models` unless `LIVE_MODEL_DIR` says
otherwise.

    hybrid  Falcon-H1-0.5B-Instruct-Q4_K_M.gguf   default
    plain   Qwen3-0.6B-Q4_K_M.gguf
    swa     gemma-3-270m-it-Q8_0.gguf

Quality is irrelevant; **the shape of the memory is not.** The default is hybrid
because production is: attention plus a recurrent state that cannot be rewound,
which is what every belief below turns on. A plain attention model answers
beliefs 1, 2 and 5 the opposite way, because any prefix of a full attention
cache can be recovered by dropping cells — swap it in and these go green while
saying nothing about the machine they were written for.

    DIR=${LIVE_MODEL_DIR:-~/.cache/llama-arbiter/live-models}
    mkdir -p "$DIR" && cd "$DIR"
    curl -sSLO https://huggingface.co/unsloth/Falcon-H1-0.5B-Instruct-GGUF/resolve/main/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf
    curl -sSLO https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
    curl -sSLO https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf

## The eight beliefs

Hybrid model, this machine. Figures are the backend's own.

| # | belief | verdict |
| --- | --- | --- |
| 1 | a saved slot restores elsewhere and serves without re-reading | **half true.** A strict extension costs nothing; the same prompt again does not, because llama.cpp steps back one token and that needs a checkpoint. 1891 of 1891 re-read. |
| 2 | a restored slot carries no checkpoints, so any rewind re-reads everything | **was true, fixed by `patches/slot-state-carries-checkpoints.patch`.** Stock says `forcing full prompt re-processing`; the same rewind on a live slot cost 4. The tests assume neither build — they read that log line. |
| 3 | `n_predict: 0` leaves the slot holding exactly the prompt | **true where it matters.** The reply carries one sampled token, never fed back, so it never enters the slot. |
| 4 | `id_slot` is honoured on all three endpoints | **true**, `/v1/messages` only via `patches/anthropic-pass-id-slot.patch`. That reply still never names its slot, which is why. |
| 5 | a prompt that exactly extends a slot skips the shared part | **true, with the off-by-one that is the whole story.** A repeat with nothing new cannot: one token must be evaluated for logits (`n_past was set to N-1`). |
| 6 | a generated token lands in the slot | **true for all but the last.** `n_predict: k` leaves prompt + k - 1. What matters holds: a slot that generated is no longer the bare prompt. |
| 7 | `?action=save` reports `n_written` and defers while working | **true, once the patch is applied.** An empty slot saves under a kilobyte, which is what `PARK_FLOOR` is for. `n_written` was taken before the checkpoint trailer — 49,214,256 for a 107,387,528 byte file — until `slot-state-carries-checkpoints.patch` added `nckpt` to it. Stock llama.cpp has no trailer to miss, so this passes either way; a half-patched build is what it catches. |
| 8 | two instances must agree on KV layout | **true, and it is four things.** `--kv-unified` (`n_stream mismatch`), flash attention (`incompatible V transposition`), the model (`mismatched layer count`). Context and slot count need not match if the state fits: 1491 tokens moved onto 2048 and onto 4 slots, refused by 512. Every failure is the same opaque 400, and the reason is only in the target's log. |

## The one that was not on the list

llama.cpp defaults to `--cache-idle-slots`: when any task starts, every idle
slot is copied into its RAM cache and, under `--kv-unified`, **cleared**.
Production does not run `--kv-unified`, so that half does not bite. Measured
both ways: a strictly extended opening re-read 609 tokens on a kv-unified
backend, 10 without.

The other half bites whatever the layout. The RAM cache is consulted only when
the server picks the slot itself, and a request that names a slot — which the
router must, to know which to save — gets it as it stands:

    named   id_slot=0   read 1491 tokens, reused    0
    unnamed             read    4 tokens, reused 1487

`--no-cache-idle-slots` stops that one, and production passes it.
