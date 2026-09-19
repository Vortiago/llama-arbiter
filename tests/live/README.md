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

## The ninth belief: a grammar does not shape what is reported

`/v1/systemone` writes one token under a grammar of single letters, and reads
the answer out of the probabilities beside it. That rests on a belief worth
stating on its own, because it is the opposite of what the feature looks like:

**A grammar decides what is written. It does not decide what is reported.**

Before sampling, llama.cpp reports a plain softmax over the whole vocabulary.
The grammar is not in it. So the same request that is forced to write `A` also
reports the words it would rather have written, and the answer letters may not
appear at all. Measured on the hybrid model, one token, `top_logprobs: 14`:

| prompt | what the letters carried |
|---|---|
| a rubric, a state, and a lettered question | 0.46 |
| prose, with no hint that an answer is wanted | 0.0007 |

Two things follow, and both are in the router:

- A confidence taken from these numbers means nothing unless the prompt itself
  asked for a letter. `/v1/systemone` returns `mass`, the share those letters
  held, so a caller can tell the two cases apart.
- Room in the report does not help. Raising `top_logprobs` from 14 to 120
  moved the share by 0.0002. The letters are near the top or nowhere.

`post_sampling_probs: true` does not fix it and makes it worse: after the
sampling chain there is nothing left to read. On `/v1/chat/completions` no
probabilities come back at all, and on `/completion` it is the written token
at 1.0, every time.

## An open question: a message added after a long one, on production

`/v1/systemone` reads a state once and asks each question against it. The
obvious split is to read the state alone and let each question extend it. On
production that cost the first question a full re-read of the state, every
time. The backend's own prompt evals, one call, three questions, a state of
348 tokens:

| read pass carries | tokens read, in order | total |
|---|---|---|
| the state alone | 348, **351**, 42, 37 | 778 |
| the state and the first question | 351, **4**, 42, 37 | 434 |
| the same call again, second way | 46, 4, 42, 37 | 129 |

Adding one message after the state made the shorter prompt worthless, while
questions two and three rolled back to the end of the state for 42 and 37
tokens. The router now sends the same prompt in both phases, and
`tests/test_systemone.py` holds it there.

**Why is not established.** Two explanations were tested here and both are
wrong:

- *The chat template moves the assistant header, so the shorter prompt is not
  a prefix.* Falcon-H1 reused 380 of 384, and Qwen3-0.6B reused 380 of 382.
  Both models, both templates, no re-read.
- *A rollback that short has no checkpoint to reach.* The same run with
  production's `--ctx-checkpoints 64 --checkpoint-min-step 2048` reused 380 of
  384, unchanged.

What production has that these do not is `--spec-type draft-mtp` and a draft
model. That is the remaining suspect and it is untested: the models here have
no MTP draft. Until someone reproduces it, there is no test for it, because a
test that passes on a model which cannot show the fault proves nothing.

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
