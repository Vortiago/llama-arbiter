# Patches to llama.cpp

Each patch is a separate change, so that each one can be read on its own and
offered upstream on its own.

`tools/get-llama.sh` clones a checkout and applies all of them. To apply one by
hand, run `git apply <name>.patch` from inside that checkout.

Three of them are required. The router does not work without
`slot-state-carries-checkpoints`, `slots-report-the-prompt-size` or
`anthropic-pass-id-slot`.

## slot-state-carries-checkpoints.patch

**Required.**

A saved slot state did not carry the slot's context checkpoints. A restored
slot therefore could not step back. llama.cpp must process one token to get
logits, so even an exact match had to step back. Without a checkpoint it read
the whole prompt again. The RAM prompt cache carries the checkpoints. The file
did not.

The patch writes the checkpoints after the state, behind a `QCKP` magic
number. It reads them back at the offset where the state ended.

Measured on an isolated pair of servers:

| case | tokens read, patched | tokens read, before |
|---|---|---|
| restored opening, extended by 12 tokens | 12 | 609 |
| restored opening, exact match | 4 | 609 |

`n_bytes` now reports the whole file, including the trailer, because a caller
budgets disk space against that number. Before the patch it was assigned before
the trailer was written, so a 107 MB file reported 49 MB. The line that fixes
it is `res->n_bytes = nwrite + nckpt`. `tests/live/` asserts that the two
numbers agree.

## slots-report-the-prompt-size.patch

**Required.**

`/slots` reported `n_prompt_tokens` as what the slot holds *now*. That number
grows while the prompt is read, and it grows again with every token generated.
Nothing outside the server could say how much was left to read. Subtracting the
cached and processed counters measures what is done, not what remains.

Watched on one slot over 40 seconds, on the same task throughout:

    n_prompt=117317  cache=114301  processed=2560
    n_prompt=117753  cache=114301  processed=3016

The patch adds `n_prompt_tokens_total`. That is the task's own prompt size, and
it does not move. `n_prompt_tokens` keeps its existing meaning.

## anthropic-pass-id-slot.patch

**Required.**

The anthropic endpoint converts a request body through a whitelist. It drops
every field that is not on that list, including `id_slot`. A router in front of
the server therefore cannot say which slot a request must use.

## anthropic-apply-template.patch

`/apply-template` reads openai-shaped messages only. A `/v1/messages` client
that writes `tool_use` and `tool_result` blocks got `unsupported
content[].type`. It had no way to ask what its prompt renders to.

The patch adds `/v1/messages/apply-template`. It is built the way
`/v1/messages/count_tokens` is already built: one handler takes the response
type, converts the body, then applies the template. `/apply-template` is
unchanged.

Verified against a running server. The rendered prefix tokenizes to the same
count that the real call reports, so a prompt prefilled from it is a true
prefix.

## mtp-fit-ctx-other.patch

Local, and not required. It fits the context to the free device memory when a
draft model is present.

## grammar-probs.patch

Not required. `/v1/systemone` uses it, and answers without it less exactly.

A grammar decides what is written. It does not decide what is reported, and
neither existing readout gives the distribution over the answers a grammar
constrained the model to. Before the sampler the probabilities cover the whole
vocabulary and the grammar is not in them, so a caller picks an `n_probs` big
enough to catch its own answers and renormalises by hand - and when the model
wanted to write something else, the answers are not in the top n at all. After
the sampling chain the grammar and the sampler have already chosen: one token,
at 1.0, and on the OpenAI chat route nothing at all.

`grammar_probs` reports every token the grammar allows, scaled to sum to one.
A classification allows a handful, so they all fit and `n_probs` does not
enter into it.

`grammar_mass` goes with them: what those tokens held of the distribution
before the scaling. It tells an answer from a letter the grammar forced out of
a model that was going to write something else, which the scaled numbers alone
cannot. Measured on Qwen3-0.6B, one prompt, changing only the chat template's
thinking flag:

| | wrote | `grammar_mass` |
|---|---|---|
| `enable_thinking: true` | A | 0.000000 |
| `enable_thinking: false` | A | 0.999618 |

Both wrote the same letter, and both distributions looked like answers.

The readout also moves above `common_sampler_accept`. Accepting advances the
grammar past the token just chosen, so applying the grammar after it asks what
may follow the answer - for a one-token grammar, end of string. Only counters
sit between the two. Three gates on `n_probs` had to learn about a request that
asks for none: the call to `populate_token_probs`, and the partial and final
result builders.

`tests/live/` covers it. Without the patch the router falls back to the older
readout: llama.cpp ignores a field it does not know.
