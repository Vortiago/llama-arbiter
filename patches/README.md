# Patches to llama.cpp

Changes to llama.cpp, kept apart so each can be read and offered upstream on
its own. `tools/get-llama.sh` clones a checkout and applies all of them; to
apply one by hand, `git apply <name>.patch` from inside that checkout.

## slot-state-carries-checkpoints.patch

A saved slot state did not carry the slot's context checkpoints, so a restored
slot could not step back — and llama.cpp must process one token for logits, so
even an exact match had to. Without a checkpoint the whole prompt was read
again. The RAM prompt cache carries them; the file did not.

Writes them after the state behind a `QCKP` magic, read back at the offset the
state ended at. Measured on an isolated pair: a restored opening extended by 12
tokens read 12, an exact match read 4. Both read 609 before.

`n_bytes` reports the whole file, trailer included, because a caller budgets
disk against it. It was assigned before the trailer was written — a 107 MB file
reported 49 MB — and `res->n_bytes = nwrite + nckpt` is the line that fixes it.
`tests/live/` asserts the two agree, and they do: every opening this box has
built reports its size on disk exactly.

## anthropic-apply-template.patch

`/apply-template` reads openai-shaped messages only, so a `/v1/messages` client
writing `tool_use` and `tool_result` blocks got `unsupported content[].type` and
had no way to ask what its prompt renders to.

Adds `/v1/messages/apply-template`, built the way `/v1/messages/count_tokens`
already is: one handler taking the response type, converting the body and then
applying the template. `/apply-template` is untouched. Verified against a
running server: the rendered prefix tokenizes to the count the real call
reports, so a prompt prefilled from it is a true prefix.

## anthropic-pass-id-slot.patch

The anthropic endpoint converts a body through a whitelist and drops everything
not on it, `id_slot` included — so a router in front of the server cannot say
which slot a request should use.

## slots-report-the-prompt-size.patch

`/slots` reported `n_prompt_tokens` as what the slot holds *now*, which grows
while the prompt is read and again with every token generated. Nothing outside
the server could say how much was left to read; subtracting the cached and
processed counters measures what is done, not what remains. Watched on one slot
over 40 s, same task throughout:

    n_prompt=117317  cache=114301  processed=2560
    n_prompt=117753  cache=114301  processed=3016

Adds `n_prompt_tokens_total`, the task's own prompt size, which does not move.
`n_prompt_tokens` keeps its meaning.

## mtp-fit-ctx-other.patch

Local. Fits the context to free device memory with a draft model present.
