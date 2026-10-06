# Patches to llama.cpp

Two sets of patches change llama.cpp for this router. Both are made against
ggml-org master `8e1642198`, the commit that `llama-ref` names.

- The server patches in this directory change `llama-server`. Three of them
  are required.
- The CPU speed patches in `cpu/` make Qwen3.8-Flash-Next faster on the CPU
  backend. None of them is required.

`tools/get-llama.sh` checks out that commit and applies the server patches in
name order. It then applies `cpu/` in number order. To apply one server patch
by hand, run `git apply <name>.patch` from inside that checkout. A CPU patch
applies only on top of the one before it.

## The server patches

Each server patch is a separate change, so that each one can be read on its
own and offered upstream on its own. Each applies to `8e1642198` on its own.

Three of them are required. The router does not work without
`slot-state-carries-checkpoints`, `slots-report-the-prompt-size` or
`anthropic-pass-id-slot`.

### slot-state-carries-checkpoints.patch

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

### slots-report-the-prompt-size.patch

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

### anthropic-pass-id-slot.patch

**Required.**

The anthropic endpoint converts a request body through a whitelist. It drops
every field that is not on that list, including `id_slot`. A router in front of
the server therefore cannot say which slot a request must use.

### anthropic-apply-template.patch

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

### mtp-fit-ctx-other.patch, removed

This patch gave the memory-fit pass a `ctx_other`, so that the pass could
measure a draft model that reads `token_embd` from the target model. It does
not apply to master `8e1642198`, and this model no longer needs it. The
`qwen4exp` graph creates its own `token_embd`, and the ggml-org MTP draft
carries one. The patch is therefore gone from this directory. To read it, run
`git show 37dc46b:patches/mtp-fit-ctx-other.patch`.

### grammar-probs.patch

Not required. `/v1/systemone` uses it, and answers without it less exactly.

A grammar decides what is written. It does not decide what is reported, and
neither existing readout gives the distribution over the answers a grammar
constrained the model to. Before the sampler the probabilities cover the whole
vocabulary and the grammar is not in them, so a caller picks an `n_probs` big
enough to catch its own answers and renormalises by hand. When the model
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
may follow the answer: for a one-token grammar, end of string. Only counters
sit between the two. Three gates on `n_probs` had to learn about a request that
asks for none: the call to `populate_token_probs`, and the partial and final
result builders.

`tests/live/` covers it. Without the patch the router falls back to the older
readout: llama.cpp ignores a field it does not know.

## The CPU speed patches

The patches in `cpu/` are one series, exported from a llama.cpp branch. Each
one applies on top of the one before it. They change the CPU backend and its
tests. Q1 also changes `ggml_argsort_top_k` in `ggml.c`, which every backend
shares. The CUDA argsort reads only the sort order, so it ignores the new
top-k hint. A CUDA build compiles the CPU backend too, so it takes the series
as well. No CUDA build of the series has been made yet.

To write the series again, for example after a commit is added, run:

    LLAMA=<llama.cpp checkout> tools/export-cpu-patches.sh <branch> [<base>]

The base is the branch that holds `8e1642198` plus the server patches. The
default is `perf/base`. The script checks, in a scratch index, that the base
is exactly that, and that the series gives the tree of the branch.

### What each one does

The machine is one EPYC 7502P with 32 cores and no GPU, and the model is
Qwen3.8-Flash-Next at Q8_0. Each change is against master plus the server
patches, in one run. A change marked *paired* is against patches 0001 to
0009 together, in two alternating rounds. The reference drifted up to 6.2%
between rounds, so a smaller change from a single run is not a result.

- **pp512** and **tg128** are `llama-bench` at 32 threads.
- **8k prefill** is one backend with the cpu-prefill flags reading 8192
  tokens.
- **generate** is one backend with the generator flags and the MTP draft,
  writing 512 tokens for each of three prompts.
- **bit-exact** means `test-backend-ops` gives the same bits as the unpatched
  CPU backend for every op the patch touches.

| patch | id | what it does | measured | exactness |
|---|---|---|---|---|
| 0001, 0002 | X1 | A Q8_0 expert with 4 or more routed rows runs through the llamafile sgemm kernel that dense Q8_0 matmuls use, not one dot product per output. | pp512 +15.0%, 8k prefill +7.4% | bit-exact on x86 AVX, AVX2 and AVX-512, mean KLD 0 |
| 0003 | X3 | At generate an expert gets 1 to 4 rows. The small experts share one work list, so a thread reads longer runs, and a late thread hands its share on. | generate +4.2 to +8.2%, tg128 +1.0% | bit-exact |
| 0004 | I4A | `get_rows` and the strided copy split a wide row over all threads. One or two threads copied the 3 MiB recurrent state of each layer. | tg128 +19.9%, generate +3.5 to +7.0% | bit-exact |
| 0005 | I4B | `gated_delta_net` writes each state snapshot straight into its cache slot, so the copy after it goes. A port of the CUDA fusion. | tg128 +6.3%, generate +6.0 to +8.9% | bit-exact |
| 0006 | M3 | A run of tiny row-wise ops runs on thread 0, with one barrier after the run, not one per op. `GGML_CPU_SERIAL_BYTES` sets the limit: 32768 by default, 0 turns it off. | generate +4.6 to +7.7%, tg128 +0.9% | bit-exact |
| 0007 | W1 | `concat` splits over destination rows and copies a transposed source in cache-line tiles. One thread copied the conv state alone. | pp512 +19.0%, 8k prefill +10.8%, generate +1.0 to +4.7% | bit-exact |
| 0008, 0009 | W2 | The norms split rows over every dimension. `dsv4_hc_post` walks one stream row at a time, without three integer divides per element. | pp512 +8.5%, 8k prefill +6.8% | bit-exact |
| 0010, 0011 | I1 | A SIMD sigmoid, which the gated `dsv4_hc_pre` now uses row by row. | *paired*: pp512 +2.4%, 8k prefill +1.7% | not bit-exact: at most 1.2e-7 per op. The model KLD is not measured yet. |
| 0012, 0013 | Q1 | `ggml_argsort_top_k` passes k as a hint, and the CPU kernel sorts only the top k of a row. The MoE router sorted 512 ids to read 10. | *paired*: generate +0.9 to +3.2%, tg128 +1.3% | bit-exact |

The gains overlap, so they do not add. Patches 0001 to 0009 together read the
8k prompt 31.9% faster than master plus the server patches. They do pp512
47.9% faster, tg128 24.4% faster and generate 20.4 to 23.5% faster. Their mean
KLD is 0 and the top token is the same at contexts 512 and 8192. The whole
series is not measured yet.
