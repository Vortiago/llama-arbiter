# Patches to llama.cpp

Two sets of patches change llama.cpp for this router. Both are made against
ggml-org master `8e1642198`, the commit that `llama-ref` names.

- The server patches in this directory change `llama-server`. Three of them
  are required.
- The CPU speed patches in `cpu/` make Qwen3.8-Flash-Next faster on the CPU
  backend. None of them is required. `cpu/core/` applies by default, and each
  set in `cpu/optional/` applies only when `CPU_OPTIONAL` names it.

`tools/get-llama.sh` checks out that commit and applies the server patches in
name order. It then applies `cpu/core/` in number order, and then the optional
sets that `CPU_OPTIONAL` names. To apply one server patch by hand, run
`git apply <name>.patch` from inside that checkout. A CPU patch applies only on
top of the one before it, and an optional set applies only on top of core.

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

### moe-sum-where-the-experts-run.patch

Not required. It speeds up a backend that keeps its experts in RAM and the
rest on a card.

The weighting of the expert outputs by the router and their sum have no
weight, so the scheduler places them by their neighbours, and its
expand-gpu-up pass put them on the card. Every layer then copied all the
used experts' rows over PCIe: at 512 tokens and 10 experts, 50 MB, where
the sum is 5 MB. The patch puts them on the backend that runs the experts'
matmul. A backend with its experts on the card, or op offload of a large
batch, keeps them there.

Measured on koishi (RTX A4000 on PCIe gen3 x8, experts in RAM, op offload
off), three alternating rounds: prompt 8.38 / 8.39 / 8.47 ms a token
against 7.48 / 7.55 / 7.51 (+10.7%); a 4-token verify step 134.5 to 131.6
ms, within noise. Over 20 chunks the outputs moved by mean KLD 0.045
against the unpatched GPU run, less than a CPU-only run of the same model
(0.047): the size of any change of summation order on this model. PPL
ratio 0.998 ± 0.005.

`weight_op_backend` repeats the rule of pass 1 of
`ggml_backend_sched_split_graph`. If upstream changes that rule, this
copy must follow it.

### sched-reserve-keeps-the-scheduler.patch

Not required. A port of upstream #28872.

With `--backend-sampling`, llama-server's slot reset detaches the sampler
after every request, so the next one sets `sched_need_reserve`, and
`sched_reserve()` destroyed and rebuilt the whole backend scheduler: every
compute buffer and the pinned host input buffer, freed and allocated again.
The patch re-reserves on the existing scheduler.

Measured on koishi's gpu backend layout, a cached 1000-token prefix plus 10
to 50 new tokens per request, three rounds: time to first token rose by 332
to 377 ms per plain request with backend sampling against without it, and
by 40 to 56 ms with the patch. A run of grammar requests never paid it,
because a grammar turns backend sampling off. Outputs, grammar
probabilities and perplexity identical.

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

The CPU patches are the final stack, branch `perf/stack4` in the llama.cpp
checkout, in two tiers:

- **Core**, in `cpu/core/`, is 17 patches for 10 changes. It applies by
  default. Every change in it gives the same bits as the base, except U4. U4
  changes the order in which batch-1 decode attention sums, so its output is
  numerically close, not the same.
- **Optional**, in `cpu/optional/<name>/`, is one directory for each change
  that moves the model's output a little: I1 and Q1. Each set is made on top
  of core alone. The sets apply in any combination, always in name order.

To build with both optional sets, run:

    CPU_OPTIONAL="I1 Q1" tools/get-llama.sh

Core plus I1 and Q1 gives the tree of `perf/stack4` exactly. `tools/README.md`
says how `get-llama.sh` finds a patch already applied, and how to write
`cpu/` again from a branch with `tools/export-cpu-patches.sh`.

The patches change the CPU backend and its tests, with two exceptions in
`ggml.c`, which every backend shares. U4's 0012 aligns a host buffer of 4 MiB
or more to 2 MiB and asks Linux for transparent huge pages. Q1 adds a top-k
hint to `ggml_argsort_top_k`. The CUDA argsort reads only the sort order, so it
ignores the hint. A CUDA build compiles the CPU backend too, so it takes the
patches as well. No CUDA build of them has been made yet.

### What each one does

The machine is one EPYC 7502P with 32 cores and no GPU, and the model is
Qwen3.8-Flash-Next at Q8_0. Every number is from
`/home/atle/llama-arbiter-bench/perf-results.tsv`. A change with no mark is a
single run of the patch alone against master plus the server patches
(`B-main`). A change marked *paired* is against the stack named, in
alternating rounds. Repeated runs of `B-main` alone spread 3.0% on 8k prefill
and 12.2% on tg128, so a smaller single-run change is not a result.
`docs/PERF.md` gives the method and every run.

- **pp512**, **pp2048** and **tg128** are `llama-bench` at 32 threads.
- **8k prefill** is one backend with the cpu-prefill flags reading 8192
  tokens.
- **generate** is one backend with the generator flags and the MTP draft,
  writing a reply to each of three prompts.
- **depth** restores a saved slot of 32k or 128k tokens, reads about 1800 more
  and writes 256.
- **bit-exact** means `test-backend-ops` gives the same bits as the base CPU
  backend for every op the patch touches. **KLD** is the mean KL divergence of
  `llama-perplexity` on wikitext against `B-main`.

| patches | id | what it does | measured | exactness |
|---|---|---|---|---|
| `core/0001, 0002` | X1 | A Q8_0 expert with 4 or more routed rows runs through the llamafile sgemm kernel that dense Q8_0 matmuls use, not one dot product per output. | pp512 +13.7%, 8k prefill +7.2% | bit-exact (947 cases), KLD 0 |
| `core/0003` | X3 | At generate an expert gets 1 to 4 rows. The small experts share one work list, so a thread reads longer runs, and a late thread hands its share on. | generate +4.2 to +8.2%. *Paired* on stack4, with and without X3: generate +6.1 to +7.3% | bit-exact (953 cases) |
| `core/0004` | I4A | `get_rows` and the strided copy split a wide row over all threads. One or two threads copied the 3 MiB recurrent state of each layer. | tg128 +20.1%, generate +3.5 to +7.0% | bit-exact (1388 cases) |
| `core/0005` | I4B | `gated_delta_net` writes each state snapshot straight into its cache slot, so the copy after it goes. A port of the CUDA fusion. | tg128 +6.4%, generate +6.0 to +8.9% | bit-exact (52 cases) |
| `core/0006` | M3 | A run of tiny row-wise ops runs on thread 0, with one barrier after the run, not one per op. `GGML_CPU_SERIAL_BYTES` sets the limit: 32768 by default, 0 turns it off. | generate +4.6 to +7.7%, tg128 +1.0% | bit-exact (5662 cases) |
| `core/0007` | W1 | `concat` splits over destination rows and copies a transposed source in cache-line tiles. One thread copied the conv state alone. | pp512 +17.7%, 8k prefill +10.6%, generate +1.0 to +4.7% | bit-exact (219 cases) |
| `core/0008, 0009` | W2 | The norms split rows over every dimension. `dsv4_hc_post` walks one stream row at a time, without three integer divides per element. | pp512 +7.3%, 8k prefill +6.6% | bit-exact (464 cases) |
| `core/0010` | K3 | The MoE weighted reduction runs as one kernel, not one multiply and one add per expert, each with a barrier. A port of the CUDA fusion. | *paired* on stack2: pp512 +2.6%, pp2048 +3.2%, 8k prefill +0.7% | bit-exact (306 cases) |
| `core/0011 to 0013` | U4 | Batch-1 flash attention decode scores KV cells in blocks of 32, with one vectorised softmax for each block. 0012 aligns large host buffers to 2 MiB for huge pages. 0013 adds test cases. A port of upstream #27478. | tg128 +1.7% | numerically close: 4574 of 5317 cases differ. KLD 0 at 8192, but the perplexity test reads in batches and does not run this path. Greedy replies differ. |
| `core/0014 to 0017` | K1 | A long prefill honours the `n_kv_max` sparse hint: each group of 8 query tokens reads only the 64-cell KV tiles in which one of its rows sees a cell. CUDA, Vulkan and Metal already do this. | on U4: 32k depth read 30.7 tok/s against 24.0 (+27.6%) | bit-exact against U4 (5338 cases, 32 and 16 threads). KLD 0, PPL ratio 1.0001 at 8192 |
| `core/0018` | N0 | `ggml_is_numa()` is true on any machine with two nodes, and matmul, `mul_mat_id`, repack, flash attention and `gated_delta_net` then split work in fixed slices per thread. A backend that numactl binds to one node lost the balance for nothing, and X3's flat work list never ran. The split now asks whether the process's own CPUs span nodes. | koishi, one node of a 2x Xeon Gold 6150: verify step of 4 tokens 322 and 317 ms against 304 and 305 (about +5% generate) idle, and about +9% on a busy box. Prompt unchanged when idle. | the same rows by other threads. PPL identical to the last printed digit (3 chunks of 512), old against new |
| `core/0019` | | `test-backend-ops` perf cases at Qwen3.8-Flash-Next's expert shapes (512 experts, 10 used, 2560 and 640 wide, 1, 4 and 512 tokens), and `GGML_TEST_THREADS` for the CPU backend's thread count. | | tests only |
| `core/0020` | N2 | A Q8_0 expert with 1 to 4 rows reads each weight row once for all its rows, with a software prefetch 4 KiB ahead, instead of once per row. | koishi, node 0: 4-token expert matmul +9 to 10%; verify step 310.9 to 305.5 ms mean (+1.7 to 2.8%, beyond the spread in two of three A/Bs) | bit-exact (27000 outputs; PPL identical) |
| `core/0021` | | upstream #29796 rebased: on one device, a stream synchronize returns at once when nothing was submitted since the last one. | no gain alone on koishi | exact |
| `core/0022` | N3 | The scheduler queues the copies between host memory and a backend's own buffer on that backend's stream: host-to-card copies run ahead of the graph launch, and the card-to-host copies of one split share one synchronize. An event lets the call return once the host memory it reads is free. | with 0021, a 4-token verify step about 2% faster on koishi; 404 syncs a step fell to 78 | exact. Tested on one card only |
| `optional/I1/0001, 0002` | I1 | A SIMD sigmoid, which the gated `dsv4_hc_pre` now uses row by row. | *paired* on stack2: pp512 +2.4%, 8k prefill +1.7% | at most 1.2e-7 per op. KLD 0.020, same top token 96.0%, PPL ratio 0.9972 ± 0.0037 at 512 |
| `optional/Q1/0001, 0002` | Q1 | `ggml_argsort_top_k` passes k as a hint, and the CPU kernel sorts only the top k of a row. The MoE router sorted 512 ids to read 10. | *paired* on stack2: generate +0.9 to +3.2%, tg128 +1.3% | bit-exact (1047 cases). KLD 0.002, same top token 99.5%, PPL ratio 0.9984 ± 0.0015 at 512 |

The PPL ratio is the perplexity of the patch over that of `B-main`. For I1 and
Q1 it is below 1, by about one uncertainty or less, so neither shows a
measurable loss. They are optional because they change the output, not because
the output got worse.

The gains overlap, so they do not add. All of them together, `perf/stack4`,
measure against `B-main`:

| measure | stack4 | `B-main` |
|---|---|---|
| 8k prefill, tok/s | 59.4 to 60.9 | 44.0 |
| read at 128k depth, 32 threads, tok/s | 38.4 and 38.9 | 19.2 and 19.8 |
| tg128, tok/s | 11.7 (+29.4%) | 9.07 |
| KLD at 512 and 8192 | 0.020 and 0.019 | 0 |
| PPL ratio at 512 and 8192 | 0.9992 ± 0.0037 and 1.0045 ± 0.0028 | 1 |

At 128k depth, stack3 (stack4 without K3, U4 and K1) read 22.4 and 22.7
tok/s. Stack2 has a KLD of 0, and stack4 has the KLD of stack3 at both
contexts. On this test, I1 and Q1 therefore account for all of it. Core alone has not
been built and measured yet.
