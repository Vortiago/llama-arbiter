# CPU speed patches for llama.cpp

This page reports the work to make Qwen3.8-Flash-Next (`qwen4exp`, Q8_0) faster
on the CPU backends. The machine is one EPYC 7502P with 32 cores and no GPU.
The numbers are from 5 and 6 October 2026. Rows marked **pending** wait for
bench queue 4.

## What was asked, and what we found

The ask had two parts. Port the speed ideas of Strata to llama.cpp as patches
and measure each one. Check upstream for work that overlaps.

| question | finding |
|---|---|
| Is Strata a llama.cpp fork we can merge from? | No. Strata is its own engine for one NVIDIA or AMD GPU. We ported ideas, not code: the adaptive draft depth of U5 follows its controller. |
| Does mainline run the model? | Yes. Mainline gained the `qwen4exp` graph (#27742) and official multi-token prediction (MTP) drafting (#29761). |
| What about our old fork? | danielhanchen's `qwen4exp/mtp` branch (`B-old`) is a dead end. Mainline has both of its features. |

We therefore based the work on ggml-org master `8e1642198` (5 October 2026)
plus our server patches, and built the CPU patches on top of that.

| branch, in `/home/atle/local_llm/llama.cpp` | holds |
|---|---|
| `perf/base` | master `8e1642198` and the arbiter's server patches (`B-main`) |
| `perf/stack` | `perf/base` + X1, X3, I4A, I4B, M3 (stack1) |
| `perf/stack2` | `perf/base` + X1, X3, I4A, I4B, M3, W1, W2 |
| `perf/stack3` | `perf/stack2` + I1, Q1 |
| `perf/s2-K3-cpu-moe-weighted-reduction` | `perf/stack2` + K3, for a paired test |
| `perf/K1-sparse-fa-tiled-prefill` | `perf/U4-batch1-decode-attn` + K1 |

Each candidate has its own `perf/<id>-*` branch and worktree
`/home/atle/local_llm/llama.cpp-perf-<id>-*`. The exported patches are in
`/home/atle/llama-arbiter-bench/patches/`.

## Method

`tools/perf-ab.py` measures one build at a time and appends every rep to
`/home/atle/llama-arbiter-bench/perf-results.tsv`. Each run starts its own
process, so two builds never share the CPU.

| test | what it runs | metric |
|---|---|---|
| kernel | `llama-bench`, 32 threads, flash attention on: prompts of 512 and 2048 tokens (pp512, pp2048), 128 generated tokens (tg128) | tok/s |
| prefill | one backend with the prefill backend's flags (32 threads, ubatch 512) reads an 8k prompt | prompt tok/s |
| generate | one backend with the generator's flags (16 threads) and the MTP draft, n-max 3, writes 512 tokens for 3 prompts | tok/s, acceptance, hash of a greedy reply |
| depth | a backend recalls a 32k copy into its slot, reads about 2k more tokens and writes 256 | prompt tok/s, tok/s |
| quality | `llama-perplexity` on wikitext against the logits of `B-main`, contexts 512 and 8192 | mean KL divergence (KLD), same top token % |

Four rules keep the numbers honest:

- **Drift.** `B-main` runs again every few hours. Over five rounds in 10 hours
  it moved 6.2% on pp512, 2.6% on 8k prefill, 2.8% on tg128 and 4.5% on the
  `explain` generate. A single-run change smaller than that is not a result.
- **Paired A/B.** A small candidate runs against its own base in two
  alternating rounds, so drift hits both sides alike.
- **Op exactness.** A dump harness (`perf/harness`) runs `test-backend-ops` once
  with the base libraries and once with the candidate's. It compares every
  output bit for the ops the patch touches.
- **Smoke.** Each candidate runs the real model greedy, with and without MTP,
  and must match the base build.

To print the summary, run this command:

```sh
python3 tools/perf-ab.py report --ref B-main
```

## Results

### The stacks against `B-main`

| build | pp512 | pp2048 | tg128 | 8k prefill | generate | 32k depth read | quality |
|---|---|---|---|---|---|---|---|
| `B-main` | 64.1 | 62.8 | 9.09 | 44.0 | 7.4 to 9.9 | 24.0 | reference |
| stack1 (X1, X3, I4A, I4B, M3) | +13.0% | +14.2% | +18.3% | +8.4% | +17.5 to +21.4% | +8.8% | KLD 0, top 100% |
| stack2 | **+47.9%** | **+45.9%** | **+24.4%** | **+31.9%** | **+20.4 to +23.5%** | pending | KLD 0, top 100% at 512 and 8192 |
| stack3 | pending | pending | pending | pending | pending | pending | pending |

The `B-main` row is in tok/s. Stack2 is bit-exact end to end: its greedy
replies hash the same as `B-main` on all three prompts.

### Each candidate

A change against `B-main` is a single run. A change marked *paired* is against
`perf/stack2` in two alternating rounds. QSA is the sparse attention of
`qwen4exp`: each token attends to a top-k pool of cells. MoE is mixture of
experts.

| id | what it does | phase | measured change | exactness | verdict |
|---|---|---|---|---|---|
| X1 | runs Q8_0 `MUL_MAT_ID` experts through llamafile sgemm | prefill | pp512 +15.0%, 8k prefill +7.4% | bit-exact (947 cases), KLD 0 | kept |
| X3 | one shared work list across small experts in `mul_mat_id` | generate | tg128 +1.0%, generate +4.2 to +8.2% | bit-exact (953 cases, 1 to 64 threads) | kept |
| I4A | splits wide `get_rows` and `dup` rows over all threads | generate | tg128 +19.9%, generate +3.5 to +7.0% | bit-exact (1388 cases) | kept |
| I4B | fuses `GATED_DELTA_NET` with its state snapshot copy (port of the CUDA fusion) | generate | tg128 +6.3%, generate +6.0 to +8.9% | bit-exact (52 cases) | kept |
| M3 | runs tiny serial ops on thread 0, with one barrier after a run of them | generate | tg128 +0.9%, generate +4.6 to +7.7% | bit-exact (5662 cases) | kept |
| W1 | splits `concat` over destination rows and tiles a transposed source | prefill | pp512 +19.0%, 8k prefill +10.8%, tg128 +3.4% | bit-exact (219 cases) | kept |
| W2 | walks `dsv4_hc_post` one stream row at a time, splits norm rows over every dimension | prefill | pp512 +8.5%, 8k prefill +6.8% | bit-exact (464 cases) | kept |
| I1 | SIMD sigmoid, and a vectorised gated `dsv4_hc_pre` | prefill | *paired*: pp512 +2.4%, 8k prefill +1.7% | numerically close: 10 of 162 cases differ, max error 1.2e-7 | kept in stack3, quality pending |
| Q1 | sorts only the top k of an `argsort_top_k` row (MoE router) | generate | *paired*: tg128 +1.3%, generate +0.9 to +3.2% | bit-exact (1047 cases) | kept in stack3 |
| K3 | fuses the MoE weighted reduction (port of the CUDA fusion) | prefill | pp512 +4.9%, 8k prefill +1.3% | bit-exact (306 cases) | **pending**: paired on stack2 |
| K1 | honours the `n_kv_max` sparse hint in tiled flash-attention prefill | prefill at depth | 32k depth read +43.6% (34.5 against 24.0 tok/s) | numerically close: 5 of 5317 cases differ from U4; KLD 0.0167, top 96.9% at 8192 | parked: a bit-exact rewrite is under test |
| U4 | batch-1 decode attention (port of upstream #27478) | generate | tg128 +1.6%, generate -15.5 to +4.7%, 32k depth +4.6% | numerically close: 4574 of 5317 cases differ; KLD 0, top 100% at 8192; greedy replies differ | parked: depth rerun pending |
| M5 | runs the QSA mask `repeat` and `sum_rows` on all threads | prefill at depth | *paired*: 8k prefill +0.4%, pp2048 +1.7% | bit-exact (30 cases) | parked: its cost grows with depth, and no depth run exists |
| X2 | four accumulators in the AVX2 `vec_dot_q8_0` | prefill | *paired*: 8k prefill +0.6%, pp512 -3.5% | numerically close: 93 of 191 cases differ | dropped: no gain past drift |
| Q2 | `nth_element` in `top_k` | prefill | *paired*: 8k prefill +0.2% | order differs in 114 of 627 cases | dropped: no gain past drift |
| U1 | each thread converts whole `src1` rows (upstream #29308) | generate | tg128 +0.7%, generate -0.5 to +3.4% | bit-exact (2862 cases) | dropped: no gain past drift |
| U5 | adaptive MTP draft depth (#27210 plumbing, Strata's controller) | generate | generate -0.2 to +2.3% | greedy replies identical | dropped: a fixed n-max 3 is as fast |
| S1 | a reduced draft vocabulary for the MTP head | generate | acceptance -1.9 to -18.1%, `list` -11.7% | target output unchanged | dropped: lost acceptance costs more than the smaller head saves |
| S3 | an attention window for the MTP layer | generate at depth | not measured | no ggml change | dropped: the ggml-org draft's MTP layer is QSA, so the window does not apply |
| I4C | row-blocked `GATED_DELTA_NET` for prefill | prefill | op microbenchmark: at most 1.33x at 32 threads, none at 16 | bit-exact | dropped: fails the 1.5x gate of its test plan |
| T1 | checks the `src1` layout before the tiled matmul | none | a correctness fix | bit-exact, with a regression test | outside the speed stack |

These designs were not built. U2, U3, U6 and G1 cannot gain on this box, by
the analysis in `designs.json`. I3 saves less than 1% of a token. S2 is already
in master: `--spec-type` takes a chain of drafters. I2, K4, K5, K6, M6 and X4
wait for a later round.

### Pending: bench queue 4

| cell | why |
|---|---|
| `B-main-256` generate | the reference for the 256-token generate runs below |
| stack3: kernel, 8k prefill, generate, quality at 512 and 8192 | stack3 is not measured yet |
| stack3 with no MTP (`--draft none`) | the stack2 run with `--spec-type none` kept speculation on: same acceptance and same greedy hash as with MTP |
| stack3 with `--spec-draft-p-min 0.5`, paired | on stack2 it gave -2.0 to +5.1% and changed the greedy replies |
| stack3 with `-ub 2048`, 8k prefill | on stack2 it gave +6.6% |
| K3 paired on stack2 | its gain alone is close to drift |
| ik_llama.cpp kernel, `-rtr` 0 and 1 | a ceiling for the CPU kernels; the queue 3 run failed, because ik's `llama-bench` has no `-o jsonl` |
| 32k depth: `B-main` again, stack3, U4, K1 with the fix | the first `B-main` depth run has 2 reps that spread 40% in tok/s |
| K1 with the fix, quality at 8192 | the fix must bring KLD to 0 |
| `B-main` kernel, prefill and generate at the end | the drift check for the queue |

## Findings worth keeping

**Where a token goes.** A profile of one generate step (119 ms) puts 16.4% in
the recurrent-state `get_rows` and `cpy` of `cache_s_l`. Its rows are few and
wide, so most threads waited. I4A and I4B target this. In a 512-token prefill
graph (11.35 s), `concat` of `conv_state_at` takes 7.8%, because one thread
copies it alone. W1 fixes that. The two `dsv4_hc_post` ops take 4.7%.

**Why master read prompts slower than the fork.** In the same round, master
read the 8k prompt 4.0% slower than the fork, and pp512 was 3.2% slower. Master
was 7.8% faster on tg128. Master fuses the hyper-connection maths into
`DSV4_HC_POST` (#28901). That removes about 770 graph nodes per token, which is
the tg gain. Its CPU kernel splits a flat index with three 64-bit integer
divides per element. The Zen 2 divider is not pipelined, so each call takes
5.5 ms against about 2.5 to 3 ms for the fork's separate ops. W2 rewrites the
kernel without the divides and recovers more than the loss.

**MTP greedy is not no-MTP greedy, on the base too.** With MTP, the target
verifies a batch of n-max + 1 tokens. A batched matmul sums in another order
than a one-token decode, so the logits differ in the last bits. In the base
smoke, one of two prompts gives a different reply after 264 characters. Compare
a candidate only with a base run in the same mode. The same holds for
`--spec-draft-p-min`, which makes the verify width vary.

**A small numeric change makes a discrete flip.** U4 and K1 change attention by
an NMSE (normalised mean squared error) of at most 1e-4. QSA picks its pool and
the router picks 10 experts by top-k, so a tiny change can flip one pick. After
that the hidden state takes another path. U4 has a mean KLD of 0 at 8192, but
all three greedy replies change and `explain` acceptance falls 13.7%. K1 shows
the same as a KLD of 0.0167 with 96.9% same top token. Treat a numerically
close attention patch as a quality change, and test it with KLD and greedy.

**Flags on stack2, against stack2's own generate and prefill:**

| flag | change | result |
|---|---|---|
| `--spec-draft-n-max 2` | -4.1 to -13.3% | keep n-max 3 |
| `--spec-draft-n-max 4` | -6.2 to -11.5% | keep n-max 3 |
| 24 generate threads | -2.3 to -3.1% | keep 16 |
| 32 generate threads | -11.0 to -24.3% | keep 16 |
| `--spec-type ngram-mod,draft-mtp` | -1.4 to -4.8% | no gain from the n-gram chain |
| `-ub 1024`, `2048`, `4096` on prefill | +5.2%, +6.6%, +7.0% | use 2048, because 4096 adds only 0.4%; the check on stack3 is pending |

## Follow-ups for the GPU box

- **Test every patch in a CUDA build there.** None of these branches was
  built with `-DGGML_CUDA=ON`. Stack3 also changes `ggml.h` and `ggml.c`, which
  every backend shares.
- **Try the VRAM expert cache.** Upstream #29887 (open) keeps hot MoE experts of
  a host-memory model in VRAM.
- **Balance `--n-cpu-moe` again.** The CPU patches change how fast the experts
  in RAM run, so the best split between card and RAM can move.
- **Measure ubatch 4096 to 8192 for prefill.** A larger ubatch reads each
  expert's weights once for more tokens.
