# CPU speed patches for llama.cpp

This page reports the work to make Qwen3.8-Flash-Next (`qwen4exp`, Q8_0) faster
on the CPU backends. The machine is one EPYC 7502P with 32 cores and no GPU.
The numbers are from bench queues 1 to 9, on 5 and 6 October 2026. Rows marked
**pending** wait for a later run.

On this box the core set reads an 8k prompt 37% faster than `B-main`, in one
backend with the server's flags, at KLD 0. Stack4, which is core with I1 and
Q1, reads a restored slot about twice as fast at 128k and 240k. Stack2, the
bit-exact part of core, generates 20 to 24% faster with MTP.

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

The final stack is `perf/stack4`. `patches/cpu/` holds it in two tiers. Core
applies by default: X1, X3, I4A, I4B, M3, W1, W2, K3, U4 and K1. Optional sets
apply only when `CPU_OPTIONAL` names them: I1 and Q1. `patches/README.md`
gives each patch with its tier.

| branch, in `/home/atle/local_llm/llama.cpp` | holds |
|---|---|
| `perf/base` | master `8e1642198` and the arbiter's server patches (`B-main`) |
| `perf/stack` | `perf/base` + X1, X3, I4A, I4B, M3 (stack1) |
| `perf/stack2` | `perf/base` + X1, X3, I4A, I4B, M3, W1, W2 |
| `perf/stack3` | `perf/stack2` + I1, Q1 |
| `perf/stack4` | `perf/stack3` + K3, U4, K1: the final stack |
| `perf/stack4-noX3` | `perf/stack4` without X3, for the paired X3 test |
| `perf/stack4-noU4dec` | `perf/stack4` without U4's batch-1 decode attention, for the U4 test of queue 9 |
| `perf/s2-<id>-*` | `perf/stack2` + one candidate, for a paired test or its quality alone |
| `perf/K1-sparse-fa-tiled-prefill` | `perf/U4-batch1-decode-attn` + K1, with the bit-exact fix |

Each candidate has its own `perf/<id>-*` branch and worktree
`/home/atle/local_llm/llama.cpp-perf-<id>-*`.

## Method

`tools/perf-ab.py` measures one build at a time and appends every rep to
`/home/atle/llama-arbiter-bench/perf-results.tsv`. Each run starts its own
process, so two builds never share the CPU.

| test | what it runs | metric |
|---|---|---|
| kernel | `llama-bench`, 32 threads, flash attention on: prompts of 512 and 2048 tokens (pp512, pp2048), 128 generated tokens (tg128) | tok/s |
| prefill | one backend with the prefill backend's flags (32 threads, ubatch 512) reads an 8k prompt | prompt tok/s |
| pair | two prefill backends on the same cores, as pre0 and pre1 run, each read their own 8k prompt at the same moment | prompt tok/s of each, and of both over the slower one's time |
| generate | one backend with the generator's flags (16 threads) and the MTP draft, n-max 3, writes 512 or 256 tokens for 3 prompts, or for the 16 prompts of `prompts-16.json` | tok/s, acceptance, hash of a greedy reply |
| depth | a backend restores a saved slot of 32k, 128k or 240k tokens, reads about 1800 more and writes 256. At 128k and 240k it reads on 32 threads (`--threads-batch 32`), as the prefill backends do. Since the end of queue 7 the reply ignores the end-of-generation token, so each rep writes all 256. | prompt tok/s, tok/s |
| quality | `llama-perplexity` on wikitext against the logits of `B-main`, contexts 512 and 8192 | mean KL divergence (KLD), same top token %, perplexity (PPL) ratio |

Five rules keep the numbers honest:

- **Drift.** `B-main` runs again every few hours. Its reps spread 8.6% on
  pp512, 3.0% on 8k prefill and 12.2% on tg128. A single-run change smaller
  than that is not a result.
- **Paired A/B.** A small candidate runs against its own base in two or three
  alternating rounds, so drift hits both sides alike.
- **Load.** Other containers share this machine. Since queue 4 each rep records
  the load average and the CPU time of other processes. A slow rep can then be
  traced to its neighbours.
- **Op exactness.** A dump harness (`perf/harness`) runs `test-backend-ops` once
  with the base libraries and once with the candidate's. It compares every
  output bit for the ops the patch touches.
- **Smoke.** Each candidate runs the real model greedy, with and without MTP,
  and must match the base build.

KLD measures how far a build moves from the base, not whether it got worse. A
rounding change on this model flips hard top-k choices and shows up as KLD.
Since queue 4b the quality test therefore also records PPL, the base PPL and
their ratio, with its uncertainty.

To print the summary, run this command:

```sh
python3 tools/perf-ab.py report --ref B-main
```

## Results

### The stacks against `B-main`

| build | pp512 | pp2048 | tg128 | 8k prefill | 32k depth read | 128k depth read | 240k depth read | quality |
|---|---|---|---|---|---|---|---|---|
| `B-main` | 64.8 | 63.3 | 9.07 | 44.1 | 24.0 | 18.3 to 19.8 | 18.2, 18.1 | reference |
| stack1 (X1, X3, I4A, I4B, M3) | +11.8% | +13.2% | +18.5% | +8.2% | +8.8% | not run | not run | KLD 0, top 100% |
| stack2 | +46.3% | +44.7% | +24.6% | +31.7% | not run | not run | not run | KLD 0, top 100% at 512 and 8192 |
| stack3 | +47.0% | +40.0% | +27.1% | +31.8% | 30.2 (+25.6%) | 22.4, 22.7 | not run | KLD 0.020 and 0.019, top 95.8% and 96.6%, PPL ratio 0.9992 ± 0.0037 at 512 |
| stack4 | *paired* against stack3: +1.9% | *paired* against stack3: +3.0% | +29.4% | **59.4 to 60.9 (+38.2%)** | not run | **38.4, 38.9** | **36.6, 38.2** | KLD 0.020 and 0.019, top 95.8% and 96.6%, PPL ratio 0.9992 ± 0.0037 at 512 and 1.0045 ± 0.0028 at 8192 |
| core | +44.2% | +45.3% | +25.9% | **60.1 to 60.4 (+36.9%)** | not run | not run | not run | KLD 0, top 100% at 512 and 8192, PPL ratio 1.0004 ± 0.0004 at 512 and 1.0001 ± 0.0001 at 8192 |
| core with I1 and Q1 | *paired* against core: +3.4% | *paired* against core: +1.5% | *paired* against core: +2.7% | *paired* against core: +1.5% | not run | not run | not run | KLD 0.019, top 96.6%, PPL ratio 1.0045 ± 0.0028 at 8192 |

The `B-main` row is in tok/s, and so are the depth cells. The 128k read of
`B-main` comes from two runs of 2 reps: 19.1 and 18.3, then 19.8 and 19.2.
Stack4 reads a restored 128k slot about twice as fast as `B-main`, and 71%
faster than stack3. Of the three changes from stack3 to stack4, K1 is the one
that targets the read at depth.

At 240k the fill read 226,745 tokens at 22.4 tok/s on average, in about 2
hours 50 minutes. The `B-main` cell is the rerun, which restores the saved slot
in a new process. The first run read in the process that did the fill, at
16.4 tok/s, so it is not comparable. Stack4 reads the restored 240k slot 2.0 to
2.1 times as fast as `B-main`.

At 240k, rep 0 of `B-main` wrote 4.73 tok/s at acceptance 0.45, and rep 0 of
stack4 wrote 5.13 tok/s at 0.38. Rep 1 of `B-main` wrote 3.53 tok/s at 0.31.
Rep 1 of stack4 wrote one token, because its first sampled token ended the
reply, so it has no generate rate. The depth test has ignored the
end-of-generation token since then.

In a single run in queue 5, stack4 did pp512 at 89.5 against 95.3 for stack3
in queue 4. The paired test in queue 7 settles it: stack4 is 1.9% faster on
pp512 and 3.0% faster on pp2048. Its two rounds gave +6.4% and +0.7% on
pp512, and +5.9% and -0.4% on pp2048.

At 512 tokens stack1 generates 17.5 to 21.4% faster than `B-main`, and stack2
20.4 to 23.5% faster. Stack2 is bit-exact end to end: its greedy replies hash
the same as `B-main` on all three prompts. Stack3 is not, because of I1 and
Q1, and stack4 is not, because of I1, Q1 and U4. Their generate rate is
therefore not a like-for-like comparison. Against `B-main` at 256
tokens (6.7 to 9.8 tok/s), stack3 generates 7.6 to 48.0% faster and stack4
30.3 to 42.1% faster. The acceptance moves with the reply: on `explain`
stack3 accepts 0.448 against 0.559 for `B-main`.

At 128k depth stack4 writes at 5.2 and 5.4 tok/s, and stack3 at 4.5 and 4.8.
`B-main` wrote 3.7 and 3.9 tok/s on the two reps that wrote 196 tokens. Its
other two replies stopped after 2 tokens, so they give no rate.

### Each candidate

A change against `B-main` is a single run. A change marked *paired* is against
the branch named, in alternating rounds. QSA is the sparse attention of
`qwen4exp`: each token attends to a top-k pool of cells. MoE is mixture of
experts.

| id | what it does | phase | measured change | exactness | verdict |
|---|---|---|---|---|---|
| X1 | runs Q8_0 `MUL_MAT_ID` experts through llamafile sgemm | prefill | pp512 +13.7%, 8k prefill +7.2% | bit-exact (947 cases), KLD 0 | core |
| X3 | one shared work list across small experts in `mul_mat_id` | generate | tg128 +1.1%, generate +4.2 to +8.2%. *Paired* on stack4: generate +6.1 to +7.3% | bit-exact (953 cases, 1 to 64 threads) | core |
| I4A | splits wide `get_rows` and `dup` rows over all threads | generate | tg128 +20.1%, generate +3.5 to +7.0% | bit-exact (1388 cases) | core |
| I4B | fuses `GATED_DELTA_NET` with its state snapshot copy (port of the CUDA fusion) | generate | tg128 +6.4%, generate +6.0 to +8.9% | bit-exact (52 cases) | core |
| M3 | runs tiny serial ops on thread 0, with one barrier after a run of them | generate | tg128 +1.0%, generate +4.6 to +7.7% | bit-exact (5662 cases) | core |
| W1 | splits `concat` over destination rows and tiles a transposed source | prefill | pp512 +17.7%, 8k prefill +10.6%, tg128 +3.6% | bit-exact (219 cases) | core |
| W2 | walks `dsv4_hc_post` one stream row at a time, splits norm rows over every dimension | prefill | pp512 +7.3%, 8k prefill +6.6% | bit-exact (464 cases) | core |
| K3 | fuses the MoE weighted reduction (port of the CUDA fusion) | prefill | *paired* on stack2: pp512 +2.6%, pp2048 +3.2%, 8k prefill +0.7% | bit-exact (306 cases) | core |
| U4 | batch-1 decode attention (port of upstream #27478), and 2 MiB aligned host buffers with huge pages | generate | tg128 +1.7%, generate -15.5 to +4.7%, 32k depth read +5.1%. *Paired* on stack4, 16 prompts: generate +0.9% ± 1.8%, acceptance +0.007 ± 0.016 | numerically close: 4574 of 5317 cases differ. KLD 0 at 8192, but perplexity reads in batches and does not run this path. Greedy replies differ. | core |
| K1 | honours the `n_kv_max` sparse hint in tiled flash-attention prefill | prefill at depth | on U4: 32k depth read +27.6% (30.7 against 24.0 tok/s) | bit-exact against U4 (5338 cases, 32 and 16 threads). KLD 0, PPL ratio 1.0001 at 8192 | core |
| I1 | SIMD sigmoid, and a vectorised gated `dsv4_hc_pre` | prefill | *paired* on stack2: pp512 +2.4%, 8k prefill +1.7% | numerically close: 10 of 162 cases differ, max error 1.2e-7. On stack2: KLD 0.020, top 96.0%, PPL ratio 0.9972 ± 0.0037 at 512 | optional |
| Q1 | sorts only the top k of an `argsort_top_k` row (MoE router) | generate | *paired* on stack2: tg128 +1.3%, generate +0.9 to +3.2% | bit-exact (1047 cases). On stack2: KLD 0.002, top 99.5%, PPL ratio 0.9984 ± 0.0015 at 512 | optional |
| M5 | runs the QSA mask `repeat` and `sum_rows` on all threads | prefill at depth | *paired*: 8k prefill +0.4%, pp2048 +1.7% | bit-exact (30 cases) | parked: its cost grows with depth, and no depth run exists |
| X2 | four accumulators in the AVX2 `vec_dot_q8_0` | prefill | *paired*: 8k prefill +0.6%, pp512 -3.5% | numerically close: 93 of 191 cases differ | dropped: no gain past drift |
| Q2 | `nth_element` in `top_k` | prefill | *paired*: 8k prefill +0.2% | order differs in 114 of 627 cases | dropped: no gain past drift |
| U1 | each thread converts whole `src1` rows (upstream #29308) | generate | tg128 +0.9%, generate -0.5 to +3.4% | bit-exact (2862 cases) | dropped: no gain past drift |
| U5 | adaptive MTP draft depth (#27210 plumbing, Strata's controller) | generate | generate -0.2 to +2.3% | greedy replies identical | dropped: a fixed n-max 3 is as fast |
| S1 | a reduced draft vocabulary for the MTP head | generate | acceptance -1.9 to -18.1%, `list` -11.7% | target output unchanged | dropped: lost acceptance costs more than the smaller head saves |
| S3 | an attention window for the MTP layer | generate at depth | not measured | no ggml change | dropped: the ggml-org draft's MTP layer is QSA, so the window does not apply |
| I4C | row-blocked `GATED_DELTA_NET` for prefill | prefill | op microbenchmark: at most 1.33x at 32 threads, none at 16 | bit-exact | dropped: fails the 1.5x gate of its test plan |
| T1 | checks the `src1` layout before the tiled matmul | none | a correctness fix | bit-exact, with a regression test | outside the speed stack |

The single-run changes come from the report against the `B-main` median over
every round. They therefore differ a little from the first version of this
page, which used the `B-main` round of the same day.

These designs were not built. U2, U3, U6 and G1 cannot gain on this box, by
the analysis in `designs.json`. I3 saves less than 1% of a token. S2 is already
in master: `--spec-type` takes a chain of drafters. I2, K4, K5, K6, M6 and X4
wait for a later round.

### What queues 4 to 7 settled

**I1 and Q1 move the output, but cost no perplexity.** Stack3 gave a KLD of
0.020 at 512 where stack2 gave 0. Queue 5 measured each patch alone on stack2.
I1 alone gives KLD 0.020 and Q1 alone 0.002, so I1 is nearly all of it. Each
PPL ratio is below 1, by about one uncertainty or less. Both are kept, in the
optional tier, because they change the output bits.

**K1 is now bit-exact, and pays for it.** The first K1 packed the live cells
of each token group into tiles. That gave KLD 0.0167 and 96.9% same top token
at 8192. The fix lists whole 64-cell tiles, as the dense tiled path sums them.
It gives KLD 0, PPL ratio 1.0001, and the same bits as U4 in every flash
attention case. The 32k read drops from 34.5 to 30.7 tok/s, because 94% of
the tiles are live against 69% of the cells.

**X3 holds in the real model.** In the microbenchmark below, stack3's expert
matmuls took 5 to 17% longer than mainline's at 1 and 4 tokens on 32 threads.
Queue 6 ran stack4 against stack4 without X3, in three alternating rounds of
2 reps at 256 tokens. With X3, generate is 6.4%, 7.3% and 6.1% faster on the
three prompts by the median, and 5.3 to 6.1% by the mean. The rounds ranged
from +1.3% to +10.5%. Both sides write the same greedy replies, by hash, and
accept the same share of drafts.

**The depth test repeats.** `B-main` at 32k read 24.0 tok/s in queue 2 and 23.8
in queue 4. At 128k its two runs read 18.7 and 19.5 tok/s. Generate at depth
is not a result: its rate follows the reply and the acceptance, which vary
from 0.19 to 0.72 between reps.

**Flags on stack3, each against stack3 in the same queue:**

| flag | change | result |
|---|---|---|
| `--draft none` | 7.1 to 7.2 tok/s, acceptance 0, against 7.5 to 12.3 with MTP | MTP is worth 4.7 to 72.3% |
| `--spec-draft-p-min 0.5`, *paired* | -4.9% on `code`, +33.2% on `explain`, +2.7% on `list` | mixed, and it changes the replies. The launch scripts do not set it. |
| `-ub 2048`, one backend, 8k prefill | +6.7% (62.0 against 58.1 tok/s) | see the two-backend test below |

**A larger ubatch does not show a gain with two prefill backends.** In the
deployed layout two prefill backends share the socket. Queue 5 ran the pair
test on stack4 at ubatch 512 and 2048, in two alternating rounds:

| round | ubatch 512, both | ubatch 2048, both | change |
|---|---|---|---|
| 1 | 89.0, 89.5 | 93.3, 94.2 | +4.6% |
| 2 | 90.1, 89.2 | 87.6, 90.7 | -0.8% |

The load average was 44 to 61 during the test, from other containers. The
result is inconclusive, so the prefill backends keep ubatch 512.

### What queues 8 and 9 settled

**The tiered patch set builds, and core alone gives KLD 0.** Queue 8 built the
patch set as a user does, with `tools/get-llama.sh`: core alone (`CORE`), and
core with `CPU_OPTIONAL="I1 Q1"` (`FULL`). The tree of `FULL` equals
`perf/stack4`. Core gives KLD 0 at 512 and 8192, with the same top token at
every position. Its PPL ratio is 1.0004 ± 0.0004 at 512. At 8192 it is
1.0001, the floor of the stored base logits, so a bit-exact build reads no
lower. `FULL` gives KLD 0.019 and PPL ratio 1.0045 ± 0.0028 at 8192.

The speed test ran `CORE` and `FULL` in 2 alternating rounds. Each cell is the
median over both rounds:

| test | `CORE` | `CORE` against `B-main` | `FULL` against `CORE` | `FULL` against `CORE`, by round |
|---|---|---|---|---|
| pp512 | 93.4 | +44.2% | +3.4% | +4.3%, +2.8% |
| pp2048 | 92.0 | +45.3% | +1.5% | +1.8%, -3.1% |
| tg128 | 11.42 | +25.9% | +2.7% | +3.1%, +2.2% |
| 8k prefill | 60.3 | +36.9% | +1.5% | +1.8%, +1.5% |

`CORE` against `B-main` is against the `B-main` median over every round, so
drift applies to it. The generate test is not a speed comparison between
`CORE` and `FULL`. The two builds write different texts, by hash, on all three
prompts, and they accept a different share of drafts. `FULL` accepts 0.535
against 0.423 on `explain`, and writes 17.8% faster there. `CORE` does not
write the text of `B-main` either, because of U4.

The arbiter's tests pass on the core build: 577 unit tests, and the 66 live
tests in `tests/live` with small models. One live test expected the old
fork's error text for a state too big to restore. Commit 3b270e4 changed it to
the wording of master.

**U4's decode change does not move MTP acceptance.** On three prompts U4
generated -15.5 to +4.7% against `B-main`, and `explain` accepted 13.7% less.
Queue 9 ran stack4 against `perf/stack4-noU4dec` on 16 prompts, in 2
alternating rounds with MTP and one round without. Each figure is the mean
over the 16 prompts, with its standard error:

| measure | U4 against no U4 |
|---|---|
| acceptance | 0.626 against 0.619, difference +0.007 ± 0.016 |
| generate with MTP | +0.9% ± 1.8% |
| generate without MTP | +0.1% ± 0.5% |
| greedy text | differs on 14 of 16 prompts with MTP |

U4 changes the text, but it costs no acceptance and gains no generate speed at
short context. U4 stays in core. Its gain is at depth: the 32k depth read is
5.1% faster.

### Lessons for the bench

- **Compare generate speed only between builds that write the same text.**
  The acceptance follows the text, so a build that samples another reply
  generates at another rate. When the texts differ, compare over many
  prompts, as queue 9 does.
- **A depth fill takes hours.** A 123k-token fill takes 78 minutes at about
  26 tok/s, and the 240k fill took about 2 hours 50 minutes. In queue 5 the
  30-minute request timeout cut off the 128k fill, so no slot was saved and
  every depth run after it failed. The fill now waits up to 6 hours.
- **Compare a depth run only with runs that restore the slot in a new
  process.** A run in the process that did the fill reads slower. At 240k it
  read 16.4 tok/s against 18.2 and 18.1 for the rerun. At 128k the gap was
  smaller: 19.1 and 18.3 against 19.8 and 19.2.

### The expert matmul microbenchmark

`/home/atle/llama-arbiter-bench/microbench/` times the expert matmuls of the
model alone: 512 Q8_0 experts, gate and up, then down. It runs four builds in
three rounds, pinned to the 32 physical cores:

- A is mainline `perf/base`.
- B is stack3, which holds X1 and X3.
- C is ik_llama.cpp with Q8_0 weights.
- D is ik_llama.cpp with the weights repacked to Q8_0_R8, as `-rtr` does.

Time over time, at 32 threads (`results.txt`):

| op | tokens | A over B | B over D |
|---|---|---|---|
| gate and up | 1, 4 | 0.94x, 0.95x | 0.98x, 0.99x |
| gate and up | 512 | 1.69x | 1.13x, and 1.19x against ik's fused op |
| down | 1, 4 | 0.86x, 0.89x | 0.98x, 0.97x |
| down | 512 | 1.44x | 1.25x |

At 512 tokens X1 makes the expert matmuls 1.44 to 1.69x faster than
mainline, and the repack is 13 to 25% faster again. At 1 and 4 tokens the
repack gains nothing. The op profile of a stack4 512-token prompt puts 26.8%
of the time in the three expert matmuls. A repack would therefore read a
prompt about 4 to 5% faster, and would not generate faster.

ik's `-rtr` cannot run in a server process here. It repacks the weights into
private memory, a copy of 175 GiB for each process. In queue 4 the `-rtr 1`
run of `llama-bench` ran out of memory and stopped the queue, so queue 4b ran
the rest without it. Without the repack, ik's `llama-bench` did pp512 84.5,
pp2048 87.3 and tg128 11.5 tok/s (`ik-rtr0.md`, not in the TSV). Stack3 did
95.3, 88.7 and 11.5 in the same queue.

### Pending

| cell | why |
|---|---|
| core alone at depth | only stack4 ran at 128k and 240k |
| U4's 2 MiB alignment: startup time and `compact_stall` | its commit asks for both before it is kept |
| M5 at depth | its cost grows with depth |

## Findings worth keeping

**Where a token goes.** A profile of one generate step (119 ms) puts 16.4% in
the recurrent-state `get_rows` and `cpy` of `cache_s_l`. Its rows are few and
wide, so most threads waited. I4A and I4B target this. In a 512-token prefill
graph (11.35 s), `concat` of `conv_state_at` takes 7.8%, because one thread
copies it alone. W1 fixes that. The two `dsv4_hc_post` ops take 4.7%.

**Where stack4 spends its time.** On stack4 a 512-token prompt graph takes
6.98 s, and a generate step on 16 threads 86 ms. The expert matmuls take 26.8%
of the prompt and 27.1% of a generate step. After a 32k depth, flash attention
is the largest op of a prompt, at 19.8%.

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

**A small numeric change makes a discrete flip.** U4 and the first K1 changed
attention by an NMSE (normalised mean squared error) of at most 1e-4. QSA picks
its pool and the router picks 10 experts by top-k, so a tiny change can flip
one pick. After that the hidden state takes another path. U4 has a mean KLD of
0 at 8192, but all three greedy replies change and `explain` acceptance falls
13.7%. Over 16 prompts the mean acceptance does not move (queue 9). The
first K1 showed the same as a KLD of 0.0167 with 96.9% same top token. Treat
a numerically close patch as a quality change, and test it with KLD, PPL and
greedy.

**Flags on stack2, against stack2's own generate and prefill:**

| flag | change | result |
|---|---|---|
| `--spec-draft-n-max 2` | -4.1 to -13.3% | keep n-max 3 |
| `--spec-draft-n-max 4` | -6.2 to -11.5% | keep n-max 3 |
| 24 generate threads | -2.3 to -3.1% | keep 16 |
| 32 generate threads | -11.0 to -24.3% | keep 16 |
| `--spec-type ngram-mod,draft-mtp` | -1.4 to -4.8% | no gain from the n-gram chain |
| `-ub 1024`, `2048`, `4096` on prefill | +5.2%, +6.6%, +7.0% | one backend alone. With two backends at once the gain does not show, so 512 stays. |

## Follow-ups for the GPU box

`docs/GPU-FOLLOWUPS.md` lists every item below, and the GPU ideas of Strata
and upstream, with a priority and a way to test each one.

- **Test every patch in a CUDA build there.** None of these branches was
  built with `-DGGML_CUDA=ON`. The patches also change `ggml.h` and `ggml.c`,
  which every backend shares: Q1's top-k hint and U4's 2 MiB alignment.
- **Try the VRAM expert cache.** Upstream #29887 (open) keeps hot MoE experts of
  a host-memory model in VRAM.
- **Balance `--n-cpu-moe` again.** The CPU patches change how fast the experts
  in RAM run, so the best split between card and RAM can move.
- **Measure ubatch 4096 to 8192 for prefill.** A larger ubatch reads each
  expert's weights once for more tokens.
