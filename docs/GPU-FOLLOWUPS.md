# Speed work to test on the GPU box

This page lists the speed ideas for Qwen3.8-Flash-Next (`qwen4exp`, Q8_0) that
need a card to test. The CPU box (llm-lab, one EPYC 7502P, no card) found them
and could not measure them. That box measured the CPU backend only: the result
is the patch series in `patches/cpu/`, reported in `docs/PERF.md` and
`patches/README.md`. `tools/get-llama.sh` builds llama.cpp with that series,
and `tools/perf-ab.py` is the A/B harness. The reader is whoever tests on the
GPU box, a person or a Claude agent.

The GPU box runs the arbiter at about 130k context on one 16 GiB card, with
`-ngl 99 --n-cpu-moe 48`. The launch default in `bin/qwen-mtp.sh` is
`CTX=150000`, which `docs/LAYOUT.md` sizes for this card. Attention, the
recurrent state and the KV cache are on the card. All 48 expert layers are in
RAM. The CPU box runs about 250k context.

## Results on koishi

koishi is the GPU box: two Xeon Gold 6150 sockets of 18 cores (AVX-512
F/BW/VL, no VNNI), an RTX A4000 16 GiB on PCIe gen3 x8, ctx 150000. Every row
is one A/B in one session, the rounds alternating. "CPU run" is a CPU-only
server bound to node 0; "GPU run" is the gpu backend's layout (experts in RAM,
the rest on the card). The tool is `tools/op-profile.cpp`: a warm read of a
1024- or 2000-token prompt, and steps of 4 tokens (an MTP verify).

### Deployed

| change | conditions | baseline | with it | change |
|---|---|---|---|---|
| pin + core CPU patches (PR #3) vs the old fork | CPU-only server, node 0 idle, 120 / 2000-token prompts | 25.1-28.9 / 34.8-36.0 tok/s | 34.6-39.3 / 50.2-50.3 tok/s | prefill +38 / +42% |
| `GGML_OP_OFFLOAD_MIN_BATCH=1000000` on the gpu backend | GPU run, idle, 120 / 2000-token prompts | 8.8-11.0 / 29.5-30.0 tok/s | 61.7-75.8 / 111.5-112.1 tok/s | prefill 4-7x |
| `--cpu-moe-draft` (needed to fit ctx 150000) | gpu backend at ctx 130000, generate, 3 prompts x 2 rounds | 15.0-15.2 / 19.7-20.0 / 17.5-18.0 tok/s | 14.4-14.5 / 18.9-19.1 / 16.8-17.1 | generate -4 to -5% |
| `--backend-sampling` | gpu backend, temperature 1.0, 3 prompts x 2 rounds | 13.2-13.6 / 18.0-18.1 / 16.5-16.7 tok/s | 13.5-14.6 / 18.7-19.4 / 16.9-17.1 | generate +2 to +5% |
| `core/0018`: chunk work when the process is on one node | CPU run, node 0 idle, 2 rounds | verify 322.1, 316.5 ms; prompt 19.56, 19.18 ms/token | verify 304.0, 304.5 ms; prompt 19.21, 19.39 | generate +5%, prompt the same. PPL identical |

### Measured, not deployed

| change | conditions | baseline | with it | verdict |
|---|---|---|---|---|
| optional I1 + Q1, together | CPU run, node 0 quiet, 3 rounds (main session) | prompt 19.51 / 19.56 / 19.62 ms/token; verify 317.0 / 317.2 / 318.5 ms | prompt 18.10 / 18.35 / 18.52; verify 302.8 / 301.8 / 301.9 | conflicts with the row below: unresolved |
| optional I1 alone, Q1 alone | CPU run, node 0 quiet, 3 rounds B / I1 / Q1 (agent optsets) | prompt 19.79 / 19.84 / 19.56; verify 323.2 / 314.8 / 310.8 | I1: prompt 22.01 / 18.58 / 18.02, verify 316.1 / 307.5 / 312.1. Q1: prompt 19.14 / 18.93 / 20.07, verify 310.8 / 309.1 / 305.1 | no end-to-end gain beyond the baseline's spread (verify 12.4 ms). Q1 was faster at verify in every round, by 6 to 12 ms |
| quality of I1 and Q1 | 20 chunks of 512, KLD against the deployed build, which is deterministic (base against base: KLD 0, 100% same top) | PPL 11.5921 | I1: KLD 0.0449, same top 89.3%, PPL 11.7051 (the SIMD sigmoid alone: KLD 0.0451). Q1: KLD 0.0017, same top 99.3%, PPL 11.5974, identical for 3 chunks | I1: KLD at the level of any change of summation order here (a CPU-only run is 0.047 from a GPU run), so quality does not rule it out; its end-to-end speed gain is unproven. Q1 not bit-exact (router ties, unverified), small |
| MoE weighted sum where the experts run (agent moesum, `exp/moesum` 5ee87eeb4, `src/llama-graph.cpp`) | GPU run, node 0 quiet, 3 rounds, 1024-token prompt | prompt 8.38 / 8.39 / 8.47 ms/token; verify 136.5 / 132.9 / 134.1 ms | prompt 7.48 / 7.55 / 7.51; verify 131.6 / 130.4 / 132.8 | prefill +10.7% (2000 tokens: 8.42 -> 7.57). Verify about 2%, within noise. The boundary copy per layer at 512 tokens fell from 50 MB to 5 MB. 20 chunks: KLD 0.0447, same top 89.8%, PPL ratio 0.998 ± 0.005, against the GPU run; a CPU-only run gives KLD 0.0471, same top 89.0%. Ready to deploy as `patches/moe-sum-where-the-experts-run.patch` |
| N1 expert cache, #29887, 1500 MiB | GPU run, ctx 150000, 3 prompts x 2 rounds | 14.5-14.8 / 18.9-20.2 / 16.9-18.7 tok/s | 1.8-1.9 / 2.1 / 2.0 tok/s | rejected: 0.00% hits on this card |
| `--spec-draft-p-min 0.5` (H3) | gpu backend, temperature 1.0, 2 rounds | see `--backend-sampling` baseline | 12.9-13.4 / 17.3-17.6 / 15.3-16.3 tok/s | rejected: -2 to -5% |
| threads 14 / 16 / 17 / 18 (`-t` = `-tb`) | CPU run and GPU run, node 0 quiet, two passes in reversed order | 18: GPU prompt 8.41, 8.37 ms/token; CPU prompt 20.27, 19.61 | GPU prompt 17: 8.69, 8.72; 16: 9.16, 9.02. CPU prompt 17: 20.25, 20.40; 14: 24.09, 23.51 | keep 18. Verify: no count beats the spread (up to 21 ms CPU, 4 ms GPU). A 4-token verify runs on the `-tb` pool, so `-t` alone changes nothing there |
| 36 threads (hyperthreads) for the experts | test-backend-ops, model shapes, box busy | 4-token matmul 0.83-0.86 ms | 10.9-12.1 ms | rejected |

### Where the time goes (op-profile, GPU run, ctx 150000, idle)

- 2000-token prompt, 9.2 ms a token: the CPU expert matmuls take 82% (gate
  29%, down 28%, up 25%). The card, the copies and the waits take 17%.
- A 4-token verify step, 145 ms: the CPU expert matmuls take 74%, the card,
  copies and waits 23%.
- At 4 tokens the expert matmul reads its weights at about 84 GB/s, against
  about 128 GB/s in theory for one socket.

### Facts about this box

- The ggml-org draft must be the 30 September upload. The 9 September file
  gives the MTP layer a compress ratio of 0, and the server aborts at load
  with `GGML_ASSERT(buffer) failed`.
- #29887 now needs upstream 6753a033f first: it was rebased after this page
  checked it.
- This model is sensitive to the order of f32 sums: a CPU-only run and a GPU
  run of the same build differ by mean KLD 0.047 and 11% of top tokens over
  20 chunks. Judge a change's KLD against that line, not against 0.
- Load on node 1 moves node-0 numbers: the same CPU-run setting read prompts
  at 19.6 to 20.3 ms/token with node 1 busy and 18.7 to 18.8 with it idle.
  Compare only inside one session's alternating rounds.
- CUDA graphs are captured per GPU split, that is per layer between two CPU
  expert phases.

## Sizes that decide most of this

These come from the model header: 48 layers, 512 experts, 10 used, an expert
width of 640 over an embedding of 2560.

| quantity | at Q8_0 |
|---|---|
| one expert (gate, up and down) | about 4.98 MiB |
| the experts of one layer | about 2.49 GiB |
| all experts | about 119.5 GiB |
| expert weights one generated token reads (10 x 48) | about 2.33 GiB |
| KV cache at 130k, f16 (36.5 KiB a token, `docs/LAYOUT.md`) | about 4.5 GiB |

Of the 36.5 KiB a token, 24 KiB is K and V: 12 layers hold KV, each with 2 KV
heads of 256 dimensions. The rest is probably the QSA indexer keys. QSA is
the sparse attention of `qwen4exp`.

Two facts about llama.cpp at the pin `8e1642198` follow from these sizes:

- **Prefill uploads the experts.** A `MUL_MAT_ID` with 32 or more tokens runs
  on the card (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32). The scheduler copies
  only the experts a ubatch uses. At ubatch 512 that is nearly all 119.5 GiB,
  so each ubatch moves about 128 GB over PCIe. PCIe 4.0 x16 moves at most
  about 25 GB/s, so prefill stays near or below 100 tok/s, before any
  compute. A larger ubatch moves the same bytes for more tokens.
- **Generate runs the experts on the CPU.** An MTP verify is 4 tokens, below
  32, so the experts stay in RAM. Speed is bound by how fast the CPU reads
  about 2.33 GiB a token.

These estimates are for planning. Measure before you trust them.

## Before anything else

No branch of the CPU series was built with `-DGGML_CUDA=ON`. Do the steps
below in order, before you test any item on this page.

### Build two trees

1. Build the patched tree (the `stack4` of `docs/PERF.md`, core plus both
   optional sets): `CUDA=1 CPU_OPTIONAL="I1 Q1" tools/get-llama.sh`.
2. Build the extra targets in that tree:
   `cmake --build llama.cpp-mtp/build -j --target llama-bench llama-perplexity test-backend-ops`.
3. Clone and patch a second tree for the reference (`B-main`):
   `BUILD=0 DIR=$PWD/llama.cpp-main tools/get-llama.sh`.
4. Remove the core CPU patches from it, last patch first:
   `for p in $(ls -r patches/cpu/core/*.patch); do git -C llama.cpp-main apply -R "$PWD/$p"; done`.
5. Configure it: `cmake -S llama.cpp-main -B llama.cpp-main/build -DGGML_CUDA=ON`.
6. Build it:
   `cmake --build llama.cpp-main/build -j --target llama-server llama-bench llama-perplexity test-backend-ops`.

### Check the ops the series touches

A CUDA build compiles the CPU backend too, and Q1 and U4 change `ggml.c`, which
every backend shares. `test-backend-ops` compares the card against the CPU backend,
so a mismatch here means the patched CPU backend and the card disagree.

    llama.cpp-mtp/build/bin/test-backend-ops test -b CUDA0 \
      -o 'MUL_MAT_ID,GET_ROWS,CPY,DUP,GATED_DELTA_NET,CONCAT,NORM,RMS_NORM,DSV4.*,SIGMOID,ARGSORT,TOP_K'

Run the same command with the `llama.cpp-main` build. A case that fails in
both is upstream's. A case that fails only in the patched build is ours.

### See where each op runs

Start one backend with `GGML_SCHED_DEBUG=2` and read which backend each node
of a generate step and a prefill ubatch gets. The last section depends on
this: a CPU patch can only gain where its op runs on the CPU.

### Take a baseline

`tools/perf-ab.py` is written for the CPU box. Change these at the top of it,
on the GPU box only:

- `BENCH` (or the `BENCH_DIR` variable), `MODELS`, `MODEL` and `DRAFT`, to that
  box's paths. Copy `prompt-8k.txt` and `wiki.test.raw` from
  `/home/atle/llama-arbiter-bench` on llm-lab into `BENCH`.
- `CTX`, to the backend's own context, about 131072.
- In `COMMON`, `--device none` becomes `-ngl 99 --n-cpu-moe 48` (or that box's
  `NGL` and `N_CPU_MOE`).
- In `ROLE_FLAGS`, threads, `--cpu-range` and ubatch, to the GPU backend's
  launch flags (`bin/qwen-mtp.sh`).
- In `run_kernel`, the thread count and the CPU mask, and add
  `-ngl 99 -ncmoe 48`. `llama-bench` has no `--device none` there, and its
  default offloads every layer.
- In `run_quality`, the thread count, and `--device none` the same way as
  `COMMON`.

Then measure each build:

    tools/perf-ab.py prefill  --label B-main --build llama.cpp-main/build
    tools/perf-ab.py generate --label B-main --build llama.cpp-main/build
    tools/perf-ab.py depth    --label B-main --build llama.cpp-main/build --depth 131072
    tools/perf-ab.py quality  --label B-main --build llama.cpp-main/build --base

Repeat the first three with `--label stack4 --build llama.cpp-mtp/build`, and
`quality` without `--base`. The first `depth` run reads 131072 tokens and parks
a copy of the slot. Later runs recall that copy. A change to the KV type (N2)
cannot recall it, so move `BENCH/slots/depth128k.bin` aside before you
measure one.

### Keep the A/B honest

The method of `docs/PERF.md` holds here too:

- **Drift.** Run `B-main` again every few hours. On the CPU box it moved up to
  6.2% between rounds. A single-run change smaller than the drift is not a
  result.
- **Paired A/B.** For a gain near the drift, run the candidate and its base in
  two or more alternating rounds.
- **KLD gate.** Write base logits with `B-main` on the GPU box. Before you keep
  a change, check its mean KL divergence (KLD) and same-top-token rate. A
  change to attention or to the KV type needs the check at context 8192 too:
  a small numeric change can flip a QSA or router pick, and then the reply
  changes.
- **Same mode.** Compare MTP with MTP, and the same `--spec-draft-p-min` with
  the same. The verify batch sums in another order than a one-token step.

## The candidates that need a card

Each upstream state was read from the GitHub API on 6 October 2026. "Applies"
means `git apply --check` of the PR diff passed on the pin `8e1642198`. For
#29887, #28785, #29797 and #27478 it also passed on top of `patches/cpu/`.
Strata items are at Strata `6f32ec0`, an MIT engine for this one model on
consumer NVIDIA or AMD cards. Its HEAD is now `1735d64`. Unless a row says
otherwise, Strata measured on an RTX 5070 12 GB with a Ryzen 5 7600. A Strata
figure that its docs and PR pages at `6f32ec0` do not hold is left out.

The priority is for a 130k-context arbiter backend on this card, with Q8_0
experts in RAM. The rows are in order of priority.

| id | what | source | claimed gain, and where | how to test | priority |
|---|---|---|---|---|---|
| N1 | VRAM expert cache: hot experts of a RAM layer live in VRAM, misses are uploaded. Batches over 32 tokens skip it. | upstream #29887 (open, applies), `--moe-cache-mib N`. Related: #27861 (open draft, does not apply), #28414, issue #29949 (open), Discussion #24528. Strata `src/core/expert_cache.cpp`. | #29887: Q4_0, RTX 4090 1.57x to 1.62x at 72 to 77% hits, RTX 5090 up to 2.20x. #27861: +31% on 2x RTX 3090. Strata: 72% of the Coder's expert reads hit a 12 GB card. | Cherry-pick #29887. Measure generate and the hit rate at the VRAM that f16 KV leaves. | **high**, but the gain here is doubtful: see the note below the table |
| N2 | KV cache at q8_0 | Strata KV int8, K8V4 (#120). In llama.cpp: `-ctk`, `-ctv`. | Strata K8V4: 85 to 99 tok/s (RTX 3090, the Coder at 198K). Rotated q4_0 (#21): +4% at 128K, perplexity +8 to 12%. | `-ctk q8_0 -ctv q8_0`. The default build has the `q8_0-q8_0` flash-attention kernels. A mixed K8V4 needs `-DGGML_CUDA_FA_QUANTS='q8_0-q8_0;q8_0-q4_0'`. `GGML_CUDA_FA_ALL_QUANTS` is a deprecated alias for all pairs. Gate with KLD at 8192. | **excluded**: a q8_0 KV cache has given poor output quality on this model in the owner's own use, so the arbiter keeps f16 KV |
| N3 | prefill in larger ubatches | Strata `--prefill auto`, `src/prefill/prefill.cpp` | Q2_0 at a chunk of 4096: 791, 6144: 878, 8192: 973 tok/s | `-ub 2048`, `4096`, `8192` with `-b` as large. Watch VRAM: the compute buffer grows with the ubatch. | **low**: the owner has tested larger prefill ubatches several times with no worthwhile gain and often a loss. Test only if N1 or N4 changes the upload cost, and only in the deployed layout |
| N4 | upload the next split's experts while this split computes | upstream #28414 (open draft, does not apply), `--prefetch-experts-slots 3`. Strata expert ring sized in bytes (#583). | #28414: time to first token -11 to -22% on 42k-token prompts, 24B and 35B A3B models, RTX 5070 Ti. Strata ring: 1130 to 1290 tok/s (Q2_0). Byte-sized ring: IQ3_XXS +18.5% at 32K. | Rebase #28414 on the pin. A/B on 8k prefill at each ubatch of N3. | **medium**: it hides the upload behind compute, but only while compute is as long as the upload |
| N5 | split the misses of N1: some copied over PCIe, some computed on the CPU | Strata `--pcie-frac`, `src/core/expert_source.cpp`, `src/core/remote_expert_opt.cu` | no figure in Strata's docs. They say `--pcie-frac 0`, all misses on the CPU, costs decode speed. | Needs a port on top of #29887, which uploads every miss. | **medium**, only if N1 shows the uploads are the limit |
| N6 | sampling on the card | Strata GPU top-k sampler (#197, `src/kernels/cuda/sampler.cu`). Upstream #29797 (open, applies): greedy for temperature-0 chains. | Strata: sampled decode +4 to +42%, by top-k. #29797: +6.6% on an RTX PRO 6000 with a dense 27B. | `--backend-sampling` and `--spec-draft-backend-sampling` are in the pin. Cherry-pick #29797 for temperature 0. | **medium**: cheap to try. Check that grammar turns (`/v1/systemone`, `grammar-probs.patch`) still answer. |
| N7 | fewer synchronisations in the output getters | upstream #29796 (open, does not apply) | +12.0% for an earlier version of the PR, on an RTX PRO 6000, dense 27B with a 7-token draft | Rebase on the pin. It bumps the backend API version. | **low**: it helps when the card sets the pace. Here the CPU experts set it. |
| N8 | skip the CPU threadpool for a graph with no CPU node | upstream #28785 (open, applies) | tg128 1.22x on an Apple M3, full offload | Cherry-pick. It can only help a graph with no CPU node, such as a draft fully on the card. | **low** |
| N9 | one CUDA graph per verify window, a pinned doorbell, fewer launches | Strata #646, `src/core/graph.cpp`, `verify.cpp` | Strata 0.1.39 took it: decode +2 to +11%, where the one-graph path does not run and the gain is its per-layer kernels. The PR claims +39 to +72% with every expert in VRAM on 2x RTX 3090. | Needs a port. First check whether llama.cpp uses CUDA graphs for the split graphs of a verify. | **low**: the one-graph path needs every expert in VRAM, which never holds here |
| N10 | Strata kernels: grouped expert launches (#372), DeltaNet 3 heads a thread (#413), GDN input prefetch (#188), QSA scores read once (#187), q4_0 KV on tensor cores (#452), per-warp histogram top-k (#603) | Strata `src/kernels/cuda/` | +2 to +26% each, on several cards. #603 gains only past about 135K tokens of context. | Needs a port each. Profile first: port one only if its op is a real share of a generate step on the card. | **low** |
| N11 | KV streaming: only the last 32768 positions stay in VRAM | Strata `--kv-resident`, `src/kernels/cuda/kv_stream.cu` | 50.9 to 62.6 tok/s at 262K | Needs a port. | **low**: at 130k the KV cache fits |
| N12 | block select and argmax with thread-block clusters | Strata `src/kernels/cuda/qsa_select.cu` | 64.5 to 76.4 tok/s at 128K, RTX 50 | Needs a port and compute capability 9.0 or later. | **low**: check the card's generation first |
| N13 | int8 tensor-core matmul for prefill, not FP16 with cuBLAS | Strata 0.1.13 and 0.1.36, `src/prefill/moe_mmq.cu`, `moe_fused*.cu` | MMQ: 1052 to 1130 tok/s (Q2_0). Fused int8 (0.1.36): +16 to 22%. | llama.cpp already uses MMQ for Q8_0. Check only that the build does not set `GGML_CUDA_FORCE_CUBLAS`. | **low**: nothing to port |
| N14 | the per-layer embedding (PLE) block as two GEMMs per chunk | Strata `src/prefill/kernels.cu` | 981 to 1053 tok/s | Needs a port. Test H2 first. | **low** |
| N15 | pinned RAM for the experts | Strata pins every expert in RAM (`docs/HOW_IT_WORKS.md`), and copies an expert it could not pin through pinned buffers | IQ3_S prefill at a chunk of 4096: 551 to 652 tok/s from the pinned copies | `--load-mode none` (the pin has no `--no-mmap`) puts the experts in pinned `CUDA_Host` buffers, about 120 GiB. The loader itself suggests it when experts are overridden to the CPU with mmap on. The lazy PLE stays mapped from the file, because the loader maps any lazy tensor. A/B the 8k prefill of N3 with and without it. | **low to medium**, but only a flag to try, not a port. Pageable copies are slower than pinned ones, which matters for N3. |
| N16 | router lookahead with `MADV_WILLNEED`, a RAM budget of hot experts | Strata `--resident-budget-gib`, `--mmap-experts` | UD-Q4_K_XL: from 3 to 7 or 8.5 tok/s | Needs a port. | **low**: the model fits RAM with `--lazy-mode auto` |
| N17 | layers split over cards, `--remote-expert-opt`, `--trim-stage-weights` | Strata `docs/MULTI_GPU.md`, `docs/SECOND_GPU.md`, `docs/BATCHING.md` | `--remote-expert-opt`: up to +132% code decode on 2x RTX 4090, over the plain helper path | | **none**: one card |
| N18 | batch slots | Strata | -11 to -24% total on a 12 GB card: each slot takes VRAM from the expert cache, and a request in a slot decodes without MTP drafts. 120 to 360 tok/s on 4 cards. | | **none**: the arbiter runs one slot a backend |

**Why N1 is high, and why it may still lose here.** The VRAM expert cache is
the largest gain in every source, and #29887 applies to the pin as it is. On
this box, the sizes work against it:

- #29887 advises a cache of 10% of the expert bytes. At Q8_0 that is about
  12 GiB. After the weights on the card and a 130k KV cache, this card has
  room for perhaps 2 to 3 GiB with a q8_0 KV cache, and less with the f16 KV cache the arbiter keeps (N2 is excluded). That is about 400 to 600 experts, or 8
  to 13 a layer, against 10 used a token.
- #27861 measured the routing of this model. A cache of 64 experts a layer
  hits about 67%, and 128 hits about 81%. A cache of 8 to 13 a layer hits far
  less. #27861 caches only a one-token decode, so an MTP verify bypasses it.
- #29887 runs every small `MUL_MAT_ID` on the card, and uploads every miss.
  At 4.98 MiB an expert and about 25 GB/s, 400 misses a token cost about
  80 ms. The CPU may read the same experts from RAM faster.

Test it, because a measured hit rate replaces these estimates. If it loses at
Q8_0, N5 is the fix: Strata computes a share of the misses on the CPU. A
smaller quant also changes the sums: #29887 itself measured at Q4_0.

### Upstream state, as verified

| item | state on 6 October 2026 | applies to `8e1642198` |
|---|---|---|
| #29887 MoE expert cache, `--moe-cache-mib` | open | yes, also on top of `patches/cpu/` |
| #27861 VRAM LRU cache for experts in RAM | open, draft | no |
| #28414 `--prefetch-experts-slots` | open, draft | no |
| #29949 issue: expert cache with VRAM LRU | open | not a PR |
| Discussion #24528: RFC, VRAM caching of hot experts | not verified: the page exists, but the REST API gives no state for a discussion | not a PR |
| #29796 fewer syncs in output getters | open | no |
| #28785 skip the threadpool without CPU work | open | yes, also on top of `patches/cpu/` |
| #29797 greedy for temperature-0 chains | open | yes, also on top of `patches/cpu/` |
| #27478 batch-1 CPU attention, 2 MiB alignment | open | yes, also on top of `patches/cpu/` |
| #20596 faster `--n-cpu-moe` generate | open | no |
| #29030 PLE rows by direct reads | open | no |
| #29599 `llama_prefetch_rows` | merged 30 September 2026, in the pin | in the pin |
| #29359 exact -128 in the x86 int8 dot | open | no. Watch it, do not take it: its q8_0 by q8_0 dot costs +23% time on AVX2 without VNNI. |

## Hybrid items: dropped on the CPU box, worth a retest

On the GPU box the CPU runs the experts and little else. A gain that was a
small share of a CPU-only step can be a larger share there.

| id | what | why it was dropped | why to retest | how |
|---|---|---|---|---|
| H1 | #20596: a fast gated path for one token (`n_tokens=1`) in the CPU experts | estimated at +0.2 to +0.4% on the CPU box from a profile, not measured | it targets `--n-cpu-moe` exactly. Note that an MTP verify is 4 tokens, so only a one-token step takes the path. | Rebase on the pin. It conflicts in `ggml-cpu.c`, beside X3 and M3. A/B generate with and without the MTP draft. |
| H2 | #29030: PLE rows by direct reads, not mmap | the CPU box holds all weights in RAM | the GPU box leaves the 50.66 GiB PLE on disk (`--lazy-mode auto`), the case it was made for. Claimed pp512 +121%, pp8192 +65% on Strix Halo, against `-lzm on` without #29599. #29599 is in the pin and took pp512 from 177 to 383 tok/s on Strix Halo by itself, so expect much less. | Rebase on the pin. A/B 8k prefill and 130k depth. |
| H3 | the MTP draft gate: draft only while its confidence is 0.5 or more | `--spec-draft-p-min 0.5` gave -2.0 to +5.1% on stack2 and changed the greedy replies | Strata gates at 0.5 (`--spec-min-p 0.5`) and measures 1.6 to 1.8x, with 2.4 to 3.2 tokens a verify pass. The cost of a verify differs on the card. | Paired A/B of generate. Compare greedy only in the same mode. |
| H4 | adaptive draft depth (U5) | -0.2 to +2.3%: a fixed n-max 3 was as fast | the ratio of draft cost to verify cost differs | Branch `perf/U5-adaptive-mtp-depth` on llm-lab. Test only if H3 gains. |
| H5 | a reduced draft vocabulary (S1) | acceptance -1.9 to -18.1% | Strata gains +15 to 38% for CJK text only | **low**: retest only for a CJK workload |
| H6 | flags: `--spec-draft-n-max` 2 to 4, `--spec-type ngram-mod,draft-mtp`, generate threads, `-ub` | n-max 3 best, no gain from the n-gram chain, 16 threads best, `-ub 2048` +6.6% for one backend alone, inconclusive with both prefill backends at once (+4.6% and -0.8% in two rounds), so 512 stays | each result measured the CPU box | Run each flag against stack4 on the GPU box. `-ub` is N3. |
| H7 | the split between card and RAM, `--n-cpu-moe` | not applicable | one expert layer on the card is 2.49 GiB. With f16 KV (N2 is excluded) there may be no room for one. The CPU series also changes how fast a RAM layer runs. | Try 47 and 46 if VRAM allows with f16 KV, against N1 with the same VRAM. |
| H8 | X2: four accumulators in the AVX2 `vec_dot_q8_0` | 8k prefill +0.6%, pp512 -3.5%: no gain past drift. Generate was not measured. Not bit-exact. | the CPU experts use this dot at generate, which is all the CPU does there. It applies only if the GPU box's CPU takes the AVX2 path. | Branch `perf/s2-X2-vec-dot-q8_0-4acc` on llm-lab. Paired A/B of generate, then the KLD gate. |

These do not apply on the GPU box, because their ops run on the card: K1, M5,
Q2 and I4C. S3 applies on neither box: the MTP layer of the ggml-org draft is
QSA, so its attention window does not apply.

## Our CPU patches, retested there

The patches in `patches/cpu/` change the CPU backend and its tests. Outside
them, two change shared code. Q1 changes `ggml_argsort_top_k` in `ggml.c`, and
a comment in `ggml.h`. U4's `core/0012` changes the alignment of large
allocations in `ggml-base`. With `--n-cpu-moe 48` the CPU still runs every
expert at generate. Read the
scheduler output from "Before anything else" before you judge a row: a patch
gains nothing when its op runs on the card.

| patch | id | runs on the CPU there? | watch for |
|---|---|---|---|
| `core/0003` | X3 | yes: each expert gets 1 to 4 rows a verify | the main gain to expect at generate. Retest the generate thread count with it. |
| `core/0006` | M3 | yes, between the expert matmuls | `GGML_CPU_SERIAL_BYTES=0` turns it off for a paired A/B |
| `core/0001`, `0002` | X1 | rarely: prefill experts run on the card at 32 tokens or more | A/B prefill with `--no-op-offload` (experts on the CPU with X1) against the upload of N3 |
| `optional/Q1` | Q1 | no: the router's argsort runs on the card | `ggml.c` changed for every backend. The CUDA argsort reads only the sort order. Check `ARGSORT` and `TOP_K` in `test-backend-ops`, a greedy hash equal to `B-main`, and no generate loss (the CUDA top-k MoE fusion must still fire). |
| `core/0004` | I4A | only if the recurrent state is in RAM | no change expected with `-ngl 99` |
| `core/0005` | I4B | only if `GATED_DELTA_NET` runs on the CPU | no change expected with `-ngl 99` |
| `core/0007` to `0009`, `optional/I1` | W1, W2, I1 | no, with `-ngl 99` | no change expected. I1 is not bit-exact, on the CPU only. |
| `core/0010` | K3 | only if the weighted sum of the experts runs on the CPU | retest if the scheduler output puts the sum on the CPU |
| `core/0011` to `0013` | U4, #27478 | its attention part does not. Its 2 MiB alignment and `MADV_HUGEPAGE` touch the CPU buffers of a CUDA build | resident memory, load time and generate. The experts are file-backed with mmap. `--load-mode none` puts them in pinned `CUDA_Host` buffers instead, so the alignment reaches them only with `GGML_CUDA_NO_PINNED=1` as well. |
| `core/0014` to `0017` | K1 | no: prefill attention runs on the card | no change expected |

### The removed fit patch

`patches/mtp-fit-ctx-other.patch` fixed the VRAM fit for a draft that
borrows tensors of the target. To read it, run
`git show 37dc46b:patches/mtp-fit-ctx-other.patch`. On the pin, `qwen4exp`
creates its own `token_embd` and the ggml-org MTP draft carries one, so the
patch should no longer be needed. Start the GPU backend with the ggml-org
draft and check that the log does not print `failed to measure the memory of
the extra model`. If it does, the fit runs without the draft, and the backend
can run out of VRAM at full context.
