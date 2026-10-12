# New speed-up candidates for qwen4exp on koishi

Research only: no code changed, nothing measured. Collected on 2026-10-07.

The model is Qwen3.8-Flash-Next at Q8_0 with an f16 KV cache. The machine is
koishi: 2x Xeon Gold 6150 (AVX-512 F/BW/VL, no VNNI, no AMX), an RTX A4000
16 GB on PCIe gen3 x8, and the pin `8e1642198`.

This list leaves out every item that `docs/GPU-FOLLOWUPS.md`, `docs/PERF.md`
and `patches/README.md` already cover. Where a source adds new evidence to a
known item, the entry says so.

## How to read this list

- **Applies to the pin** means `git apply --check` of the PR diff passed on
  a clean worktree of `8e1642198`. It does not mean that it applies on top of
  `patches/cpu/`.
- **Unverified** marks a claim that I did not trace to a measurement. A
  **forum** claim comes from a PR comment, a discussion or a fork README.
- Numbers are quoted as the source gives them. A line marked "my arithmetic"
  is an estimate, not a measurement.
- Reference figures for koishi, from `docs/GPU-FOLLOWUPS.md`:
  - A 4-token verify step takes about 143 ms. The CPU and llama.cpp take
    105 ms, CUDA graph execution 23.4 ms, and host time in CUDA calls 10.5 ms.
  - The CPU expert matmuls take 82% of a 2000-token prompt and 74% of a
    verify step.

## Summary

| # | idea | phase | effort | priority |
|---|---|---|---|---|
| R1 | keep the scheduler when a request sets its sampler (#28872) | prefill (time to first token) | small port | **high** |
| R2 | measure plain decode and MTP depth on the gpu backend | generation | flag | **high** (cheap) |
| R3 | NUMA: check where the expert pages are, then try a mirror (#27986, ik #2396) | generation | check, then cherry-pick or port | **medium-high** |
| R4 | prefill: the card computes the most-routed experts while the CPU computes the rest | prefill | port / research | **medium-high** |
| R5 | the CPU expert phase inside the CUDA stream (doorbell or host function) | generation | port / research | medium |
| R6 | MTP draft `hnorm`: llama.cpp normalises each stream, vLLM all four | generation (acceptance) | small change + A/B | medium |
| R7 | limit the experts a verify step reads (AcceptMoE, MoE-Spec, fewer experts used) | generation | flag / port, lossy | medium |
| R8 | 2 MiB pages for the expert mapping (tmpfs `huge=always`, or XFS) | both | ops | medium-low |
| R9 | Gumbel-coupled MTP drafts for sampled requests | generation | port | medium-low |
| R10 | a pinned checkpoint at the shared prefix end and the last turn boundary | prefill | port / check | medium-low |
| R11 | #29308 (= U1) again, with new dual Cascade Lake data | generation | rebase | low-medium |
| R12 | expert deferral (kTransformers) | generation | research, lossy | low-medium |
| R13 | GDN CUDA kernels: #30087, #29187, #29353, #21897 | both, mostly prefill | cherry-pick | low |
| R14 | CUDA graph fixes after the pin: #29986, #29768 | both | cherry-pick | low |
| R15 | the shared expert on the card overlaps the routed experts on the CPU | generation | port | low |
| R16 | repacked experts written into the file, shared through mmap | prefill | port | low |
| R17 | fewer barriers in a CPU MoE layer, and SwiGLU fused with the Q8_0 quantize | generation | port | low |
| R18 | MTP drafter: `--mtp-q4`, skip the K/V of rejected rows | generation | port | low |
| R19 | a cost-aware prompt lookup chained after the MTP drafts | generation | port | low |
| R20 | fastllm as a second engine | both | test | low |

## Candidates, by priority

### R1. Keep the backend scheduler when a request sets its sampler

- **Source:** llama.cpp PR #28872 (open, 2026-09-14),
  https://github.com/ggml-org/llama.cpp/pull/28872
- **What it does:** `llama_context::sched_reserve()` destroys and recreates
  `ggml_backend_sched` whenever `sched_need_reserve` is set.
  `llama_set_sampler()` sets that flag, and `llama-server` calls it for every
  request. Each request therefore frees and reallocates every compute buffer
  and the pinned host input buffer, and then page-faults on the new buffers.
  The PR calls `ggml_backend_sched_reserve()` on the existing scheduler
  instead.
- **Claimed gain:** on a 4-GPU layer split of Qwen3.8-Flash-Next, the time to
  first token of a 4-17 token continuation fell from 0.8-1.1 s to
  0.25-0.4 s, with identical outputs. "The same mechanism cost 0.2-0.4 s per
  request at `-ub 256`."
- **Applies to koishi: yes, likely.**
  - The gpu backend runs `--backend-sampling`, which was deployed for +2 to
    +5% generate.
  - In the pin, `llama_set_sampler` sets `sched_need_reserve = true` when it
    attaches or detaches an offloadable sampler (`src/llama-context.cpp`,
    around line 1317). `sched_reserve` then calls `sched.reset(ggml_backend_sched_new(...))`
    (line 643). I read both.
  - The prompts are short: 100-2000 tokens, and a 120-token prompt reads in
    about 1.6-1.9 s. A fixed per-request cost of a few hundred ms is a large
    share of that.
  - Check whether the 210 ms CUDA graph build of the first two verify steps
    after a prompt is part of the same cost.
- **Phase:** prefill (time to first token), and the first decode steps.
- **Effort:** small port. The diff does not apply to the pin (conflict in
  `src/llama-context.cpp:602`).
- **Priority:** high. First measure the request overhead with and without
  `--backend-sampling`.

### R2. Measure plain decode and the MTP depth on the gpu backend

- **Sources:**
  - ktransformers PR #2088, merged 2026-07-15, a doc of community results:
    https://github.com/kvcache-ai/ktransformers/pull/2088
  - ik_llama.cpp PR #2396, comment by st3fk3 (**forum**):
    https://github.com/ikawrakow/ik_llama.cpp/pull/2396
  - The PixelML model card, vLLM MTP on this model:
    https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash
- **What they report:**
  - kt #2088 used 1x RTX 5090 with 2x Xeon Gold 6138 (Skylake-SP, no VNNI,
    no AMX), DeepSeek-V4-Flash with experts on the CPU. "EAGLE / MTP
    speculative decode: ~28 → 27.2 tok/s (net-negative)." The reason it
    gives: "verifying a batch of draft tokens that route to *different*
    experts multiplies the CPU expert reads that are the bottleneck".
  - st3fk3 ran Qwen3.8-Flash-Next UD-Q3_K_XL on 2x Xeon E5-2697A v4 with an
    RTX 3090, 42 of 48 expert layers on the CPU: "MTP showed low or negative
    impact on decode".
  - PixelML, vLLM on GPUs: accepted length 1.844 / 2.997 / 3.367 / 3.825 at
    k = 1 / 3 / 4 / 6. k = 3 and k = 4 were "indistinguishable"; k = 1 and
    k = 6 were worse.
- **Applies to koishi: yes.**
  - `docs/PERF.md` measured MTP against `--draft none` only on the CPU box,
    where MTP was worth 4.7 to 72.3%.
  - H6 of `docs/GPU-FOLLOWUPS.md` plans an n-max sweep on koishi, but no
    `--draft none` baseline in the gpu layout. On koishi a 4-token verify
    reads more distinct experts than one token does (see R7).
- **Phase:** generation.
- **Effort:** flags: `--draft none` against n-max 1, 2 and 3, over many
  prompts, because the acceptance follows the text.
- **Priority:** high, because it is cheap and sets the frame for R7, R9 and
  R18.

### R3. NUMA: find where the expert pages are, then try a weight mirror

- **Sources:**
  - llama.cpp PR #27986, `--numa mirror` (open, 2026-08-29):
    https://github.com/ggml-org/llama.cpp/pull/27986
  - ik_llama.cpp PR #2396, NUMA mirror (open):
    https://github.com/ikawrakow/ik_llama.cpp/pull/2396
  - llama.cpp PR #16000, the original `--numa mirror` (open):
    https://github.com/ggml-org/llama.cpp/pull/16000
- **What it does:** each NUMA node gets its own copy of the large CPU
  weights. In #27986 the CPU matmul reads (`mul_mat`, `mul_mat_id`, sgemm,
  repack) go to the copy local to the calling thread, and `tensor->data` does
  not change.
- **Claimed gains:**
  - #27986, DeepSeek-V4-Flash Q8 on 2x EPYC 7532 and 5x RTX 3090: steady
    decode equals a fully warmed first-touch `--numa distribute` (16.4
    against 17.3 tok/s with mmap). With pinned memory (`--no-mmap`), mirror
    gave 128.6 tok/s prefill and 18.2 tok/s decode, against 125.1 and 11.5
    for distribute.
  - ik #2396, st3fk3 (**forum**), Qwen3.8-Flash-Next on dual Broadwell with
    an RTX 3090: mirror 25.754 tok/s, numactl interleave 22.376 (+15.10%),
    distribute 21.910. RSS was 145.9 against 73.7 GiB. Each of these
    compares two-socket policies, not one bound socket.
  - #16000, two-socket Xeon 6238R (Cascade Lake), Qwen3-32B Q6_K: tg 1.91 to
    2.70.
  - In a discussion of ik #2396, rrubberr reports (**forum**) for MoE on dual
    Cascade Lake that the second socket gives "at best ~1.3x over NUMA
    isolate".
- **Applies to koishi: partly. Check first.**
  - The four backends share one page cache of the GGUF. The gpu backend is
    bound to node 0, and two CPU backends run on node 1. A page lives on the
    node that first touched it. Part of the gpu backend's expert reads may
    therefore be remote, which could explain part of the gap between 84 and
    128 GB/s. `numastat -p <pid>` of the gpu backend answers this without a
    benchmark.
  - A mirror for the gpu backend costs about 120 GB more anonymous RAM.
  - It gains only if the gpu backend may use node-1 cores. Today those
    cores serve `cpu1_0` and `cpu1_1`. The trade is gpu generate speed
    against the CPU backends.
  - #27986 does not apply to the pin. It conflicts in `ggml-cpu.c:1299`,
    where `core/0018` and X3 also change the code.
- **Phase:** generation, and prefill if the experts become pinned.
- **Effort:** a check (`numastat`), then a cherry-pick with conflicts.
- **Priority:** medium-high for the check, medium for the mirror.

### R4. Prefill: the card computes the most-routed experts while the CPU computes the rest

- **Sources:**
  - Strata PR #1282 (open, but in main), commit `978d355`, and
    `docs/DETAILS.md`, "Short prompts: let the CPU share the experts". I read
    the doc in the local clone at HEAD `e8ca9af`:
    https://github.com/Niko1221/Strata/pull/1282
  - fastllm commit `2756bd122` (`FT_MOE_ASSIST_OVERLAP`), through the
    engines agent: https://github.com/ztxz16/fastllm
- **What it does:**
  - In Strata, for a chunk below 1,024 tokens, the CPU computes the
    non-resident experts that the fewest tokens route to. The card streams
    and computes the rest.
  - `auto` times both sides each layer and gives the CPU the share g/(c+g),
    so that both finish together.
- **Claimed gain:** medians of 10 interleaved pairs, off against auto, in ms:

  | machine | 512 tokens | 1,000 tokens |
  |---|---|---|
  | RTX 5070, Ryzen 5 7600, Q2_0 | 1,376 / 1,019 (-26%) | 1,788 / 1,392 (-22%) |
  | RTX 3060, Core Ultra 7 265, IQ3_XXS | 1,620 / 1,159 (-28%) | 2,196 / 1,770 (-19%) |
  | Tesla P100, Xeon E5-2690 v4, IQ3_XXS | 4,805 / 3,126 (-35%) | 6,821 / 5,085 (-25%) |

  The output is not bit-identical: first-token KL mean 0.006, max 0.026.
  fastllm: "16K chunk prefill 吞吐由 1496.44 提升至 1791.38 tokens/s", on
  2x RTX 3090 Ti with a synthetic model. That is a commit message, and the
  engines agent read it; I did not.
- **Applies to koishi: yes, in reverse.**
  - koishi runs every prefill expert on the CPU, because op offload is off
    (`GGML_OP_OFFLOAD_MIN_BATCH=1000000`). The card waits for most of a
    prompt.
  - The useful direction here is the opposite of Strata's: the card takes
    the experts that the most tokens route to, because their upload pays
    for itself.
  - My arithmetic: a Q8_0 expert is about 5.2 MB, so about 0.75 ms on
    gen3 x8 at about 7 GB/s. On the CPU a 2000-token prompt spends about
    16 us per token per expert-layer, so an expert with 100 or more tokens
    costs the CPU about 1.6 ms or more.
  - The split must overlap the upload with the CPU work. It probably needs
    pinned host memory: N15 is a flag that already gives that.
  - llama.cpp cannot split one `MUL_MAT_ID` between two backends today.
- **Phase:** prefill.
- **Effort:** port / research: a split `MUL_MAT_ID` in the scheduler, or a
  custom op. Measure the token count per expert in a real prompt first.
- **Priority:** medium-high. It targets the largest share of prefill time,
  but it is the largest piece of work on this list.

### R5. The CPU expert phase inside the CUDA stream (doorbell)

- **Sources:**
  - The kTransformers SOSP'25 paper:
    https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
  - The sergqwer/qwen4exp-5090 write-up, section 9 (**forum**: a fork
    README):
    https://github.com/sergqwer/qwen4exp-5090/blob/HEAD/results/optim-2026-09-21/README.md
  - FlashML-org/FreeToken, `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`:
    https://github.com/FlashML-org/FreeToken
- **What it does:**
  - The GPU stream writes a "ready" flag (`cuStreamWriteValue64`). A
    persistent CPU worker pool computes the experts. The stream then waits
    on a "done" value (`cuStreamWaitValue64`).
  - kTransformers wraps the hand-off in `cudaLaunchHostFunc` instead. Either
    way the whole decode step fits in one CUDA graph, with no per-layer
    stream synchronisation.
- **Claimed gain:**
  - kTransformers: "up to 1.23×" on its own engine.
  - sergqwer, on this model with an RTX 5090 and a 9950X3D: about 200 us per
    host expert layer, of which about 130-170 us is neither memory nor
    arithmetic. He estimates the doorbell at "~5 milliseconds per token".
    That is **unverified**: it was not built, because stream memops fail on
    Windows WDDM.
- **Applies to koishi: yes.** Linux supports stream memops.
  - koishi's nsys showed 48 graph launches of 114 us (5.4 ms) and 10.5 ms of
    host time in CUDA calls a step.
  - The local queued copies (`exp/verifygpu`) already cut synchronisations
    from 404 to 78 a step for about +2%. The ceiling left is therefore
    roughly the rest of the 10.5 ms, about 5-7% of a step (my arithmetic).
  - Note that the Strata N9 item (one graph per verify window) needed every
    expert in VRAM. This one does not.
- **Phase:** generation.
- **Effort:** port / research. It changes how ggml-sched runs a CPU split
  that sits between two GPU splits.
- **Priority:** medium.

### R6. The MTP draft's `hnorm`: one norm for each stream, or one over all four

- **Sources:** Strata commit `2312c83` (2026-10-03), `--mtp-hnorm stream`.
  I read the commit message and `docs/DETAILS.md` line 274 in the local
  clone.
- **What it says:** Strata applies the draft layer's `pre_fc_norm_hidden` as
  "one over all four (vLLM's qwen4_exp MTP, the default)". It offers "one
  RMS per hyper-connection stream (llama.cpp's qwen4exp MTP graph)" as an
  option.
- **Checked in the pin:** `src/models/qwen4exp.cpp` reshapes `h` to
  `[n_embd, hc, n_tokens]` before `build_norm(... nextn.hnorm ...)`, so it
  computes the RMS for each 2560-wide stream. The `nextn.hnorm.weight` in
  `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` is F32 of 10240 = 4 x 2560.
- **Claimed gain:** none published. Strata gives no acceptance figure for
  either form.
- **Applies to koishi: yes.** If the vLLM form is the reference, the
  llama.cpp draft may accept less than it could. If it is not, nothing
  changes.
- **Phase:** generation (draft acceptance).
- **Effort:** a small graph change (RMS over the 10240 vector, then
  reshape), and an acceptance A/B over the 16 prompts of `prompts-16.json`.
  First read vLLM's `qwen4_exp` MTP code or the HF reference to see which
  form is correct.
- **Priority:** medium: cheap, and each point of acceptance counts.

### R7. Limit the experts a verify step reads (lossy)

- **Sources:**
  - AcceptMoE, read in full by the speculative-decoding agent:
    https://arxiv.org/pdf/2608.02989
  - MoE-Spec: https://arxiv.org/html/2602.16052v1
  - MoESD, for the cost formula: https://arxiv.org/html/2505.19645v4
  - ik_llama.cpp SER, PR #239 (merged 2025-03-02):
    https://github.com/ikawrakow/ik_llama.cpp/pull/239
- **What it does:**
  - A verify step of 4 tokens reads the union of the experts all 4 tokens
    route to. MoESD's formula gives about 38.8 distinct experts per layer at
    E = 512, K = 10 and 4 tokens, for uniform routing, against 10 for one
    token. That figure is the agent's arithmetic. Real routing overlaps
    more, so measure it.
  - AcceptMoE and MoE-Spec cap that union during verify. AcceptMoE weights
    the router scores by the chance each draft position is accepted, keeps
    the root token's own top-k, and sizes the set from the entropy of the
    demand (about 23-34 experts per layer).
  - SER drops the lowest-weight experts per token. ikawrakow now advises
    `--override-kv <arch>.expert_used_count=int:N` instead. llama.cpp
    mainline accepts that flag too.
- **Claimed gain:**
  - AcceptMoE: "1.290× the throughput of this baseline [EAGLE-3] with all
    expert weights in GPU memory, and 2.06× under physical expert
    offloading". Mean accuracy "0.27 percentage points lower". Measured on
    an RTX PRO 6000 Blackwell and an RTX 5090 with SGLang; Qwen3-30B-A3B,
    Qwen3-Coder-30B and GPT-OSS-120B.
  - MoE-Spec: "10–30% higher throughput than … EAGLE-3", on an A100.
  - SER: "5-7%" TG at Kmin = 4 and t = 0.2, on DeepSeek-Lite with an RTX
    4080 and a 7950X.
  - Keep-7 experts (**forum**, ik discussion #2106, as the engines agent
    reported it; I did not open it): +10% generation at 92.4% top-1
    agreement.
- **Applies to koishi: yes, but lossy.** These target the exact cost on
  koishi: CPU bytes per verify step. They change the output, so they need
  the KLD and greedy gates. With MTP a capped verify accepts tokens from a
  changed model.
- **Phase:** generation (SER and expert_used_count: both).
- **Effort:** measure the union first, by logging the selected experts per
  verify step. `expert_used_count` is a flag. A verify-time cap is a port in
  the MoE graph.
- **Priority:** medium. Measure the union first. Run the flag only if the
  quality gate allows it.

### R8. 2 MiB pages for the expert mapping

- **Sources:**
  - The kernel tmpfs doc: https://docs.kernel.org/6.8/filesystems/tmpfs.html
  - llama.cpp PR #22022, `MADV_HUGEPAGE` on the mmap (open; "Neutral on an
    unloaded machine"): https://github.com/ggml-org/llama.cpp/pull/22022
  - llama.cpp PR #12552, mmap from hugetlbfs (open):
    https://github.com/ggml-org/llama.cpp/pull/12552
  - ik_llama.cpp PR #278 (**forum**-level numbers): "~0.5-1% in TG" on a
    Ryzen 7950X, about 20% slower on a 5975WX.
- **What I checked on koishi:**
  - `/boot/config-6.12.90+deb13.1-amd64` has `# CONFIG_READ_ONLY_THP_FOR_FS is not set`.
  - THP is `[always]`, and shmem THP is `[never]`.
  - The model is on ext4.
  - So `MADV_HUGEPAGE` or `MADV_COLLAPSE` on the GGUF mapping cannot give
    huge pages today. This matches the `FilePmdMapped 0 kB` that
    `docs/GPU-FOLLOWUPS.md` records.
- **Ways to get them:**
  - **(a)** Copy the GGUF to a tmpfs mounted with `huge=always`. Its
    `mpol=` option can also set the NUMA placement.
  - **(b)** XFS, or ext4 on kernel 6.16 or later, has large page-cache
    folios. The agent reports (from `mm/filemap.c` in 6.12) that a
    `VM_HUGEPAGE` mapping then reads ahead in PMD-size folios. I did not
    read that code.
  - **(c)** hugetlbfs with #12552.
- **Applies to koishi:** perhaps. A 4-token verify reads about 480 experts of
  5 MB each, so about 1280 4-KiB pages per expert. No source measured
  Skylake.
  - A tmpfs copy uses RAM in place of the page cache. All four backends
    must map the same copy, or RAM use doubles.
  - The file is 197 GB with the 54 GB per-layer embedding table. The table
    can stay on disk only if it is not in the tmpfs copy, so a tmpfs copy
    may need a re-split of the GGUF.
- **Phase:** both.
- **Effort:** ops (mount and copy). No llama.cpp change for (a) and (b).
- **Priority:** medium-low. Test (a) on one backend first.

### R9. Gumbel-coupled MTP drafts for sampled requests

- **Sources:** Strata PR #1281, commits `6381df3` and `2088bd4`:
  https://github.com/Niko1221/Strata/pull/1281
- **What it does:** the draft samples with the target's own sampler chain
  and seed, and both pick argmax p/E with E ~ Exp(1), keyed by (seed,
  counter, token id). Draft and target then agree on shared tokens even
  when their top-k or top-p lists differ. The verify rule stays an exact
  match.
- **Claimed gain:**
  - Ryzen AI Max+ 395 iGPU, temperature 1.0: accepted 52.8% to 59.9%,
    output 41.1 to 44.9 tok/s (+9%).
  - RTX 3060: acceptance 62.1% to 68.0%, but decode within noise.
- **Applies to koishi:** for requests with temperature > 0 only. The
  `--backend-sampling` tests ran at temperature 1.0, so such requests exist.
  Greedy requests do not change.
- **Phase:** generation.
- **Effort:** port into the llama.cpp sampler and the MTP draft sampling.
- **Priority:** medium-low.

### R10. A pinned checkpoint at the shared prefix end and the last turn boundary

- **Sources:**
  - Strata commits `13d14a4` and `feccbfe` ("strata_prefix", engine
    `pin=N`).
  - Strata PR #734, `STRATA_CACHE_MESSAGE_BOUNDARY=1`, and
    `docs/MESSAGE_BOUNDARY_CACHE.md`:
    https://github.com/Niko1221/Strata/pull/734
- **What it does:** it saves a never-evicted recurrent-state checkpoint at
  the shared prefix end, and one at the previous turn boundary for prefixes
  of 8,192 tokens or more. Without them, a request that changes the last
  message reads again from the last periodic checkpoint.
- **Claimed gain:**
  - Prefix pin, P100: a 64K document, questions 2-5, from 48.6 s to
    0.35-0.53 s.
  - Message boundary, 2x RTX 3080: first edit from 17.236 s to 11.527 s.
    The doc itself says this "is a COMBINED boundary-policy and chunk-size
    comparison, not the isolated benefit".
- **Applies to koishi:** perhaps. The router already keeps layered prompt
  caches. Check whether llama-server's context checkpoints for a hybrid
  model land at the turn boundaries the router reuses, or only near the
  prompt end.
- **Phase:** prefill.
- **Effort:** a check, then a port into the server's checkpoint placement.
- **Priority:** medium-low, until the check shows re-reads.

### R11. #29308 (U1) again, with new dual Cascade Lake data

- **Source:** llama.cpp PR #29308 (open):
  https://github.com/ggml-org/llama.cpp/pull/29308
- **What it does:** each `src1` row of `mul_mat` and `mul_mat_id` is
  converted by one thread. This is U1 in `docs/PERF.md`, which was dropped on
  the EPYC box: "no gain past drift".
- **New evidence:** the PR's numbers are on "2x Xeon Gold 6262 (Cascade
  Lake, 24 cores per socket)", OLMoE 1B-7B Q5_K_M: 30.08 / 30.30 to
  38.01 / 38.33. One socket (`numactl -N 0 -m 0`): 28.43 / 29.67 to
  32.91 / 32.91. That is the closest hardware to koishi.
- **Applies to koishi:** perhaps. It does not apply to the pin (conflict in
  `ggml-cpu.c:1333`), and it touches the same code as the local
  `mul_mat_id` patches.
- **Phase:** generation.
- **Effort:** rebase.
- **Priority:** low-medium.

### R12. Expert deferral (lossy)

- **Sources:**
  - The kTransformers SOSP'25 paper (link in R5).
  - https://www.lmsys.org/blog/2025-10-22-KTransformers
- **What it does:** the low-score experts of layer k are added at layer k+2,
  not k+1. The CPU computes them while the GPU runs the next layer's
  attention.
- **Claimed gain:** with 3 of 8 experts deferred, "reduced single-layer
  execution time by 26%, and increased end-to-end decoding throughput by
  33%"; average accuracy drop "only 0.5%". Measured on 2x Xeon Platinum
  8452Y (AMX) with an A100 40GB or an RTX 4080, on DeepSeek-V3, V2.5 and
  Qwen2-57B.
- **Applies to koishi: partly.** The ceiling is the GPU share of a step,
  about 23 ms of 143. It changes the model's output, and the MTP head was
  trained without it. It is untested on GDN or sparse attention.
- **Phase:** generation.
- **Effort:** research: a graph change, and true CPU/GPU concurrency in the
  scheduler.
- **Priority:** low-medium.

### R13. GDN CUDA kernels

- **Sources:**
  - #30087, two GDN state columns per warp (open, 2026-10-07; **applies to
    the pin**): https://github.com/ggml-org/llama.cpp/pull/30087. Claim on
    an RTX 4090: pp512 +5.5%, kernel "487 to 304 us per instance".
  - #29187, fused GDN alpha/beta projections, handles Q8_0 (open; **applies
    to the pin**): https://github.com/ggml-org/llama.cpp/pull/29187. On
    koishi `ssm_alpha` and `ssm_beta` are Q8_0 [2560, 48].
  - #29353, chunked GDN kernel (open; does not apply, conflict in
    `mma.cuh`): https://github.com/ggml-org/llama.cpp/pull/29353. Claim:
    RTX 3090, Qwen3.6-35B-A3B, 2048 tokens, 4,435.95 to 5,489.94 (+23.76%),
    full offload.
  - #21897, concurrent CUDA streams for linear attention (open; does not
    apply): https://github.com/ggml-org/llama.cpp/pull/21897. Claim: 5090,
    gemma4, 1.07x.
- **Applies to koishi:** yes, the GDN layers run on the A4000 (sm_86). But
  the card holds about 17% of a prompt and 23% of a verify step, so the gain
  is a fraction of that.
- **Phase:** both, mostly prefill.
- **Effort:** cherry-pick (#30087, #29187); rebase (#29353).
- **Priority:** low. #30087 is the cheapest to try.

### R14. CUDA graph fixes merged or opened after the pin

- **Sources:**
  - #29986, makes the CUDA `alloc_deps` check independent of the batch
    (merged 2026-10-05 13:44 UTC, after the pin; **applies to the pin**):
    https://github.com/ggml-org/llama.cpp/pull/29986. It fixes a re-reserve
    per ubatch size caused by #29184, which is in the pin. Issue #29980
    reports prompt processing about 2x slower on Qwen3.6-35B-A3B on 3x RTX
    PRO 6000.
  - #29768, recapture at once after a stable replay (open; **applies to the
    pin**): https://github.com/ggml-org/llama.cpp/pull/29768. Claim:
    +0.3 to +1.3% on a 5090.
- **Applies to koishi:** #29986 only where a routed `MUL_MAT_ID` runs on
  CUDA. Check with `GGML_SCHED_DEBUG_REALLOC=1` whether the gpu backend
  re-reserves during a request.
- **Phase:** both.
- **Effort:** cherry-pick.
- **Priority:** low, but cheap.

### R15. The shared expert on the card overlaps the routed experts on the CPU

- **Sources:**
  - ik_llama.cpp PR #1191: https://github.com/ikawrakow/ik_llama.cpp/pull/1191.
    Claim: "4-5% better performance for GLM-4.5-AIR with all routed experts
    left on the CPU".
  - Strata commit `8b5dd7a` (#783), no separate figure.
- **Applies to koishi:** the idea does, because ggml-sched runs the CPU and
  GPU splits one after the other. But on koishi the shared expert is one
  640-wide Q8_0 FFN (about 5 MB), so the card's share to hide is small.
- **Phase:** generation.
- **Effort:** port.
- **Priority:** low.

### R16. Repacked experts written into the file, shared through mmap

- **Sources:**
  - ik_llama.cpp PR #272 / #274 (`llama-quantize --repack --repack-pattern exps`):
    https://github.com/ikawrakow/ik_llama.cpp/pull/272 and
    https://github.com/ikawrakow/ik_llama.cpp/pull/274
  - ktransformers PR #2190, a `MAP_SHARED` file arena for repacked experts
    (open): https://github.com/kvcache-ai/ktransformers/pull/2190. It claims
    memory, not speed: anonymous RSS from about 105 GiB to 2.4 GiB, decode
    13.1 against 13.0 tok/s.
- **What it does:** it removes the per-process private copy that rules out
  `-rtr` on this host.
- **Applies to koishi: low.**
  - `docs/PERF.md` estimated a repack at about 4-5% of a prompt, with no
    generate gain.
  - The engines agent read that ik's `iqk_config.h` enables its AVX-512
    paths only with VNNI. On koishi ik would run its AVX2 kernels.
- **Phase:** prefill.
- **Effort:** port (a new ggml type and its kernels).
- **Priority:** low.

### R17. Fewer barriers in a CPU MoE layer, and SwiGLU fused with the Q8_0 quantize

- **Sources:**
  - Strata PR #949, `--pool-tasks N`:
    https://github.com/Niko1221/Strata/pull/949. On an i5-14600K, the
    median CPU-layer call fell 4.41-21.00%, with no claimed end-to-end gain.
  - shyringo/qwen3.8-flash-next-in-c, README (**self-reported**): one
    parallel region for the ten routed and the shared expert.
    https://github.com/shyringo/qwen3.8-flash-next-in-c
  - fastllm commit `94461e1b4`: SwiGLU and the Q8_0 quantize in one pass,
    "单轮解码提升 3.6%". A later profile was "基本持平" (about flat).
- **Applies to koishi:** perhaps. sergqwer found the CPU side barrier-bound
  on a 16-core desktop with an expert cache. koishi reads its experts at 84
  of 128 GB/s, so part of a step may be barriers. An op profile of barrier
  wait time would show it.
- **Phase:** generation.
- **Effort:** port.
- **Priority:** low.

### R18. MTP drafter: `--mtp-q4`, skip the K/V of rejected rows

- **Sources:** Strata commits `e35a061` (`--mtp-q4`) and `3fd0460` (the
  drafter's catch-up skips the K/V of rejected rows).
- **Claimed gain:** `--mtp-q4` on an RTX 5070: "draft ms per window 2.55 ->
  2.23 (q4)", with tokens per step "2.64 -> 2.63". The catch-up skip has no
  figure.
- **Applies to koishi:** a Q4_0 draft layer would cut draft time, which is
  small next to a 143 ms verify. `models/MTP/` already holds
  `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`; its origin is not checked
  here. Check whether
  llama.cpp's MTP catch-up repeats work for rejected rows.
- **Phase:** generation.
- **Effort:** a flag (the Q4_K_M draft file) or research.
- **Priority:** low.

### R19. A cost-aware prompt lookup chained after the MTP drafts

- **Sources:** Strata commits `2312c83`, `5e0c43b`, `48d2e6c` (#1252), and
  open PR #1316: https://github.com/Niko1221/Strata/pull/1316
- **What it does:** a suffix matcher appends up to K tokens after the MTP
  drafts. A policy keeps a chain only if its expected tokens per ms beats
  the cost of the longer window.
- **Claimed gain:** before the policy, "+6.4% tokens per step for +7% window
  time", so no net gain. No final figure.
- **Applies to koishi:** low. `--spec-type ngram-mod,draft-mtp` lost 1.4 to
  4.8% on the CPU box, and the expert kernel is tuned for up to 4 tokens.
- **Phase:** generation.
- **Effort:** port.
- **Priority:** low.

### R20. fastllm as a second engine

- **Source:** https://github.com/ztxz16/fastllm,
  `docs/qwen3.8-flash-next/README_EN.md`
- **What it is:** native qwen4exp support from GGUF, with MTP, a NUMA expert
  backend, a CPU/GPU split, and the per-layer embedding table on disk.
- **Claimed gain:** on an EPYC 7452 with 2x 22 GiB 2080 Ti, NVFP4: prefill
  919.43, decode 22.36, MTP=3 30.08 tok/s. That is a different quant and
  machine.
- **Applies to koishi:** unverified. Its AVX-512 path on Skylake without
  VNNI is unknown. Its NUMA backend does a "destructive NUMA repacking": a
  private copy for each process.
- **Phase:** both.
- **Effort:** a test.
- **Priority:** low.

## Checked and not applicable

| item | source | why not |
|---|---|---|
| #28118, speculative checkpoints kept on the device | https://github.com/ggml-org/llama.cpp/pull/28118 | The pin rolls `qwen4exp` back with in-graph recurrent snapshots (`llm_arch_supports_rs_rollback` lists `LLM_ARCH_QWEN4EXP`, and `n_rs_seq` comes from the speculative params). The host checkpoint runs only when the draft is longer than `n_rs_seq`. |
| ik #2412 / #1669, per-step GDN state for MTP | https://github.com/ikawrakow/ik_llama.cpp/pull/2412 | The same reason. The pin already keeps the GDN state of each draft position. |
| #28699, incremental pooled-key cache for the QSA indexer | https://github.com/ggml-org/llama.cpp/pull/28699 | The pin's k-pool input already re-pools only the blocks a ubatch completes (`new_pool_idxs`, `new_pool_rep`). |
| #28671, radix-select TOP_K for CCCL below 3.4.3 | https://github.com/ggml-org/llama.cpp/pull/28671 | Applicable, and applied as core/0047. The pin's gate is CCCL >= 3.4.3 and koishi has 3.3.4, so the CUDA top-k fell into argsort + copy and sorted every column; the row's first reading compared against the PR's description (3.2). core/0047 compiles the radix-select for the CUB build and uses it for ncols >= 8192; test-backend-ops perf -o TOP_K median 1.91x on wide shapes, correctness 525/525. |
| ik #2373 and #28875, a thin F32 GEMM for `hc_*_inject` | https://github.com/ikawrakow/ik_llama.cpp/pull/2373 | In this GGUF `hc_attn_inject` and `hc_ffn_inject` are Q8_0 [10240, 4], not F32. |
| #30036, VNNI repack kernels for Q8_0 and Q4_0 | https://github.com/ggml-org/llama.cpp/pull/30036 | Every kernel needs AVX-VNNI or AVX512-VNNI. Its claim is on a 9950X: pp2048 53.84 to 209.10. |
| ik_llama.cpp AVX-512 kernels in general | `ggml/src/iqk/iqk_config.h` (read by the engines agent) | `HAVE_FANCY_SIMD` needs AVX512-VNNI, so Skylake-SP takes the AVX2 path. |
| DFlash block drafter (#22105, merged 2026-06-28, in the pin) | https://github.com/ggml-org/llama.cpp/pull/22105 | Its drafter for this model exists for vLLM with NVFP4 only. The card says "Code is a wash and chat is slower". A wider block also makes the verify read more experts. |
| GPU expert prefetch and prediction (ProMoE, Fate, SP-MoE, MoE-SpeQ, HOBBIT, AdapMoE, Speculating Experts, QwFNfer routing prediction) | listed in the speculative-decoding agent's report | Each moves experts to the GPU at generate. On gen3 x8 one Q8_0 expert takes about 0.75 ms to upload, against about 0.06 ms for the CPU to read it (my arithmetic). They also suit an expert cache, and N1 got 0% hits. |
| EcoSpec, EVICT, tree drafts | https://arxiv.org/html/2607.12696v1, https://arxiv.org/pdf/2605.00342 | They need tree verify, which needs one GDN state per branch, and a tree grows the expert union. Every serving stack found uses a top-1 chain for this family. |
| Pre-gated MoE, EAGLE-3 | https://arxiv.org/pdf/2308.12066, https://arxiv.org/abs/2503.01840 | They need training. |
| Strata `STRATA_GDN_CHUNKED` (PR #1372) | https://github.com/Niko1221/Strata/pull/1372 | It needs at least 128 SMs. The A4000 has 48. |
| FlashMLA, DeepGEMM, FlashQLA | https://github.com/deepseek-ai/FlashMLA | They need SM90 or later. The A4000 is sm_86. |
| Strata `--expert-cache-per-layer`, AVX-VNNI rows, IQ kernels, file tier | Strata `818ec1c`, `b211ba1` | They need mixed quants, VNNI, or a RAM shortfall. |
| Strata "experimental speed projection" | Strata `docs/DETAILS.md`, line 1398 | It is a refusal-direction control vector. Its own doc says it costs 0.2-0.4% a token. It is not a speed-up. |
| Strata `STRATA_MMVQ_IL` | Strata `ffac97f` | Its commit says that the Q8_0 parts are "left out". |
| MoE-Lightning, MoE-Gen, SpecMoEOff, SpecMoE, LayerScope | listed in the speculative-decoding agent's report | They target throughput at large batch sizes. |
| #26610, RPC `-sm tensor` across sockets | https://github.com/ggml-org/llama.cpp/pull/26610 | It is unclear whether it mixes with a local CUDA device. Research only. |

## Sources checked with nothing new

- PowerInfer: no code since 2025, and it depends on ReLU sparsity.
- mistral.rs: only PR #2385, which matches the R-not-applicable GDN rollback
  row.
- ktransformers `Qwen3-Next.md`: no numbers. kt-kernel has no Q8_0 backend
  and no qwen4exp support.
- unslothai/llama.cpp: its `moe-cache-auto` branch is the known expert cache.
- r/LocalLLaMA: WebSearch refused reddit.com, so no Reddit post was read.

## Notes on method

- Four research agents read the sources: other engines, Strata, papers on
  speculative decoding and hybrid MoE, and kernels and NUMA. I spot-checked
  their key claims:
  - Strata `docs/DETAILS.md` (CPU share table) and commit `2312c83`, in the
    local clone at `e8ca9af`.
  - ik #2396 (st3fk3's comment), kt #2088, #27986 and #28872 through `gh`.
  - The sergqwer README through WebFetch.
  - The pin's code for R1, R6 and the not-applicable rows.
  - Tensor types in the Q8_0 and MTP GGUF headers.
  - The kernel config and the CUB version on koishi.
- I did not open the following myself. Their figures come from the agents'
  reading:
  - the arXiv papers (AcceptMoE, MoE-Spec, MoESD, EcoSpec, EVICT and the
    prefetch family);
  - fastllm's commits;
  - ik discussion #2106;
  - FreeToken's source.
- `git apply --check` was run on a temporary worktree of `8e1642198` for
  #28872, #29986, #29187, #30087, #29353, #29768, #21897, #27986, #30036 and
  #29308. The worktree was then removed.
- Strata rewrote its main history on 2026-10-06 (#1276). The old hashes map
  as `6f32ec0` → `a1641e9` and `1735d64` → `1cbcacb` (equal trees). HEAD is
  now `e8ca9af` (2026-10-07), and the latest release is v0.1.40.2.
