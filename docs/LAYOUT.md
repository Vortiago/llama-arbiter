# What the defaults are based on

The numbers in the launch scripts and in `bin/router.py` are not arbitrary, and
they are not laws either. Below is what each was measured against, split by
whether it is a property of llama.cpp, of Linux, or only of the box it was
measured on — two Xeon sockets and an A4000.

## llama.cpp and the model

**Generated tokens are not reprocessed.** Their KV was written while decoding,
so a 600-token reply fed back as history cost **24** prompt tokens, not 600.
Only *injected* content — tool results, file reads, pasted code — is prefilled,
and only once. What limits usable context is therefore cache survival, not
throughput:

| what breaks the prefix cache | cost |
|---|---|
| compaction or truncating history | reprocess from the divergence |
| a system prompt that changes (a timestamp will do) | reprocess, every turn |
| more concurrent conversations than slots | LRU evicts a slot |
| a conversation moved to another backend | reprocess everything |

The last two are what pinning prevents. That table is why this router exists.

**A slot reading a long prompt stops every other slot on its instance** —
0.02 to 0.06 tokens/s for the 55-65 s a 2040-token chunk takes. A second slot
buys a queue, not a second prefill; two instances are two schedulers. Hence
`--parallel 1` and a backend per socket rather than one backend with two slots.

**Every backend has to offer the same context.** A conversation that outgrew
one could never move back to it, and re-reading a full context is a single
unavoidable pass: 150k tokens at 26.6 tok/s is 94 minutes.

**A draft model is not a quality lever.** The target verifies every drafted
token, so a smaller draft can only change acceptance, never output: Q8 18.64
and Q4 18.33 tokens/s over three runs each, Q4's range containing Q8's
entirely. KV quantisation is a different matter, and stays f16.

**`--agent` limits CORS to localhost** unless `--cors-origins` is set
(`common/arg.cpp:935`). Browser origins only; API clients send no `Origin`.

## Linux, on any two-socket machine

**One page cache per file, placed on whichever node faults it first.** Two
servers mapping one weights file means one socket reads every expert over the
interconnect. Memory policy cannot fix it — the second mapper allocates nothing
— and a hard link or a reflink shares the same pages. A separate copy per node
is the only thing that works:

```
one shared file      gpu N0=62.8GiB(83%)  cpu N0=63.3GiB(82%)   <- 82% remote
a copy per node      gpu N0=76.0GiB(100%) cpu N1=76.4GiB(99%)
```

At 2k prompts and ctx 150000 that was +14% on the GPU alone, +35% across three
CPU streams, **+25% for the box**. Spilling one instance across both nodes
costs about 30% of generation (node distance 21 against 10), which is why each
is bound to one socket.

**Stale page cache silently undoes it.** Placement is decided at first fault
and never revisited, so priming a file already in cache moves nothing. Two runs
were lost to this: node 1 held ~40 GiB of stale cache, the prime evicted what
it had just loaded, and the server sat in D state at 100 MB/s. A prime that
finishes in 31 s instead of 247 is the tell. `start-all.sh` drops the cache for
every model file first, and starts backends one at a time — each reads ~125 GiB
and together they only queue on the disk.

Checking a binding takes more than `/proc/<pid>/status`, which reports the main
thread only. On a GPU instance that reads `0-71`, because the CUDA driver
widens its own threads:

```sh
P=$(ss -ltnp | awk '/:8080 /{match($0,/pid=[0-9]+/); print substr($0,RSTART+4,RLENGTH-4); exit}')
for t in /proc/$P/task/*; do awk '/Cpus_allowed_list/{print $2}' $t/status; done | sort | uniq -c
#  95 0-17,36-53     <- node 0, correct
#   2 0-71           <- CUDA driver threads, harmless

awk '/file/ && /N[01]=/ { for (i=1;i<=NF;i++) {
       if ($i ~ /^N0=/) { split($i,a,"="); n0 += a[2] }
       if ($i ~ /^N1=/) { split($i,a,"="); n1 += a[2] } } }
     END { t = n0+n1; printf "N0=%.1fGiB(%.0f%%) N1=%.1fGiB(%.0f%%)\n",
           n0*4/1048576, 100*n0/t, n1*4/1048576, 100*n1/t }' /proc/$P/numa_maps
```

## Sized to one machine, so yours will differ

- **`CTX=150000`** is what fits 16376 MiB of VRAM at f16 KV and about 36.5 KiB
  a token. 160000 loads too, at 15701 MiB, but leaves 675 MiB — not enough for
  the compute buffers at full context. Size your own card by that rate.
- **`N_CPU_MOE=48`** puts attention and KV on the card and the experts in RAM:
  ~2.7x on generation, nothing on prefill. VRAM is the limit, not RAM.
- **One socket per instance** because Q8 is 175 GiB resident against a 187 GiB
  node, with `--lazy-mode auto` leaving the 50.66 GiB `per_layer_token_embd` on
  disk for a working set of ~125 GiB.
- **The disk changes startup, not speed.** Three streams measured 12.15 tok/s
  on SATA against 12.24 on NVMe: a 2048-token prompt plus 192 generated tokens
  reads only 6.7 MiB. NVMe pays for priming — 85 s instead of 247.
- **`PARK_BUDGET_GB` and `BLOCK_BUDGET_GB`**, 256 and 64, are sized to a large
  SATA disk with the openings moved to NVMe. The cost of setting them too low
  is a full re-read.
