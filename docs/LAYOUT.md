# What the defaults are based on

The numbers in the launch scripts and in `bin/router.py` are measured, not
guessed. They are also not laws. This page gives the measurement behind each
one, in three groups: properties of llama.cpp, properties of Linux, and figures
that belong to one machine. That machine has two Xeon sockets and an A4000.

## llama.cpp and the model

**Generated tokens are not prefilled again.** The server wrote their KV while
it decoded them. A 600-token reply, fed back as history, cost **24** prompt
tokens rather than 600.

Only injected content is prefilled: tool results, file reads and pasted code.
Each piece is prefilled once. What limits usable context is therefore cache
survival, not throughput:

| what breaks the prefix cache | cost |
|---|---|
| compaction or truncating history | reprocess from the divergence |
| a system prompt that changes (a timestamp will do) | reprocess, every turn |
| more concurrent conversations than slots | LRU evicts a slot |
| a conversation moved to another backend | reprocess everything |

Pinning prevents the last two. That table is why this router exists.

**A slot that reads a long prompt stops every other slot on its instance.**
Those slots run at 0.02 to 0.06 tokens a second, for the 55 to 65 seconds that
a 2040-token chunk takes. A second slot buys a queue, not a second prefill. Two
instances are two schedulers. That is why each backend runs `--parallel 1`, and
why a socket gets its own backend rather than a second slot.

**Every backend must offer the same context size.** A conversation that
outgrows one backend can never move back to it. Reading a full context again is
one unavoidable pass: 150k tokens at 26.6 tokens a second takes 94 minutes.

**A draft model does not change output quality.** The target model verifies
every drafted token, so a smaller draft can only change the acceptance rate.
Measured over three runs each: Q8 reached 18.64 tokens a second, Q4 reached
18.33, and the Q4 range contained the Q8 range entirely. KV quantisation is a
separate question, and stays at f16.

**`--agent` limits CORS to localhost**, unless `--cors-origins` is set
(`common/arg.cpp:935`). This affects browser origins only. API clients send no
`Origin` header.

## Linux, on any two-socket machine

**Linux keeps one page cache per file. It places those pages on whichever
node faults them first.** Two servers that map one weights file therefore share
one cache, and one socket reads every expert over the interconnect.

Memory policy cannot fix this, because the second mapper allocates nothing. A
hard link cannot fix it either, and neither can a reflink: both share the same
pages. Only a separate copy per node works:

```
one shared file      gpu N0=62.8GiB(83%)  cpu N0=63.3GiB(82%)   <- 82% remote
a copy per node      gpu N0=76.0GiB(100%) cpu N1=76.4GiB(99%)
```

Measured at 2k prompts and ctx 150000, a copy per node gained 14% on the GPU
alone, 35% across three CPU streams, and **25% for the whole machine**.

One instance spread across both nodes loses about 30% of its generation speed.
The node distance is 21, against 10 for local memory. That is why each instance
binds to one socket.

**A stale page cache undoes all of this, and says nothing.** Linux decides
placement at the first fault and never revisits it. Priming a file that is
already in the cache therefore moves nothing.

Two measurement runs were lost to this. Node 1 held about 40 GiB of stale
cache. The prime evicted what it had just loaded. The server then sat in D
state at 100 MB/s. The symptom to watch for is a prime that finishes in 31
seconds instead of 247.

`start-all.sh` avoids this in two ways. It drops the cache for every model file
before it starts anything. It then starts the backends one at a time, because
each one reads about 125 GiB, and together they only queue on the disk.

To check a binding, do not read `/proc/<pid>/status`. It reports the main
thread only, and on an instance with a GPU it reads `0-71`, because the CUDA
driver widens its own threads. Read every task instead:

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

**`CTX=150000`** fits 16376 MiB of VRAM at f16 KV. That model costs about 36.5
KiB a token, so multiply your own card by that rate. 160000 also loads, at
15701 MiB, but it leaves only 675 MiB. That is not enough for the compute
buffers at full context.

**`N_CPU_MOE=48`** puts attention and KV on the card, and the experts in RAM.
It gains about 2.7 times on generation, and nothing on prefill. VRAM is the
limit here, not RAM.

**One socket per instance.** Q8 is 175 GiB resident, against a node of 187 GiB.
`--lazy-mode auto` leaves the 50.66 GiB `per_layer_token_embd` tensor on disk,
which gives a working set of about 125 GiB.

**The disk changes startup, not speed.** Three streams measured 12.15 tokens a
second on SATA, against 12.24 on NVMe. A 2048-token prompt plus 192 generated
tokens reads only 6.7 MiB. NVMe pays for itself in priming instead: 85 seconds
against 247.

**`PARK_BUDGET_GB` is 256 and `BLOCK_BUDGET_GB` is 64.** Both are sized to a
large SATA disk, with the saved openings moved to NVMe. Set either one too low
and the cost is a full re-read.
