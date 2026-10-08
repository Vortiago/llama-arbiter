#!/usr/bin/env bash
# Copy to config.local.sh and edit. That file is not tracked, so no local
# path reaches the repository.
#
# bin/common.sh reads it first, so every launch script, the prefill harness
# and the experiment runners see it. Export any name the router reads: the
# router is a separate process.

# Where the weights are. Default: models/ in the checkout, which suits one
# socket. Two sockets want one copy each on the fastest disk, as separate
# files. A hard link or a reflink shares the page cache, which the second
# copy exists to avoid. Exported, because the prefill harness is a separate
# process.
# export MODELS=/mnt/nvme/models
# export MODELS2=/mnt/nvme/models-node1

# Where llama-server is. Default: the patched llama.cpp that tools/get-llama.sh
# builds under the checkout. patches/README.md says what the patches do and why.
# SERVER_MTP=/opt/llama.cpp/build/bin/llama-server
# koishi builds the pin in a second worktree, llama.cpp-pin, so the old fork
# kept running while it built: DIR=$PWD/llama.cpp-pin LLAMA_REF= CUDA=1
# CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=86 tools/get-llama.sh. Its
# SERVER_MTP points at the newest release worktree (llama.cpp-rel3), built
# the same way from the current patches/; the one before it (llama.cpp-rel2)
# stays for a rollback and as the baseline of open experiments. llama.cpp-pin stays
# at the commit the experiments in docs/GPU-FOLLOWUPS.md measure against.

# A different model. MODELS only moves the directory, so a model with another
# name needs these too. DRAFT is the MTP draft model. Default: ggml-org's
# mtp-Qwen3.8-Flash-Next-Q8_0.gguf, which carries its own token_embd. The
# unsloth "shared" draft does not load on llama.cpp master. With no draft,
# leave the --model-draft flags out of ARGS.
# MODEL_Q8=$MODELS/my-model-00001-of-00004.gguf
# MODEL2_Q8=$MODELS2/my-model-00001-of-00004.gguf
# DRAFT=$MODELS/my-draft.gguf
# DRAFT2=$MODELS2/my-draft.gguf
# MMPROJ=$MODELS/mmproj-F16.gguf      # vision. VISION=0 starts without it.

# Threads per backend. Default: the physical cores on the node it is bound
# to, which is usually right.
# THREADS=18

# One NUMA node: nothing to bind away from, so no numactl and no --numa.
# export NUMA_MODE=off

# A podman container for the tool calls --agent makes, and the MCP servers it
# reaches, for a launch script that calls tools_args. Unset: no tools sandbox.
# TOOLS_CONTAINER=llama-tools
# MCP_CONFIG=/srv/llama-tools/mcp-servers.json

# Context per slot. Every backend must use the same number: a conversation
# that outgrew one can never move back to it. 150000 needs about 15.3 GiB of
# VRAM at f16 KV, about 36.5 KiB a token. Size it to your own card;
# docs/LAYOUT.md has the measurements. NGL and N_CPU_MOE divide the model
# between card and RAM. A lower N_CPU_MOE keeps more experts on the card and
# needs the VRAM to hold them.
# export CTX=150000
# export N_CPU_MOE=48
# export NGL=99

# Where a run writes logs, pids, saved slots and cache events. Default: the
# checkout's run/. Exported, because the router reads it too. A router that
# looks anywhere but --slot-save-path finds no caches.
# export RUN=/var/lib/llama-arbiter

# More than one backend needs two tables that agree. BACKENDS says what
# start-all.sh launches, ROUTER_BACKENDS where the router sends a turn. Both
# default to one instance on :8080 that reads and generates.
#
# BACKENDS='
# gpu0_0 8080 qwen-mtp.sh
# cpu1_0 8081 qwen-mtp-cpu.sh SLOTS=1 NODE=1
# '
#
# One socket and no GPU: two readers over every core, and a generator that
# floats over the same cores at a higher priority.
#
# BACKENDS='
# pre0 8080 cpu-prefill.sh  CPUSET=0-31 THREADS=32
# pre1 8081 cpu-prefill.sh  CPUSET=0-31 THREADS=32
# gen0 8082 cpu-generate.sh CPUSET=0-31 THREADS=16 SLOTS=2
# '
#
# The JSON has the same names plus what each instance may do: "prefill" and
# "generate". "pref" orders the instances a turn would rather generate on,
# lowest first. A turn read on one instance moves to the first instance in
# that order that generates and can take it, if that one comes before the
# reader; otherwise it is answered where it was read. Peers get the same pref:
# a move between two equal instances costs a copy and buys nothing.
# Prefill off: an instance that only answers. Generate off: read for the pool
# and hand every turn on. The router refuses a table with nothing to prefill
# on, nothing to generate on, or an instance that does neither.
#
# "prefill" can also list the paths an instance reads for. At least one
# instance must read every path.
# koishi runs gpu0_0 with "prefill": ["/v1/systemone"]. "prefill": true made it
# one reader among four: the router reads on it last, and a cpu reader that
# generates keeps the turn, so chat generated on a cpu at 5-7 tok/s instead of
# the gpu. It first went back
# to false: the gpu read 120-token questions at 9.4 tokens/s, because each
# batch copied the experts over PCIe. With GGML_OP_OFFLOAD_MIN_BATCH high
# (bin/qwen-mtp.sh) it reads them at about 68, against 37 on a cpu socket.
#
#   [{"name": "gpu0_0", "url": "http://127.0.0.1:8080", "pref": 0,
#     "prefill": false, "generate": true,  "node": 0},
#    {"name": "cpu1_0", "url": "http://127.0.0.1:8081", "pref": 1,
#     "prefill": true,  "generate": true,  "node": 1}]
#
# export ROUTER_BACKENDS=$ROOT/backends.local.json

# The provider id in a generated client config. It names the machine, not
# the model: one client can reach two of these. Default: the hostname.
# export PROVIDER=myhost

# Where a system block goes. Default: beside the rest of run/slots. Every new
# session reads a block at its start. On a machine with two disks, put it on
# the faster one.
# export BLOCK_DIR=/mnt/nvme/qwen-blocks

# Disk the conversation copies may take, in GB. One copy per live
# conversation in RUN/slots, 0.2 to 5.4 GB each at ctx 150000. When full, the
# least recently used goes, which costs a full re-read. Default 256.
# export PARK_BUDGET_GB=256

# Disk the saved openings may take, in GB. Size it to the disk BLOCK_DIR is
# on. One block is about 0.6 to 3.7 GB. The least recently used goes first,
# and a deeper cut goes before a system prompt. Default 64.
# export BLOCK_BUDGET_GB=80

# Extra backend paths the router forwards, comma separated. The router serves
# the two chat apis and a short list of endpoints a client asks about itself,
# and refuses the rest. The backends run with --agent, which is shell and file
# access, and the router is the public port. Widen this only for a path a
# client of yours needs.
# export PASS_THROUGH=/v1/rerank,/lora-adapters

# Where the router listens. start-all.sh reads both. bin/restart-backend.sh
# reaches the router at ROUTER_PORT unless ROUTER names the whole address.
# ROUTER_PORT=8090
# ROUTER_HOST=::

# Read and generate in the same slot, for one instance with one slot. Set 0
# when the backend table has no generate-only instance, or when llama.cpp
# cannot carry a slot's checkpoints across a restore (see
# patches/slot-state-carries-checkpoints.patch).
# export HANDOFF=1

# One json line per cache decision, in a dated file. CACHE_LOG=0 turns it
# off. CACHE_LOG_DIR moves it off run/. tools/cache-report.py reads them.
# export CACHE_LOG=1
# export CACHE_LOG_DIR=$ROOT/run

# Debugging only: write every request body to a file, to compare two turns
# offline. Off unless this names a directory.
# export CAPTURE=$ROOT/run/captures

# The small models tests/live downloads. Default: under ~/.cache.
# export LIVE_MODEL_DIR=/mnt/nvme/live-e2e-models
