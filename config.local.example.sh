#!/usr/bin/env bash
# Copy to config.local.sh and edit. That file is not tracked, so the defaults
# in the repository stay defaults and nobody's paths reach the history.
#
# bin/common.sh reads it before deciding anything, so every launch script, the
# prefill harness and the experiment runners see it. Export any name the router
# reads, because the router is a separate process.

# Where the weights are. The default is models/ in the checkout, which suits
# one socket. Two sockets want a copy each on the fastest disk, as separate
# files: a hard link or a reflink shares the page cache, which is the thing the
# second copy exists to avoid. Exported, because the prefill harness is a
# separate process.
# export MODELS=/mnt/nvme/models
# export MODELS2=/mnt/nvme/models-node1

# Where llama-server is. The default is the patched fork built under the
# checkout; see patches/README.md for what the patches are and why.
# SERVER_MTP=/opt/llama.cpp/build/bin/llama-server

# A different model. MODELS above only moves the directory, so a model with
# another name needs these too. DRAFT is the MTP draft model; leave the
# --model-draft flags out of ARGS if there is none.
# MODEL_Q8=$MODELS/my-model-00001-of-00004.gguf
# MODEL2_Q8=$MODELS2/my-model-00001-of-00004.gguf
# DRAFT=$MODELS/my-draft.gguf
# DRAFT2=$MODELS2/my-draft.gguf
# MMPROJ=$MODELS/mmproj-F16.gguf      # vision. VISION=0 starts without it.

# Threads per backend. Counted from the physical cores on the node it is bound
# to if this is unset, which is usually right.
# THREADS=18

# Context per slot. Every backend must use the same number: a conversation that
# outgrew one could never move back to it. 150000 needs about 15.3 GiB of VRAM
# at f16 KV, which is roughly 36.5 KiB a token - size it to your own card, and
# see docs/LAYOUT.md for how that was measured. NGL and
# N_CPU_MOE divide the model between card and RAM - lower N_CPU_MOE keeps more
# experts on the card, which needs the VRAM to hold them.
# export CTX=150000
# export N_CPU_MOE=48
# export NGL=99

# Where everything a run writes goes: logs, pids, saved slots, cache events.
# The checkout's own run/ if unset. Exported, because the router reads it too,
# and a router looking somewhere other than --slot-save-path finds no caches.
# export RUN=/var/lib/llama-arbiter

# More than one backend takes two tables that have to agree: BACKENDS says what
# start-all.sh launches, ROUTER_BACKENDS where the router sends a turn. Both
# default to a single instance on :8080 that reads and generates.
#
# BACKENDS='
# gpu0_0 8080 qwen-mtp.sh
# cpu1_0 8081 qwen-mtp-cpu.sh SLOTS=1 NODE=1
# '
#
# The JSON carries the same names plus what each instance may do: "prefill"
# and "generate", with "pref" ordering where a turn would rather generate.
# Both on is read-and-answer in place; prefill off makes it the generator that
# turns migrate to; generate off makes it read for the pool and hand every turn
# on. A table with nothing to prefill on, nothing to generate on, or an
# instance that does neither is refused at startup.
#
#   [{"name": "gpu0_0", "url": "http://127.0.0.1:8080", "pref": 0,
#     "prefill": false, "generate": true,  "node": 0},
#    {"name": "cpu1_0", "url": "http://127.0.0.1:8081", "pref": 1,
#     "prefill": true,  "generate": true,  "node": 1}]
#
# export ROUTER_BACKENDS=$ROOT/backends.local.json

# The provider id in a generated client config. It names the machine, not the
# model, because one client can reach two of these. The hostname if unset.
# export PROVIDER=myhost

# Where a system block goes. Beside the rest of run/slots by default. A block
# is read at the start of every new session, so on a machine with two disks it
# belongs on the faster one.
# export BLOCK_DIR=/mnt/nvme/qwen-blocks

# Deeper openings: a cut where two conversations diverge, read and saved so a
# branch of a session can start from it. Off, because here nothing ever loaded
# one. The detection still runs - tools/cache-report.py says how often a fork
# could have started deeper - so turn this on if it says they are real.
# export DEEP_OPENINGS=1

# Disk the conversation copies may take, in GB - one per live conversation, in
# RUN/slots, 0.2 to 5.4 GB each at ctx 150000. The least recently used goes
# when it is full, and that costs a full re-read. Default 256.
# export PARK_BUDGET_GB=256

# Disk the saved openings may take, in GB. Size it to the disk BLOCK_DIR is
# on: one block runs about 0.6 to 3.7 GB, the least recently used goes first,
# and a deeper cut goes before a system prompt. Default 64.
# export BLOCK_BUDGET_GB=80

# Extra backend paths the router will forward, comma separated. It serves the
# two chat apis and a short list of the endpoints a client asks about itself,
# and refuses the rest: the backends run with --agent, which is shell and file
# access, and the router is the public port. Widen it only for something a
# client of yours really needs.
# export PASS_THROUGH=/v1/rerank,/lora-adapters

# Where the router listens. start-all.sh reads both, and bin/restart-backend.sh
# reaches the router at ROUTER_PORT unless ROUTER names the whole address.
# ROUTER_PORT=8090
# ROUTER_HOST=::

# Read and generate in the same slot, for one instance with one slot. Set 0
# where the backend table has nothing set to generate only, or where
# llama.cpp cannot carry a slot's checkpoints across a restore (see
# patches/slot-state-carries-checkpoints.patch).
# export HANDOFF=1

# One json line per cache decision, in a dated file. CACHE_LOG=0 turns it off;
# CACHE_LOG_DIR moves it off run/. tools/cache-report.py reads them.
# export CACHE_LOG=1
# export CACHE_LOG_DIR=$ROOT/run

# Debugging only: write every request body down, for comparing two turns
# offline. Off unless this names a directory.
# export CAPTURE=$ROOT/run/captures

# The small models tests/live downloads. Under ~/.cache if unset.
# export LIVE_MODEL_DIR=/mnt/nvme/live-e2e-models
