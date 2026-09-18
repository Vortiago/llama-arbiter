# llama-arbiter

A router that gives every conversation a warm prompt cache, not only the few
that fit in a slot. These are the words this project uses for that. Use them
in code, in comments and in commit messages. Where two words exist for one
thing, the one here is the one to write.

## The work

**Conversation**:
One client's continuing run of turns with the model. It has one pin, one slot
and one parked copy. One turn runs at a time.
_Avoid_: session, chat, thread

**Turn**:
One prompt the client sends, and the reply it gets, inside a conversation.
_Avoid_: exchange, round, request as a name for the whole turn

**Ask**:
What one client sent for one turn, as the router reads it: the path, the
body, and the conversation a header named, when it did.
_Avoid_: request, payload, params

**Client**:
Whoever asked, behind the one seam a turn writes to. The public port is the
only real one; a test writes its own and runs a whole turn with no socket.
_Avoid_: caller, consumer, peer, connection

**Prefill**:
To read a prompt. The work is compute bound and runs for tens of minutes.
_Avoid_: ingest, process, evaluate

**Generate**:
To write the reply. The work is memory bound and runs for seconds.
_Avoid_: inference, decode, completion

**Handoff**:
The move of a turn from the backend that read its prompt to the one that
writes the reply.
_Avoid_: migration, transfer, failover

## Where the work runs

**Backend**:
One `llama-server` process. Each one prefills, generates, or does both.
_Avoid_: instance, server, worker, node

**Generator**:
A backend that generates but never prefills. Turns move to it after another
backend reads the prompt.
_Avoid_: writer, responder

**Slot**:
One place inside a backend that can hold a prompt. Each backend runs one,
because a slot reading a long prompt stops every other slot beside it.
_Avoid_: worker, lane, seat

**Pin**:
The record that ties a conversation to the backend and slot holding its cache.
The router sends the next turn where the pin says.
_Avoid_: assignment, binding, lease, affinity

## What is kept

**Park**:
To copy a conversation's cache out of its slot and onto disk, before anything
can take the slot.
_Avoid_: save, evict, spill, persist

**Parked copy**:
What a park leaves on disk. Short form: **copy**.
_Avoid_: conversation cache, live cache, state file, snapshot

**Recall**:
To put a parked copy back into a slot.
_Avoid_: load, restore, rehydrate

**Opening**:
A parked cache of the prompt prefix that new conversations start from, so that
none of them has to read it again. Usually a system prompt.
_Avoid_: block, prefix cache, template, preamble

**Cut**:
A point in a prompt where a shared prefix can end. The router can park an
opening there.
_Avoid_: split, boundary, checkpoint

**Want**:
An opening the router has found a use for but has not built yet.
_Avoid_: pending, todo, backlog

**Hold**:
What a slot has in it now, named by the cuts it covers.
_Avoid_: contents, state

**Shelf**:
The kind an opening belongs to. A **base** opening is a system prompt that
every conversation shares. A **deep** opening is parked at a cut where two
conversations diverge.
_Avoid_: tier, class, category

**Store**:
Everything one run keeps on disk: the parked copies, the openings, and the two
files that say what they are. A caller names a file. Where it lives is the
store's business.
_Avoid_: disk, filesystem, storage, or slot directory, as a name for the store
