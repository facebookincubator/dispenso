# Three-Tier Queue Architecture

Design document for dispenso's three-tier task scheduling infrastructure:
locality rings, steal rings, and the central MPMC queue.

**Status:** Implemented

## Overview

ThreadPool maintains three tiers of task storage, each optimized for a
different scheduling pattern:

```
            ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
            │  Locality Rings  │  │   Steal Rings     │  │  Central Queue   │
            │  (Tier 1)        │  │   (Tier 2)        │  │  (Tier 3)        │
            ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
            │ 16 slots/thread  │  │ 64 slots/group   │  │ unbounded        │
            │ per-thread       │  │ per-group shared  │  │ global shared    │
            │ fork-join affine │  │ proactive wake    │  │ overflow/general │
            └──────────────────┘  └──────────────────┘  └──────────────────┘
                    ▲                      ▲                      ▲
                    │                      │                      │
            scheduleBulkToRings     schedule()            scheduleBulkEnqueue
            (parallel_for)         (pipeline,futures)     (bulk, overflow)
```

All three tiers feed into a unified consumer loop (`tryFindAndExecuteWork`)
that each pool thread runs.

## Tier 1: Locality Rings

Per-thread MPMC ring buffers (`MpmcRingBuffer<OnceFunction, 16>`) that
provide deterministic thread-to-chunk affinity for fork-join parallel_for.

### When used

Populated by `scheduleBulkToRings()` via the fork-join fast path in
`TaskSetBase::scheduleBulkImpl()`. Activates when:

- `count * 4 >= numPool` — enough tasks to meaningfully distribute
- `count <= numPool` — at most one task per ring (kStatic pattern)
- Not pool-recursive — no nested TaskSet
- Not overloaded — `outstandingTaskCount_ <= taskSetLoadFactor_`

### Layout

Tasks are distributed linearly: task i goes to ring i. The fast path
requires `count <= numPool`, so at most one task per ring (kStatic
pattern). kAuto (count > numPool) bypasses locality rings entirely and
uses the central queue via `scheduleBulkEnqueue`.

### Wake integration

After distribution, `wakeRange(count)` wakes exactly the threads whose
rings received work, avoiding spurious wakes on empty rings.

## Tier 2: Steal Rings

Per-group MPMC ring buffers (`MpmcRingBuffer<OnceFunction, 64>`) for
non-locality work distribution. 16 threads share one steal ring, aligned
with wake group boundaries.

```
Threads 0-15   → StealRing 0  (Group 0)
Threads 16-31  → StealRing 1  (Group 1)
Threads 32-47  → StealRing 2  (Group 2)
...
```

### When populated

Two paths within `schedule(ForceQueuingTag)` → `scheduleImpl()`:

**Proactive wake:** `claimAndWakeOne()` finds a sleeping thread, pushes
the task to its group's steal ring, and bumps its subgroup's EpochWaiter.
All subgroup peers wake and can see the task in their shared steal ring.

**No-sleeper round-robin:** When all threads are busy, distributes to
steal rings via per-thread round-robin target, skipping the caller's own
group. Guarded by `!isPoolRecursive` (see below).

### Capacity sizing

`kStealSlotsPerThread (4) × kStealRingSharing (16) = 64` slots per ring.
Provides buffering for bursty `schedule()` calls without excessive memory.

## Tier 3: Central Queue

Unbounded MPMC queue (`moodycamel::ConcurrentQueue<OnceFunction>`) that
handles general-purpose scheduling and overflow from the other two tiers.

### When used

- Non-pool callers (no producer token) after proactive-wake attempt
- Steal ring overflow (try_push fails)
- Ring overflow from `scheduleBulkToRings`
- `scheduleBulkEnqueue` (standard bulk path, bypasses rings entirely)
- Pool-recursive schedule (steals rings skipped, goes straight to queue)

Pool threads use per-thread `ProducerToken` for better throughput.

## Consumer: tryFindAndExecuteWork

Each pool thread checks sources in an order determined by the `preferRing`
sticky hint:

```
if preferRing:
    [1] locality ring → [2] central queue → [3] steal ring (empty() guard)
else:
    [1] central queue → [2] locality ring
    (no steal ring check — found in outer loop instead)
```

### The preferRing sticky hint

- **Starts true** (after creation or waking from sleep)
- **Switches to false** when ring is empty but queue has work
- **Switches to true** when queue is empty but ring has work
- Persists until the opposite source produces a hit

This prevents oscillation and keeps threads checking the most productive
source first. During sustained parallel_for, threads stay ring-preferred
and never touch the contended central queue. During pipeline/futures work,
threads become queue-preferred and skip the locality ring check.

### Asymmetric steal ring check

The steal ring is checked as tier 3 **only on the ring-preferred path**.
Queue-preferred threads skip it to avoid cache line traffic when steal
rings are rarely populated (futures/pipeline patterns). Steal ring work
is still consumed in the **outer thread loop** before backoff/sleep.

## Pool-Recursive Guard

The no-sleeper steal ring distribution is skipped when
`isPoolRecursive(this)` is true (a pool thread scheduling back into its
own pool). In recursive patterns:

1. Most work is inlined by the tight `quickLoadFactor` (1.5× numThreads)
2. The small burst that escapes inlining is better served by the central
   queue, which is designed for high-contention multi-producer access
3. Steal ring distribution adds overhead (round-robin counter, skip-self
   check, try_push) without proportional benefit for small bursts

## Load Factor and Inline Decisions

Two checks gate the transition from queuing to inline execution:

| Context | Threshold | Effect |
|---------|-----------|--------|
| Pool-recursive | `numThreads * 1.5` | Aggressive inline for nested work |
| Non-recursive | `numThreads * 32` | Deep queuing for maximum parallelism |
| TaskSet-level | `numThreads * 4` | Per-TaskSet overload prevention |
| Graph executor | `numThreads * 3.0` via `poolRecursiveLoadFactor` | Less inline for graph distribution |

The `poolRecursiveLoadFactor` is a per-call parameter on
`ConcurrentTaskSet::schedule` and `ConcurrentTaskSetExecutor::operator()`,
allowing workload-specific tuning.

## Invariants

1. **Every queued task is counted.** `workRemaining_` is incremented
   before enqueueing. Decrement happens after execution (batched in the
   thread loop).
2. **No task is lost.** If try_push to a ring fails, the task falls
   through to the central queue (unbounded).
3. **Shutdown drains all tiers.** Destructor and `resizeLocked()` drain
   central queue, all locality rings, and all steal rings after joining.
4. **Proactive wake is at-most-once.** `claimAndWakeOne()` atomically
   clears the sleep bit; concurrent callers skip to the next sleeper.
5. **Missed wakes are safe.** The EpochWaiter timeout provides a safety
   net — threads self-wake within `sleepLengthUs_`.
