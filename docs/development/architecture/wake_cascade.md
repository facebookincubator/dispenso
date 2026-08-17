# Wake Cascade System

Design document for dispenso's PoolWakeState: per-thread wake
infrastructure with leader-team parallel cascade.

**Status:** Implemented

## Overview

PoolWakeState manages thread sleep/wake lifecycle with these goals:

- **O(1) single-wake latency** for the common `schedule()` path
- **O(log N) bulk-wake latency** for fork-join and scheduleBulk
- **Zero overhead** when no cascade budget is pending (common case)
- **No redundant futex calls** via atomic sleep mask claims

## Groups and Subgroups

### Groups

Threads are partitioned into groups of `groupSize` (default 16 on Linux,
8 on Windows). Each group has a 64-bit atomic sleep mask where bit i
indicates thread `(group * groupSize + i)` is sleeping:

```
Group 0: threads [0, 16)    sleepMask: uint64_t
Group 1: threads [16, 32)   sleepMask: uint64_t
...
```

Group size matches topology-aware thread groups from CpuSet, so threads
in the same group typically share an L3 cache.

### Subgroups

Within each group, `kWaiterSubgroupSize` threads (default 4) share one
EpochWaiter (one futex address). A single `bumpAndWakeAll()` wakes all
4 with one syscall:

```
Group 0 (16 threads):
  Subgroup 0: threads 0-3    → EpochWaiter[0]  ← cascade team
  Subgroup 1: threads 4-7    → EpochWaiter[1]
  Subgroup 2: threads 8-11   → EpochWaiter[2]
  Subgroup 3: threads 12-15  → EpochWaiter[3]
```

## Sleep/Wake Protocol

### enterSleep(threadIdx)

Called before EpochWaiter sleep. Sets the thread's bit in its group's
sleep mask (`fetch_or`, release).

### exitSleep(threadIdx)

Called after waking. Clears the bit (`fetch_and`, relaxed). Handles the
timeout-wake case where no waker cleared the bit.

### tryClaimSleeper(threadIdx)

Called by wakers to atomically claim exclusive wake responsibility.
Returns true if this caller cleared the bit (owns the wake). The
`fetch_and` with `acq_rel` ensures the claim is visible to concurrent
wakers.

## Wake Methods

### 1. wakeOne()

Fast path for `conditionallyWake()`. Scans from a round-robin hint,
finds the first sleeping thread, bumps its waiter. No sleep mask claim
(thread clears its own bit in exitSleep). Redundant wakes are benign.

### 2. claimAndWakeOne()

Proactive wake for `schedule()`. Claims a sleeper atomically, bumps its
waiter, returns the thread index so the caller can push work to that
thread's steal ring.

### 3. wakeRange(count)

Targeted wake for fork-join. After `scheduleBulkToRings` pushes tasks to
rings 0..count-1, wakes exactly those threads. Masks off bits beyond the
range in the last group.

### 4. wakeOneWithBudget(budget)

Entry point for budget cascade. Claims a sleeping thread, assigns it a
wake budget, bumps its waiter. Prefers cascade-team threads (first
subgroup, bits 0..kWaiterSubgroupSize-1) via `kCascadeTeamMask`.

Budget counts cascade actions, not threads. Each action issues one futex
call that wakes a subgroup (up to kWaiterSubgroupSize threads). The budget
is deducted by kWaiterSubgroupSize/2 per action, which mildly over-wakes:
each action wakes up to 4 threads but only consumes 2 from the budget.
This is intentional — over-waking is a no-op (already-awake threads are
unaffected), while under-waking causes latency (threads wait for spin
timeout).

### 5. wakeAll()

Shutdown. Claims and bumps every sleeping thread. Spinning threads exit
via their `running_` flag.

## Leader-Team Parallel Cascade

### Cascade Team

The first subgroup of each group (threads 0-3 within each group)
participates in cascade wake. Only these threads have budget slots and
call `processBudget` in the thread loop:

```
Group 0: cascade team = threads 0,1,2,3     (have budget slots)
         regular      = threads 4,5,...,15   (skip budget check)

Group 1: cascade team = threads 16,17,18,19
         regular      = threads 20,...,31
```

Total budget slots = `numGroups * kWaiterSubgroupSize` (e.g., 11 × 4 =
44 on a 166-thread EPYC). Each slot is cache-line padded (2.8KB total).

### Budget Flow

To wake N threads via `wakeN(N)`:

```
Step 0: Caller
  wakeOneWithBudget(N-1)
  → claims thread 0 (group 0, cascade team)
  → budget[0] = N-1
  → bumpAndWakeAll() on subgroup 0
  → threads 0,1,2,3 all wake

Step 1: Thread 0 runs processBudget(0)
  → exchanges budget: N-1
  → scans for sleeping cascade-team in group 1
  → claims thread 16, deducts kWaiterSubgroupSize/2 = 2
  → remaining = N-3, distributes among 16,17,18,19:
      thread 17: (N-3)/4
      thread 18: (N-3)/4
      thread 19: (N-3)/4
      thread 16: remainder
  → bumpAndWakeAll() on group 1 subgroup 0
  → threads 16-19 all wake

Step 1 (parallel): Threads 1,2,3 run processBudget
  → budget = 0 → return immediately (fast path)

Step 2: Threads 16,17,18 each have budget > 0
  → each finds and wakes cascade-team in another group
  → fan-out width = kWaiterSubgroupSize per level
```

### Fan-Out Depth

```
Level 0: 1 futex call              →  up to 4 threads woken
Level 1: up to 4 parallel futex calls  → up to 16 additional threads woken
Level 2: up to 16 parallel futex calls → up to 64 additional threads woken
Level 3: up to 64 parallel futex calls → up to 256 additional threads woken

Depth to wake N threads: ceil(log_4(N))
(assumes kWaiterSubgroupSize=4; each futex call wakes one subgroup)
```

### Inlined processBudget Fast Path

The budget check runs at the top of every outer thread loop iteration.
The fast path is fully inlined:

```cpp
int32_t processBudget(int32_t threadIdx) {
    uint16_t slot = budgetSlotTable_[threadIdx];  // precomputed lookup
    if (slot == kNoCascade) return 0;              // not cascade team (75%)
    if (wakeBudgets_[slot].value.load(relaxed) <= 0) return 0;  // no budget
    return processBudgetSlow(threadIdx);            // out-of-line slow path
}
```

**Cost for non-cascade threads** (75%): 1 table lookup + 1 comparison.
**Cost for idle cascade threads**: + 1 relaxed load + 1 comparison.
**Slow path**: only when a non-zero budget is found (rare, cascade-only).

### Precomputed budgetSlotTable_

Maps `threadIdx → budget slot index` (or `kNoCascade` sentinel).
Eliminates division and modulo in the hot path:

```
Thread 0  → slot 0       Thread 4  → kNoCascade
Thread 1  → slot 1       Thread 5  → kNoCascade
Thread 2  → slot 2       ...
Thread 3  → slot 3       Thread 15 → kNoCascade
Thread 16 → slot 4       Thread 20 → kNoCascade
...
```

Allocated once at pool construction. 166 × 2 bytes = 332 bytes.

## Tuning Knobs

| Define | Default | Effect |
|--------|---------|--------|
| `DISPENSO_TUNE_WAKE_BRANCH_FACTOR` | 4 (Linux), 8 (Windows) | Unused after leader-team refactor (kept for API compat) |
| `DISPENSO_TUNE_WAKE_GROUP_SIZE` | 16 (Linux), 8 (Windows) | Threads per wake group |
| `DISPENSO_TUNE_WAITER_SUBGROUP_SIZE` | 4 | Threads sharing one EpochWaiter |
| `DISPENSO_TUNE_SPIN_CHECK_INTERVAL` | 64 (Linux), 32 (macOS), 256 (Windows) | Spin iterations between time checks |
| `DISPENSO_TUNE_MAX_SPIN_US` | 128 (Linux/macOS), 256 (Windows) | Max spin duration before sleeping |
| `DISPENSO_TUNE_MIN_SPIN_US` | 1 | Min spin duration after idle sleeps |

### Adaptive Spin Timeout

```
Found work during spin:   spinTimeout = min(kMax, spinTimeout * 2)
Woke from sleep, no work: spinTimeout = max(kMin, spinTimeout * 0.5)
```

Keeps threads hot during sustained bursts (128us spin) but backs off to
near-zero (1us) when idle.

## Correctness Properties

1. **No lost wakes.** EpochWaiter epoch ensures races between
   `bumpAndWakeAll` and `waitFor` are resolved correctly.
2. **No stale sleep bits.** Every `enterSleep` is paired with `exitSleep`.
3. **Budget is never double-spent.** `exchange(0)` atomically claims the
   budget; losers see 0.
4. **Cascade terminates.** Budget is strictly decreasing (each step
   deducts kWaiterSubgroupSize/2). Exhausts in at most
   2*N/kWaiterSubgroupSize steps.
5. **Shutdown is complete.** `wakeAll()` bumps every sleeping thread's
   waiter. Spinning threads exit via `running_` flag.
