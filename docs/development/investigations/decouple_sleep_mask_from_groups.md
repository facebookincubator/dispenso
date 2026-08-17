# Decouple sleep mask from group concept

**Context.** The wake-cascade design (see [wake_cascade.md](../architecture/wake_cascade.md))
bundles three things into the "group" abstraction:
1. **Steal-ring locality** — threads in a group share a steal ring
2. **EpochWaiter / futex address** — one futex per group; `bumpAndWakeAll`
   wakes every parked thread in the group
3. **Sleep mask** — a 64-bit atomic per group where bit *i* tracks thread
   *i*'s sleep state, used by the bitmask cascade to find wake targets

The 1.6 default of `kDefaultWakeGroupSize=8` works well across Linux, macOS,
and Windows. The cross-platform tuning sweep did not surface a workload
that wanted these three to be sized differently — but the coupling is
implicit, not measured.

**Why investigate.** A few hypothetical cases where the coupling could hurt:

- **Steal-ring contention vs wake bucket contention** scale with different
  factors. Steal-ring contention grows with active producers + consumers
  on the ring; wake bucket contention grows with concurrent
  `WakeByAddress*` callers on the same address. A workload heavy on one
  but not the other might benefit from independent sharding.
- **Per-platform asymmetry.** On Windows where wake calls are 7-10 μs each,
  a smaller wake-mask shard (fewer potential targets per `wakeN`) might
  reduce spurious wake amplification, while keeping a larger steal-ring
  shard for locality. Today both are tied to `kDefaultWakeGroupSize`.
- **>64 thread groups.** The 64-bit sleep mask caps group size at 64.
  Not a limit anyone has hit — current G=8 leaves 56 bits unused on every
  platform — but a different sleep-tracking representation would lift it
  if a future design wanted very large groups.

**Proposed experiments (in order of cost):**

1. **Add separate `kSleepMaskShardSize`** that defaults to `groupSize_` (preserves
   current behavior). Sweep it independently in `run_tuning_experiment.py`.
   Expected to be a no-op based on current data; the value is ruling out
   the hypothesis that they want to be different.
2. **Wider sleep-mask representation** (e.g. an array of 64-bit words per
   group) only if (1) shows a workload preferring shard size > 64.
3. **Independent steal-ring sharing** sweep — already partially exposed
   via `kStealRingSharing` but historically tied to `kDefaultWakeGroupSize`
   in defaults. Decouple in defaults, sweep, see if any platform wants
   them different.

**Cost.** Mostly mechanical refactor — ~200 lines across `thread_pool_wake.h/cpp`
and `thread_pool.cpp`. The risk is tuning regression: more knobs means
more parameter space to validate per platform.

**When to revisit.** If a user reports a workload with > 64 contending
threads on a single steal-ring, or if a future architecture extension
(e.g. NUMA-node-aware wake) wants different sharding for wakes vs steal.
Otherwise, the current coupling is the simplest design that's correct.
