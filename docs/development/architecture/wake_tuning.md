# Wake-cost tuning guide

How to tune the wake-side parameters of dispenso's thread pool for a new
platform (or revalidate an existing one). The defaults in the source are
chosen for a typical 16/32/96/166-core x86 Linux machine; mac and Windows
have higher per-syscall overhead and benefit from different settings.

## Background

Workers wait on a per-group `EpochWaiter` (one futex address per group).
Producers wake threads via three primitives:

| primitive | semantics | platform mapping |
|---|---|---|
| `bumpAndWake()` | wake one waiter | `futex_wake(addr, 1)` / `__ulock_wake_one` / `WakeByAddressSingle` |
| `bumpAndWakeAll()` | wake all parked waiters on this address | `futex_wake(addr, INT_MAX)` / `__ulock_wake_all` / `WakeByAddressAll` |
| `bumpAndWakeN(n, total)` | wake N waiters | Linux: single syscall. Mac/Windows: loop wake-one or fall through to wake-all (see threshold below). |

Producers also fan out across groups via the **bitmask cascade** (see
`thread_pool_wake.h::wakeN`) and may pre-seed a second group via
**promote-seed** so cascade can fan out from two seeds in parallel with
just one producer syscall.

## The knobs

All overridable at compile time via `-D` flags:

| macro | default | description |
|---|---|---|
| `DISPENSO_TUNE_WAKE_GROUP_SIZE` | 8 (all platforms) | Threads per group. Larger = fewer cascade hops, more bucket-lock contention, more queue-walk per wake. |
| `DISPENSO_TUNE_STEAL_RING_SHARING` | 8 | Threads per steal ring. Lower = less ring contention, more rings to scan (mitigated by has-work bitmask). Typically matched to group size. |
| `DISPENSO_TUNE_WAKE_ALL_THRESHOLD` | INT_MAX (Linux/Mac), 3 (Windows) | K at which `bumpAndWakeN` switches from looping wake-one to a single wake-all. Linux's exact-K syscall makes this irrelevant; on macOS Mach IPC wake-one is so cheap that K wake-ones cost about the same as one wake-all (so wake-one wins by avoiding spurious wakes); on Windows the per-call cost is high enough that wake-all wins early. |
| `DISPENSO_TUNE_PROMOTE_SEED` | 1 (Linux/Mac), 0 (Windows) | Whether to use the promote-seed cascade trick when `numGroups > groupSize`. Disabled on Windows because the extra in-thread `WakeByAddressSingle` to bootstrap g1 doesn't pay for itself there. |
| `DISPENSO_TUNE_WAKE_BRANCH_FACTOR` | 4 (Linux/Mac), 8 (Windows) | Producer-side fan-out branching for `wakeBudget` cascade. |

## Tuning process

### Step 1 — measure platform wake costs

Build and run `wake_cost_bench`:

```bash
cmake -S . -B build -DDISPENSO_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target wake_cost_bench
./build/bin/wake_cost_bench
```

Output looks like:

```
Spurious-wake round-trip: 11200 ns

Producer-side wake costs (avg over 200 trials):
    N |     wake-1 |   wake-all | K threshold (wake-all wins for K >=)
------+------------+------------+------------------------------------
    4 |     3300 ns |    10300 ns | 4
    8 |     3700 ns |    20700 ns | 6
   16 |     3300 ns |    38400 ns | 12
   32 |     3400 ns |    75300 ns | 22
   ...
```

Read off the K threshold for your intended group size. Set
`DISPENSO_TUNE_WAKE_ALL_THRESHOLD` to that value.

For a default group size of 16 on Linux, the threshold is irrelevant
(Linux uses exact-K). On macOS the row N=16 might show K=4-6; on Windows
the threshold tends to land at K=2-3 because each `WakeByAddressSingle`
syscall is more expensive.

### Step 2 — choose group size

The default G=8 came out of a sweep across G ∈ {16, 8, 4} on a 166-thread
Linux machine; G=8 + promote-seed was the net winner (see commit log).
G=16 was the previous default and may still be worth measuring on
platforms with very expensive per-syscall costs.

Smaller groups (G=8 vs G=16) win when:

- **Steal-ring contention** dominates: workloads with many small placed
  tasks (e.g. tree benchmarks) push tasks onto specific group rings.
  Fewer threads per ring = less contention on the ring's atomics.
- **Bucket-lock contention on the group futex** is meaningful: every
  per-thread wake acquires the group's futex bucket lock. Half the
  threads per group = half the contenders.

Smaller groups cost:

- More cascade hops to wake the full pool.
- More producer-side syscalls if the cascade can't single-indirect
  (numGroups > groupSize without promote-seed).

To find the sweet spot, sweep G ∈ {8, 16, 32} across the benchmarks
that matter for your workload (`pipeline_benchmark`,
`idle_pool_benchmark` for burst patterns, `tree_benchmark` for
fine-grained task dispatch).

Set per-platform defaults in `thread_pool_wake.h`:

```cpp
#elif defined(__APPLE__)
constexpr int32_t kDefaultWakeGroupSize = 16;  // tune per measurements
#elif defined(_WIN32)
constexpr int32_t kDefaultWakeGroupSize = 8;   // tune per measurements
```

### Step 3 — promote-seed on/off

Promote-seed adds one extra in-thread wake call when wakeN's cascade
target count exceeds the chosen group size (i.e., the seed group's
threads can't claim all the bits in parallel). For a 96-thread pool with
G=16, numGroups=6 ≤ G, so promote-seed never fires — keeping it on is
free.

For G=8 on the same pool, numGroups=12 > G, so promote-seed engages on
every full-pool wake. Worth keeping on unless wake_cost_bench shows the
extra in-thread syscall is unusually expensive on this platform.

### Step 4 — validate with a benchmark sweep

Run the standard benchmark suite against the previous default to confirm
the new tuning is a net improvement:

```bash
python3 scripts/run_benchmarks.py --build -o new_results.json
# Compare against your previous baseline.
```

Pay attention to:

- **`pipeline_benchmark`** — frequent small wakes via `claimAndWakeOne`.
- **`idle_pool_benchmark`** (periodic burst) — full-pool wake latency.
- **`tree_benchmark`** small/medium variants — high-rate per-task wakes
  through `scheduleImplPlaced` + steal-ring contention.

## Current per-platform defaults

These are starting points, not necessarily optimal. Each entry includes
links to the historical sweep that informed the choice. Update this
table as you tune.

| platform | groupSize | sharing | wakeAllThreshold | promoteSeed | source |
|---|---:|---:|---:|---:|---|
| Linux x86_64 | 8 | 8 | INT_MAX (exact-K syscall) | 1 | group-size sweep, 166-thread machine: G=8 + promote-seed beat G=16 net |
| macOS | 8 | 8 | INT_MAX | 1 | wake_cost_bench on M-series: K × wake-1 ≈ wake-all (Mach IPC wake-one is cheap), so wake-1 × K wins by avoiding spurious wakes. Group-size sweep showed no significant variation (low core count → cascade rarely engages). |
| Windows | 8 | 8 | 2 | 0 | per-bench sweep on 96-thread Xeon: G/sharing/wake_factor flat across the swept range; promote_seed=0 wins ~1.5% mean (clearest on `BM_dispenso_blocking/*`) because each WakeByAddressSingle is expensive enough that the extra in-thread bootstrap wake doesn't amortize. |

## Caveats

- `bumpAndWake()` (wake-one) on Windows is currently implemented as
  `WakeByAddressAll` — the comment in `epoch_waiter.h` explains why
  (per-call cost similar, keeps threads in their spin phase). For
  workloads that wake one specific thread very frequently and want
  zero spurious wakes, this may be worth revisiting on a measured
  Windows machine.
- Spurious-wake CPU cost (worker wakes, finds no work, re-sleeps) is
  bounded above by `spurious_round_trip * (groupSize - K)` per
  wake-all but happens in parallel on otherwise-idle cores, so its
  wall-time impact is much smaller than the raw CPU cost suggests.
