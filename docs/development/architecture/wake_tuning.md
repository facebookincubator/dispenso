# Wake-cost tuning guide

How to tune the wake-side parameters of dispenso's thread pool for a new
platform, or revalidate an existing one, and why each current default was
chosen. How the wake machinery works is documented next to the code, in
`dispenso/detail/thread_pool_wake.h`.

## Background

Threads are partitioned into wake groups of G consecutive threads. Each
group shares one `EpochWaiter` (one futex address) and one sleep mask.
Producers wake threads via three primitives:

| primitive | semantics | platform mapping |
|---|---|---|
| `bumpAndWake()` | wake one waiter | `futex_wake(addr, 1)` / `__ulock_wake_one` / `WakeByAddressAll` on Windows (see Caveats) |
| `bumpAndWakeAll()` | wake all parked waiters on this address | `futex_wake(addr, INT_MAX)` / `__ulock_wake_all` / `WakeByAddressAll` |
| `bumpAndWakeN(n, total)` | wake N waiters | Linux/FreeBSD: one exact-K syscall. macOS: loop wake-one, or fall through to wake-all at the threshold below. Windows: `WakeByAddressSingle` if the group has one waiter, otherwise `WakeByAddressAll`. |

A wake of up to `branchFactor` threads claims and wakes sleepers one at a
time. Larger wakes bump and wake every affected group directly; on the
fork-join ring path, the first threads to wake also run pre-staged tasks
that wake further groups.

## The knobs

All are compile-time overrides passed as `-D` flags:

| macro | default | description |
|---|---|---|
| `DISPENSO_TUNE_WAKE_GROUP_SIZE` | 8 (all platforms) | Threads per wake group (max 64). Larger = fewer groups to wake, more contention on each group's futex and sleep mask. |
| `DISPENSO_TUNE_STEAL_RING_SHARING` | 8 | Threads per steal ring. Lower = less ring contention, more rings to scan. Normally matched to the group size. |
| `DISPENSO_TUNE_WAKE_BRANCH_FACTOR` | 4 (Linux/macOS), 8 (Windows) | Largest wake count served by claiming and waking sleepers one at a time; larger wakes wake whole groups. |
| `DISPENSO_TUNE_WAKE_ALL_THRESHOLD` | INT_MAX | macOS only: K at which `bumpAndWakeN` switches from looping wake-one to a single wake-all. K wake-ones cost about the same as one wake-all there, so wake-one wins by avoiding spurious wakes. Linux/FreeBSD wake exactly K in one syscall, and Windows never loops wake-one (see Background), so no other platform reads it. |
| `DISPENSO_TUNE_FIXED_SPIN_ITERS` | 400 (Linux/macOS), 200 (Windows) | Iterations of the work-finding loop before an idle worker parks. 0 parks immediately. |
| `DISPENSO_TUNE_SPIN_CHECK_INTERVAL` | 64 (Linux), 32 (macOS), 256 (Windows) | Length of the lean-spin warm-up, during which an idle worker checks only its own ring. Must be a power of 2. |

## Tuning process

`scripts/run_tuning_experiment.py` rebuilds dispenso once per configuration,
runs the representative cases in its `BENCHMARK_SUITE`, prints each
configuration's change against a baseline at the current defaults, and
writes `tuning_results_<os>.json`. `--list` shows every configuration, and
`-c <category>` runs one category plus the baseline:

| category | sweeps |
|---|---|
| `wake_all_threshold` | `WAKE_ALL_THRESHOLD` ∈ {2, 3, 4, 6, 8, 12, 16} |
| `wake_group` | `WAKE_GROUP_SIZE` ∈ {8, 16, 32, 64} |
| `wake_combo` | `WAKE_GROUP_SIZE` = `STEAL_RING_SHARING` ∈ {4, 8, 16, 32} |
| `wake_factor` | `WAKE_BRANCH_FACTOR` ∈ {2, 4, 8, 16} |
| `spin_strategy` | `FIXED_SPIN_ITERS` ∈ {0, 32, 64, 128, 200, 400, 800} |
| `spin_interval` | `SPIN_CHECK_INTERVAL` ∈ {16, 32, 64, 128, 256} |

Categories whose knob the host platform does not read are skipped with an
explanation (today only `wake_all_threshold`, outside macOS). `--force` runs
them anyway, for bringing up a platform whose implementation has not been
characterized; on systems without a tailored `EpochWaiter` backend (e.g.
NetBSD, OpenBSD, DragonFly BSD) the script notes that results describe the
generic fallback.

### Step 1 — wake-all threshold (macOS only)

Only macOS reads the threshold (see the knobs table); on other platforms
the script skips this category, so go to step 2. On macOS:

```bash
python3 scripts/run_tuning_experiment.py -c wake_all_threshold
```

Measuring end to end matters here, not just timing the syscalls: a
wake-all also wakes the group's other (G - K) parked threads, and their
spurious wake-and-repark costs CPU that a syscall microbenchmark does
not see.

Earlier tuning used a standalone syscall microbenchmark,
`tools/wake_cost_bench.cpp` (removed after 2e41885). On Windows (Xeon
Gold 6450C, August 2026) it measured `WakeByAddressSingle` and
`WakeByAddressAll` within 5% of each other at every N, consistent with
Windows using wake-all whenever more than one thread is waiting.

### Step 2 — group size

Sweep `-c wake_combo`, which keeps steal-ring sharing matched to the group
size as the defaults do; `-c wake_group` varies the group size alone.

Smaller groups win when:

- **Steal-ring contention** dominates: workloads with many small placed
  tasks (e.g. tree benchmarks) push tasks onto specific group rings.
  Fewer threads per ring = less contention on the ring's atomics.
- **Contention on the group futex** is meaningful: every wake into a group
  goes through that group's futex. Half the threads per group = half the
  contenders.

Smaller groups cost more groups to wake when the whole pool is needed.

To change a default, edit the `groupSize` default argument of
`PoolWakeState` in `thread_pool_wake.h`, and `kStealRingSharing` in
`thread_pool.h`.

### Step 3 — branch factor and spinning

`-c wake_factor`, `-c spin_strategy` and `-c spin_interval` cover the
remaining knobs. The spin settings trade wake latency against CPU burned
by idle workers, so judge them on `idle_pool_benchmark`'s CPU-usage cases
as well as on throughput.

### Step 4 — validate with a benchmark sweep

Run the standard benchmark suite with and without the new tuning and
compare:

```bash
python3 scripts/run_benchmarks.py --build -o before.json
# ... change the default ...
python3 scripts/run_benchmarks.py --build -o after.json
python3 scripts/compare_benchmarks.py before.json after.json
```

Pay attention to:

- **`pipeline_benchmark`** — frequent small wakes via `claimAndWakeOne`.
- **`idle_pool_benchmark`** (periodic burst) — full-pool wake latency.
- **`tree_benchmark`** small/medium variants — high-rate per-task wakes
  through `scheduleImplPlaced` + steal-ring contention.

## Current per-platform defaults

These are starting points, not necessarily optimal. Update this table as
you tune.

| platform | groupSize | sharing | wakeAllThreshold | spin limit | source |
|---|---:|---:|---:|---:|---|
| Linux x86_64 | 8 | 8 | INT_MAX (exact-K syscall) | 400 | group-size sweep on a 166-thread machine: G=8 beat G=16 net. |
| macOS | 8 | 8 | INT_MAX | 400 | syscall microbenchmark (since removed) on M-series: K × wake-1 ≈ wake-all (Mach IPC wake-one is cheap), so wake-1 × K wins by avoiding spurious wakes. Group-size sweep showed no significant variation (low core count). |
| Windows | 8 | 8 | n/a (wake-all for >1 waiter) | 200 | per-bench sweep on a 96-thread Xeon: group size, sharing and branch factor flat across the swept range. A fixed spin of 200 was ~6% faster by geomean than the earlier adaptive spin on a 48-thread Xeon Platinum 8259CL. |

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
