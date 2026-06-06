# Windows dispenso tuning notes, 2026-05-18

Machine for the Windows sweep:

- Windows 11, AMD64
- Intel Xeon Platinum 8259CL @ 2.50 GHz
- 96 logical CPUs reported by Python, 48 CPUs reported by Google Benchmark

Files from this sweep:

- `windows_sweep_20260518_171038/spin_strategy_tuning.json`
- `windows_sweep_20260518_171038/promote_seed_tuning.json`
- `windows_sweep_20260518_171038/full_benchmark_sweep.json`
- `rerun_rw_lock_long_timeout.json`
- `rerun_trivial_compute_long_timeout.json`

## Sweep status

The main full benchmark sweep completed 21 of 24 benchmark targets. The 600 second per-target timeout was hit by:

- `trivial_compute_benchmark`
- `rw_lock_benchmark`
- `idle_pool_benchmark`

The longer timeout reruns completed `trivial_compute_benchmark` and `rw_lock_benchmark`, so those were long-running targets rather than hangs. `idle_pool_benchmark` still needs a separate bounded rerun if a complete Windows full-sweep artifact is required.

## Spin strategy tuning

`spin_fixed_200` was the best aggregate candidate in this run, about 6% faster than baseline by geomean over the tuning set. The result is mixed, with notable regressions in some locality cases, so this should be treated as a candidate that needs a focused confirmation pass rather than an automatic default change.

## Promote seed tuning

Promote-seed remains unconvincing. `promote_seed_0` was roughly flat to slightly positive overall, while `promote_seed_1` was slightly negative and mixed. There is no strong reason from this sweep to enable promote seed on Windows.

## DistributedRWLock benchmark interpretation

The `rw_lock_benchmark` timeout is explained by very slow, write-heavy `DistributedRWLock` cases, not by deadlock or livelock. The full long-timeout rerun completed.

The data shows that this is true of `DistributedRWLock` in general, not only `DistributedRWLock<128>`:

- Write period 2: all distributed variants are bad. `DistributedRWLock<8>` is roughly 3x to 19x slower than the best non-distributed lock, `DistributedRWLock<16>` is roughly 5x to 24x slower, and `DistributedRWLock<128>` is up to roughly 131x slower.
- Write period 8: all distributed variants are still the wrong answer. `DistributedRWLock<8>` is roughly 1.3x to 3.4x slower, `DistributedRWLock<16>` roughly 2.5x to 5.4x slower, and `DistributedRWLock<128>` up to roughly 25x slower.
- Write period 32: mixed. `DistributedRWLock<8>` can win at some concurrencies, but distributed locks are still risky at high concurrency.
- Write periods 128 and 512: distributed locks become useful. `DistributedRWLock<8>` and `DistributedRWLock<16>` usually win, and `DistributedRWLock<128>` is only attractive in very read-heavy, high-concurrency cases.

This should not be hidden by deleting benchmark coverage. The useful benchmark split is:

- Keep representative read-mostly `DistributedRWLock` cases in the default benchmark sweep.
- Move write-heavy distributed-lock cases into a stress or misuse benchmark path, or run them with smaller problem sizes and a longer timeout.
- Keep enough write-heavy data to document the failure mode, but do not let a known misuse case dominate the normal benchmark runtime.

Suggested default/stress split for `rw_lock_benchmark`:

- Default sweep: keep `std::shared_mutex` and `RWLock` at write periods `{2, 8, 32, 128, 512}`.
- Default sweep: run `DistributedRWLock<N>` primarily at write periods `{128, 512}`, optionally `{32, 128, 512}` for `DistributedRWLock<8>`.
- Stress sweep: keep the full `DistributedRWLock<N>` matrix at write periods `{2, 8, 32, 128, 512}`, but use a smaller value count or a separate benchmark target so the full suite remains bounded.

## idle_pool_benchmark interpretation

The `idle_pool_benchmark` timeout is likely the known `BM_dispenso_mostly_idle` recursive one-task-at-a-time cliff. Earlier probes showed `BM_dispenso_mostly_idle/8/1000000` and `BM_dispenso_mostly_idle/48/1000000` taking roughly 1.0 to 1.6 seconds per Google Benchmark iteration on Windows. With broad argument coverage and repetitions, that can exceed a 600 second target-level timeout.

This does not appear to implicate the normal idle/wake workloads:

- `BM_dispenso_very_idle` is intentionally dominated by the 100 ms sleep and should remain bounded.
- `BM_dispenso_periodic_burst` is the more realistic idle-then-wake burst case and should remain in the default suite.
- The pathological case is the recursive `mostly_idle` chain with large element counts and higher worker counts.

Suggested default/stress split for `idle_pool_benchmark`:

- Default sweep: keep `mostly_idle` at smaller element counts such as `{1000, 10000}`.
- Default sweep: keep `very_idle` and `periodic_burst`.
- Stress sweep: move `mostly_idle` with `1000000` elements to a separate stress path or run it with fewer repetitions and an explicit longer timeout.

## Benchmark runner policy

The broad benchmark sweep should stay representative and bounded. Use separate modes for:

- Full sweep: representative benchmark coverage, no known multi-minute stress cases, reliable completion.
- Stress sweep: pathological or misuse cases that are useful for documentation and regression detection.
- Confirmation sweep: selected high-signal cases with repetitions for final tuning decisions.

This keeps the data honest without allowing known pathological cases to make routine benchmark collection take hours.
