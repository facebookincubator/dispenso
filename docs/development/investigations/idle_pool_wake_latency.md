# Idle-Pool Wake Latency vs CPU Cost (mostly-idle then burst)

**Objective — dual.** This suite targets *both* low burst latency (wall time)
**and** low CPU spent staying responsive. A keep-alive strategy that only
minimized wall time by spinning would just trade the problem for idle CPU/power
(mobile battery, shared hosts), so future work must be judged on both axes. The
Plotly dashboard now plots `cpu_time` as a dotted companion to wall time on the
idle_pool charts to make the tradeoff visible.

**Context.** The `idle_pool_benchmark` `mostly_idle` scenario — a pool that sits
idle and then receives periodic bursts of work — is dispenso's weakest suite
versus TBB, consistently ~0.24-0.26x (TBB ~4x faster by geomean) across
platforms, with far larger worst-case gaps:

| Platform | Case | dispenso | TBB | Ratio |
|----------|------|----------|-----|-------|
| Apple M4 Pro (12c) | `mostly_idle/12/1e6` | 3.76 s | 30 ms | ~125x |
| Pixel 9 Pro XL / Tensor G4 (8c) | `mostly_idle/8/1e6` | 2.3 s | 166 ms | ~14x |

**Why.** dispenso aggressively parks idle workers (futex / `WaitOnAddress` /
ulock) to yield the CPU when there is no work; waking them for each burst pays
the full wake-from-park latency (scheduler + context switch + cache warm-up),
which dominates when the burst itself is small. TBB keeps workers alive/spinning
adaptively so a burst finds them ready, and its SPMC-deque self-stealing tends
to re-run a repeating burst on the same core with a warm cache. dispenso has no
special-casing for the mostly-idle-then-burst pattern.

**Related: sustained static loops vs OpenMP.** The same park/wake cost — plus
dispenso's runtime chunk/steal-ring setup vs OpenMP's compile-time static
scheduling — leaves dispenso ~15-20% behind OpenMP on *large, sustained*
`parallel_for` (simple_for / summing_for / locality at large sizes, ~0.81-0.86x;
far worse at small sizes where OpenMP's `OMP_WAIT_POLICY=active` team is already
hot). dispenso still beats TBB ~2x on the same locality cases, so this is
specific to OpenMP's persistent hot team. The removed `AwakeRef`/`keepAwake` API
targeted part of this; a bounded keep-alive hint (below) would help here too.

**Possible approaches:**
- Adaptive park delay / hysteresis: keep recently-active workers spinning longer
  before parking when the pool has seen recent bursty activity, trading a little
  idle CPU for burst latency. Composes with the adaptive spin backoff and
  wake-cascade tuning (see [wake_cascade.md](../architecture/wake_cascade.md)).
- Locality-preserving burst placement: bias re-scheduling of a repeating burst
  back onto the cores that last ran it (warm cache), analogous to TBB's
  self-steal.
- An opt-in pool hint/mode for latency-sensitive intermittent workloads that
  favors keep-alive over CPU yield.

**Cost / tradeoff.** Any keep-alive strategy burns CPU while "idle", so it must
be opt-in or bounded (e.g. decay after N idle periods) to avoid regressing the
genuinely-idle case and hurting battery/thermals on mobile. Validate with
`idle_pool_benchmark` on frequency-pinned runs (the retail-Pixel numbers above
are unpinned and carry extra variance).
