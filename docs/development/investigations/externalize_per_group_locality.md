# Experiment: externalize per-group locality; simplify thread-pool tiers

**Context.** The pool has three tiers — per-thread rings, per-group steal rings,
central queue. `kAdaptive` already shows algorithms can own their locality
(stripes + own victim policy). Hypothesis: letting known-layout algorithms
(`parallel_for`/`reduce`/…) manage locality externally — **in addition to**, not
replacing, the steal rings — could give better-tailored behavior and trim
per-iteration `threadLoop` work (also helping the idle-pool CPU objective above).

**Keep.** Per-thread rings stay — the preferred path for bulk known-layout
scheduling of `parallel_for`/`reduce`/etc.

**Key risk.** Steal rings are the *universal* cross-algorithm work-stealing
fallback. If each algorithm only manages its own structure, a thread committed to
algorithm A's work cannot be stolen for algorithm B — risking load imbalance and,
for nested / `wait`-blocking patterns, potential **deadlock**. So steal rings
most likely remain as the safety net and external management layers on top,
rather than replacing them.

**Scope.** (1) Profile `threadLoopImpl` to quantify each tier's per-iteration
cost, so any simplification is justified by measured idle-path savings.
(2) Prototype an external placed-locality hook for `parallel_for`/`reduce`
alongside steal rings. (3) Prove cross-algorithm steal and deadlock-freedom hold.
Supersedes the "steal-ring round-robin for placed scheduling" backlog idea;
interacts with "Decouple sleep mask from group concept".

