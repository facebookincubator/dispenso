# Recursive fork-join vs TBB task_group (deep / heavy trees)

**Context.** With fork-join scheduling complete (per-thread + steal rings),
dispenso is on par with `tbb::task_group` on shallow traversal — basic tree
~1.03x, kdtree ~1.01x, light-per-node work ~1.13x — but trails on deep and
heavy-per-node recursion: `tree_work` heavy-per-node ~0.71x and 4-ary deep
fork-join ~0.69x vs TBB (EPYC-Genoa 166c). It beats folly ~5.7x throughout, so
the gap is TBB-specific and is the clearest remaining fork-join weakness.

**Why.** TBB's continuation stealing keeps a deep recursion's working set on the
stealing core and avoids re-scheduling overhead at every node; dispenso
re-schedules each `parallel_invoke` / recursive TaskSet child through the pool.

**Possible approaches:**
- Continuation-style stealing for `parallel_invoke` / recursive TaskSet so a
  stolen child resumes its parent on the stealing thread (warm working set).
- A recursion-depth / subtree-size cutoff that runs deep sub-trees inline past a
  threshold — the fork-join analogue of `parallel_for`'s `minItemsPerChunk`
  guardrail, which already wins the trivial-work cases.
- Validate against `tbb::task_group` on the `tree_work` (heavy) and `4ary` suites.
