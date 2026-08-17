# ConcurrentVector Access-Path Optimizations

How `ConcurrentVector`'s indexed access path is optimized, and how those
optimizations interact with its traits. Planned follow-on work is on the
[roadmap](../roadmap.md#concurrentvector).

## Optimizations Applied (Default Traits)

Three categories of optimization have been applied to ConcurrentVector:

1. **Inline asm `bsr` for `detail::log2`** on x86 GCC/Clang, plus 32-bit
   overloads and `unsigned long` disambiguation for macOS. Prevents Clang from
   decomposing `63 - __builtin_clzll` back into `bsrq + xorq` when inlined
   into arithmetic.

2. **Platform-adaptive `bucketAndSubIndexForIndex`**: branching fast path
   (early return for `index < firstBucketLen_`) on MSVC and ARM where branch
   predictors handle the sequential pattern well; branchless cmov path on
   Clang/GCC x86 where cmovs avoid misprediction penalties.

3. **Non-atomic buffer pointer cache (`cachedPtrs_[]`)** on non-ARM platforms.
   Packs 8 pointers per cache line (vs 1 per line for `AlignedAtomic
   buffers_[]`), dramatically improving `operator[]` and iterator read paths.
   Disabled on ARM where cache-line invalidation on every write exceeds the
   read benefit. Cache stores are ordered before the release store to
   `buffers_[]`, so any acquire on `buffers_[]` guarantees cache visibility.

## Impact on Alternative Traits

| Trait | Values | Optimization Interaction |
|-------|--------|--------------------------|
| `kPreferBuffersInline` | `false` | Cache is *more* valuable — bypasses the extra indirection through heap-allocated `buffers_[]` pointer |
| `kIteratorPreferSpeed` | `false` (compact iterator) | Benefits *disproportionately* — compact iterator calls `operator[]` (and thus `cachedBuffer` + `bucketAndSubIndexForIndex`) on every dereference, vs speed iterator which only calls on bucket transitions |
| `kReallocStrategy` | `kHalfBufferAhead`, `kFullBufferAhead` | No interaction — earlier allocation just means cache is populated earlier |

**Conclusion:** All optimizations apply uniformly across trait combinations.
The current defaults (`kPreferBuffersInline=true`, `kIteratorPreferSpeed=true`,
`kReallocStrategy=kAsNeeded`) remain the best general-purpose configuration.
The compact iterator (`kIteratorPreferSpeed=false`) benefits the most from the
`cachedPtrs_` and `log2` optimizations in relative terms, since it hits the
indexed access path on every element access.

