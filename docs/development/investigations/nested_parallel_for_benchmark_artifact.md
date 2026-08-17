# Nested parallel_for Benchmark Optimization Artifact

**Context.** `nested_for_benchmark` serial benchmarks show near-zero times on
Windows (0.00x and 0.01x ratio vs Linux), indicating the compiler is optimizing
away the computation. The benchmark loop body needs `benchmark::DoNotOptimize`
or equivalent to prevent dead code elimination.

**TODO:** Add `benchmark::DoNotOptimize` to the serial benchmark loop in
`nested_for_benchmark.cpp` to ensure the computation is not elided.
