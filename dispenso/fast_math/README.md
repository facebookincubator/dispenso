# dispenso::fast_math

> **EXPERIMENTAL** — This sublibrary is under active development. The API is
> unstable and subject to breaking changes without notice. Do not depend on it
> in production code.
>
> fast_math is not built by default. To include it in the CMake build, set
> `-DDISPENSO_BUILD_FAST_MATH=ON`.

A fast, accurate, SIMD-friendly math library for `float` transcendentals.

- [Design Goals](#design-goals)
- [Quick Start](#quick-start)
- [Accuracy Traits](#accuracy-traits)
- [SIMD Support](#simd-support)
- [Function Reference](#function-reference)
- [Performance](#performance)

## Design Goals

- **Fast**: 1.5-6x faster than glibc scalar math, with near-linear SIMD scaling
- **Accurate**: Drop-in replacements for `std::` math functions (1-4 ULP typical)
- **SIMD-native**: Same template works for `float`, `__m128`, `__m256`, `__m512`,
  `float32x4_t`, and Highway vectors — no source changes needed
- **Configurable**: `AccuracyTraits` template parameter trades speed for precision
  and edge-case handling at compile time

## Quick Start

```cpp
#include <dispenso/fast_math/fast_math.h>

namespace dfm = dispenso::fast_math;

// Scalar — drop-in replacement for std::sinf
float y = dfm::sin(x);

// SSE4.1 (4-wide) — same function, SIMD type
__m128 y4 = dfm::sin(x4);

// AVX2 (8-wide)
__m256 y8 = dfm::sin(x8);

// Max accuracy with bounds checking
float y = dfm::sin<float, dfm::MaxAccuracyTraits>(x);
```

For auto-detected best SIMD width:
```cpp
#include <dispenso/fast_math/simd.h>

using F = dispenso::fast_math::DefaultSimdFloat;
F result = dispenso::fast_math::sin(F(1.0f));
```

## Accuracy Traits

All functions accept an optional `AccuracyTraits` template parameter:

| Trait | `kMaxAccuracy` | `kBoundsValues` | Description |
|:------|:-:|:-:|:---|
| `DefaultAccuracyTraits` | false | false | Fastest. Correct over normal-range inputs. |
| `MaxAccuracyTraits` | true | true | Highest accuracy + edge-case handling (NaN, Inf, denorm). |

Custom traits can mix these independently:
```cpp
struct BoundsOnly {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;  // Handle NaN/Inf, keep fast polynomials
};
float y = dfm::exp<float, BoundsOnly>(x);
```

## SIMD Support

### Supported Types

| Backend | Type | Width | Guard Macro |
|:--------|:-----|:-----:|:------------|
| Scalar | `float` | 1 | always |
| SSE4.1 | `__m128` | 4 | `__SSE4_1__` |
| AVX2 | `__m256` | 8 | `__AVX2__` |
| AVX-512 | `__m512` | 16 | `__AVX512F__` |
| NEON | `float32x4_t` | 4 | `__aarch64__` |
| Highway | `HwyFloat` | variable | `__has_include("hwy/highway.h")` |

All SIMD backends provide `FloatTraits` specializations with the required operations
(`fma`, `sqrt`, `conditional`, `bit_cast`, etc.). Include `<dispenso/fast_math/fast_math.h>`
and the appropriate backend headers are auto-included based on platform macros.

### How It Works

Every function is templated on `Flt`. When `Flt` is a SIMD vector type, all
operations (arithmetic, comparisons, bit manipulation) dispatch through
`FloatTraits<Flt>` specializations that map to the corresponding intrinsics.
Polynomial coefficients stay as scalar `float` arrays — the implicit
`SseFloat(float)` / `_mm256_set1_ps` broadcast is efficient on modern hardware.

## Function Reference

### Trigonometric

| Function | Signature | Domain | Max ULP (Default) | Max ULP (MaxAccuracy) |
|:---------|:----------|:-------|:--:|:--:|
| `sin` | `sin(x)` | all float | 1 ([-128pi,128pi]), 2 ([-1Mpi,1Mpi]) | 1 ([-1Mpi,1Mpi]), 2 (full) |
| `cos` | `cos(x)` | all float | 1 ([-128pi,128pi]), 2 ([-1Mpi,1Mpi]) | 1 ([-128pi,128pi]), 2 ([-1Mpi,1Mpi]) |
| `tan` | `tan(x)` | all float | 3 ([-128Kpi,128Kpi]), 32 ([-1Mpi,1Mpi]) | 3 ([-128Kpi,128Kpi]), 4 ([-1Mpi,1Mpi]) |

### Inverse Trigonometric

| Function | Signature | Domain | Max ULP |
|:---------|:----------|:-------|:--:|
| `acos` | `acos(x)` | [-1, 1] | 3 |
| `asin` | `asin(x)` | [-1, 1] | 2 |
| `atan` | `atan(x)` | all float | 3 |
| `atan2` | `atan2(y, x)` | all float | 3 (Default), 3 (MaxAccuracy, +Inf handling) |

### Exponential

| Function | Signature | Domain | Max ULP (Default) | Max ULP (MaxAccuracy) |
|:---------|:----------|:-------|:--:|:--:|
| `exp` | `exp(x)` | [-89, 89] | 3 | 1 |
| `exp2` | `exp2(x)` | [-127, 128] | 1 | 1 |
| `exp10` | `exp10(x)` | [-40, 40] | 2 | 2 |

With `kBoundsValues = true`, exp/exp2/exp10 correctly handle NaN, +/-Inf, and
out-of-range inputs (returning 0 for large negative, Inf for large positive).

### Logarithmic

| Function | Signature | Domain | Max ULP (Default) | Max ULP (MaxAccuracy) |
|:---------|:----------|:-------|:--:|:--:|
| `log` | `log(x)` | (0, +Inf) | 2 | 2 |
| `log2` | `log2(x)` | (0, +Inf) | 1 | 1 |
| `log10` | `log10(x)` | (0, +Inf) | 3 | 3 |

With `kBoundsValues = true`, log/log2/log10 correctly handle 0, denormals,
Inf, and NaN inputs.

### Other

| Function | Signature | Max Error | Notes |
|:---------|:----------|:----------|:------|
| `sqrt` | `sqrt(x)` | 0 ULP | Delegates to hardware `sqrt` |
| `cbrt` | `cbrt(x)` | 12 ULP (Default), 3 ULP (MaxAccuracy) | |
| `frexp` | `frexp(x, &e)` | 0 | Bit-accurate |
| `ldexp` | `ldexp(x, e)` | 0 | Bit-accurate |

## Performance

Speedup ratios relative to platform libc (scalar) or scalar fast_math (SIMD).
Absolute throughput varies by clock speed and microarchitecture; ratios are
more meaningful for comparison.

### Linux x86-64 — AMD EPYC Genoa

166 cores, clang 21.1.7, `-O2 -march=native`, CMake Release build.

#### Scalar: fast_math vs glibc

| Function | Speedup | Function | Speedup |
|:---------|:-:|:---------|:-:|
| sin | 1.6x | exp | 3.8x |
| cos | 1.6x | exp2 | 2.7x |
| tan | 2.9x | exp10 | 2.5x |
| acos | 2.2x | log | 3.4x |
| asin | 3.4x | log2 | 3.0x |
| atan | 1.7x | log10 | 3.3x |
| atan2 | 3.3x | cbrt | 5.6x |
| frexp | 3.8x | ldexp | 5.8x |

#### SIMD Scaling (per-element throughput relative to scalar fast_math)

| Function | SSE (4) | AVX (8) | AVX-512 (16) | Highway (16) |
|:---------|:-:|:-:|:-:|:-:|
| sin | 2.7x | 5.4x | 6.5x | 5.8x |
| cos | 2.6x | 5.2x | 6.5x | 5.7x |
| tan | 3.6x | 7.0x | 8.7x | 7.7x |
| acos | 4.9x | 9.8x | 10.1x | 10.2x |
| asin | 1.9x | 3.8x | 4.3x | 4.3x |
| atan | 4.8x | 9.5x | 11.9x | 11.9x |
| atan2 | 3.5x | 7.0x | 9.6x | 9.6x |
| exp | 3.7x | 7.4x | 7.7x | 7.3x |
| exp2 | 3.7x | 7.5x | 8.8x | 7.2x |
| exp10 | 3.7x | 7.3x | 8.6x | 7.1x |
| log | 3.7x | 7.4x | 8.8x | 8.8x |
| log2 | 3.6x | 7.3x | 9.6x | 9.6x |
| log10 | 3.7x | 7.4x | 9.2x | 9.2x |
| cbrt | 2.9x | 5.7x | 7.0x | 7.1x |
| frexp | 7.4x | 14.8x | 23.0x | 22.9x |
| ldexp | 4.8x | 9.7x | 16.6x | 16.6x |

Highway dispatches to AVX-512 (16-lane) on this hardware.

#### Highway vs hwy::contrib (AVX-512 dispatch)

| Function | Ratio | Function | Ratio |
|:---------|:-:|:---------|:-:|
| sin | 0.76x | log | 1.60x |
| cos | 0.74x | log2 | 1.62x |
| exp | 1.63x | log10 | 1.67x |
| exp2 | 2.21x | atan | 1.12x |
| acos | 1.63x | asin | 0.87x |
| atan2 | 1.67x | | |

fast_math is 1.6-2.2x faster for exp/log family, comparable for inverse trig,
and ~25% slower for sin/cos (contrib uses a different polynomial fit).

#### MaxAccuracyTraits overhead (SSE4.1)

| Function | Overhead | Function | Overhead |
|:---------|:-:|:---------|:-:|
| sin | ~0% | log | 46% |
| cos | ~0% | cbrt | 24% |
| exp | 37% | | |

### Mac ARM — Apple M4 Pro

12 cores, 48 GB, Apple clang, `-O2`.

#### Scalar: fast_math vs Apple libm

| Function | Speedup | Function | Speedup |
|:---------|:-:|:---------|:-:|
| sin | 1.0x | exp | 1.7x |
| cos | 1.4x | exp2 | 1.7x |
| tan | 1.2x | exp10 | 1.6x |
| acos | 1.6x | log | 1.6x |
| asin | 1.8x | log2 | 1.0x |
| atan | 1.8x | log10 | 1.6x |
| atan2 | 1.7x | cbrt | 2.6x |
| frexp | 1.4x | ldexp | 2.0x |

Apple's libm is highly optimized for M-series silicon; the smaller speedups
reflect a stronger baseline rather than slower fast_math.

#### SIMD Scaling (per-element throughput relative to scalar fast_math)

| Function | NEON (4) | Highway (4) |
|:---------|:-:|:-:|
| sin | 5.4x | 4.9x |
| cos | 3.7x | 3.5x |
| tan | 4.6x | 4.1x |
| acos | 4.8x | 3.5x |
| asin | 2.8x | 2.9x |
| atan | 3.4x | 2.3x |
| atan2 | 2.2x | 1.3x |
| exp | 3.9x | 3.6x |
| exp2 | 4.0x | 3.9x |
| exp10 | 3.9x | 2.7x |
| log | 4.4x | 2.7x |
| log2 | 3.5x | 1.1x |
| log10 | 2.5x | 1.6x |
| cbrt | 3.1x | 2.4x |
| frexp | 4.0x | 4.3x |
| ldexp | 4.0x | 4.0x |

Highway dispatches to NEON (4-lane) on AArch64. NEON backend is generally
faster than Highway on ARM due to lower abstraction overhead.
