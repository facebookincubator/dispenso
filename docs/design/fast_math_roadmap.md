# dispenso::fast_math Roadmap

This document tracks planned features and improvements for the fast_math sublibrary.

## V1 Scope (Current)

### Functions Shipped

| Category | Functions |
|----------|-----------|
| Trigonometric | sin, cos, tan, asin, acos, atan, atan2 |
| Exponential | exp, exp2, exp10 |
| Logarithmic | log, log2, log10 |
| Other | cbrt, sqrt, frexp, ldexp |

All functions are templated on float type and support:
- Scalar `float`
- SSE4.1 (`SseFloat` / `__m128`)
- AVX2 (`AvxFloat` / `__m256`)
- AVX-512 (`Avx512Float` / `__m512`)
- NEON/aarch64 (`NeonFloat` / `float32x4_t`)
- Highway (`HwyFloat` / `hn::Vec<>`)

### V1 TODO

| Task | Status | Notes |
|------|--------|-------|
| Complete SIMD benchmark coverage | Done | All functions covered across SSE/AVX/AVX-512/NEON/Highway |
| Benchmark on Linux x86 | Done | Results recorded in README |
| Benchmark on Mac ARM | Pending | Compare vs Apple libm (different algorithms than glibc) |
| Benchmark on Windows MSVC | Pending | Compare vs MSVC CRT (different algorithms than glibc) |
| Rewrite README | Done | Structured docs with SIMD benchmarks, accuracy tables |
| Audit SIMD test edge cases | Done | NaN/inf/denorm coverage across all backends |
| Update sin_pi_4 via Sollya fpminimax | Done | 3.3x better sup-norm error (2.44e-10 → 7.32e-11) |
| Remove sin_wide/cos_wide | Done | Marginal 30% SIMD gain not worth maintaining; cos_wide unviable |
| Consolidate eval tools | Done | Single `ulp_eval` tool with dispenso::parallel_for, covers 12 functions |
| CMake build support | Done | Gated behind `-DDISPENSO_BUILD_FAST_MATH=ON` (experimental) |
| Evaluate additional transcendentals | Pending | sinh, cosh, tanh, pow, hypot, expm1, log1p (see Post-V1 section) |

---

## Post-V1: Additional Functions

Prioritized by demand in real-time graphics, physics, and ML inference workloads.

### Priority 1 — Strong candidates

| Function | Rationale |
|----------|-----------|
| `hypot(x, y)` | `sqrt(x*x + y*y)` with overflow/underflow protection. Very common in geometry code — dispenso's primary audience. Simple to implement well. |
| `pow(x, y)` | General power function. `exp(y * log(x))` works as a stopgap but loses accuracy for integer exponents and has edge case issues (negative base, 0^0). Dedicated implementation can be faster and more correct. |
| `sincos(x)` | Compute sin and cos simultaneously. Many workloads (rotation matrices, complex multiplication) need both. Shared range reduction makes this ~1.5x faster than separate sin+cos calls. |
| `sinpi(x)`, `cospi(x)`, `sincospi(x)` | Compute sin(πx), cos(πx), or both. Exact at integer and half-integer arguments without range reduction error accumulation. Important for FFT twiddle factors, Chebyshev nodes, and anywhere the argument is naturally in units of π. |

### Priority 2 — Useful in specific domains

| Function | Rationale |
|----------|-----------|
| `expm1(x)` | `exp(x) - 1` with precision near zero. Important for numerical stability in financial, scientific, and signal processing code. |
| `log1p(x)` | `log(1 + x)` with precision near zero. Paired with expm1 for stable computation. |
| `tanh(x)` | ML activation function. Most ML runs on GPU, but CPU inference (edge devices, ONNX Runtime) would benefit. Can be derived from exp. |

### Priority 3 — Niche / low demand

| Function | Rationale |
|----------|-----------|
| `sinh(x)`, `cosh(x)` | Hyperbolic functions. Rare in hot loops outside of specialized physics simulations. Derivable from exp. |
| `erf(x)`, `erfc(x)` | Error function. Statistical/ML use cases. GELU activation uses erf, but typically GPU-side. |
| `pow(x, int n)` | Integer power specialization. Faster than general pow via repeated squaring. |
| `fmod(x, y)`, `remainder(x, y)` | Modular arithmetic. Less clear speedup opportunity vs libc — these are integer-like. |
| `lgamma(x)`, `tgamma(x)` | Gamma functions. Very niche. |

### Not planned

| Function | Reason |
|----------|--------|
| Double precision (`double`) variants | fast_math targets `float` workloads (graphics, real-time). Double-precision demand is low for our audience. Could be added later if justified. |
| Complex math | Out of scope — different domain. |

---

## Post-V1: Infrastructure

| Task | Priority | Notes |
|------|----------|-------|
| Benchmark automation script | High | Run all benchmarks, collect results, generate markdown tables. Per SLEEF: document CPU model, clock speed (turbo off), compiler version, OS. |
| Accuracy measurement automation | High | Sweep full domain of each function, compute max ULP error vs MPFR (like SLEEF does). Currently ULP numbers are manually measured. |
| Compiler Explorer (Godbolt) examples | Medium | Show generated assembly for scalar vs SIMD to demonstrate vectorization. |
| Comparison benchmarks vs other libs | Medium | Benchmark against SLEEF, Eve (Expressive Vector Engine), and Highway contrib/math. Compare speed and accuracy on identical hardware/inputs. Useful for positioning and identifying where we can improve. |
| Comparison table vs SLEEF/SVML | Medium | Side-by-side accuracy and speed comparison. Helps users decide. |
| Refit asin polynomials via Sollya with explicit FMA | Medium | `asin_0_pt5` uses non-FMA polynomial (`a * b + c`), which the compiler may auto-contract differently on SIMD vs scalar, causing 1 ULP divergence. Refit coefficients via Sollya `fpminimax` targeting explicit FMA evaluation (like `sin_pi_4` was done). This will make scalar and SIMD paths produce identical results. |
| Investigate Estrin polynomial evaluation | Medium | Current polynomials use Horner evaluation (sequential FMA chain). Estrin's scheme reduces critical-path depth from ~9 to ~4 FMAs, potentially improving SIMD throughput. However, coefficients were fitted for Horner evaluation order — switching to Estrin with existing coefficients regresses atan ULP from 3 to 4. Need to refit coefficients via Remez (requires modifying the boost-based Remez harness to use Estrin evaluation during fitting). Candidates: atan_poly, asin polynomials, and any other deep Horner chains. Also consider converting atan_poly to odd-only form (evaluate in x^2) to halve polynomial degree. |
| CMake SIMD backend options | Medium | Add CMake options to enable specific SIMD backends (SSE4.1, AVX2, AVX-512, NEON, Highway) and set appropriate compiler flags (e.g. `-mavx2`, `-mavx512f`). Include option for `-march=native`. Also handle finding/fetching Highway when the Highway backend is enabled. |
| Binary float constant representations | Medium | Replace decimal float literals with C99 hex float constants (e.g. `0x1.921fb6p+1f`) where appropriate. Hex floats are exact, avoid questions about truncation from excess decimal digits, and match the format used by Sollya/MPFR output. |
| Doxygen documentation for public API | Medium | Add doxygen comments to all public functions documenting precision, domain, AccuracyTraits behavior, and SIMD compatibility. Note which functions ignore AccuracyTraits. |
| `DefaultSimdFloat` usage examples | Medium | Show how to write portable SIMD code using the type alias. |
| Power accounting hooks | Low | Original design goal — compile-time zero-overhead instrumentation for power profiling. Deferred. |

---

## Design Notes

### Why not just use SLEEF?

dispenso::fast_math occupies a different niche:

1. **Tighter integration with dispenso** — Same build system, same namespace, same coding conventions. No external dependency to manage.
2. **FloatTraits abstraction** — Generic programming over any SIMD type (including raw intrinsics) via a single template parameter. SLEEF requires calling platform-specific function names.
3. **AccuracyTraits** — Compile-time selection between speed and accuracy variants via template parameter, not function name suffixes.
4. **Simpler subset** — 17 functions vs SLEEF's 70+. Less code to maintain, audit, and understand.
5. **Scalar-first design** — Every function works on plain `float` with the same API. SIMD is opt-in, not required.

For users who need the full breadth of SLEEF's function coverage, SLEEF is the right choice. For users who want fast transcendentals integrated into a C++ parallel computing library with a clean generic API, dispenso::fast_math fills that gap.
