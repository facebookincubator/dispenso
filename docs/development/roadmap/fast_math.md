# dispenso::fast_math Roadmap

This document tracks planned features and improvements for the fast_math sublibrary.

## V1 Scope (Current)

### Functions Shipped

| Category | Functions |
|----------|-----------|
| Trigonometric | sin, cos, sincos, sinpi, cospi, sincospi, tan, asin, acos, atan, atan2 |
| Exponential | exp, exp2, exp10, expm1 |
| Logarithmic | log, log1p, log2, log10 |
| Hyperbolic | tanh |
| Power | pow, sqrt, cbrt, hypot |
| Other | frexp, ldexp |

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

### Future: GPU Device Functions (CUDA / HIP)

The tableless double-precision pow core (`pow_double_poly_core`) and its building
blocks (`DoubleVec`, `exp2_split`, `logarithmSep`) are pure arithmetic with no
platform-specific intrinsics — they use only `+`, `-`, `*`, `fma`, and
widening/narrowing conversions. This makes them natural candidates for GPU device
functions (`__device__`).

**Motivation**: GPU shader math libraries (CUDA `__powf`, HIP `__ocml_pow_f32`)
trade accuracy for speed. A <1 ULP pow on-device would serve:
- Physically-based rendering (PBR) and tone mapping
- ML inference with numerical sensitivity (small-exponent power laws)
- Scientific computing where double is too slow but single needs accuracy

**Approach — double-float (df64)**: Consumer GPUs have 1/32 double-precision
throughput (only datacenter GPUs like A100/H100 have 1/2). A true `double`
detour would be a net loss on most hardware. Instead, use **double-float**
arithmetic: carry extended precision as a `(hi, lo)` pair of `float` values,
using single-precision FMA for error-free transforms. This runs at full `float`
throughput on all GPUs.

The existing `log2_ext` in `fast_math_impl.h` is essentially a double-float
algorithm (extended-precision log2 with FMA error tracking), but it was designed
for CPU SIMD where the division and extra ops are tolerable. A GPU df64 version
would:
- Use the same `logarithmSep` range reduction (pure float/uint32)
- Evaluate `log2(m)` as a df64 polynomial (hi+lo pair, FMA error tracking)
- Multiply by `y` in df64 (Dekker splitting or FMA-based TwoProduct)
- Feed the result to `exp2` (existing float exp2 + EFT correction)

This avoids double entirely while achieving similar accuracy (~46 mantissa
bits in the y*log2(x) product). The CPU `DoubleVec` approach is the right
reference for algorithm structure, but the GPU implementation would substitute
df64 pairs for actual doubles.

**Key considerations**:
- `__fma_rn` (round-to-nearest FMA) is available on all CUDA architectures
  (sm_20+). HIP has `__fma`. Critical for df64 error-free transforms.
- Float↔double conversions are free on datacenter GPUs; on consumer GPUs the
  double path is prohibitive — df64 sidesteps this entirely.
- The `pow_double_hybrid_core` table-based path could also work on GPU
  (L1/shared memory lookups are fast), adapting gathers to `__ldg` or shared
  memory preload. Worth benchmarking against pure-polynomial df64.
- On datacenter GPUs with fast double (A100: 1/2 rate, H100: 1/2 rate), the
  existing `DoubleVec<float>` scalar-double approach may win outright — worth
  benchmarking both paths and dispatching based on `__CUDA_ARCH__`.

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
| Support targets without hardware FMA via double-precision reduction | Medium | Every SIMD backend currently requires hardware FMA (`DISPENSO_FAST_MATH_HAS_FMA`), because the Cody-Waite reductions assume single rounding. Without it the failure is catastrophic rather than gradual: measured on SSE4.1-without-FMA, `sin`/`cos` reach 3.9e6 / 5.6e6 ULP beyond +-pi, and `exp10` reached 71 ULP before its constant was truncated. More Cody-Waite terms do **not** fix this -- without fusion every term but the last must itself be truncated so `n*t` stays exact, giving only `24 - bits(n)` bits each, so `\|x\| <= 1Mpi` would need ~16 terms. The cheap fix is to carry the reduction in **double**: `n` (2^21) x `hi` (24 bits) = 45 bits fits exactly in double's 53, so no fusion and no truncation are needed. Measured with the existing constants, double-based reduction gives max \|dr\| of 2.8e-8 at 128pi and 3.0e-8 at 1Mpi against a 4.7e-8 budget -- correct across the full tested range. `detail/double_promote.h` (`DoubleVec`) already provides the machinery. Payne-Hanek is **not** the right tool here; it only earns its cost beyond ~2^28 where a bit-expansion genuinely runs out. Doing this would let the FMA gate on the SSE and Highway backends be relaxed. The live exposure is Highway on WASM SIMD128, whose `relaxed_madd` is explicitly permitted not to fuse. |
| Explicit FMA for deterministic out-of-range results | Medium | Under `DefaultAccuracyTraits` the result for non-finite input is unspecified by contract, but today it is also *irreproducible*: Cody-Waite reduction computes `x - n*ln2`, which is `inf - inf` for infinite input, so whether a function yields `inf` or `NaN` depends on how the compiler contracted the surrounding FMAs. The same source passed locally and failed in CI for `expm1`/`cos`. Writing the reductions with explicit `fma()` would fix the result per ISA path. Note this does **not** make the value correct — `inf - inf` is `NaN` either way — so it is about reproducibility, not accuracy; making these match std would mean guarding on the default path, which is exactly the overhead `kBoundsValues=false` exists to avoid. Same underlying issue as the asin refit above. |
| Unify `convert_to_int`'s non-finite result on `INT_MIN` | Medium | The value is currently unspecified and target-dependent: the five SIMD backends and CUDA mask non-finite lanes to `0`, while the scalar paths yield `INT_MIN`. `0` is the worst available choice — it is what any x in (-0.5, 0.5) legitimately converts to, and every caller consumes a range-reduced exponent where `0` is dead centre. That is exactly how `expm1`'s `n == 0` shortcut came to answer from the polynomial for `inf`; it is now ordered so the bounds checks take precedence, but the hazard remains for the next caller. `INT_MIN` aliases only `x == -2^31`, outside every caller's reduced range, and is **free on x86**: `_mm_cvtps_epi32` and `_mm_cvtss_si32` already return it for inf/NaN/out-of-range, so the backends spend three ops (`bit_cast`, compare, AND) converting it into the more dangerous value — deleting that mask is a hot-path win. NEON needs normalisation either way (`FCVTNS` saturates, NaN maps to 0), as do CUDA and the portable scalar path. Do this with benchmarks, then tighten the `util.h` contract from "unspecified" to `INT_MIN`. `INT_MAX` is the only genuinely unreachable value (float spacing near 2^31 is 128, so nothing rounds to 2147483647) but costs normalisation on x86, so it buys little over `INT_MIN`. |
| Investigate Estrin polynomial evaluation | Medium | Current polynomials use Horner evaluation (sequential FMA chain). Estrin's scheme reduces critical-path depth from ~9 to ~4 FMAs, potentially improving SIMD throughput. However, coefficients were fitted for Horner evaluation order — switching to Estrin with existing coefficients regresses atan ULP from 3 to 4. Need to refit coefficients via Remez (requires modifying the boost-based Remez harness to use Estrin evaluation during fitting). Candidates: atan_poly, asin polynomials, and any other deep Horner chains. Also consider converting atan_poly to odd-only form (evaluate in x^2) to halve polynomial degree. |
| CMake SIMD backend options | Medium | Add CMake options to enable specific SIMD backends (SSE4.1, AVX2, AVX-512, NEON, Highway) and set appropriate compiler flags (e.g. `-mavx2`, `-mavx512f`). Include option for `-march=native`. Also handle finding/fetching Highway when the Highway backend is enabled. |
| Binary float constant representations | Medium | Replace decimal float literals with C99 hex float constants (e.g. `0x1.921fb6p+1f`) where appropriate. Hex floats are exact, avoid questions about truncation from excess decimal digits, and match the format used by Sollya/MPFR output. |
| Doxygen documentation for public API | Medium | Add doxygen comments to all public functions documenting precision, domain, AccuracyTraits behavior, and SIMD compatibility. Note which functions ignore AccuracyTraits. |
| `DefaultSimdFloat` usage examples | Medium | Show how to write portable SIMD code using the type alias. |
| Power accounting hooks | Low | Original design goal — compile-time zero-overhead instrumentation for power profiling. Deferred. |

---

## Post-V1: Public API surface

The set of names users may rely on is currently implicit -- whatever happens to
be reachable from `fast_math.h` is public by accident. That needs to become a
deliberate list, for one concrete reason: **explicit backend selection is a
first-class use case, not a fallback.** AVX-512 downclocks on many Intel parts
and is frequently the slower choice, so users will deliberately name `AvxFloat`
on AVX-512 hardware rather than accept `DefaultSimdFloat`. All backends coexist
(on an AVX-512 host built with AVX2, `SseFloat` / `AvxFloat` / `Avx512Float` are
all available), so the named per-backend types *are* the interface and
`DefaultSimdFloat` is a convenience on top.

**Decision: user-facing types stay at top level in `dispenso::fast_math`.** An
earlier sketch defined them in `detail/` and re-exported (`using
detail::AvxFloat;`, the pattern `rw_lock.h` already uses). That was rejected
because moving the types changes ADL: unqualified `sin(v)` on an `AvxFloat`
would search `detail` and stop finding the public overloads -- a silent lookup
change rather than a compile error. Keeping the types where they are avoids it
entirely.

### Work

| Task | Priority | Notes |
|------|----------|-------|
| Named user-facing boolean types | Medium | Users can write `AvxInt32` but there is no `AvxBool`. Today `BoolType` is *the float type itself* on SSE/AVX/NEON/Highway (lane mask held in the vector), a distinct `Avx512Mask` (`__mmask16`) on AVX-512, and plain `bool` scalar (`kBoolIsMask = false`). Add named types so generic code has something to spell. |
| Promote the `_t` aliases to user-facing spellings | Medium | `SimdType_t`, `IntType_t`, `UintType_t`, `BoolType_t` in `float_traits.h` read as metafunction machinery but are what generic user code actually needs. Give them names intended for users, where that is cheap. |
| Enumerate and document the public list | Medium | Float types, the `*Int32` companions (definitely public), the boolean spelling, and a decision on `FloatTraits`. Then say so in the README rather than leaving it to inference. |

### Open questions

1. **Does `AvxBool` alias `AvxFloat`, or become a distinct wrapper?** Aliasing
   matches the representation and costs nothing, but erases the type distinction
   between a mask and a value, so `AvxFloat x = someComparison;` compiles.
   A wrapper restores safety at the cost of conversions and churn. Note the
   asymmetry: on AVX-512 the type is genuinely distinct already, so aliasing
   makes four of five backends behave one way and one the other.
2. **Rename `Avx512Mask` to `Avx512Bool`?** Consistency argues yes; it is the
   only backend whose bool type is separately named today.
3. **What is the scalar spelling?** For `float`, `BoolType` is plain `bool` and
   `kBoolIsMask` is false. Generic code that switches on the mask
   representation needs this to be uniform, or explicitly not.
4. **Is `FloatTraits` public?** `SimdType_t` plausibly is -- generic user code
   needs it. `FloatTraits` itself reads as internal, but the `_t` aliases are
   defined in terms of it, so exposing them exposes it indirectly.
5. **Do the backend headers stay non-self-contained?** `float_traits_avx512.h`
   cannot compile alone: it includes `util.h`, which references `Avx512Float`
   at the point it selects `DefaultSimdFloat`, before this header has defined
   it. The cycle only works because `util.h` is always entered first. Since the
   README already documents these as auto-included and never a user entry
   point, this may simply want a comment saying so. Fixing it properly means
   splitting `util.h` into a `util_core.h` (helpers, no backend dependency)
   that the backends include -- worth doing if the headers move anyway.

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

### SIMD Architecture Evolution

The current SIMD wrapper design (`FloatTraits<T>`, `SimdTypeFor<T>`) supports
fixed-width architectures (SSE, AVX, AVX-512, NEON) with native wrappers and
variable-width via Highway. This section outlines the path toward broader
architecture support and integration with dispenso's parallel algorithms.

#### Variable-Length Vector Architectures

**ARM SVE/SVE2** (128-2048 bit) and **RISC-V Vector Extension (RVV)** use
runtime-determined vector lengths. Key differences from fixed-width SIMD:

- Vector types are **sizeless** (`svfloat32_t`, `vfloat32m1_t`) — can't take
  sizeof, store in arrays, or (portably) place in structs.
- Comparisons return **predicate registers** (`svbool_t`, `vbool_t`), not
  lane-wide float masks. This differs from SSE/AVX (float masks) and AVX-512
  (bit masks), but the existing `BoolType_t<Flt>` / `bool_as_mask` abstraction
  extends naturally to predicates.
- RVV uses a **vector-length-agnostic (VLA)** programming model: `vsetvl(n)`
  sets the number of elements to process per iteration, and the hardware
  handles tail elements natively. This is closer to Cray-style vector
  processing than traditional SIMD.
- RVV's **LMUL** (register grouping) gangs 1-8 registers together, trading
  register count for width. Effective width is `VLEN * LMUL / element_bits`,
  all runtime-determined.

**Short-term strategy**: Use Highway for SVE/RVV. Highway already dispatches
to `HWY_SVE`, `HWY_SVE2`, and `HWY_RVV` targets. The `HwyFloat` wrapper
works regardless of underlying width. This avoids sizeless type complexity.

**Medium-term strategy**: If SVE becomes a high-priority target (e.g. for
Neoverse server workloads), add native `SveFloat` / `RvvFloat` wrappers with
predicate-aware `FloatTraits`. The existing `BoolType` pattern extends cleanly.

**POWER VMX/VSX**: Fixed 128-bit (4 lanes), similar to SSE. Low priority —
niche HPC audience. Highway covers it via `HWY_PPC8` if needed.

#### SimdOps: Cross-Lane Operations for Parallel Algorithms

`FloatTraits<T>` provides per-element operations (arithmetic, comparisons,
conditional blending) sufficient for fast_math. Dispenso's parallel algorithms
(reductions, scans, partitions) additionally need **cross-lane** operations.

Proposed `SimdOps<T>` trait alongside `FloatTraits<T>`:

```cpp
template <typename Flt>
struct SimdOps {
  // Runtime vector width (compile-time for SSE/AVX, runtime for SVE/RVV).
  static size_t width();

  // Horizontal operations.
  static float reduce_sum(Flt v);
  static float reduce_min(Flt v);
  static float reduce_max(Flt v);

  // Prefix sums (within a single SIMD register).
  static Flt inclusive_prefix_sum(Flt v);
  static Flt exclusive_prefix_sum(Flt v);

  // Lane permutations.
  static Flt shuffle(Flt v, IntType_t<Flt> indices);
  static Flt broadcast_lane(Flt v, int lane);
  static Flt rotate(Flt v, int amount);
};
```

These are **building blocks** for dispenso's task-parallel algorithms, not
full algorithms. The SIMD block operations compose with dispenso's existing
task scheduler:

1. **Phase 1** (parallel): Each task processes SIMD-width blocks using
   `SimdOps` (e.g. block prefix sum + block total).
2. **Phase 2** (sequential or recursive): Combine block totals across tasks.
3. **Phase 3** (parallel): Each task applies its offset to its block results.

This decomposition keeps SIMD building blocks pure and stateless, while
dispenso's `parallel_for` / `parallel_reduce` handles work distribution.

#### Widening Operations

Many reductions and scans require **widening accumulation** to avoid precision
loss or overflow:

- Sum of `float` buffer → `double` accumulator (catastrophic cancellation)
- Prefix sum of `int16_t` → `int32_t` output (overflow prevention)
- Histogram of `uint8_t` → `uint32_t` counts

This is a fundamental pattern, not a special case. The SIMD building blocks
must support it from the start:

```cpp
template <typename Flt>
struct SimdOps {
  using WideType = ...;  // SseFloat → SseDouble, NeonFloat → NeonDouble, etc.

  // Widening convert: N narrow elements → N wide elements (2 output vectors).
  static std::pair<WideType, WideType> widen(Flt v);

  // Widening reduce: accumulate narrow elements in wide precision.
  static scalar_t<WideType> reduce_sum_wide(Flt v);
};
```

Every architecture has native widening support: SSE `cvtps_pd`, NEON
`vcvt_f64_f32`, SVE `svcvt_f64_f32_x`, RVV `vfwcvt`. RVV's LMUL system
handles widening especially naturally — `vfwcvt.f.f.v` converts N floats to
N doubles using 2x the register group width with no loop splitting.
