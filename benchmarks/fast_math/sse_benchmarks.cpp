/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "benchmark_helpers.h"

#if defined(__SSE4_1__)

namespace dfm = dispenso::fast_math;
namespace bench = dispenso::fast_math::bench;
using Flt = __m128;

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// --- One-arg benchmarks ---

void BM_sin_sse(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::sin(x); });
}
void BM_sin_sse_accurate(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) {
    return dfm::sin<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_cos_sse(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::cos(x); });
}
void BM_cos_sse_accurate(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) {
    return dfm::cos<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_tan_sse(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::tan(x); });
}
void BM_atan_sse(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::atan(x); });
}
void BM_acos_sse(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::acos(x); });
}
void BM_asin_sse(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::asin(x); });
}

void BM_exp_sse(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp(x); });
}
void BM_exp_sse_accurate(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) {
    return dfm::exp<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_exp_sse_bounds(benchmark::State& state) {
  bench::runBench(
      state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp<Flt, BoundsTraits>(x); });
}
void BM_exp2_sse(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp2(x); });
}
void BM_exp10_sse(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp10(x); });
}
void BM_expm1_sse(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::expm1(x); });
}

void BM_log_sse(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log(x); });
}
void BM_log_sse_accurate(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    return dfm::log<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_log2_sse(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log2(x); });
}
void BM_log10_sse(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log10(x); });
}
void BM_log1p_sse(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](Flt x) {
    Flt ax = _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
    return dfm::log1p(ax);
  });
}

void BM_cbrt_sse(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::cbrt(x); });
}
void BM_cbrt_sse_accurate(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    return dfm::cbrt<Flt, dfm::MaxAccuracyTraits>(x);
  });
}

void BM_frexp_sse(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    dfm::IntType_t<Flt> e;
    return dfm::frexp(x, &e);
  });
}
void BM_ldexp_sse(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return dfm::ldexp(x, _mm_set1_epi32(3)); });
}

void BM_tanh_sse(benchmark::State& state) {
  bench::runBench(state, bench::tanhInputs<Flt>(), [](auto x) { return dfm::tanh(x); });
}
void BM_erf_sse(benchmark::State& state) {
  bench::runBench(state, bench::erfInputs<Flt>(), [](auto x) { return dfm::erf(x); });
}

// --- Two-arg benchmarks ---

void BM_atan2_sse(benchmark::State& state) {
  bench::runBench2(state, bench::expInputs<Flt>(), bench::sinInputs<Flt>(), [](auto y, auto x) {
    return dfm::atan2(y, x);
  });
}

void BM_hypot_sse(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot(x, y);
  });
}
void BM_hypot_sse_bounds(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot<Flt, dfm::MaxAccuracyTraits>(x, y);
  });
}

void BM_pow_sse(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow(b, e);
      });
}
void BM_pow_sse_accurate(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow<Flt, dfm::MaxAccuracyTraits>(b, e);
      });
}
void BM_pow_sse_scalar_exp(benchmark::State& state) {
  bench::runBench(state, bench::powBaseInputs<Flt>(), [](auto x) { return dfm::pow(x, 2.5f); });
}

// --- Libc-packed comparisons (SSE-specific, kept hand-written) ---

void BM_hypot_libc(benchmark::State& state) {
  const auto& inputs = bench::hypotInputs<Flt>();
  const auto& inputs2 = bench::sinInputs<Flt>();
  size_t idx = 0;
  Flt sum = _mm_setzero_ps();
  for (auto _ : state) {
    (void)_;
    alignas(16) float x[4], y[4], r[4];
    _mm_store_ps(x, inputs[idx]);
    _mm_store_ps(y, inputs2[idx]);
    r[0] = ::hypotf(x[0], y[0]);
    r[1] = ::hypotf(x[1], y[1]);
    r[2] = ::hypotf(x[2], y[2]);
    r[3] = ::hypotf(x[3], y[3]);
    sum = _mm_add_ps(sum, _mm_load_ps(r));
    idx = (idx + 1) & bench::kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  bench::consumeResult(sum);
}

void BM_pow_libc_sse(benchmark::State& state) {
  const auto& bases = bench::powBaseInputs<Flt>();
  const auto& exps = bench::powExpInputs<Flt>();
  size_t idx = 0;
  Flt sum = _mm_setzero_ps();
  for (auto _ : state) {
    (void)_;
    alignas(16) float x[4], y[4], r[4];
    _mm_store_ps(x, bases[idx]);
    _mm_store_ps(y, exps[idx]);
    r[0] = ::powf(x[0], y[0]);
    r[1] = ::powf(x[1], y[1]);
    r[2] = ::powf(x[2], y[2]);
    r[3] = ::powf(x[3], y[3]);
    sum = _mm_add_ps(sum, _mm_load_ps(r));
    idx = (idx + 1) & bench::kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  bench::consumeResult(sum);
}

// --- Registrations ---

BENCHMARK(BM_sin_sse);
BENCHMARK(BM_sin_sse_accurate);
BENCHMARK(BM_cos_sse);
BENCHMARK(BM_cos_sse_accurate);
BENCHMARK(BM_tan_sse);
BENCHMARK(BM_atan_sse);
BENCHMARK(BM_acos_sse);
BENCHMARK(BM_asin_sse);
BENCHMARK(BM_exp_sse);
BENCHMARK(BM_exp_sse_accurate);
BENCHMARK(BM_exp_sse_bounds);
BENCHMARK(BM_exp2_sse);
BENCHMARK(BM_exp10_sse);
BENCHMARK(BM_expm1_sse);
BENCHMARK(BM_log_sse);
BENCHMARK(BM_log_sse_accurate);
BENCHMARK(BM_log2_sse);
BENCHMARK(BM_log10_sse);
BENCHMARK(BM_log1p_sse);
BENCHMARK(BM_cbrt_sse);
BENCHMARK(BM_cbrt_sse_accurate);
BENCHMARK(BM_frexp_sse);
BENCHMARK(BM_ldexp_sse);
BENCHMARK(BM_tanh_sse);
BENCHMARK(BM_erf_sse);
BENCHMARK(BM_atan2_sse);
BENCHMARK(BM_hypot_sse);
BENCHMARK(BM_hypot_sse_bounds);
BENCHMARK(BM_hypot_libc);
BENCHMARK(BM_pow_sse);
BENCHMARK(BM_pow_sse_accurate);
BENCHMARK(BM_pow_sse_scalar_exp);
BENCHMARK(BM_pow_libc_sse);

#else // !defined(__SSE4_1__)

int main() {
  std::cout << "SSE4.1 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__SSE4_1__)
