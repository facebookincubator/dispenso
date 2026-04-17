/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "benchmark_helpers.h"

#if defined(__AVX2__)

namespace dfm = dispenso::fast_math;
namespace bench = dispenso::fast_math::bench;
using Flt = __m256;

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// --- One-arg benchmarks ---

void BM_sin_avx(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::sin(x); });
}
void BM_sin_avx_accurate(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) {
    return dfm::sin<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_cos_avx(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::cos(x); });
}
void BM_cos_avx_accurate(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) {
    return dfm::cos<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_tan_avx(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::tan(x); });
}
void BM_atan_avx(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::atan(x); });
}
void BM_acos_avx(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::acos(x); });
}
void BM_asin_avx(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::asin(x); });
}

void BM_exp_avx(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp(x); });
}
void BM_exp_avx_accurate(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) {
    return dfm::exp<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_exp_avx_bounds(benchmark::State& state) {
  bench::runBench(
      state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp<Flt, BoundsTraits>(x); });
}
void BM_exp2_avx(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp2(x); });
}
void BM_exp10_avx(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp10(x); });
}
void BM_expm1_avx(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::expm1(x); });
}

void BM_log_avx(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log(x); });
}
void BM_log_avx_accurate(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    return dfm::log<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_log2_avx(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log2(x); });
}
void BM_log10_avx(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log10(x); });
}
void BM_log1p_avx(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](Flt x) {
    Flt ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    return dfm::log1p(ax);
  });
}

void BM_cbrt_avx(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::cbrt(x); });
}
void BM_cbrt_avx_accurate(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    return dfm::cbrt<Flt, dfm::MaxAccuracyTraits>(x);
  });
}

void BM_frexp_avx(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    dfm::IntType_t<Flt> e;
    return dfm::frexp(x, &e);
  });
}
void BM_ldexp_avx(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return dfm::ldexp(x, _mm256_set1_epi32(3)); });
}

void BM_tanh_avx(benchmark::State& state) {
  bench::runBench(state, bench::tanhInputs<Flt>(), [](auto x) { return dfm::tanh(x); });
}
void BM_erf_avx(benchmark::State& state) {
  bench::runBench(state, bench::erfInputs<Flt>(), [](auto x) { return dfm::erf(x); });
}

// --- Two-arg benchmarks ---

void BM_atan2_avx(benchmark::State& state) {
  bench::runBench2(state, bench::expInputs<Flt>(), bench::sinInputs<Flt>(), [](auto y, auto x) {
    return dfm::atan2(y, x);
  });
}

void BM_hypot_avx(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot(x, y);
  });
}
void BM_hypot_avx_bounds(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot<Flt, dfm::MaxAccuracyTraits>(x, y);
  });
}

void BM_pow_avx(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow(b, e);
      });
}
void BM_pow_avx_accurate(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow<Flt, dfm::MaxAccuracyTraits>(b, e);
      });
}
void BM_pow_avx_scalar_exp(benchmark::State& state) {
  bench::runBench(state, bench::powBaseInputs<Flt>(), [](auto x) { return dfm::pow(x, 2.5f); });
}

// --- Libc-packed comparison (AVX-specific, kept hand-written) ---

void BM_pow_libc_avx(benchmark::State& state) {
  const auto& bases = bench::powBaseInputs<Flt>();
  const auto& exps = bench::powExpInputs<Flt>();
  size_t idx = 0;
  Flt sum = _mm256_setzero_ps();
  for (auto _ : state) {
    (void)_;
    alignas(32) float x[8], y[8], r[8];
    _mm256_store_ps(x, bases[idx]);
    _mm256_store_ps(y, exps[idx]);
    for (int32_t i = 0; i < 8; ++i) {
      r[i] = ::powf(x[i], y[i]);
    }
    sum = _mm256_add_ps(sum, _mm256_load_ps(r));
    idx = (idx + 1) & bench::kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  bench::consumeResult(sum);
}

// --- Registrations ---

BENCHMARK(BM_sin_avx);
BENCHMARK(BM_sin_avx_accurate);
BENCHMARK(BM_cos_avx);
BENCHMARK(BM_cos_avx_accurate);
BENCHMARK(BM_tan_avx);
BENCHMARK(BM_atan_avx);
BENCHMARK(BM_acos_avx);
BENCHMARK(BM_asin_avx);
BENCHMARK(BM_exp_avx);
BENCHMARK(BM_exp_avx_accurate);
BENCHMARK(BM_exp_avx_bounds);
BENCHMARK(BM_exp2_avx);
BENCHMARK(BM_exp10_avx);
BENCHMARK(BM_expm1_avx);
BENCHMARK(BM_log_avx);
BENCHMARK(BM_log_avx_accurate);
BENCHMARK(BM_log2_avx);
BENCHMARK(BM_log10_avx);
BENCHMARK(BM_log1p_avx);
BENCHMARK(BM_cbrt_avx);
BENCHMARK(BM_cbrt_avx_accurate);
BENCHMARK(BM_frexp_avx);
BENCHMARK(BM_ldexp_avx);
BENCHMARK(BM_tanh_avx);
BENCHMARK(BM_erf_avx);
BENCHMARK(BM_atan2_avx);
BENCHMARK(BM_hypot_avx);
BENCHMARK(BM_hypot_avx_bounds);
BENCHMARK(BM_pow_avx);
BENCHMARK(BM_pow_avx_accurate);
BENCHMARK(BM_pow_avx_scalar_exp);
BENCHMARK(BM_pow_libc_avx);

#else // !defined(__AVX2__)

int main() {
  std::cout << "AVX2 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__AVX2__)
