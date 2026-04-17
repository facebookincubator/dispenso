/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "benchmark_helpers.h"

#if defined(__AVX512F__)

namespace dfm = dispenso::fast_math;
namespace bench = dispenso::fast_math::bench;
using Flt = __m512;

// --- One-arg benchmarks ---

void BM_sin_avx512(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::sin(x); });
}
void BM_cos_avx512(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::cos(x); });
}
void BM_tan_avx512(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::tan(x); });
}
void BM_atan_avx512(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::atan(x); });
}
void BM_acos_avx512(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::acos(x); });
}
void BM_asin_avx512(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::asin(x); });
}

void BM_exp_avx512(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp(x); });
}
void BM_exp2_avx512(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp2(x); });
}
void BM_exp10_avx512(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp10(x); });
}
void BM_expm1_avx512(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::expm1(x); });
}

void BM_log_avx512(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log(x); });
}
void BM_log2_avx512(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log2(x); });
}
void BM_log10_avx512(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log10(x); });
}
void BM_log1p_avx512(benchmark::State& state) {
  bench::runBench(
      state, bench::sinInputs<Flt>(), [](Flt x) { return dfm::log1p(_mm512_abs_ps(x)); });
}

void BM_cbrt_avx512(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::cbrt(x); });
}

void BM_frexp_avx512(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    dfm::IntType_t<Flt> e;
    return dfm::frexp(x, &e);
  });
}
void BM_ldexp_avx512(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return dfm::ldexp(x, _mm512_set1_epi32(3)); });
}

void BM_tanh_avx512(benchmark::State& state) {
  bench::runBench(state, bench::tanhInputs<Flt>(), [](auto x) { return dfm::tanh(x); });
}
void BM_erf_avx512(benchmark::State& state) {
  bench::runBench(state, bench::erfInputs<Flt>(), [](auto x) { return dfm::erf(x); });
}

// --- Two-arg benchmarks ---

void BM_atan2_avx512(benchmark::State& state) {
  bench::runBench2(state, bench::expInputs<Flt>(), bench::sinInputs<Flt>(), [](auto y, auto x) {
    return dfm::atan2(y, x);
  });
}

void BM_hypot_avx512(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot(x, y);
  });
}
void BM_hypot_avx512_bounds(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot<Flt, dfm::MaxAccuracyTraits>(x, y);
  });
}

void BM_pow_avx512(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow(b, e);
      });
}
void BM_pow_avx512_accurate(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow<Flt, dfm::MaxAccuracyTraits>(b, e);
      });
}
void BM_pow_avx512_scalar_exp(benchmark::State& state) {
  bench::runBench(state, bench::powBaseInputs<Flt>(), [](auto x) { return dfm::pow(x, 2.5f); });
}

// --- Libc-packed comparisons (AVX-512-specific, kept hand-written) ---

void BM_hypot_libc_avx512(benchmark::State& state) {
  const auto& inputs = bench::hypotInputs<Flt>();
  const auto& inputs2 = bench::sinInputs<Flt>();
  size_t idx = 0;
  Flt sum = _mm512_setzero_ps();
  for (auto _ : state) {
    (void)_;
    alignas(64) float x[16], y[16], r[16];
    _mm512_store_ps(x, inputs[idx]);
    _mm512_store_ps(y, inputs2[idx]);
    for (int32_t i = 0; i < 16; ++i) {
      r[i] = ::hypotf(x[i], y[i]);
    }
    sum = _mm512_add_ps(sum, _mm512_load_ps(r));
    idx = (idx + 1) & bench::kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  bench::consumeResult(sum);
}

void BM_pow_libc_avx512(benchmark::State& state) {
  const auto& bases = bench::powBaseInputs<Flt>();
  const auto& exps = bench::powExpInputs<Flt>();
  size_t idx = 0;
  Flt sum = _mm512_setzero_ps();
  for (auto _ : state) {
    (void)_;
    alignas(64) float x[16], y[16], r[16];
    _mm512_store_ps(x, bases[idx]);
    _mm512_store_ps(y, exps[idx]);
    for (int32_t j = 0; j < 16; ++j)
      r[j] = ::powf(x[j], y[j]);
    sum = _mm512_add_ps(sum, _mm512_load_ps(r));
    idx = (idx + 1) & bench::kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  bench::consumeResult(sum);
}

// --- Registrations ---

BENCHMARK(BM_sin_avx512);
BENCHMARK(BM_cos_avx512);
BENCHMARK(BM_tan_avx512);
BENCHMARK(BM_atan_avx512);
BENCHMARK(BM_acos_avx512);
BENCHMARK(BM_asin_avx512);
BENCHMARK(BM_exp_avx512);
BENCHMARK(BM_exp2_avx512);
BENCHMARK(BM_exp10_avx512);
BENCHMARK(BM_expm1_avx512);
BENCHMARK(BM_log_avx512);
BENCHMARK(BM_log2_avx512);
BENCHMARK(BM_log10_avx512);
BENCHMARK(BM_log1p_avx512);
BENCHMARK(BM_cbrt_avx512);
BENCHMARK(BM_frexp_avx512);
BENCHMARK(BM_ldexp_avx512);
BENCHMARK(BM_tanh_avx512);
BENCHMARK(BM_erf_avx512);
BENCHMARK(BM_atan2_avx512);
BENCHMARK(BM_hypot_avx512);
BENCHMARK(BM_hypot_avx512_bounds);
BENCHMARK(BM_hypot_libc_avx512);
BENCHMARK(BM_pow_avx512);
BENCHMARK(BM_pow_avx512_accurate);
BENCHMARK(BM_pow_avx512_scalar_exp);
BENCHMARK(BM_pow_libc_avx512);

#else // !defined(__AVX512F__)

int main() {
  std::cout << "AVX-512 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__AVX512F__)
