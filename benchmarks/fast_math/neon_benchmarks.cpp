/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "benchmark_helpers.h"

#if defined(__aarch64__)

namespace dfm = dispenso::fast_math;
namespace bench = dispenso::fast_math::bench;
using Flt = float32x4_t;

// --- One-arg benchmarks ---

void BM_sin_neon(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::sin(x); });
}
void BM_cos_neon(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::cos(x); });
}
void BM_tan_neon(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::tan(x); });
}
void BM_atan_neon(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::atan(x); });
}
void BM_acos_neon(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::acos(x); });
}
void BM_asin_neon(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::asin(x); });
}

void BM_exp_neon(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp(x); });
}
void BM_exp2_neon(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp2(x); });
}
void BM_exp10_neon(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp10(x); });
}
void BM_expm1_neon(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::expm1(x); });
}

void BM_log_neon(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log(x); });
}
void BM_log2_neon(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log2(x); });
}
void BM_log10_neon(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log10(x); });
}
void BM_log1p_neon(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](Flt x) { return dfm::log1p(vabsq_f32(x)); });
}

void BM_cbrt_neon(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::cbrt(x); });
}

void BM_frexp_neon(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    dfm::IntType_t<Flt> e;
    return dfm::frexp(x, &e);
  });
}
void BM_ldexp_neon(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return dfm::ldexp(x, vdupq_n_s32(3)); });
}

void BM_tanh_neon(benchmark::State& state) {
  bench::runBench(state, bench::tanhInputs<Flt>(), [](auto x) { return dfm::tanh(x); });
}
void BM_erf_neon(benchmark::State& state) {
  bench::runBench(state, bench::erfInputs<Flt>(), [](auto x) { return dfm::erf(x); });
}

// --- Two-arg benchmarks ---

void BM_atan2_neon(benchmark::State& state) {
  bench::runBench2(state, bench::expInputs<Flt>(), bench::sinInputs<Flt>(), [](auto y, auto x) {
    return dfm::atan2(y, x);
  });
}

void BM_hypot_neon(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot(x, y);
  });
}
void BM_hypot_neon_bounds(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot<Flt, dfm::MaxAccuracyTraits>(x, y);
  });
}

void BM_pow_neon(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow(b, e);
      });
}
void BM_pow_neon_accurate(benchmark::State& state) {
  bench::runBench2(
      state, bench::powBaseInputs<Flt>(), bench::powExpInputs<Flt>(), [](auto b, auto e) {
        return dfm::pow<Flt, dfm::MaxAccuracyTraits>(b, e);
      });
}
void BM_pow_neon_scalar_exp(benchmark::State& state) {
  bench::runBench(state, bench::powBaseInputs<Flt>(), [](auto x) { return dfm::pow(x, 2.5f); });
}

// --- Libc-packed comparison (NEON-specific, kept hand-written) ---

void BM_pow_libc_neon(benchmark::State& state) {
  const auto& bases = bench::powBaseInputs<Flt>();
  const auto& exps = bench::powExpInputs<Flt>();
  size_t idx = 0;
  Flt sum = vdupq_n_f32(0.0f);
  for (auto _ : state) {
    (void)_;
    alignas(16) float x[4], y[4], r[4];
    vst1q_f32(x, bases[idx]);
    vst1q_f32(y, exps[idx]);
    for (int32_t i = 0; i < 4; ++i) {
      r[i] = ::powf(x[i], y[i]);
    }
    sum = vaddq_f32(sum, vld1q_f32(r));
    idx = (idx + 1) & bench::kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  bench::consumeResult(sum);
}

// --- Registrations ---

BENCHMARK(BM_sin_neon);
BENCHMARK(BM_cos_neon);
BENCHMARK(BM_tan_neon);
BENCHMARK(BM_atan_neon);
BENCHMARK(BM_acos_neon);
BENCHMARK(BM_asin_neon);
BENCHMARK(BM_exp_neon);
BENCHMARK(BM_exp2_neon);
BENCHMARK(BM_exp10_neon);
BENCHMARK(BM_expm1_neon);
BENCHMARK(BM_log_neon);
BENCHMARK(BM_log2_neon);
BENCHMARK(BM_log10_neon);
BENCHMARK(BM_log1p_neon);
BENCHMARK(BM_cbrt_neon);
BENCHMARK(BM_frexp_neon);
BENCHMARK(BM_ldexp_neon);
BENCHMARK(BM_tanh_neon);
BENCHMARK(BM_erf_neon);
BENCHMARK(BM_atan2_neon);
BENCHMARK(BM_hypot_neon);
BENCHMARK(BM_hypot_neon_bounds);
BENCHMARK(BM_pow_neon);
BENCHMARK(BM_pow_neon_accurate);
BENCHMARK(BM_pow_neon_scalar_exp);
BENCHMARK(BM_pow_libc_neon);

#else // !defined(__aarch64__)

int main() {
  std::cout << "AArch64 NEON not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__aarch64__)
