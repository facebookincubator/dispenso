/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "benchmark_helpers.h"

#if __has_include("hwy/highway.h")
#include "hwy/contrib/math/math-inl.h"

namespace dfm = dispenso::fast_math;
namespace bench = dispenso::fast_math::bench;
namespace hn = hwy::HWY_NAMESPACE;
using Flt = dfm::HwyFloat;
using HwyFloatTag = dfm::HwyFloatTag;

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// --- One-arg benchmarks ---

void BM_sin_hwy(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::sin(x); });
}
void BM_sin_hwy_accurate(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) {
    return dfm::sin<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_cos_hwy(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::cos(x); });
}
void BM_cos_hwy_accurate(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) {
    return dfm::cos<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_tan_hwy(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::tan(x); });
}
void BM_atan_hwy(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::atan(x); });
}
void BM_acos_hwy(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::acos(x); });
}
void BM_asin_hwy(benchmark::State& state) {
  bench::runBench(state, bench::acosInputs<Flt>(), [](auto x) { return dfm::asin(x); });
}

void BM_exp_hwy(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp(x); });
}
void BM_exp_hwy_accurate(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) {
    return dfm::exp<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_exp_hwy_bounds(benchmark::State& state) {
  bench::runBench(
      state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp<Flt, BoundsTraits>(x); });
}
void BM_exp2_hwy(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp2(x); });
}
void BM_exp10_hwy(benchmark::State& state) {
  bench::runBench(state, bench::expInputs<Flt>(), [](auto x) { return dfm::exp10(x); });
}
void BM_expm1_hwy(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::expm1(x); });
}

void BM_log_hwy(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log(x); });
}
void BM_log_hwy_accurate(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    return dfm::log<Flt, dfm::MaxAccuracyTraits>(x);
  });
}
void BM_log2_hwy(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log2(x); });
}
void BM_log10_hwy(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::log10(x); });
}
void BM_log1p_hwy(benchmark::State& state) {
  bench::runBench(state, bench::sinInputs<Flt>(), [](auto x) { return dfm::log1p(hn::Abs(x.v)); });
}

void BM_cbrt_hwy(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) { return dfm::cbrt(x); });
}
void BM_cbrt_hwy_accurate(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    return dfm::cbrt<Flt, dfm::MaxAccuracyTraits>(x);
  });
}

void BM_frexp_hwy(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    dfm::IntType_t<Flt> e;
    return dfm::frexp(x, &e);
  });
}
void BM_ldexp_hwy(benchmark::State& state) {
  bench::runBench(state, bench::logInputs<Flt>(), [](auto x) {
    const hn::RebindToSigned<HwyFloatTag> di;
    return dfm::ldexp(x, hn::Set(di, 3));
  });
}

void BM_tanh_hwy(benchmark::State& state) {
  bench::runBench(state, bench::tanhInputs<Flt>(), [](auto x) { return dfm::tanh(x); });
}
void BM_erf_hwy(benchmark::State& state) {
  bench::runBench(state, bench::erfInputs<Flt>(), [](auto x) { return dfm::erf(x); });
}

// --- Two-arg benchmarks ---

void BM_atan2_hwy(benchmark::State& state) {
  bench::runBench2(state, bench::expInputs<Flt>(), bench::sinInputs<Flt>(), [](auto y, auto x) {
    return dfm::atan2(y, x);
  });
}

void BM_hypot_hwy(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot(x, y);
  });
}
void BM_hypot_hwy_bounds(benchmark::State& state) {
  bench::runBench2(state, bench::hypotInputs<Flt>(), bench::sinInputs<Flt>(), [](auto x, auto y) {
    return dfm::hypot<Flt, dfm::MaxAccuracyTraits>(x, y);
  });
}

// --- Highway contrib/math comparison benchmarks ---

void BM_sin_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::sinInputs<Flt>(), [](auto x) { return hn::Sin(HwyFloatTag(), x.v); });
}
void BM_cos_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::sinInputs<Flt>(), [](auto x) { return hn::Cos(HwyFloatTag(), x.v); });
}
void BM_exp_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::expInputs<Flt>(), [](auto x) { return hn::Exp(HwyFloatTag(), x.v); });
}
void BM_exp2_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::expInputs<Flt>(), [](auto x) { return hn::Exp2(HwyFloatTag(), x.v); });
}
void BM_log_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return hn::Log(HwyFloatTag(), x.v); });
}
void BM_log2_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return hn::Log2(HwyFloatTag(), x.v); });
}
void BM_log10_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::logInputs<Flt>(), [](auto x) { return hn::Log10(HwyFloatTag(), x.v); });
}
void BM_atan_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::sinInputs<Flt>(), [](auto x) { return hn::Atan(HwyFloatTag(), x.v); });
}
void BM_acos_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::acosInputs<Flt>(), [](auto x) { return hn::Acos(HwyFloatTag(), x.v); });
}
void BM_asin_hwy_contrib(benchmark::State& state) {
  bench::runBench(
      state, bench::acosInputs<Flt>(), [](auto x) { return hn::Asin(HwyFloatTag(), x.v); });
}
void BM_atan2_hwy_contrib(benchmark::State& state) {
  bench::runBench2(state, bench::expInputs<Flt>(), bench::sinInputs<Flt>(), [](auto y, auto x) {
    return hn::Atan2(HwyFloatTag(), y.v, x.v);
  });
}

// --- Registrations ---

BENCHMARK(BM_sin_hwy);
BENCHMARK(BM_sin_hwy_accurate);
BENCHMARK(BM_sin_hwy_contrib);
BENCHMARK(BM_cos_hwy);
BENCHMARK(BM_cos_hwy_accurate);
BENCHMARK(BM_cos_hwy_contrib);
BENCHMARK(BM_tan_hwy);
BENCHMARK(BM_atan_hwy);
BENCHMARK(BM_atan_hwy_contrib);
BENCHMARK(BM_acos_hwy);
BENCHMARK(BM_acos_hwy_contrib);
BENCHMARK(BM_asin_hwy);
BENCHMARK(BM_asin_hwy_contrib);
BENCHMARK(BM_exp_hwy);
BENCHMARK(BM_exp_hwy_accurate);
BENCHMARK(BM_exp_hwy_bounds);
BENCHMARK(BM_exp_hwy_contrib);
BENCHMARK(BM_exp2_hwy);
BENCHMARK(BM_exp2_hwy_contrib);
BENCHMARK(BM_exp10_hwy);
BENCHMARK(BM_expm1_hwy);
BENCHMARK(BM_log_hwy);
BENCHMARK(BM_log_hwy_accurate);
BENCHMARK(BM_log_hwy_contrib);
BENCHMARK(BM_log2_hwy);
BENCHMARK(BM_log2_hwy_contrib);
BENCHMARK(BM_log10_hwy);
BENCHMARK(BM_log10_hwy_contrib);
BENCHMARK(BM_log1p_hwy);
BENCHMARK(BM_cbrt_hwy);
BENCHMARK(BM_cbrt_hwy_accurate);
BENCHMARK(BM_frexp_hwy);
BENCHMARK(BM_ldexp_hwy);
BENCHMARK(BM_tanh_hwy);
BENCHMARK(BM_erf_hwy);
BENCHMARK(BM_atan2_hwy);
BENCHMARK(BM_atan2_hwy_contrib);
BENCHMARK(BM_hypot_hwy);
BENCHMARK(BM_hypot_hwy_bounds);

#else // !__has_include("hwy/highway.h")

int main() {
  std::cout << "Highway not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // __has_include("hwy/highway.h")
