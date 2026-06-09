/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA correctness tests for dispenso fast_math.
//
// Verifies that fast_math functions compiled and executed on CUDA GPUs meet
// the same accuracy contract as the CPU implementations. Ground truth uses
// double-precision reference computations (matching the CPU test approach)
// to avoid false failures from imprecise single-precision references.
//
// Tests exhaustively evaluate every representable float in the valid input
// domain, processed in chunks to stay within GPU memory limits. CPU ground
// truth is computed in parallel via dispenso::parallel_for.

#include <dispenso/fast_math/fast_math.h>
#include <dispenso/fast_math/util.h>
#include <dispenso/parallel_for.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace dfm = dispenso::fast_math;

// ---------------------------------------------------------------------------
// CUDA RAII helpers
// ---------------------------------------------------------------------------

struct CudaBuffer {
  float* ptr = nullptr;
  CudaBuffer() = default;
  explicit CudaBuffer(uint32_t n) {
    cudaMalloc(&ptr, n * sizeof(float));
  }
  ~CudaBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

// ---------------------------------------------------------------------------
// Generic evaluation kernel
// ---------------------------------------------------------------------------

template <typename Fn>
__global__ void evalKernel(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Fn fn;
    out[i] = fn(in[i]);
  }
}

template <typename Fn>
__global__ void eval2Kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Fn fn;
    out[i] = fn(a[i], b[i]);
  }
}

// ---------------------------------------------------------------------------
// Input domain helpers: ordered-integer representation for monotonic float
// iteration. Maps IEEE 754 floats to a uint32_t space where the natural
// integer order matches the float order, enabling simple iteration over
// every representable float in a range.
// ---------------------------------------------------------------------------

static uint32_t floatToOrdered(float f) {
  uint32_t u = dfm::bit_cast<uint32_t>(f);
  uint32_t mask = -static_cast<int32_t>(u >> 31) | 0x80000000u;
  return u ^ mask;
}

static float orderedToFloat(uint32_t u) {
  uint32_t mask = ((u >> 31) - 1) | 0x80000000u;
  return dfm::bit_cast<float>(u ^ mask);
}

static constexpr uint32_t kChunkSize = 1u << 24; // 16M floats per chunk

// ---------------------------------------------------------------------------
// Shared comparison helper — compares CPU vs GPU results, updating stats.
// ---------------------------------------------------------------------------

struct AccuracyStats {
  uint32_t maxUlp = 0;
  uint32_t mismatchCount = 0;
  uint64_t totalTested = 0;
};

template <typename PrintFn>
void compareFloatResults(
    const float* expected,
    const float* actual,
    uint32_t n,
    uint32_t maxAllowedUlps,
    AccuracyStats& stats,
    PrintFn printMismatch) {
  for (uint32_t i = 0; i < n; ++i) {
    if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
      if (std::isnan(expected[i]) && std::isnan(actual[i]))
        continue;
      if (expected[i] == actual[i])
        continue;
      ++stats.mismatchCount;
      continue;
    }
    uint32_t d = dfm::float_distance(expected[i], actual[i]);
    if (d > stats.maxUlp) {
      stats.maxUlp = d;
    }
    if (d > maxAllowedUlps) {
      if (stats.mismatchCount < 10) {
        printMismatch(i, d);
      }
      ++stats.mismatchCount;
    }
  }
  stats.totalTested += n;
}

// ---------------------------------------------------------------------------
// Exhaustive unary evaluation — tests every representable float in [lo, hi]
// ---------------------------------------------------------------------------
// Double-buffers CPU ground truth computation (via dispenso::parallel_for)
// against GPU kernel execution. While the GPU evaluates chunk N, the CPU
// computes ground truth for chunk N+1 and compares results from chunk N-1.

template <typename GpuFn>
::testing::AssertionResult
evalCudaAccuracyExhaustive(float (*cpuFn)(float), float lo, float hi, uint32_t maxAllowedUlps) {
  uint32_t ordLo = floatToOrdered(lo);
  uint32_t ordHi = floatToOrdered(hi);
  uint64_t totalCount = static_cast<uint64_t>(ordHi) - ordLo + 1;
  uint64_t numChunks = (totalCount + kChunkSize - 1) / kChunkSize;
  if (numChunks == 0)
    return ::testing::AssertionSuccess();

  std::vector<float> inputs[2], cpuRef[2], gpuOut[2];
  for (int b = 0; b < 2; ++b) {
    inputs[b].resize(kChunkSize);
    cpuRef[b].resize(kChunkSize);
    gpuOut[b].resize(kChunkSize);
  }

  CudaBuffer dIn0(kChunkSize), dOut0(kChunkSize);
  CudaBuffer dIn1(kChunkSize), dOut1(kChunkSize);
  float* dInPtrs[2] = {dIn0.ptr, dIn1.ptr};
  float* dOutPtrs[2] = {dOut0.ptr, dOut1.ptr};
  if (!dIn0.ptr || !dIn1.ptr)
    return ::testing::AssertionFailure() << "cudaMalloc failed";

  cudaStream_t streams[2] = {};
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  AccuracyStats stats;
  uint32_t counts[2] = {};

  auto prepareChunk = [&](int buf, uint64_t chunkIdx) -> uint32_t {
    uint64_t offset = chunkIdx * kChunkSize;
    uint32_t n =
        static_cast<uint32_t>(std::min(static_cast<uint64_t>(kChunkSize), totalCount - offset));
    for (uint32_t i = 0; i < n; ++i) {
      inputs[buf][i] = orderedToFloat(static_cast<uint32_t>(ordLo + offset + i));
    }
    float* in = inputs[buf].data();
    float* out = cpuRef[buf].data();
    dispenso::parallel_for(0u, n, [in, out, cpuFn](uint32_t i) { out[i] = cpuFn(in[i]); });
    return n;
  };

  auto launchGpu = [&](int buf, uint32_t n) {
    cudaMemcpyAsync(
        dInPtrs[buf], inputs[buf].data(), n * sizeof(float), cudaMemcpyHostToDevice, streams[buf]);
    constexpr uint32_t kBlockSize = 256;
    uint32_t gridSize = (n + kBlockSize - 1) / kBlockSize;
    evalKernel<GpuFn><<<gridSize, kBlockSize, 0, streams[buf]>>>(dInPtrs[buf], dOutPtrs[buf], n);
    cudaMemcpyAsync(
        gpuOut[buf].data(), dOutPtrs[buf], n * sizeof(float), cudaMemcpyDeviceToHost, streams[buf]);
  };

  auto compareChunk = [&](int buf, uint32_t n) {
    compareFloatResults(
        cpuRef[buf].data(),
        gpuOut[buf].data(),
        n,
        maxAllowedUlps,
        stats,
        [&](uint32_t i, uint32_t d) {
          printf(
              "  %u ULP: f(%.9g): cpu=%.9g gpu=%.9g\n",
              d,
              inputs[buf][i],
              cpuRef[buf][i],
              gpuOut[buf][i]);
        });
  };

  // Prolog: prepare chunk 0 and launch on GPU.
  counts[0] = prepareChunk(0, 0);
  launchGpu(0, counts[0]);

  // Steady state: prepare next chunk on CPU while GPU runs previous.
  for (uint64_t chunk = 1; chunk < numChunks; ++chunk) {
    int cur = static_cast<int>(chunk & 1);
    int prev = 1 - cur;

    counts[cur] = prepareChunk(cur, chunk);

    cudaStreamSynchronize(streams[prev]);
    compareChunk(prev, counts[prev]);

    launchGpu(cur, counts[cur]);
  }

  // Epilog: wait and compare last chunk.
  int last = static_cast<int>((numChunks - 1) & 1);
  cudaStreamSynchronize(streams[last]);
  compareChunk(last, counts[last]);

  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

  if (stats.maxUlp > maxAllowedUlps || stats.mismatchCount > 0) {
    return ::testing::AssertionFailure()
        << "max ULP = " << stats.maxUlp << " (allowed " << maxAllowedUlps << "), "
        << stats.mismatchCount << " mismatches, " << stats.totalTested
        << " values tested exhaustively";
  }
  return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Input generation for binary tests: bit-uniform sampling
// ---------------------------------------------------------------------------

// Sample representable floats uniformly in bit-space within [lo, hi].
static std::vector<float> generateInputs(float lo, float hi, uint32_t targetCount = 1000000) {
  std::vector<float> result;
  result.reserve(targetCount + 16);

  auto sampleRange = [&](uint32_t bitLo, uint32_t bitHi, uint32_t n, bool negate) {
    if (bitHi <= bitLo || n == 0)
      return;
    uint64_t range = static_cast<uint64_t>(bitHi) - bitLo;
    uint64_t step = std::max(range / n, uint64_t{1});
    for (uint64_t u = bitLo; u <= bitHi; u += step) {
      float f = dfm::bit_cast<float>(static_cast<uint32_t>(u));
      result.push_back(negate ? -f : f);
    }
  };

  if (lo >= 0.0f) {
    sampleRange(dfm::bit_cast<uint32_t>(lo), dfm::bit_cast<uint32_t>(hi), targetCount, false);
  } else if (hi <= 0.0f) {
    sampleRange(dfm::bit_cast<uint32_t>(-hi), dfm::bit_cast<uint32_t>(-lo), targetCount, true);
  } else {
    uint32_t negCount = targetCount / 2;
    uint32_t posCount = targetCount - negCount;
    sampleRange(0, dfm::bit_cast<uint32_t>(-lo), negCount, true);
    result.push_back(0.0f);
    sampleRange(0, dfm::bit_cast<uint32_t>(hi), posCount, false);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Host-side accuracy evaluation (binary) — sampled, not exhaustive
// ---------------------------------------------------------------------------

template <typename GpuFn>
::testing::AssertionResult evalCudaAccuracy2(
    float (*cpuFn)(float, float),
    const std::vector<float>& aInputs,
    const std::vector<float>& bInputs,
    uint32_t maxAllowedUlps) {
  uint32_t n = static_cast<uint32_t>(aInputs.size());
  if (n == 0)
    return ::testing::AssertionSuccess();

  std::vector<float> cpuResults(n);
  for (uint32_t i = 0; i < n; ++i) {
    cpuResults[i] = cpuFn(aInputs[i], bInputs[i]);
  }

  CudaBuffer dA(n), dB(n), dOut(n);
  if (!dA.ptr || !dB.ptr || !dOut.ptr) {
    return ::testing::AssertionFailure() << "cudaMalloc failed";
  }

  cudaError_t err;
  err = cudaMemcpy(dA.ptr, aInputs.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    return ::testing::AssertionFailure() << "cudaMemcpy: " << cudaGetErrorString(err);
  err = cudaMemcpy(dB.ptr, bInputs.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    return ::testing::AssertionFailure() << "cudaMemcpy: " << cudaGetErrorString(err);

  constexpr uint32_t kBlockSize = 256;
  uint32_t gridSize = (n + kBlockSize - 1) / kBlockSize;
  eval2Kernel<GpuFn><<<gridSize, kBlockSize>>>(dA.ptr, dB.ptr, dOut.ptr, n);
  err = cudaGetLastError();
  if (err != cudaSuccess)
    return ::testing::AssertionFailure() << "kernel launch: " << cudaGetErrorString(err);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    return ::testing::AssertionFailure() << "sync: " << cudaGetErrorString(err);

  std::vector<float> gpuResults(n);
  err = cudaMemcpy(gpuResults.data(), dOut.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    return ::testing::AssertionFailure() << "cudaMemcpy D2H: " << cudaGetErrorString(err);

  AccuracyStats stats;
  compareFloatResults(
      cpuResults.data(), gpuResults.data(), n, maxAllowedUlps, stats, [&](uint32_t i, uint32_t d) {
        printf(
            "  %u ULP: f(%.9g, %.9g): cpu=%.9g, gpu=%.9g\n",
            d,
            aInputs[i],
            bInputs[i],
            cpuResults[i],
            gpuResults[i]);
      });

  if (stats.maxUlp > maxAllowedUlps || stats.mismatchCount > 0) {
    return ::testing::AssertionFailure()
        << "max ULP = " << stats.maxUlp << " (allowed " << maxAllowedUlps << "), "
        << stats.mismatchCount << " mismatches, " << n << " samples tested";
  }
  return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Device functors — wrap each fast_math function for kernel invocation
// ---------------------------------------------------------------------------

#define DEFINE_UNARY_FUNCTOR(Name, expr)              \
  struct Name {                                       \
    DISPENSO_INLINE float operator()(float x) const { \
      return expr;                                    \
    }                                                 \
  }

#define DEFINE_BINARY_FUNCTOR(Name, expr)                      \
  struct Name {                                                \
    DISPENSO_INLINE float operator()(float a, float b) const { \
      return expr;                                             \
    }                                                          \
  }

// Unary functions.
DEFINE_UNARY_FUNCTOR(GpuExp, dfm::exp(x));
DEFINE_UNARY_FUNCTOR(GpuExp2, dfm::exp2(x));
DEFINE_UNARY_FUNCTOR(GpuExp10, dfm::exp10(x));
DEFINE_UNARY_FUNCTOR(GpuLog, dfm::log(x));
DEFINE_UNARY_FUNCTOR(GpuLog2, dfm::log2(x));
DEFINE_UNARY_FUNCTOR(GpuLog10, dfm::log10(x));
DEFINE_UNARY_FUNCTOR(GpuSin, dfm::sin(x));
DEFINE_UNARY_FUNCTOR(GpuCos, dfm::cos(x));
DEFINE_UNARY_FUNCTOR(GpuTan, dfm::tan(x));
DEFINE_UNARY_FUNCTOR(GpuAsin, dfm::asin(x));
DEFINE_UNARY_FUNCTOR(GpuAcos, dfm::acos(x));
DEFINE_UNARY_FUNCTOR(GpuAtan, dfm::atan(x));
DEFINE_UNARY_FUNCTOR(GpuSqrt, dfm::sqrt(x));
DEFINE_UNARY_FUNCTOR(GpuCbrt, dfm::cbrt(x));
DEFINE_UNARY_FUNCTOR(GpuErf, dfm::erf(x));
DEFINE_UNARY_FUNCTOR(GpuTanh, dfm::tanh(x));
DEFINE_UNARY_FUNCTOR(GpuRsqrt, dfm::rsqrt(x));
DEFINE_UNARY_FUNCTOR(GpuRcp, dfm::rcp(x));
DEFINE_UNARY_FUNCTOR(GpuSinpi, dfm::sinpi(x));

// Binary functions.
DEFINE_BINARY_FUNCTOR(GpuAtan2, dfm::atan2(a, b));
DEFINE_BINARY_FUNCTOR(GpuPow, dfm::pow(a, b));

// ---------------------------------------------------------------------------
// CPU ground-truth wrappers
// ---------------------------------------------------------------------------
// Reference values use double-precision computation where needed (matching
// the CPU test approach) so that ULP distances reflect the quality of our
// fast_math approximation, not the quality of the reference.

static float cpuExp(float x) {
  return std::exp(x);
}
static float cpuExp2(float x) {
  return std::exp2(x);
}
static float cpuExp10(float x) {
  return std::pow(10.0f, x);
}
static float cpuLog(float x) {
  return std::log(x);
}
static float cpuLog2(float x) {
  return std::log2(x);
}
static float cpuLog10(float x) {
  return std::log10(x);
}
static float cpuSin(float x) {
  return std::sin(x);
}
static float cpuCos(float x) {
  return std::cos(x);
}
static float cpuTan(float x) {
  return std::tan(x);
}
static float cpuAsin(float x) {
  return std::asin(x);
}
static float cpuAcos(float x) {
  return std::acos(x);
}
static float cpuAtan(float x) {
  return std::atan(x);
}
static float cpuSqrt(float x) {
  return std::sqrt(x);
}
static float cpuCbrt(float x) {
  return std::cbrt(x);
}
static float cpuErf(float x) {
  return std::erf(x);
}
static float cpuTanh(float x) {
  return static_cast<float>(std::tanh(static_cast<double>(x)));
}
static float cpuRsqrt(float x) {
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
}
static float cpuRcp(float x) {
  return 1.0f / x;
}
static float cpuSinpi(float x) {
  double xd = static_cast<double>(x);
  double j = std::round(2.0 * xd);
  double r = xd - j * 0.5;
  double t = r * M_PI;
  int qi = static_cast<int>(j) & 3;
  double sv = std::sin(t);
  double cv = std::cos(t);
  double result;
  switch (qi) {
    case 0:
      result = sv;
      break;
    case 1:
      result = cv;
      break;
    case 2:
      result = -sv;
      break;
    case 3:
      result = -cv;
      break;
    default:
      result = 0.0;
      break;
  }
  return static_cast<float>(result);
}
static float cpuAtan2(float y, float x) {
  return std::atan2(y, x);
}
static float cpuPow(float base, float exp) {
  return std::pow(base, exp);
}

// ---------------------------------------------------------------------------
// ULP thresholds match the CPU unit test budgets. With accurate double-
// precision ground truth, GPU fast_math meets the same budgets as CPU.
// ---------------------------------------------------------------------------

static constexpr float kPi = static_cast<float>(M_PI);

// ---- Exponential / logarithm ----

TEST(CudaFastMath, Exp) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuExp>(cpuExp, -88.0f, 88.0f, 3)));
}

TEST(CudaFastMath, Exp2) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuExp2>(cpuExp2, -126.0f, 127.0f, 1)));
}

TEST(CudaFastMath, Exp10) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuExp10>(cpuExp10, -38.0f, 38.0f, 2)));
}

TEST(CudaFastMath, Log) {
  EXPECT_TRUE(
      (evalCudaAccuracyExhaustive<GpuLog>(cpuLog, std::numeric_limits<float>::min(), 1e30f, 2)));
}

TEST(CudaFastMath, Log2) {
  EXPECT_TRUE(
      (evalCudaAccuracyExhaustive<GpuLog2>(cpuLog2, std::numeric_limits<float>::min(), 1e30f, 1)));
}

TEST(CudaFastMath, Log10) {
  EXPECT_TRUE((
      evalCudaAccuracyExhaustive<GpuLog10>(cpuLog10, std::numeric_limits<float>::min(), 1e30f, 3)));
}

// ---- Trigonometric ----

TEST(CudaFastMath, Sin) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuSin>(cpuSin, -128.0f * kPi, 128.0f * kPi, 1)));
}

TEST(CudaFastMath, Cos) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuCos>(cpuCos, -128.0f * kPi, 128.0f * kPi, 1)));
}

TEST(CudaFastMath, Tan) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuTan>(cpuTan, -128.0f * kPi, 128.0f * kPi, 3)));
}

TEST(CudaFastMath, Sinpi) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuSinpi>(cpuSinpi, -128.0f, 128.0f, 2)));
}

// ---- Inverse trigonometric ----

TEST(CudaFastMath, Asin) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuAsin>(cpuAsin, -1.0f, 1.0f, 3)));
}

TEST(CudaFastMath, Acos) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuAcos>(cpuAcos, -1.0f, 1.0f, 4)));
}

TEST(CudaFastMath, Atan) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuAtan>(cpuAtan, -100.0f, 100.0f, 3)));
}

// ---- Hyperbolic ----

TEST(CudaFastMath, Tanh) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuTanh>(cpuTanh, -10.0f, 10.0f, 2)));
}

// ---- Special functions ----

TEST(CudaFastMath, Erf) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuErf>(cpuErf, -4.0f, 4.0f, 2)));
}

// ---- Root / reciprocal ----

TEST(CudaFastMath, Sqrt) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuSqrt>(cpuSqrt, 0.0f, 1e30f, 1)));
}

TEST(CudaFastMath, Cbrt) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuCbrt>(cpuCbrt, -1e10f, 1e10f, 13)));
}

TEST(CudaFastMath, Rsqrt) {
  EXPECT_TRUE((
      evalCudaAccuracyExhaustive<GpuRsqrt>(cpuRsqrt, std::numeric_limits<float>::min(), 1e30f, 2)));
}

TEST(CudaFastMath, Rcp) {
  EXPECT_TRUE((evalCudaAccuracyExhaustive<GpuRcp>(cpuRcp, 0.001f, 1e30f, 1)));
}

// ---- Binary functions ----

TEST(CudaFastMath, Atan2) {
  // Generate pairs (y, x) covering all quadrants.
  std::vector<float> ys, xs;
  auto yRange = generateInputs(-100.0f, 100.0f, 250000);
  auto xRange = generateInputs(-100.0f, 100.0f, 250000);
  uint32_t n = std::min(yRange.size(), xRange.size());
  ys.assign(yRange.begin(), yRange.begin() + n);
  xs.assign(xRange.begin(), xRange.begin() + n);

  EXPECT_TRUE((evalCudaAccuracy2<GpuAtan2>(cpuAtan2, ys, xs, 3)));
}

TEST(CudaFastMath, Pow) {
  // Base in (0, 10], exponent in [-10, 10].
  std::vector<float> bases, exps;
  auto baseRange = generateInputs(0.001f, 10.0f, 250000);
  auto expRange = generateInputs(-10.0f, 10.0f, 250000);
  uint32_t n = std::min(baseRange.size(), expRange.size());
  bases.assign(baseRange.begin(), baseRange.begin() + n);
  exps.assign(expRange.begin(), expRange.begin() + n);

  EXPECT_TRUE((evalCudaAccuracy2<GpuPow>(cpuPow, bases, exps, 4)));
}
