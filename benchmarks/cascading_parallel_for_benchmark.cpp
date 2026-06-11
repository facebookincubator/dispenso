/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark demonstrating dispenso's cascading parallel_for advantage.
 *
 * Simulates multi-resolution image processing: a set of buffers at different
 * resolutions undergo independent operations of varying per-pixel cost, then
 * a dependent composite pass combines selected outputs. dispenso can overlap
 * the independent operations using ParForOptions{.wait = false}, while TBB
 * and OpenMP impose implicit barriers between parallel regions.
 *
 * Buffer layout (fraction of total pixels, relative cost per pixel):
 *   - Full-res gamma correction:   40%, cheap   (pow)
 *   - Full-res box blur:           40%, moderate (neighbor reads + averaging)
 *   - Half-res edge sharpen:       10%, expensive (Laplacian + clamp)
 *   - Quarter-res histogram EQ:     5%, moderate (log + scale)
 *   - Eighth-res threshold:         5%, cheap   (compare + blend)
 *   - Composite (dependent):      100%, cheap   (weighted sum)
 */

#include <dispenso/parallel_for.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_group.h"
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include "thread_benchmark_common.h"

// ---------------------------------------------------------------------------
// Image operations — each is a realistic per-pixel transform.
// ---------------------------------------------------------------------------

// Gamma correction: out = pow(in, 1/2.2). Cheap (~1 transcendental).
inline float gammaCorrect(float pixel) {
  return std::pow(std::max(pixel, 0.0f), 1.0f / 2.2f);
}

// 1D box blur: average of pixel and its neighbors. Moderate (3 reads + math).
inline float boxBlur(const float* row, size_t i, size_t width) {
  float left = (i > 0) ? row[i - 1] : row[i];
  float right = (i + 1 < width) ? row[i + 1] : row[i];
  return (left + row[i] + right) * (1.0f / 3.0f);
}

// Edge-aware sharpen: Laplacian-based unsharp mask. Expensive (5 reads + mul).
inline float edgeSharpen(const float* row, size_t i, size_t width) {
  float left = (i > 0) ? row[i - 1] : row[i];
  float right = (i + 1 < width) ? row[i + 1] : row[i];
  float laplacian = left + right - 2.0f * row[i];
  float sharpened = row[i] - 1.5f * laplacian;
  // Reinhard-style soft clamp to avoid ringing
  return sharpened / (1.0f + std::abs(sharpened));
}

// Histogram equalization approximation: log-scale remap. Moderate (log + mul).
inline float histogramEQ(float pixel) {
  return std::log1p(std::max(pixel, 0.0f) * 10.0f) * (1.0f / std::log(11.0f));
}

// Adaptive threshold: smooth step blend. Cheap (compare + lerp).
inline float adaptiveThreshold(float pixel, float threshold) {
  float t = (pixel - threshold + 0.1f) * 5.0f; // smoothstep input
  t = std::max(0.0f, std::min(1.0f, t));
  return t * t * (3.0f - 2.0f * t); // smoothstep
}

// ---------------------------------------------------------------------------
// Data setup
// ---------------------------------------------------------------------------

static constexpr int32_t kNumBuffers = 5;

struct ImageBuffers {
  // Input/output buffer pairs at different resolutions
  struct Buffer {
    std::vector<float> input;
    std::vector<float> output;
  };
  std::array<Buffer, kNumBuffers> buffers;
  std::vector<float> composite; // final output, full resolution
  int32_t fullResSize;
};

static ImageBuffers& getImageBuffers(int32_t totalPixels) {
  static std::unordered_map<int32_t, ImageBuffers> bufferMap;
  auto it = bufferMap.find(totalPixels);
  if (it != bufferMap.end()) {
    return it->second;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  ImageBuffers ib;
  ib.fullResSize = totalPixels;

  // Buffer sizes: full, full, half, quarter, eighth
  int32_t sizes[kNumBuffers] = {
      totalPixels,
      totalPixels,
      totalPixels / 2,
      totalPixels / 4,
      totalPixels / 8,
  };

  for (int32_t k = 0; k < kNumBuffers; ++k) {
    auto sz = static_cast<size_t>(std::max(sizes[k], 1));
    ib.buffers[static_cast<size_t>(k)].input.resize(sz);
    ib.buffers[static_cast<size_t>(k)].output.resize(sz);
    for (size_t i = 0; i < sz; ++i) {
      ib.buffers[static_cast<size_t>(k)].input[i] = dist(rng);
    }
  }
  ib.composite.resize(static_cast<size_t>(totalPixels), 0.0f);

  auto res = bufferMap.emplace(totalPixels, std::move(ib));
  return res.first->second;
}

// Process each buffer with its operation
template <typename ProcessFunc>
void processAllBuffers(ImageBuffers& ib, ProcessFunc processBuffer) {
  auto& b0 = ib.buffers[0];
  auto n0 = static_cast<int32_t>(b0.input.size());
  processBuffer(0, n0, [&b0](int32_t i) {
    b0.output[static_cast<size_t>(i)] = gammaCorrect(b0.input[static_cast<size_t>(i)]);
  });

  auto& b1 = ib.buffers[1];
  auto n1 = static_cast<int32_t>(b1.input.size());
  auto w1 = static_cast<size_t>(n1);
  processBuffer(0, n1, [&b1, w1](int32_t i) {
    b1.output[static_cast<size_t>(i)] = boxBlur(b1.input.data(), static_cast<size_t>(i), w1);
  });

  auto& b2 = ib.buffers[2];
  auto n2 = static_cast<int32_t>(b2.input.size());
  auto w2 = static_cast<size_t>(n2);
  processBuffer(0, n2, [&b2, w2](int32_t i) {
    b2.output[static_cast<size_t>(i)] = edgeSharpen(b2.input.data(), static_cast<size_t>(i), w2);
  });

  auto& b3 = ib.buffers[3];
  auto n3 = static_cast<int32_t>(b3.input.size());
  processBuffer(0, n3, [&b3](int32_t i) {
    b3.output[static_cast<size_t>(i)] = histogramEQ(b3.input[static_cast<size_t>(i)]);
  });

  auto& b4 = ib.buffers[4];
  auto n4 = static_cast<int32_t>(b4.input.size());
  processBuffer(0, n4, [&b4](int32_t i) {
    b4.output[static_cast<size_t>(i)] = adaptiveThreshold(b4.input[static_cast<size_t>(i)], 0.5f);
  });
}

// Composite: weighted sum, sampling from each buffer at appropriate resolution
inline void compositePixel(ImageBuffers& ib, int32_t i) {
  auto idx = static_cast<size_t>(i);
  float result = 0.0f;
  // Full-res buffers: direct index
  result += ib.buffers[0].output[idx] * 0.3f;
  result += ib.buffers[1].output[idx] * 0.3f;
  // Lower-res buffers: nearest-neighbor sample. Clamp the down-sampled index to
  // the (smaller) buffer's last element so integer division can never read past
  // the end for odd / non-power-of-two full-res sizes.
  auto sample = [](const std::vector<float>& buf, size_t s) {
    return buf[std::min(s, buf.size() - 1)];
  };
  result += sample(ib.buffers[2].output, idx / 2) * 0.2f;
  result += sample(ib.buffers[3].output, idx / 4) * 0.1f;
  result += sample(ib.buffers[4].output, idx / 8) * 0.1f;
  ib.composite[idx] = result;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

static constexpr int32_t kMediumSize = 100000;
static constexpr int32_t kLargeSize = 10000000;

void BM_serial(benchmark::State& state) {
  const int32_t totalPixels = state.range(0);
  auto& ib = getImageBuffers(totalPixels);

  for (auto UNUSED_VAR : state) {
    processAllBuffers(ib, [](int32_t begin, int32_t end, auto&& f) {
      for (int32_t i = begin; i < end; ++i) {
        f(i);
      }
    });
    for (int32_t i = 0; i < totalPixels; ++i) {
      compositePixel(ib, i);
    }
  }
}

void BM_dispenso_blocking(benchmark::State& state) {
  const int32_t numThreads = state.range(0) - 1;
  const int32_t totalPixels = state.range(1);
  auto& ib = getImageBuffers(totalPixels);

  dispenso::ThreadPool pool(numThreads);

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    processAllBuffers(ib, [&tasks](int32_t begin, int32_t end, auto&& f) {
      dispenso::parallel_for(tasks, begin, end, std::forward<decltype(f)>(f));
    });
    dispenso::parallel_for(tasks, 0, totalPixels, [&ib](int32_t i) { compositePixel(ib, i); });
  }
}

void BM_dispenso_cascaded(benchmark::State& state) {
  const int32_t numThreads = state.range(0) - 1;
  const int32_t totalPixels = state.range(1);
  auto& ib = getImageBuffers(totalPixels);

  dispenso::ThreadPool pool(numThreads);
  dispenso::ParForOptions noWait;
  noWait.wait = false;

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    // All image operations are independent — cascade without waiting.
    processAllBuffers(ib, [&tasks, &noWait](int32_t begin, int32_t end, auto&& f) {
      dispenso::parallel_for(tasks, begin, end, std::forward<decltype(f)>(f), noWait);
    });
    // Composite depends on all buffers — must wait.
    dispenso::parallel_for(tasks, 0, totalPixels, [&ib](int32_t i) { compositePixel(ib, i); });
  }
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int32_t numThreads = state.range(0);
  const int32_t totalPixels = state.range(1);
  auto& ib = getImageBuffers(totalPixels);

  omp_set_num_threads(numThreads);

  for (auto UNUSED_VAR : state) {
    processAllBuffers(ib, [](int32_t begin, int32_t end, auto&& f) {
#pragma omp parallel for schedule(static)
      for (int32_t i = begin; i < end; ++i) {
        f(i);
      }
    });
#pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < totalPixels; ++i) {
      compositePixel(ib, i);
    }
  }
}
#endif // _OPENMP

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int32_t numThreads = state.range(0);
  const int32_t totalPixels = state.range(1);
  auto& ib = getImageBuffers(totalPixels);

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(numThreads);
    processAllBuffers(ib, [](int32_t begin, int32_t end, auto&& f) {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(begin, end), [&f](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              f(i);
            }
          });
    });
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(0, totalPixels), [&ib](const tbb::blocked_range<int32_t>& r) {
          for (int32_t i = r.begin(); i != r.end(); ++i) {
            compositePixel(ib, i);
          }
        });
  }
}

void BM_tbb_task_group(benchmark::State& state) {
  const int32_t numThreads = state.range(0);
  const int32_t totalPixels = state.range(1);
  auto& ib = getImageBuffers(totalPixels);

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(numThreads);
    tbb::task_group tg;

    // Launch all buffer operations concurrently via task_group.
    // Each still has an internal barrier from tbb::parallel_for, but they
    // can overlap with each other.
    auto& b0 = ib.buffers[0];
    auto n0 = static_cast<int32_t>(b0.input.size());
    tg.run([&b0, n0]() {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, n0), [&b0](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              b0.output[static_cast<size_t>(i)] = gammaCorrect(b0.input[static_cast<size_t>(i)]);
            }
          });
    });

    auto& b1 = ib.buffers[1];
    auto n1 = static_cast<int32_t>(b1.input.size());
    auto w1 = static_cast<size_t>(n1);
    tg.run([&b1, n1, w1]() {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, n1), [&b1, w1](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              b1.output[static_cast<size_t>(i)] =
                  boxBlur(b1.input.data(), static_cast<size_t>(i), w1);
            }
          });
    });

    auto& b2 = ib.buffers[2];
    auto n2 = static_cast<int32_t>(b2.input.size());
    auto w2 = static_cast<size_t>(n2);
    tg.run([&b2, n2, w2]() {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, n2), [&b2, w2](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              b2.output[static_cast<size_t>(i)] =
                  edgeSharpen(b2.input.data(), static_cast<size_t>(i), w2);
            }
          });
    });

    auto& b3 = ib.buffers[3];
    auto n3 = static_cast<int32_t>(b3.input.size());
    tg.run([&b3, n3]() {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, n3), [&b3](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              b3.output[static_cast<size_t>(i)] = histogramEQ(b3.input[static_cast<size_t>(i)]);
            }
          });
    });

    auto& b4 = ib.buffers[4];
    auto n4 = static_cast<int32_t>(b4.input.size());
    tg.run([&b4, n4]() {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, n4), [&b4](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              b4.output[static_cast<size_t>(i)] =
                  adaptiveThreshold(b4.input[static_cast<size_t>(i)], 0.5f);
            }
          });
    });

    tg.wait();

    tbb::parallel_for(
        tbb::blocked_range<int32_t>(0, totalPixels), [&ib](const tbb::blocked_range<int32_t>& r) {
          for (int32_t i = r.begin(); i != r.end(); ++i) {
            compositePixel(ib, i);
          }
        });
  }
}
#endif // !BENCHMARK_WITHOUT_TBB

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kMediumSize, kLargeSize}) {
    for (int i : benchmarkThreadCounts()) {
      b->Args({i, j});
    }
  }
}

BENCHMARK(BM_serial)->Args({kMediumSize})->Args({kLargeSize})->UseRealTime();

#if defined(_OPENMP)
BENCHMARK(BM_omp)->Apply(CustomArguments)->UseRealTime();
#endif // OPENMP
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_tbb_task_group)->Apply(CustomArguments)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

BENCHMARK(BM_dispenso_blocking)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_dispenso_cascaded)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
