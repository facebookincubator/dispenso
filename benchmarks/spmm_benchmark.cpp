/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Sparse matrix-dense matrix multiply (SpMM) benchmark: C = A * B where A
// is a CSR-format sparse matrix with a power-law row distribution and B is
// a dense column-major matrix with kRhsCols columns.
//
// Row nnz follows nnz_i = meanNnz * u^(-1.5), u ~ Uniform(0.01, 1).
// With meanNnz=2 this gives: median ~6, 75th ~16, 95th ~179, top 1.5%
// at the 512 cap. Overall density is <1% — realistic for CSR without
// suggesting a specialised format (ELL+COO, HYB).
//
// Each row's cost is proportional to nnz * kRhsCols, so the per-row
// variance (1-100x+) creates strong load imbalance under static
// partitioning. Adaptive scheduling re-balances by letting idle workers
// steal heavy-row work from busy ones.

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <dispenso/parallel_for.h>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb_compat.h"
#endif

#include "thread_benchmark_common.h"

namespace {

struct CsrMatrix {
  std::vector<int32_t> rowPtr; // size = numRows + 1
  std::vector<int32_t> colIdx; // size = totalNnz
  std::vector<double> values; // size = totalNnz
  int32_t numRows;
  int32_t numCols;
};

CsrMatrix generateCsr(int32_t numRows, int32_t numCols, double meanNnz, int32_t maxNnz) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::uniform_int_distribution<int32_t> colDist(0, numCols - 1);

  CsrMatrix m;
  m.numRows = numRows;
  m.numCols = numCols;
  m.rowPtr.resize(static_cast<size_t>(numRows) + 1);

  // Power-law: nnz_i = meanNnz * u^(-1.5), u ~ Uniform(0.01, 1).
  m.rowPtr[0] = 0;
  for (int32_t r = 0; r < numRows; ++r) {
    double u = std::max(0.01, uni(rng));
    double nnzD = meanNnz * std::pow(u, -1.5);
    int32_t nnz = static_cast<int32_t>(std::min(nnzD, static_cast<double>(maxNnz)));
    m.rowPtr[static_cast<size_t>(r + 1)] = m.rowPtr[static_cast<size_t>(r)] + nnz;
  }
  int32_t totalNnz = m.rowPtr.back();
  m.colIdx.resize(static_cast<size_t>(totalNnz));
  m.values.resize(static_cast<size_t>(totalNnz));
  for (int32_t i = 0; i < totalNnz; ++i) {
    m.colIdx[static_cast<size_t>(i)] = colDist(rng);
    m.values[static_cast<size_t>(i)] = uni(rng) - 0.5;
  }
  return m;
}

const CsrMatrix& getMatrix(int32_t numRows, double meanNnz) {
  struct Key {
    int32_t numRows;
    int32_t meanNnzMilli;
  };
  static std::vector<std::pair<Key, CsrMatrix>> cache;
  int32_t mm = static_cast<int32_t>(meanNnz * 1000);
  for (auto& kv : cache) {
    if (kv.first.numRows == numRows && kv.first.meanNnzMilli == mm) {
      return kv.second;
    }
  }
  cache.push_back({{numRows, mm}, generateCsr(numRows, numRows, meanNnz, /*maxNnz=*/512)});
  return cache.back().second;
}

// Multiply one row of A against all columns of B (row-major, numCols x kRhsCols).
// Writes one row of C (row-major, numRows x kRhsCols).
template <int32_t kRhsCols>
inline void mulRow(const CsrMatrix& m, const double* B, double* C, int32_t r) {
  double* cRow = C + static_cast<size_t>(r) * kRhsCols;
  for (int32_t k = 0; k < kRhsCols; ++k) {
    cRow[k] = 0.0;
  }
  int32_t i = m.rowPtr[static_cast<size_t>(r)];
  int32_t end = m.rowPtr[static_cast<size_t>(r + 1)];
  for (; i < end; ++i) {
    double aVal = m.values[static_cast<size_t>(i)];
    int32_t col = m.colIdx[static_cast<size_t>(i)];
    const double* bRow = B + static_cast<size_t>(col) * kRhsCols;
    for (int32_t k = 0; k < kRhsCols; ++k) {
      cRow[k] += aVal * bRow[k];
    }
  }
}

} // namespace

static constexpr int32_t kMedRows = 64 * 1024;
static constexpr int32_t kLargeRows = 1024 * 1024;
static constexpr double kMeanNnz = 2.0;
static constexpr int32_t kRhsCols = 64;

// Serial baseline.
template <int32_t kRows>
void BM_serial(benchmark::State& state) {
  const CsrMatrix& m = getMatrix(kRows, kMeanNnz);
  std::vector<double> B(static_cast<size_t>(m.numCols) * kRhsCols, 1.0);
  std::vector<double> C(static_cast<size_t>(m.numRows) * kRhsCols, 0.0);
  for (auto UNUSED_VAR : state) {
    for (int32_t r = 0; r < m.numRows; ++r) {
      mulRow<kRhsCols>(m, B.data(), C.data(), r);
    }
    benchmark::DoNotOptimize(C);
  }
}
BENCHMARK_TEMPLATE(BM_serial, kMedRows);
BENCHMARK_TEMPLATE(BM_serial, kLargeRows);

template <int32_t kRows>
void BM_dispenso_static_reference(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const CsrMatrix& m = getMatrix(kRows, kMeanNnz);
  std::vector<double> B(static_cast<size_t>(m.numCols) * kRhsCols, 1.0);
  std::vector<double> C(static_cast<size_t>(m.numRows) * kRhsCols, 0.0);

  dispenso::ThreadPool pool(num_threads);

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    dispenso::parallel_for(
        tasks,
        dispenso::makeChunkedRange(0, m.numRows, dispenso::ParForChunking::kStatic),
        [&m, &B, &C](int32_t r0, int32_t r1) {
          for (int32_t r = r0; r < r1; ++r) {
            mulRow<kRhsCols>(m, B.data(), C.data(), r);
          }
        });
    benchmark::DoNotOptimize(C);
  }
}

template <int32_t kRows>
void BM_dispenso_auto(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const CsrMatrix& m = getMatrix(kRows, kMeanNnz);
  std::vector<double> B(static_cast<size_t>(m.numCols) * kRhsCols, 1.0);
  std::vector<double> C(static_cast<size_t>(m.numRows) * kRhsCols, 0.0);

  dispenso::ThreadPool pool(num_threads);
  dispenso::ParForOptions options;
  options.defaultChunking = dispenso::ParForChunking::kAdaptive;

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    dispenso::parallel_for(
        tasks,
        0,
        m.numRows,
        [&m, &B, &C](int32_t r0, int32_t r1) {
          for (int32_t r = r0; r < r1; ++r) {
            mulRow<kRhsCols>(m, B.data(), C.data(), r);
          }
        },
        options);
    benchmark::DoNotOptimize(C);
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
template <int32_t kRows>
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const CsrMatrix& m = getMatrix(kRows, kMeanNnz);
  std::vector<double> B(static_cast<size_t>(m.numCols) * kRhsCols, 1.0);
  std::vector<double> C(static_cast<size_t>(m.numRows) * kRhsCols, 0.0);

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(0, m.numRows),
        [&m, &B, &C](const tbb::blocked_range<int32_t>& r) {
          for (int32_t i = r.begin(); i != r.end(); ++i) {
            mulRow<kRhsCols>(m, B.data(), C.data(), i);
          }
        });
    benchmark::DoNotOptimize(C);
  }
}
#endif

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i : benchmarkThreadCounts()) {
    b->Args({i});
  }
}

BENCHMARK_TEMPLATE(BM_dispenso_static_reference, kMedRows)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_static_reference, kLargeRows)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_auto, kMedRows)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_auto, kLargeRows)->Apply(CustomArguments)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK_TEMPLATE(BM_tbb, kMedRows)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb, kLargeRows)->Apply(CustomArguments)->UseRealTime();
#endif

BENCHMARK_MAIN();
