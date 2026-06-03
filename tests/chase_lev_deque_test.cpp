/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/chase_lev_deque.h>

#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using dispenso::ChaseLevDeque;

// =============================================================================
// Basic Functionality
// =============================================================================

TEST(ChaseLevDeque, DefaultConstructionIsEmpty) {
  ChaseLevDeque<int> deque;
  EXPECT_TRUE(deque.empty());
  EXPECT_EQ(deque.size(), 0u);
  EXPECT_EQ(deque.capacity(), 32u);
}

TEST(ChaseLevDeque, CustomCapacity) {
  EXPECT_EQ((ChaseLevDeque<int, 8>::capacity()), 8u);
  EXPECT_EQ((ChaseLevDeque<int, 1024>::capacity()), 1024u);
}

TEST(ChaseLevDeque, PushAndPopSingle) {
  ChaseLevDeque<int, 8> deque;
  EXPECT_TRUE(deque.try_push(42));
  EXPECT_FALSE(deque.empty());
  EXPECT_EQ(deque.size(), 1u);

  int v = 0;
  EXPECT_TRUE(deque.try_pop(v));
  EXPECT_EQ(v, 42);
  EXPECT_TRUE(deque.empty());
}

TEST(ChaseLevDeque, PushAndStealSingle) {
  ChaseLevDeque<int, 8> deque;
  EXPECT_TRUE(deque.try_push(42));

  int v = 0;
  EXPECT_TRUE(deque.try_steal(v));
  EXPECT_EQ(v, 42);
  EXPECT_TRUE(deque.empty());
}

TEST(ChaseLevDeque, PopFromEmpty) {
  ChaseLevDeque<int, 8> deque;
  int v = 999;
  EXPECT_FALSE(deque.try_pop(v));
  EXPECT_EQ(v, 999);
  EXPECT_TRUE(deque.empty());
}

TEST(ChaseLevDeque, StealFromEmpty) {
  ChaseLevDeque<int, 8> deque;
  int v = 999;
  EXPECT_FALSE(deque.try_steal(v));
  EXPECT_EQ(v, 999);
}

// =============================================================================
// Order Semantics: LIFO Pop, FIFO Steal
// =============================================================================

TEST(ChaseLevDeque, OwnerPopIsLIFO) {
  ChaseLevDeque<int, 8> deque;
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(deque.try_push(i));
  }
  // Owner pop returns most recent first.
  for (int i = 4; i >= 0; --i) {
    int v;
    EXPECT_TRUE(deque.try_pop(v));
    EXPECT_EQ(v, i);
  }
  EXPECT_TRUE(deque.empty());
}

TEST(ChaseLevDeque, StealIsFIFO) {
  ChaseLevDeque<int, 8> deque;
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(deque.try_push(i));
  }
  // Steal returns oldest first.
  for (int i = 0; i < 5; ++i) {
    int v;
    EXPECT_TRUE(deque.try_steal(v));
    EXPECT_EQ(v, i);
  }
  EXPECT_TRUE(deque.empty());
}

// =============================================================================
// Capacity and Wrap-Around
// =============================================================================

TEST(ChaseLevDeque, FillToCapacity) {
  ChaseLevDeque<int, 4> deque;
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(deque.try_push(i));
  }
  EXPECT_EQ(deque.size(), 4u);
  EXPECT_FALSE(deque.try_push(99));
}

TEST(ChaseLevDeque, WrapAroundManyCycles) {
  ChaseLevDeque<int, 4> deque;
  // Repeatedly push and pop more elements than capacity to exercise modular indexing.
  for (int cycle = 0; cycle < 100; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(deque.try_push(cycle * 10 + i));
    }
    for (int i = 2; i >= 0; --i) {
      int v;
      EXPECT_TRUE(deque.try_pop(v));
      EXPECT_EQ(v, cycle * 10 + i);
    }
  }
  EXPECT_TRUE(deque.empty());
}

TEST(ChaseLevDeque, WrapAroundWithSteals) {
  ChaseLevDeque<int, 4> deque;
  // Mix push + steal; both top and bottom advance, exercising index wrap.
  for (int cycle = 0; cycle < 100; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(deque.try_push(cycle * 10 + i));
    }
    for (int i = 0; i < 3; ++i) {
      int v;
      EXPECT_TRUE(deque.try_steal(v));
      EXPECT_EQ(v, cycle * 10 + i);
    }
  }
  EXPECT_TRUE(deque.empty());
}

// =============================================================================
// Pointer Payload (typical fork-join shape: store a heap-allocated task pointer)
// =============================================================================

TEST(ChaseLevDeque, PointerPayloadSurvivesPushPop) {
  ChaseLevDeque<int*, 4> deque;
  int a = 1, b = 2;
  EXPECT_TRUE(deque.try_push(&a));
  EXPECT_TRUE(deque.try_push(&b));

  int* out = nullptr;
  EXPECT_TRUE(deque.try_pop(out));
  EXPECT_EQ(out, &b);
  EXPECT_TRUE(deque.try_steal(out));
  EXPECT_EQ(out, &a);
}

// =============================================================================
// Non-Default-Constructible Type
// =============================================================================

namespace {
struct PodNoDefault {
  int value;
  // No default constructor.
  explicit PodNoDefault(int v) : value(v) {}
  PodNoDefault(const PodNoDefault&) = default;
  PodNoDefault& operator=(const PodNoDefault&) = default;
};
static_assert(std::is_trivially_copyable<PodNoDefault>::value, "");
} // namespace

TEST(ChaseLevDeque, NonDefaultConstructibleViaPopInto) {
  ChaseLevDeque<PodNoDefault, 4> deque;
  EXPECT_TRUE(deque.try_push(PodNoDefault(42)));

  alignas(PodNoDefault) char storage[sizeof(PodNoDefault)];
  EXPECT_TRUE(deque.try_pop_into(reinterpret_cast<PodNoDefault*>(storage)));
  auto* obj = reinterpret_cast<PodNoDefault*>(storage);
  EXPECT_EQ(obj->value, 42);
  obj->~PodNoDefault();
}

TEST(ChaseLevDeque, NonDefaultConstructibleViaStealInto) {
  ChaseLevDeque<PodNoDefault, 4> deque;
  EXPECT_TRUE(deque.try_push(PodNoDefault(42)));

  alignas(PodNoDefault) char storage[sizeof(PodNoDefault)];
  EXPECT_TRUE(deque.try_steal_into(reinterpret_cast<PodNoDefault*>(storage)));
  auto* obj = reinterpret_cast<PodNoDefault*>(storage);
  EXPECT_EQ(obj->value, 42);
  obj->~PodNoDefault();
}

// =============================================================================
// POD Struct Payload
// =============================================================================

namespace {
struct Pair {
  int a;
  int b;
};
static_assert(std::is_trivially_copyable<Pair>::value, "");
} // namespace

TEST(ChaseLevDeque, PodStructRoundTrips) {
  ChaseLevDeque<Pair, 4> deque;
  EXPECT_TRUE(deque.try_push(Pair{1, 2}));
  Pair out{};
  EXPECT_TRUE(deque.try_pop(out));
  EXPECT_EQ(out.a, 1);
  EXPECT_EQ(out.b, 2);
}

// =============================================================================
// Concurrent: Single Owner + Single Stealer
// =============================================================================

TEST(ChaseLevDeque, OwnerPushPopConcurrentWithSingleStealer) {
  ChaseLevDeque<int, 32> deque;
  constexpr int kIters = 100000;

  std::atomic<bool> ownerDone{false};
  std::atomic<int64_t> stolenSum{0};
  std::atomic<int64_t> stolenCount{0};
  std::atomic<int64_t> poppedSum{0};
  std::atomic<int64_t> poppedCount{0};

  std::thread owner([&] {
    int64_t localSum = 0;
    int64_t localCount = 0;
    for (int i = 1; i <= kIters; ++i) {
      while (!deque.try_push(i)) {
        // Deque full; pop one to drain.
        int v;
        if (deque.try_pop(v)) {
          localSum += v;
          ++localCount;
        }
      }
      // Occasionally pop our own work.
      if ((i & 3) == 0) {
        int v;
        if (deque.try_pop(v)) {
          localSum += v;
          ++localCount;
        }
      }
    }
    int v;
    while (deque.try_pop(v)) {
      localSum += v;
      ++localCount;
    }
    poppedSum.store(localSum, std::memory_order_relaxed);
    poppedCount.store(localCount, std::memory_order_relaxed);
    ownerDone.store(true, std::memory_order_release);
  });

  std::thread stealer([&] {
    int64_t localSum = 0;
    int64_t localCount = 0;
    while (true) {
      int v;
      if (deque.try_steal(v)) {
        localSum += v;
        ++localCount;
      } else if (ownerDone.load(std::memory_order_acquire) && deque.empty()) {
        break;
      }
    }
    stolenSum.store(localSum, std::memory_order_relaxed);
    stolenCount.store(localCount, std::memory_order_relaxed);
  });

  owner.join();
  stealer.join();

  const int64_t expectedSum = static_cast<int64_t>(kIters) * (kIters + 1) / 2;
  EXPECT_EQ(poppedCount.load() + stolenCount.load(), kIters);
  EXPECT_EQ(poppedSum.load() + stolenSum.load(), expectedSum);
}

// =============================================================================
// Concurrent: Single Owner + Many Stealers
// =============================================================================

TEST(ChaseLevDeque, OwnerVsManyStealers) {
  ChaseLevDeque<int, 64> deque;
  constexpr int kIters = 200000;
  constexpr int kStealers = 4;

  std::atomic<bool> ownerDone{false};
  std::atomic<int64_t> takenSum{0};
  std::atomic<int64_t> takenCount{0};

  std::thread owner([&] {
    int64_t localSum = 0;
    int64_t localCount = 0;
    for (int i = 1; i <= kIters; ++i) {
      while (!deque.try_push(i)) {
        int v;
        if (deque.try_pop(v)) {
          localSum += v;
          ++localCount;
        }
      }
    }
    int v;
    while (deque.try_pop(v)) {
      localSum += v;
      ++localCount;
    }
    takenSum.fetch_add(localSum, std::memory_order_relaxed);
    takenCount.fetch_add(localCount, std::memory_order_relaxed);
    ownerDone.store(true, std::memory_order_release);
  });

  std::vector<std::thread> stealers;
  for (int s = 0; s < kStealers; ++s) {
    stealers.emplace_back([&] {
      int64_t localSum = 0;
      int64_t localCount = 0;
      while (true) {
        int v;
        if (deque.try_steal(v)) {
          localSum += v;
          ++localCount;
        } else if (ownerDone.load(std::memory_order_acquire) && deque.empty()) {
          break;
        }
      }
      takenSum.fetch_add(localSum, std::memory_order_relaxed);
      takenCount.fetch_add(localCount, std::memory_order_relaxed);
    });
  }

  owner.join();
  for (auto& t : stealers) {
    t.join();
  }

  const int64_t expectedSum = static_cast<int64_t>(kIters) * (kIters + 1) / 2;
  EXPECT_EQ(takenCount.load(), kIters);
  EXPECT_EQ(takenSum.load(), expectedSum);
}

// =============================================================================
// Last-Element Race: Owner Pop vs Stealer
// =============================================================================

TEST(ChaseLevDeque, LastElementRaceNoLossNoDuplicate) {
  // Hammer the single-element race: owner pushes one, then pop races stealers.
  ChaseLevDeque<int, 4> deque;
  constexpr int kRounds = 50000;
  constexpr int kStealers = 3;

  std::atomic<int> roundBarrier{0};
  std::atomic<int> takenCount{0};
  std::atomic<int64_t> takenSum{0};
  std::atomic<bool> done{false};

  std::vector<std::thread> stealers;
  for (int s = 0; s < kStealers; ++s) {
    stealers.emplace_back([&] {
      int lastSeenRound = -1;
      while (!done.load(std::memory_order_acquire)) {
        const int round = roundBarrier.load(std::memory_order_acquire);
        if (round == lastSeenRound) {
          continue;
        }
        int v;
        if (deque.try_steal(v)) {
          takenCount.fetch_add(1, std::memory_order_relaxed);
          takenSum.fetch_add(v, std::memory_order_relaxed);
          lastSeenRound = round;
        }
      }
    });
  }

  int64_t expectedSum = 0;
  for (int round = 1; round <= kRounds; ++round) {
    expectedSum += round;
    EXPECT_TRUE(deque.try_push(round));
    roundBarrier.store(round, std::memory_order_release);
    int v;
    if (deque.try_pop(v)) {
      takenCount.fetch_add(1, std::memory_order_relaxed);
      takenSum.fetch_add(v, std::memory_order_relaxed);
    }
    // Spin until the deque is drained for this round (steal won the race).
    for (int spin = 0; !deque.empty(); ++spin) {
      ASSERT_LT(spin, 10000000) << "Deque not drained in round " << round;
    }
  }
  done.store(true, std::memory_order_release);
  for (auto& t : stealers) {
    t.join();
  }

  EXPECT_EQ(takenCount.load(), kRounds);
  EXPECT_EQ(takenSum.load(), expectedSum);
}
