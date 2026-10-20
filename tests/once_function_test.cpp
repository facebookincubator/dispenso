/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/once_function.h>

#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

constexpr size_t kExtraSmall = 8;
constexpr size_t kSmall = 24;
constexpr size_t kMedium = 120;
constexpr size_t kLarge = 248;
constexpr size_t kExtraLarge = 10000;

using dispenso::OnceFunction;

TEST(OnceFunction, Empty) {
  OnceFunction f([]() {});
  f();
}

TEST(OnceFunction, MoveConstructor) {
  OnceFunction f([]() {});
  OnceFunction g(std::move(f));
  g();
}

TEST(OnceFunction, MoveOperator) {
  OnceFunction f([]() {});
  OnceFunction g;
  g = std::move(f);
  g();
}

template <size_t kSize>
void testSize() {
  constexpr size_t kNumElts = kSize - sizeof(int*);
  struct Foo {
    void operator()() {
      int s = 0;
      for (uint8_t b : buf) {
        s += b;
      }
      *sum = s;
    }
    uint8_t buf[kNumElts];
    int* sum;
  } foo;
  for (size_t i = 0; i < kNumElts; ++i) {
    foo.buf[i] = static_cast<uint8_t>(i);
  }
  int answer;
  foo.sum = &answer;
  OnceFunction f(foo);
  OnceFunction g(foo);
  g();
  f();
  int expected = 0;
  for (size_t i = 0; i < kNumElts; ++i) {
    expected += static_cast<int>(i & 255);
  }
  EXPECT_EQ(answer, expected);
}

template <>
void testSize<8>() {
  struct Foo {
    void operator()() {
      int s = 0;
      *sum = s;
    }
    int* sum;
  } foo;
  int answer;
  foo.sum = &answer;
  OnceFunction f(foo);
  OnceFunction g(foo);
  g();
  f();
  int expected = 0;
  EXPECT_EQ(answer, expected);
}

TEST(OnceFunction, ExtraSmall) {
  testSize<kExtraSmall>();
}

TEST(OnceFunction, Small) {
  testSize<kSmall>();
}

TEST(OnceFunction, Medium) {
  testSize<kMedium>();
}

TEST(OnceFunction, Large) {
  testSize<kLarge>();
}

TEST(OnceFunction, ExtraLarge) {
  testSize<kExtraLarge>();
}

TEST(OnceFunction, MoveWithResult) {
  int result = 5;
  OnceFunction f([&result]() { result = 17; });
  EXPECT_EQ(result, 5);
  OnceFunction g(std::move(f));
  EXPECT_EQ(result, 5);
  g();
  EXPECT_EQ(result, 17);
}

template <size_t kNumElts>
void ensureDestructor() {
  int value = 0;
  struct FooWithDestructor {
    void operator()() {
      ++*value;
    }
    ~FooWithDestructor() {
      ++*value;
    }
    uint8_t buf[kNumElts];
    int* value;
  } foo;

  foo.value = &value;

  OnceFunction f(foo);
  f();
  EXPECT_EQ(value, 2);
}

TEST(OnceFunction, EnsureDestructionExtraSmall) {
  ensureDestructor<kExtraSmall>();
}

TEST(OnceFunction, EnsureDestructionSmall) {
  ensureDestructor<kSmall>();
}

TEST(OnceFunction, EnsureDestructionMedium) {
  ensureDestructor<kMedium>();
}

TEST(OnceFunction, EnsureDestructionLarge) {
  ensureDestructor<kLarge>();
}

TEST(OnceFunction, EnsureDestructionExtraLarge) {
  ensureDestructor<kExtraLarge>();
}

template <size_t alignment>
struct EnsureAlign {
  void operator()() {
    uintptr_t bloc = reinterpret_cast<uintptr_t>(&b);
    EXPECT_EQ(0, bloc & (alignment - 1)) << "broken for alignment: " << alignment;
  }

  alignas(alignment) char b = 0;
};

TEST(OnceFunction, EnsureAlignment1) {
  EnsureAlign<1> e;
  OnceFunction f(e);
  f();
}

TEST(OnceFunction, EnsureAlignment2) {
  EnsureAlign<2> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment4) {
  EnsureAlign<4> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment8) {
  EnsureAlign<8> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment16) {
  EnsureAlign<16> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment32) {
  EnsureAlign<32> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment64) {
  EnsureAlign<64> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment128) {
  EnsureAlign<128> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment256) {
  EnsureAlign<256> e;
  OnceFunction f(e);
  f();
}

// Moves a OnceFunction through a chain of move constructions and move assignments, as the
// scheduler does when tasks pass through queues.
OnceFunction moveAround(OnceFunction f) {
  OnceFunction a(std::move(f));
  OnceFunction b;
  b = std::move(a);
  OnceFunction c(std::move(b));
  return c;
}

TEST(OnceFunction, ShortStringCapture) {
  // Short enough for the small-string optimisation, which on libstdc++ points into the string
  // object itself, so a byte copy of the capture would leave it pointing at the old location.
  std::string captured = "short";
  std::string result;
  OnceFunction f = moveAround([captured, &result]() { result = captured + "!"; });
  f();
  EXPECT_EQ(result, "short!");
}

TEST(OnceFunction, LongStringCapture) {
  std::string captured(100, 'x');
  std::string result;
  OnceFunction f = moveAround([captured, &result]() { result = captured; });
  f();
  EXPECT_EQ(result, std::string(100, 'x'));
}

TEST(OnceFunction, MoveOnlyCapture) {
  auto p = std::make_unique<int>(42);
  int result = 0;
  OnceFunction f = moveAround([p = std::move(p), &result]() { result = *p; });
  f();
  EXPECT_EQ(result, 42);
}

namespace {
// Holds a pointer into itself, which only a move constructor can keep correct.
struct SelfReferential {
  SelfReferential(int* out) : self(buf), out(out) {
    buf[0] = 7;
  }
  SelfReferential(SelfReferential&& other) noexcept : self(buf), out(other.out) {
    buf[0] = other.buf[0];
  }
  void operator()() {
    *out = self == buf ? *self : -1;
  }
  int buf[4];
  int* self;
  int* out;
};

// As SelfReferential, but its move may throw, so OnceFunction cannot store it inline.
struct SelfReferentialThrowingMove {
  SelfReferentialThrowingMove(int* out) : self(buf), out(out) {
    buf[0] = 9;
  }
  SelfReferentialThrowingMove(SelfReferentialThrowingMove&& other) : self(buf), out(other.out) {
    buf[0] = other.buf[0];
  }
  void operator()() {
    *out = self == buf ? *self : -1;
  }
  int buf[4];
  int* self;
  int* out;
};

// Counts live instances, to check that every constructed functor is destroyed exactly once.
template <size_t kPad>
struct Counted {
  explicit Counted(int* live, int* calls) : live(live), calls(calls) {
    ++*live;
  }
  Counted(const Counted& other) : live(other.live), calls(other.calls) {
    ++*live;
  }
  Counted(Counted&& other) noexcept : live(other.live), calls(other.calls) {
    ++*live;
  }
  ~Counted() {
    --*live;
  }
  void operator()() {
    ++*calls;
  }
  int* live;
  int* calls;
  char pad[kPad];
};
} // namespace

TEST(OnceFunction, SelfReferentialFunctor) {
  int result = 0;
  OnceFunction f = moveAround(SelfReferential(&result));
  f();
  EXPECT_EQ(result, 7);
}

TEST(OnceFunction, SelfReferentialFunctorWithThrowingMove) {
  int result = 0;
  OnceFunction f = moveAround(SelfReferentialThrowingMove(&result));
  f();
  EXPECT_EQ(result, 9);
}

template <size_t kPad>
void testLifetime() {
  int live = 0;
  int calls = 0;
  {
    OnceFunction f = moveAround(Counted<kPad>(&live, &calls));
    f();
  }
  EXPECT_EQ(live, 0);
  EXPECT_EQ(calls, 1);

  {
    OnceFunction f = moveAround(Counted<kPad>(&live, &calls));
    f.cleanupNotRun();
  }
  EXPECT_EQ(live, 0);
  EXPECT_EQ(calls, 1);
}

TEST(OnceFunction, LifetimeInline) {
  testLifetime<8>();
}

TEST(OnceFunction, LifetimeSpilled) {
  testLifetime<200>();
}

TEST(OnceFunction, OverAlignedNontrivialCapture) {
  struct alignas(128) Aligned {
    std::string s = "aligned";
  } aligned;
  std::string result;
  OnceFunction f = moveAround([aligned, &result]() {
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(&aligned) & 127);
    result = aligned.s;
  });
  f();
  EXPECT_EQ(result, "aligned");
}

TEST(OnceFunction, MovedBetweenArrays) {
  // Moves many string-capturing functions from one array to another, as a task queue does when it
  // grows. (Not a std::vector: before C++17 std::allocator ignores OnceFunction's 64-byte
  // alignment.)
  constexpr size_t kNum = 100;
  std::vector<std::string> results(kNum);
  OnceFunction from[kNum];
  OnceFunction to[kNum];
  for (size_t i = 0; i < kNum; ++i) {
    std::string s = std::to_string(i);
    from[i] = OnceFunction([s, &results, i]() { results[i] = s; });
  }
  for (size_t i = 0; i < kNum; ++i) {
    to[i] = std::move(from[i]);
  }
  for (auto& f : to) {
    f();
  }
  for (size_t i = 0; i < kNum; ++i) {
    EXPECT_EQ(results[i], std::to_string(i));
  }
}

TEST(OnceFunction, MoveAfterInvoke) {
  // Moving an already-invoked OnceFunction must not touch the destroyed functor.
  std::string captured = "once";
  std::string result;
  OnceFunction f([captured, &result]() { result = captured; });
  f();
  OnceFunction g(std::move(f));
  OnceFunction h;
  h = std::move(g);
  EXPECT_EQ(result, "once");
}

#if defined(__cpp_exceptions)
namespace {
struct ThrowOnCopy {
  ThrowOnCopy() = default;
  ThrowOnCopy(const ThrowOnCopy&) {
    throw std::runtime_error("copy");
  }
  void operator()() {}
  // Larger than the pooled small-buffer sizes, so the spill goes to the heap and a leaked buffer is
  // visible to LeakSanitizer.
  char pad[1000];
};
} // namespace

TEST(OnceFunction, ThrowingSpilledConstructionReleasesBuffer) {
  // The spill buffer must be released when the callable's constructor throws.
  ThrowOnCopy t;
  for (int i = 0; i < 1000; ++i) {
    EXPECT_THROW(OnceFunction f(t), std::runtime_error);
  }
}
#endif // __cpp_exceptions
