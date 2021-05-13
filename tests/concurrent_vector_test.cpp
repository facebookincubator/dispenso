// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/concurrent_vector.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <dispenso/parallel_for.h>
#include <gtest/gtest.h>

template <typename Traits>
class ConcurrentVectorTest : public testing::Test {
 public:
};

using dispenso::ConcurrentVectorReallocStrategy;

struct TestTraitsA {
  static constexpr bool kPreferBuffersInline = false;
  static constexpr ConcurrentVectorReallocStrategy kReallocStrategy =
      ConcurrentVectorReallocStrategy::kHalfBufferAhead;
  static constexpr bool kIteratorPreferSpeed = false;
};

struct TestTraitsB {
  static constexpr bool kPreferBuffersInline = true;
  static constexpr ConcurrentVectorReallocStrategy kReallocStrategy =
      ConcurrentVectorReallocStrategy::kFullBufferAhead;
  static constexpr bool kIteratorPreferSpeed = true;
};

using TestTraitsTypes =
    ::testing::Types<dispenso::DefaultConcurrentVectorTraits, TestTraitsA, TestTraitsB>;
TYPED_TEST_SUITE(ConcurrentVectorTest, TestTraitsTypes);

template <typename CVec, typename Func>
void runVariedTest(int num, Func func) {
  {
    CVec vec;
    func(num, vec);
  }
  {
    CVec vec(num / 3, dispenso::ReserveTag);
    func(num, vec);
  }
  {
    CVec vec;
    vec.reserve(num / 2);
    func(num, vec);
  }
}

#define RUN_VARIED_TEST(len, func, EltType)                      \
  runVariedTest<dispenso::ConcurrentVector<EltType, TypeParam>>( \
      (len), func<dispenso::ConcurrentVector<EltType, TypeParam>>)

template <typename CVec, typename Func>
void runVariedTest2(int num, int oth, Func func) {
  {
    CVec vec;
    func(num, oth, vec);
  }
  {
    CVec vec(num / 3, dispenso::ReserveTag);
    func(num, oth, vec);
  }
  {
    CVec vec;
    vec.reserve(num / 2);
    func(num, oth, vec);
  }
}

#define RUN_VARIED_TEST2(len, oth, func, EltType)                 \
  runVariedTest2<dispenso::ConcurrentVector<EltType, TypeParam>>( \
      (len), (oth), func<dispenso::ConcurrentVector<EltType, TypeParam>>)

template <typename CVec>
void indexCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  for (int index = 0; index < vec.size(); ++index) {
    EXPECT_EQ(index, *vec[index]);
  }

  const auto& cvec = vec;

  for (int index = 0; index < cvec.size(); ++index) {
    EXPECT_EQ(index, *cvec[index]);
  }
}

TYPED_TEST(ConcurrentVectorTest, IndexCorrectTiny) {
  RUN_VARIED_TEST(4, indexCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IndexCorrectSmall) {
  RUN_VARIED_TEST(12, indexCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IndexCorrectMedium) {
  RUN_VARIED_TEST(513, indexCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IndexCorrectLarge) {
  RUN_VARIED_TEST(1 << 13, indexCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IndexCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 13) + 1, indexCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void iterateCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  int index = 0;

  for (auto& v : vec) {
    EXPECT_EQ(index, *v);
    ++index;
  }

  const auto& cvec = vec;

  index = 0;
  for (auto& v : cvec) {
    EXPECT_EQ(index, *v);
    ++index;
  }
}

TYPED_TEST(ConcurrentVectorTest, IterateCorrectTiny) {
  RUN_VARIED_TEST(3, iterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IterateCorrectSmall) {
  RUN_VARIED_TEST(37, iterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IterateCorrectMedium) {
  RUN_VARIED_TEST(768, iterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IterateCorrectLarge) {
  RUN_VARIED_TEST(1 << 12, iterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, IterateCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 12) + 1, iterateCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void clearAndReuseCorrect(int num, CVec& vec) {
  for (int round = 0; round < 4; ++round) {
    vec.clear();
    for (int i = 0; i < num; ++i) {
      vec.push_back(std::make_unique<int>(i));
    }

    int index = 0;

    for (auto& v : vec) {
      EXPECT_EQ(index, *v);
      ++index;
    }
  }
}

TYPED_TEST(ConcurrentVectorTest, ClearAndReuseCorrectTiny) {
  RUN_VARIED_TEST(1, clearAndReuseCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ClearAndReuseCorrectSmall) {
  RUN_VARIED_TEST(90, clearAndReuseCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ClearAndReuseCorrectMedium) {
  RUN_VARIED_TEST(129, clearAndReuseCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ClearAndReuseCorrectLarge) {
  RUN_VARIED_TEST(1 << 11, clearAndReuseCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ClearAndReuseCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, clearAndReuseCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void copyConstructor(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  for (int round = 0; round < 4; ++round) {
    CVec newVec(vec);
    int index = 0;
    for (auto& v : newVec) {
      EXPECT_EQ(index, *v);
      ++index;
    }
  }
  CVec newVec(vec);
  for (int i = num; i < 2 * num; ++i) {
    newVec.push_back(std::make_unique<int>(i));
  }

  size_t sum = 0;
  for (auto& v : newVec) {
    sum += *v;
  }
  size_t maxV = 2 * num - 1;

  EXPECT_EQ(sum, (maxV * (maxV + 1)) / 2);
}

TYPED_TEST(ConcurrentVectorTest, CopyConstructorTiny) {
  RUN_VARIED_TEST(1, copyConstructor, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyConstructorSmall) {
  RUN_VARIED_TEST(90, copyConstructor, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyConstructorMedium) {
  RUN_VARIED_TEST(129, copyConstructor, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyConstructorLarge) {
  RUN_VARIED_TEST(1 << 11, copyConstructor, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyConstructorLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, copyConstructor, std::shared_ptr<int>);
}

template <typename CVec>
void moveConstructor(int num, CVec& vec) {
  for (int round = 0; round < 4; ++round) {
    EXPECT_TRUE(vec.empty());
    for (int i = 0; i < num; ++i) {
      vec.push_back(std::make_unique<int>(i));
    }

    CVec newVecMoved(std::move(vec));
    EXPECT_EQ(newVecMoved.size(), num);
    EXPECT_TRUE(vec.empty());
    int index = 0;
    for (auto& v : newVecMoved) {
      EXPECT_EQ(index, *v);
      ++index;
    }
  }
}

TYPED_TEST(ConcurrentVectorTest, MoveConstructorTiny) {
  RUN_VARIED_TEST(1, moveConstructor, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveConstructorSmall) {
  RUN_VARIED_TEST(90, moveConstructor, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveConstructorMedium) {
  RUN_VARIED_TEST(129, moveConstructor, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveConstructorLarge) {
  RUN_VARIED_TEST(1 << 11, moveConstructor, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveConstructorLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, moveConstructor, std::unique_ptr<int>);
}

template <typename CVec>
void copyOperator(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_shared<int>(i));
  }
  CVec newVec;
  for (int round = 0; round < 4; ++round) {
    newVec = vec;
    EXPECT_EQ(newVec.size(), vec.size());
    int index = 0;
    for (auto& v : newVec) {
      EXPECT_EQ(index, *v);
      ++index;
    }
  }

  for (int i = num; i < 2 * num; ++i) {
    newVec.push_back(std::make_shared<int>(i));
  }

  size_t sum = 0;
  for (auto& v : newVec) {
    sum += *v;
  }
  size_t maxV = 2 * num - 1;

  EXPECT_EQ(sum, (maxV * (maxV + 1)) / 2);
}

TYPED_TEST(ConcurrentVectorTest, CopyOperatorTiny) {
  RUN_VARIED_TEST(7, copyOperator, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyOperatorSmall) {
  RUN_VARIED_TEST(127, copyOperator, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyOperatorMedium) {
  RUN_VARIED_TEST(511, copyOperator, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyOperatorLarge) {
  RUN_VARIED_TEST(1 << 9, copyOperator, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, CopyOperatorLargeP1) {
  RUN_VARIED_TEST((1 << 9) + 1, copyOperator, std::shared_ptr<int>);
}

template <typename CVec>
void moveOperator(int num, CVec& vec) {
  CVec newVecMoved;
  for (int round = 0; round < 4; ++round) {
    EXPECT_TRUE(vec.empty());
    for (int i = 0; i < num; ++i) {
      vec.push_back(std::make_unique<int>(i));
    }
    newVecMoved = std::move(vec);
    EXPECT_EQ(newVecMoved.size(), num);
    int index = 0;
    for (auto& v : newVecMoved) {
      EXPECT_EQ(index, *v);
      ++index;
    }
  }
}

TYPED_TEST(ConcurrentVectorTest, MoveOperatorTiny) {
  RUN_VARIED_TEST(17, moveOperator, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveOperatorSmall) {
  RUN_VARIED_TEST(901, moveOperator, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveOperatorMedium) {
  RUN_VARIED_TEST(1102, moveOperator, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveOperatorLarge) {
  RUN_VARIED_TEST(1 << 11, moveOperator, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, MoveOperatorLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, moveOperator, std::unique_ptr<int>);
}

template <typename CVec>
void shrinkToFit(int num, CVec& vec) {
  vec.shrink_to_fit();

  EXPECT_EQ(vec.capacity(), vec.default_capacity());

  for (size_t i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  size_t maxCapacity = vec.capacity();

  EXPECT_LE(
      maxCapacity,
      2 * std::max<size_t>(dispenso::detail::nextPow2(vec.size() + 1), vec.default_capacity()))
      << "Num: " << num << " default: " << vec.default_capacity();

  for (size_t i = 0; i < num / 2; ++i) {
    vec.pop_back();
  }

  EXPECT_EQ(vec.capacity(), maxCapacity);

  vec.shrink_to_fit();

  EXPECT_LE(vec.capacity(), maxCapacity);

  size_t afterPopNShrinkCap = vec.capacity();

  EXPECT_LE(
      afterPopNShrinkCap,
      2 * std::max<size_t>(dispenso::detail::nextPow2(vec.size() + 1), vec.default_capacity()))
      << "Size: " << vec.size() << " Num: " << num << " default: " << vec.default_capacity();

  vec.clear();

  EXPECT_EQ(vec.capacity(), afterPopNShrinkCap);

  vec.shrink_to_fit();

  EXPECT_EQ(vec.capacity(), vec.default_capacity());
}

TYPED_TEST(ConcurrentVectorTest, ShrinkToFitTiny) {
  RUN_VARIED_TEST(17, shrinkToFit, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ShrinkToFitSmall) {
  RUN_VARIED_TEST(901, shrinkToFit, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ShrinkToFitMedium) {
  RUN_VARIED_TEST(1102, shrinkToFit, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ShrinkToFitLarge) {
  RUN_VARIED_TEST(1 << 13, shrinkToFit, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ShrinkToFitLargeP1) {
  RUN_VARIED_TEST((1 << 13) + 1, shrinkToFit, std::unique_ptr<int>);
}

template <typename T>
struct ReverseWrapper {
  T& iterable;
};

template <typename T>
auto begin(ReverseWrapper<T> w) {
  return std::rbegin(w.iterable);
}

template <typename T>
auto end(ReverseWrapper<T> w) {
  return std::rend(w.iterable);
}

template <typename T>
ReverseWrapper<T> reverse(T&& iterable) {
  return {iterable};
}

template <typename CVec>
void reverseIterateCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  int index = num;

  for (auto& v : reverse(vec)) {
    --index;
    EXPECT_EQ(index, *v);
  }

  const auto& cvec = vec;
  index = num;
  for (auto& v : reverse(cvec)) {
    --index;
    EXPECT_EQ(index, *v);
  }
}

TYPED_TEST(ConcurrentVectorTest, ReverseIterateCorrectTiny) {
  RUN_VARIED_TEST(6, reverseIterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ReverseIterateCorrectSmall) {
  RUN_VARIED_TEST(73, reverseIterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ReverseIterateCorrectMedium) {
  RUN_VARIED_TEST(677, reverseIterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ReverseIterateCorrectLarge) {
  RUN_VARIED_TEST(1 << 13, reverseIterateCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ReverseIterateCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 13) + 1, reverseIterateCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void binarySearchCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(
        std::lower_bound(
            std::begin(vec), std::end(vec), i, [](const auto& up, int val) { return *up < val; }) -
            std::begin(vec),
        i);
  }
}

TYPED_TEST(ConcurrentVectorTest, BinarySearchCorrectTiny) {
  RUN_VARIED_TEST(11, binarySearchCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, BinarySearchCorrectSmall) {
  RUN_VARIED_TEST(59, binarySearchCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, BinarySearchCorrectMedium) {
  RUN_VARIED_TEST(711, binarySearchCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, BinarySearchCorrectLarge) {
  RUN_VARIED_TEST(1 << 11, binarySearchCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, BinarySearchCorrectLargeP3) {
  RUN_VARIED_TEST((1 << 11) + 3, binarySearchCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void operatorPlusCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  auto start = std::begin(vec);
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(**(start + i), i);
  }
  auto end = std::end(vec) - 1;
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(**(end - i), num - i - 1);
  }
}

TYPED_TEST(ConcurrentVectorTest, OperatorPlusCorrectTiny) {
  RUN_VARIED_TEST(11, operatorPlusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorPlusCorrectSmall) {
  RUN_VARIED_TEST(59, operatorPlusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorPlusCorrectMedium) {
  RUN_VARIED_TEST(711, operatorPlusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorPlusCorrectLarge) {
  RUN_VARIED_TEST(1 << 11, operatorPlusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorPlusCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 3, operatorPlusCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void operatorMinusCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  auto start = std::begin(vec);
  auto it = start;
  for (int i = 0; i < num; ++i, ++it) {
    EXPECT_EQ(it - start, i);
  }

  auto end = std::end(vec);

  EXPECT_EQ(end, it);

  for (int i = 0; i < num; ++i, --it) {
    EXPECT_EQ(end - it, i);
  }

  auto mid = start + num / 2;
  it = start;
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(mid - it++, num / 2 - i);
  }
}

TYPED_TEST(ConcurrentVectorTest, OperatorMinusCorrectTiny) {
  RUN_VARIED_TEST(11, operatorMinusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorMinusCorrectSmall) {
  RUN_VARIED_TEST(59, operatorMinusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorMinusCorrectMedium) {
  RUN_VARIED_TEST(711, operatorMinusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorMinusCorrectLarge) {
  RUN_VARIED_TEST(711, operatorMinusCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorMinusCorrectLargeP1) {
  RUN_VARIED_TEST(711, operatorMinusCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void operatorComparisonsCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  auto start = std::begin(vec);
  auto it = start;
  EXPECT_FALSE(it > start);
  EXPECT_FALSE(start < it);
  EXPECT_FALSE(it < start);
  EXPECT_FALSE(start > it);
  EXPECT_TRUE(it >= start);
  EXPECT_TRUE(start <= it);
  for (int i = 0; i < num; ++i) {
    ++it;
    EXPECT_TRUE(it > start);
    EXPECT_TRUE(it >= start);
    EXPECT_TRUE(start < it);
    EXPECT_TRUE(start <= it);
  }

  auto end = std::end(vec);

  EXPECT_EQ(end, it);

  EXPECT_TRUE(end >= it);
  EXPECT_TRUE(it <= end);
  EXPECT_FALSE(end > it);
  EXPECT_FALSE(it < end);
  EXPECT_FALSE(it > end);
  EXPECT_FALSE(end < it);
  for (int i = 0; i < num; ++i) {
    --it;
    EXPECT_TRUE(it < end);
    EXPECT_TRUE(it <= end);
    EXPECT_TRUE(end > it);
    EXPECT_TRUE(end >= it);
  }

  EXPECT_EQ(start, it);

  int midLen = num / 2;
  auto mid = start + midLen;

  for (int i = 0; i < midLen; ++i, ++it) {
    EXPECT_TRUE(mid > it);
    EXPECT_TRUE(mid >= it);
    EXPECT_TRUE(it < mid);
    EXPECT_TRUE(it <= mid);
  }

  EXPECT_TRUE(mid == it);
  EXPECT_TRUE(mid >= it);
  EXPECT_TRUE(it <= mid);
  EXPECT_FALSE(mid < it);
  EXPECT_FALSE(it > mid);
  ++it;
  for (int i = midLen; i < num; ++i, ++it) {
    EXPECT_TRUE(mid < it);
    EXPECT_TRUE(mid <= it);
    EXPECT_TRUE(it > mid);
    EXPECT_TRUE(it >= mid);
  }
}

TYPED_TEST(ConcurrentVectorTest, OperatorComparisonsCorrectTiny) {
  RUN_VARIED_TEST(0, operatorComparisonsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorComparisonsCorrectSmall) {
  RUN_VARIED_TEST(65, operatorComparisonsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorComparisonsCorrectMedium) {
  RUN_VARIED_TEST(511, operatorComparisonsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorComparisonsCorrectLarge) {
  RUN_VARIED_TEST(1 << 12, operatorComparisonsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorComparisonsCorrectLargeP3) {
  RUN_VARIED_TEST((1 << 12) + 3, operatorComparisonsCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void operatorBracketsCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  auto start = std::begin(vec);
  auto it = start;
  for (int i = 0; i < num; ++i, ++it) {
    for (int j = -i, k = 0; j < 0; ++j, ++k) {
      EXPECT_EQ(*it[j], k);
    }
    for (int j = 0, k = i; j < num - i; ++j, ++k) {
      EXPECT_EQ(*it[j], k);
    }
  }
}

TYPED_TEST(ConcurrentVectorTest, OperatorBracketsCorrectTiny) {
  RUN_VARIED_TEST(13, operatorBracketsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorBracketsCorrectSmall) {
  RUN_VARIED_TEST(90, operatorBracketsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorBracketsCorrectMedium) {
  RUN_VARIED_TEST(399, operatorBracketsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorBracketsCorrectLarge) {
  RUN_VARIED_TEST(1 << 9, operatorBracketsCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, OperatorBracketsCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 9) + 2, operatorBracketsCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void growByDefaultCorrect(int num, CVec& vec) {
  for (int i = 0; i < 5; ++i) {
    vec.grow_by(num);
  }

  EXPECT_EQ(vec.size(), num * 5);

  for (auto& uv : vec) {
    EXPECT_TRUE(!uv);
  }
}

TYPED_TEST(ConcurrentVectorTest, GrowByDefaultCorrectTiny) {
  RUN_VARIED_TEST(17, growByDefaultCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByDefaultCorrectSmall) {
  RUN_VARIED_TEST(91, growByDefaultCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByDefaultCorrectMedium) {
  RUN_VARIED_TEST(499, growByDefaultCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByDefaultCorrectLarge) {
  RUN_VARIED_TEST(1 << 10, growByDefaultCorrect, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByDefaultCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 10) + 1, growByDefaultCorrect, std::unique_ptr<int>);
}

template <typename CVec>
void growByConstantCorrect(int num, CVec& vec) {
  for (int i = 0; i < 5; ++i) {
    vec.grow_by(num, std::make_shared<int>(4));
  }

  size_t result = 0;

  for (auto& v : vec) {
    result += *v + 1;
  }

  EXPECT_EQ(result, 5 * vec.size());
  EXPECT_EQ(result, 5 * 5 * num);
}

TYPED_TEST(ConcurrentVectorTest, GrowByConstantCorrectTiny) {
  RUN_VARIED_TEST(12, growByConstantCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConstantCorrectSmall) {
  RUN_VARIED_TEST(81, growByConstantCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConstantCorrectMedium) {
  RUN_VARIED_TEST(300, growByConstantCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConstantCorrectLarge) {
  RUN_VARIED_TEST(1 << 9, growByConstantCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConstantCorrectLargeP7) {
  RUN_VARIED_TEST((1 << 9) + 7, growByConstantCorrect, std::shared_ptr<int>);
}

template <typename CVec>
void growByInitListCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.grow_by(
        {std::make_shared<int>(0),
         std::make_shared<int>(1),
         std::make_shared<int>(2),
         std::make_shared<int>(3),
         std::make_shared<int>(4),
         std::make_shared<int>(5),
         std::make_shared<int>(6),
         std::make_shared<int>(7),
         std::make_shared<int>(8),
         std::make_shared<int>(9),
         std::make_shared<int>(10)});
  }

  EXPECT_EQ(vec.size(), 11 * num);

  size_t result = 0;

  for (auto& v : vec) {
    result += *v;
  }

  EXPECT_EQ(result, 55 * num);
}

TYPED_TEST(ConcurrentVectorTest, GrowByInitListCorrectTiny) {
  RUN_VARIED_TEST(17, growByInitListCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByInitListCorrectSmall) {
  RUN_VARIED_TEST(91, growByInitListCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByInitListCorrectMedium) {
  RUN_VARIED_TEST(499, growByInitListCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByInitListCorrectLarge) {
  RUN_VARIED_TEST(1 << 10, growByInitListCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByInitListCorrectLargeP1) {
  RUN_VARIED_TEST((1 << 10) + 1, growByInitListCorrect, std::shared_ptr<int>);
}

template <typename CVec>
void growBySquaredCorrect(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.grow_by(num, std::make_shared<int>(10));
  }

  size_t result = 0;

  for (auto& v : vec) {
    result += *v;
  }

  EXPECT_EQ(result, 10 * vec.size());
  EXPECT_EQ(result, 10 * num * num);
}

TYPED_TEST(ConcurrentVectorTest, GrowBySquaredCorrectTiny) {
  RUN_VARIED_TEST(14, growBySquaredCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowBySquaredCorrectSmall) {
  RUN_VARIED_TEST(79, growBySquaredCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowBySquaredCorrectMedium) {
  RUN_VARIED_TEST(129, growBySquaredCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowBySquaredCorrectLarge) {
  RUN_VARIED_TEST(191, growBySquaredCorrect, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowBySquaredCorrectLargeP1) {
  RUN_VARIED_TEST(193, growBySquaredCorrect, std::shared_ptr<int>);
}

template <typename CVec>
void growByConcurrent(int num, int growBy, CVec& vec) {
  dispenso::parallel_for(
      dispenso::ChunkedRange(0, num, dispenso::ChunkedRange::Static()),
      [&vec, growBy](size_t i, size_t end) {
        while (i + growBy <= end) {
          vec.grow_by_generator(growBy, [i]() mutable { return std::make_unique<int>(i++); });
          i += growBy;
        }
        vec.grow_by_generator(end - i, [i]() mutable { return std::make_unique<int>(i++); });
      });

  EXPECT_EQ(vec.size(), num);

  std::vector<uint8_t> which(num);
  size_t idx = 0;
  for (auto& i : vec) {
    ++which[*i];
    ++idx;
  }

  for (size_t i = 0; i < which.size(); ++i) {
    EXPECT_EQ(1, which[i]) << "mismatch for index " << i;
  }
}

TYPED_TEST(ConcurrentVectorTest, GrowByConcurrentTiny) {
  RUN_VARIED_TEST2(14, 2, growByConcurrent, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConcurrentSmall) {
  RUN_VARIED_TEST2(79, 4, growByConcurrent, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConcurrentMedium) {
  RUN_VARIED_TEST2(200, 8, growByConcurrent, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConcurrentLarge) {
  RUN_VARIED_TEST2(1 << 8, 16, growByConcurrent, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, GrowByConcurrentLargeP1) {
  RUN_VARIED_TEST2((1 << 8) + 1, 1, growByConcurrent, std::unique_ptr<int>);
}

template <typename CVec>
void assignThenPush(int num, CVec& vec) {
  std::vector<std::shared_ptr<int>> vals;
  for (int i = 0; i < num; ++i) {
    vals.push_back(std::make_shared<int>(i));
  }

  vec.assign(std::begin(vals), std::end(vals));

  for (int i = num; i < 2 * num; ++i) {
    vec.push_back(std::make_shared<int>(i));
  }

  size_t result = 0;

  for (auto& v : vec) {
    result += *v;
  }

  size_t maxV = 2 * num - 1;

  EXPECT_EQ(result, (maxV * (maxV + 1)) / 2);
}

TYPED_TEST(ConcurrentVectorTest, AssignThenPushTiny) {
  RUN_VARIED_TEST(11, assignThenPush, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AssignThenPushSmall) {
  RUN_VARIED_TEST(81, assignThenPush, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AssignThenPushMedium) {
  RUN_VARIED_TEST(317, assignThenPush, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AssignThenPushLarge) {
  RUN_VARIED_TEST(1 << 9, assignThenPush, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AssignThenPushLargeP1) {
  RUN_VARIED_TEST((1 << 9) + 1, assignThenPush, std::shared_ptr<int>);
}

template <typename CVec>
void at(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  size_t result = 0;
  for (int i = 0; i < num; ++i) {
    result += *vec.at(i);
  }

  size_t maxV = num - 1;

  EXPECT_EQ(result, (maxV * (maxV + 1)) / 2);

#if defined(__cpp_exceptions)
  bool caught = false;
  try {
    EXPECT_EQ(0, *vec.at(num));
  } catch (const std::out_of_range& e) {
    caught = true;
  }

  EXPECT_TRUE(caught);
#endif // exceptions
}

TYPED_TEST(ConcurrentVectorTest, AtTiny) {
  RUN_VARIED_TEST(8, at, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AtSmall) {
  RUN_VARIED_TEST(65, at, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AtMedium) {
  RUN_VARIED_TEST(255, at, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AtLarge) {
  RUN_VARIED_TEST(1 << 9, at, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, AtLargeP2) {
  RUN_VARIED_TEST((1 << 9) + 2, at, std::unique_ptr<int>);
}

template <typename CVec>
void resize(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  vec.resize(num / 2);

  size_t result = 0;
  for (auto& v : vec) {
    if (v) {
      result += *v;
    }
  }

  size_t maxV = num / 2 - 1;

  EXPECT_EQ(result, (maxV * (maxV + 1)) / 2);

  // default init for new values should be null
  vec.resize(num);

  result = 0;
  for (auto& v : vec) {
    if (v) {
      result += *v;
    }
  }

  EXPECT_EQ(result, (maxV * (maxV + 1)) / 2);

  vec.resize(num * 2, std::make_shared<int>(5));

  result = 0;
  for (auto& v : vec) {
    if (v) {
      result += *v;
    }
  }

  EXPECT_EQ(result, 5 * num + (maxV * (maxV + 1)) / 2);
}

TYPED_TEST(ConcurrentVectorTest, ResizeTiny) {
  RUN_VARIED_TEST(9, resize, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ResizeSmall) {
  RUN_VARIED_TEST(63, resize, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ResizeMedium) {
  RUN_VARIED_TEST(256, resize, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ResizeLarge) {
  RUN_VARIED_TEST(1 << 10, resize, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, ResizeLargeP1) {
  RUN_VARIED_TEST((1 << 10) + 1, resize, std::shared_ptr<int>);
}

template <typename CVec>
void comparison(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(i);
  }

  EXPECT_EQ(vec, vec);

  auto vec2 = vec;

  EXPECT_GE(vec, vec2);
  EXPECT_LE(vec, vec2);

  EXPECT_FALSE(vec < vec2);
  EXPECT_FALSE(vec2 < vec);

  vec2[num - 1] = 2 * num;

  EXPECT_LE(vec, vec2);
  EXPECT_LT(vec, vec2);

  EXPECT_GE(vec2, vec);
  EXPECT_GT(vec2, vec);

  EXPECT_FALSE(vec > vec2);
  EXPECT_FALSE(vec >= vec2);
  EXPECT_FALSE(vec2 < vec);
  EXPECT_FALSE(vec2 <= vec);

  EXPECT_NE(vec, vec2);
  EXPECT_FALSE(vec == vec2);

  vec2.pop_back();

  EXPECT_LE(vec2, vec);
  EXPECT_LT(vec2, vec);

  EXPECT_GE(vec, vec2);
  EXPECT_GT(vec, vec2);

  EXPECT_FALSE(vec2 > vec);
  EXPECT_FALSE(vec2 >= vec);
  EXPECT_FALSE(vec < vec2);
  EXPECT_FALSE(vec <= vec2);

  EXPECT_NE(vec, vec2);
  EXPECT_FALSE(vec == vec2);

  vec2.push_back(num - 1);

  EXPECT_EQ(vec, vec2);
}

TYPED_TEST(ConcurrentVectorTest, ComparisonTiny) {
  RUN_VARIED_TEST(12, comparison, int);
}
TYPED_TEST(ConcurrentVectorTest, ComparisonSmall) {
  RUN_VARIED_TEST(77, comparison, int);
}
TYPED_TEST(ConcurrentVectorTest, ComparisonMedium) {
  RUN_VARIED_TEST(222, comparison, int);
}
TYPED_TEST(ConcurrentVectorTest, ComparisonLarge) {
  RUN_VARIED_TEST(1 << 10, comparison, int);
}
TYPED_TEST(ConcurrentVectorTest, ComparisonLargeP1) {
  RUN_VARIED_TEST((1 << 10) + 1, comparison, int);
}

template <typename V>
std::string printVec(const V& vec) {
  if (vec.size() > 20) {
    return "too long to print";
  }
  std::string ret;
  for (auto& v : vec) {
    ret += std::to_string(*v) + " ";
  }
  return ret;
}

std::string printVec(const dispenso::ConcurrentVector<int>& vec) {
  if (vec.size() > 20) {
    return "too long to print";
  }
  std::string ret;
  for (auto v : vec) {
    ret += std::to_string(v) + " ";
  }
  return ret;
}

std::string printVec(const std::vector<int>& vec) {
  if (vec.size() > 20) {
    return "too long to print";
  }
  std::string ret;
  for (auto v : vec) {
    ret += std::to_string(v) + " ";
  }
  return ret;
}

#define EXPECT_VEC_SUM(vec, num)                                                        \
  do {                                                                                  \
    size_t maxV = (num)-1;                                                              \
    size_t result = 0;                                                                  \
    for (auto& v : vec) {                                                               \
      result += *v;                                                                     \
    }                                                                                   \
    EXPECT_EQ(result, (maxV * (maxV + 1)) / 2)                                          \
        << "vec size is " << vec.size() << " num is " << (num) << " " << printVec(vec); \
  } while (false)

#define EXPECT_VEC_SUM_MINUS(vec, num, minus)                   \
  do {                                                          \
    size_t maxV = (num)-1;                                      \
    size_t result = 0;                                          \
    for (auto& v : vec) {                                       \
      result += *v;                                             \
    }                                                           \
    EXPECT_EQ(result, (maxV * (maxV + 1)) / 2 - (minus))        \
        << "vec size is " << vec.size() << " num is " << (num); \
  } while (false)

template <typename CVec>
void swap(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  CVec vec2;
  for (int i = 0; i < num / 2; ++i) {
    vec2.push_back(std::make_unique<int>(i));
  }

  EXPECT_VEC_SUM(vec, num);
  EXPECT_VEC_SUM(vec2, num / 2);

  swap(vec, vec2);

  EXPECT_VEC_SUM(vec, num / 2);
  EXPECT_VEC_SUM(vec2, num);

  vec.swap(vec2);

  EXPECT_VEC_SUM(vec, num);
  EXPECT_VEC_SUM(vec2, num / 2);
}

TYPED_TEST(ConcurrentVectorTest, SwapTiny) {
  RUN_VARIED_TEST(12, swap, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, SwapSmall) {
  RUN_VARIED_TEST(77, swap, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, SwapMedium) {
  RUN_VARIED_TEST(222, swap, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, SwapLarge) {
  RUN_VARIED_TEST(1 << 11, swap, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, SwapLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, swap, std::unique_ptr<int>);
}

TYPED_TEST(ConcurrentVectorTest, IterToConstIter) {
  dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam> vec;
  typename dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam>::iterator it = vec.begin();
  typename dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam>::const_iterator cit = it;
  EXPECT_EQ(cit, it);
}

template <typename CVec>
void eraseOne(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }
  // 0 1 2 3 4 5 6 7 8 9 10 11
  EXPECT_VEC_SUM(vec, num);

  ASSERT_EQ(vec.end(), vec.erase(vec.end()));

  // 0 1 2 3 4 5 6 7 8 9 10 11
  ASSERT_EQ(vec.size(), num);

  vec.erase(vec.end() - 1);

  // 0 1 2 3 4 5 6 7 8 9 10
  ASSERT_EQ(vec.size(), num - 1);

  EXPECT_VEC_SUM(vec, num - 1);

  vec.erase(vec.end() - 1);

  // 0 1 2 3 4 5 6 7 8 9
  EXPECT_VEC_SUM(vec, num - 2);

  vec.erase(vec.begin());

  // 1 2 3 4 5 6 7 8 9
  ASSERT_EQ(vec.size(), num - 3) << printVec(vec);

  EXPECT_VEC_SUM(vec, num - 2);

  vec.erase(vec.begin() + 1);

  // 1 3 4 5 6 7 8 9
  EXPECT_VEC_SUM_MINUS(vec, num - 2, 2);
}

TYPED_TEST(ConcurrentVectorTest, EraseOneTiny) {
  RUN_VARIED_TEST(12, eraseOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseOneSmall) {
  RUN_VARIED_TEST(77, eraseOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseOneMedium) {
  RUN_VARIED_TEST(222, eraseOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseOneLarge) {
  RUN_VARIED_TEST(1 << 11, eraseOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseOneLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, eraseOne, std::unique_ptr<int>);
}

template <typename CVec>
void eraseRange(int num, CVec& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
  }

  // 0 1 2 3 4 5 6 7 8 9 10 11
  EXPECT_VEC_SUM(vec, num);

  vec.erase(vec.end() - 1);

  // 0 1 2 3 4 5 6 7 8 9 10
  EXPECT_VEC_SUM(vec, num - 1);

  vec.erase(vec.end() - 5, vec.end());

  // 0 1 2 3 4 5
  EXPECT_VEC_SUM(vec, num - 6);

  vec.erase(vec.begin() + 1, vec.begin() + 3);

  // 0 3 4 5
  EXPECT_EQ(vec.size(), num - 8);

  EXPECT_VEC_SUM_MINUS(vec, num - 6, 3);

  //
  vec.erase(vec.begin(), vec.end());

  EXPECT_EQ(vec.size(), 0);
}

TYPED_TEST(ConcurrentVectorTest, EraseRangeTiny) {
  RUN_VARIED_TEST(12, eraseRange, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseRangeSmall) {
  RUN_VARIED_TEST(66, eraseRange, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseRangeMedium) {
  RUN_VARIED_TEST(111, eraseRange, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseRangeLarge) {
  RUN_VARIED_TEST(1 << 9, eraseRange, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, EraseRangeLargeP1) {
  RUN_VARIED_TEST((1 << 9) + 1, eraseRange, std::unique_ptr<int>);
}

#define EXPECT_CONTAINER_EQ(a, b)                                                \
  do {                                                                           \
    ASSERT_EQ(a.size(), b.size());                                               \
    for (size_t i = 0; i < a.size(); ++i) {                                      \
      EXPECT_EQ(a[i], *b[i]) << "index: " << i << "\n"                           \
                             << printVec(a) << "\n===========================\n" \
                             << printVec(b);                                     \
    }                                                                            \
  } while (0)

template <typename CVec>
void insertOne(int num, CVec& vec) {
  // Utilize std::vector as our baseline

  std::vector<int> baseline;
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_unique<int>(i));
    baseline.push_back(i);
  }

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(vec.begin(), std::make_unique<int>(5));
  baseline.insert(baseline.begin(), 5);

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(vec.begin() + 7, std::make_unique<int>(77));
  baseline.insert(baseline.begin() + 7, 77);

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(vec.end() - 1, std::make_unique<int>(33));
  baseline.insert(baseline.end() - 1, 33);

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(vec.end(), std::make_unique<int>(999));
  baseline.insert(baseline.end(), 999);

  EXPECT_CONTAINER_EQ(baseline, vec);
}

TYPED_TEST(ConcurrentVectorTest, InsertOneTiny) {
  RUN_VARIED_TEST(11, insertOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertOneSmall) {
  RUN_VARIED_TEST(40, insertOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertOneMedium) {
  RUN_VARIED_TEST(200, insertOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertOneLarge) {
  RUN_VARIED_TEST(1 << 10, insertOne, std::unique_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertOneLargeP1) {
  RUN_VARIED_TEST((1 << 10) + 1, insertOne, std::unique_ptr<int>);
}

template <typename CVec>
void insertRange(int num, CVec& vec) {
  // Utilize std::vector as our baseline

  std::vector<int> baseline;
  for (int i = 0; i < num; ++i) {
    vec.push_back(std::make_shared<int>(i));
    baseline.push_back(i);
  }

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(vec.begin(), 4, std::make_shared<int>(5));
  baseline.insert(baseline.begin(), 4, 5);

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(
      vec.begin() + 7,
      {std::make_shared<int>(1),
       std::make_shared<int>(2),
       std::make_shared<int>(3),
       std::make_shared<int>(4)});
  baseline.insert(baseline.begin() + 7, {1, 2, 3, 4});

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(
      vec.end() - 1,
      {std::make_shared<int>(1),
       std::make_shared<int>(2),
       std::make_shared<int>(3),
       std::make_shared<int>(4)});
  baseline.insert(baseline.end() - 1, {1, 2, 3, 4});

  EXPECT_CONTAINER_EQ(baseline, vec);

  vec.insert(
      vec.end(),
      {std::make_shared<int>(1),
       std::make_shared<int>(2),
       std::make_shared<int>(3),
       std::make_shared<int>(4)});
  baseline.insert(baseline.end(), {1, 2, 3, 4});

  EXPECT_CONTAINER_EQ(baseline, vec);
}

TYPED_TEST(ConcurrentVectorTest, InsertRangeTiny) {
  RUN_VARIED_TEST(13, insertRange, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertRangeSmall) {
  RUN_VARIED_TEST(111, insertRange, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertRangeMedium) {
  RUN_VARIED_TEST(600, insertRange, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertRangeLarge) {
  RUN_VARIED_TEST(1 << 11, insertRange, std::shared_ptr<int>);
}
TYPED_TEST(ConcurrentVectorTest, InsertRangeLargeP1) {
  RUN_VARIED_TEST((1 << 11) + 1, insertRange, std::shared_ptr<int>);
}

template <typename ContainerInit, typename ContainerPush>
void parallelImplGrowBy(
    size_t length,
    size_t growBy,
    ContainerInit containerInit,
    ContainerPush containerPush) {
  auto values = containerInit();
  dispenso::parallel_for(
      dispenso::ChunkedRange(0, length, dispenso::ChunkedRange::Static()),
      [&values, containerPush, growBy](size_t i, size_t end) {
        while (i + growBy <= end) {
          containerPush(values, i, i + growBy);
          i += growBy;
        }
        containerPush(values, i, end);
      });

  EXPECT_VEC_SUM(values, length);
}

TYPED_TEST(ConcurrentVectorTest, InsertRangeLargeGrowBy10) {
  parallelImplGrowBy(
      (1 << 18),
      10,
      []() { return dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam>(); },
      [](dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam>& c, int i, int end) {
        c.grow_by_generator(end - i, [i]() mutable { return std::make_unique<int>(i++); });
      });
}

TYPED_TEST(ConcurrentVectorTest, InsertRangeLargeGrowBy100) {
  parallelImplGrowBy(
      (1 << 18),
      100,
      []() { return dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam>(); },
      [](dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam>& c, int i, int end) {
        c.grow_by_generator(end - i, [i]() mutable { return std::make_unique<int>(i++); });
      });
}

struct NonMovable {
  int i;
  NonMovable(int val) : i(val) {}
  NonMovable(NonMovable&& other) = delete;
  NonMovable& operator=(NonMovable&& other) = delete;
};

TYPED_TEST(ConcurrentVectorTest, NonMovableObjects) {
  dispenso::ConcurrentVector<NonMovable, TypeParam> vec;
  for (int i = 0; i < 10; ++i) {
    vec.emplace_back(i);
  }
  int idx = 0;
  for (auto& nmv : vec) {
    EXPECT_EQ(nmv.i, idx++);
  }

  dispenso::ConcurrentVector<NonMovable, TypeParam> other(std::move(vec));

  idx = 0;
  for (auto& nmv : other) {
    EXPECT_EQ(nmv.i, idx++);
  }
}

TYPED_TEST(ConcurrentVectorTest, OtherConstructorsTest) {
  dispenso::ConcurrentVector<std::unique_ptr<int>, TypeParam> a(193);
  size_t remaining = 193;
  for (auto& ap : a) {
    EXPECT_FALSE(ap);
    --remaining;
  }
  EXPECT_EQ(remaining, 0);

  dispenso::ConcurrentVector<std::shared_ptr<int>, TypeParam> b(211, std::make_shared<int>(5));
  remaining = 211;
  for (auto& bp : b) {
    ASSERT_TRUE(bp);
    EXPECT_EQ(*bp, 5);
    --remaining;
  }
  EXPECT_EQ(remaining, 0);

  dispenso::ConcurrentVector<int, TypeParam> c({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  int v = 0;
  for (auto cv : c) {
    EXPECT_EQ(cv, v++);
  }
}
