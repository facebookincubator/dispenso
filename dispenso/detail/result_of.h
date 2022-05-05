/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <type_traits>

namespace dispenso {
namespace detail {

#if __cplusplus < 201703L
template <typename F, typename... Args>
using ResultOf =
    typename std::result_of<typename std::decay<F>::type(typename std::decay<Args>::type...)>::type;
#else
template <typename F, typename... Args>
using ResultOf = typename std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
#endif // c++17

} // namespace detail
} // namespace dispenso
