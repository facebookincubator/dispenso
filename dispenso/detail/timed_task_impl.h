/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/thread_pool.h>

namespace dispenso {
namespace detail {

enum FunctionFlags : uint32_t { kFFlagsNone = 0, kFFlagsDetached = 1, kFFlagsCancelled = 2 };

struct TimedTaskImpl {
  alignas(kCacheLineSize) std::atomic<size_t> count{0};
  std::atomic<size_t> timesToRun;
  std::atomic<uint32_t> flags{kFFlagsNone};
  std::atomic<uint32_t> inProgress{0};
  double nextAbsTime;
  double period;
  bool steady;
  std::function<void(std::shared_ptr<TimedTaskImpl>)> func;

  template <typename F, typename Schedulable>
  TimedTaskImpl(size_t times, double next, double per, F&& f, Schedulable& sched, bool stdy)
      : timesToRun(times), nextAbsTime(next), period(per), steady(stdy) {
    func = [&sched, f = std::move(f), this](std::shared_ptr<TimedTaskImpl> me) {
      if (flags.load(std::memory_order_acquire) & kFFlagsCancelled) {
        return;
      }

      inProgress.fetch_add(1, std::memory_order_acq_rel);

      auto wrap = [&f, this, me = std::move(me)]() mutable {
        if (!(flags.load(std::memory_order_acquire) & kFFlagsCancelled)) {
          if (!f()) {
            timesToRun.store(0, std::memory_order_release);
            flags.fetch_or(kFFlagsCancelled, std::memory_order_acq_rel);
            func = {};
          }
          count.fetch_add(1, std::memory_order_acq_rel);
        }

        inProgress.fetch_sub(1, std::memory_order_release);
        me.reset();
      };
      sched.schedule(wrap, ForceQueuingTag());
    };
  }
};

} // namespace detail
} // namespace dispenso
