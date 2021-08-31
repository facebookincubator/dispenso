#include <dispenso/detail/per_thread_info.h>

namespace dispenso {
namespace detail {

PerThreadInfo& PerPoolPerThreadInfo::info() {
  static DISPENSO_THREAD_LOCAL PerThreadInfo perThreadInfo;
  return perThreadInfo;
}

} // namespace detail
} // namespace dispenso
