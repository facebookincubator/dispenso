/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/small_buffer_allocator_impl.h>
#include <dispenso/small_buffer_allocator.h>

#include <new>

namespace dispenso {
namespace detail {

template <size_t kChunkSize>
SmallBufferGlobals& getSmallBufferGlobals() {
  // controlled leak here
  static SmallBufferGlobals* globals = new SmallBufferGlobals();
  return *globals;
}

char* allocSmallBufferImpl(size_t ordinal) {
  switch (ordinal) {
    case 0:
      return detail::SmallBufferAllocator<4>::alloc();
    case 1:
      return detail::SmallBufferAllocator<8>::alloc();
    case 2:
      return detail::SmallBufferAllocator<16>::alloc();
    case 3:
      return detail::SmallBufferAllocator<32>::alloc();
    case 4:
      return detail::SmallBufferAllocator<64>::alloc();
    case 5:
      return detail::SmallBufferAllocator<128>::alloc();
    case 6:
      return detail::SmallBufferAllocator<256>::alloc();
    default:
      assert(false && "Invalid small buffer ordinal requested");
      return nullptr;
  }
}

void deallocSmallBufferImpl(size_t ordinal, void* buf) {
  switch (ordinal) {
    case 0:
      detail::SmallBufferAllocator<4>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 1:
      detail::SmallBufferAllocator<8>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 2:
      detail::SmallBufferAllocator<16>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 3:
      detail::SmallBufferAllocator<32>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 4:
      detail::SmallBufferAllocator<64>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 5:
      detail::SmallBufferAllocator<128>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 6:
      detail::SmallBufferAllocator<256>::dealloc(reinterpret_cast<char*>(buf));
      break;
    default:
      assert(false && "Invalid small buffer ordinal requested");
  }
}

size_t approxBytesAllocatedSmallBufferImpl(size_t ordinal) {
  switch (ordinal) {
    case 0:
      return detail::SmallBufferAllocator<4>::bytesAllocated();
    case 1:
      return detail::SmallBufferAllocator<8>::bytesAllocated();
    case 2:
      return detail::SmallBufferAllocator<16>::bytesAllocated();
    case 3:
      return detail::SmallBufferAllocator<32>::bytesAllocated();
    case 4:
      return detail::SmallBufferAllocator<64>::bytesAllocated();
    case 5:
      return detail::SmallBufferAllocator<128>::bytesAllocated();
    case 6:
      return detail::SmallBufferAllocator<256>::bytesAllocated();
    default:
      assert(false && "Invalid small buffer ordinal requested");
      return 0;
  }
}

template <size_t kChunkSize>
SmallBufferAllocator<kChunkSize>::PerThreadQueuingData::~PerThreadQueuingData() {
  enqueue_bulk(buffers_, count_);

  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
  ptoken().~ProducerToken();
  ctoken().~ConsumerToken();
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
}

template class SmallBufferAllocator<4>;
template class SmallBufferAllocator<8>;
template class SmallBufferAllocator<16>;
template class SmallBufferAllocator<32>;
template class SmallBufferAllocator<64>;
template class SmallBufferAllocator<128>;
template class SmallBufferAllocator<256>;

} // namespace detail
} // namespace dispenso
