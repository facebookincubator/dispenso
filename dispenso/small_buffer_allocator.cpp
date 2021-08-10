#include <dispenso/detail/small_buffer_allocator_impl.h>
#include <dispenso/small_buffer_allocator.h>

#include <new>

namespace dispenso {
namespace detail {
static int smallBufferSchwarzCounter = 0;

#define SMALL_BUFFER_GLOBALS_DECL(N)                           \
  static AlignedBuffer<SmallBufferGlobals> g_globalsBuffer##N; \
  static SmallBufferGlobals& g_globals##N =                    \
      reinterpret_cast<SmallBufferGlobals&>(g_globalsBuffer##N)

SMALL_BUFFER_GLOBALS_DECL(8);
SMALL_BUFFER_GLOBALS_DECL(16);
SMALL_BUFFER_GLOBALS_DECL(32);
SMALL_BUFFER_GLOBALS_DECL(64);
SMALL_BUFFER_GLOBALS_DECL(128);
SMALL_BUFFER_GLOBALS_DECL(256);

#define SMALL_BUFFER_GLOBAL_FUNC_DEFS(N)           \
  template <>                                      \
  SmallBufferGlobals& getSmallBufferGlobals<N>() { \
    return g_globals##N;                           \
  }

SMALL_BUFFER_GLOBAL_FUNC_DEFS(8)
SMALL_BUFFER_GLOBAL_FUNC_DEFS(16)
SMALL_BUFFER_GLOBAL_FUNC_DEFS(32)
SMALL_BUFFER_GLOBAL_FUNC_DEFS(64)
SMALL_BUFFER_GLOBAL_FUNC_DEFS(128)
SMALL_BUFFER_GLOBAL_FUNC_DEFS(256)

SchwarzSmallBufferInit::SchwarzSmallBufferInit() {
  if (smallBufferSchwarzCounter++ == 0) {
    ::new (&g_globals8) SmallBufferGlobals();
    ::new (&g_globals16) SmallBufferGlobals();
    ::new (&g_globals32) SmallBufferGlobals();
    ::new (&g_globals64) SmallBufferGlobals();
    ::new (&g_globals128) SmallBufferGlobals();
    ::new (&g_globals256) SmallBufferGlobals();
  }
}
SchwarzSmallBufferInit::~SchwarzSmallBufferInit() {
  if (--smallBufferSchwarzCounter == 0) {
    g_globals8.~SmallBufferGlobals();
    g_globals16.~SmallBufferGlobals();
    g_globals32.~SmallBufferGlobals();
    g_globals64.~SmallBufferGlobals();
    g_globals128.~SmallBufferGlobals();
    g_globals256.~SmallBufferGlobals();
  }
}

char* allocSmallBufferImpl(size_t ordinal) {
  switch (ordinal) {
    case 0:
      return detail::SmallBufferAllocator<8>::alloc();
    case 1:
      return detail::SmallBufferAllocator<16>::alloc();
    case 2:
      return detail::SmallBufferAllocator<32>::alloc();
    case 3:
      return detail::SmallBufferAllocator<64>::alloc();
    case 4:
      return detail::SmallBufferAllocator<128>::alloc();
    case 5:
      return detail::SmallBufferAllocator<256>::alloc();
    default:
      assert(false && "Invalid small buffer ordinal requested");
      return nullptr;
  }
}

void deallocSmallBufferImpl(size_t ordinal, void* buf) {
  switch (ordinal) {
    case 0:
      detail::SmallBufferAllocator<8>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 1:
      detail::SmallBufferAllocator<16>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 2:
      detail::SmallBufferAllocator<32>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 3:
      detail::SmallBufferAllocator<64>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 4:
      detail::SmallBufferAllocator<128>::dealloc(reinterpret_cast<char*>(buf));
      break;
    case 5:
      detail::SmallBufferAllocator<256>::dealloc(reinterpret_cast<char*>(buf));
      break;
    default:
      assert(false && "Invalid small buffer ordinal requested");
  }
}

size_t approxBytesAllocatedSmallBufferImpl(size_t ordinal) {
  switch (ordinal) {
    case 0:
      return detail::SmallBufferAllocator<8>::bytesAllocated();
    case 1:
      return detail::SmallBufferAllocator<16>::bytesAllocated();
    case 2:
      return detail::SmallBufferAllocator<32>::bytesAllocated();
    case 3:
      return detail::SmallBufferAllocator<64>::bytesAllocated();
    case 4:
      return detail::SmallBufferAllocator<128>::bytesAllocated();
    case 5:
      return detail::SmallBufferAllocator<256>::bytesAllocated();
    default:
      assert(false && "Invalid small buffer ordinal requested");
      return 0;
  }
}

template class SmallBufferAllocator<8>;
template class SmallBufferAllocator<16>;
template class SmallBufferAllocator<32>;
template class SmallBufferAllocator<64>;
template class SmallBufferAllocator<128>;
template class SmallBufferAllocator<256>;

} // namespace detail
} // namespace dispenso
