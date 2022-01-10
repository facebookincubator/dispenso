// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

/**
 * @file tsan_annotations.h
 * This file exposes a set of macros for ignoring tsan errors.  These should generally not
 * be used just to shut up TSAN, because most of the time, TSAN reports real bugs.  They should be
 * used only when there is a high level of certainty that TSAN is spitting out a false positive, as
 * can occasionally happen with lock-free algorithms.
 *
 * When these are required, it is best to keep the scope as small as possible to avoid blinding TSAN
 * to real bugs. Note that several libraries already expose macros like these, but we want to
 * keep dependencies to a bare minimum.
 **/

#pragma once

#include <dispenso/platform.h>

#if defined(__SANITIZE_THREAD__)
#define DISPENSO_HAS_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define DISPENSO_HAS_TSAN 1
#else
#define DISPENSO_HAS_TSAN 0
#endif // TSAN
#else
#define DISPENSO_HAS_TSAN 0
#endif // feature

#if DISPENSO_HAS_TSAN

namespace dispenso {
namespace detail {

DISPENSO_DLL_ACCESS void annotateIgnoreWritesBegin(const char* f, int l);
DISPENSO_DLL_ACCESS void annotateIgnoreWritesEnd(const char* f, int l);
DISPENSO_DLL_ACCESS void annotateIgnoreReadsBegin(const char* f, int l);
DISPENSO_DLL_ACCESS void annotateIgnoreReadsEnd(const char* f, int l);
DISPENSO_DLL_ACCESS void
annotateNewMemory(const char* f, int l, const volatile void* address, long size);
DISPENSO_DLL_ACCESS void annotateHappensBefore(const char* f, int l, const volatile void* address);
DISPENSO_DLL_ACCESS void annotateHappensAfter(const char* f, int l, const volatile void* address);

} // namespace detail
} // namespace dispenso

#define DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN() \
  ::dispenso::detail::annotateIgnoreWritesBegin(__FILE__, __LINE__)
#define DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END() \
  ::dispenso::detail::annotateIgnoreWritesEnd(__FILE__, __LINE__)
#define DISPENSO_TSAN_ANNOTATE_IGNORE_READS_BEGIN() \
  ::dispenso::detail::annotateIgnoreReadsBegin(__FILE__, __LINE__)
#define DISPENSO_TSAN_ANNOTATE_IGNORE_READS_END() \
  ::dispenso::detail::annotateIgnoreReadsEnd(__FILE__, __LINE__)
#define DISPENSO_TSAN_ANNOTATE_NEW_MEMORY(address, size) \
  ::dispenso::detail::annotateNewMemory(__FILE__, __LINE__, address, size)
#define DISPENSO_TSAN_ANNOTATE_HAPPENS_BEFORE(address) \
  ::dispenso::detail::annotateHappensBefore(__FILE__, __LINE__, address)
#define DISPENSO_TSAN_ANNOTATE_HAPPENS_AFTER(address) \
  ::dispenso::detail::annotateHappensAfter(__FILE__, __LINE__, address)

#else

#define DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN()
#define DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END()
#define DISPENSO_TSAN_ANNOTATE_IGNORE_READS_BEGIN()
#define DISPENSO_TSAN_ANNOTATE_IGNORE_READS_END()
#define DISPENSO_TSAN_ANNOTATE_NEW_MEMORY(address, size)
#define DISPENSO_TSAN_ANNOTATE_HAPPENS_BEFORE(address)
#define DISPENSO_TSAN_ANNOTATE_HAPPENS_AFTER(address)

#endif // DISPENSO_HAS_TSAN
