/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/tsan_annotations.h>

#if DISPENSO_HAS_TSAN

#ifdef __GNUC__
#define ATTRIBUTE_WEAK __attribute__((weak))
#else
#define ATTRIBUTE_WEAK
#endif

// These are found in the accompanying libtsan, but there is no header exposing them.  We want to
// also avoid exposing them in a header to to discourage folks from calling them directly.
extern "C" {
void AnnotateIgnoreReadsBegin(const char* f, int l) ATTRIBUTE_WEAK;

void AnnotateIgnoreReadsEnd(const char* f, int l) ATTRIBUTE_WEAK;

void AnnotateIgnoreWritesBegin(const char* f, int l) ATTRIBUTE_WEAK;

void AnnotateIgnoreWritesEnd(const char* f, int l) ATTRIBUTE_WEAK;

void AnnotateNewMemory(const char* f, int l, const volatile void* address, long size)
    ATTRIBUTE_WEAK;

void AnnotateHappensBefore(const char* f, int l, const volatile void* address) ATTRIBUTE_WEAK;
void AnnotateHappensAfter(const char* f, int l, const volatile void* address) ATTRIBUTE_WEAK;
}

namespace dispenso {
namespace detail {

void annotateIgnoreWritesBegin(const char* f, int l) {
  AnnotateIgnoreWritesBegin(f, l);
}
void annotateIgnoreWritesEnd(const char* f, int l) {
  AnnotateIgnoreWritesEnd(f, l);
}
void annotateIgnoreReadsBegin(const char* f, int l) {
  AnnotateIgnoreReadsBegin(f, l);
}
void annotateIgnoreReadsEnd(const char* f, int l) {
  AnnotateIgnoreReadsEnd(f, l);
}

void annotateNewMemory(const char* f, int l, const volatile void* address, long size) {
  AnnotateNewMemory(f, l, address, size);
}

void annotateHappensBefore(const char* f, int l, const volatile void* address) {
  AnnotateHappensBefore(f, l, address);
}

void annotateHappensAfter(const char* f, int l, const volatile void* address) {
  AnnotateHappensAfter(f, l, address);
}

} // namespace detail
} // namespace dispenso

#endif // TSAN
