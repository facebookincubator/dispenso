# Release Checklist

Post-release tasks and reminders for package manager updates.

## vcpkg: Remove temporary patches

The v1.5.0 vcpkg port (`microsoft/vcpkg` PR #49633) includes two workarounds
for upstream bugs. Remove them once the release includes the fixes:

1. **`fix-arm64-platform-define.patch`** — `notifier_common.h` defined `_ARM_`
   instead of `_ARM64_` on ARM64 Windows, causing `winnt.h` compilation
   failures. Fixed on `main`. Remove the patch file and the `PATCHES` block
   from `portfile.cmake`.

2. **`-DDISPENSO_SHARED_LIB=${DISPENSO_SHARED}`** — dispenso's
   `DISPENSO_SHARED_LIB` option ignored `BUILD_SHARED_LIBS`, producing DLLs in
   static triplets. Fixed on `main` (defaults to `BUILD_SHARED_LIBS` when set).
   Remove the `string(COMPARE EQUAL ...)` line and the `-DDISPENSO_SHARED_LIB`
   option from `portfile.cmake`.
