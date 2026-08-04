# Release Checklist

Post-release tasks and reminders for package manager updates.

## Compiler Explorer (Godbolt)

Compiler Explorer pins each library version to an explicit git ref; it does not
track HEAD. Each release must be registered so the README "Try on Godbolt" link
offers the new version.

On each release (after the `vX.Y.Z` tag exists on GitHub):

1. Add the new tag as a version in the Compiler Explorer library config via PRs
   to `compiler-explorer/infra` (the build recipe) and
   `compiler-explorer/compiler-explorer` (the version list). Follow their
   current library-onboarding docs for exact file locations.
2. dispenso is **not** header-only — it uses the compiled-library path (a build
   recipe producing `libdispenso` per compiler), not just an include path. This
   is a one-time onboarding; subsequent releases only add the new version.
3. After the PRs merge, confirm the new version is selectable and that a minimal
   `parallel_for` example compiles and runs on godbolt.org.

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
