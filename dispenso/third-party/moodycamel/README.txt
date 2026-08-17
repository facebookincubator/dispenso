https://github.com/cameron314/concurrentqueue

Vendored at tag v1.0.5, commit 9afb99746f0f5fc94ac8aef737053ae0481ba8d1.

Files: concurrentqueue.h, blockingconcurrentqueue.h, lightweightsemaphore.h,
LICENSE.md. They include each other with relative quotes, so only the entry
points matter to callers.

------------------------------------------------------------------------------
NO LOCAL MODIFICATIONS -- keep it that way
------------------------------------------------------------------------------

These files are byte-identical to upstream. Updating is a straight overwrite:
drop in the new release, update the ref above, rebuild.

dispenso adds this directory to the build as a SYSTEM include (see the
DISPENSO_USE_SYSTEM_CONCURRENTQUEUE branch in ../../CMakeLists.txt), so the
compiler does not diagnose it and our warning flags cannot force a patch here.

Earlier versions did carry three local edits -- `override` on two producer
destructors, and a clang -Wglobal-constructors suppression -- purely to keep
-Werror quiet. They were removed once the SYSTEM include made them
unnecessary, and nothing here should need patching again for a warning. If a
future change genuinely requires modifying these files, send it upstream
rather than carrying it: a patch that only silences a diagnostic is not worth
re-applying on every update, and one that changes behaviour belongs upstream
anyway.
