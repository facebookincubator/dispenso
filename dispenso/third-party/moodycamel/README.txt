https://github.com/cameron314/concurrentqueue

Vendored at tag v1.0.5, commit 9afb99746f0f5fc94ac8aef737053ae0481ba8d1.

Files: concurrentqueue.h, blockingconcurrentqueue.h, lightweightsemaphore.h,
LICENSE.md. They include each other with relative quotes, so only the entry
points matter to callers.

------------------------------------------------------------------------------
LOCAL MODIFICATIONS -- re-apply all of these when updating
------------------------------------------------------------------------------

All three are in concurrentqueue.h, all exist only to survive dispenso's
-Werror, and none change behaviour. Upstream has taken none of them as of
v1.0.5, so an update is "overwrite, then re-apply". Diff against the upstream
ref above before assuming this list is complete.

1. `override` on ~ExplicitProducer(), and
2. `override` on ~ImplicitProducer().

   Both derive from ProducerBase, which declares `virtual ~ProducerBase()`, so
   both implicitly override a virtual destructor. Clang reports that via
   -Winconsistent-missing-destructor-override, which -Werror makes fatal.

3. Inside the existing __GNUC__ diagnostic block, after the -Wconversion
   pragma:

       #if defined(__clang__)
       #pragma GCC diagnostic ignored "-Wglobal-constructors"
       #endif //__clang__

   Status uncertain -- it may be obsolete. -Wglobal-constructors is not in
   -Wall or -Wextra, so dispenso's own CMake flags never enable it, and the
   Buck build suppresses it separately (-Wno-global-constructors,
   -Wno-error=global-constructors). It may predate a change that removed the
   construct it was guarding. See the "Upstream contributions" table in
   docs/development/roadmap/packaging_and_outreach.md.

Getting 1 and 2 accepted upstream would retire them permanently; both are
two-line changes with no behavioural effect.
