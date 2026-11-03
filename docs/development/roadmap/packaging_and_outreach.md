<!-- Part of the dispenso roadmap; index at ../roadmap.md -->

# Roadmap: release, packaging, and outreach

Package managers, listings, and the tooling around publishing a release.
Per-release steps live in [process/release_checklist.md](../process/release_checklist.md).

## Planned

| Feature | Description | Priority |
|---------|-------------|----------|
| Benchmark automation | Script to run benchmarks and generate charts | High |
| Compiler Explorer examples | Godbolt links in README for try-it-now experience | High |

## External submissions

| Target | Status | Notes |
|--------|--------|-------|
| awesome-cpp | Listed | fffaraz/awesome-cpp |
| awesome-modern-cpp | Listed | rigtorp/awesome-modern-cpp |
| awesome-high-performance-computing | Listed | Already present in dstansby/awesome-high-performance-computing |
| awesome-scientific-computing | Not applicable | Focus is numerical methods, not parallelism libraries |
| awesome-hpc | Not applicable | Focus is cluster infrastructure, not app-level parallelism |
| vcpkg | In progress | PR to microsoft/vcpkg |
| Conan | In progress | PR to conan-center-index |

## Patches against vendored dependencies

**We carry none, and adding one should be a deliberate decision.** The bundled
moodycamel is byte-identical to upstream, so updating it is a straight
overwrite.

It was not always so. Three local edits accumulated in `concurrentqueue.h` —
`override` on two producer destructors, and a clang `-Wglobal-constructors`
suppression — every one of them there only to keep a warning quiet under
`-Werror`. Nothing recorded that they existed, so they had to be rediscovered
by diffing against the upstream ref, and the v1.0.5 update nearly dropped them
silently.

They are gone because the directory is now a SYSTEM include: the compiler does
not diagnose code we do not maintain, so our warning flags cannot force a patch
there. That covers any warning we enable in future, not just the three we
happened to hit.

The rule that falls out of it, for the next person tempted to patch a vendored
header:

- **A patch that only silences a diagnostic is not worth carrying.** It says
  nothing about dispenso, and it has to be re-applied on every update forever.
  Mark the include SYSTEM instead.
- **A patch that changes behaviour belongs upstream**, and should be sent there
  rather than vendored — otherwise we own a fork of someone else's library.

## Backlog

- Integration examples (game engines, scientific computing)
- Discord/Slack community channel
- Benchmark dashboard follow-ups:
  - mandelbrot is drawn as one bar per library, region, thread count and grain size, like spmm was. Its names are `<region>/threads/grain`, so it needs thread-scaling charts that keep the grain sizes apart, not just the spmm parsing.
  - In every rw_lock chart, on all four platforms, each lock is missing results at some thread counts, and `std::shared_mutex` and `dispenso::RWLock` are missing different ones. Find out why the benchmark's thread sweeps differ.
  - On macOS (M4 Pro) the plain binary tree builds fastest serially, about 2.4x ahead of every parallel version, TBB included. The tree sizes may be too small for that machine to be a useful comparison.

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **CI coverage for `-fno-exceptions`** (`tests/CMakeLists.txt`). The library guards its throwing paths with `__cpp_exceptions`, but no CI job builds without exceptions.
