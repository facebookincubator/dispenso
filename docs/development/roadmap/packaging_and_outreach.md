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

## Upstream contributions

Local patches carried against vendored dependencies. Each one has to be
re-applied by hand every time the dependency is updated, and nothing forces
that to happen, so getting them upstream is how the cost actually goes away.
The patches themselves are listed in the vendored directory's `README.txt`.

| Patch | Upstream | Status | Notes |
|-------|----------|--------|-------|
| `override` on `~ExplicitProducer` / `~ImplicitProducer` | cameron314/concurrentqueue | Not submitted | Both override `virtual ~ProducerBase()`. Clang's `-Winconsistent-missing-destructor-override` makes this fatal under our `-Werror`. Two lines, no behaviour change, still absent in v1.0.5 — the cheapest one to eliminate. |
| clang `-Wglobal-constructors` suppression | cameron314/concurrentqueue | Needs triage | Establish whether it is still needed before submitting anything. It is not enabled by `-Wall`/`-Wextra`, and our Buck build suppresses it separately, so the patch may be obsolete — in which case dropping it beats upstreaming it. |

## Backlog

- Integration examples (game engines, scientific computing)
- Discord/Slack community channel
