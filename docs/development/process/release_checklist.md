# Release Checklist

## Cutting the tag

A published tag cannot be walked back. The package manager ports pin the SHA-256
of the tarball GitHub generates for it, so moving a tag invalidates hashes that
are already recorded downstream. If a tag turns out to be wrong, release a new
patch version rather than repointing the old one.

1. Confirm CI is green on the commit you intend to tag. Nothing runs on `main`
   pushes for a specific past commit, and the export keeps moving `main`, so tag
   the commit CI validated rather than whatever `main` currently points at.
2. Check the vendored dependency against upstream. `dispenso/third-party/moodycamel/`
   is a copy of [cameron314/concurrentqueue](https://github.com/cameron314/concurrentqueue);
   `README.txt` records the upstream ref it came from. Compare that against
   upstream's latest release, and check it still matches the version the
   conan-center recipe requires — the two can drift apart silently, because
   conan builds with `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON` and never touches
   the bundled copy, while vcpkg and source builds use nothing else. Nothing
   forces this comparison, which is exactly how the bundled copy reached 1.6.0
   four years behind upstream while conan users were on a different queue
   entirely. Updating it is a scheduling-hot-path change: run the full suite
   under ASAN and TSAN, and do it in its own release rather than alongside
   other work.
3. Create the tag:
   ```bash
   python3 scripts/release.py tag --version X.Y.Z --commit <sha>
   ```
   This verifies that all seven places the version appears agree with each other
   — `CHANGELOG.md`, `dispenso/platform.h`, `CMakeLists.txt`, `METADATA.bzl`,
   `docs/Doxyfile`, `docs/faq.md` and `README.md` — and refuses to tag if any
   disagree. 1.5.0 shipped with the `platform.h` macros still reading 1.4.1;
   this is the check that would have caught it. It also rejects a changelog date
   that is not a specific day, because a placeholder month reads like a real
   heading and survives review.

   The tag is signed and annotated by default, and is *not* pushed. Review it
   with `git show vX.Y.Z` first.
4. `git push origin vX.Y.Z`. The `Release` workflow then re-runs the same checks
   against the tagged tree, requires the tag to be annotated, extracts the
   changelog section for that version, and publishes the GitHub release.

## Post-release

Package manager updates and reminders.

## Package manager updates

`scripts/update_package_managers.py` drives conan, vcpkg, homebrew and
macports. It downloads the release tarball, computes sha256/sha512/rmd160 and
size, then for each manager clones the repo if missing, rewrites the port,
runs the ecosystem's own tooling and tests, pushes to your fork, and opens a
PR.

```bash
python3 scripts/update_package_managers.py --version X.Y.Z --guided
```

`--guided` prompts before each manager and lets you skip or quit; local
commits survive a quit. Useful variants: `--dry-run` to see the plan without
touching anything, `--managers conan,vcpkg` to do a subset, and `--skip-push`
to stop after committing and testing.

### Before you start

- Public network access. None of this runs from a devserver — the script
  downloads the release tarball and clones upstream package repos.
- macOS if you end up touching macports. On Linux the script degrades homebrew
  and macports to checksum verification only; vcpkg and conan are unaffected,
  so a Linux machine with public network access is enough for those.
- `gh` installed and authenticated (`gh auth login`). The guided flow opens
  PRs through it.
- The script clones into `--repos-dir` (default `~/repos`) and pushes to the
  fork named by `--github-user` (default `graphicsMan`). Override both if
  either is wrong for the machine. conan-center-index and vcpkg are large
  clones — budget disk and time.
- **Refresh the clones before you start, and ideally between releases.** Run
  `git -C ~/repos/vcpkg fetch origin` and the same for
  `~/repos/conan-center-index`, well ahead of the release rather than during
  it. Both are enormous, and after a few months untouched the catch-up fetch
  is large enough that GitHub rate-limits it: doing both back to back during
  the 1.6.0 update earned an HTTP 429 on the git endpoint that blocked all git
  traffic for roughly half an hour. It is not an API-quota problem — `gh api
  rate_limit` will look untouched, and the cheap `info/refs` endpoint keeps
  returning 200 while the `git-upload-pack` RPC is refused, so the only honest
  probe is retrying the real operation. There is nothing to do but wait it
  out, and nothing partial is left behind: the fetch fails before any file is
  touched, and the script is safe to re-run.

### Which ports actually need us

Not all four are ours to push. Who bumped each one to 1.5.1 is the best guide
we have:

| Port | Who bumped it to 1.5.1 | Lag |
|------|------------------------|-----|
| Homebrew | `BrewTestBot`, automated | same day |
| vcpkg | a community contributor (PR #50867) | 1 day |
| MacPorts | us | 1 day |
| Conan | us (PR #29887) | ~1 month |

- **vcpkg — by hand, and early.** Someone else will usually bump the version
  within a day or two, but they will not remove our workarounds: the 1.5.1
  bump left `DISPENSO_SHARED_LIB` exactly where it was. Getting there first is
  the only way a version bump and a cleanup land in the same PR.
- **Conan — by hand.** Every bump since the recipe was contributed in 2024 has
  been ours, and the one time we waited it took a month.
- **Homebrew — nothing to do.** `BrewTestBot` handles it.
- **MacPorts — wait, then decide.** The Portfile has read `nomaintainer` since
  April 2026, and the community has updated it before (1.3.0, 1.4.1). Run the
  script for it only if nobody has moved after a few days.

Conda-forge and Fedora/RHEL appear in the README's install list but have never
been ours; their own packagers track releases. Nothing to do, and nothing to
chase.

**Do not start a release's port round until the previous one has merged
upstream.** Branches are per-version, and `checkout_branch` creates each one
fresh from the upstream default branch — so if the previous release's PR is
still open, the new branch starts from a port that has none of its changes and
silently reverts them. That is not hypothetical: the 1.6.0 vcpkg PR carried a
devendoring, two workaround removals and a `usage`-file deletion, none of which
existed upstream while the PR sat in review. Starting 1.6.1 against that would
have undone all four and looked like a regression to the reviewer who had just
asked for them.

**Do not put a version on `find_package(concurrentqueue ...)`.** It looks like
the right way to keep the system copy in step with the bundled one, and it
cannot work. 1.6.1 tried it and had to be superseded by 1.6.2 within a day.

concurrentqueue's own `CMakeLists.txt` has said `project(concurrentqueue VERSION
1.0.0)` at every release, v1.0.5 included, so the `concurrentqueueConfigVersion.cmake`
it installs always reports `1.0.0` — a number matching no release; there has
never been a `v1.0.0` tag. Any version argument is therefore compared against a
constant that is wrong, and `find_package(concurrentqueue 1.0.5 CONFIG REQUIRED)`
is rejected by a correctly installed 1.0.5. The header carries no version macro
either, so there is nothing else to test against.

The underlying concern is real: `moodycamel::ConcurrentQueue` is a by-value
member of `ThreadPool`, so its layout is part of dispenso's ABI, and a system
copy that disagrees with the bundled one is an ODR problem. It simply cannot be
expressed through this package. If it ever has to be enforced, it needs a
mechanism that does not rely on the dependency's own version metadata.

Note this bites unevenly, which is what makes it easy to miss: conan is
unaffected, because CMakeDeps generates the version file from the *conan
package* version rather than the upstream project's, so it reports the real
1.0.5. vcpkg, and anyone consuming upstream's own CMake install, gets 1.0.0.
Testing only the conan round would have shown all green.

### Before trusting a green run

- **Hashes are pinned against the tag's tarball.** This is why a published tag
  must never be moved: all four ports record a hash of the archive GitHub
  generates for it.
- **The script only reasons about `.patch` files.** It detects a patch whose
  fix has been upstreamed, but it has no notion of a workaround written
  directly into a port's build script. Those have to be found and removed by
  hand — see the vcpkg section for the standing example.

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

Note that dispenso has not been onboarded yet, so the first pass is that
one-time setup rather than a version bump — and there is no "Try on Godbolt"
link in `README.md` to update, because none exists. Adding it is part of the
onboarding.
