# Release candidate tags

**Context.** dispenso releases go straight from a version-bump commit to a
signed annotated tag, which the `Release` workflow turns into a published
GitHub release. The package manager ports are then updated from that tag. There
is no intermediate artifact: the first tarball anyone builds is the one the
ports pin a SHA-256 of, and a published tag cannot be walked back.

1.6.1 was superseded by 1.6.2 within a day, which raised the question below.

## The question

Would a release-candidate flow -- `v1.6.1-rc1`, exercised end to end before
cutting `v1.6.1` -- have prevented the release failures we have actually had?

Answering it needs the record rather than intuition, because the honest answer
changes as the other safety nets improve.

## The record

Each row is a failure that reached a tag or shipped in one. The last three
columns ask which mechanism would have caught it *before* the real tag existed.

| Release | Failure | RC + port round | `release.py check` | `preflight` |
|---|---|---|---|---|
| 1.5.0 | `platform.h` macros still read 1.4.1 | no | **yes** | yes |
| 1.6.0 | changelog date was a placeholder month | no | **yes** | yes |
| 1.6.0 | bundled moodycamel four years behind upstream | no | no | no |
| 1.6.1 | annotated tag rejected as lightweight by the release workflow | **yes** | no | no |
| 1.6.1 | `find_package(concurrentqueue 1.0.5)` unsatisfiable, so `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON` could not configure | **yes** | no | **yes** |
| 1.6.2 | release workflow rejected the annotated tag again, after the first fix | **yes** | no | no |

Two entries need their own note.

The moodycamel staleness is caught by neither, and never would be: no automated
check knows what upstream's latest release is. It is a checklist step ("check
the vendored dependency against upstream") and has to stay one.

The concurrentqueue floor was in fact caught, by the vcpkg port round refusing
to push when its own `vcpkg install` failed. The safety net worked. What it did
not do was work *early* -- it ran after the tag was published, so the cost was a
burnt version number and a release that is broken for anyone building 1.6.1
from source against a system concurrentqueue. Homebrew and MacPorts users are
unaffected; both build the bundled copy.

## What the record says

**RC uniquely catches the lightweight-tag rejection.** It is the only failure
that lives in the publish path itself, where nothing short of a real tag
exercises the code. Everything else is caught earlier and more cheaply by
something we already have or have since added.

**That row now has two members, not one.** The first fix was wrong: it blamed
`actions/checkout` fetching with `--no-tags` and added `fetch-tags: true`. The
real cause is that the check reads a local `refs/tags/` entry which checkout
builds pointing at the commit, so fetching tags changes nothing. 1.6.2 was
rejected identically and published by hand, and the check now asks the API
instead.

This is the strongest argument for RC in the table, and it cuts against the
recommendation below. The class RC uniquely covers is the one where a real tag
is the only way to exercise the code -- and it is precisely the class where our
reasoning failed twice, because we could not test the fix without publishing a
tag. Every other row was diagnosed correctly the first time.

## What RC would cost

Not nothing, and the cost recurs every release:

- `release.yml` triggers on `tags: ['v*']` and its first step rejects anything
  not matching `^[0-9]+\.[0-9]+\.[0-9]+$`. An RC tag would fire the workflow and
  immediately fail that check. The trigger needs narrowing or the regex
  widening, plus prerelease handling on the published release.
- `release.py check` compares against a plain `x.y.z` and would need to
  understand suffixes, as would the changelog heading rule.
- Every release grows a stage, and the ports would need to distinguish an RC
  tarball from a real one so an RC hash never reaches a port.

## What we are doing instead

1. **`release.py preflight`.** Builds what packagers build -- vendored shared
   and static, each installed and then consumed by a translation unit compiled
   with nothing but `-I<prefix>/include -L<prefix>/lib -ldispenso`, plus
   `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON`. It runs against the candidate tree,
   so it fails before a tag exists, and an unpublished failure costs nothing. It
   treats an unrunnable system-concurrentqueue check as a failure rather than a
   skip, because a preflight that reports success without it launders an
   unverified release as a verified one.
2. **The port round as a release gate, not a post-release step.** The round
   already builds and tests every port locally, and
   `update_package_managers.py --skip-push` stops before anything is pushed.
   Running it before the release is considered done is a checklist change rather
   than new machinery, and it is what caught 1.6.1.
3. **CI covers the packaged configuration.** A `build-system-concurrentqueue`
   job builds with `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON` on every pull
   request. That is the configuration the vcpkg and conan ports ship, and
   nothing exercised it before 1.6.2.

Together those move the concurrentqueue row from "caught after publishing" to
"caught on the pull request", which is where the 1.6.1 cost actually came from.

## When to revisit

Reopen this if any of the following happens. Each is a signal that the cheap
mechanisms have stopped covering the ground. The conclusion above was reached
when the publish-path class had one member; it now has two, so the case is
closer than the recommendation implies.

- A third failure that is **only** discoverable from a real tag, tarball, or
  publish. The first two triggers below have already fired, so this one decides
  it.
- Two consecutive releases need a patch follow-up. **Fired:** 1.6.1 and 1.6.2.
- The publish path changes materially: a `release.yml` rewrite, a different
  release action, or a change to how the tarball is produced. Both RC-shaped
  failures came from `actions/checkout` behaviour nobody had reason to suspect,
  and the first attempt to fix it was itself wrong.
- A port round starts failing on something preflight cannot model, such as
  vcpkg's own portfile logic or `conan create` against the recipe.

## Keeping this current

Add a row to the record for every future release failure, including the ones the
new mechanisms catch -- those are evidence too, and a mechanism that never fires
is worth knowing about. The decision above is only as good as the table, and the
table is the reason to prefer it over a fresh argument each time.
