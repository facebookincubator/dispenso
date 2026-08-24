#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Release helper for dispenso: verify version consistency, then tag.

A release version is spelled out in six places. When one of them is missed the
mistake ships: 1.5.0 went out with the `platform.h` macros still reading 1.4.1,
and the 1.6.0 changelog carried a placeholder month until it was audited by hand.
`check` compares every one of them against a single expected version so that
class of mistake fails before a tag exists rather than after.

Ordering matters here. A published tag is effectively immutable: the package
manager ports pin the SHA-256 of the tarball GitHub generates for it, so moving
a tag invalidates hashes that are already recorded downstream. Everything that
can be validated is therefore validated first, and `tag` refuses to run when
`check` fails.

    python3 scripts/release.py check --version 1.6.0
    python3 scripts/release.py tag --version 1.6.0 [--commit SHA] [--no-sign]

`tag` creates a signed annotated tag locally and stops. It deliberately does not
push: review `git show v<version>` before making it permanent.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Months as spelled in CHANGELOG.md headings.
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _read(relpath):
    with open(os.path.join(REPO_ROOT, relpath), encoding="utf-8") as f:
        return f.read()


def _check_changelog(version, errors):
    """The top section must name this version and carry a full, specific date.

    A bare "June 2026" is rejected: it reads like a real heading, survives review,
    and is only ever noticed once the release is out under a different date.
    """
    first = _read("CHANGELOG.md").splitlines()[0].strip()
    m = re.match(r"^(\S+)\s+\((.+)\)$", first)
    if not m:
        errors.append(
            f"CHANGELOG.md: first line is {first!r}, expected '<version> (<Month D, YYYY>)'"
        )
        return
    found_version, date = m.group(1), m.group(2)
    if found_version != version:
        errors.append(
            f"CHANGELOG.md: top section is version {found_version!r}, expected {version!r}"
        )
    if not re.match(rf"^({MONTHS}) \d{{1,2}}, \d{{4}}$", date):
        errors.append(
            f"CHANGELOG.md: date {date!r} is not a specific day; "
            "use 'Month D, YYYY' so the entry cannot ship with a placeholder"
        )


def _check_pattern(relpath, pattern, expected, errors, label=None):
    """Assert every capture of `pattern` in `relpath` equals `expected`."""
    text = _read(relpath)
    found = re.findall(pattern, text, flags=re.MULTILINE)
    if not found:
        errors.append(f"{relpath}: no match for {label or pattern}")
        return
    for value in found:
        if value != expected:
            errors.append(
                f"{relpath}: {label or pattern} is {value!r}, expected {expected!r}"
            )


def check(version):
    major, minor, patch = version.split(".")
    errors = []

    _check_changelog(version, errors)
    _check_pattern(
        "dispenso/platform.h",
        r"#define DISPENSO_MAJOR_VERSION (\d+)",
        major,
        errors,
        "DISPENSO_MAJOR_VERSION",
    )
    _check_pattern(
        "dispenso/platform.h",
        r"#define DISPENSO_MINOR_VERSION (\d+)",
        minor,
        errors,
        "DISPENSO_MINOR_VERSION",
    )
    _check_pattern(
        "dispenso/platform.h",
        r"#define DISPENSO_PATCH_VERSION (\d+)",
        patch,
        errors,
        "DISPENSO_PATCH_VERSION",
    )
    _check_pattern(
        "CMakeLists.txt",
        r"^\s*VERSION (\d+\.\d+\.\d+)",
        version,
        errors,
        "project VERSION",
    )
    _check_pattern(
        "docs/Doxyfile",
        r"^PROJECT_NUMBER\s*=\s*(\S+)",
        version,
        errors,
        "PROJECT_NUMBER",
    )
    _check_pattern(
        "docs/faq.md", r"GIT_TAG\s+v(\d+\.\d+\.\d+)", version, errors, "GIT_TAG"
    )
    _check_pattern(
        "README.md",
        r"conan install --requires=dispenso/(\d+\.\d+\.\d+)",
        version,
        errors,
        "conan requires",
    )

    if errors:
        print(f"Release consistency check FAILED for {version}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"Release consistency check passed for {version}.")
    return 0


def _git(*args, **kwargs):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, **kwargs
    )


def tag(version, commit, sign):
    if check(version) != 0:
        print("\nRefusing to tag while the checks above fail.", file=sys.stderr)
        return 1

    tag_name = f"v{version}"
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{tag_name}").returncode == 0:
        print(
            f"{tag_name} already exists. A published tag must not be moved: the "
            "package manager ports pin the SHA-256 of its tarball. Release a new "
            "patch version instead.",
            file=sys.stderr,
        )
        return 1

    target = commit or "HEAD"
    resolved = _git("rev-parse", "--verify", f"{target}^{{commit}}")
    if resolved.returncode != 0:
        print(f"Cannot resolve {target!r}: {resolved.stderr.strip()}", file=sys.stderr)
        return 1
    sha = resolved.stdout.strip()

    args = ["tag", "-s" if sign else "-a", tag_name, sha, "-m", f"dispenso {version}"]
    result = _git(*args)
    if result.returncode != 0:
        print(f"Failed to create {tag_name}: {result.stderr.strip()}", file=sys.stderr)
        return 1

    print(f"Created {'signed ' if sign else ''}annotated tag {tag_name} at {sha[:12]}.")
    print("\nReview it, then publish:")
    print(f"    git show {tag_name}")
    print(f"    git push origin {tag_name}")
    return 0


# A consumer that reaches dispenso the way a packaged install does: one include
# directory, one library, no CMake target. `thread_pool.h` is included
# deliberately -- it is the header that pulls in moodycamel, so it is what
# breaks when the bundled headers are installed somewhere unreachable.
CONSUMER_SOURCE = """\
#include <cstdio>
#include <vector>

#include <dispenso/parallel_for.h>
#include <dispenso/thread_pool.h>

int main() {
  dispenso::ThreadPool pool(4);
  std::vector<int> v(1000);
  dispenso::parallel_for(
      0, v.size(), [&](size_t i) { v[i] = static_cast<int>(i * i); });
  std::printf("%d\\n", v[999]);
  return v[999] == 998001 ? 0 : 1;
}
"""

CONCURRENTQUEUE_PROBE = """\
cmake_minimum_required(VERSION 3.12)
project(cqprobe LANGUAGES CXX)
find_package(concurrentqueue CONFIG REQUIRED)
"""

# Each entry is (label, extra cmake args). Both link shapes are built because
# the ports disagree: vcpkg builds static, the source default is shared.
PREFLIGHT_BUILDS = [
    ("vendored shared", ["-DDISPENSO_SHARED_LIB=ON"]),
    ("vendored static", ["-DDISPENSO_SHARED_LIB=OFF"]),
]

CONSUMER_STANDARDS = ["14", "20"]


def _run(cmd, cwd=None, env=None):
    """Run one command, returning (ok, a transcript of it)."""
    result = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )
    return result.returncode == 0, f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}"


def _cmake_install(build_dir, prefix, cmake_args, jobs, log):
    """Configure, build and install dispenso into prefix."""
    steps = [
        [
            "cmake",
            "-S",
            REPO_ROOT,
            "-B",
            build_dir,
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DDISPENSO_BUILD_TESTS=OFF",
            "-DDISPENSO_BUILD_BENCHMARKS=OFF",
            *cmake_args,
        ],
        ["cmake", "--build", build_dir, "--parallel", str(jobs)],
        ["cmake", "--install", build_dir],
    ]
    for step in steps:
        ok, out = _run(step)
        log.append(out)
        if not ok:
            return False
    return True


def _consume(prefix, std, workdir, log):
    """Compile and run the consumer against an installed prefix.

    Only -I<prefix>/include and -L<prefix>/lib are passed: no CMake target, no
    second include directory. An install that needs more than this is one the
    package managers cannot ship.
    """
    source = os.path.join(workdir, f"consumer{std}.cpp")
    with open(source, "w", encoding="utf-8") as f:
        f.write(CONSUMER_SOURCE)
    exe = os.path.join(workdir, f"consumer{std}")

    ok, out = _run(
        [
            os.environ.get("CXX", "c++"),
            f"-std=c++{std}",
            source,
            f"-I{os.path.join(prefix, 'include')}",
            f"-L{os.path.join(prefix, 'lib')}",
            "-ldispenso",
            "-o",
            exe,
        ]
    )
    log.append(out)
    if not ok:
        return False

    env = dict(os.environ)
    libdir = os.path.join(prefix, "lib")
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = os.pathsep.join([libdir, env.get(var, "")]).rstrip(os.pathsep)
    ok, out = _run([exe], env=env)
    log.append(out)
    return ok


def _have_concurrentqueue(workdir, prefix_path, log):
    """Whether a system concurrentqueue CMake package is visible to find_package."""
    probe = os.path.join(workdir, "cqprobe")
    os.makedirs(probe, exist_ok=True)
    with open(os.path.join(probe, "CMakeLists.txt"), "w", encoding="utf-8") as f:
        f.write(CONCURRENTQUEUE_PROBE)
    cmd = ["cmake", "-S", probe, "-B", os.path.join(probe, "build")]
    if prefix_path:
        cmd.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")
    ok, out = _run(cmd)
    log.append(out)
    return ok


def _preflight_installed_builds(workdir, jobs, results, logs):
    """Build, install and consume dispenso the way a packaged install is used."""
    for label, cmake_args in PREFLIGHT_BUILDS:
        log = []
        slug = label.replace(" ", "-")
        prefix = os.path.join(workdir, f"prefix-{slug}")
        built = _cmake_install(
            os.path.join(workdir, f"build-{slug}"), prefix, cmake_args, jobs, log
        )
        consumed = built and all(
            _consume(prefix, std, workdir, log) for std in CONSUMER_STANDARDS
        )
        results.append((label, "PASS" if consumed else "FAIL"))
        logs[label] = log


def _preflight_system_concurrentqueue(workdir, jobs, prefix_path, results, logs):
    """Build the configuration the vcpkg and conan ports ship.

    Not being able to run this is a failure, not a pass with a footnote. It is
    the only check that catches the 1.6.1 class of bug, and a preflight that
    reports success without it is worse than no preflight: it launders an
    unverified release as a verified one. Install concurrentqueue, or pass
    --concurrentqueue-prefix.
    """
    label = "system concurrentqueue"
    log = []
    args = ["-DDISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON"]
    if prefix_path:
        args.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")

    if not _have_concurrentqueue(workdir, prefix_path, log):
        results.append(
            (
                label,
                "FAIL: no concurrentqueue CMake package is visible, so the "
                "configuration the vcpkg and conan ports ship went untested. "
                "Install it, or pass --concurrentqueue-prefix.",
            )
        )
        logs[label] = log
        return

    ok = _cmake_install(
        os.path.join(workdir, "build-syscq"),
        os.path.join(workdir, "prefix-syscq"),
        args,
        jobs,
        log,
    )
    results.append((label, "PASS" if ok else "FAIL"))
    logs[label] = log


def _report_preflight(results, logs):
    """Print the summary, and the transcript of anything that failed."""
    failed = [label for label, status in results if status.startswith("FAIL")]
    for label, status in results:
        head, _, detail = status.partition(": ")
        print(f"  {head:5s}  {label}")
        if detail:
            print(f"         {detail}")
    for label in failed:
        print(f"\n--- {label} ---", file=sys.stderr)
        print("\n".join(logs.get(label, [])), file=sys.stderr)
    return failed


def preflight(version, jobs, keep, concurrentqueue_prefix):
    """Build what packagers build, before a tag makes the mistake permanent.

    `check` compares version strings; this compares against reality. Every
    release failure so far has been of a kind `check` cannot see: 1.6.1 passed
    every version check, tagged cleanly, and could not configure against a
    system concurrentqueue at all. That was found days later by a port round.
    An unpublished failure costs nothing; a published one costs a release.
    """
    results = [("version consistency", "PASS" if check(version) == 0 else "FAIL")]
    logs = {}

    workdir = tempfile.mkdtemp(prefix="dispenso-preflight-")
    print(f"\nBuilding in {workdir}\n")
    try:
        _preflight_installed_builds(workdir, jobs, results, logs)
        _preflight_system_concurrentqueue(
            workdir, jobs, concurrentqueue_prefix, results, logs
        )
    finally:
        if keep:
            print(f"\nKept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nPreflight for {version}:")
    failed = _report_preflight(results, logs)
    if failed:
        print(f"\nPreflight FAILED for {version}: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nPreflight passed for {version}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="verify every version reference agrees")
    p_check.add_argument(
        "--version", required=True, help="expected version, e.g. 1.6.0"
    )

    p_tag = sub.add_parser("tag", help="check, then create a signed annotated tag")
    p_tag.add_argument("--version", required=True, help="version to tag, e.g. 1.6.0")
    p_tag.add_argument(
        "--commit",
        help="commit to tag; defaults to HEAD. Use the commit CI validated, not "
        "whatever the branch has drifted to.",
    )
    p_tag.add_argument(
        "--no-sign",
        action="store_true",
        help="create an unsigned annotated tag (signing is the default)",
    )

    p_pre = sub.add_parser(
        "preflight", help="build and consume dispenso the way packagers do"
    )
    p_pre.add_argument("--version", required=True, help="version to validate")
    p_pre.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 4, help="parallel build jobs"
    )
    p_pre.add_argument(
        "--keep", action="store_true", help="keep the build tree for inspection"
    )
    p_pre.add_argument(
        "--concurrentqueue-prefix",
        help="where a system concurrentqueue is installed, if find_package "
        "cannot locate it unaided",
    )

    args = parser.parse_args()
    if args.command == "check":
        return check(args.version)
    if args.command == "preflight":
        return preflight(
            args.version, args.jobs, args.keep, args.concurrentqueue_prefix
        )
    return tag(args.version, args.commit, not args.no_sign)


if __name__ == "__main__":
    sys.exit(main())
