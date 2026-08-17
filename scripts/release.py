#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Release helper for dispenso: verify version consistency, then tag.

A release version is spelled out in seven places. When one of them is missed the
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
import subprocess
import sys

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
        "METADATA.bzl", r'"version":\s*"([^"]+)"', version, errors, "version"
    )
    _check_pattern(
        "METADATA.bzl",
        r'"package_url":\s*"pkg:github/[^@"]+@([^"]+)"',
        version,
        errors,
        "package_url version",
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

    args = parser.parse_args()
    if args.command == "check":
        return check(args.version)
    return tag(args.version, args.commit, not args.no_sign)


if __name__ == "__main__":
    sys.exit(main())
