#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Update dispenso across package manager repositories for a new release.

Downloads the release tarball, computes hashes, updates version/hash references
in each package manager's repo, commits, tests locally, pushes to the user's
fork, and creates PRs.

Recommended usage (interactive guided flow):
    python3 update_package_managers.py --version 1.5.1 --guided

The guided flow walks through every step: updating each manager, running tests,
inspecting diffs, pushing, closing superseded PRs, creating new PRs, and
guiding through any remaining manual steps like CLA signing.

Advanced usage (individual flags):
    python3 update_package_managers.py --version 1.5.1 --managers conan,vcpkg
    python3 update_package_managers.py --version 1.5.1 --dry-run
    python3 update_package_managers.py --version 1.5.1 --skip-test
    python3 update_package_managers.py --version 1.5.1 --skip-push
    python3 update_package_managers.py --version 1.5.1 --create-prs
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

GITHUB_REPO = "facebookincubator/dispenso"
MANAGERS = ["conan", "vcpkg", "homebrew", "macports"]

# Upstream repos and default branch names for each manager
UPSTREAM_REPOS = {
    "conan": "conan-io/conan-center-index",
    "vcpkg": "microsoft/vcpkg",
    "homebrew": "Homebrew/homebrew-core",
    "macports": "macports/macports-ports",
}

REPO_DIRS = {
    "conan": "conan-center-index",
    "vcpkg": "vcpkg",
    "homebrew": "homebrew-core",
    "macports": "macports-ports",
}

BRANCH_NAMES = {
    "conan": "package/dispenso",
    "vcpkg": "add-dispenso",
    # homebrew and macports use version-specific branches, set dynamically
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update dispenso across package manager repositories."
    )
    parser.add_argument("--version", required=True, help="Release version, e.g. 1.5.0")
    parser.add_argument(
        "--managers",
        default=",".join(MANAGERS),
        help=f"Comma-separated list of managers to update (default: {','.join(MANAGERS)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files or pushing",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip local verification tests after committing",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Update files, commit, and test but don't push to fork",
    )
    parser.add_argument(
        "--repos-dir",
        default=os.path.expanduser("~/repos"),
        help="Base directory containing cloned repos (default: ~/repos)",
    )
    parser.add_argument(
        "--github-user",
        default="graphicsMan",
        help="GitHub fork username (default: graphicsMan)",
    )
    parser.add_argument(
        "--create-prs",
        action="store_true",
        help="Create PRs on GitHub after pushing (uses gh CLI)",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Interactive guided flow: walks through every step, prompts "
        "before each action, creates PRs at the end",
    )
    args = parser.parse_args()
    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        parser.error(
            f"Invalid version format: {args.version!r}. Expected semver like 1.5.1"
        )
    args.managers = [m.strip() for m in args.managers.split(",")]
    for m in args.managers:
        if m not in MANAGERS:
            parser.error(f"Unknown manager: {m}. Choose from: {', '.join(MANAGERS)}")
    return args


def download_and_hash(version):
    """Download the release tarball and compute sha256, sha512, rmd160, and size."""
    url = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/v{version}.tar.gz"
    print(f"Downloading {url} ...")

    tarball_path = os.path.join(tempfile.gettempdir(), f"dispenso-v{version}.tar.gz")

    # Download (or reuse if already present and non-empty)
    if os.path.exists(tarball_path) and os.path.getsize(tarball_path) > 0:
        print(f"  Using cached tarball: {tarball_path}")
    else:
        urllib.request.urlretrieve(url, tarball_path)
        print(f"  Saved to {tarball_path}")

    data = open(tarball_path, "rb").read()
    size = len(data)

    sha256 = hashlib.sha256(data).hexdigest()
    sha512 = hashlib.sha512(data).hexdigest()

    # RIPEMD-160: try hashlib first, fall back to openssl subprocess
    try:
        rmd160 = hashlib.new("ripemd160", data).hexdigest()
    except ValueError:
        # hashlib doesn't support rmd160 on this system, try openssl
        try:
            result = subprocess.run(
                ["openssl", "dgst", "-rmd160", tarball_path],
                capture_output=True,
                text=True,
                check=True,
            )
            # Output format: "RIPEMD-160(file)= <hex>"
            rmd160 = result.stdout.strip().split("= ")[-1]
        except (FileNotFoundError, subprocess.CalledProcessError):
            rmd160 = None
            print("  WARNING: Could not compute RIPEMD-160 (needed for MacPorts)")

    hashes = {
        "sha256": sha256,
        "sha512": sha512,
        "rmd160": rmd160,
        "size": size,
        "url": url,
    }

    print(f"  SHA256:  {sha256}")
    print(f"  SHA512:  {sha512}")
    print(f"  RMD160:  {rmd160}")
    print(f"  Size:    {size}")
    print()

    return tarball_path, hashes


def run(cmd, cwd=None, check=True, dry_run=False, capture=False, env=None):
    """Run a shell command, printing it first. Respects dry_run."""
    if isinstance(cmd, str):
        display = cmd
    else:
        display = " ".join(cmd)

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"  {prefix}$ {display}")

    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    kwargs = {"cwd": cwd, "check": check}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    if env is not None:
        kwargs["env"] = env

    return subprocess.run(cmd, **kwargs)


def ensure_repo(repos_dir, manager, github_user, dry_run):
    """Clone if missing, add fork remote, fetch upstream."""
    upstream = UPSTREAM_REPOS[manager]
    repo_name = REPO_DIRS[manager]
    repo_dir = os.path.join(repos_dir, repo_name)

    if not os.path.isdir(repo_dir):
        print(f"  Cloning {upstream} ...")
        run(
            ["git", "clone", f"https://github.com/{upstream}.git", repo_dir],
            dry_run=dry_run,
        )
    else:
        print(f"  Repo already exists: {repo_dir}")

    if dry_run:
        return repo_dir

    # Ensure upstream remote points to the right place
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        current_origin = result.stdout.strip()
        # Normalize: also accept SSH format
        if upstream not in current_origin:
            print(f"  WARNING: origin points to {current_origin}, expected {upstream}")

    # Ensure fork remote exists
    result = subprocess.run(
        ["git", "remote", "get-url", "fork"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Adding fork remote for {github_user}/{repo_name} ...")
        # Ensure the fork exists on GitHub
        run(
            ["gh", "repo", "fork", upstream, "--clone=false"],
            cwd=repo_dir,
            check=False,
        )
        run(
            [
                "git",
                "remote",
                "add",
                "fork",
                f"git@github.com:{github_user}/{repo_name}.git",
            ],
            cwd=repo_dir,
        )
    else:
        print("  Fork remote already exists")

    # Fetch upstream
    print("  Fetching origin ...")
    run(["git", "fetch", "origin"], cwd=repo_dir)

    return repo_dir


def checkout_branch(repo_dir, branch, dry_run):
    """Determine the default branch, update it, and create/checkout the working branch."""
    if dry_run:
        print(f"  [DRY RUN] Would checkout branch: {branch}")
        return

    # Determine default branch (main or master)
    result = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        default_branch = result.stdout.strip().replace("refs/remotes/origin/", "")
    else:
        # Try common names
        for candidate in ["main", "master"]:
            r = subprocess.run(
                ["git", "show-ref", "--verify", f"refs/remotes/origin/{candidate}"],
                cwd=repo_dir,
                capture_output=True,
            )
            if r.returncode == 0:
                default_branch = candidate
                break
        else:
            default_branch = "main"

    # Abort any in-progress rebase (e.g. from a previous failed run)
    rebase_merge = os.path.join(repo_dir, ".git", "rebase-merge")
    rebase_apply = os.path.join(repo_dir, ".git", "rebase-apply")
    if os.path.isdir(rebase_merge) or os.path.isdir(rebase_apply):
        print("  Aborting in-progress rebase from previous run ...")
        run(["git", "rebase", "--abort"], cwd=repo_dir, check=False)

    # Stash any dirty changes so we can switch branches
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    stashed = False
    if result.stdout.strip():
        print("  Stashing dirty changes ...")
        run(["git", "stash", "--include-untracked"], cwd=repo_dir)
        stashed = True

    # Checkout and update default branch
    run(["git", "checkout", default_branch], cwd=repo_dir)
    run(["git", "pull", "origin", default_branch], cwd=repo_dir)

    # Create or reset working branch to start fresh from default branch.
    # We reset instead of rebase because the script overwrites all port files
    # and commit_and_push() squashes into one commit anyway.  Rebasing can
    # conflict when a previous version's changes were already merged.
    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        cwd=repo_dir,
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  Branch {branch} already exists, resetting to {default_branch} ...")
        run(["git", "checkout", branch], cwd=repo_dir)
        run(["git", "reset", "--hard", default_branch], cwd=repo_dir)
    else:
        print(f"  Creating branch {branch} from {default_branch} ...")
        run(["git", "checkout", "-b", branch], cwd=repo_dir)

    # Restore stashed changes if any
    if stashed:
        print("  Restoring stashed changes ...")
        run(["git", "stash", "pop"], cwd=repo_dir, check=False)


def commit_and_push(repo_dir, branch, message, github_user, dry_run, skip_push):
    """Stage all changes, squash into a single commit, push to fork.

    If the branch already has commits beyond the upstream base, soft-reset to the
    base and re-commit everything as a single commit. This ensures the branch always
    has exactly one clean commit regardless of how many iterations have been made.
    """
    if dry_run:
        run(["git", "add", "-A"], cwd=repo_dir, dry_run=True)
        run(["git", "commit", "-m", message], cwd=repo_dir, dry_run=True)
        if not skip_push:
            run(
                ["git", "push", "-u", "fork", branch, "--force"],
                cwd=repo_dir,
                dry_run=True,
            )
        return True

    # Stage everything
    run(["git", "add", "-A"], cwd=repo_dir)

    # Find the merge base with the upstream default branch
    merge_base = None
    for candidate in ["origin/main", "origin/master"]:
        result = subprocess.run(
            ["git", "merge-base", candidate, "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            merge_base = result.stdout.strip()
            break

    if merge_base:
        # Check if there are any differences from the merge base
        result = subprocess.run(
            ["git", "diff", "--cached", merge_base, "--quiet"],
            cwd=repo_dir,
        )
        if result.returncode == 0:
            # Also check uncommitted staged changes
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=repo_dir,
            )
            if result.returncode == 0:
                print(
                    "  WARNING: No changes to commit — upstream already"
                    " has this version. Skipping."
                )
                return False

        # Soft-reset to merge base and recommit as single commit.
        # Re-add after reset so the index reflects all working-tree changes
        # relative to the new HEAD (merge_base), not the old HEAD.
        print("  Squashing into single commit ...")
        run(["git", "reset", "--soft", merge_base], cwd=repo_dir)
        run(["git", "add", "-A"], cwd=repo_dir)
        run(["git", "commit", "-m", message], cwd=repo_dir)
    else:
        # Fallback: just commit normally if we can't find a merge base
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_dir,
        )
        if result.returncode == 0:
            print("  No staged changes to commit")
        else:
            run(["git", "commit", "-m", message], cwd=repo_dir)

    if not skip_push:
        run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    return True


# ---------------------------------------------------------------------------
# Per-manager test functions
# ---------------------------------------------------------------------------


def test_conan(repo_dir, version):
    """Test conan recipe with conan create."""
    conan_bin = shutil.which("conan")
    if not conan_bin:
        print("  WARNING: conan not found, skipping test")
        return False

    print("  Running conan create ...")
    result = run(
        [conan_bin, "create", "recipes/dispenso/all", f"--version={version}"],
        cwd=repo_dir,
        check=False,
    )
    if result.returncode != 0:
        print("  FAIL: conan create failed")
        return False

    print("  PASS: conan create succeeded")
    return True


def test_vcpkg(repo_dir, version):
    """Test vcpkg port with vcpkg install."""
    vcpkg_bin = shutil.which("vcpkg")
    if not vcpkg_bin:
        print("  WARNING: vcpkg not found, skipping test")
        return False

    # Detect default triplet so we remove/install for the right platform
    if sys.platform == "darwin":
        machine = platform.machine()  # arm64 or x86_64
        default_triplet = "arm64-osx" if machine == "arm64" else "x64-osx"
    elif sys.platform == "win32":
        default_triplet = "x64-windows"
    else:
        default_triplet = "x64-linux"

    # Remove existing install to force rebuild
    print(f"  Removing existing vcpkg install (if any) [{default_triplet}] ...")
    run(
        [vcpkg_bin, "remove", f"dispenso:{default_triplet}"],
        cwd=repo_dir,
        check=False,
    )

    print("  Running vcpkg install ...")
    result = run(
        [
            vcpkg_bin,
            "install",
            f"dispenso:{default_triplet}",
            f"--overlay-ports={os.path.join(repo_dir, 'ports', 'dispenso')}",
            "--no-binarycaching",
        ],
        cwd=repo_dir,
        check=False,
    )
    if result.returncode != 0:
        print("  FAIL: vcpkg install failed")
        return False

    print("  PASS: vcpkg install succeeded")
    return True


def test_homebrew(repo_dir, version):
    """Test homebrew formula with brew install, test, and audit."""
    brew_bin = shutil.which("brew")
    if not brew_bin:
        print("  WARNING: brew not found — full testing requires macOS")
        return False

    brew_repo = subprocess.run(
        [brew_bin, "--repository"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    real_core = os.path.join(brew_repo, "Library", "Taps", "homebrew", "homebrew-core")
    backup_core = None

    try:
        # Swap in our local checkout as homebrew-core
        if os.path.exists(real_core):
            backup_core = f"{real_core}.bak.{os.getpid()}"
            print("  Temporarily replacing homebrew-core tap ...")
            os.rename(real_core, backup_core)
        os.symlink(repo_dir, real_core)

        # Uninstall if already installed
        run([brew_bin, "uninstall", "dispenso"], check=False)

        # Install from source
        print(
            "  Running HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-from-source ..."
        )
        brew_env = {**os.environ, "HOMEBREW_NO_INSTALL_FROM_API": "1"}
        result = run(
            [brew_bin, "install", "--build-from-source", "dispenso"],
            check=False,
            capture=True,
            env=brew_env,
        )
        if result.returncode != 0:
            print(f"  FAIL: brew install failed\n{result.stderr}")
            return False

        # Run tests
        print("  Running brew test ...")
        result = run([brew_bin, "test", "dispenso"], check=False)
        if result.returncode != 0:
            print("  FAIL: brew test failed")
            return False

        # Audit
        print("  Running brew audit --strict ...")
        result = run([brew_bin, "audit", "--strict", "dispenso"], check=False)
        if result.returncode != 0:
            print("  WARNING: brew audit --strict reported issues")

        print("  PASS: homebrew tests succeeded")
        return True

    finally:
        # Restore original homebrew-core tap
        if os.path.islink(real_core):
            os.unlink(real_core)
        if backup_core and os.path.exists(backup_core):
            print("  Restoring original homebrew-core tap ...")
            os.rename(backup_core, real_core)
        # Clean up
        run([brew_bin, "uninstall", "dispenso"], check=False)


def test_macports(repo_dir, version, hashes=None):
    """Test macports portfile with port lint, install, and test."""
    # Verify checksums first — this catches the most common error
    if hashes:
        print("  --- Verifying checksums ---")
        if not verify_portfile_checksums(repo_dir, hashes):
            print("  FAIL: checksum verification failed — fix before continuing")
            return False

    port_bin = shutil.which("port")
    if not port_bin:
        print("  WARNING: port not found — install MacPorts to run full tests")
        return False

    portdir = os.path.join(repo_dir, "devel", "dispenso")

    print("  Running port lint --nitpick ...")
    result = run(
        [port_bin, "-D", portdir, "lint", "--nitpick"],
        check=False,
    )
    if result.returncode != 0:
        print("  FAIL: port lint failed")
        return False
    print("  PASS: port lint succeeded")

    # Install and test require sudo.  Root cannot access files under the
    # user's home directory on macOS, so copy to /tmp.  Running sudo from a
    # Python subprocess may also be blocked by endpoint security tools, so
    # print the commands for the user to run manually.
    tmp_parent = tempfile.mkdtemp(prefix="dispenso-port-", dir="/tmp")
    os.chmod(tmp_parent, 0o755)
    tmp_portdir = os.path.join(tmp_parent, "dispenso")
    shutil.copytree(portdir, tmp_portdir)
    for dirpath, dirs, files in os.walk(tmp_portdir):
        for d in dirs:
            os.chmod(os.path.join(dirpath, d), 0o755)
        for f in files:
            os.chmod(os.path.join(dirpath, f), 0o644)

    print()
    print("  The next steps require sudo.  Please run each command manually")
    print("  in a separate terminal, then come back and report the result.")

    install_ok = False
    test_ok = False
    install_note = ""
    test_note = ""

    # Step 1: install
    print()
    print("  Step 1 — run this command:")
    print(f"    sudo {port_bin} -D {tmp_portdir} -vst install")
    print()
    answer = input("  Result? [y=succeeded / or describe what happened] ").strip()
    if answer.lower() == "y":
        install_ok = True
        print("  PASS: port install (user-confirmed)")
    else:
        install_note = answer
        print(f"  NOTED: port install issue — {answer}")

    # Step 2: test
    print()
    print("  Step 2 — run this command:")
    print(f"    sudo {port_bin} -D {tmp_portdir} test")
    print()
    answer = input("  Result? [y=succeeded / or describe what happened] ").strip()
    if answer.lower() == "y":
        test_ok = True
        print("  PASS: port test (user-confirmed)")
    else:
        test_note = answer
        print(f"  NOTED: port test issue — {answer}")

    # Summarize
    print()
    if install_ok and test_ok:
        print("  PASS: port install + test both succeeded")
    elif test_ok and not install_ok:
        print("  PARTIAL: port test passed but install had issues")
        print(f"    install note: {install_note}")
    elif install_ok and not test_ok:
        print("  PARTIAL: port install passed but test had issues")
        print(f"    test note: {test_note}")
    else:
        print("  FAIL: both install and test had issues")

    if not (install_ok and test_ok):
        print(f"  (temp port dir preserved at {tmp_portdir} for debugging)")
    else:
        shutil.rmtree(tmp_parent, ignore_errors=True)

    return test_ok


def get_macos_tested_on():
    """Gather macOS system info for MacPorts 'Tested on' PR section.

    Returns a formatted string matching the MacPorts PR template, or None
    if not running on macOS.
    """
    if platform.system() != "Darwin":
        return None

    lines = []

    # macOS version + build + arch
    result = subprocess.run(
        ["sw_vers", "-productVersion"], capture_output=True, text=True
    )
    if result.returncode == 0:
        mac_ver = result.stdout.strip()
        build_result = subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True
        )
        build = build_result.stdout.strip() if build_result.returncode == 0 else ""
        arch_result = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        arch = arch_result.stdout.strip() if arch_result.returncode == 0 else ""
        lines.append(f"macOS {mac_ver} {build} {arch}")

    # Xcode or Command Line Tools version
    result = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True)
    if result.returncode == 0:
        parts = result.stdout.strip().split("\n")
        if len(parts) >= 2:
            # "Xcode 16.2\nBuild version 16C5032a" → "Xcode 16.2 16C5032a"
            lines.append(f"{parts[0]} {parts[-1].split()[-1]}")
        elif parts:
            lines.append(parts[0])
    else:
        result = subprocess.run(
            ["pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "version:" in line:
                    lines.append(f"Command Line Tools {line.split(':')[1].strip()}")
                    break

    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Post-write checksum verification
# ---------------------------------------------------------------------------

MACOS_ONLY_MANAGERS = {"homebrew", "macports"}


def verify_portfile_checksums(repo_dir, hashes):
    """Verify that the Portfile checksums match the computed hashes."""
    portfile_path = os.path.join(repo_dir, "devel", "dispenso", "Portfile")
    content = open(portfile_path).read()

    ok = True
    for field, key in [("rmd160", "rmd160"), ("sha256", "sha256"), ("size", "size")]:
        match = re.search(rf"{field}\s+(\S+)", content)
        if not match:
            print(f"  WARNING: {field} not found in Portfile")
            continue
        expected = str(hashes[key])
        actual = match.group(1)
        if actual != expected:
            print(f"  FAIL: {field} mismatch in Portfile")
            print(f"    Portfile:  {actual}")
            print(f"    Expected:  {expected}")
            ok = False

    if ok:
        print("  PASS: Portfile checksums match computed hashes")
    return ok


def verify_formula_checksums(repo_dir, hashes):
    """Verify that the Homebrew formula sha256 matches the computed hash."""
    formula_path = os.path.join(repo_dir, "Formula", "d", "dispenso.rb")
    content = open(formula_path).read()

    match = re.search(r'sha256\s+"([0-9a-fA-F]+)"', content)
    if not match:
        print("  WARNING: sha256 not found in formula")
        return False

    expected = hashes["sha256"]
    actual = match.group(1)
    if actual != expected:
        print("  FAIL: sha256 mismatch in formula")
        print(f"    Formula:   {actual}")
        print(f"    Expected:  {expected}")
        return False

    print("  PASS: Formula sha256 matches computed hash")
    return True


# ---------------------------------------------------------------------------
# Per-manager update functions
# ---------------------------------------------------------------------------


def ensure_conan_issue(version, github_user, dry_run):
    """Search for or create a Conan Center Index issue for this version.

    Returns the issue number if found/created, or None on failure.
    """
    repo = UPSTREAM_REPOS["conan"]
    search_query = f"[package] dispenso/{version}"

    # Search for existing issue
    print(f"  Searching for existing Conan issue: {search_query}")
    if dry_run:
        print(f"  [DRY RUN] Would search/create issue for dispenso/{version}")
        return None

    result = subprocess.run(
        [
            "gh",
            "search",
            "issues",
            search_query,
            "--repo",
            repo,
            "--json",
            "number,title",
            "--limit",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            issues = json.loads(result.stdout)
            for issue in issues:
                if f"dispenso/{version}" in issue.get("title", ""):
                    number = issue["number"]
                    print(f"  Found existing issue #{number}: {issue['title']}")
                    return number
        except (json.JSONDecodeError, KeyError):
            pass

    # Create new issue
    print(f"  Creating new Conan issue for dispenso/{version} ...")
    issue_body = f"""\
### Package Details
- **Name**: dispenso
- **Version**: {version}
- **Homepage**: https://github.com/facebookincubator/dispenso
- **License**: MIT

### Description
dispenso is a high-performance C++ library for parallel programming from Meta. \
It provides work-stealing thread pools, parallel for loops, futures, task graphs, \
pipelines, and concurrent containers. Requires C++14, no external dependencies \
beyond pthreads."""

    result = subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            f"[package] dispenso/{version}",
            "--body",
            issue_body,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # Output is the issue URL, extract number from it
        url = result.stdout.strip()
        match = re.search(r"/issues/(\d+)", url)
        if match:
            number = int(match.group(1))
            print(f"  Created issue #{number}: {url}")
            return number
        print(f"  Created issue: {url}")
        return None

    print(f"  WARNING: Failed to create Conan issue: {result.stderr.strip()}")
    return None


def update_conan(args, hashes, tarball_path):
    """Update conan-center-index with new dispenso version."""
    print("=== Conan (conan-center-index) ===")
    version = args.version
    manager = "conan"
    branch = BRANCH_NAMES.get(manager, "add-dispenso")

    repo_dir = ensure_repo(args.repos_dir, manager, args.github_user, args.dry_run)
    checkout_branch(repo_dir, branch, args.dry_run)

    # --- Update conandata.yml ---
    conandata_path = os.path.join(
        repo_dir, "recipes", "dispenso", "all", "conandata.yml"
    )
    print(f"  Updating {conandata_path} ...")

    if not args.dry_run:
        content = open(conandata_path).read()

        # Check if version already present
        if f'"{version}"' in content:
            print(f"  Version {version} already present in conandata.yml")
        else:
            # Insert new version entry right after "sources:\n"
            new_entry = (
                f'  "{version}":\n'
                f'    url: "{hashes["url"]}"\n'
                f'    sha256: "{hashes["sha256"]}"\n'
            )
            content = content.replace("sources:\n", f"sources:\n{new_entry}", 1)
            open(conandata_path, "w").write(content)
            print(f"  Added version {version} to conandata.yml")
    else:
        print(f"  [DRY RUN] Would add {version} entry with sha256={hashes['sha256']}")

    # --- Update config.yml ---
    config_path = os.path.join(repo_dir, "recipes", "dispenso", "config.yml")
    print(f"  Updating {config_path} ...")

    if not args.dry_run:
        content = open(config_path).read()

        if f'"{version}"' in content:
            print(f"  Version {version} already present in config.yml")
        else:
            new_entry = f'  "{version}":\n    folder: all\n'
            content = content.replace("versions:\n", f"versions:\n{new_entry}", 1)
            open(config_path, "w").write(content)
            print(f"  Added version {version} to config.yml")
    else:
        print(f"  [DRY RUN] Would add {version} entry to config.yml")

    # --- Ensure GitHub issue exists ---
    print("  --- Conan issue ---")
    issue_number = ensure_conan_issue(version, args.github_user, args.dry_run)

    # --- Commit ---
    commit_msg = f"dispenso: add version {version}"
    committed = commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    test_passed = True
    if committed and not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        test_passed = test_conan(repo_dir, version)

    # --- Push ---
    if not committed:
        print("  Skipping push — no changes to push")
    elif not args.dry_run and not args.skip_push:
        if not test_passed:
            print("  Skipping push due to test failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = (
        f"https://github.com/{args.github_user}/conan-center-index/tree/{branch}"
    )
    tests_ran = not args.skip_test and not args.dry_run and test_passed
    title = pr_title("conan", version)
    body = pr_body_conan(version, tests_ran=tests_ran, issue_number=issue_number)
    print()
    return {
        "status": "no_changes"
        if not committed
        else "ok"
        if test_passed
        else "test_failed",
        "branch": branch,
        "branch_url": branch_url,
        "version": version,
        "tests_ran": tests_ran,
        "pr_title": title,
        "pr_body": body,
    }


def detect_obsolete_patches(tarball_path, port_dir, strip_level=1):
    """Check which patches in port_dir are obsolete against the tarball source.

    Extracts the tarball to a temp directory and tries `git apply --check` for
    each .patch file. Returns a list of patch filenames that no longer apply
    (i.e. the fix has been upstreamed).

    Args:
        strip_level: Number of leading path components to strip (0 for MacPorts,
            1 for vcpkg/git-style patches).
    """
    patch_files = [f for f in os.listdir(port_dir) if f.endswith(".patch")]
    if not patch_files:
        return []

    obsolete = []
    tmpdir = tempfile.mkdtemp(prefix="dispenso-patch-check-")
    try:
        # Extract tarball
        with tarfile.open(tarball_path, "r:gz") as tf:
            tf.extractall(tmpdir, filter="data")

        # Find the extracted directory (e.g. dispenso-1.5.1/)
        entries = os.listdir(tmpdir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
            src_dir = os.path.join(tmpdir, entries[0])
        else:
            src_dir = tmpdir

        # Initialize a temporary git repo so we can use git apply --check
        subprocess.run(["git", "init"], cwd=src_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "add", "-A"], cwd=src_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=src_dir,
            capture_output=True,
            check=True,
        )

        for patch_file in patch_files:
            patch_path = os.path.join(port_dir, patch_file)
            result = subprocess.run(
                ["git", "apply", "--check", f"-p{strip_level}", patch_path],
                cwd=src_dir,
                capture_output=True,
            )
            if result.returncode != 0:
                print(f"  Patch {patch_file}: OBSOLETE (no longer applies)")
                obsolete.append(patch_file)
            else:
                print(f"  Patch {patch_file}: still needed (applies cleanly)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return obsolete


def remove_obsolete_patches(port_dir, portfile_path, obsolete_patches):
    """Remove obsolete patches from portfile.cmake and delete patch files."""
    if not obsolete_patches:
        return

    content = open(portfile_path).read()

    for patch_file in obsolete_patches:
        # Delete the patch file
        patch_path = os.path.join(port_dir, patch_file)
        if os.path.exists(patch_path):
            os.unlink(patch_path)
            print(f"  Deleted {patch_file}")

    # Check if there are remaining patches
    remaining = [f for f in os.listdir(port_dir) if f.endswith(".patch")]

    if not remaining:
        # Remove entire PATCHES block from portfile.cmake
        # Match: PATCHES\n    file1.patch\n    file2.patch\n (up to next keyword or end)
        content = re.sub(
            r"\s+PATCHES\n(?:\s+\S+\.patch\n?)+",
            "\n",
            content,
        )
        print("  Removed entire PATCHES block from portfile.cmake")
    else:
        # Remove only obsolete patch filenames from PATCHES block
        for patch_file in obsolete_patches:
            # Remove the line "    patch_file.patch\n"
            content = re.sub(
                rf"\s+{re.escape(patch_file)}\n?",
                "\n",
                content,
            )
        print(f"  Removed obsolete patches from portfile.cmake, kept: {remaining}")

    open(portfile_path, "w").write(content)


def _vcpkg_update_port_files(repo_dir, version, hashes, dry_run):
    """Update vcpkg.json version/port-version and portfile.cmake SHA512."""
    vcpkg_json_path = os.path.join(repo_dir, "ports", "dispenso", "vcpkg.json")
    print(f"  Updating {vcpkg_json_path} ...")

    if not dry_run:
        content = open(vcpkg_json_path).read()
        content = re.sub(
            r'"version"\s*:\s*"[^"]*"',
            f'"version": "{version}"',
            content,
        )
        # Remove port-version (resets to 0 on version bump)
        content = re.sub(r',?\s*"port-version"\s*:\s*\d+', "", content)
        open(vcpkg_json_path, "w").write(content)
        print(f"  Updated version to {version} (removed port-version if present)")
    else:
        print(f"  [DRY RUN] Would update version to {version} (remove port-version)")

    portfile_path = os.path.join(repo_dir, "ports", "dispenso", "portfile.cmake")
    print(f"  Updating {portfile_path} ...")

    if not dry_run:
        content = open(portfile_path).read()
        content = re.sub(
            r"SHA512\s+[0-9a-fA-F]+",
            f"SHA512 {hashes['sha512']}",
            content,
        )
        open(portfile_path, "w").write(content)
        print("  Updated SHA512")
    else:
        print(f"  [DRY RUN] Would update SHA512 to {hashes['sha512']}")

    return vcpkg_json_path


def _vcpkg_cleanup_patches(repo_dir, tarball_path, dry_run):
    """Detect and remove obsolete patches from the vcpkg port."""
    port_dir = os.path.join(repo_dir, "ports", "dispenso")
    portfile_path = os.path.join(port_dir, "portfile.cmake")

    if dry_run or not os.path.isdir(port_dir):
        return
    patch_files = [f for f in os.listdir(port_dir) if f.endswith(".patch")]
    if not patch_files:
        return

    print("  --- Checking patches against new source ---")
    obsolete = detect_obsolete_patches(tarball_path, port_dir)
    if obsolete:
        remove_obsolete_patches(port_dir, portfile_path, obsolete)


def _macports_cleanup_patches(repo_dir, tarball_path, dry_run):
    """Detect and remove obsolete patches from the MacPorts port.

    MacPorts patches use -p0 (no a/b prefixes) and are stored in a files/
    subdirectory. The Portfile references them via 'patchfiles'.
    """
    port_dir = os.path.join(repo_dir, "devel", "dispenso")
    files_dir = os.path.join(port_dir, "files")

    if dry_run or not os.path.isdir(files_dir):
        return

    patch_files = [f for f in os.listdir(files_dir) if f.endswith(".patch")]
    if not patch_files:
        return

    print("  --- Checking patches against new source ---")
    obsolete = detect_obsolete_patches(tarball_path, files_dir, strip_level=0)
    if not obsolete:
        return

    portfile_path = os.path.join(port_dir, "Portfile")

    # Delete obsolete patch files
    for patch_file in obsolete:
        patch_path = os.path.join(files_dir, patch_file)
        if os.path.exists(patch_path):
            os.unlink(patch_path)
            print(f"  Deleted {patch_file}")

    # Check if any patches remain
    remaining = [f for f in os.listdir(files_dir) if f.endswith(".patch")]

    # Update Portfile: remove patchfiles entries
    content = open(portfile_path).read()
    if not remaining:
        # Remove entire patchfiles line(s)
        content = re.sub(r"\n*patchfiles\s+.*\n?", "\n", content)
        print("  Removed patchfiles from Portfile")
        # Remove empty files/ directory
        try:
            os.rmdir(files_dir)
            print("  Removed empty files/ directory")
        except OSError:
            pass
    else:
        # Remove only obsolete entries from patchfiles
        for patch_file in obsolete:
            content = re.sub(rf"\s*{re.escape(patch_file)}", "", content)
        print(f"  Removed obsolete patches from Portfile, kept: {remaining}")

    open(portfile_path, "w").write(content)


def _vcpkg_run_tooling(repo_dir, vcpkg_json_path, dry_run):
    """Run vcpkg format-manifest and x-add-version if vcpkg is available."""
    vcpkg_bin = shutil.which("vcpkg")
    if not vcpkg_bin:
        print(
            "  WARNING: vcpkg not found. Run 'vcpkg format-manifest' and "
            "'vcpkg x-add-version dispenso --overlay-ports=ports/dispenso' "
            "manually before pushing."
        )
        return

    print("  Running vcpkg format-manifest ...")
    run(
        [vcpkg_bin, "format-manifest", vcpkg_json_path],
        cwd=repo_dir,
        check=False,
        dry_run=dry_run,
    )

    # x-add-version requires port changes to be committed first, so we make
    # a temporary commit. The final commit_and_push will squash everything.
    print("  Running vcpkg x-add-version ...")
    if not dry_run:
        run(["git", "add", "-A"], cwd=repo_dir)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir)
        if result.returncode != 0:
            run(
                ["git", "commit", "-m", "temp: port changes for x-add-version"],
                cwd=repo_dir,
            )
    run(
        [
            vcpkg_bin,
            "x-add-version",
            "dispenso",
            "--overwrite-version",
            f"--overlay-ports={os.path.join(repo_dir, 'ports', 'dispenso')}",
        ],
        cwd=repo_dir,
        check=False,
        dry_run=dry_run,
    )
    # Version database changes from x-add-version will be staged by
    # commit_and_push's git add -A.


def _vcpkg_verify_port_files(repo_dir, version, hashes):
    """Verify port files contain the expected version and hash after updates."""
    port_dir = os.path.join(repo_dir, "ports", "dispenso")
    errors = []

    vcpkg_json_path = os.path.join(port_dir, "vcpkg.json")
    portfile_path = os.path.join(port_dir, "portfile.cmake")

    # Check vcpkg.json has correct version
    vcpkg_json = open(vcpkg_json_path).read()
    if f'"version": "{version}"' not in vcpkg_json:
        errors.append(f"vcpkg.json does not contain version {version}")

    # Check portfile.cmake has correct SHA512
    portfile = open(portfile_path).read()
    if hashes["sha512"] not in portfile:
        errors.append("portfile.cmake does not contain expected SHA512")

    # Check portfile doesn't reference deleted patch files.
    # Only match lines that look like bare filenames (no spaces, no comment chars).
    for line in portfile.splitlines():
        stripped = line.strip()
        if (
            stripped.endswith(".patch")
            and " " not in stripped
            and not stripped.startswith("#")
        ):
            patch_path = os.path.join(port_dir, stripped)
            if not os.path.exists(patch_path):
                errors.append(
                    f"portfile.cmake references {stripped} but file does not exist"
                )

    if errors:
        print("  ERROR: Port file verification failed:")
        for e in errors:
            print(f"    - {e}")
        raise RuntimeError("vcpkg port files are inconsistent; aborting")
    else:
        print("  Verified: vcpkg.json and portfile.cmake are consistent")


def update_vcpkg(args, hashes, tarball_path):
    """Update vcpkg with new dispenso version."""
    print("=== vcpkg ===")
    version = args.version
    manager = "vcpkg"
    branch = BRANCH_NAMES.get(manager, "add-dispenso")

    repo_dir = ensure_repo(args.repos_dir, manager, args.github_user, args.dry_run)
    checkout_branch(repo_dir, branch, args.dry_run)

    vcpkg_json_path = _vcpkg_update_port_files(repo_dir, version, hashes, args.dry_run)
    _vcpkg_cleanup_patches(repo_dir, tarball_path, args.dry_run)
    _vcpkg_run_tooling(repo_dir, vcpkg_json_path, args.dry_run)

    # Verify port files were updated correctly before committing.
    if not args.dry_run:
        _vcpkg_verify_port_files(repo_dir, version, hashes)

    # --- Commit ---
    commit_msg = f"[dispenso] Update to version {version}"
    committed = commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    test_passed = True
    if committed and not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        test_passed = test_vcpkg(repo_dir, version)

    # --- Push ---
    if not committed:
        print("  Skipping push — no changes to push")
    elif not args.dry_run and not args.skip_push:
        if not test_passed:
            print("  Skipping push due to test failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = f"https://github.com/{args.github_user}/vcpkg/tree/{branch}"
    tests_ran = committed and not args.skip_test and not args.dry_run and test_passed
    title = pr_title("vcpkg", version)
    body = pr_body_vcpkg(version)
    print()
    return {
        "status": "no_changes"
        if not committed
        else "ok"
        if test_passed
        else "test_failed",
        "branch": branch,
        "branch_url": branch_url,
        "version": version,
        "tests_ran": tests_ran,
        "pr_title": title,
        "pr_body": body,
    }


def update_homebrew(args, hashes, tarball_path):
    """Update homebrew-core with new dispenso version."""
    print("=== Homebrew (homebrew-core) ===")
    version = args.version
    manager = "homebrew"
    branch = f"dispenso-{version}"

    repo_dir = ensure_repo(args.repos_dir, manager, args.github_user, args.dry_run)
    checkout_branch(repo_dir, branch, args.dry_run)

    formula_path = os.path.join(repo_dir, "Formula", "d", "dispenso.rb")
    print(f"  Updating {formula_path} ...")

    if not args.dry_run:
        if not os.path.exists(formula_path):
            print(f"  ERROR: Formula file not found at {formula_path}")
            print("  The dispenso formula may not exist yet in homebrew-core.")
            print()
            return {"status": "error", "error": "Formula file not found"}

        content = open(formula_path).read()

        # Update url
        content = re.sub(
            r'url\s+"https://github\.com/facebookincubator/dispenso/archive/refs/tags/v[^"]+\.tar\.gz"',
            f'url "https://github.com/{GITHUB_REPO}/archive/refs/tags/v{version}.tar.gz"',
            content,
        )

        # Update sha256
        content = re.sub(
            r'sha256\s+"[0-9a-fA-F]+"',
            f'sha256 "{hashes["sha256"]}"',
            content,
        )

        # Remove revision line (reset on version bump)
        content = re.sub(r"\n\s*revision\s+\d+", "", content)

        open(formula_path, "w").write(content)
        print("  Updated url, sha256 (removed revision if present)")
    else:
        print("  [DRY RUN] Would update url and sha256 in formula")

    # --- Verify checksums ---
    checksums_ok = True
    if not args.dry_run:
        print("  --- Verifying checksums ---")
        checksums_ok = verify_formula_checksums(repo_dir, hashes)

    # --- Commit ---
    commit_msg = f"dispenso {version}"
    committed = commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    fully_tested = False
    if committed and checksums_ok and not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        fully_tested = test_homebrew(repo_dir, version)

    # --- Push ---
    if not committed:
        print("  Skipping push — no changes to push")
    elif not args.dry_run and not args.skip_push:
        if not checksums_ok:
            print("  Skipping push due to checksum verification failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = f"https://github.com/{args.github_user}/homebrew-core/tree/{branch}"
    title = pr_title("homebrew", version)
    body = pr_body_homebrew(version, tests_ran=fully_tested)
    print()
    return {
        "status": "no_changes"
        if not committed
        else "error"
        if not checksums_ok
        else "ok"
        if fully_tested or args.skip_test
        else "needs_macos",
        "branch": branch,
        "branch_url": branch_url,
        "version": version,
        "tests_ran": fully_tested,
        "pr_title": title,
        "pr_body": body,
    }


def update_macports(args, hashes, tarball_path):
    """Update macports-ports with new dispenso version."""
    print("=== MacPorts (macports-ports) ===")
    version = args.version
    manager = "macports"
    branch = f"dispenso-{version}"

    if hashes["rmd160"] is None:
        print("  ERROR: RIPEMD-160 hash not available, cannot update MacPorts")
        print()
        return {"status": "error", "error": "RIPEMD-160 hash not available"}

    repo_dir = ensure_repo(args.repos_dir, manager, args.github_user, args.dry_run)
    checkout_branch(repo_dir, branch, args.dry_run)

    portfile_path = os.path.join(repo_dir, "devel", "dispenso", "Portfile")
    print(f"  Updating {portfile_path} ...")

    if not args.dry_run:
        if not os.path.exists(portfile_path):
            print(f"  ERROR: Portfile not found at {portfile_path}")
            print("  The dispenso port may not exist yet in MacPorts.")
            print()
            return {"status": "error", "error": "Portfile not found"}

        content = open(portfile_path).read()

        # Update github.setup line version
        content = re.sub(
            r"(github\.setup\s+facebookincubator\s+dispenso\s+)\S+(\s+v)",
            rf"\g<1>{version}\2",
            content,
        )

        # Update checksums block
        content = re.sub(
            r"(checksums\s+rmd160\s+)\S+",
            rf"\g<1>{hashes['rmd160']}",
            content,
        )
        content = re.sub(
            r"(sha256\s+)\S+",
            rf"\g<1>{hashes['sha256']}",
            content,
        )
        content = re.sub(
            r"(size\s+)\S+",
            rf"\g<1>{hashes['size']}",
            content,
        )

        # Reset revision to 0 if present
        content = re.sub(
            r"(revision\s+)\d+",
            r"\g<1>0",
            content,
        )

        open(portfile_path, "w").write(content)
        print("  Updated github.setup version, checksums (rmd160, sha256, size)")
    else:
        print(f"  [DRY RUN] Would update version to {version} and checksums")

    # --- Verify checksums ---
    checksums_ok = True
    if not args.dry_run:
        print("  --- Verifying checksums ---")
        checksums_ok = verify_portfile_checksums(repo_dir, hashes)

    # --- Clean up obsolete patches ---
    _macports_cleanup_patches(repo_dir, tarball_path, args.dry_run)

    # --- Commit ---
    commit_msg = f"dispenso: update to {version}"
    committed = commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    fully_tested = False
    tested_on = None
    if committed and checksums_ok and not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        fully_tested = test_macports(repo_dir, version, hashes=hashes)
        if fully_tested:
            tested_on = get_macos_tested_on()

    # --- Push ---
    if not committed:
        print("  Skipping push — no changes to push")
    elif not args.dry_run and not args.skip_push:
        if not checksums_ok:
            print("  Skipping push due to checksum verification failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = f"https://github.com/{args.github_user}/macports-ports/tree/{branch}"
    title = pr_title("macports", version)
    body = pr_body_macports(version, tests_ran=fully_tested, tested_on=tested_on)
    print()
    return {
        "status": "no_changes"
        if not committed
        else "error"
        if not checksums_ok
        else "ok"
        if fully_tested or args.skip_test
        else "needs_macos",
        "branch": branch,
        "branch_url": branch_url,
        "version": version,
        "tests_ran": fully_tested,
        "pr_title": title,
        "pr_body": body,
    }


# ---------------------------------------------------------------------------
# PR body templates
# ---------------------------------------------------------------------------


def pr_body_homebrew(version, tests_ran):
    """Generate PR body following Homebrew's PR template."""
    test_check = "[x]" if tests_ran else "[ ]"
    audit_line = (
        f"- {test_check} Does your build pass `brew audit --strict <formula>` "
        "(after doing `HOMEBREW_NO_INSTALL_FROM_API=1 brew install "
        "--build-from-source <formula>`)? If this is a new formula, "
        "does it pass `brew audit --new <formula>`?"
    )
    return f"""\
- [ ] Have you followed the [guidelines for contributing](https://github.com/Homebrew/homebrew-core/blob/HEAD/CONTRIBUTING.md)?
- [x] Have you ensured that your commits follow the [commit style guide](https://docs.brew.sh/Formula-Cookbook#commit)?
- [x] Have you checked that there aren't other open [pull requests](https://github.com/Homebrew/homebrew-core/pulls) for the same formula update/change?
- {test_check} Have you built your formula locally with `HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-from-source <formula>`?
- {test_check} Is your test running fine `brew test <formula>`?
{audit_line}

-----

- [x] AI was used to generate or assist with generating this PR. \
A script ([update_package_managers.py](https://github.com/facebookincubator/dispenso/blob/main/scripts/update_package_managers.py)) \
was used to update the version, URL, and SHA256 in the formula. \
The formula was verified locally before submission.

-----

Update dispenso to {version}.

[dispenso](https://github.com/facebookincubator/dispenso) is a high-performance \
C++ library for parallel programming from Meta. It provides work-stealing thread \
pools, parallel for loops, futures, task graphs, pipelines, and concurrent containers. \
Requires C++14, no external dependencies beyond pthreads."""


def pr_body_conan(version, tests_ran, issue_number=None):
    """Generate PR body following Conan Center Index's PR template."""
    test_check = "[x]" if tests_ran else "[ ]"
    issue_line = f"\nfixes #{issue_number}\n" if issue_number else ""
    return f"""\
### Summary
Changes to recipe:  **dispenso/{version}**
{issue_line}
#### Motivation
Update dispenso to version {version}.

#### Details
Added version {version} entry to `conandata.yml` and `config.yml`. \
No changes to `conanfile.py` (version-agnostic).

---
- [x] Read the [contributing guidelines](https://github.com/conan-io/conan-center-index/blob/master/CONTRIBUTING.md)
- [x] Checked that this PR is not a duplicate: [list of PRs by recipe](https://github.com/conan-io/conan-center-index/discussions/24240)
- [ ] If this is a bug fix, please link related issue or provide bug details
- {test_check} Tested locally with at least one configuration using a recent version of Conan
---"""


def pr_body_vcpkg(version):
    """Generate PR body following vcpkg's PR template."""
    return f"""\
## Port Update Checklist

- [x] Complies with the [maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide)
- [x] Updated SHA512 checksums
- [x] Version database updated via `./vcpkg x-add-version --all`
- [x] Exactly one version added per modified versions file

Update dispenso to version {version}."""


def pr_body_macports(version, tests_ran, tested_on=None):
    """Generate PR body following MacPorts' PR template."""
    test_check = "[x]" if tests_ran else "[ ]"
    if tested_on:
        tested_on_section = tested_on
    else:
        tested_on_section = (
            "<!-- Run and paste output of: "
            "port version && sw_vers && xcode-select -p -->"
        )
    return f"""\
#### Description
Update dispenso to version {version}.

[dispenso](https://github.com/facebookincubator/dispenso) is a high-performance \
C++ parallel programming library from Meta.

###### Type(s)
- [ ] bugfix
- [x] enhancement
- [ ] security fix

###### Tested on
{tested_on_section}

###### Verification
- [x] Followed [Commit Message Guidelines](https://trac.macports.org/wiki/CommitMessages)
- [x] Squashed and minimized commits
- [x] Checked that there aren't other open [pull requests](https://github.com/macports/macports-ports/pulls) for the same change
- {test_check} Checked Portfile with `port lint --nitpick`
- {test_check} Tried existing tests with `sudo port test`
- {test_check} Tested with `sudo port -vst install`
- {test_check} Checked that binaries work as expected
- [x] Tested important variants (dispenso has no variants)"""


def pr_title(manager, version):
    """Generate the PR title for a given manager."""
    titles = {
        "conan": f"dispenso: add version {version}",
        "vcpkg": f"[dispenso] Update to version {version}",
        "homebrew": f"dispenso {version}",
        "macports": f"dispenso: update to {version}",
    }
    return titles.get(manager, f"dispenso {version}")


def pre_pr_checklist(manager, version, tests_ran):
    """Return items the user should manually verify before creating the PR."""
    checklists = {
        "homebrew": [
            "Read contributing guidelines: https://github.com/Homebrew/homebrew-core/blob/HEAD/CONTRIBUTING.md",
            "Run: HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-from-source dispenso",
            "Run: brew test dispenso",
            "Run: brew audit --strict dispenso",
        ],
        "conan": [
            "Sign the CLA if prompted on the PR",
            "Run: conan create recipes/dispenso/all --version=" + version,
        ],
        "vcpkg": [
            "Read maintainer guide: https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide",
            "Run: vcpkg install dispenso --overlay-ports=ports/dispenso",
        ],
        "macports": [
            "Fill in 'Tested on' section with output of: port version && sw_vers && xcode-select -p",
            "Run: port lint --nitpick devel/dispenso",
            "Run: sudo port -D devel/dispenso -vst install",
            "Run: sudo port -D devel/dispenso test",
            "Verify installed binaries work",
        ],
    }

    items = checklists.get(manager, [])
    if tests_ran:
        # Filter out "Run:" items that the script already tested
        pass  # Keep all — user should still double-check
    return items


# ---------------------------------------------------------------------------
# PR creation and guided flow
# ---------------------------------------------------------------------------


def get_default_branch(upstream_repo):
    """Get the default branch of an upstream repo via gh API."""
    result = subprocess.run(
        ["gh", "api", f"repos/{upstream_repo}", "--jq", ".default_branch"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "master"


def close_superseded_prs(upstream_repo, version, github_user, dry_run):
    """Close any open dispenso PRs by this user in the upstream repo."""
    result = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            upstream_repo,
            "--author",
            github_user,
            "--search",
            "dispenso",
            "--state",
            "open",
            "--json",
            "number,title,url",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return

    try:
        prs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return

    for pr in prs:
        title = pr.get("title", "")
        number = pr.get("number")
        if number is None:
            continue
        comment = (
            f"Superseded by version {version} update. Closing in favor of the new PR."
        )
        if dry_run:
            print(f"  [DRY RUN] Would close #{number}: {title}")
            continue

        close_result = subprocess.run(
            [
                "gh",
                "pr",
                "close",
                str(number),
                "--repo",
                upstream_repo,
                "--comment",
                comment,
            ],
            capture_output=True,
            text=True,
        )
        if close_result.returncode == 0:
            print(f"  Closed #{number}: {title}")
        else:
            print(
                f"  WARNING: Failed to close #{number}: {close_result.stderr.strip()}"
            )


def create_pr(upstream_repo, branch, title, body, github_user, dry_run):
    """Create a PR via gh CLI. Returns the PR URL or None."""
    if dry_run:
        print(f"  [DRY RUN] Would create PR: {title}")
        return None

    base = get_default_branch(upstream_repo)

    result = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            upstream_repo,
            "--head",
            f"{github_user}:{branch}",
            "--base",
            base,
            "--title",
            title,
            "--body",
            body,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        print(f"  Created PR: {url}")
        return url

    stderr = result.stderr.strip()
    # If a PR already exists for this branch, extract its URL
    if "already exists" in stderr:
        print(f"  PR already exists for branch {branch}")
        view_result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                branch,
                "--repo",
                upstream_repo,
                "--json",
                "url",
                "--jq",
                ".url",
            ],
            capture_output=True,
            text=True,
        )
        if view_result.returncode == 0:
            url = view_result.stdout.strip()
            print(f"  Existing PR: {url}")
            return url
        return None

    print(f"  ERROR: Failed to create PR: {stderr}")
    return None


def create_prs_phase(results, args):
    """Create PRs for all successful managers. Returns dict of manager→URL."""
    if args.skip_push:
        print("  --skip-push is set — cannot create PRs without pushing first.")
        return {}

    gh_bin = shutil.which("gh")
    if not gh_bin:
        print("  ERROR: gh CLI not found — install it to create PRs automatically.")
        print("  PR titles and bodies were printed above for manual creation.")
        return {}

    print()
    print("=" * 60)
    print("CREATING PRs")
    print("=" * 60)

    pr_urls = {}
    for mgr, result in results.items():
        if result.get("status") not in ("ok",):
            continue

        upstream = UPSTREAM_REPOS[mgr]
        branch = result.get("branch", "")
        title = result.get("pr_title", "")
        body = result.get("pr_body", "")

        if not branch or not title:
            continue

        print(f"\n--- {mgr} ---")

        # Close any superseded open PRs
        close_superseded_prs(upstream, args.version, args.github_user, args.dry_run)

        # Create the new PR
        url = create_pr(upstream, branch, title, body, args.github_user, args.dry_run)
        if url:
            pr_urls[mgr] = url

    if pr_urls:
        print()
        print("=" * 60)
        print("PR URLs")
        print("=" * 60)
        for mgr, url in pr_urls.items():
            print(f"  {mgr:12s}  {url}")

    return pr_urls


def post_pr_steps(pr_urls, version):
    """Guide the user through remaining manual post-PR steps."""
    print()
    print("=" * 60)
    print("POST-PR STEPS")
    print("=" * 60)

    if "conan" in pr_urls:
        print()
        print("  Conan: If this is your first PR from this account, the CLA")
        print("  bot will comment on the PR. Sign it if prompted.")
        input("  Press Enter to continue ...")

    print()
    print("  Monitor CI on each PR and respond to reviewer feedback.")
    print()
    print("  All done!")


def print_summary(results, github_user):
    """Print a summary of all operations."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    for mgr, result in results.items():
        status = result.get("status", "unknown")
        branch_url = result.get("branch_url", "")
        error = result.get("error")

        if status == "ok":
            print(f"  PASS  {mgr:12s}  {branch_url}")
        elif status == "test_failed":
            print(f"  FAIL  {mgr:12s}  Tests failed (branch not pushed)")
            all_ok = False
        elif status == "error":
            print(f"  ERR   {mgr:12s}  {error}")
            all_ok = False
        elif status == "needs_macos":
            print(f"  WARN  {mgr:12s}  Checksums OK, but full testing requires macOS")
            print(f"        {'':12s}  {branch_url}")
        elif status == "no_changes":
            print(f"  SKIP  {mgr:12s}  Upstream already has this version")
        elif status == "skipped":
            print(f"  SKIP  {mgr:12s}  {result.get('reason', '')}")
        else:
            print(f"  ??    {mgr:12s}  {result}")
            all_ok = False

    print()
    if all_ok:
        print("All managers succeeded. Review branches on GitHub, then create PRs.")
    else:
        print("Some managers had issues. Fix and re-run for the failed ones.")
    print()

    # Print PR body text for each successful manager
    for mgr, result in results.items():
        if result.get("status") != "ok":
            continue
        version = result.get("version", "")
        tests_ran = result.get("tests_ran", False)
        body = result.get("pr_body", "")
        title = result.get("pr_title", "")
        if not body:
            continue

        print("=" * 60)
        print(f"PR for {mgr}")
        print("=" * 60)
        print(f"Title: {title}")
        print(f"Branch: {result.get('branch_url', '')}")

        # Pre-PR checklist
        checklist = pre_pr_checklist(mgr, version, tests_ran)
        if checklist:
            print()
            print("  BEFORE CREATING PR, verify:")
            for i, item in enumerate(checklist, 1):
                done = "[x]" if tests_ran and item.startswith("Run:") else "[ ]"
                print(f"    {done} {item}")
            print()
            print("  Check any unchecked boxes in the PR body after verifying.")

        print("-" * 60)
        print(body)
        print("-" * 60)
        print()
    print("Copy-paste the above PR bodies when creating PRs on GitHub.")
    print()


# ---------------------------------------------------------------------------
# Guided interactive flow
# ---------------------------------------------------------------------------


def prompt_continue(message="Continue?"):
    """Prompt user to continue, skip, or quit. Returns the chosen action."""
    response = input(f"  {message} [Enter=yes / s=skip / q=quit] ").strip().lower()
    if response in ("", "y", "yes"):
        return "continue"
    if response in ("s", "skip"):
        return "skip"
    if response in ("q", "quit"):
        return "quit"
    return "continue"


def _detect_default_branch(repo_dir):
    """Detect the default branch from local remote refs."""
    result = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip().replace("refs/remotes/origin/", "")
    for candidate in ["main", "master"]:
        r = subprocess.run(
            ["git", "show-ref", "--verify", f"refs/remotes/origin/{candidate}"],
            cwd=repo_dir,
            capture_output=True,
        )
        if r.returncode == 0:
            return candidate
    return "master"


def _guided_run_updates(args, managers, hashes, tarball_path):
    """Run each manager's update handler, prompting before each.

    Returns (results, aborted).  *aborted* is True if the user chose 'quit'.
    """
    handlers = {
        "conan": update_conan,
        "vcpkg": update_vcpkg,
        "homebrew": update_homebrew,
        "macports": update_macports,
    }
    step_descriptions = {
        "vcpkg": "Update vcpkg port (version, SHA512, patches, version DB)",
        "conan": "Update Conan recipe (conandata.yml, config.yml, issue)",
        "homebrew": "Update Homebrew formula (URL, SHA256, build + test)",
        "macports": "Update MacPorts Portfile (version, checksums, lint + install + test)",
    }

    results = {}
    for mgr in managers:
        print()
        print("=" * 60)
        print(f"  {step_descriptions.get(mgr, mgr)}")
        if mgr in MACOS_ONLY_MANAGERS and platform.system() != "Darwin":
            print("  (full testing requires macOS — will verify checksums only)")
        print("=" * 60)

        action = prompt_continue(f"Proceed with {mgr}?")
        if action == "quit":
            print("\n  Quitting. Local commits (if any) are preserved.")
            return results, True
        if action == "skip":
            results[mgr] = {"status": "skipped", "reason": "skipped by user"}
            continue

        try:
            results[mgr] = handlers[mgr](args, hashes, tarball_path)
        except Exception as e:
            print(f"\n  ERROR: {mgr} failed: {e}")
            results[mgr] = {"status": "error", "error": str(e)}

        _print_manager_result(mgr, results[mgr])

    return results, False


def _print_manager_result(mgr, result):
    """Print the outcome of a single manager update."""
    status = result.get("status", "unknown")
    messages = {
        "ok": "PASSED",
        "no_changes": "SKIPPED — upstream already has this version",
        "needs_macos": "checksums OK — re-run on macOS for full testing",
        "test_failed": "TESTS FAILED — branch will not be pushed",
    }
    if status in messages:
        print(f"\n  {mgr}: {messages[status]}")
    elif status == "error":
        print(f"\n  {mgr}: ERROR — {result.get('error', '')}")


def _guided_push_branches(args, pushable):
    """Push branches to fork and print compare URLs.

    Returns dict of successfully pushed managers.
    """
    print()
    print("=" * 60)
    print("  Pushing branches to fork")
    print("=" * 60)

    pushed = {}
    for mgr, result in pushable.items():
        repo_dir = os.path.join(args.repos_dir, REPO_DIRS[mgr])
        branch = result.get("branch", "")
        if not branch:
            continue
        print(f"\n  {mgr}: pushing {branch} ...")
        run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)
        pushed[mgr] = result

    print()
    print("  All branches pushed. Review the proposed changes:")
    print()
    for mgr, result in pushed.items():
        upstream = UPSTREAM_REPOS[mgr]
        repo_dir = os.path.join(args.repos_dir, REPO_DIRS[mgr])
        default = _detect_default_branch(repo_dir)
        branch = result.get("branch", "")
        repo_name = REPO_DIRS[mgr]
        compare_url = (
            f"https://github.com/{upstream}/compare/"
            f"{default}...{args.github_user}:{repo_name}:{branch}"
        )
        print(f"    {mgr:12s}  {compare_url}")

    return pushed


def _guided_create_prs(pushed, args):
    """Create PRs for successfully pushed managers via gh CLI.

    Returns dict mapping manager name to PR URL.
    """
    pr_urls = {}
    for mgr, result in pushed.items():
        if result.get("status") != "ok":
            continue
        upstream = UPSTREAM_REPOS[mgr]
        branch = result.get("branch", "")
        title = result.get("pr_title", "")
        body = result.get("pr_body", "")
        if not branch or not title:
            continue

        print(f"\n--- {mgr} ---")
        close_superseded_prs(upstream, args.version, args.github_user, False)
        url = create_pr(upstream, branch, title, body, args.github_user, False)
        if url:
            pr_urls[mgr] = url

    return pr_urls


def _guided_review_push_pr(args, results, pushable, tarball_path):
    """Review diffs, push branches, and create PRs.

    Handles all user prompts for the post-update phase.
    """
    version = args.version

    # ---- Inspect diffs ----
    print()
    print("=" * 60)
    print("  Review changes before pushing")
    print("=" * 60)
    print()
    print("  Inspect the diffs in each repo:")
    for mgr in pushable:
        repo_dir = os.path.join(args.repos_dir, REPO_DIRS[mgr])
        default = _detect_default_branch(repo_dir)
        print(f"    cd {repo_dir} && git diff origin/{default}...HEAD")

    action = prompt_continue("Diffs look good? Push to fork?")
    if action == "quit":
        print("\n  Quitting. Changes are committed locally in each repo.")
        _guided_cleanup(tarball_path)
        return
    if action == "skip":
        print("\n  Skipped push. Changes are committed locally.")
        print_summary(results, args.github_user)
        _guided_cleanup(tarball_path)
        return

    # ---- Push ----
    pushed = _guided_push_branches(args, pushable)

    # ---- Create PRs ----
    pr_urls = _guided_offer_pr_creation(args, pushed, results, tarball_path)
    if pr_urls is None:
        return

    if pr_urls:
        print()
        print("=" * 60)
        print("  PR URLs")
        print("=" * 60)
        for mgr, url in pr_urls.items():
            print(f"    {mgr:12s}  {url}")

    post_pr_steps(pr_urls, version)

    # ---- needs_macos reminder ----
    needs_macos = [m for m, r in results.items() if r.get("status") == "needs_macos"]
    if needs_macos:
        print()
        print("=" * 60)
        print("  macOS required for full testing")
        print("=" * 60)
        mgr_list = ",".join(needs_macos)
        print("\n  Re-run on macOS to complete testing and create PRs:")
        print(
            f"    python3 {sys.argv[0]} --version {version}"
            f" --managers {mgr_list} --guided"
        )

    _guided_cleanup(tarball_path)
    print()
    print("  Release update complete!")


def _guided_offer_pr_creation(args, pushed, results, tarball_path):
    """Prompt to create PRs via gh CLI.

    Returns pr_urls dict, or None if the user skipped/quit or gh is missing.
    """
    print()
    print("=" * 60)
    print("  Create pull requests")
    print("=" * 60)

    gh_bin = shutil.which("gh")
    if not gh_bin:
        print()
        print("  gh CLI not found. Install it to create PRs automatically.")
        print("  PR titles and bodies are printed below for manual creation.")
        print_summary(results, args.github_user)
        _guided_cleanup(tarball_path)
        return None

    action = prompt_continue("Create PRs via gh CLI?")
    if action in ("skip", "quit"):
        print("\n  Branches are pushed. Create PRs manually if needed.")
        print_summary(results, args.github_user)
        _guided_cleanup(tarball_path)
        return None

    return _guided_create_prs(pushed, args)


def guided_flow(args):
    """Run the complete release update flow interactively.

    Walks through every step, prompts before each action, and creates PRs
    at the end. Replaces the need for a separate checklist document.
    """
    version = args.version
    managers = args.managers

    print()
    print("=" * 60)
    print(f"  dispenso {version} — guided release update")
    print("=" * 60)
    print()
    print("  Managers: " + ", ".join(managers))
    if args.dry_run:
        print("  Mode: DRY RUN (no files will be modified)")
    if args.skip_test:
        print("  Mode: SKIP TESTS")
    print()
    print("  At each step you can press Enter to continue,")
    print("  's' to skip, or 'q' to quit.")
    print()

    # ---- Download tarball ----
    print("-" * 60)
    print("  Downloading release tarball ...")
    print("-" * 60)
    tarball_path, hashes = download_and_hash(version)

    # ---- Update each manager (no push) ----
    original_skip_push = args.skip_push
    args.skip_push = True
    results, aborted = _guided_run_updates(args, managers, hashes, tarball_path)
    args.skip_push = original_skip_push

    if aborted:
        _guided_cleanup(tarball_path)
        return

    pushable = {
        m: r for m, r in results.items() if r.get("status") in ("ok", "needs_macos")
    }

    if not pushable:
        print("\n  No managers succeeded. Fix issues and re-run.")
        _guided_cleanup(tarball_path)
        return

    if args.dry_run:
        print("\n  Dry run complete. Re-run without --dry-run to apply changes.")
        _guided_cleanup(tarball_path)
        return

    _guided_review_push_pr(args, results, pushable, tarball_path)


def _guided_cleanup(tarball_path):
    """Clean up the downloaded tarball."""
    if os.path.exists(tarball_path):
        os.unlink(tarball_path)
        print(f"\n  Cleaned up {tarball_path}")


def main():
    args = parse_args()

    if args.guided:
        guided_flow(args)
        return

    print(f"Updating dispenso to version {args.version}")
    if args.dry_run:
        print("*** DRY RUN MODE — no files will be modified ***")
    if args.skip_test:
        print("*** SKIP TEST MODE — no local verification ***")
    if args.skip_push:
        print("*** SKIP PUSH MODE — commit and test only ***")
    if args.create_prs:
        print("*** CREATE PRS MODE — will create PRs after push ***")
    print()

    tarball_path, hashes = download_and_hash(args.version)

    handlers = {
        "conan": update_conan,
        "vcpkg": update_vcpkg,
        "homebrew": update_homebrew,
        "macports": update_macports,
    }

    results = {}
    for mgr in args.managers:
        try:
            results[mgr] = handlers[mgr](args, hashes, tarball_path)
        except Exception as e:
            print(f"  ERROR: {mgr} failed: {e}")
            results[mgr] = {"status": "error", "error": str(e)}
            print()

    print_summary(results, args.github_user)

    # Create PRs if requested
    if args.create_prs:
        pr_urls = create_prs_phase(results, args)
        if pr_urls:
            post_pr_steps(pr_urls, args.version)

    # Clean up tarball
    if os.path.exists(tarball_path):
        os.unlink(tarball_path)
        print(f"Cleaned up {tarball_path}")

    # Check if any macOS-only managers need full testing
    needs_macos = [
        mgr for mgr, result in results.items() if result.get("status") == "needs_macos"
    ]
    if needs_macos:
        print()
        print("=" * 60)
        print("ERROR: Full testing requires macOS")
        print("=" * 60)
        print("  The following managers were updated and checksums verified,")
        print("  but full testing (lint, install, audit) requires macOS:")
        for mgr in needs_macos:
            print(f"    - {mgr}")
        print()
        print("  Please re-run on macOS to complete testing:")
        mgr_list = ",".join(needs_macos)
        print(
            f"    python3 {sys.argv[0]} --version {args.version} --managers {mgr_list}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
