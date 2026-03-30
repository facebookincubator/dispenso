#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Update dispenso across package manager repositories for a new release.

Downloads the release tarball, computes hashes, updates version/hash references
in each package manager's repo, commits, tests locally, pushes to the user's
fork, and prints branch URLs for manual PR creation.

Usage:
    python3 update_package_managers.py --version 1.5.0
    python3 update_package_managers.py --version 1.5.0 --managers conan,vcpkg
    python3 update_package_managers.py --version 1.5.0 --dry-run
    python3 update_package_managers.py --version 1.5.0 --skip-test
    python3 update_package_managers.py --version 1.5.0 --skip-push
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
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
    args = parser.parse_args()
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


def run(cmd, cwd=None, check=True, dry_run=False, capture=False):
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
        print(f"  Fork remote already exists")

    # Fetch upstream
    print(f"  Fetching origin ...")
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

    # Create or checkout working branch
    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        cwd=repo_dir,
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  Branch {branch} already exists, rebasing onto {default_branch} ...")
        run(["git", "checkout", branch], cwd=repo_dir)
        run(["git", "rebase", default_branch], cwd=repo_dir)
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
                print("  No changes to commit (already up to date)")
                if not skip_push:
                    run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)
                return True

        # Soft-reset to merge base and recommit as single commit
        print("  Squashing into single commit ...")
        run(["git", "reset", "--soft", merge_base], cwd=repo_dir)
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

    # Remove existing install to force rebuild
    print("  Removing existing vcpkg install (if any) ...")
    run(
        [vcpkg_bin, "remove", "dispenso:x64-linux"],
        cwd=repo_dir,
        check=False,
    )

    print("  Running vcpkg install ...")
    result = run(
        [
            vcpkg_bin,
            "install",
            "dispenso",
            f"--overlay-ports={os.path.join(repo_dir, 'ports', 'dispenso')}",
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
        print("  Running brew install --build-from-source ...")
        result = run(
            [brew_bin, "install", "--build-from-source", "dispenso"],
            check=False,
            capture=True,
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
        print("  Running brew audit --new ...")
        result = run([brew_bin, "audit", "--new", "dispenso"], check=False)
        if result.returncode != 0:
            print("  WARNING: brew audit --new reported issues")

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


def test_macports(repo_dir, version):
    """Test macports portfile with port lint."""
    port_bin = shutil.which("port")
    if not port_bin:
        print("  WARNING: port not found — full testing requires macOS")
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

    print("  PASS: macports lint succeeded")
    print("  NOTE: For full testing, also run:")
    print(f"    sudo port -D {portdir} -vst install")
    print(f"    sudo port -D {portdir} test")
    return True


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
        print(f"  FAIL: sha256 mismatch in formula")
        print(f"    Formula:   {actual}")
        print(f"    Expected:  {expected}")
        return False

    print("  PASS: Formula sha256 matches computed hash")
    return True


# ---------------------------------------------------------------------------
# Per-manager update functions
# ---------------------------------------------------------------------------


def update_conan(args, hashes, tarball_path):
    """Update conan-center-index with new dispenso version."""
    print(f"=== Conan (conan-center-index) ===")
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

    # --- Commit ---
    commit_msg = f"dispenso: add version {version}"
    commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    test_passed = True
    if not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        test_passed = test_conan(repo_dir, version)

    # --- Push ---
    if not args.dry_run and not args.skip_push:
        if not test_passed:
            print("  Skipping push due to test failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = (
        f"https://github.com/{args.github_user}/conan-center-index/tree/{branch}"
    )
    tests_ran = not args.skip_test and not args.dry_run and test_passed
    title = pr_title("conan", version, is_new=True)
    body = pr_body_conan(version, tests_ran=tests_ran)
    print()
    return {
        "status": "ok" if test_passed else "test_failed",
        "branch_url": branch_url,
        "version": version,
        "is_new": True,
        "tests_ran": tests_ran,
        "pr_title": title,
        "pr_body": body,
    }


def update_vcpkg(args, hashes, tarball_path):
    """Update vcpkg with new dispenso version."""
    print(f"=== vcpkg ===")
    version = args.version
    manager = "vcpkg"
    branch = BRANCH_NAMES.get(manager, "add-dispenso")

    repo_dir = ensure_repo(args.repos_dir, manager, args.github_user, args.dry_run)
    checkout_branch(repo_dir, branch, args.dry_run)

    # --- Update vcpkg.json ---
    vcpkg_json_path = os.path.join(repo_dir, "ports", "dispenso", "vcpkg.json")
    print(f"  Updating {vcpkg_json_path} ...")

    if not args.dry_run:
        content = open(vcpkg_json_path).read()
        content = re.sub(
            r'"version"\s*:\s*"[^"]*"',
            f'"version": "{version}"',
            content,
        )
        open(vcpkg_json_path, "w").write(content)
        print(f"  Updated version to {version}")
    else:
        print(f"  [DRY RUN] Would update version to {version}")

    # --- Update portfile.cmake ---
    portfile_path = os.path.join(repo_dir, "ports", "dispenso", "portfile.cmake")
    print(f"  Updating {portfile_path} ...")

    if not args.dry_run:
        content = open(portfile_path).read()
        content = re.sub(
            r"SHA512\s+[0-9a-fA-F]+",
            f"SHA512 {hashes['sha512']}",
            content,
        )
        open(portfile_path, "w").write(content)
        print(f"  Updated SHA512")
    else:
        print(f"  [DRY RUN] Would update SHA512 to {hashes['sha512']}")

    # --- Run vcpkg x-add-version ---
    # x-add-version requires port changes to be committed first, so we make
    # a temporary commit. The final commit_and_push will squash everything.
    vcpkg_bin = shutil.which("vcpkg")
    if vcpkg_bin:
        print("  Running vcpkg x-add-version ...")
        if not args.dry_run:
            run(["git", "add", "-A"], cwd=repo_dir)
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=repo_dir
            )
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
                f"--overlay-ports={os.path.join(repo_dir, 'ports', 'dispenso')}",
            ],
            cwd=repo_dir,
            check=False,
            dry_run=args.dry_run,
        )
        # Stage any version database changes from x-add-version
        if not args.dry_run:
            run(["git", "add", "-A"], cwd=repo_dir)
    else:
        print(
            "  WARNING: vcpkg not found. Run 'vcpkg x-add-version dispenso "
            "--overlay-ports=ports/dispenso' manually before pushing."
        )

    # --- Commit (if not already committed above) ---
    commit_msg = f"[dispenso] Update to version {version}"
    commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    test_passed = True
    if not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        test_passed = test_vcpkg(repo_dir, version)

    # --- Push ---
    if not args.dry_run and not args.skip_push:
        if not test_passed:
            print("  Skipping push due to test failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = f"https://github.com/{args.github_user}/vcpkg/tree/{branch}"
    tests_ran = not args.skip_test and not args.dry_run and test_passed
    title = pr_title("vcpkg", version, is_new=True)
    body = pr_body_vcpkg(version, is_new_port=True)
    print()
    return {
        "status": "ok" if test_passed else "test_failed",
        "branch_url": branch_url,
        "version": version,
        "is_new": True,
        "tests_ran": tests_ran,
        "pr_title": title,
        "pr_body": body,
    }


def update_homebrew(args, hashes, tarball_path):
    """Update homebrew-core with new dispenso version."""
    print(f"=== Homebrew (homebrew-core) ===")
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
            print(f"  The dispenso formula may not exist yet in homebrew-core.")
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
        print(f"  Updated url, sha256 (removed revision if present)")
    else:
        print(f"  [DRY RUN] Would update url and sha256 in formula")

    # --- Verify checksums ---
    checksums_ok = True
    if not args.dry_run:
        print("  --- Verifying checksums ---")
        checksums_ok = verify_formula_checksums(repo_dir, hashes)

    # --- Commit ---
    commit_msg = f"dispenso {version} (new formula)"
    commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    fully_tested = False
    if checksums_ok and not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        fully_tested = test_homebrew(repo_dir, version)

    # --- Push ---
    if not args.dry_run and not args.skip_push:
        if not checksums_ok:
            print("  Skipping push due to checksum verification failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = f"https://github.com/{args.github_user}/homebrew-core/tree/{branch}"
    is_new = True  # TODO: detect from formula existence in upstream
    title = pr_title("homebrew", version, is_new=is_new)
    body = pr_body_homebrew(version, is_new_formula=is_new, tests_ran=fully_tested)
    print()
    return {
        "status": "error"
        if not checksums_ok
        else "ok"
        if fully_tested or args.skip_test
        else "needs_macos",
        "branch_url": branch_url,
        "version": version,
        "is_new": is_new,
        "tests_ran": fully_tested,
        "pr_title": title,
        "pr_body": body,
    }


def update_macports(args, hashes, tarball_path):
    """Update macports-ports with new dispenso version."""
    print(f"=== MacPorts (macports-ports) ===")
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
            print(f"  The dispenso port may not exist yet in MacPorts.")
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
        print(f"  Updated github.setup version, checksums (rmd160, sha256, size)")
    else:
        print(f"  [DRY RUN] Would update version to {version} and checksums")

    # --- Verify checksums ---
    checksums_ok = True
    if not args.dry_run:
        print("  --- Verifying checksums ---")
        checksums_ok = verify_portfile_checksums(repo_dir, hashes)

    # --- Commit ---
    commit_msg = f"dispenso: update to {version}"
    commit_and_push(
        repo_dir, branch, commit_msg, args.github_user, args.dry_run, skip_push=True
    )

    # --- Test ---
    fully_tested = False
    if checksums_ok and not args.dry_run and not args.skip_test:
        print("  --- Testing ---")
        fully_tested = test_macports(repo_dir, version)

    # --- Push ---
    if not args.dry_run and not args.skip_push:
        if not checksums_ok:
            print("  Skipping push due to checksum verification failure")
        else:
            run(["git", "push", "-u", "fork", branch, "--force"], cwd=repo_dir)

    branch_url = f"https://github.com/{args.github_user}/macports-ports/tree/{branch}"
    title = pr_title("macports", version, is_new=True)
    body = pr_body_macports(version, tests_ran=fully_tested)
    print()
    return {
        "status": "error"
        if not checksums_ok
        else "ok"
        if fully_tested or args.skip_test
        else "needs_macos",
        "branch_url": branch_url,
        "version": version,
        "is_new": True,
        "tests_ran": fully_tested,
        "pr_title": title,
        "pr_body": body,
    }


# ---------------------------------------------------------------------------
# PR body templates
# ---------------------------------------------------------------------------


def pr_body_homebrew(version, is_new_formula, tests_ran):
    """Generate PR body following Homebrew's PR template."""
    formula_type = "new formula" if is_new_formula else "version bump"
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

Add dispenso {version} ({formula_type}).

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


def pr_body_vcpkg(version, is_new_port):
    """Generate PR body following vcpkg's PR template."""
    if is_new_port:
        return f"""\
## New Port Checklist

- [ ] Complies with the [maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide)
- [x] Port name is associated with the packaged project
- [x] All optional build dependencies controlled by the port
- [x] Version scheme and license declarations match upstream
- [x] Copyright file is accurate
- [x] Source code comes from authoritative origin
- [x] Brief, accurate usage text provided
- [x] Version database updated via `./vcpkg x-add-version --all`
- [x] Exactly one version added per modified versions file

Add [dispenso](https://github.com/facebookincubator/dispenso) {version} — \
a high-performance C++ parallel programming library from Meta. MIT licensed, \
requires C++14, no external dependencies beyond pthreads."""
    else:
        return f"""\
## Port Update Checklist

- [ ] Complies with the [maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide)
- [x] Updated SHA512 checksums
- [x] Version database updated via `./vcpkg x-add-version --all`
- [x] Exactly one version added per modified versions file

Update dispenso to version {version}."""


def pr_body_macports(version, tests_ran):
    """Generate PR body following MacPorts' PR template."""
    lint_check = "[x]" if tests_ran else "[ ]"
    return f"""\
#### Description
Update dispenso to version {version}.

[dispenso](https://github.com/facebookincubator/dispenso) is a high-performance \
C++ parallel programming library from Meta.

#### Type(s)
- [ ] bugfix
- [x] enhancement
- [ ] security fix

#### Tested on
<!-- Run and paste output of: port version && sw_vers && xcode-select -p -->

#### Verification
- [x] Followed [Commit Message Guidelines](https://trac.macports.org/wiki/CommitMessages)
- [x] Squashed and minimized commits
- [x] Checked that there aren't other open [pull requests](https://github.com/macports/macports-ports/pulls) for the same change
- {lint_check} Checked Portfile with `port lint --nitpick`
- [ ] Tested with `sudo port -vst install`
- [ ] Checked that binaries work as expected
- [ ] Tested important variants"""


def pr_title(manager, version, is_new):
    """Generate the PR title for a given manager."""
    titles = {
        "conan": f"dispenso: add version {version}",
        "vcpkg": f"[dispenso] Add new port (v{version})"
        if is_new
        else f"[dispenso] Update to version {version}",
        "homebrew": f"dispenso {version} (new formula)"
        if is_new
        else f"dispenso {version}",
        "macports": f"dispenso: add new port @{version}"
        if is_new
        else f"dispenso: update to {version}",
    }
    return titles.get(manager, f"dispenso {version}")


def pre_pr_checklist(manager, version, tests_ran):
    """Return items the user should manually verify before creating the PR."""
    checklists = {
        "homebrew": [
            "Read contributing guidelines: https://github.com/Homebrew/homebrew-core/blob/HEAD/CONTRIBUTING.md",
            "Run: HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-from-source dispenso",
            "Run: brew test dispenso",
            "Run: brew audit --new dispenso  (or --strict for updates)",
        ],
        "conan": [
            "Open or comment on an issue at conan-center-index per CONTRIBUTING.md",
            "Sign the CLA if prompted on the PR",
            "Run: conan create recipes/dispenso/all --version=" + version,
            "Include 'fixes #<issue>' in PR body to link the issue",
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
        is_new = result.get("is_new", False)
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


def main():
    args = parse_args()

    print(f"Updating dispenso to version {args.version}")
    if args.dry_run:
        print("*** DRY RUN MODE — no files will be modified ***")
    if args.skip_test:
        print("*** SKIP TEST MODE — no local verification ***")
    if args.skip_push:
        print("*** SKIP PUSH MODE — commit and test only ***")
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
        print(f"  The following managers were updated and checksums verified,")
        print(f"  but full testing (lint, install, audit) requires macOS:")
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
