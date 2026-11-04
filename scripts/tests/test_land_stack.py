# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for scripts/land_stack.py against a real local repository and a fake GitHub.

    python3 -m unittest discover -s scripts/tests
"""

import datetime
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import land_stack  # noqa: E402

NOW = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.timezone.utc)
REQUIRED = ["Build", "CLA"]
GOVERNANCE = """# Governance

| Name | GitHub | Role | Affiliation |
|---|---|---|---|
| Alice | @Alice | Maintainer | Independent |
| Bob | @bob | Committer | Elsewhere |
"""
RELEASED_BODY = "- [x] Alters or removes something present in the most recent tagged release\n"


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Repo:
    """A bare origin, a clone to build pull requests in, and a clone for the lander."""

    def __init__(self, tmp):
        self.origin = os.path.join(tmp, "origin.git")
        self.work = os.path.join(tmp, "work")
        self.lander = os.path.join(tmp, "lander")
        git(tmp, "init", "-q", "--bare", "-b", "main", self.origin)
        git(tmp, "clone", "-q", self.origin, self.work)
        self.main = self.commit(None, {"GOVERNANCE.md": GOVERNANCE, "CHANGELOG.md": "Unreleased\n"}, "Initial")
        self.push(self.main, "main")
        git(tmp, "clone", "-q", self.origin, self.lander)

    def commit(self, parent, files, message, author="Alice <alice@example.com>"):
        # Commits are pushed under refs/scratch so the origin, and so the fake GitHub, knows them.
        git(self.work, "fetch", "-q", "origin")
        if parent:
            git(self.work, "checkout", "-q", "--detach", parent)
        for path, text in files.items():
            with open(os.path.join(self.work, path), "w") as f:
                f.write(text)
            git(self.work, "add", path)
        git(self.work, "commit", "-q", "--author", author, "-m", message)
        sha = git(self.work, "rev-parse", "HEAD")
        git(self.work, "push", "-q", "origin", f"{sha}:refs/scratch/{sha}")
        return sha

    def read(self, rev, path):
        return git(self.work, "show", f"{rev}:{path}")

    def push(self, sha, branch):
        git(self.work, "push", "-q", "-f", "origin", f"{sha}:refs/heads/{branch}")

    def origin_rev(self, ref):
        return git(self.origin, "rev-parse", ref)

    def origin_branches(self):
        return set(git(self.origin, "for-each-ref", "--format=%(refname:short)", "refs/heads").split())


class FakeGitHub:
    repo = "o/r"

    def __init__(self, r):
        self.r = r
        self.prs, self.checks, self.check_starts = {}, {}, {}
        self.reviews_, self.pushes = {}, {}
        self.comments, self.deleted = [], []

    def add(self, n, head_ref, base_ref, sha, body="", author="alice", created=NOW - datetime.timedelta(days=5)):
        self.r.push(sha, head_ref)
        self.prs[n] = {
            "number": n,
            "state": "open",
            "title": f"Change {n}",
            "body": body,
            "draft": False,
            "labels": [],
            "user": {"login": author},
            "created_at": iso(created),
            "head": {"ref": head_ref, "sha": sha, "repo": {"full_name": self.repo}},
            "base": {"ref": base_ref},
        }
        self.checks[sha] = {name: "success" for name in REQUIRED}
        self.check_starts[sha] = created

    def default_branch(self):
        return "main"

    def pull(self, n):
        return self.prs[n]

    def open_pulls(self):
        return [p for p in self.prs.values() if p["state"] == "open"]

    def required_checks(self, branch):
        return REQUIRED

    def check_results(self, sha):
        return self.checks.get(sha, {})

    def check_start(self, sha):
        return self.check_starts.get(sha)

    def reviews(self, n):
        return self.reviews_.get(n, [])

    def force_pushes(self, n):
        return self.pushes.get(n, [])

    def tree(self, sha):
        return git(self.r.origin, "rev-parse", f"{sha}^{{tree}}")

    def comment(self, n, body):
        self.comments.append((n, body))

    def close(self, n):
        self.prs[n]["state"] = "closed"

    def delete_branch(self, name):
        self.deleted.append(name)
        git(self.r.work, "push", "-q", "origin", f":refs/heads/{name}")


class LandStackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = dict(os.environ)
        os.environ.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_COMMITTER_NAME="Tester",
            GIT_COMMITTER_EMAIL="tester@example.com",
        )
        self.r = Repo(self._tmp.name)
        self.gh = FakeGitHub(self.r)
        self.git = land_stack.Git(self.r.lander, ("Lander", "lander@example.com"))
        self.log = []

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    def stack(self):
        """Three stacked pull requests that each add a CHANGELOG line below the last.

        GitHub's squash merge conflicts on every one of them after the first lands.
        """
        r = self.r
        a = r.commit(r.main, {"a.txt": "a\n", "CHANGELOG.md": "Unreleased\n* one\n"}, "Add a\n\nBody of a.")
        b = r.commit(a, {"CHANGELOG.md": "Unreleased\n* one\n* two\n"}, "Note two")
        c1 = r.commit(
            b,
            {"c.txt": "c\n"},
            "Add c\n\nFirst part.\n\nCo-Authored-By: Helper <helper@example.com>",
            author="Bob <bob@example.com>",
        )
        c2 = r.commit(
            c1,
            {"CHANGELOG.md": "Unreleased\n* one\n* two\n* three\n"},
            "Note three\n\nSecond part.\n\nCo-Authored-By: Helper <helper@example.com>",
        )
        self.gh.add(1, "s/1", "main", a)
        self.gh.add(2, "s/2", "s/1", b)
        self.gh.add(3, "s/3", "s/2", c2, author="bob")
        return a, b, c2

    def land(self, top, **kw):
        return land_stack.land(self.gh, self.git, top, now=NOW, out=self.log.append, **kw)

    def assert_nothing_landed(self):
        self.assertEqual(self.r.origin_rev("main"), self.r.main)
        self.assertTrue(all(p["state"] == "open" for p in self.gh.prs.values()))
        self.assertEqual(self.gh.comments, [])

    def test_lands_a_chain_in_one_push_with_tested_trees(self):
        heads = self.stack()
        landed = self.land(3, run_url="https://example.com/run")
        main = self.r.origin_rev("main")
        commits = git(self.r.origin, "rev-list", "--reverse", f"{self.r.main}..{main}").split()
        self.assertEqual(commits, [c for _, c in landed])
        for commit, head in zip(commits, heads):
            self.assertEqual(self.gh.tree(commit), self.gh.tree(head))
        self.assertEqual(git(self.r.origin, "log", "-1", "--format=%s", commits[0]), "Change 1 (#1)")
        self.assertEqual(git(self.r.origin, "log", "-1", "--format=%b", commits[0]).strip(), "Body of a.")
        third = git(self.r.origin, "log", "-1", "--format=%B", commits[2])
        self.assertIn("* Add c\n\nFirst part.\n\n* Note three\n\nSecond part.", third)
        self.assertEqual(third.count("Co-authored-by: Helper <helper@example.com>"), 1)
        self.assertEqual(git(self.r.origin, "log", "-1", "--format=%an", commits[2]), "Bob")
        self.assertEqual(git(self.r.origin, "log", "-1", "--format=%cn", commits[2]), "Lander")
        self.assertEqual({n for n, p in self.gh.prs.items() if p["state"] == "closed"}, {1, 2, 3})
        self.assertIn((2, f"Landed on `main` as {commits[1]} by [Land stack](https://example.com/run)."), self.gh.comments)
        self.assertEqual(self.r.origin_branches(), {"main"})

    def test_lands_the_lower_part_and_keeps_the_branch_the_rest_is_based_on(self):
        self.stack()
        self.land(2)
        self.assertEqual(self.gh.prs[3]["state"], "open")
        self.assertEqual(self.r.origin_branches(), {"main", "s/2", "s/3"})
        self.assertTrue(any("kept s/2" in line for line in self.log))
        # The rest lands later on top: its base branch is gone from the open chain but has main's tree.
        self.land(3)
        self.assertEqual(self.r.origin_branches(), {"main"})
        self.assertEqual(self.gh.tree(self.r.origin_rev("main")), self.gh.tree(self.gh.prs[3]["head"]["sha"]))

    def test_refuses_when_main_has_moved_under_the_chain(self):
        self.stack()
        other = self.r.commit(self.r.main, {"other.txt": "x\n"}, "Unrelated")
        self.r.push(other, "main")
        self.r.main = other
        with self.assertRaisesRegex(land_stack.LandError, "#1 does not contain"):
            self.land(3)
        self.assert_nothing_landed()

    def test_refuses_a_leftover_base_whose_tree_is_not_main(self):
        self.stack()
        self.land(2)
        other = self.r.commit(self.r.origin_rev("main"), {"other.txt": "x\n"}, "Unrelated")
        self.r.push(other, "main")
        with self.assertRaisesRegex(land_stack.LandError, "whose tree is not main's"):
            self.land(3)
        self.assertEqual(self.gh.prs[3]["state"], "open")

    def test_refuses_a_pull_request_that_does_not_contain_its_parent(self):
        a, _, _ = self.stack()
        newer = self.r.commit(a, {"a.txt": "a2\n"}, "Amend a")
        self.r.push(newer, "s/1")
        self.gh.prs[1]["head"]["sha"] = newer
        self.gh.checks[newer] = {name: "success" for name in REQUIRED}
        with self.assertRaisesRegex(land_stack.LandError, "#2 does not contain"):
            self.land(3)
        self.assert_nothing_landed()

    def test_refuses_failed_and_missing_checks(self):
        _, b, c = self.stack()
        self.gh.checks[b]["CLA"] = "failure"
        del self.gh.checks[c]["Build"]
        with self.assertRaisesRegex(land_stack.LandError, "#2, #3 cannot land"):
            self.land(3)
        self.assertTrue(any("CLA=failure" in line for line in self.log))
        self.assertTrue(any("Build=missing" in line for line in self.log))
        self.assert_nothing_landed()

    def test_refuses_drafts_labels_and_requests_for_changes(self):
        self.stack()
        self.gh.prs[1]["draft"] = True
        self.gh.prs[2]["labels"] = [{"name": "do-not-land"}]
        self.gh.reviews_[3] = [
            {
                "user": {"login": "carol"},
                "state": "CHANGES_REQUESTED",
                "submitted_at": iso(NOW),
                "commit_id": self.gh.prs[3]["head"]["sha"],
                "author_association": "MEMBER",
            }
        ]
        with self.assertRaisesRegex(land_stack.LandError, "#1, #2, #3 cannot land"):
            self.land(3)
        self.assert_nothing_landed()

    def test_notice_waits_24h_and_72h_for_released_surface(self):
        r = self.r
        a = r.commit(r.main, {"a.txt": "a\n"}, "Add a")
        self.gh.add(1, "s/1", "main", a, created=NOW - datetime.timedelta(hours=23))
        with self.assertRaisesRegex(land_stack.LandError, "#1 cannot land"):
            self.land(1)
        self.assertTrue(any("notice wait of 24h" in line for line in self.log))
        self.gh.prs[1]["body"] = RELEASED_BODY
        self.gh.check_starts[a] = NOW - datetime.timedelta(hours=30)
        self.gh.prs[1]["created_at"] = iso(NOW - datetime.timedelta(hours=30))
        with self.assertRaisesRegex(land_stack.LandError, "#1 cannot land"):
            self.land(1)
        self.assertTrue(any("notice wait of 72h" in line for line in self.log))
        self.gh.check_starts[a] = NOW - datetime.timedelta(hours=73)
        self.gh.prs[1]["created_at"] = iso(NOW - datetime.timedelta(hours=73))
        self.land(1)
        self.assertEqual(self.gh.tree(r.origin_rev("main")), self.gh.tree(a))

    def test_an_approval_on_the_current_tree_replaces_the_wait(self):
        r = self.r
        a = r.commit(r.main, {"a.txt": "a\n"}, "Add a")
        old = r.commit(r.main, {"a.txt": "draft\n"}, "Draft a")
        self.gh.add(1, "s/1", "main", a, created=NOW - datetime.timedelta(hours=1))
        review = {"user": {"login": "bob"}, "state": "APPROVED", "submitted_at": iso(NOW), "author_association": "COLLABORATOR"}
        self.gh.reviews_[1] = [dict(review, commit_id=old)]
        with self.assertRaisesRegex(land_stack.LandError, "#1 cannot land"):
            self.land(1)
        self.gh.reviews_[1] = [dict(review, commit_id=a)]
        self.land(1)
        self.assertEqual(self.gh.tree(r.origin_rev("main")), self.gh.tree(a))

    def test_released_surface_changes_need_a_maintainer(self):
        r = self.r
        a = r.commit(r.main, {"a.txt": "a\n"}, "Add a", author="Bob <bob@example.com>")
        self.gh.add(1, "s/1", "main", a, body=RELEASED_BODY, author="bob")
        with self.assertRaisesRegex(land_stack.LandError, "#1 cannot land"):
            self.land(1)
        self.assertTrue(any("no maintainer as author or approver" in line for line in self.log))
        self.gh.reviews_[1] = [
            {"user": {"login": "alice"}, "state": "APPROVED", "submitted_at": iso(NOW), "commit_id": a, "author_association": "MEMBER"}
        ]
        self.land(1)

    def test_tree_identical_force_push_keeps_the_notice_clock(self):
        r = self.r
        a = r.commit(r.main, {"a.txt": "a\n"}, "Add a")
        rebased = r.commit(r.main, {"a.txt": "a\n"}, "Add a, rebased")
        self.gh.add(1, "s/1", "main", rebased, created=NOW - datetime.timedelta(days=3))
        self.gh.check_starts[rebased] = NOW - datetime.timedelta(hours=1)
        self.gh.pushes[1] = [(NOW - datetime.timedelta(hours=30), None, a), (NOW - datetime.timedelta(hours=1), a, rebased)]
        self.land(1, dry_run=True)
        changed = r.commit(r.main, {"a.txt": "other\n"}, "Add a, differently")
        self.gh.pushes[1] = [(NOW - datetime.timedelta(hours=30), None, changed), (NOW - datetime.timedelta(hours=1), changed, rebased)]
        with self.assertRaisesRegex(land_stack.LandError, "#1 cannot land"):
            self.land(1, dry_run=True)

    def test_tree_identical_force_push_keeps_the_original_plain_push_clock(self):
        r = self.r
        a = r.commit(r.main, {"a.txt": "a\n"}, "Add a")
        rebased = r.commit(r.main, {"a.txt": "a\n"}, "Add a, rebased")
        self.gh.add(1, "s/1", "main", rebased, created=NOW - datetime.timedelta(days=3))
        self.gh.check_starts[a] = NOW - datetime.timedelta(hours=30)
        self.gh.check_starts[rebased] = NOW - datetime.timedelta(hours=1)
        # The only force push replaced a plain push of the same tree.
        self.gh.pushes[1] = [(NOW - datetime.timedelta(hours=1), a, rebased)]
        self.land(1, dry_run=True)
        changed = r.commit(r.main, {"a.txt": "other\n"}, "Add a, differently")
        self.gh.check_starts[changed] = NOW - datetime.timedelta(hours=30)
        self.gh.pushes[1] = [(NOW - datetime.timedelta(hours=1), changed, rebased)]
        with self.assertRaisesRegex(land_stack.LandError, "#1 cannot land"):
            self.land(1, dry_run=True)

    def test_dry_run_pushes_nothing(self):
        self.stack()
        landed = self.land(3, dry_run=True)
        self.assertEqual(len(landed), 3)
        self.assert_nothing_landed()
        self.assertEqual(self.r.origin_branches(), {"main", "s/1", "s/2", "s/3"})

    def test_push_is_rejected_if_main_moves_during_the_landing(self):
        self.stack()
        r, real = self.r, self.git.__class__.__call__

        def racing(git_self, *args, **kw):
            if args and args[0] == "push":
                other = r.commit(r.main, {"other.txt": "x\n"}, "Raced in")
                r.push(other, "main")
            return real(git_self, *args, **kw)

        self.git.__class__.__call__ = racing
        try:
            with self.assertRaisesRegex(land_stack.LandError, "git push"):
                self.land(3)
        finally:
            self.git.__class__.__call__ = real
        self.assertTrue(all(p["state"] == "open" for p in self.gh.prs.values()))
        self.assertEqual(git(r.origin, "log", "-1", "--format=%s", "main"), "Raced in")

    def test_reads_maintainers_from_the_governance_roster(self):
        self.assertEqual(land_stack.governance_maintainers(GOVERNANCE), {"alice"})


if __name__ == "__main__":
    unittest.main()
