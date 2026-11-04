#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Land a chain of stacked pull requests on the default branch in one push.

GitHub can only merge stacked pull requests one at a time. Under squash-only
merging that is slow and fragile: each squash leaves the next pull request's
branch with no history in common with the default branch, so a pull request
that touches lines a lower one changed (CHANGELOG.md, nearly always) conflicts
with its own parent's squash, and has to be rebased and re-tested before it can
merge.

This script lands the chain the way it was tested instead. Each pull request's
CI ran on its head, and that head's tree is exactly what the default branch
should hold once it lands. So for each pull request, bottom first, the script
writes one commit whose tree is that head's tree and whose parent is the commit
before it, then pushes them all in a single fast-forward. No merge is computed,
so nothing can conflict, and nothing is re-run. If the default branch moved in
the meantime the push is rejected and nothing lands.

Every pull request in the chain must satisfy GOVERNANCE.md's merging rules:
open and not a draft, every required check passed on its head, no outstanding
"request changes" review and no `do-not-land` label, and either approved by a
committer other than its author or past its notice wait. The wait is 72 hours
for a change that alters the stable surface of the latest tagged release (which
also needs a maintainer as author or approver), and 24 hours otherwise.

    python3 scripts/land_stack.py --repo OWNER/NAME --top 123 [--dry-run]

The chain is found by following each pull request's base branch down to the
default branch. A base branch left behind by an earlier landing also ends the
chain, provided its tree is the default branch's tree: that is the branch the
bottom pull request was tested against.

After pushing, each landed pull request is closed with a comment naming its
commit, and its branch is deleted, except a branch that an open pull request is
still based on; that one goes when its child lands.

Run from a clone with full history whose `origin` is the repository. GitHub
access goes through the `gh` CLI, authenticated by GH_TOKEN.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ALTERS_RELEASED = re.compile(
    r"^\s*-\s*\[[xX]\]\s*Alters or removes something present in the most recent tagged release",
    re.M,
)
DO_NOT_LAND = "do-not-land"
COMMITTER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
PASSED = {"success", "neutral", "skipped"}
CO_AUTHOR = re.compile(r"^co-authored-by:\s*(.+)$", re.I)
WAIT_RELEASED = datetime.timedelta(hours=72)
WAIT_DEFAULT = datetime.timedelta(hours=24)


class LandError(Exception):
    pass


def _parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


class GitHub:
    """The few REST calls this script needs, through `gh api`."""

    def __init__(self, repo):
        self.repo = repo

    def _api(self, path, method="GET", body=None):
        cmd = ["gh", "api", "-X", method, f"repos/{self.repo}" + (f"/{path}" if path else "")]
        if body is not None:
            cmd += ["--input", "-"]
        out = subprocess.run(
            cmd,
            input=json.dumps(body) if body is not None else None,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            raise LandError(f"gh api {method} {path} failed: {out.stderr.strip()}")
        return json.loads(out.stdout) if out.stdout.strip() else None

    def _pages(self, path, key=None):
        sep = "&" if "?" in path else "?"
        items, page = [], 1
        while True:
            data = self._api(f"{path}{sep}per_page=100&page={page}")
            chunk = data[key] if key else data
            items += chunk
            if len(chunk) < 100:
                return items
            page += 1

    def default_branch(self):
        return self._api("")["default_branch"]

    def pull(self, number):
        return self._api(f"pulls/{number}")

    def open_pulls(self):
        return self._pages("pulls?state=open")

    def required_checks(self, branch):
        names = []
        for rule in self._api(f"rules/branches/{branch}"):
            if rule["type"] == "required_status_checks":
                names += [c["context"] for c in rule["parameters"]["required_status_checks"]]
        return names

    def check_results(self, sha):
        """Latest result per check name: check runs, then commit statuses."""
        results = {}
        for run in sorted(self._pages(f"commits/{sha}/check-runs", "check_runs"), key=lambda r: r["id"]):
            results[run["name"]] = run["conclusion"] if run["status"] == "completed" else run["status"]
        for status in self._api(f"commits/{sha}/status")["statuses"]:
            results.setdefault(status["context"], status["state"])
        return results

    def check_start(self, sha):
        """When CI first started on this commit, which is no earlier than its push."""
        times = [r["started_at"] for r in self._pages(f"commits/{sha}/check-runs", "check_runs")]
        return min(_parse_time(t) for t in times if t) if any(times) else None

    def reviews(self, number):
        return self._pages(f"pulls/{number}/reviews")

    def force_pushes(self, number):
        """(time, sha before, sha after) of each force push, oldest first.

        GraphQL, because the REST timeline does not say which commit a force push replaced.
        """
        owner, name = self.repo.split("/")
        query = (
            "query($owner: String!, $name: String!, $number: Int!, $cursor: String) {"
            " repository(owner: $owner, name: $name) { pullRequest(number: $number) {"
            " timelineItems(itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT], first: 100, after: $cursor) {"
            " pageInfo { hasNextPage endCursor }"
            " nodes { ... on HeadRefForcePushedEvent { createdAt beforeCommit { oid } afterCommit { oid } } } } } } }"
        )
        pushes, cursor = [], None
        while True:
            cmd = ["gh", "api", "graphql", "-f", f"query={query}", "-F", f"owner={owner}", "-F", f"name={name}",
                   "-F", f"number={number}"]
            if cursor:
                cmd += ["-F", f"cursor={cursor}"]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                raise LandError(f"gh api graphql (force pushes of #{number}) failed: {out.stderr.strip()}")
            items = json.loads(out.stdout)["data"]["repository"]["pullRequest"]["timelineItems"]
            for e in items["nodes"]:
                if e.get("afterCommit"):
                    before = (e.get("beforeCommit") or {}).get("oid")
                    pushes.append((_parse_time(e["createdAt"]), before, e["afterCommit"]["oid"]))
            if not items["pageInfo"]["hasNextPage"]:
                return pushes
            cursor = items["pageInfo"]["endCursor"]

    def tree(self, sha):
        return self._api(f"commits/{sha}")["commit"]["tree"]["sha"]

    def comment(self, number, body):
        self._api(f"issues/{number}/comments", "POST", {"body": body})

    def close(self, number):
        self._api(f"pulls/{number}", "PATCH", {"state": "closed"})

    def delete_branch(self, name):
        self._api(f"git/refs/heads/{name}", "DELETE")


class Git:
    def __init__(self, cwd, committer):
        self.cwd = cwd
        self.committer = committer

    def __call__(self, *args, env=None, input=None):
        out = subprocess.run(
            ["git", *args],
            cwd=self.cwd,
            env={**os.environ, **(env or {})},
            input=input,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            raise LandError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
        return out.stdout.strip()

    def is_ancestor(self, a, b):
        return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=self.cwd).returncode == 0

    def tree(self, rev):
        return self("rev-parse", f"{rev}^{{tree}}")


def governance_maintainers(text):
    """GitHub logins of the active maintainers in GOVERNANCE.md's roster table."""
    logins = set()
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and "Maintainer" in cells:
            for cell in cells:
                logins.update(login.lower() for login in re.findall(r"^@([A-Za-z0-9-]+)$", cell))
    return logins


def find_chain(gh, top, default):
    """The open pull requests from the bottom of the chain up to `top`."""
    by_head = {}
    for pr in gh.open_pulls():
        if pr["head"]["repo"] and pr["head"]["repo"]["full_name"] == gh.repo:
            by_head[pr["head"]["ref"]] = pr
    chain = [gh.pull(top)]
    while chain[-1]["base"]["ref"] != default and chain[-1]["base"]["ref"] in by_head:
        chain.append(by_head[chain[-1]["base"]["ref"]])
        if len(chain) > 200:
            raise LandError("base branches form a cycle")
    chain.reverse()
    for pr in chain:
        if pr["state"] != "open":
            raise LandError(f"#{pr['number']} is {pr['state']}")
        if not pr["head"]["repo"] or pr["head"]["repo"]["full_name"] != gh.repo:
            raise LandError(f"#{pr['number']} comes from a fork; stacks must live in {gh.repo}")
    return chain


def notice_clock(gh, pr, head_tree):
    """When the head's current tree was first pushed.

    A force push that keeps the tree (a rebase onto a landed parent) is not a
    material change, so the clock runs from the earliest push in the latest run
    of pushes that all carry this tree. When every force push kept it, the tree
    predates them all: the clock then runs from the replaced commit's push.
    """
    sha = pr["head"]["sha"]
    pushes = gh.force_pushes(pr["number"])
    start = None
    for time, before, pushed in reversed(pushes):
        if gh.tree(pushed) != head_tree:
            break
        start = time
    else:
        # The first force push replaced a plain push. If that commit carried the same tree,
        # its CI start dates the push conservatively (CI starts no earlier than the push).
        before = pushes[0][1] if pushes else None
        if before and gh.tree(before) == head_tree:
            older = gh.check_start(before)
            if older and older < start:
                start = older
    if start is None or not any(pushed == sha for _, _, pushed in pushes):
        # The head arrived by a plain push, which the timeline does not date. CI starts on
        # push, so its earliest start is no earlier than the push: a conservative stand-in.
        ci = gh.check_start(sha)
        start = min(t for t in (start, ci) if t) if (start or ci) else None
    if start is None:
        start = _parse_time(pr["created_at"])
    return start


def check_pull(gh, git, pr, required, maintainers, now):
    """Problems that stop this pull request landing; empty if it may land."""
    n, sha, author = pr["number"], pr["head"]["sha"], pr["user"]["login"]
    problems = []
    if pr.get("draft"):
        problems.append("is a draft")
    if any(label["name"] == DO_NOT_LAND for label in pr.get("labels", [])):
        problems.append(f"has the {DO_NOT_LAND} label")

    results = gh.check_results(sha)
    failed = [f"{name}={results.get(name, 'missing')}" for name in required if results.get(name) not in PASSED]
    if failed:
        problems.append("required checks not passed on its head: " + ", ".join(failed))

    head_tree = git.tree(sha)
    latest = {}
    for review in sorted(gh.reviews(n), key=lambda r: r["submitted_at"] or ""):
        if review["state"] in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            latest[review["user"]["login"]] = review
    if any(r["state"] == "CHANGES_REQUESTED" for r in latest.values()):
        problems.append("has an outstanding request for changes")
    approvers = {
        login
        for login, r in latest.items()
        if r["state"] == "APPROVED"
        and login != author
        and r.get("author_association") in COMMITTER_ASSOCIATIONS
        and gh.tree(r["commit_id"]) == head_tree
    }

    alters_released = bool(ALTERS_RELEASED.search(pr.get("body") or ""))
    if alters_released and not {login.lower() for login in {author} | approvers} & maintainers:
        problems.append("alters the latest release's stable surface but has no maintainer as author or approver")
    if not approvers:
        wait = WAIT_RELEASED if alters_released else WAIT_DEFAULT
        clock = notice_clock(gh, pr, head_tree)
        if now - clock < wait:
            left = wait - (now - clock)
            problems.append(
                f"notice wait of {wait.total_seconds() / 3600:.0f}h not over: started "
                f"{clock:%Y-%m-%d %H:%M}Z, {left.total_seconds() / 3600:.1f}h left"
            )
    return problems


def squash_message(git, pr, base):
    """Title "<PR title> (#N)", then the pull request's own commit messages."""
    commits = git("rev-list", "--reverse", f"{base}..{pr['head']['sha']}").split()
    parts, co_authors = [], []
    for c in commits:
        subject = git("log", "-1", "--format=%s", c)
        body = []
        for line in git("log", "-1", "--format=%b", c).splitlines():
            m = CO_AUTHOR.match(line.strip())
            if m:
                if m.group(1) not in co_authors:
                    co_authors.append(m.group(1))
            else:
                body.append(line)
        parts.append((subject, "\n".join(body).strip()))
    if len(parts) == 1:
        text = parts[0][1]
    else:
        text = "\n\n".join(f"* {s}\n\n{b}".rstrip() for s, b in parts)
    trailers = "\n".join(f"Co-authored-by: {c}" for c in co_authors)
    return f"{pr['title']} (#{pr['number']})\n\n" + "\n\n".join(p for p in (text.strip(), trailers) if p) + "\n"


def land(gh, git, top, dry_run=False, now=None, run_url=None, governance_text=None, out=print):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    default = gh.default_branch()
    git("fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*")
    main = git("rev-parse", f"origin/{default}")
    chain = find_chain(gh, top, default)

    # The chain must be internally consistent and sit on what the default branch holds now.
    bottom_base = chain[0]["base"]["ref"]
    base_sha = git("rev-parse", f"origin/{bottom_base}")
    if git.tree(base_sha) != git.tree(main):
        raise LandError(
            f"#{chain[0]['number']} is based on {bottom_base}, whose tree is not {default}'s. "
            f"Rebase it onto {default} and let CI run."
        )
    below = base_sha
    for pr in chain:
        sha = pr["head"]["sha"]
        if git("rev-parse", f"origin/{pr['head']['ref']}") != sha:
            raise LandError(f"#{pr['number']}'s branch has moved since GitHub last reported it; retry")
        if not git.is_ancestor(below, sha):
            raise LandError(
                f"#{pr['number']} does not contain the head of the branch below it, so CI tested a "
                "merge rather than its head. Rebase it and let CI run."
            )
        below = sha

    required = gh.required_checks(default)
    maintainers = governance_maintainers(governance_text or git("show", f"origin/{default}:GOVERNANCE.md"))
    blocked = {pr["number"]: check_pull(gh, git, pr, required, maintainers, now) for pr in chain}
    for pr in chain:
        status = "; ".join(blocked[pr["number"]]) or "ok"
        out(f"#{pr['number']} {pr['head']['ref']} {pr['head']['sha'][:9]}: {status}")
    if any(blocked.values()):
        raise LandError("nothing landed: " + ", ".join(f"#{n}" for n, p in blocked.items() if p) + " cannot land")

    # One commit per pull request, each with exactly the tree that pull request's CI tested.
    parent, landed, below = main, [], base_sha
    for pr in chain:
        sha = pr["head"]["sha"]
        author = git("log", "-1", "--format=%an%x00%ae", git("rev-list", "--reverse", f"{below}..{sha}").split()[0])
        name, email = author.split("\0")
        commit = git(
            "commit-tree",
            git.tree(sha),
            "-p",
            parent,
            "-F",
            "-",
            input=squash_message(git, pr, below),
            env={
                "GIT_AUTHOR_NAME": name,
                "GIT_AUTHOR_EMAIL": email,
                "GIT_COMMITTER_NAME": git.committer[0],
                "GIT_COMMITTER_EMAIL": git.committer[1],
            },
        )
        landed.append((pr, commit))
        parent, below = commit, sha
    for pr, commit in landed:
        out(f"  {commit[:9]} {git('log', '-1', '--format=%s', commit)}")
    if dry_run:
        out(f"dry run: would push {len(landed)} commit(s) to {default}")
        return landed

    # A plain push only fast-forwards: if the default branch moved, it is rejected and nothing lands.
    git("push", "--quiet", "origin", f"{parent}:refs/heads/{default}")
    out(f"pushed {len(landed)} commit(s) to {default}")

    landed_numbers = {pr["number"] for pr, _ in landed}
    still_based = {p["base"]["ref"] for p in gh.open_pulls() if p["number"] not in landed_numbers}
    where = f" by [Land stack]({run_url})" if run_url else ""
    for pr, commit in landed:
        gh.comment(pr["number"], f"Landed on `{default}` as {commit}{where}.")
        gh.close(pr["number"])
    stale = [pr["head"]["ref"] for pr, _ in landed]
    if bottom_base != default:
        stale.append(bottom_base)
    for branch in stale:
        if branch in still_based:
            out(f"kept {branch}: an open pull request is based on it")
        else:
            gh.delete_branch(branch)
    return landed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="OWNER/NAME")
    ap.add_argument("--top", type=int, required=True, help="the top pull request of the chain to land")
    ap.add_argument("--dry-run", action="store_true", help="check and print the plan without pushing")
    ap.add_argument("--run-url", help="link to this run, for the landing comments")
    ap.add_argument("--committer", default="dispenso-lander[bot] <noreply@github.com>")
    args = ap.parse_args()
    m = re.match(r"^(.*\S)\s*<([^>]+)>$", args.committer)
    if not m:
        sys.exit("--committer must look like 'Name <email>'")
    try:
        land(GitHub(args.repo), Git(os.getcwd(), (m.group(1), m.group(2))), args.top, args.dry_run, run_url=args.run_url)
    except LandError as e:
        sys.exit(f"land_stack: {e}")


if __name__ == "__main__":
    main()
