# Landing stacked pull requests

A stack is a chain of pull requests where each one's base branch is the branch
of the one below it, and the bottom one is based on `main`. Each pull request
is reviewed on its own, and each lands as its own squashed commit.

Land a stack, or its lower part, with the **Land stack** workflow (Actions →
Land stack → Run workflow), giving the top pull request to land. It runs as a
dry run by default: it checks every pull request in the chain and prints the
commits it would push. Run it again with the dry-run box cleared to land.

A single pull request based on `main` can still be merged with GitHub's merge
button.

## Why not GitHub's merge button

`main` takes only squash merges. Once the bottom of a stack is squashed onto
`main`, the next pull request's branch shares no history with `main` beyond
the commit the stack started from, so GitHub computes its squash as a
three-way merge against that old commit. Any pull request that edits lines a
lower one also changed (CHANGELOG.md, nearly always) then conflicts with its
own parent's squash, and has to be rebased and re-tested before it can merge.
Landing a stack that way is serial, one CI run per pull request.

## What the workflow does

`scripts/land_stack.py` follows base branches down from the top pull request
and checks each pull request in the chain:

- It is open, not a draft, has no `do-not-land` label and no outstanding
  request for changes.
- Every check the `main` ruleset requires passed on its head.
- Its head contains the head of the branch below it, so its CI tested its head
  itself rather than a merge. The bottom pull request's base must have the
  same tree as `main`.
- It meets GOVERNANCE.md's merging rules: approved by a committer other than
  its author on its current tree, or past its notice wait. The wait is 72 hours
  if the pull request's Stable surface section ticks "Alters or removes
  something present in the most recent tagged release", which also needs a
  maintainer as author or approver; otherwise 24 hours. The wait runs from when
  the head's current tree was first pushed, so a rebase that keeps the tree
  does not restart it.

If every pull request passes, it writes one commit per pull request, bottom
first, each with exactly its head's tree, titled `<title> (#N)` with the pull
request's own commit messages as the body, and pushes them to `main` in one
fast-forward push. Nothing is merged, so nothing can conflict, and nothing is
re-tested: each tree is the one CI already tested. If `main` moved after the
checks, the push is rejected and nothing lands.

Afterwards it comments on each landed pull request with its commit, closes it,
and deletes its branch. GitHub shows these pull requests as closed rather than
merged, because their head commits are not themselves on `main`. A branch
that an open pull request is still based on is kept, and deleted when that
pull request lands.

The workflow always runs the script and its tests from `main`, never from the
pull requests being landed.

## Setup

Done once by a repository administrator.

1. Create a GitHub App owned by the organisation, with repository permissions
   **Contents: read and write** and **Pull requests: read and write**, and no
   webhook. Install it on this repository only.
2. Generate a private key for it. In the repository settings, store the App's
   ID as the Actions variable `LANDER_APP_ID` and the key as the Actions secret
   `LANDER_APP_PRIVATE_KEY`.
3. In the `main` branch ruleset, add the App to the bypass list. Only the Land
   stack workflow holds its key, and the workflow only pushes what the script
   has checked.

## Testing the script

    python3 -m unittest discover -s scripts/tests

The tests build stacks in a local repository and land them through a fake
GitHub, including the refusals: failed checks, notice waits, a moved `main`,
and a push that loses a race.
