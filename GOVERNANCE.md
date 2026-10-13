# Governance

dispenso is a small project and this document is short by design. It describes
how decisions are actually made, not an aspirational process.

> The project is mid-transfer from `facebookincubator` to a neutral
> organisation. This document applies now and continues to apply afterwards; the
> move changes where the repository lives, not how it is run.

## Roles

- **Contributors** open issues and pull requests; anyone may, and nothing below
  constrains them.
- **Committers** have write access: they review, merge and may object. No review
  quota.
- **Maintainers** are committers who also decide direction: release content and
  timing, and what the project will not do.
- **Active** means a seat-holder who has not resigned, gone emeritus, or been
  removed. **Maintainer involvement** means a maintainer as author or approver.

| Name | Role | Affiliation |
|---|---|---|
| Brian Budge | Maintainer | Independent |
| Pierre Moulon | Maintainer | Meta |
| Michael Ranieri | Committer | Meta |

The roster records decisions made under this document rather than forming part
of it, so updating it is not an amendment. Affiliations are self-reported.

Administrative access — organisation ownership, and Meta infrastructure accounts
while the project remains under `facebookincubator` — is custodial: no
decision-making authority, exercised per maintainer decisions, except where
security or account recovery leaves no time to ask.

## Merging changes

Changes are made by pull request, and required automated checks must pass. One
may be merged once either **reviewed** — a committer other than its author has
approved it — or **noticed**: its author is a committer, no committer has
objected, and the wait below has elapsed.

| Change | Maintainer | Notice wait | Release note |
|---|---|---|---|
| Internal implementation | — | 24h | — |
| Compatible stable-surface addition | — | 24h | Yes |
| Alters unreleased stable surface | — | 24h | Yes |
| Alters the stable surface of the latest tagged release | Yes | 72h | Yes |
| Undisclosed security fix | Yes | none | With the release |

The **stable surface** is public API and ABI, default behaviour, and documented
concurrency guarantees — ordering, forward progress, cancellation, lifetime and
safety. Undocumented scheduling choices and internal threading implementation
are not, and change in the normal course of the work. A pull request that adds
to or alters it must say so.

An addition that could reasonably be expected to break existing documented use
counts as an alteration. The test is reasonable expectation, not bare
possibility; when genuinely unsure, treat it as an alteration and let review
decide.

Released promises get 72 hours because downstream builds against them. The
notice route itself accommodates uneven review capacity, keeping the chance to
object without requiring anyone to be present; it is for uncontroversial work,
not a way around review.

- A material change restarts the notice period.
- Objections belong on the pull request, or another durable channel.
- An unresolved objection blocks either route. An author may not resolve one
  against their own change; the active maintainers may, by consensus, having
  engaged with it and recorded the outcome.

If the maintainers cannot agree, the change does not land.

## Neutrality

At least two maintainers must not share an employer; an independent maintainer
satisfies this. Below that, the remaining maintainers say so in the README and
treat recruiting as the priority.

Seats belong to individuals, not employers, and do not lapse on a job change. No
employer may appoint or remove a committer, direct a vote, or hold a controlling
one; several sharing an employer confers no control. An organisation that
depends on dispenso may propose a candidate where its perspective has gone
unrepresented; the maintainers weigh them on the usual criteria and may decline.
It is a right to be heard, not a right to a seat. Meta is the present example.

## Joining and leaving

| Action | Rule |
|---|---|
| Appoint a committer or maintainer | Proposed by any maintainer; consensus of the active, non-recused maintainers, after a reasonable interval to respond |
| Resign or take emeritus | On request, at any time |
| Suspend access in an emergency | Any maintainer or organisation owner, where reasonably necessary to protect the project; temporary, and reported promptly |
| Remove for cause | Majority of the other active maintainers, after notice and an opportunity to respond; the person is recused |
| Move an inactive member to emeritus | Any maintainer, after six months without participation and an unanswered 30 days' notice |
| Reinstate an emeritus member | Any maintainer, on request |
| Amend this document | All active maintainers |

The bar for a seat is sustained contribution and knowing the limits of one's own
knowledge — concurrency bugs are easy to introduce and expensive to find. A
maintainer must also know which changes to decline.

A removal is recorded, publicly or privately as circumstances warrant. Where
recusal would leave no other active maintainer, access may be suspended but
removal waits. Emeritus removes write access and the vote; it is not a
judgement.

If there are no active maintainers left, whoever notices should mark the project
unmaintained in the README and open a pinned issue: saying so does far less harm
than merely appearing maintained.
