#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Maintain the "Timing test health" issue for the Timing diagnostics workflow.

The timing tests are expected to fail now and then on shared runners; what
matters is whether a test fails consistently. Each run of the workflow adds a
column to a table of recent runs kept in a single open issue, and the issue
gets a comment only when the set of consistently failing tests changes, so
nobody has to remember to look.

Inputs are the per-OS result files written by the workflow
(<dir>/timing-<os>/results.tsv): a first line "outcome<TAB><step outcome>",
then one "<test><TAB>pass|fail" line per test. A missing file, or an outcome
other than success/failure, means the tests did not run (e.g. the build
failed), which is recorded as a failure of the pseudo-test "(no results)".

The run history lives in a hidden HTML comment in the issue body, so the
workflow needs no storage of its own. With --dry-run the issue body is read
from and written to files and gh is not called, for local testing.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LABEL = "timing-health"
TITLE = "Timing test health"
HISTORY = 8  # runs shown and kept
WINDOW = 4  # a test failing in at least THRESHOLD of the last WINDOW runs...
THRESHOLD = 3  # ...is reported as consistently failing
NO_RESULTS = "(no results)"
STATE_RE = re.compile(r"<!-- timing-health-state: (.*?) -->", re.S)


def read_results(results_dir):
    """Return {os: {test: "pass"|"fail"}} for every timing-<os> directory."""
    results = {}
    for d in sorted(Path(results_dir).glob("timing-*")):
        os_name = d.name[len("timing-") :]
        tests = {}
        tsv = d / "results.tsv"
        lines = tsv.read_text().splitlines() if tsv.is_file() else []
        outcome = lines[0].split("\t")[1] if lines and "\t" in lines[0] else ""
        if outcome in ("success", "failure"):
            for line in lines[1:]:
                name, _, status = line.partition("\t")
                if name:
                    tests[name] = status
        if not tests:
            tests[NO_RESULTS] = "fail"
        results[os_name] = tests
    return results


def consistent_failures(runs):
    """(os, test) pairs failing in at least THRESHOLD of the last WINDOW runs."""
    recent = runs[-WINDOW:]
    counts = {}
    for run in recent:
        for os_name, tests in run["results"].items():
            for test, status in tests.items():
                if status == "fail":
                    counts[(os_name, test)] = counts.get((os_name, test), 0) + 1
    return sorted(k for k, n in counts.items() if n >= THRESHOLD)


def render(runs, failing):
    """Issue body: a summary, a table of tests that failed recently, and the state."""
    out = [
        "Maintained by the Timing diagnostics workflow (`.github/workflows/timing.yml`); "
        "edits to this body are overwritten.",
        "",
        f"A test is flagged when it fails in {THRESHOLD} of the last {WINDOW} runs; each test "
        "gets three attempts per run, so a single failure is usually a busy runner.",
        "",
    ]
    if failing:
        out.append("**Consistently failing:** " + ", ".join(f"`{t}` ({o})" for o, t in failing))
    else:
        out.append("**Consistently failing:** none")
    out.append("")

    flaky = sorted(
        {
            (o, t)
            for run in runs
            for o, tests in run["results"].items()
            for t, s in tests.items()
            if s == "fail"
        }
    )
    total = sum(len(t) for t in runs[-1]["results"].values()) if runs else 0
    if not flaky:
        out.append(f"All {total} tests passed in each of the last {len(runs)} runs.")
    else:
        header = "| OS | Test | " + " | ".join(
            f"[{r['date']}]({r['url']})" for r in runs
        ) + " |"
        out += [header, "|---|---|" + "---|" * len(runs)]
        for o, t in flaky:
            cells = []
            for r in runs:
                s = r["results"].get(o, {}).get(t)
                cells.append({"pass": "✓", "fail": "✗"}.get(s, "–"))
            out.append(f"| {o} | `{t}` | " + " | ".join(cells) + " |")
        out += ["", f"Tests not listed passed in every run shown ({total} in the latest run)."]
    state = {"runs": runs, "alerted": [list(f) for f in failing]}
    out += ["", f"<!-- timing-health-state: {json.dumps(state, sort_keys=True)} -->", ""]
    return "\n".join(out)


def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results_dir")
    p.add_argument("--date", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--run-url", required=True)
    p.add_argument("--mention", default="", help="e.g. @someone; added to alert comments")
    p.add_argument("--dry-run", nargs=2, metavar=("BODY_IN", "OUT_PREFIX"))
    args = p.parse_args()

    if args.dry_run:
        body_in = Path(args.dry_run[0])
        issue, old_body = ("dry", body_in.read_text()) if body_in.is_file() else (None, "")
    else:
        found = json.loads(
            gh("issue", "list", "--label", LABEL, "--state", "open", "--json", "number,body")
        )
        issue, old_body = (found[0]["number"], found[0]["body"]) if found else (None, "")

    m = STATE_RE.search(old_body)
    state = json.loads(m.group(1)) if m else {"runs": [], "alerted": []}
    run = {
        "date": args.date,
        "sha": args.sha[:7],
        "url": args.run_url,
        "results": read_results(args.results_dir),
    }
    runs = (state["runs"] + [run])[-HISTORY:]
    failing = consistent_failures(runs)
    previously = sorted(tuple(f) for f in state.get("alerted", []))
    body = render(runs, failing)

    comment = None
    new = [f for f in failing if f not in previously]
    recovered = [f for f in previously if f not in failing]
    if new or recovered:
        parts = []
        if new:
            names = "\n".join(f"- `{t}` on {o}" for o, t in new)
            parts.append(
                f"{args.mention + ' ' if args.mention else ''}Now failing in {THRESHOLD} of the "
                f"last {WINDOW} runs:\n\n{names}"
            )
        if recovered:
            names = "\n".join(f"- `{t}` on {o}" for o, t in recovered)
            parts.append(f"No longer failing consistently:\n\n{names}")
        parts.append(f"As of {run['sha']}: {args.run_url}")
        comment = "\n\n".join(parts)

    if args.dry_run:
        prefix = args.dry_run[1]
        Path(prefix + "body.md").write_text(body)
        Path(prefix + "comment.md").write_text(comment or "")
        return 0
    if issue is None:
        gh("label", "create", LABEL, "--color", "FBCA04", "--force",
           "--description", "Weekly timing/priority test results")
        url = gh("issue", "create", "--title", TITLE, "--label", LABEL, "--body", body).strip()
        issue = url.rsplit("/", 1)[-1]
    else:
        gh("issue", "edit", str(issue), "--body", body)
    if comment:
        gh("issue", "comment", str(issue), "--body", comment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
