# Repository agent workflow

## Branch naming

Use one short-lived task branch with a case-sensitive prefix followed by a lowercase,
hyphenated description:

- `Feature/`: new user-facing functionality, e.g. `Feature/scan-search`.
- `Bugfix/`: defect fixes, e.g. `Bugfix/playback-orientation`.
- `Enhance/`: improvements to existing functionality, performance, UX, documentation,
  or maintenance, e.g. `Enhance/bilingual-ui-docs-workflow`.

Never use `codex/`, lowercase variants of these prefixes, or additional long-lived
branches for routine work. Preserve unrelated changes; use an isolated worktree
when the existing checkout contains work belonging to another task.

## Commit and pull request

1. Use a specific English commit subject beginning with `feat:`, `fix:`, `enhance:`,
   or `docs:`.
2. Push the task branch and create a PR.
3. Write the PR in English first: describe the final behavior, important tradeoffs,
   test results, and physical-device checks still pending. Include a Traditional
   Chinese summary afterward.
4. If the application provides task attachment tools, link or attach the PR to the
   current task.

## Validation and merge

- Run the relevant checks. Repository CI runs `ruff check .`,
  `mypy pipeline/config.py`, and `pytest` (install with `pip install -e '.[dev]'`).
- For UI changes, check desktop and narrow layouts, keyboard navigation, and affected
  interactions. Distinguish browser emulation from validation on physical devices.
- Check PR mergeability, required checks, and required reviews. Fix failures, rerun
  affected checks, and update the PR.
- Merge the verified HEAD, normally with a squash merge. Ensure the HEAD being merged
  matches the commit that passed validation.
- Never push directly to `main`, use administrator bypass, disable protections, or
  bypass required reviews.
- If an external gate blocks merging, report its exact name and leave the PR and
  task branch intact.

## Cleanup after confirmed merge

1. Confirm the PR successfully merged the verified HEAD and the task worktree is clean.
2. Delete only this task's branch from `origin` and locally.
3. Update local `main` with a fast-forward only and prune stale remote-tracking refs.
4. After a squash merge, `git branch -D` is allowed only after the merge and clean
   worktree checks above.

Never delete unrelated branches or worktrees, or unmerged work. If the original
checkout is dirty, keep it intact while updating `main` from a clean task worktree.

## Project boundaries

Keep the web panel and pipeline imports free of heavy training dependencies.
`templates/`, `static/css/`, and `static/js/` contain the UI; `web/` handles HTTP and
form processing; `pipeline/` orchestrates external tools. Preserve form field names,
request endpoints, and job log contracts when improving presentation.
