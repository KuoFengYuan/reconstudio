# CLAUDE.md

@AGENTS.md

## Claude Code notes

- `AGENTS.md` (imported above) is the single source of agent rules; edit it, not
  a copy here. This file only adds what is specific to Claude Code.
- A user question about a running job, a failed stage or "why didn't my change
  apply" usually needs the job record, not the code: `<RECON_STUDIO_DATA>/jobs/<id>/job.json`
  and `console.log`. `RECON_STUDIO_DATA` comes from `local.env`. Read only that
  job's files.
- Before restarting a live panel, list the panel's child processes (trainers run
  as children of the uvicorn PID) and say what would be killed; restart only when
  idle or when the user confirms.
- For UI work, run a sandbox panel (AGENTS.md §3) and look at it in a real
  browser at desktop and 390 px widths before reporting the change as done.
