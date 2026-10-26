# Detailed Telegram failure alerts — 2026-10-26

Scope: Daily papers, Code series (work/publish/notify), and Deploy Hugo site to Pages. The user approved implementation, push and deployment.

Each repository has a separate `Detailed failure alerts` workflow triggered on completion of its production workflows. It executes only default-branch monitor code, filters source repository/branch/event, and reads logs/reports as data. Initial checkout/setup failures can be reported because the monitor runs independently. No source artifacts are executed or extracted to disk.

Messages include workflow, branch/commit, failed job and steps, explicit error-line excerpts, paper/task/notification identifiers from partial-failure reports, source attempt and log links. Series job names include the project slug. The message explicitly says the alert does not initiate an automatic workflow retry; internal retry details remain in the logs. Successful, clean runs and intentional cancellations are silent. Failed diagnostic retrieval does not label a successful source run as failed; the monitor fails visibly and remains retryable instead.

Per-source-run/attempt concurrency and a delivery receipt cache suppress duplicate sends when a completed monitor is rerun. This is best effort: Telegram acceptance followed by runner/cache failure or cache eviction can still duplicate a message. The monitor never retries ambiguous Telegram requests automatically. A failed send leaves no delivery receipt. Secrets are redacted before excerpts and UTF-16 length limits are applied; links are retained and no Telegram HTML/Markdown parsing is enabled.

The standalone standard-library script is mirrored at `automation/core/workflow_alert.py` and `scripts/workflow_alert.py` because public blog runners cannot check out the private automation submodule. Both copies must remain byte-identical and run the same behavioral tests. Existing Telegram settings were copied to the blog repository’s encrypted Actions secrets without displaying their values.

Manual `workflow_dispatch` on the monitor sends a clearly labelled fictional connection test, never a fake production incident. Production failures are inspected automatically; manual connection checks do not run generation, publication or deployment commands.

Validation: 15 new behavioral tests pass (14 pure behavior/transport tests for both copies and one real workflow command test); automation suite 1,076 passed / 5 Hugo-dependent tests skipped locally (CI installs pinned Hugo). Ruff and actionlint 1.7.12 pass. Existing blog link tests (8) and Node/browser regression tests (12) pass. Independent review found and verified fixes for series partial-failure omissions and diagnostic-only false alerts; final review found no important blockers. Live monitor dispatches, repository CIs and Pages deployment are verified after push.

GitHub documents completion-triggered workflows and permissions at https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run and job-log downloads at https://docs.github.com/en/rest/actions/workflow-jobs#download-job-logs-for-a-workflow-run .

Live connection testing caught `core/http.py` shadowing Python’s standard library when running a file directly; both workflows now use `python3 -I`. A regression executes the actual workflow command with `--help`. Both repositories delivered labelled connection-test messages. Rerunning the blog connection test restored its receipt and skipped sending. A real successful Pages completion triggered the monitor, which reported no failure and sent no message. Partial fetch source errors are also included in the report artifact.
