# Stop automation when existing posts cannot be scanned

The existing-post scan silently ignored unreadable Markdown and directory traversal errors. A partial set of reviewed paper IDs could therefore authorize another draft for a paper already covered by a manual post or page bundle.

The scanner now raises `PostScanError` with the affected path for read, UTF-8 decoding, and directory enumeration errors. `status`, `score`, and `materialize` return configuration exit code 4 instead of treating an incomplete scan as valid. A missing posts directory remains valid for a fresh workspace; a file at that path is an error.

Materialization rechecks existing posts after each PDF download. If this later scan fails, it stops the batch, reports the error, and checkpoints any earlier successful drafts and queue removals. With no earlier progress, state and queue contents and modification times remain unchanged. A downloaded PDF may remain cached. Scan errors do not count as paper download failures.

## Verification

- Before the fix, 10 of the 11 new regression cases failed. Coverage includes three CLI commands, unreadable/invalid UTF-8 bundle files, directory enumeration errors, an invalid posts root, and errors during the first or second PDF download.
- After the fix: 1,046 automation tests passed with production Hugo 0.165.0 enabled; Ruff passed. A separate review found no significant actionable issues.
- Read-only status against the real workspace completed: 434 reviewed IDs and 63 history entries, with no pending summary or translation. These IDs include references in article bodies and are not a count of standalone reviews.
- Fault injection used temporary test workspaces only. No production post was corrupted, no live paper generation or notification was triggered.

The full site build, link checks, math/copy checks, Mermaid rendering, and both repository CI runs are verified before this change is reported as deployed.
