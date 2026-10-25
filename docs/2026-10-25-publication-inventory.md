# Publication checks require a complete post inventory

The body checker recursively inspected Markdown, but its URL collision pass used a flat glob and silently skipped read failures. Directory enumeration failures could also turn a partial/default scan into success. A partial `check --path ... --fix` could therefore alter the selected post while ignoring an unreadable nested peer.

## Behavior

A shared `post_scan` module now provides deterministic Markdown enumeration and strict UTF-8 reads. Directory enumeration, invalid posts roots and read failures carry the affected path in `PostScanError`. Reviewed-ID scanning reuses the same implementation, with its existing public error import retained. Missing posts directories remain valid for empty workspaces.

`check` preflights the full inventory before applying fixes. Failures return structured non-fixable read problems and configuration exit code 4. Explicit paths still limit body lint/fix scope; the whole workspace remains relevant for collision and read validation.

The conservative collision pass covers nested regular pages and leaf-bundle indexes, including English editions. Section indexes and Markdown resources inside leaf bundles do not become independent article-route candidates. Resources are still read during preflight so unreadable Markdown cannot be silently ignored.

If fixes change front matter, collision groups are recomputed from the resulting text. If another file changes during fixing and cannot be reread, the command returns a read failure and reports the already-completed fix count. It does not promise atomic rollback of completed edits.

## Evidence

- Controlled temporary workspaces reproduced skipped nested collisions, directory errors and unsafe partial fixes. The native Hugo fixture was given a page template so actual output collisions were observable; production Hugo 0.165.0 confirms a flat post and equal-title bundle index collide, while a bundle Markdown resource does not.
- Regression coverage includes an unstaged nested collision during `--path`, unreadable bundle resources, missing versus invalid roots, post-fix title collisions and a late read failure. Existing state/materialization replay tests still pass.
- Full automation suite with Hugo enabled: 1,066 passed. Ruff passed. Independent review found no significant actionable issues, including the post-fix rescan delta.
- Read-only checks against the actual workspace found 688 posts without publication lint problems, 435 reviewed IDs and 64 history entries. The IDs include body references and are not a standalone-review count. No production generation, notification or `--fix` was invoked.

Normalized/custom-URL collision handling and malformed front-matter behavior beyond the prior heuristic remain outside this change; Hugo's full output-path gate is authoritative. Full site checks, both repository CIs, Pages deployment and live HTTP are verified before reporting this chunk deployed. Repository-wide source auditing remains ongoing.
