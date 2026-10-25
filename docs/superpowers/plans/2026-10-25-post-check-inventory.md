# Complete post inventory for publication checks

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Work in the existing authorized workspace and preserve unrelated pipeline changes.

The publication gate must not declare success or apply fixes after a directory enumeration/read failure, including when explicit paths omit an unreadable peer. Certain duplicate title/slug checks must cover nested regular pages and bundle indexes without treating section indexes or leaf-bundle Markdown resources as independently published pages. Hugo remains authoritative for normalized/custom-route collisions beyond this conservative check.

- [x] Add failing CLI, nested collision and resource-exclusion regressions. Use production Hugo 0.165.0 fixtures to prove the collision/resource distinction.
- [x] Share strict Markdown enumeration/read errors with reviewed-ID scanning; preflight URL inventory before fixes and return structured configuration errors. Preserve explicit-path body-check scope and missing-directory behavior.
- [ ] Pass targeted/full tests, lint and actual workspace read-only checks; independent review; commit/push automation, then root pointer/docs, verify both CIs, Pages deployment and live site.

Files: `automation/tasks/papers/post_scan.py` (strict inventory), `state.py` (reuse/re-export), `lint.py` (recursive regular-page collisions), `cli.py` (preflight/report), and corresponding papers tests. No production paper generation or notifications.

Ruling: retain the conservative title/slug collision heuristic and Hugo deployment gate; this change extends inventory coverage rather than reimplementing Hugo URL normalization. Section indexes and leaf-bundle Markdown resources are excluded only from article-route grouping, not from read validation.

Ruling: preflight failures prevent fixes, but this is not a transactional filesystem operation. If a peer changes during fixes, report the late read error with the actual completed fix count; recheck collision keys after any formatting changes. Regression tests cover both cases.
