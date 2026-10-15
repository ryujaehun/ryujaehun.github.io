# Recent posts and reading UX improvements

Scope: October 11–15, 2026 paper reviews in Korean and English, shared home and post metadata UI, and paper-pipeline regressions.

Changes:

- Use descriptive Korean titles, retain the English paper titles, remove duplicate editorial headings, and provide dedicated card/search descriptions for the five reviews. Existing slugs and translation URLs are unchanged.
- Correct SparseDecoding math, N:M example, and bit order. Regenerate all six original tables from arXiv 2610.12327v1 LaTeX, removing color and dashed-rule artifacts without changing numeric cells.
- Correct the DCOCO TPS/accepted-length heading and broken acronym formatting; identify the Full KV and GDN memory baselines in the HLA-WM introduction.
- Collapse original table appendices in the ten revised posts using native keyboard-operable details elements.
- Replace the large home biography with a compact introduction and About link. Show three recent posts before topic hubs; use descriptive topic link titles.
- Collapse post taxonomy metadata and display model attribution separately from subject category links. Preserve existing series navigation and taxonomy archive URLs.
- Fix automatic math normalization to preserve starred operator names and avoid raw HTML collisions. Fix textcolor and cdashline table conversion with regression tests.
- Run dependency-free Chrome smoke checks before deployment: latest five posts and an existing code-series episode in both languages, plus both home pages, at 390px and 1440px.

Validation:

- Paper automation suite: 483 passed (including the latest upstream quality-gate changes).
- Existing paper linter: ten revised source files, zero findings.
- Production Hugo 0.165.0: complete site build succeeds with no duplicate-path warnings.
- Browser smoke checks: 28 page/viewport combinations; no KaTeX errors, malformed HTML elements, or horizontal page overflow; native appendix expansion works with Enter; series navigation is present.
- The first recent-post card begins at y=346px on the Korean 390px home viewport, compared with being below the initial screen before the change.
- SparseDecoding now reports 14 minutes / 6546 Korean characters after repairing its malformed HTML, compared with 3 minutes / 1227 characters previously.

The content review did not independently reproduce every paper's experimental result. Broader article rewrites and a full accessibility/performance audit are outside this change.
