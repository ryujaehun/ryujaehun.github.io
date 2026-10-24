# Materialize replay safety and paper identity corrections

## Failure and protection

A saved scored JSONL could bypass `score` exclusions when passed directly to `materialize`. In isolated test workspaces, replay replaced a reviewed published body with a prompt-only draft and reset its state to `materialized`. A later date/version could create another post.

`materialize` now reapplies existing-post/state exclusions after the original top-k boundary, deduplicates selected rows, and returns skipped IDs. Pure skips do not download PDFs or save state. Draft creation rechecks posts after downloads and uses an atomic no-clobber hard link from a unique complete sibling temporary file. Normal summary/translation writes retain overwrite behavior. Bundle posts and arXiv abs/PDF/HTML links participate in existing exclusions. As before, body references also count as known IDs.

This protects sequential replays and competing file creation; it does not introduce global multi-process state transactions. Production workflow serialization remains necessary. Failed downloads remain retryable and cannot be hidden by a later collision skip.

## Real workspace audit

All 63 state draft paths existed: 62 published and one existing unpublished SpecAgent draft (2610.21463). The draft was already unpublished in its original commit and was left unchanged. No production materialize write was performed. A real saved-input dry run for 2610.28559 returned no candidates and one skipped ID; actual post/state SHA-256 hashes and mtimes remained unchanged.

## Paper corrections

- Seedream 2.0 linked to FlexWorld. Corrected to [Seedream 2.0 v1](https://arxiv.org/abs/2503.07703v1).
- The older MoE review also linked to FlexWorld and contained unsupported FasterMoE/MegaBlocks comparisons and an incorrect FLOP explanation. Replaced the generated Q&A with a concise account of [Towards MoE Deployment v2](https://arxiv.org/abs/2303.06182v2), using its [PDF](https://arxiv.org/pdf/2303.06182v2): Table I uses eight V100 32GB GPUs per node; §V.B/Figure 9 single-node maxima against static gating are 6.21×/5.75×/2.58×; Figure 10 reports activation memory, not total GPU memory. Preserved both the abstract’s 11.23× and body’s 11.55× discrepancy explicitly. The illustrative token packing example is labeled as such.
- Corrected MoE title/tags and set an explicit slug preserving `/posts/toward-efficient-inference-for-mixture-of-experts/`. Original publication date retained.

## Validation

- 20 targeted replay/atomic-creation cases passed; complete automation suite: 1,017 passed.
- Ruff and diff whitespace checks passed; both edited posts passed article checks.
- Hugo build succeeded; 2,158 generated HTML files had zero broken local links/images. Eight local-link validator tests and two reading-path tests passed.
- Headless Chrome at 390px and 1440px: stable MoE URL, correct source link, two tables, rendered Mermaid SVG, no horizontal page overflow or KaTeX errors.
- Independent review found no significant actionable issues and checked the numerical claims against the v2 PDF.
