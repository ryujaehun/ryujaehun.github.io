# Full Mermaid rendering audit

The text lint gate passed all 686 Markdown posts but missed six balanced yet invalid Mermaid diagrams. Recent-post browser checks did not cover these older articles.

## Confirmed failures

The actual minified Hugo build contained 162 Mermaid blocks across 131 HTML files. Rendering every block with the build’s Mermaid 11 module produced six parser errors:

- Deadline-Aware Adaptive Prefill Chunking (KO/EN): unquoted square labels containing `min(C_max, L)` and `T(n, c)`.
- Native Sparse Attention (KO/EN): unquoted parenthesized labels in the compression/selection/window pipeline. Replaced literal backslash-n separators with `<br/>` in the same labels.
- An Empirical Study of Qwen3 Quantization: unquoted parenthesized algorithm labels.
- Compress, Gather, and Recompute: unquoted parenthesized cosine similarity label.

Quoted the affected labels without changing graph edges, article claims, metadata, or URLs. An initial Markdown-regex probe also reported Kimi K2 EN incorrectly because it did not recognize a four-backtick closing fence. The production gate reads generated HTML and correctly accepts that valid diagram.

## Deployment gate

`scripts/check-mermaid.mjs public` discovers Mermaid blocks throughout the Hugo output, decodes their text with browser DOMParser, imports the renderer URL present in that build, and calls `mermaid.render` for every block. A parser/import/browser error fails the command; per-diagram failures include HTML path and diagram number. It uses the existing Chrome/Node environment and adds no dependencies. GitHub Pages now runs it as a separate deployment gate.

## Evidence

- Before the content fixes: 162 rendered blocks, six failures, exit 1.
- After the fixes: 162 rendered blocks, zero failures, exit 0.
- All 686 articles pass the existing article lint gate.
- Independent code review found no significant actionable issues.
- All six affected URLs passed headless Chrome checks at 390px and 1440px: rendered SVGs, no Mermaid/KaTeX error markers, no page overflow. Inspected a mobile diagram screenshot.
- Hugo build succeeded; 2,158 HTML files had zero broken local references. Eight local-link unit tests and two reading-path tests passed.
