# Serve KaTeX from the blog

A previous production smoke check observed zero rendered formulas once, then passed on recheck; that observation alone did not establish the original network cause. A controlled experiment blocked only `*://cdn.jsdelivr.net/npm/katex@*` in Chrome: the existing article still loaded and rendered Mermaid, but its formulas remained unrendered. This established a reproducible external dependency failure mode.

## Change

Serve the same KaTeX 0.16.22 core, auto-render extension, stylesheet and all 60 font files from `/vendor/katex@0.16.22/`. The original MIT license, provenance README and upstream-file SHA256 manifest are included. The official npm archive's SHA512 integrity was verified before copying, and SHA384 for the three CSS/JS entry files matches the SRI values previously used with the CDN. The 64 unchanged upstream files total 1,381,199 bytes.

A site-level `features/katex.html` override retains the theme's enablement, content detection, delimiters, rendering options and initialization. Only asset URLs change, using Hugo `relURL`. The matching resource hint preloads local CSS; jsDelivr preconnection remains conditional on Mermaid. This change does not self-host Mermaid, analytics or comments, and does not upgrade KaTeX.

The math audit now reads the actual local JavaScript from the generated build. Missing assets fail instead of silently substituting another version. Existing CDN fixtures remain supported. The invalid-math regression fixture and its temporary asset copies use local files, removing their KaTeX network requirement.

The automation configuration reader now honors an active site partial override before reading a theme partial, so it reports the deployed renderer version rather than an unused theme version. An override without a version does not fall back to the shadowed theme.

## Verification

- Before implementation: the three local-renderer cases failed against the CDN-only math gate, and two override-precedence cases failed against the theme-only configuration reader. Controlled CDN blocking also reproduced the user-visible rendering failure.
- Node tests cover valid local rendering, unsupported-command rejection despite permissive page options, missing assets, upstream hashes and every CSS font reference; all 12 asset/renderer/reading-path tests pass.
- Full automation suite with Hugo 0.165.0: 1,048 passed. Ruff and all 686 posts' publication lint passed. Independent code review found no significant actionable issues.
- Reading UI checks now always block the former KaTeX CDN and verify local asset origins, loaded fonts and absence of font errors. The 24 local mobile/desktop page checks passed, as did the housing report UI checks. A separate article smoke check rendered ten math nodes with the CDN blocked and observed local CSS/JS/font responses.

Full-site local-reference/math/copy/Mermaid gates and both repository CI runs are verified before reporting deployment. Production is checked again with the former KaTeX CDN blocked. This removes the identified KaTeX CDN dependency; it does not claim that all possible browser or network failures are eliminated.
