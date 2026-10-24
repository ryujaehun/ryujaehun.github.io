# Retranslation route stability

## Reproduced problem

A read-only audit inspected 111 Korean/English pairs. Before the fix, a separate Hugo site built from their front matter showed one changed URL among 110 published English articles when applying the current retranslation metadata logic:

- Qwen3: `/en/posts/paper-review-qwen-3-technical-report/` became `/en/posts/paper-review-qwen3-technical-report/` because the source title has no space before 3.

The actual published bodies were not retranslated. The previous metadata builder also replaced English-specific slugs, custom URLs and aliases with Korean fields (or dropped them), demonstrated with isolated regression cases.

## Change

Existing English `url` and `aliases` remain owned by that article, including an explicitly empty alias list. Korean custom URLs/aliases are not newly copied over existing English defaults. Existing English slugs are retained. When a source title/slug update would change an implicit English title-based route, the old title is pinned as its slug and Hugo performs the normalization. Titles and other shared metadata can still update according to the existing policy; localized previews remain preserved. New translations retain their shared source slug behavior.

The real Qwen3 English article now explicitly pins its already-published slug. Its title, body and current address are unchanged.

## CI and engine version

CI now installs the same Hugo 0.165.0 version used by the blog's Pages workflow, verifies the official release archive SHA-256, and sets `HUGO_BIN` for native route tests. The CLI itself does not gain a Hugo dependency. Without `HUGO_BIN`, the three native tests are explicitly skipped; CI supplies it and executes them.

This version alignment matters: local Hugo 0.168 changes how title-based path separators are normalized, while explicit slugs retain path separators. An additional slash/Unicode fixture exposed that difference. The routing checks target the actual deployment version and the blog's `/posts/:slug/` permalink configuration. Keep the two workflow pins aligned and rerun these cases before changing the deployment engine.

## Verification

- Nine route regression cases cover explicit and implicit slugs, custom URLs, aliases, source-field updates, first translation and CLI-level retranslation with a stubbed model. Seven original cases failed before the change; an additional case caught loss of an explicit slug matching the new title.
- Three native Hugo cases cover the Qwen title update, custom URL/redirect preservation, and Unicode/punctuation/path separators.
- A larger native audit using Hugo 0.165.0 checked all 110 published English articles plus three synthetic cases: zero changed URLs; four alias redirect files and their targets remained identical.
- Full automation suite with native tests enabled: 1,035 passed. Ruff and whitespace checks passed. The CI installer shell script also passed syntax validation.
- Independent review found no significant actionable issues in metadata preservation or CI setup.
- The full site was built with deployment Hugo 0.165.0: 30,468 math nodes, 2,878 copy sources and 162 diagrams passed. All 2,158 HTML files passed local-reference checks; eight validator unit tests passed.
- Qwen3 English passed Chrome at 390px and 1440px: existing canonical URL, title, Korean alternate link, two math nodes, one rendered diagram, no math errors or page overflow.
