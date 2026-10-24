# Runtime math and code-copy integrity audit

## Evidence and fixes

A standards-mode Chrome audit used the actual KaTeX scripts and per-page options from minified Hugo output. The initial probe reported 19 parser failures in ten pages (30,549 rendered math nodes across 454 pages loading KaTeX):

- Seven paper articles contained invalid text-mode underscores, unescaped percentage signs, a double superscript on a primed matrix, or doubled command backslashes. Fixed their TeX syntax. The English DMS article also had four stray `$$$$` fragments; restored the surrounding display blocks and preserved actual cases row separators.
- Three shell articles showed errors in hidden code-copy sources rather than visible prose. The theme stored `.Inner` in a hidden `div`; HTML minification removed indentation, and KaTeX processed dollar signs in that div. On the deployed Docker-tag article, the copy source lost the `$` from `$1`/`$3` and contained KaTeX nodes.

An expanded audit compared every hidden source with its highlighted code before and after math rendering: 1,166 source mismatches across 280 pages, with 19 sources changed by math rendering. This covered all 2,908 source blocks in 607 generated HTML pages containing math or copyable code.

The site now overrides the theme code-block hook. The only behavioral change is storing `.Inner` in hidden `pre > code`, which preserves whitespace during minification and is ignored by KaTeX. The nested code element also prevents HTML parsing from dropping an initial source newline. Existing copy/collapse selectors and visible highlighting remain compatible; the theme submodule is unchanged. Keep this override synchronized with future theme hook changes.

The article lint then exposed an automation bug: mathfix treated valid `\text{Per\_token}` labels as corrupted subscripts, removing the escape and breaking rendering again. Detection and repair now preserve balanced text/operator arguments, including nested groups and escaped braces, while still correcting escaped subscripts outside those arguments. Six regression cases failed before this change and pass afterward.

## Permanent gate

`scripts/check-math.mjs public` uses standards-mode headless Chrome, the build’s KaTeX and auto-render scripts, and each page’s actual delimiter/options object. It checks parse-error spans and callbacks, compares copy sources with Chroma output, and verifies that math rendering leaves those sources intact. Missing scripts, incompatible renderer URLs, browser failures, or malformed options fail the command. Only one terminal newline is normalized for comparison; internal/leading whitespace must match.

GitHub Pages runs this check after the full Mermaid gate. The runtime check proves syntactic rendering and copy integrity; paper claims still require source review.

## Validation

- Full minified build: 30,519 math nodes and 2,908 code sources in 607 HTML pages; zero failures. Counts changed because hidden code no longer becomes math and malformed display delimiters were repaired.
- Complete automation suite: 1,023 passed; Ruff and whitespace checks passed. All 686 articles passed text lint.
- All 162 Mermaid diagrams in 131 HTML pages rendered successfully. All 2,158 HTML pages passed local-link/image checks; eight validator unit tests passed.
- Real copy-button interactions for three shell articles: 18 code blocks at each of 390px and 1440px matched their Markdown source, retained indentation/dollar signs, and stayed hidden. The same probe failed against the previous deployed version.
- Independent reviews found no significant actionable issues in the gate, override, or automation correction.
- All ten affected URLs passed real-page Chrome checks at 390px and 1440px with zero KaTeX errors or page overflow; the English cases equation rendered as math.
