# Fragment links and external images — 2026-10-24

A fresh Hugo build contained 2,156 HTML files. All 64,326 checked local fragment
links pointed to an existing element ID or named anchor. Empty fragments, browser
`#top`, and text-fragment directives were excluded. No fragment repair was needed.

The rendered pages referenced 26 unique external image URLs. Network probes found
three failing URLs: a GitHub Camo Qwen logo (403), and two old Docker diagram URLs
(404). Three S3 images returned 200 with a generic binary content type; this alone
was not treated as an image failure.

Repairs:

- Qwen's Korean and English covers now use the original public logo URL encoded
  by the Camo proxy: https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png
- The Docker article's three diagrams now use the current official assets under
  https://www.docker.com/app/uploads/2021/11/. Their source page remains
  https://www.docker.com/resources/what-container/.
- A blank line followed by indentation in the raw HTML comparison table made the
  VM diagram render as escaped code instead of an image. The table is now one
  contiguous HTML block, with descriptive alternative text and automatic height.
- The existing head override now uses `.IsBranch` instead of deprecated `.IsNode`.
  The local Hugo build no longer reports the deprecation warning.

Validation:

- All four replacement image URLs returned 200 with an image content type.
- Chrome at 390px and 1440px decoded all six Docker article images and the Qwen
  logo on both language pages; none of these pages overflowed horizontally.
- Before/after HTML comparison changed only 182 Docker/Qwen image or card pages;
  all unrelated HTML files stayed byte-identical.
- The full rendered local-link/image check passed across 2,156 HTML files.
- External network probes were an audit, not a new network-dependent deploy gate.
