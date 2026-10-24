# Local link and image audit — 2026-10-24

The rendered build contained four broken references across three posts. The
existing desktop/mobile smoke check samples recent posts, so it did not cover
these older pages.

Restored original SVG assets (trailing blank lines normalized):

- `static/img/flat_hash_map.svg`: https://abseil.io/img/flat_hash_map.svg
- `static/img/node_hash_map.svg`: https://abseil.io/img/node_hash_map.svg
- `static/assets/images/papers/2610.20556/figure5-quality-efficiency.svg`:
  https://arxiv.org/html/2610.20556v1/x1.svg (Figure 5, CC BY 4.0).
  The post referenced a WebP file that was never present in the committed assets.

The Protocol Buffers link in Tip 144 used `(protobuf)` instead of the existing
`[protobuf]` reference definition. It now resolves to https://protobuf.dev/.

The Pages workflow now runs `scripts/check_local_links.py` after Hugo and before
uploading the artifact. It checks every generated HTML file's `a[href]` and
`img[src]` against the build, including same-host absolute URLs and page-relative
URLs. It does not fetch external sites or validate fragment IDs, CSS URLs,
JavaScript resources, or `srcset` candidates.

Verification: 8 unit tests passed; the prior build reproduced all 4 broken
references; a fresh Hugo build contained 2,153 HTML files with 0 broken local
links/images. Chrome rendered the restored Figure 5 SVG successfully.

Run locally after building:

```sh
python3 -m unittest discover -s scripts -p 'test_check_local_links.py'
python3 scripts/check_local_links.py public --base-url https://jaehun.me/
```
