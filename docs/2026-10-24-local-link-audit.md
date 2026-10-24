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

## Translation recovery

The same paper's English edition had exhausted three automated retries. Numeric
validation treated `95th` as a word and rejected a faithful translation of
`95 백분위수`. The validator now normalizes English ordinal suffixes before
comparing numeric values; changed or missing percentiles still fail. Six new
regression cases and the full 990-test automation suite passed.

The recovered translation passes the new validator; the previous validator
rejects the same text with only `수치 누락 또는 변경: 95`. It was generated with
`opencode/glm-5.3-flash-free` using the production-pinned OpenCode 1.18.29 after
local Go credentials returned an expiry error. Reported translation cost: $0.

After recovery, the 686-post source check passed, and a fresh Hugo build passed
the local-link audit across 2,156 HTML files. Every published paper recorded in
the automation state now has a published English counterpart.
