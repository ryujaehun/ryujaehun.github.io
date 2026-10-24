# English metadata rerun audit — 2026-10-24

Published English pages already had correct descriptions. The translation writer,
however, copied Korean front matter back over localized metadata on a rerun.
Replaying the existing pairs exposed 12 English descriptions and 5 paper titles
that would regress to Korean.

The translation writer now receives the existing English front matter. It keeps
curated English descriptions and summaries, and keeps an existing English title
when the source title remains Korean after the standard prefix conversion.
English source/arXiv titles remain authoritative. Dates, slugs, categories, and
tags continue to come from the source.

For a new translation, Korean description/summary fields are omitted so the shared
Hugo description partial can extract a preview from the translated English body.
No additional model call is needed. This does not add translation of a wholly
Korean title on a first run; normal generated drafts already use English arXiv
titles.

Validation:

- Seven regression tests cover new previews, preserved descriptions/summaries,
  localized titles, empty metadata, and the actual CLI rerun path.
- Full automation suite: 997 passed.
- Replayed 110 existing pairs without changing their files: all 12 curated
  previews and 5 localized paper titles survived; shared metadata stayed equal.
- Isolated Hugo rendering with the production description partial verified both
  an English body-derived preview and exact preservation of a curated preview.
- Full build: 2,156 HTML files, zero broken local links/images.

The source audit also normalized two legacy long-dash front-matter terminators
(`2025-07-20-daily3-post.md` and `2025-07-26-daily1-post.en.md`). All 686 Markdown
front matters now parse with the automation reader. The English July 20 post's
cover URL contained Markdown backslash escapes; it now matches the original
Korean cover URL and returned HTTP 200 with `image/jpeg`.
