# Keyformer source review

Reviewed `2024-10-31-paper-keyformer.md` against [2403.09054v1](https://arxiv.org/html/2403.09054v1), including Algorithm 1, the scoring formulation and Tables 1–3.

## Corrections

The old Q&A claimed higher temperature could retain more tokens and change compression. The retained-token budget is `k`; temperature changes score magnitudes and their accumulation. Changing a positive temperature while holding the adjusted logits fixed does not change their instantaneous score ranking. Recent-window size changes the split within `k`, unless the total budget itself changes.

The prior pseudocode selected key tokens across all candidates, allowing overlap with the recent window, and contained undefined variables/functions. Replaced it with an executable standard-library helper for selection only: exclude the recent window, choose the remaining highest-score IDs, union without duplication and return chronological original IDs. It explicitly handles `w=0` rather than relying on Python's empty `[:-0]` slice. Its inputs and outputs are illustrative position IDs and scores, not tokenizer words or a complete LLM implementation.

Clarified Q–K-logit Gumbel adjustment and cumulative cache-selection scores without describing a second learned attention layer or promising semantic importance. The time-index notation does not specify a noise resampling implementation. Added a diagram showing the bounded retained cache and stated the inability to recover evicted KV through the selector.

Added Table 3's original/new-position and per-layer/shared-score differences. The “99%” threshold is relative to baseline ROUGE, not universal absolute answer accuracy; cache percentages indicate retained fractions. Distinguished same-batch throughput from the larger batch enabled by lower memory usage. Added actual Table 2 quality losses instead of unreported T5 failure claims or universal losslessness.

## Verification

Executed the exact Python fence from the Markdown. Its example returns `[0, 1, 4]`; checks covered all `k=1..6`, `w=0..k`, empty and short inputs, uniqueness, chronological order and the expected retained count. Independently computed throughput ratios `17/8.3 = 2.04819` at equal batch and `19.85/8.3 = 2.39157` using the larger Keyformer batch.

Independent source/code review found no significant factual, mathematical or code issues. The initial 686-post snapshot passed publication lint before the browser exposed the masked-math case; after the lint fix and pipeline synchronization, all 688 current posts passed. Production Hugo 0.165.0 retains the original title, date, categories, tags and published path; lastmod and description are added. There is no English companion. A browser check also exposed the lint gap described below.

Full-site link/math/copy/Mermaid gates, mobile/desktop browser checks with the former KaTeX CDN blocked, deployment CI and production content are verified before reporting deployment. An initial validation attempt was not executed because automatic approval review hit a usage limit; the same authorized validation subsequently ran successfully. No approval mechanism was bypassed. No model generation or notification was invoked. Repository-wide source review remains ongoing.


## Math-ending bold regression

The first browser check caught literal `**` around `**... $k$**이므로`, although publication lint passed. `_broken_bold` replaced the complete math span with spaces and lost the closing delimiter's punctuation, which controls CommonMark flanking for the surrounding emphasis. The scanner now keeps that final character while continuing to hide internal math markup.

Three new cases failed before the fix; two valid bold-with-math cases remained passing. All 62 lint tests and the full 1,053-test automation suite passed afterward, as did Ruff. A separate code review found no significant issues. The 686-post audit found one additional occurrence in `2026-10-22-paper-2610.20556.md`; its rendered HTML also contained literal markers. Both affected paragraphs now use `<strong>` without changing the math content. The extra post received only that formatting correction.

While validation was interrupted, the remote daily pipeline published a new bilingual review and state/queue commits. Fast-forwarded the workspace to `726220a` and restored the isolated edits without changing those generated files, then rebuilt and revalidated the combined site.
