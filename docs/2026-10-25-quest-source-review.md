# Quest source review

Reviewed `2024-11-01-paper-quest.md` against [2406.10774v1](https://arxiv.org/html/2406.10774v1), including Algorithm 1, the traffic model, Table 1 and the kernel/decode evaluation settings.

## Corrections

The prior Q&A contradicted itself about cache retention, called omitted pages temporarily evicted, and suggested fixed page budgets bound total memory. Quest retains the whole KV cache and chooses pages to read for the current Q vector. Metadata grows with context length. The new explanation separates stored bytes, read bytes, selected page count, selected token budget and per-kernel latency.

Clarified that the per-page min/max Key score bounds raw Q–K dot products rather than being a softmax probability or a direct Q–Value correlation. Different channels can take extrema from different tokens, so ranking page upper bounds need not recover exact top tokens. Added a worked counterexample and a page-selection diagram showing continued KV retention.

Removed the unsupported universal page-size-16 recommendation. The paper evaluates estimation latency across page sizes and uses 16 for approximate-attention comparisons; this does not prove a universal joint accuracy/latency optimum. The text distinguishes fixed page count from fixed token budget and the dense first two layers from later layers using sparse selection.

Replaced broad quality/speed claims with selected Table 1 results, the modified H2O comparison setup, and the scope of Figure 11's qualitative baseline analysis. Self-attention acceleration on RTX 4090 is separate from single-batch model decode latency on Ada 6000. The latter excludes sampling and does not establish prefill, TTFT or full-request speedup; its 4-bit condition concerns weights, not automatically KV storage.

## Independent calculations

Decimal arithmetic confirms the illustrative query `(1,0.5,-0.2)` and two example Keys produce a page upper bound 1.36, while actual dot products are -0.49 and 1.12. A second page with Key `(1.2,0,0)` has a lower upper-bound score than the first page but a higher true maximum dot product.

For 65,536 tokens, page size 16 and 256 selected pages, the selected token budget is 4,096 and the modeled read ratio is `1/16 + 4096/65536 = 1/8`. The source v1 example says “4K pages,” which does not yield its claimed eightfold reduction under its own formula. The article explicitly flags the unit inconsistency and labels its corrected calculation as illustrative.

Independent source review found no significant factual or mathematical issues. Original title, publication date, categories, tags and route are retained; lastmod and description are added. All 686 posts pass publication lint. No English companion or automation logic is changed, and no generation or notification is invoked.

Production-version Hugo build, mobile/desktop checks at the original route with the KaTeX CDN blocked, full local-link/math/copy/Mermaid gates and deployment CI are verified before reporting deployment. Repository-wide source review remains ongoing.
