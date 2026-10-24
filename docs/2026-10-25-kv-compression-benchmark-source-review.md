# KV cache compression benchmark source review

Reviewed `2024-11-01-paper-KVCacheCompression.md` against the [original v1 paper](https://arxiv.org/html/2407.01527v1), including Table 2, the four observations in §3.3, and Appendices B/D.

## Corrections

The old generated Q&A attributed the evaluated KV quantization methods' quality losses to compressed prefill representations. The source explicitly says quantized KV is not used in prefill computation for the quantization methods evaluated here. Rewrote the explanation to distinguish prefill-time information reduction from full-prefill KV used in compressed form during decoding. The new conceptual Mermaid shows that phase distinction without claiming an exact buffer-allocation schedule.

Removed the unsupported universal 1–2% accuracy loss and 20–30% speed increase. Added selected Llama-3-8B-Instruct rows from Table 2, identifying the LongBench average as an aggregate score rather than a universal accuracy percentage. Included the harder 64-digit passkey result to limit the interpretation of the ordinary needle test's perfect score.

Distinguished ideal 4-bit payload savings from effective cache compression. The paper reports FlexGen-4bit at 3.20× and KIVI-4bit at 3.11×; residual-window methods' ratios use a 10K-token input. Group size 32 and KIVI residual length 128 are attributed to Appendix B.3. Memory savings, whole-model FLOPs and execution time are kept separate.

Corrected the claim that the benchmark primarily covers summarization. It includes seven task categories including code completion, but focuses on long inputs with short outputs. The review also notes baseline truncation, word versus token units in the needle experiment, Llama-3's modified RoPE setting, small-model scope and unmatched pretraining conditions.

## Independent examples and review

The memory illustration assumes 32 FP16 values (64 bytes), 4-bit packed payload (16 bytes) and two FP16 metadata values (4 bytes). Its 3.2× compression and 68.75% savings are arithmetic examples, not a reconstruction of FlexGen/KIVI's full formats.

A separate affine quantization example uses Decimal arithmetic and explicitly chosen half-up rounding: `[0, 0.7, 1.4, 3]` becomes codes `[0, 4, 7, 15]` at scale 0.2 and reconstructs `[0, 0.8, 1.4, 3]`, maximum absolute error 0.1. The text names the constant-group special case and distinguishes a value-domain offset from an integer-code zero point.

Independent source review found no significant factual or mathematical issues. Original title, date, tags and categories remain unchanged; lastmod and description were added. No English companion or automation logic was changed, and no generation or notification was invoked.

Publication lint, production-version Hugo build, full local-reference/math/copy/Mermaid gates, mobile/desktop browser checks and deployment CI are verified before reporting this change deployed. Repository-wide source auditing remains ongoing.
