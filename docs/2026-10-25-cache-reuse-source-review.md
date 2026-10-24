# CacheBlend and Prompt Cache source review

Reviewed the two Korean posts dated 2024-10-31 against [CacheBlend v3](https://arxiv.org/html/2405.16444v3) and [Prompt Cache v2](https://arxiv.org/html/2311.04934v2). The CacheBlend post explicitly identifies its use of the 2025 revision; original publication dates and titles are retained. No English companion exists for either post.

## Corrections

- Removed CacheBlend's unsupported TF-IDF/BERT query-keyword extraction and priority KV-loading algorithm. The retriever chooses chunks; CacheBlend looks up and fuses their cached states. Explained selective QKV recomputation, reuse of other tokens' KV, progressive HKVD filtering, and layer loading/computation overlap.
- Distinguished position correction from recovering preceding-context effects. Prompt Cache's schema-assigned, potentially discontinuous position IDs allow modular reuse beyond an exact prefix. Independent module encoding still approximates cross-module attention; scaffolding trades extra memory for joint encoding.
- Removed `2048² FLOPs`, the FLOPs/FLOP/s confusion, and the claim that half the cached tokens establishes half the computation or latency. A clearly labeled causal-prefix example counts valid query-key pairs, not GPU operations or Prompt Cache measurements: 2,098,176 full pairs versus 1,573,376 remaining pairs, a 25.012201% reduction. Independently recomputed with integer arithmetic.
- Replaced universal quality guarantees with concrete conditions and limitations. Benchmark tables identify devices, cache location, baseline and revision. Prompt Cache's abstract summary and detailed measurements are distinguished; its Table 1 includes both score increases and decreases. CacheBlend's warm-cache throughput evaluation excludes the initial 1,000 requests.

Added these unit and pipeline distinctions to the automated editorial contract. This is prompt guidance, not a semantic correctness gate. No model invocation, paper generation or notification was needed.

## Verification

Independent source review found no significant factual or mathematical issues. Automation's full local suite passed 1,046 tests with Hugo 0.165.0 enabled; Ruff passed. Both revised posts passed the publication linter after correcting a punctuation-adjacent bold delimiter.

Real Chrome at 390px and 1440px checked both routes, canonical URLs, descriptions, revised content, math rendering and horizontal overflow. CacheBlend renders one formula and one Mermaid diagram; Prompt Cache renders ten math nodes. Original title-based paths remain unchanged.

Full site build, local-reference validation, math/copy and Mermaid rendering gates and the deployed revision are verified before reporting completion of this change. The complete repository-wide content audit remains ongoing.
