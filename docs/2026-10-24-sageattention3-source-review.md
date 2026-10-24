# SageAttention3 bilingual source review

Reviewed both `2025-07-09-paper-2505.11594v1` articles against the [v1 paper](https://arxiv.org/html/2505.11594v1) and the separate [NVIDIA NVFP4 format reference](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/features/low_precision_training/nvfp4/nvfp4.html).

## Problems corrected

The prior generated Q&A conflated temporary attention-tile quantization with persistent KV-cache storage and asserted a 75% VRAM reduction, larger batches and longer context without establishing the storage format. It omitted scale metadata from its bit calculation, described FP4 as integer-like with an incorrect maximum, counted all attention matmuls as backward operations, and mixed kernel speed with full training speed. It also generalized fine-tuning quality to all training and supplied unsupported software-version and compute-budget tables.

Replaced these sections in both languages with a compact account separating inference/training precision, kernel/end-to-end measurements and pretraining limitations. The result table retains the paper’s actual benchmark rows and identifies Table 4’s `Original` baseline without assuming it is FlashAttention2. Section and algorithm references are attached to the relevant claims. The separate NVIDIA format reference is not presented as the paper’s complete implementation or softmax scaling algorithm.

## Independent calculations

These examples are explicitly labeled as illustrative, not measured paper results:

- A 16-value block `[0,0.7,1.4,3]` plus twelve zeros, global scale 1, block scale 0.5. Nearest E2M1 codes are `[0,1.5,3,6]`, reconstructing `[0,0.75,1.5,3]`. Decimal arithmetic verifies maximum absolute error 0.1 and block MSE 0.00078125.
- Packed FP4 payload plus one FP8 scale totals 9 bytes against 32 bytes for sixteen FP16 values. The 71.875% reduction excludes global scale, alignment and other metadata; it does not establish KV-cache or total VRAM savings.
- A hypothetical 100 ms workload with 40 ms attention falls to 68 ms when attention alone becomes 5× faster, about 1.47× overall before extra overhead.

## Assets, metadata and automation

Copied the unmodified [official Figure 2 PNG](https://arxiv.org/html/2505.11594v1/alg_overview.png) locally (99,620 bytes), credited Zhang et al. and CC BY 4.0 in both articles, and reused it as their cover. Retained publication dates and exact titles; added lastmod and language-appropriate descriptions. Existing Korean/English canonical URLs and reciprocal alternate links are unchanged.

Added the relevant precision/storage and measurement-scope distinctions to the existing automated review quality contract. The contract is appended to the review prompt in `automation/tasks/papers/cli.py`; this is editorial guidance, not a semantic proof or substitute for source review.

## Validation

- Independent content review found no significant factual issues in either language.
- Translation structure/numeric checks returned no problems; all 686 articles passed lint.
- Full automation suite: 1,023 passed; Ruff and whitespace checks passed.
- Minified build: 30,468 math nodes, 2,878 copy sources and 162 Mermaid diagrams passed. All 2,158 HTML pages passed local-reference validation; eight validator tests passed.
- Real Chrome at 390px and 1440px for each language: six math nodes and three tables, loaded local figure, correct canonical/alternate URLs, English description without Korean, no math errors or horizontal overflow. Figure lightbox opens and closes with Escape. Inspected light/dark screenshots.
