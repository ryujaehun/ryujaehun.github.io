# Attention Drop source review

Reviewed `2024-11-01-paper-Not_All_Attention_is_Needed.md` against [2406.15786v1](https://arxiv.org/html/2406.15786v1), especially Eqs. 3–6, Algorithm 1, Tables 1/3 and the calibration analyses.

## Corrections

- Clarified the residual-inclusive hidden-state boundary for cosine scoring and removal of the associated LayerNorm. Comparing the raw attention output to its input is a different metric. Explained static one-shot scoring and Joint Drop's pooled attention/MLP score list, rather than adding two scores into a block score.
- Replaced contradictory claims of guaranteed calibration robustness and certain overfitting with the actual scope of the experiments: 256 C4 validation samples of length 2,048 by default, sample-count and dataset-type ablations, and remaining uncertainty for deployment distributions. Kept the author's original distribution-shift concern as a clearly labeled application question.
- Added Table 3's model-specific results, identifying its four-task average. Half attention removal lowers Llama-3-8B's reported average from 67.6 to 55.4, versus 75.1 to 73.3 for Llama-3-70B; comparable performance is not universal losslessness.
- Separated Table 1's Mistral speed/memory measurements from module-removal fractions and theoretical KV payload ratios. Included the specialized-math/output-format caveat and the paper's stated limitations.
- Removed the stray word “Fusion” from the title. An explicit slug retains the existing published `/posts/what-matters-in-transformers-not-all-attention-is-needed-fusion/` route. Date, categories and tags are unchanged; lastmod and description are added. There is no English companion.

## Verification

Independently computed the illustrative residual score `1 - 1/sqrt(1.01) = 0.00496280979` and the 12.2-point Table 3 difference. The example is labeled as illustrative and includes a same-direction/different-magnitude counterexample to distinguish cosine similarity from a guarantee of equivalent behavior.

Independent source review found no significant factual or mathematical issues. Publication lint passed for all 686 posts. Production Hugo 0.165.0 builds the corrected title at the original route. Browser checks cover 390px and 1440px, canonical URL, description, updated content, math and Mermaid rendering, and horizontal overflow with the former KaTeX CDN blocked.

Full-site local-reference/math/copy/Mermaid gates and deployment CI are verified before reporting this change deployed. Automation logic is unchanged; no model generation or notification is invoked. Repository-wide source review remains ongoing.
