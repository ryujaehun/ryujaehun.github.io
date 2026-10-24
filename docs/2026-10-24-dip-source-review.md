# DIP article source and arithmetic review

Reviewed `content/posts/2025-03-12-paper-2412.01380.md` against [arXiv v2](https://arxiv.org/html/2412.01380v2), explicitly identified in the revised article. Publication date and existing URL remain unchanged; lastmod records the correction.

## Confirmed problems

- The article labeled `2048 × 4096 × 4096` as 34.6 TFLOPs, then called a hypothetical two-dimension reduction 8.6 TFLOPs. The product is 34,359,738,368 MAC, approximately 34.36 GMAC or 68.72 GFLOP under the two-operations-per-MAC convention. A single projection is not the full Mistral model, and selecting half the input columns does not halve its output dimension.
- The article supplied GPU timings, VRAM figures, development-day estimates and cost savings without source evidence. It also presented LoRA fine-tuning as mandatory despite §4 and Table 1 reporting DIP without adapters.
- The headline 46% memory reduction and 40% throughput increase lacked their baseline and simulator scope. §6.1 models Flash/DRAM transfers for Apple A18 specifications (60 GB/s DRAM, 1 GB/s Flash) without simulating NPU compute time. The representative Phi-3-Medium result is relative to dense Flash streaming, with under 0.1 perplexity loss (§6.2/abstract).

## Revised article

Replaced repetitive generated Q&A and unsupported numerical tables with a compact source-backed description of DIP input/GLU column masks (§4, equations 7–9), optional LoRA, and DIP-CA's MLP weight cache (§5.2, equation 10, Algorithm 1). Explicitly distinguishes that cache from KV caching.

Added an independent worked example: x=[0.1,0.5,-0.2,0.8,-0.9], cached columns=[1,3,5], gamma=0.2, K=3. Magnitude selects [5,4,2], cache reweighting [5,3,4]. Missing columns decline from two to one. A hypothetical eviction of unused column 1 yields cache [3,4,5]. For a weight row of all ones, dense/magnitude/cache-aware inner products are 0.3/0.4/-0.3: less transfer does not establish lower approximation error. Decimal arithmetic verified all selections and sums.

The matrix-work example is explicitly illustrative, identifies MAC/FLOP conventions, and preserves the output dimension when halving input columns. It does not predict end-to-end speed or engineering effort.

Added the authors' Figure 1 from [the official SVG](https://arxiv.org/html/2412.01380v2/figure_1.svg), with attribution and CC BY 4.0 link. Added only a white background rectangle for legibility on dark pages; the caption declares this modification.

## Validation

- Independent content review found no significant factual issues; numerical and cache examples checked separately.
- Article lint and minified Hugo build passed.
- Whole build: 30,460 math nodes, 2,895 copy sources and 162 Mermaid diagrams rendered without failures. All 2,158 HTML files passed local-link/image checks.
- Mobile/desktop and light/dark screenshots checked for the revised article; correct source link, two tables, 16 rendered math nodes, loaded local SVG, no KaTeX errors or horizontal page overflow.
