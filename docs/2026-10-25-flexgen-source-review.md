# FlexGen source review

Reviewed `2024-11-01-paper-FlexGen.md` against [2303.06865v2](https://arxiv.org/html/2303.06865v2). The article now explicitly identifies the revision.

## Corrections

The old generated Q&A described blocks as groups of several layers and minimized total latency without dividing by batch size. The rewrite explains same-layer weight reuse across GPU batches, effective batch size, enumeration of batch configurations and placement LP objective `T/B`. It distinguishes automatic policy search from manual corrections needed for relaxation, memory fragmentation and model inaccuracies.

Separated CPU memory storage from CPU attention delegation. The source disables CPU delegation when fine-grained quantization is enabled because CPU compression/decompression is costly. The 4-bit weight/KV format is converted back to FP16 before computation; it is not a claim of full INT4 execution or universal lossless compression.

Replaced broad DeepSpeed/ZeRO-Offload realtime and multi-GPU assertions with the actual paper baselines, ZeRO-Inference and Accelerate, limited to the evaluated settings. The single-GPU result includes substantial CPU RAM and SSD resources, and the compressed policy can avoid disk swapping. Throughput is aggregate generated tokens divided by prefill plus decode time, not per-request streaming latency.

Added Table 2's OPT-175B prompt-512/output-32 values and effective batch 144 for compressed FlexGen. The text states that throughput measurements use dummy weights, quality evaluations use real weights, and some slow-system results are extrapolated. Table 5 distinguishes Lambada accuracy and WikiText perplexity; the OPT-6.7B prompt-1024 row shows compression is not universally faster.

## Verification

Independent arithmetic confirms the illustrative one-step weight-read example, four batches × three 2GB layers = 24GB versus 6GB with reuse, excluding other transfers and compute. The throughput example gives `8*32/4 = 64` versus `32*32/8 = 128` tokens/s, demonstrating why longer total latency can accompany higher throughput. Neither example is a measured paper result.

Independent source review found no significant factual or mathematical issues. Original title, publication date, tags, categories and route are retained; lastmod and description are added. All 688 posts pass publication lint. No English companion or automation logic was changed, and no model generation or notification was invoked.

Production Hugo 0.165.0 build, mobile/desktop browser checks with the former KaTeX CDN blocked, full-site link/math/copy/Mermaid gates and deployment CI are verified before reporting this change deployed. Repository-wide source review remains ongoing.
