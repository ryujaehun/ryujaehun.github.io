# Teola source review

Reviewed `2024-11-01-paper-teola.md` against [2407.00326v1](https://arxiv.org/html/2407.00326v1), especially Table 1, the four graph passes, Algorithm 2, implementation and evaluation settings.

## Corrections

The prior generated Q&A blurred partial decoding with arbitrary partial-sentence forwarding and parallel token generation. The rewrite preserves autoregressive dependency within one call and explains overlap between downstream work and continued generation. It requires splittable output and a structured parser extracting complete pieces. A Mermaid shows sequential expanded-query generation feeding overlapping embedding/search branches.

Partial prefilling is limited by causal prefix dependencies. Full Prefilling in the paper handles the remaining prompt after partial prefilling, rather than requiring the prefix to be recomputed. The article distinguishes per-query graph construction from a completely static graph, and Condition primitives from support for arbitrary dynamic loops.

Explained the four optimization passes, profile-limited stage decomposition, two scheduling tiers, reverse-topological depth and query-bucket arrival ordering. Depth is a heuristic, not an exact execution-time critical-path oracle or proof of globally optimal batching.

Teola uses a modified vLLM backend. Its evaluated baselines are the authors' LlamaDist and LlamaDistPC, each under PO/TO scheduling, using the same engines. The reported maximum gains are not a direct universal speedup over standalone or current vLLM. Added workload, per-instance GPU and resource-allocation conditions for the paper's application latency measurements, and stated the backend-coupling/dynamic-graph limitations.

## Verification

Independently checked the illustrative ideal schedule: `800+600+250+150+400 = 2200ms`; overlapping the first three independent tasks gives `max(800,600,250)+150+400 = 1350ms`, or approximately 1.63×. This example explicitly assumes no resource contention and does not reproduce the paper's 2.09× result.

Independent source review found no significant factual or mathematical issues. All 688 posts pass publication lint. Original title, date, tags, categories and route are retained; lastmod and description are added. There is no English companion or automation change, and no model generation or notification is invoked.

Production Hugo 0.165.0 build, mobile/desktop browser checks with the former KaTeX CDN blocked, full-site link/math/copy/Mermaid gates and deployment CI are verified before reporting deployment. Repository-wide source review remains ongoing.
