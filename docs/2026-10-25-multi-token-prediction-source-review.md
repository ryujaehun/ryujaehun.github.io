# Multi-token Prediction source review — 2026-10-25

Source: https://arxiv.org/html/2404.19737v1 (sections 2–3 and appendices A, C, D, E).

Reviewed `content/posts/2024-10-31-paper-multi-token-prediction.md`; retained the original title/date and published route.

Corrections:
- Explained independent Transformer heads, shared trunk/unembedding, and a position-wise auxiliary loss; candidates require verification during greedy self-speculative decoding.
- Removed unsupported 4× whole-model FLOP/activation claims, 120–150 GB Llama 2 estimates, and DDR 2×/quarter-bandwidth predictions. Distinguished storage from traffic and the limited logit-memory complexity from total VRAM.
- Added matched-parameter comparisons and measured Table S5 overhead (6.7B: 1.07×; 13B: 1.09×), rather than claiming all implementations have zero cost.
- Scoped Table S2 decoding speeds to its models, workload, greedy mode and batch conditions; separated accepted tokens from measured speedup.
- Replaced speculative small-model failure explanations with observed results and explicitly identified hypotheses. Included the limited Llama 2 finetuning results.
- Reported Table S7 pass@1 changes as percentage points and relative percentages. Independently checked HumanEval 12.1% and MBPP 17.3%; illustrative speed formula gives 2× or 1× under its stated assumptions.

Validation: publication inventory/lint passed for 688 posts; independent source review found no significant actionable issues. Hugo 0.165.0 production build passed with no path warnings; 30,561 formulas and 2,878 code sources passed, as did 171 Mermaid diagrams. Targeted Chrome checks passed at 390/1440 px with the former KaTeX CDN blocked: 19 rendered formulas, canonical URL and description, revised text, and no horizontal overflow. Published route: `/posts/better-faster-large-language-models-via-multi-token-prediction/`.

Local link/image validation: 2,167 HTML files, zero broken targets.
