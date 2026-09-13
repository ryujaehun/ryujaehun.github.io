# Series review: tiny-vllm (10 chapters, ko/en)

Review scope: `guide.md`, `series.yaml`, `briefs.md`, `evidence/claims.md`,
`evidence/trace.md`, and `articles/ko-01..10.md` / `articles/en-01..10.md`.
Checked coverage and ordering, terminology, cross-chapter contradictions, claim
ownership and boundaries, Korean-English pairing, and citation consistency.

## Decision

REVISE

## Coverage

- All ten chapters exist in both Korean (`ko-01`…`ko-10`) and English
  (`en-01`…`en-10`) in `series.yaml` order 1–10; none is missing, out of order,
  or duplicated.
- Ordering follows the `guide.md:24-29` arc: runnable program and build (1),
  weight format (2), token→logit forward path (3, 4), attention/KV cache (5),
  prefill/decode structure (6), batching and scheduling (7, 8), paged attention
  kernel (9 = online softmax, 10 = traversal and data structures). Chapter 9
  precedes 10 inside the final paged-attention pair, matching
  `briefs.md:194-209` and `:213-233`.
- Each chapter's scope matches `series.yaml` and the briefs' Scope/Claims/
  Exclusions: build system (1), loader and `json.hpp` (2), tokenizer/embedding
  (3), transformer block (4), attention/KV cache (5), prefill/decode bottlenecks
  (6), static batching (7), continuous batching + block-table resync (8), online
  softmax (9), paged KV cache and paged attention (10).
- Claim ownership follows the briefs: Claim 1 (ch2); Claims 2–3 + Claim 7 ref
  (ch4); Claims 5–7 (ch5); Claims 3–4 + Claims 5–6 refs (ch6); Claim 9 (ch7);
  Claims 9–10 + Claim 6 ref (ch8); Claim 8 softmax/warp + Claim 2 limitation
  (ch9); Claim 8 kernel + Claims 5–7, 10 (ch10). Each chapter also carries its
  briefed limitations.
- Visuals from every brief are present as diagrams/tables in the corresponding
  chapters: main() flow + constants table (1); safetensors layout + pointer table
  (2); gather flow + comparison table (3); layer pipeline + transpose-trick
  diagram (4); block_table/free_blocks + GQA diagrams (5); side-by-side sequence
  + step table (6); slot state table + timeline (7); iteration timeline +
  sync-point diagram (8); online-softmax update + warp tree-sum diagrams (9);
  block pool + kernel flow + causal-masking diagrams (10).

## Cross-chapter consistency

- Constants agree wherever they recur: `N_LAYERS=16`, `EMBEDDING_LENGTH=2048`,
  `HIDDEN_DIM=8192`, `KV_DIM=512`, `HEAD_DIM=64`, `NUM_Q_HEADS=32`,
  `NUM_K_HEADS=NUM_V_HEADS=8`, `GQA_Q_TO_K_RATIO=4`, `VOCAB_SIZE=128256`,
  `MAX_SEQ_LEN=2048`, `BATCH_SIZE=2`, `MAX_PROMPT_LEN=512`, `BLOCK_SIZE=16`,
  `V_OFFSET=16384`, `BLOCK_BYTES=32768`, `KV_CACHE_SIZE_BYTES=2GiB`,
  `MAX_BLOCKS_PER_SEQ=128`, `NUM_BLOCKS=65536`. The derivations
  `2GiB/32768=65536` and `2048/16=128` are computed identically in ch1, 5, 10.
- Prompt queue lengths 17·14·13·14 (ch1, 3, 7) and the "prompts ≤ 17 tokens so
  the 1024-thread guards never trigger" statement (ch3, 4, 6) agree.
- Decode-loop exit (`queue.empty() && num_active_slots==0` → break, else
  continue), the unused `MAX_NEW_TOKENS_GENERATED`, and EOT IDs 128001/128009
  are stated identically in ch1, 7, 8.
- GQA 4:1 sharing (`k_head_idx = i/4` at `src/main.cpp:303`,
  `v_head_idx = i/4` at `:346`, `kv_head_idx = q_head_id/4` at
  `src/kernels.cu:468`) is consistent across ch4, 5, 10.
- `pagedAttentionKernel` launch (grid `(num_active_slots, NUM_Q_HEADS)`, block
  `HEAD_DIM=64`, `src/kernels.cu:525-527`), the `gpu_active_slots` mapping,
  `dot_products[2]` + `__shfl_down_sync` offsets, `acc/d` output, and "masking =
  not reading unwritten tokens" via `num_blocks`/`tokens_in_block` agree across
  ch8, 9, 10. `WARP_FULL_MASK` (`src/cuda_to_hip.h:50` HIP 64-bit, `:59` CUDA
  32-bit) is consistent in ch1, 9, 10.
- KV scatter (prefill block-wise `:251-288`, decode token-wise `:851-873`) and
  the GEMM widths (`n=prompt_len` vs `n=num_active_slots`) agree across ch4, 5,
  6, 10.
- Terminology is uniform: 산포/scatter vs gather, `block_table`/`block_table_gpu`/
  `free_blocks`, online softmax/온라인 softmax, and the ch10 phrase "최대
  BLOCK_SIZE 토큰 청크(마지막은 일부)" all match their ko/en counterparts.
- Korean–English pairs are faithful section-by-section translations with matching
  citations and figures; the only pairing deviation is the en-08 cross-reference
  (Problems item 1).

## Problems

1. **en-08 cross-references the Korean article.** `en-08.md:81` cites
   `` `ko-07.md:140-141` `` for the batch-width claim, while `ko-08.md:78`
   correctly cites `ko-07.md:140-141`. The English chapter should point to its
   English counterpart `en-07.md:140-141` (equivalent content at those lines).
2. **`softmaxDecode` range differs between ch1 and ch9.** `ko-01.md:191-194` /
   `en-01.md:215-217` give the wrapper as `src/kernels.cu:442-458` (guard
   `:444-448`), while `ko-09.md:180-182` / `en-09.md:188-190` give
   `` `src/kernels.cu:408-458` `` for the same symbol. `trace.md:258-259`
   distinguishes wrapper `:442-458` from kernel `:408-439`; trace.md:217 itself
   conflates the two, which is the likely origin.
3. **Minor boundary overlap in ch9.** `ko-09.md:142-144` / `en-09.md:147-149`
   state the grid `(num_active_slots, NUM_Q_HEADS)` and block `HEAD_DIM=64`
   (`src/kernels.cu:525-527`), which `briefs.md:213-233` assigns to ch10, and the
   same chapter's "이 장에서 다루지 않는 것" lists grid/block placement as
   ch10's. The mention is needed for ch9's `dot_products[2]`/two-warp argument
   and is framed as call-site context, so it is not a contradiction — but the
   boundary is softer than the briefs intend.
4. **Minor evidence gap in ch4.** `ko-04.md:190-191` / `en-04.md:209-210` state
   the prefill argmax "casts bf16 to float" (`:533-543`), but `claims.md:47`
   documents the bf16→float cast only for the decode argmax (`:1003-1010`); no
   attached evidence attests the cast for prefill.

## Required fixes

1. In `en-08.md:81`, change `` `ko-07.md:140-141` `` → `` `en-07.md:140-141` ``.
2. Align the `softmaxDecode` range: in both `ko-09.md` and `en-09.md`, cite the
   wrapper as `src/kernels.cu:442-458` (kernel `softmaxKernelDecode` as
   `:408-439`), mirroring `trace.md:258-259`, so ch1 and ch9 agree.
3. (Recommended) In `ko-09.md`/`en-09.md`, explicitly defer the grid/block
   placement detail to ch10 at the call-site mention, matching
   `briefs.md:213-233`.
4. (Recommended) In `ko-04.md`/`en-04.md`, either remove "casts bf16 to float"
   from the prefill-argmax sentence or add a supporting citation; otherwise keep
   only the source-attested fact that argmax runs on the CPU after the D2H copy
   (`:529`, `:533-543`).