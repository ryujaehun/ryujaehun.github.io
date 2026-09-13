# Series review — tiny-vllm: CUDA로 읽는 LLM 추론 엔진

## Decision

REVISE

The series is structurally sound: all 10 chapters exist in both Korean and English, follow
`series.yaml` ordering and scope, respect the claim ownership in `briefs.md`, and stay
consistent on every numeric and structural fact. The revision is warranted not by content
errors but by a handful of citation-range inconsistencies that cut against the series' own
"every claim cites exact line ranges" discipline (`guide.md:11-16`). Fixes are surgical and
listed below.

## Coverage

- **Chapter coverage.** All 10 chapters are present as ko-01..10 and en-01..10. Titles match
  `series.yaml` exactly (Ch7 "static batching", Ch8 "continuous batching과 scheduler", Ch9
  "online softmax CUDA kernel", Ch10 "paged KV cache와 paged attention").
- **Ordering.** The narrative follows `guide.md:24-29`: runnable program + build (Ch1) →
  weight format (Ch2) → token→logit forward path (Ch3-6) → memory/scheduling for many
  requests (Ch7-8) → paged attention as the meeting point of data structure and kernel
  (Ch9-10). Ch10 explicitly closes the arc as the series' end point.
- **File scope.** Each chapter stays within its `series.yaml` scope. README.md is never used
  as primary evidence (Ch6 explicitly defers it per `guide.md:7-9`); `python/tokenizer.py` is
  in Ch3's scope but correctly handled as "no claim backs its internals"; `test.sh` and
  `CMakeLists.txt` internals are not claimed by any chapter.
- **Claim ownership.** Matches `briefs.md`: Ch1 makes no direct claims (references Claims
  5/9/3/4 only); Ch2 owns Claim 1; Ch3 owns Claims 3/4/9 (partial); Ch4 owns Claims 2/3 with
  Claim 7 reference; Ch5 owns Claims 5/6/7; Ch6 owns Claims 3/4; Ch7 owns Claim 9 with Claim
  4 reference; Ch8 owns Claims 9/10 with Claim 6 reference; Ch9 owns Claim 8's online-softmax
  and warp-tree-sum parts; Ch10 owns Claim 8's grid/traversal part with Claims 5/6/7/10 as
  references.
- **Deferred-topic hand-offs.** Consistent everywhere: attention scores and KV scatter →
  Ch5; `pagedAttentionKernel` traversal → Ch10; online softmax details → Ch9; batching and
  the scheduler → Ch7-8; `block_table_gpu` re-sync → Ch8 (Claim 10); dead `softmaxDecode` →
  Ch1/9. No chapter duplicates another's owned content.
- **Korean-English pairing.** Every chapter has a structurally parallel ko/en pair; spot
  checks of tables, code blocks, diagrams, and citations show matched content (e.g., the
  online-softmax update block in Ch9, the slot-release table in Ch7/8, the prefill/decode
  comparison table in Ch6).
- **Terminology.** Consistent across the series: `KV 캐시`/KV cache, `논리/물리 블록`
  (logical/physical block), `산포`/scatter, `전치 트릭`/transpose trick, `온라인 softmax`/
  online softmax, `warp 트리 합`/warp tree sum, and untranslated `prefill`/`decode`/`attention`
  in both languages.

## Cross-chapter consistency

- **Constants.** Agree everywhere: `BATCH_SIZE=2`, `BLOCK_SIZE=16`, `KV_DIM=512`,
  `V_OFFSET=16384`, `BLOCK_BYTES=32768`, `NUM_BLOCKS=65536`, `MAX_BLOCKS_PER_SEQ=128`, 2GiB
  reservation, `attn_alpha=1/8` ≡ `SQRT_HEAD_DIM=8` (both `1/sqrt(64)`), EOT/EOT_ID
  128001/128009, `MAX_SEQ_LEN-1=2047`, prompt lengths 17/14/13/14, and the 1024-thread guard.
- **Facts.** No substantive contradictions found. Prefill's score GEMM reads the temp buffers
  rather than the cache (Ch5) with no conflict in Ch4/6/10; decode's paged attention reads
  only accumulated K tokens, standing in for causal masking (Ch6/9/10); GQA 4:1 appears in
  both prefill (`k_head_idx`, `v_head_idx`) and the kernel (`kv_head_idx`); `block_table`
  sync sites `:876`/`:552`/`:1030` are consistent (Ch5/8/10); the four-state slot-release
  table is identical in Ch7 and Ch8; the 16KiB full-table sync (2×16×128×4 bytes) is
  consistent in Ch8/Ch10.
- **Hand-offs.** Each chapter opens by citing the previous chapter's endpoint accurately
  (e.g., Ch8 opens on Ch7's slot release at `:1015-1031`; Ch9 opens on Ch8's sync at
  `:876`/`:878`; Ch10 opens on Ch9's softmax/traversal split). Internal cross-references
  resolve correctly (e.g., Ch8 cites `ko-07.md:140-141`, which matches Ch7's slot-bound
  batch-width paragraph).
- **Evidence rules.** Citations to claims.md/trace.md line ranges are internally consistent
  across chapters except for the items in Problems below.

## Problems

1. **Guard citation range mismatch (Ch1 vs the rest).** ko-01/en-01 cite the
   `causalMask`/`softmax` launch guard as `src/kernels.cu:241-245`, `:295-299`, while
   `claims.md`, `trace.md`, and Ch3/4/6 consistently cite `:241-244`, `:295-298`. Same code,
   two ranges inside the series.
2. **Break-branch citation range mismatch.** ko-01/en-01 cite the loop-exit `break` as
   `src/main.cpp:739-746`, while `trace.md` and Ch7/8 cite `:739-744` (with `continue` at
   `:745`). Minor, but it is the same branch cited two ways.
3. **Uncorroborated Python filename.** Ch4's not-covered section names
   `python/rms_norm.py`; Ch5 and Ch6 name `python/reference.py`. Only `python/tokenizer.py`
   is corroborated (`series.yaml:10-12`, `trace.md:69`); the briefs corroborate only
   `python/reference.py` (`briefs.md`, Ch5 exclusions). `python/rms_norm.py` appears nowhere
   else and conflicts with the other chapters' example.
4. **`series.yaml` citation-style inconsistency.** Ch1 omits the `series.yaml` scope citation
   that every other chapter includes, and Ch9 cites `series.yaml:29-30` (a 2-line slice)
   while Ch2-8 and Ch10 cite their full 3-line chapter block (e.g., `:10-12`).
5. **Minor: Ch1 constants table over-reach.** The table adds rows with claim "—"
   (`N_LAYERS`, `EMBEDDING_LENGTH`, `HIDDEN_DIM`, `HEAD_DIM`, `VOCAB_SIZE`, `MAX_SEQ_LEN`,
   `MAX_PROMPT_LEN`) beyond the briefs' "인용한 값만 담는다" visual spec (`briefs.md:41`).
   Transparently marked, so low risk.

## Required fixes

1. Change ko-01/en-01 to `src/kernels.cu:241-244`, `:295-298`, matching claims.md, trace.md,
   and Ch3/4/6.
2. Pick one break-branch range (`:739-744` per trace/Ch7/8, or `:739-746` per Ch1) and apply
   it consistently; align trace.md if Ch1's wider range is kept.
3. Verify which Python reference files exist in the pinned commit and cite only corroborated
   filenames; align Ch4's `python/rms_norm.py` with Ch5/Ch6's `python/reference.py` (or state
   both files explicitly if both exist).
4. Add the missing `series.yaml` scope citation to Ch1 and change Ch9's citation to the full
   3-line block (`series.yaml:28-30`).
5. Optional: trim or annotate the unclaimed rows in Ch1's constants table to stay within the
   briefs' "인용한 값만 담는다" spec.