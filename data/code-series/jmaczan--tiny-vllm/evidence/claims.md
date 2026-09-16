# tiny-vLLM Evidence Claims

이 문서는 첨부된 여섯 파일(`guide.md`, `series.yaml`, `evidence/recon.md`,
`evidence/trace.md`, `src/main.cpp`, `src/kernels.cu`)만을 근거로 고정 revision
`e25bf1994efa90bc98b721ba7c527402f86fbeaf` 에서 수집한 주장이다. 모든 편의
`required_evidence.runtime` 은 0 이며(`guide.md:85`), 실행 결과 파일과
`runtime-claims.json` 이 존재하지 않아 보존해야 할 runtime claim ID·execution ID
는 없다. 커널 실행 시간·처리량·메모리 사용량은 측정하지 않는다(결정 003,
`guide.md:75-87`).

## Claim 1

**ID:** product-translation-units

**Claim:** 제품 바이너리 `tiny-vllm` 은 `src/main.cpp` 와 `src/kernels.cu` 두
번역 단위로만 빌드된다. `include/json.hpp` 는 제품 코드가 아니라 include 경로로
들어오는 단일 헤더 의존이다.

**Confidence:** High

**Evidence:** `trace.md:17-22` 가 `CMakeLists.txt:49-52` 의
`add_executable(tiny-vllm src/main.cpp src/kernels.cu)` 와 `CMakeLists.txt:62-63`
의 include 경로(`src`, `include`)를 인용한다. `src/main.cpp:4-8` 이
`cuda_to_hip.h`·`json.hpp`·`kernels.cuh` 를 include 하고, `src/main.cpp:10` 이
`nlohmann::json` 을 사용한다.

**Limitation:** `CMakeLists.txt` 본문은 첨부에 없어 위 사실은 `trace.md` 의
인용으로만 접근한다(`trace.md:216-218`). `include/json.hpp` 는 제3자 단일 헤더
(`JSON for Modern C++ 3.12.0`, MIT)로 제품 소스에서 제외된다(`recon.md:24-26`).

## Claim 2

**ID:** dual-backend-branch

**Claim:** `USE_HIP` 옵션이 같은 CUDA 소스를 hipcc/hipBLAS 경로로 돌리며,
`src/cuda_to_hip.h` 가 bfloat16 타입과 BLAS 호출의 CUDA/HIP 차이를 흡수하는
shim 이다.

**Confidence:** Medium

**Evidence:** `trace.md:19` 가 `CMakeLists.txt:10-16` 의 `USE_HIP` 언어 분기를
인용한다. `src/main.cpp:4` 와 `src/kernels.cu:1` 이 `cuda_to_hip.h` 를 첫
include 로 끌어오고, 제품 전역에서 `__nv_bfloat16` 을 타입으로 쓴다
(`src/main.cpp:66-76`, `src/kernels.cu:32-40`).

**Limitation:** `CMakeLists.txt` 와 `src/cuda_to_hip.h` 본문은 첨부에 없어 분기의
구체적 코드는 검증되지 않았다. `recon.md:33-36` 과 `guide.md:94-101` 의 서술에
의존한다.

## Claim 3

**ID:** weights-safetensors-load

**Claim:** `loadWeights` 는 `model.safetensors` 를 열어 헤더 JSON 의
`data_offsets` 로 각 텐서의 경계와 `max_offset` 을 산정하고, 가중치 바이트 전체를
한 번의 `cudaMalloc` 으로 잡은 뒤 CPU 버퍼를 거쳐 H2D 로 복사한다. 이후 텐서
이름→오프셋으로 `Weights` 구조체의 포인터를 채운다.

**Confidence:** High

**Evidence:** `src/main.cpp:87-93`(이진 오픈), `src/main.cpp:96-101`(헤더 크기·
헤더 읽기), `src/main.cpp:103-118`(`data_offsets`·`max_offset`),
`src/main.cpp:120-127`(`cudaMalloc`·H2D 복사), `src/main.cpp:132-145`(포인터
매핑).

**Limitation:** GPU 가 없어 실제 파일 적재는 실행되지 않았다(`guide.md:75-87`).
가중치 파일은 gated 모델 `meta-llama/Llama-3.2-1B-Instruct` 여서 확보도 제약이다
(`recon.md:91-92`).

## Claim 4

**ID:** buffer-allocation-table

**Claim:** prefill 어텐션 점수 버퍼 `prefill_attn_scores` 는
`MAX_PROMPT_LEN(512) × MAX_PROMPT_LEN × NUM_Q_HEADS(32)` 크기로 잡힌다. decode
전용 버퍼(`gpu_last_tokens`, `k_proj_batched_buffer`, `v_proj_batched_buffer`)와
K/V 투영 임시 버퍼(`k_proj_temp_buf`, `v_proj_temp_buf`)는 별도로 할당되어
재사용된다.

**Confidence:** High

**Evidence:** `src/main.cpp:647-648`(`prefill_attn_scores`),
`src/main.cpp:683-693`(decode 전용 3개), `src/main.cpp:635-639`(K/V 임시
버퍼). 크기 상수는 `src/main.cpp:19-31`.

**Limitation:** 크기는 컴파일 타임 상수에 의존하며 실행 시 실제 할당은 검증되지
않았다.

## Claim 5

**ID:** buffer-aliasing

**Claim:** `buf_2048_1` 은 q 프로젝션과 어텐션 점수×V 출력이 공유하고,
`buf_2048_2` 는 O 프로젝션과 down 프로젝션이 공유한다. 같은 시점에는 한 역할만
쓰는 버퍼 별칭 설계다.

**Confidence:** High

**Evidence:** `src/main.cpp:628-629`(주석 "shared between q_proj and
attn_scores_v"), `src/main.cpp:656-657`(주석 "shared between o_proj and down"),
`src/main.cpp:177`·`343`(전자 사용), `src/main.cpp:377`·`470`(후자 사용).

**Limitation:** in-place 재사용의 정합성은 소스 인용으로만 뒷받침된다
(`trace.md:184-189`).

## Claim 6

**ID:** prefill-kernel-sequence

**Claim:** 한 번의 `prefill` 호출이 프롬프트의 모든 토큰을 함께 처리하며,
레이어마다 embeddingGather → rmsNorm → Q/K/V 투영 → RoPE → K/V 블록 분산 →
헤드별 어텐션 점수 → causalMask → softmax → 점수×V → O 투영 → residual →
post-attn rmsNorm → SwiGLU(gate/up/silu/down) → residual 순으로 커널과 cuBLAS
를 호출한다.

**Confidence:** High

**Evidence:** `src/main.cpp:157-158`(embeddingGather), `src/main.cpp:166`
(rmsNorm), `src/main.cpp:178-240`(Q/K/V 투영), `src/main.cpp:244-245`(RoPE),
`src/main.cpp:251-288`(K/V 분산), `src/main.cpp:301-327`(점수),
`src/main.cpp:329`(causalMask), `src/main.cpp:331`(softmax),
`src/main.cpp:343-396`(점수×V, O 투영), `src/main.cpp:399`(residual),
`src/main.cpp:401`(post-attn rmsNorm), `src/main.cpp:414-489`(SwiGLU),
`src/main.cpp:492`(residual). 커널 구현은 `src/kernels.cu:32-344`.

**Limitation:** 커널 호출 순서는 소스 순서 인용이며 실행 검증은 없다
(`guide.md:75-87`).

## Claim 7

**ID:** rmsnorm-reduction

**Claim:** `rmsNormKernel` 은 블록 내 1024 스레드가 shared memory
(`rms_vector[1024]`)에 요소 제곱을 누적하고, 트리 리덕션으로 합을 구한 뒤
스레드 0 이 `sqrt(합/2048 + 1e-5)` 로 rms 를 계산해 정규화에 쓴다.

**Confidence:** High

**Evidence:** `src/kernels.cu:55-81`(`rmsNormKernel`), 특히 `src/kernels.cu:57`
(shared), `src/kernels.cu:61`(제곱 합), `src/kernels.cu:62-71`(트리 리덕션),
`src/kernels.cu:72-75`(`sqrt`), `src/kernels.cu:78-79`(정규화·가중치 곱). 런치는
`src/kernels.cu:84-86`.

**Limitation:** 수치 교차검증 대상 `python/rms_norm_crosscheck.txt` 는 저장소에
커밋돼 있으나 첨부에 없고 실행도 없다(`recon.md:42-44`).

## Claim 8

**ID:** prefill-gqa-attention

**Claim:** prefill 어텐션은 Q 헤드 32개 각각에 대해 K 헤드 인덱스 `i/4`(GQA)를
쓰고, `attn_alpha = 1/sqrt(64)` 로 스케일된 `Q_head·K_head^T` 점수를 cuBLAS 로
계산한다. causalMask 로 미래 마스킹한 뒤 행별 softmax 가 적용된다.

**Confidence:** High

**Evidence:** `src/main.cpp:301-327`(헤드별 cuBLAS, `src/main.cpp:303`
`k_head_idx = i / GQA_Q_TO_K_RATIO`), `src/main.cpp:329`(causalMask),
`src/main.cpp:331`(softmax), `src/main.cpp:649-650`(`attn_alpha = 1.0f / 8.0f`).
상수는 `src/main.cpp:19-24`, 커널은 `src/kernels.cu:224-309`.

**Limitation:** 참조 출력으로 선언된 `python/reference.py` 는 실행되지 않았고
수치 일치 검증도 없다(`recon.md:37-42`).

## Claim 9

**ID:** column-row-major-trick

**Claim:** 가중치·활성은 행 우선(row-major)인데 cuBLAS 는 열 우선을 가정하므로,
코드는 데이터를 그대로 두고 전치 플래그(`CUBLAS_OP_T`)와 lda/ldb/ldc 만으로
행렬곱을 호출해 결과를 전치된 채로 해석한다. 이 트릭은 소스 주석에 명시돼 있다.

**Confidence:** High

**Evidence:** `src/main.cpp:168-196`(Q 투영, `src/main.cpp:179-180`
`CUBLAS_OP_T`/`CUBLAS_OP_N`, `src/main.cpp:184-194` lda·ldb·ldc), 동일 패턴
`src/main.cpp:201-219`(K), `src/main.cpp:378-396`(O),
`src/main.cpp:414-432`(gate), `src/main.cpp:509-527`(로그잇).

**Limitation:** `series.yaml` 의 시각 자료 `column-row-major-trick` 필수 주장 중
"alpha 와 beta 인자가 잔차 누적을 별도 커널 없이 처리한다"는 고정 revision 의
코드로는 뒷받침되지 않는다. 모든 beta 는 0.0 이고(예: `src/main.cpp:631-632`),
잔차는 별도 커널 `residualAdd` 가 담당한다(`src/main.cpp:399`·`492`,
`src/kernels.cu:311-329`). 유일한 비-1.0 alpha 는 어텐션 스케일 `1/8` 이다
(`src/main.cpp:649`).

## Claim 10

**ID:** prefill-decode-contrast

**Claim:** decode 계열 커널은 시퀀스 길이 1(슬롯당 토큰 1개)을 전제로 인덱싱을
단순화한다. `embeddingGatherKernelDecode` 는 `blockIdx.x` 를 슬롯으로 삼고,
`ropeKernelDecode` 는 단일 블록으로 현재 위치 하나만 회전하며,
`softmaxKernelDecode` 는 `MAX_SEQ_LEN` 스트라이드 레이아웃을 가정한다.

**Confidence:** High

**Evidence:** `src/kernels.cu:347-369`(`embeddingGatherKernelDecode`,
`src/kernels.cu:349` `blockIdx.x`), `src/kernels.cu:371-405`
(`ropeKernelDecode`, `src/kernels.cu:397` `<<<1, num_threads>>>`),
`src/kernels.cu:408-458`(`softmaxKernelDecode`, `src/kernels.cu:413`
`blockIdx.x * MAX_SEQ_LEN`). 대비되는 prefill 커널은 `src/kernels.cu:32-344`.

**Limitation:** `softmaxKernelDecode` 의 `MAX_SEQ_LEN` 스트라이드
(`src/kernels.cu:413`)는 prefill softmax 의 `num_tokens` 스트라이드
(`src/kernels.cu:262`)와 다른 레이아웃 가정이며 실행 검증되지 않았다
(`trace.md:229-231`).

## Claim 11

**ID:** rope-tables-recompute

**Claim:** prefill 의 RoPE 는 `init_rope_frequencies` 가 사전 계산한
`d_cos_table`/`d_sin_table` 을 커널이 참조한다. 반면 `ropeKernelDecode` 는
호출마다 `theta = 1/pow(500000, 2i/64)` 를 다시 계산하고 상수 `500000.0` 과
`32` 를 하드코딩한다.

**Confidence:** High

**Evidence:** `src/kernels.cu:96-152`(테이블 생성·H2D), `src/kernels.cu:173-200`
(테이블 참조), `src/kernels.cu:371-384`(재계산, `src/kernels.cu:377-378` 상수),
`src/main.cpp:572`(초기화 호출).

**Limitation:** decode 계열의 재계산은 TODO 로 남아 있고
(`trace.md:231-233`), 두 경로의 수치 일치는 실행 검증되지 않았다.

## Claim 12

**ID:** decode-kv-scatter

**Claim:** decode 는 슬롯별 K/V 투영을 `k_proj_batched_buffer`/
`v_proj_batched_buffer` 에 모은 뒤, `token_in_block_idx == 0` 일 때만
`free_blocks` 에서 새 물리 블록을 할당하고 그 외에는 기존 블록의 위치에
`cudaMemcpy`(D2D)로 기록한다.

**Confidence:** High

**Evidence:** `src/main.cpp:851-873`, 특히 `src/main.cpp:856-865`(블록 인덱스·
할당 분기), `src/main.cpp:866-872`(캐시 기록). 배치 K/V 투영은
`src/main.cpp:802-842`.

**Limitation:** 매 레이어마다 `block_table` 전체를 H2D 동기화하므로
(`src/main.cpp:876`, TODO `src/main.cpp:551`) 이 비용은 측정되지 않았다
(`trace.md:234-235`).

## Claim 13

**ID:** decode-termination

**Claim:** decode 루프는 생성 토큰이 `<|end_of_text|>`(128001) 또는
`<|eot_id|>`(128009) 이거나 `current_prompt_len == MAX_SEQ_LEN-1` 이면 슬롯을
해제하고, 해당 슬롯이 소유한 모든 블록을 `free_blocks` 로 반납하며 `block_table`
을 `-1` 로 재초기화한다.

**Confidence:** High

**Evidence:** `src/main.cpp:1015-1031`(종료 분기), 반납 루프
`src/main.cpp:1018-1029`, 동기화 `src/main.cpp:1030`. 상수는 `src/main.cpp:26-28`.

**Limitation:** `MAX_NEW_TOKENS_GENERATED = 20`(`src/main.cpp:12`)은 선언만 있고
참조되지 않아 종료 조건으로 동작하지 않는다(`trace.md:222-224`).

## Claim 14

**ID:** block-table-indexing

**Claim:** `pagedAttentionKernel` 은 `block_table_gpu` 를 거쳐 논리 블록을 물리
블록으로 바꾸고, `kv_cache` 의 비연속 블록을 순회하며 읽는다. 블록 할당은
`free_blocks` 의 `pop_back`, 반납은 `push_back` 으로 관리된다.

**Confidence:** High

**Evidence:** `src/kernels.cu:478-485`(블록 순회·주소 계산),
`src/kernels.cu:480`(block_table_gpu 참조). 할당·반납은 `src/main.cpp:264-277`
(prefill), `src/main.cpp:859-865`(decode), `src/main.cpp:1015-1031`(반납).
초기화는 `src/main.cpp:575-581`.

**Limitation:** 논리 인덱스 산술(`slot*N_LAYERS*MAX_BLOCKS_PER_SEQ +
layer*MAX_BLOCKS_PER_SEQ + block_idx`)의 실행 정합성은 GPU 없이 검증되지
않았다.

## Claim 15

**ID:** kv-block-layout

**Claim:** KV cache 의 블록 하나는 `BLOCK_SIZE(16) × KV_DIM(512)` bf16 요소를
담으며, K 는 블록 시작에, V 는 `V_OFFSET = 16×512×2` 바이트 뒤에 배치된다.
블록당 `BLOCK_BYTES = V_OFFSET×2` 여서 2GB 캐시는 65536 블록으로 나뉜다.

**Confidence:** High

**Evidence:** `src/main.cpp:33-37`(상수), `src/kernels.cu:17-19`(동일 상수),
`src/kernels.cu:484-485`(K·V 주소 계산), `src/main.cpp:280-287`·`866-872`(기록).

**Limitation:** 블록 크기·오프셋은 상수이며 실행 시 레이아웃 검증은 없다.

## Claim 16

**ID:** paged-attention-inputs

**Claim:** `pagedAttentionKernel` 은 `kv_cache`, `block_table_gpu`,
`gpu_seq_lens`, `gpu_active_slots` 를 함께 받는다. `gpu_active_slots` 로 활성
슬롯을, `gpu_seq_lens` 로 각 슬롯의 현재 길이를 얻으며, 이 커널만 이 네 상태를
함께 쓴다.

**Confidence:** High

**Evidence:** `src/kernels.cu:461`(시그니처), `src/kernels.cu:464-471`(활성 슬롯·
길이 소비), `src/kernels.cu:480`(block_table 참조). 호출부 `src/main.cpp:878`,
인자 준비 `src/main.cpp:749-757`.

**Limitation:** decode 계열에서 이 커널만 네 상태를 함께 받는 비대칭은
`guide.md:107-110` 의 서술과 일치하나, 다른 커널의 시그니처는 첨부에 없어 직접
비교되지 않았다.

## Claim 17

**ID:** pagedattention-online-softmax

**Claim:** `pagedAttentionKernel` 은 워프 내 `__shfl_down_sync` 5단계 리덕션으로
32개 스레드(1 워프)의 내적을 모으고, 스레드 0 과 스레드 32 가 `dot_products[2]`
로 두 워프의 합을 더해 `sqrt(HEAD_DIM)` 으로 나눈다. 이후 온라인 소프트맥스
(FlashAttention 방식)로 가중 평균을 누적한다.

**Confidence:** High

**Evidence:** `src/kernels.cu:463`(shared `dot_products[2]`),
`src/kernels.cu:486-507`(내적·워프 리덕션·두 워프 결합), `src/kernels.cu:510-519`
(온라인 softmax), `src/kernels.cu:522`(출력 기록).

**Limitation:** `dot_products[2]` 와 스레드 0·32 협력은 `HEAD_DIM=64` 전제이며
그 외 크기에서는 성립하지 않는다(`trace.md:225-227`). 실행 검증은 없다.

## Claim 18

**ID:** pagedattention-inplace

**Claim:** `pagedAttentionKernel` 의 출력은 입력 q 프로젝션과 같은 버퍼
(`buf_2048_1`)에 쓰인다. 읽기 인덱스(`src/kernels.cu:469`)와 쓰기 인덱스
(`src/kernels.cu:522`)가 같아 in-place 가 안전하며, 이것이 `pagedAttention` 호출
결과가 곧바로 O 투영 입력으로 쓰이는 이유다.

**Confidence:** High

**Evidence:** `src/kernels.cu:469`(입력 q 읽기) 대 `src/kernels.cu:522`(출력
기록), 호출 `src/main.cpp:878`, O 투영 입력 `src/main.cpp:892`.
`trace.md:185-189` 도 이 별칭을 기록한다.

**Limitation:** in-place 안전성은 인덱스 동일성이라는 정적 주장이며 동시성·
경합의 실행 검증은 없다.

## Claim 19

**ID:** slot-lifecycle

**Claim:** `is_slot_free` 벡터가 슬롯의 점유 상태를 관리하고, decode 루프의 매
반복에서 빈 슬롯은 큐의 다음 프롬프트를 `prefill` 로 채운다. 점유 슬롯만
`active_slots`·`active_tokens` 에 모여 `gpu_active_slots` 로 커널에 전달된다.

**Confidence:** High

**Evidence:** `src/main.cpp:597`(초기화), `src/main.cpp:722-737`(슬롯 재구성·
prefill 채움), `src/main.cpp:749-750`(H2D), `src/main.cpp:878`(커널 전달).
초기 채움은 `src/main.cpp:695-708`.

**Limitation:** 슬롯 종료는 EOS/EOT 토큰 또는 `MAX_SEQ_LEN-1` 도달뿐이다
(`trace.md:209-210`).

## Claim 20

**ID:** batching-queue

**Claim:** 큐에는 채팅 템플릿 토큰 ID 로 된 프롬프트 4개가 들어 있고,
`BATCH_SIZE=2` 이므로 초기 prefill 루프가 슬롯 0·1 을 채우고 프롬프트 2·3 은
큐에 남는다. decode 루프가 슬롯 해제 시 큐의 다음 프롬프트로 채우는 흐름이
continuous batching 의 구현이다.

**Confidence:** High

**Evidence:** `src/main.cpp:583-594`(큐 4개), `src/main.cpp:29`
(`BATCH_SIZE=2`), `src/main.cpp:695-708`(초기 prefill), `src/main.cpp:724-737`
(재채움), `src/main.cpp:738-746`(빈 큐·비활성 시 break).

**Limitation:** `active_slots` 가 0 이고 큐가 비면 break 하므로, 상시 운영 서버를
의도한 주석(`src/main.cpp:720`)과 달리 실제 실행은 유한하다(`trace.md:162-164`).

## Claim 21

**ID:** static-verification

**Claim:** 이 시리즈의 모든 편은 `required_evidence.runtime: 0` 이며 커널 실행
시간·처리량·메모리 사용량은 측정하지 않는다. 커밋된 참조 출력(`reference.txt`,
`python/rms_norm_crosscheck.txt`)이 수치 교차검증의 유일한 근거다.

**Confidence:** High

**Evidence:** `guide.md:75-87`(결정 003, 정적 대체 검증), `recon.md:42-44`(참조
출력), `recon.md:87-90`(측정 부재).

**Limitation:** 실행 결과 파일과 `runtime-claims.json` 이 존재하지 않아 보존해야
할 runtime claim ID·execution ID 는 없다. GPU·CUDA 툴체인 부재로 어떤 실행
검증도 수행되지 않았으며, gated 가중치 확보는 후속 과제로 남는다
(`recon.md:91-92`).