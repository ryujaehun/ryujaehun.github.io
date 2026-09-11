# 증거 주장: jmaczan/tiny-vllm

고정 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61`. 첨부된 파일만 인용한다.
각 주장은 소스 경로와 줄 범위를 함께 제시하고, 코드와 테스트를 README 설명보다
우선한다 (`guide.md:11-16`).

## Claim 1: 가중치는 safetensors 오프셋을 단일 GPU 할당으로 매핑한다

- **Claim**: 로더는 `model.safetensors`를 이진으로 열어 8바이트 헤더 크기와 JSON
  헤더를 읽고, 텐서별 `data_offsets[0]`을 모아 `max_offset`을 구한 뒤 한 번의
  `cudaMalloc`으로 잡은 버퍼에 전체 가중치를 H2D 복사한다. 이후 각 텐서
  포인터를 그 단일 할당의 오프셋으로 설정한다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:87`, `src/main.cpp:96-104`, `src/main.cpp:103-118`,
  `src/main.cpp:120-127`, `src/main.cpp:132-145`
- **Limitation**: 텐서 형상·dtype 검증이 없고 `__metadata__`만 건너뛴다
  (`src/main.cpp:106-111`). 모델 경로가 `"model.safetensors"`로 하드코딩되어 있다
  (`src/main.cpp:87`).

## Claim 2: 모델 차원은 컴파일 상수로 하드코딩되어 있다

- **Claim**: `N_LAYERS=16`, `EMBEDDING_LENGTH=2048`, `HIDDEN_DIM=8192`,
  `KV_DIM=512`, `HEAD_DIM=64`, `NUM_Q_HEADS=32`, `NUM_K_HEADS=NUM_V_HEADS=8`,
  `VOCAB_SIZE=128256`이 `constexpr`로 고정된다. `Weights` 구조체는 임베딩·최종
  norm과 레이어별 input/post-attn layernorm, q/k/v/o, gate/up/down 포인터 배열을
  가진다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:12-38`, `src/main.cpp:64-77`
- **Limitation**: 여러 값에 "hardcoded for llama 3.2 1B" TODO가 붙어 있어 다른
  모델에는 그대로 적용되지 않는다 (`src/main.cpp:15`).

## Claim 3: prefill은 토큰→로짓 전체 forward를 CPU argmax까지 수행한다

- **Claim**: `prefill`은 큐에서 프롬프트를 꺼내 슬롯을 점유하고, 토큰을 GPU로
  복사한 뒤 `embeddingGather`로 임베딩을 모은다. 16개 레이어마다 RMSNorm →
  Q/K/V `cublasGemmEx` → RoPE → KV 블록 scatter → head별 attention score →
  causal mask → softmax → score×V → output projection → residual → post-attn
  RMSNorm → gate/up → SiLU → down → residual을 수행한다. 레이어 후 최종 RMSNorm과
  logit projection을 거쳐 CPU에서 argmax를 계산한다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:149-553`, `src/kernels.cu:32-53`,
  `src/kernels.cu:55-94`, `src/kernels.cu:173-222`, `src/kernels.cu:224-255`,
  `src/kernels.cu:257-309`, `src/kernels.cu:331-344`
- **Limitation**: `cublasStatus_t` 반환값이 대부분 검사되지 않는다
  (`src/main.cpp:178`, `src/main.cpp:201`, `src/main.cpp:222`). argmax가 CPU에서
  수행되어 logit 전체를 D2H 복사한다 (`src/main.cpp:529`, `src/main.cpp:533-543`).

## Claim 4: decode는 슬롯별 1토큰 forward를 pagedAttention으로 수행한다

- **Claim**: decode 루프는 활성 슬롯과 마지막 토큰을 모아
  `embeddingGatherDecode`로 임베딩을 만들고, 레이어마다 RMSNorm → Q/K/V → 슬롯별
  `ropeDecode` → KV 블록 scatter → `block_table` 동기화 → `pagedAttention` →
  output projection → residual → MLP를 수행한다. 최종 RMSNorm·logit projection 후
  CPU argmax로 다음 토큰을 정한다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:720-1039`, `src/kernels.cu:347-369`,
  `src/kernels.cu:388-405`, `src/kernels.cu:460-528`
- **Limitation**: 매 스텝마다 `block_table` 전체를 H2D 복사하고
  (`src/main.cpp:876`), logit 전체를 D2H 복사해 CPU argmax를 돌린다
  (`src/main.cpp:996`, `src/main.cpp:1000-1014`).

## Claim 5: KV 캐시는 16토큰 페이지 단위로 쪼개진 2GB 버퍼다

- **Claim**: `KV_CACHE_SIZE_BYTES`는 2GB이고 `BLOCK_SIZE=16`이므로
  `NUM_BLOCKS=65536`이다. 한 블록은 `BLOCK_SIZE*KV_DIM*bf16` 크기의 K 영역과
  `V_OFFSET`만큼 떨어진 동일 크기 V 영역으로 구성되어 `BLOCK_BYTES`가 된다.
  `block_table`은 `(slot, layer, logical_block)` 인덱스로 물리 블록 번호를
  저장하며 초기값은 `-1`이다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:32-37`, `src/main.cpp:575-581`,
  `src/main.cpp:251-288`, `src/main.cpp:852-873`
- **Limitation**: `NUM_BLOCKS` 고갈 검사가 없다. `free_blocks.back()`을 검사 없이
  호출하므로 고갈 시 미정의 동작이 된다 (`src/main.cpp:268`, `src/main.cpp:861`).

## Claim 6: 블록 할당은 prefill과 decode에서 규칙이 다르다

- **Claim**: prefill에서 해당 logical block이 `-1`이면 `free_blocks`에서 pop해
  `block_table`에 기록한다. decode에서는 `token_in_block_idx == 0`일 때만 새
  블록을 할당한다. 시퀀스 종료 시 슬롯이 쓴 모든 non-`-1` 블록을 `free_blocks`에
  반납하고 `-1`로 되돌린다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:264-277`, `src/main.cpp:859-865`,
  `src/main.cpp:1017-1031`
- **Limitation**: prefill에서 기존 블록을 만나면 `assert(false)`로 중단한다
  (`src/main.cpp:273-277`).

## Claim 7: 배칭은 BATCH_SIZE=2 슬롯과 활성 슬롯 배열로 구현된다

- **Claim**: `BATCH_SIZE=2`, `MAX_SEQUENCES=BATCH_SIZE`다. `is_slot_free`가 슬롯
  상태를 관리하며, 초기 prefill 루프와 decode 루프가 빈 슬롯에 큐의 프롬프트를
  prefill한다. 활성 슬롯/토큰을 `active_slots`·`active_tokens`로 모아 GEMM과
  attention 커널에 연속 배열로 전달한다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:29`, `src/main.cpp:38`, `src/main.cpp:596-610`,
  `src/main.cpp:695-708`, `src/main.cpp:720-757`
- **Limitation**: 작성자 스스로 `BATCH_SIZE`가 "not even close to being good"이라
  TODO로 표시한다 (`src/main.cpp:29`). 우선순위·선점 등 스케줄링은 없다.

## Claim 8: pagedAttention 커널은 온라인 소프트맥스로 페이지를 순회한다

- **Claim**: `pagedAttentionKernel`은 grid `(num_active_slots, NUM_Q_HEADS)`,
  block `HEAD_DIM`으로 실행된다. `head_dim` 스레드가 Q 원소를 하나씩 맡아
  `block_table`을 따라 K/V 페이지를 읽고, warp shuffle로 QK 점곱을 축소한 뒤
  FlashAttention 방식 online softmax(러닝 max와 보정 계수)로 정규화된 V 가중합을
  출력한다.
- **Confidence**: High
- **Evidence**: `src/kernels.cu:460-528`
- **Limitation**: GQA는 `q_head_id / GQA_Q_TO_K_RATIO`로 K/V head를 고른다
  (`src/kernels.cu:468`). `dot_products`가 2칸이고 thread 0/32만 쓰므로
  `HEAD_DIM=64`(2 warp) 전제다 (`src/kernels.cu:463`, `src/kernels.cu:496-508`).

## Claim 9: RoPE가 prefill과 decode에서 서로 다른 방식으로 구현된다

- **Claim**: prefill의 `ropeKernel_llama3`는 `init_rope_frequencies`가 llama3
  저/고주파 스케일링(`factor`, `low_freq_factor`, `high_freq_factor`,
  `original_max_len`)으로 만든 cos/sin 테이블을 사용한다. 반면 decode의
  `ropeKernelDecode`는 `500000.0`과 `HEAD_DIM`으로 theta를 매 호출 재계산하며
  스케일링된 테이블을 쓰지 않는다.
- **Confidence**: Medium
- **Evidence**: `src/kernels.cu:96-152`, `src/kernels.cu:173-222`,
  `src/kernels.cu:371-405`
- **Limitation**: 첨부 파일만으로 이 불일치가 의도된 것인지, 검증된 것인지 판단할
  수 없다 (`evidence/trace.md:190-194`).

## Claim 10: 커널 런치 가드가 1024 스레드 초과 시 실행을 건너뛴다

- **Claim**: prefill의 `rope`·`causalMask`·`softmax`와 decode의
  `ropeDecode`·`softmaxDecode`는 필요한 스레드 수가 1024를 넘으면 메시지만
  출력하고 커널을 실행하지 않는다. `MAX_SEQ_LEN=2048`이므로 1024를 넘는 구간은
  조용히 계산에서 빠진다.
- **Confidence**: High
- **Evidence**: `src/kernels.cu:202-209`, `src/kernels.cu:239-245`,
  `src/kernels.cu:293-299`, `src/kernels.cu:388-395`,
  `src/kernels.cu:442-448`, `src/main.cpp:28`
- **Limitation**: 장문에서의 실제 동작은 첨부 파일만으로 확정할 수 없다.

## Claim 11: 토크나이저가 없고 최대 생성 토큰 수가 강제되지 않는다

- **Claim**: C++ 경로에 토크나이저 호출이 없으며 프롬프트는 길이 17·14·13·14의
  하드코딩된 토큰 ID 벡터 4개다. `MAX_NEW_TOKENS_GENERATED`는 선언만 되고 사용되지
  않는다. 종료는 argmax 결과가 `END_OF_TEXT_TOKEN_ID` 또는 `EOT_ID_TOKEN_ID`이거나
  `current_prompt_len == MAX_SEQ_LEN - 1`일 때만 일어난다.
- **Confidence**: High
- **Evidence**: `src/main.cpp:12`, `src/main.cpp:583-594`,
  `src/main.cpp:1015-1037`
- **Limitation**: 최대 생성 토큰 수는 실제로는 강제되지 않으며, 서버 루프는 큐가
  비고 모든 슬롯이 해제될 때만 종료된다 (`src/main.cpp:738-746`).

## Claim 12: 시리즈 범위와 증거 규칙은 guide.md가 규정한다

- **Claim**: 시리즈는 고정 커밋의 모델 로딩 → 단일 토큰 생성 → 배칭 → paged
  KV-cache attention 경로를 다루며, 학습·분산 서빙·저장소 외 성능 주장은 범위
  밖이다. 모든 구현 주장은 소스 경로와 줄 범위를 인용해야 하고 충돌 시 코드와
  테스트가 README보다 우선한다.
- **Confidence**: High
- **Evidence**: `guide.md:3-29`
- **Limitation**: `guide.md`는 범위 문서이지 구현 증거가 아니므로, 구현 서술은
  항상 소스 인용으로 뒷받침해야 한다.
