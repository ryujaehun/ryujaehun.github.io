# tiny-vllm: CUDA로 읽는 LLM 추론 엔진 — 챕터 브리프

고정 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61` 기준. 모든 서술은
`evidence/claims.md`의 주장(Claim)과 그 증거 인용만을 근거로 하며, 새 인용을
발명하지 않는다.

## Chapter 1 — tiny-vllm의 실행 경로와 빌드

### Scope

- `CMakeLists.txt`, `src/main.cpp`, `test.sh` (`series.yaml:6`).
- 시리즈 전반의 범위와 증거 규칙을 규정하는 `guide.md` (`guide.md:3-29`).

### Claims

- **Claim 12**: 시리즈는 고정 커밋의 모델 로딩 → 단일 토큰 생성 → 배칭 → paged
  KV-cache attention 경로를 다루며, 학습·분산 서빙·저장소 외 성능 주장은 범위
  밖이다. 구현 주장은 소스 경로와 줄 범위로 뒷받침하고, 충돌 시 코드·테스트가
  README보다 우선한다. (Confidence: High)
- **Claim 2**: 실행 프로그램의 모델 차원(`N_LAYERS=16`, `EMBEDDING_LENGTH=2048`,
  `HIDDEN_DIM=8192`, `KV_DIM=512`, `HEAD_DIM=64`, `NUM_Q_HEADS=32`,
  `NUM_K_HEADS=NUM_V_HEADS=8`, `VOCAB_SIZE=128256`)이 `constexpr`로 컴파일 상수
  고정된다. (Confidence: High, `src/main.cpp:12-38`)
- **Claim 10**: 커널 런치 가드가 1024 스레드 초과 시 커널 실행을 건너뛴다.
  `MAX_SEQ_LEN=2048`이므로 장문에서 1024를 넘는 구간은 조용히 계산에서 빠진다.
  (Confidence: High, `src/kernels.cu:202-209` 등)
- **Claim 11**: C++ 경로에 토크나이저가 없고 프롬프트는 하드코딩된 토큰 ID
  벡터 4개다. `MAX_NEW_TOKENS_GENERATED`는 선언만 되고 미사용이며, 서버 루프는
  큐가 비고 모든 슬롯이 해제될 때만 종료된다. (Confidence: High,
  `src/main.cpp:583-594`, `src/main.cpp:738-746`)

### Exclusions

- 학습, 분산 서빙, 저장소 외부 성능 주장 (Claim 12).
- 가중치 로딩·포맷 내부(Chapter 2), forward 커널 상세(Chapter 4 이후).
- CUDA/HIP 빌드 분기 등 claims.md에 인용되지 않은 빌드 상세.

### Visuals

- 시리즈 아크 지도: 모델 로딩 → 토큰 생성 → 배칭 → paged attention 4단계를
  챕터별로 배치한 다이어그램 (Claim 12).
- 컴파일 상수 표: `src/main.cpp:12-38`의 하드코딩 차원 값을 표로 정리
  (Claim 2).

## Chapter 2 — Safetensors와 Llama 가중치 적재

### Scope

- `src/main.cpp`, `include/json.hpp` (`series.yaml:7`).

### Claims

- **Claim 1**: 로더는 `model.safetensors`를 이진으로 열어 8바이트 헤더 크기와
  JSON 헤더를 읽고, 텐서별 `data_offsets[0]`을 모아 `max_offset`을 구한 뒤 한 번의
  `cudaMalloc`으로 잡은 버퍼에 전체 가중치를 H2D 복사한다. 이후 각 텐서 포인터를
  단일 할당의 오프셋으로 설정한다. (Confidence: High, `src/main.cpp:87`,
  `src/main.cpp:96-104`, `src/main.cpp:120-145`)
- **Claim 2**: `Weights` 구조체는 임베딩·최종 norm과 레이어별 input/post-attn
  layernorm, q/k/v/o, gate/up/down 포인터 배열을 가진다. 여러 값에 "hardcoded for
  llama 3.2 1B" TODO가 붙어 있어 다른 모델에는 그대로 적용되지 않는다.
  (Confidence: High, `src/main.cpp:64-77`, `src/main.cpp:15`)

### Exclusions

- 텐서 형상·dtype 검증과 `__metadata__` 처리 등 Claim 1의 한계로 밝혀진 부분은
  "없는 기능"으로만 언급한다 (`src/main.cpp:106-111`).
- forward/추론 경로 전체는 이후 챕터로.

### Visuals

- 로딩 파이프라인 도해: 파일 → 8바이트 크기 → JSON 헤더 → 오프셋 맵 →
  `max_offset` → 단일 `cudaMalloc` 버퍼 → 포인터 오프셋 배정 (Claim 1).
- 단일 할당 버퍼 위에 텐서들이 오프셋으로 쌓이는 레이아웃 그림 (Claim 1).

## Chapter 3 — 토크나이저, 임베딩, 첫 토큰

### Scope

- `src/main.cpp`, `python/tokenizer.py` (`series.yaml:8`).

### Claims

- **Claim 11**: C++ 경로에 토크나이저 호출이 없으며 프롬프트는 길이 17·14·13·14의
  하드코딩된 토큰 ID 벡터 4개다. 종료는 argmax 결과가
  `END_OF_TEXT_TOKEN_ID`/`EOT_ID_TOKEN_ID`이거나 `current_prompt_len ==
  MAX_SEQ_LEN - 1`일 때만 일어난다. (Confidence: High, `src/main.cpp:583-594`,
  `src/main.cpp:1015-1037`)
- **Claim 3**: prefill은 토큰을 GPU로 복사한 뒤 `embeddingGather`로 임베딩을
  모으고, 16개 레이어를 거쳐 최종 RMSNorm과 logit projection을 수행한 다음
  CPU에서 argmax를 계산해 첫 토큰을 정한다. argmax가 CPU 수행이라 logit 전체를
  D2H 복사한다. (Confidence: High, `src/kernels.cu:32-53`,
  `src/main.cpp:529`, `src/main.cpp:533-543`)
- **Claim 2**: 임베딩 차원(`EMBEDDING_LENGTH=2048`, `VOCAB_SIZE=128256`)이 컴파일
  상수로 고정된다. (Confidence: High, `src/main.cpp:12-38`)

### Exclusions

- 레이어 내부 연산(RMSNorm, RoPE, GEMM)은 Chapter 4에서 상세화.
- 토크나이저 구현 자체: C++ 경로에는 없고 `python/tokenizer.py`는 참조 구현으로만
  두되, claims.md에 인용이 없으므로 동작을 주장하지 않는다.
- `MAX_NEW_TOKENS_GENERATED`가 강제되지 않는 점은 Claim 11의 한계로 서술
  (`src/main.cpp:12`).

### Visuals

- 토큰 ID → `embeddingGather` → hidden state → ... → logit → CPU argmax의 첫
  토큰 생성 흐름도 (Claim 3).
- 하드코딩 프롬프트 4개(길이 17·14·13·14)를 큐로 보여주는 그림 (Claim 11).

## Chapter 4 — RMSNorm, RoPE, GEMM으로 만드는 transformer block

### Scope

- `src/main.cpp`, `src/kernels.cu` (`series.yaml:9`).

### Claims

- **Claim 3**: 레이어마다 RMSNorm → Q/K/V `cublasGemmEx` → RoPE → KV 블록
  scatter → attention → output projection → residual → post-attn RMSNorm →
  gate/up → SiLU → down → residual 순으로 수행된다. `cublasStatus_t` 반환값은
  대부분 검사되지 않는다. (Confidence: High, `src/main.cpp:149-553`,
  `src/kernels.cu:55-94`, `src/kernels.cu:331-344`)
- **Claim 9**: prefill의 `ropeKernel_llama3`는 `init_rope_frequencies`가 llama3
  저/고주파 스케일링(`factor`, `low_freq_factor`, `high_freq_factor`,
  `original_max_len`)으로 만든 cos/sin 테이블을 사용한다. 반면 decode의
  `ropeKernelDecode`는 `500000.0`과 `HEAD_DIM`으로 theta를 재계산하며 스케일링된
  테이블을 쓰지 않는다. (Confidence: Medium, `src/kernels.cu:96-152`,
  `src/kernels.cu:173-222`, `src/kernels.cu:371-405`)
- **Claim 10**: prefill의 `rope` 커널은 1024 스레드 초과 시 실행되지 않고
  메시지만 출력한다. (Confidence: High, `src/kernels.cu:202-209`)

### Exclusions

- attention 내부(score→mask→softmax→가중합)는 Chapter 5에서 상세화.
- Claim 9의 RoPE 불일치가 의도된 것인지·검증되었는지는 첨부 자료로 판단하지
  않는다 (Claim 9 Limitation).
- 커널 성능 주장은 다루지 않는다 (Claim 12).

### Visuals

- 단일 transformer block의 데이터 흐름 다이어그램: RMSNorm → QKV GEMM → RoPE →
  attention → O projection → residual → post-attn norm → MLP(gate/up→SiLU→down)
  → residual (Claim 3).
- prefill/decode RoPE 두 경로를 나란히 놓은 비교 그림 (Claim 9).

## Chapter 5 — attention과 KV cache

### Scope

- `src/main.cpp`, `src/kernels.cu` (`series.yaml:10`).

### Claims

- **Claim 3**: prefill attention은 head별 attention score → causal mask → softmax →
  score×V 순으로 수행된다. (Confidence: High, `src/kernels.cu:173-222`,
  `src/kernels.cu:224-255`, `src/kernels.cu:257-309`)
- **Claim 5**: KV 캐시는 2GB 버퍼고 `BLOCK_SIZE=16`이므로 `NUM_BLOCKS=65536`이다.
  한 블록은 `BLOCK_SIZE*KV_DIM*bf16` 크기의 K 영역과 `V_OFFSET`만큼 떨어진 동일
  크기 V 영역으로 구성된다. `block_table`은 `(slot, layer, logical_block)` 인덱스로
  물리 블록 번호를 저장하며 초기값은 `-1`이다. `NUM_BLOCKS` 고갈 검사가 없고
  `free_blocks.back()`을 검사 없이 호출한다. (Confidence: High,
  `src/main.cpp:32-37`, `src/main.cpp:575-581`, `src/main.cpp:251-288`)
- **Claim 6**: prefill에서 해당 logical block이 `-1`이면 `free_blocks`에서 pop해
  `block_table`에 기록한다. prefill에서 기존 블록을 만나면 `assert(false)`로
  중단한다. (Confidence: High, `src/main.cpp:264-277`)
- **Claim 10**: prefill의 `causalMask`·`softmax` 커널은 1024 스레드 초과 시 실행을
  건너뛴다. (Confidence: High, `src/kernels.cu:239-245`,
  `src/kernels.cu:293-299`)

### Exclusions

- paged attention 커널의 online softmax는 Chapter 9·10에서 상세화.
- decode 경로의 attention은 Chapter 6 이후.

### Visuals

- KV 캐시 블록 구조: 16토큰 페이지의 K 영역 + V 영역 레이아웃과
  `block_table`의 물리 블록 매핑 그림 (Claim 5).
- causal mask + softmax가 attention score에 적용되는 단계 그림 (Claim 3).

## Chapter 6 — prefill과 decode의 서로 다른 병목

### Scope

- `src/main.cpp`, `README.md` (`series.yaml:11`). README는 보조 증거로만 사용.

### Claims

- **Claim 3**: prefill은 큐에서 프롬프트를 꺼내 슬롯을 점유하고 전체 forward를
  CPU argmax까지 수행한다. logit 전체를 D2H 복사해 CPU에서 argmax를 계산한다.
  (Confidence: High, `src/main.cpp:149-553`, `src/main.cpp:529`,
  `src/main.cpp:533-543`)
- **Claim 4**: decode는 활성 슬롯과 마지막 토큰을 모아 슬롯별 1토큰 forward를
  수행한다. 매 스텝마다 `block_table` 전체를 H2D 복사하고 logit 전체를 D2H 복사해
  CPU argmax를 돌린다. (Confidence: High, `src/main.cpp:720-1039`,
  `src/main.cpp:876`, `src/main.cpp:996`, `src/main.cpp:1000-1014`)
- **Claim 7**: 배칭은 `BATCH_SIZE=2` 슬롯과 활성 슬롯 배열로 구현된다.
  (Confidence: High, `src/main.cpp:29`, `src/main.cpp:38`)
- **Claim 11**: 서버 루프는 큐가 비고 모든 슬롯이 해제될 때만 종료된다.
  (Confidence: High, `src/main.cpp:738-746`)

### Exclusions

- 배치·스케줄링 구조 자체는 Chapter 7·8에서 상세화.
- 성능 측정·프로파일링 주장은 범위 밖 (Claim 12).

### Visuals

- prefill(전 토큰 일괄) vs decode(슬롯당 1토큰)의 forward 경로를 나란히 비교한
  그림 (Claim 3, 4).
- H2D/D2H 복사 지점을 표시한 데이터 이동 흐름도 (Claim 3, 4 한계).

## Chapter 7 — static batching

### Scope

- `src/main.cpp` (`series.yaml:12`).

### Claims

- **Claim 7**: `BATCH_SIZE=2`, `MAX_SEQUENCES=BATCH_SIZE`다. `is_slot_free`가
  슬롯 상태를 관리하며, 초기 prefill 루프와 decode 루프가 빈 슬롯에 큐의
  프롬프트를 prefill한다. 활성 슬롯/토큰을 `active_slots`·`active_tokens`로 모아
  GEMM과 attention 커널에 연속 배열로 전달한다. 작성자 스스로 `BATCH_SIZE`가
  "not even close to being good"이라 TODO로 표시한다. (Confidence: High,
  `src/main.cpp:29`, `src/main.cpp:38`, `src/main.cpp:596-610`,
  `src/main.cpp:695-708`, `src/main.cpp:720-757`)
- **Claim 4**: decode 루프는 활성 슬롯과 마지막 토큰을 모아 연속 배열로 forward에
  넘긴다. (Confidence: High, `src/main.cpp:720-757`)

### Exclusions

- 우선순위·선점 등 스케줄링은 Claim 7의 한계로 명시 (없는 기능으로만 언급).
- prefill/decode 내부 연산 상세는 Chapter 4·6.

### Visuals

- 고정 2개 슬롯에서 프롬프트가 prefill되고 나머지 슬롯이 대기하는 상태 그림
  (Claim 7).
- `active_slots`/`active_tokens` 배열이 커널로 전달되는 연속 배열 도식
  (Claim 7).

## Chapter 8 — continuous batching과 scheduler

### Scope

- `src/main.cpp` (`series.yaml:13`).

### Claims

- **Claim 7 한계**: 우선순위·선점 등 스케줄링은 없으며 `BATCH_SIZE`는 고정 2로
  TODO다. (Confidence: High, `src/main.cpp:29`)
- **Claim 4**: decode 루프가 매 반복마다 빈 슬롯에 큐의 프롬프트를 prefill하고
  활성 슬롯을 모은다. (Confidence: High, `src/main.cpp:724-757`)
- **Claim 6**: 시퀀스 종료 시 슬롯이 쓴 모든 non-`-1` 블록을 `free_blocks`에
  반납하고 `-1`로 되돌린다. decode에서는 `token_in_block_idx == 0`일 때만 새
  블록을 할당한다. (Confidence: High, `src/main.cpp:859-865`,
  `src/main.cpp:1017-1031`)
- **Claim 11**: 종료는 argmax가 종료 토큰이거나 `MAX_SEQ_LEN - 1`일 때만
  일어나고, 서버 루프는 큐가 비고 모든 슬롯이 해제될 때 종료된다.
  (Confidence: High, `src/main.cpp:1015-1037`, `src/main.cpp:738-746`)

### Exclusions

- "연속 배칭"의 일반론(동적 스케줄러, 우선순위, 선점)은 저장소에 없으므로
  교육적 구현과의 대비로만 언급.
- Claim 6의 prefill `assert(false)` 분기는 Chapter 5에서 다룸.

### Visuals

- 슬롯이 EOT로 해제되며 블록이 `free_blocks`로 돌아가고 빈 슬롯이 큐에서 새
  프롬프트를 받는 상태 전이 그림 (Claim 4, 6, 11).

## Chapter 9 — online softmax CUDA kernel

### Scope

- `src/kernels.cu`, `tests/test_softmax.cu` (`series.yaml:14`). 테스트 파일은
  claims.md에 인용이 없으므로 존재만 언급하고 동작을 주장하지 않는다.

### Claims

- **Claim 8**: `pagedAttentionKernel`은 warp shuffle로 QK 점곱을 축소한 뒤
  FlashAttention 방식 online softmax(러닝 max와 보정 계수)로 정규화된 V 가중합을
  출력한다. (Confidence: High, `src/kernels.cu:460-528`)
- **Claim 3**: prefill softmax는 `src/kernels.cu:293-309`의 커널로 수행된다.
  (Confidence: High)
- **Claim 10**: prefill의 `softmax`·`causalMask` 커널은 1024 스레드 초과 시 실행을
  건너뛴다. (Confidence: High, `src/kernels.cu:293-299`,
  `src/kernels.cu:239-245`)

### Exclusions

- online softmax 일반 이론은 구현 인용으로만 뒷받침.
- paged attention의 페이지 순회·GQA 매핑은 Chapter 10에서 상세화.

### Visuals

- 러닝 max와 보정 계수가 업데이트되는 online softmax 단계 그림 (Claim 8).
- warp shuffle로 QK 점곱이 축소되는 과정 도식 (Claim 8).

## Chapter 10 — paged KV cache와 paged attention

### Scope

- `src/main.cpp`, `src/kernels.cu` (`series.yaml:15`).

### Claims

- **Claim 5**: KV 캐시는 `BLOCK_SIZE=16` 토큰 페이지로 쪼개진 2GB 버퍼다. 한
  블록은 K 영역과 `V_OFFSET`만큼 떨어진 V 영역으로 구성되고 `block_table`이 물리
  블록 번호를 저장한다. (Confidence: High, `src/main.cpp:32-37`,
  `src/main.cpp:575-581`)
- **Claim 8**: `pagedAttentionKernel`은 grid `(num_active_slots, NUM_Q_HEADS)`,
  block `HEAD_DIM`으로 실행된다. `head_dim` 스레드가 Q 원소를 하나씩 맡아
  `block_table`을 따라 K/V 페이지를 읽고, warp shuffle로 QK 점곱을 축소한 뒤
  online softmax로 정규화된 V 가중합을 출력한다. GQA는 `q_head_id /
  GQA_Q_TO_K_RATIO`로 K/V head를 고른다. `dot_products`가 2칸이고 thread 0/32만
  쓰므로 `HEAD_DIM=64`(2 warp) 전제다. (Confidence: High, `src/kernels.cu:460-528`)
- **Claim 6**: decode에서는 `token_in_block_idx == 0`일 때만 새 블록을 할당하고,
  시퀀스 종료 시 블록을 `free_blocks`에 반납한다. (Confidence: High,
  `src/main.cpp:859-865`, `src/main.cpp:1017-1031`)
- **Claim 4**: decode 레이어마다 `block_table` 동기화 후 `pagedAttention`을
  수행한다. (Confidence: High, `src/main.cpp:876`, `src/main.cpp:878`)

### Exclusions

- 커널 성능 비교·최적화 수치 주장은 범위 밖 (Claim 12).
- `NUM_BLOCKS` 고갈 시 동작은 Claim 5의 한계로만 언급.

### Visuals

- `block_table`이 물리 페이지를 따라가며 K/V를 읽는 paged attention 순회 그림
  (Claim 8).
- GQA에서 Q head가 K/V head로 사상되는 다이어그램 (Claim 8).
- 블록 할당/반납이 `free_blocks` 스택을 오가는 흐름도 (Claim 6).