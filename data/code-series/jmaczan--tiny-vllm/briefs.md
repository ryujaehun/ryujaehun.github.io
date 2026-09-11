# tiny-vllm 챕터 브리프

기준 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61`. 챕터 구성은 `series.yaml`, 근거는 `evidence/claims.md`의 Claim 1~10만 사용한다. 각 챕터는 Scope(대상 파일), Claims(사용할 주장), Exclusions(다루지 않을 것), Visuals(시각자료)로 나눈다. 챕터 제목은 `series.yaml`을 그대로 따른다.

## 근거 색인

- **Claim 1** — safetensors 헤더의 `data_offsets`로 `max_offset`을 잡고 원시 바이트 전체를 1회 H2D 복사한 뒤 오프셋 기반으로 텐서 포인터를 배정한다(`src/main.cpp:96-117`, `:120-127`, `:132-145`). `__metadata__`는 건너뛴다.
- **Claim 2** — 모든 GEMM이 cuBLAS 전치 트릭(`CUBLAS_OP_T/CUBLAS_OP_N`)으로 행-주 메모리를 처리한다(`src/main.cpp:168-176`, `:177-196`, `:509-527`).
- **Claim 3** — prefill은 레이어마다 rmsNorm → Q/K/V → RoPE → KV 산포 → attention → MLP 순서를 따른다(`src/main.cpp:150-163`, `:164-492`, `:494-548`; `src/kernels.cu:42-53`).
- **Claim 4** — decode는 활성 슬롯을 묶어 배치 처리하고 argmax는 CPU에서 수행한다(`src/main.cpp:722-737`, `:759`, `:763-842`, `:996-1012`; `src/kernels.cu:358-369`).
- **Claim 5** — KV 캐시는 2GiB 고정 블록 풀과 CPU/GPU 블록 테이블로 관리된다(`src/main.cpp:32-37`, `:576-581`; `src/kernels.cu:16-19`).
- **Claim 6** — KV 산포는 prefill에서 블록 단위, decode에서 토큰 단위로 일어난다(`src/main.cpp:251-288`, `:851-873`, `:552`, `:876`, `:1030`).
- **Claim 7** — GQA는 4개 Q 헤드가 1개 K/V 헤드를 공유한다(`src/main.cpp:20-24`, `:303`, `:346`; `src/kernels.cu:468`).
- **Claim 8** — paged attention 커널은 블록 테이블 순회 + warp 내 트리 합 + 온라인 softmax를 하나로 합친다(`src/kernels.cu:461-523`, `:525-527`; `src/main.cpp:878`).
- **Claim 9** — 배칭은 `BATCH_SIZE=2` 고정이고 여유 슬롯이 생기면 decode 중에도 즉시 prefill을 채운다(`src/main.cpp:29`, `:695-708`, `:726-737`, `:1015-1037`, `:584-594`).
- **Claim 10** — decode는 매 레이어마다 블록 테이블을 CPU→GPU로 전체 재동기화한다(`src/main.cpp:876`, `:552`, `:1030`).

---

## Chapter 1 — tiny-vllm의 실행 경로와 빌드

### Scope
- `CMakeLists.txt`, `src/main.cpp`, `test.sh` (`series.yaml` 1장).

### Claims
- **직접 근거 없음.** Claim 1~10 중 빌드 시스템, 컴파일 플래그, 링크 대상, `test.sh` 내부를 다루는 주장은 없다. 이 챕터는 빌드/테스트 상세를 주장하지 않는다.
- **참조(소유 아님):**
  - Claim 5(`src/main.cpp:32-37`, `:576-581`) — 실행 상수와 KV 캐시 초기화 위치.
  - Claim 9(`src/main.cpp:29`, `:584-594`) — `BATCH_SIZE=2`와 하드코딩된 프롬프트 큐.
  - Claim 3(`src/main.cpp:150-163`) — `prefill` 진입점.
  - Claim 4(`src/main.cpp:722-737`) — decode 루프 진입점.

### Exclusions
- CMake 옵션·CUDA arch·링크 라이브러리·`test.sh` 내용은 근거 주장이 없으므로 서술하지 않는다.
- CUDA/HIP 이식 계층은 어떤 주장도 다루지 않는다.
- 가중치 로딩 내부(2장), forward 커널 상세(4장 이후)는 소유 챕터로 넘긴다.
- 학습·분산 서빙·저장소 외부 성능 주장은 `guide.md`가 범위 밖으로 둔다.

### Visuals
- `main()` 준비 → prefill → decode → 종료 흐름도. 각 단계에 Claim 5·9·3·4의 근거 줄을 라벨로 표시.
- 실행 상수 표. Claim 5(`BLOCK_SIZE=16`, `KV_DIM=512`, `BLOCK_BYTES=32768`, `KV_CACHE_SIZE_BYTES=2GiB`, `MAX_BLOCKS_PER_SEQ=128`, `NUM_BLOCKS=65536`), Claim 9(`BATCH_SIZE=2`), Claim 7(`NUM_Q_HEADS=32`, `NUM_K_HEADS=8`, `GQA_Q_TO_K_RATIO=4`)가 인용한 값만 담는다.

---

## Chapter 2 — Safetensors와 Llama 가중치 적재

### Scope
- `src/main.cpp`, `include/json.hpp` (`series.yaml` 2장).

### Claims
- **Claim 1 (주 소유):** 8바이트 헤더 크기 → JSON 헤더 파싱 → `data_offsets[1]` 최대치를 `max_offset`으로 → `cudaMalloc` 후 원시 바이트 1회 H2D 복사 → 오프셋 기반 포인터 배정. `__metadata__` 제외. 근거: `src/main.cpp:96-117`, `:120-127`, `:132-145`.
- **Claim 1 Limitation:** 실제 `model.safetensors`로 런타임 검증하지 않았다. `json.hpp` 파싱 세부는 서드파티라 검증 범위 밖이다.

### Exclusions
- `include/json.hpp` 내부 구현은 다루지 않는다(서드파티, Claim 1이 범위 밖으로 명시).
- 실제 파일의 메모리 배치를 런타임 검증한 것처럼 서술하지 않는다.
- 저장소에 없는 포맷 변환·양자화 처리는 다루지 않는다.

### Visuals
- safetensors 레이아웃 도식: `[8바이트 헤더 길이][JSON 헤더][원시 텐서 바이트]`와 `max_offset` 경계.
- 오프셋 → 텐서 포인터 표(`embed_tokens`, `norm`, 레이어별 `input_layernorm`/`mlp_*`/`w_q/k/v/o`). Claim 1의 `src/main.cpp:132-145` 목록만 사용.

---

## Chapter 3 — 토크나이저, 임베딩, 첫 토큰

### Scope
- `src/main.cpp`, `python/tokenizer.py` (`series.yaml` 3장).

### Claims
- **Claim 3 (일부):** prefill이 프롬프트 토큰을 `embeddingGather`로 임베딩으로 gather한다(`src/main.cpp:150-163`의 `:158`; `src/kernels.cu:42-53`).
- **Claim 4 (일부):** decode의 `embeddingGatherDecode`와 CPU argmax로 다음 토큰을 고른다(`src/kernels.cu:358-369`; `src/main.cpp:996-1012`, `:1003-1010`).
- **Claim 9 (일부):** 프롬프트가 하드코딩된 토큰 ID 벡터 4개이므로(`src/main.cpp:584-594`) C++ 경로에 문자열→토큰 변환 단계가 없다.

### Exclusions
- `python/tokenizer.py` 내부는 Claim 1~10 어느 것도 근거로 삼지 않으므로 다루지 않는다.
- 토크나이저 알고리즘·vocab 처리·특수 토큰 규칙은 근거가 없어 서술하지 않는다.
- RMSNorm·GEMM 등 transformer block 내부는 4장으로 넘긴다.

### Visuals
- 토큰 ID → `embeddingGather` → `hidden_state` → (최종) 로짓 → CPU argmax 흐름도.
- prefill용 gather(토큰당 2048요소, 1024스레드×2)와 decode용 gather(블록=활성 슬롯 수) 비교표. Claim 3·4 근거 줄 표기.

---

## Chapter 4 — RMSNorm, RoPE, GEMM으로 만드는 transformer block

### Scope
- `src/main.cpp`, `src/kernels.cu` (`series.yaml` 4장).

### Claims
- **Claim 2 (주 소유):** cuBLAS 전치 트릭. `CUBLAS_OP_T/CUBLAS_OP_N`, Q 투영 `m=EMBEDDING_LENGTH, n=prompt_len, k=EMBEDDING_LENGTH`, 로짓 `m=VOCAB_SIZE, n=prompt_len, k=EMBEDDING_LENGTH`. 근거: `src/main.cpp:168-176`, `:177-196`, `:509-527`.
- **Claim 3 (주 소유):** 레이어 순서 rmsNorm → Q/K/V GEMM → RoPE → KV 산포 → attention → o_proj → residual → post-attn rmsNorm → SwiGLU MLP → residual, 최종 rmsNorm 후 로짓 GEMM. 근거: `src/main.cpp:164-492`, `:494-548`.
- **Claim 7 (참조):** Q 32헤드·K/V 8헤드와 GQA 비율 4(`src/main.cpp:20-24`).

### Exclusions
- attention 스코어·KV cache 산포 상세는 5장, paged attention은 10장으로 넘긴다.
- Claim 2 Limitation: 전치 트릭의 수치 정확성은 테스트로 확인되지 않았다. 커널 테스트는 `tests/test_softmax.cu` 하나뿐이다.
- Claim 3 Limitation: `prompt_len > 1024`면 `causalMask`/`softmax` 커널이 실행되지 않고 메시지만 출력된다(`src/kernels.cu:241-244`, `:295-298`).

### Visuals
- 레이어 파이프라인 다이어그램(rmsNorm → Q/K/V → RoPE → attention → o_proj → residual → MLP)과 단계별 근거 줄.
- 행-주 메모리를 열-주로 해석하는 cuBLAS 전치 트릭 도식(`Q^T = w_q^T * inputs`), 출력을 다시 전치하지 않아도 되는 이유 강조.

---

## Chapter 5 — attention과 KV cache

### Scope
- `src/main.cpp`, `src/kernels.cu` (`series.yaml` 5장).

### Claims
- **Claim 5 (주 소유):** KV 캐시는 `KV_CACHE_SIZE_BYTES=2GiB`로 한 번에 할당되고, `BLOCK_SIZE=16`×`KV_DIM=512`×bf16 단위 블록이 `BLOCK_BYTES=32768`바이트다. `NUM_BLOCKS=65536`개를 `free_blocks`가 소유하고, 논리→물리 매핑은 CPU `block_table`(-1 초기화)이 담당한다. 근거: `src/main.cpp:32-37`, `:576-581`; 상수 중복 정의 `src/kernels.cu:16-19`.
- **Claim 6 (주 소유):** prefill은 `BLOCK_SIZE` 단위로 물리 블록을 pop해 K(`+0`)/V(`+V_OFFSET`)를 D2D 복사하고, decode는 `seq_len % BLOCK_SIZE == 0`일 때만 새 블록을 pop한 뒤 해당 토큰 위치에 덮어쓴다. 근거: `src/main.cpp:251-288`, `:851-873`.
- **Claim 7 (주 소유):** prefill·paged attention 모두 `k_head_idx = q_head_id / GQA_Q_TO_K_RATIO(=4)`로 4개 Q 헤드가 1개 K/V 헤드를 공유한다(`src/main.cpp:303`, `:346`; `src/kernels.cu:468`).

### Exclusions
- paged attention 커널 자체는 10장으로 넘긴다.
- 배칭·스케줄러 정책은 7·8장으로 넘긴다.
- Claim 6 Limitation: 산포는 커널이 아니라 D2D `cudaMemcpy`다. `assert(false)` 분기는 정상 경로에서 도달하지 않는다고 가정한다.
- Claim 5 Limitation: 상수 중복은 소스가 TODO로 인지한다. 2GiB는 사용량과 무관한 예약 할당이다.

### Visuals
- 논리 블록 → 물리 블록 `block_table` 매핑과 `free_blocks` 스택 도식.
- K(`+0`)/V(`+V_OFFSET`) 산포 오프셋, prefill(블록 단위)과 decode(토큰 단위) 채움 방식 비교.
- Q 헤드 32개가 K/V 헤드 8개를 4:1로 공유하는 GQA 도식.

---

## Chapter 6 — prefill과 decode의 서로 다른 병목

### Scope
- `src/main.cpp`, `README.md` (`series.yaml` 6장).

### Claims
- **Claim 3 (참조/소유):** prefill 경로 전체와 그 한계(`src/main.cpp:150-548`; `prompt_len>1024` 가드 `src/kernels.cu:241-244`, `:295-298`).
- **Claim 4 (주 소유):** decode는 활성 슬롯을 묶어 Q/K/V GEMM을 `n=num_active_slots`로 처리하고, 로짓은 D2H 복사 후 CPU argmax로 토큰을 고른다(`src/main.cpp:722-737`, `:759`, `:763-842`, `:996-1012`). Limitation: 매 decode 반복마다 D2H 복사가 필요하다(`src/main.cpp:531-532`, `:686`, `:1003-1010`).
- **Claim 6 (참조):** prefill은 프롬프트 전체 블록을, decode는 한 토큰씩 KV 캐시를 채운다(`src/main.cpp:251-288`, `:851-873`).
- **Claim 5 (참조):** KV 캐시는 사용량과 무관하게 2GiB를 예약한다.

### Exclusions
- `README.md`의 성능 수치·설계 주장은 코드 근거와 분리하며, 시리즈 범위 밖이므로 인용하지 않는다.
- 측정된 병목 수치는 Claim 1~10에 없다. "병목"은 작업량·복사 횟수의 구조적 차이로만 서술하고, 실행 시간·처리량을 주장하지 않는다.
- 배칭 정책 상세는 7·8장, paged attention은 10장으로 넘긴다.

### Visuals
- prefill 루프와 decode 루프의 나란한 시퀀스 다이어그램(프롬프트 전체 vs 슬롯당 1토큰).
- 단계별 성격 표: prefill(토큰 전체 GEMM, KV 블록 단위) vs decode(배치 GEMM, 토큰 단위 KV, 매 반복 D2H argmax).

---

## Chapter 7 — static batching

### Scope
- `src/main.cpp` (`series.yaml` 7장).

### Claims
- **Claim 9 (주 소유):** `BATCH_SIZE=2` 고정 슬롯 배칭. 초기 루프가 큐의 프롬프트 2개를 슬롯 0·1에 prefill하고, 종료 조건(EOT/EOT_ID 또는 `MAX_SEQ_LEN-1`)에서 슬롯을 해제하고 블록을 `free_blocks`에 반환한다. 근거: `src/main.cpp:29`, `:695-708`, `:1015-1037`, `:584-594`.
- **Claim 4 (참조):** decode가 활성 슬롯을 한 번에 묶어 `n=num_active_slots` 배치로 처리한다(`src/main.cpp:722-737`, `:763-842`).

### Exclusions
- continuous batching과 스케줄러는 8장으로 넘긴다.
- Claim 9 Limitation: `MAX_NEW_TOKENS_GENERATED`는 선언만 되고 루프 조건에 쓰이지 않는다. 큐가 빈 뒤 모든 슬롯이 끝날 때까지 루프가 지속된다.

### Visuals
- 슬롯 2칸 상태 표(초기 prefill → decode → 해제)와 타임라인.
- 정적 배치에서 배치 크기가 `BATCH_SIZE=2`로 고정되는 지점을 강조한 도식.

---

## Chapter 8 — continuous batching과 scheduler

### Scope
- `src/main.cpp` (`series.yaml` 8장).

### Claims
- **Claim 9 (주 소유):** decode 루프에서 여유 슬롯이 있으면 큐에서 다음 프롬프트를 prefill해 같은 반복의 decode에 즉시 포함시킨다(`src/main.cpp:726-737`).
- **Claim 10 (주 소유):** decode의 각 레이어에서 KV 산포로 `block_table`이 바뀐 뒤 `block_table_gpu` 전체를 H2D로 재복사한 다음에야 `pagedAttention`을 호출한다. prefill 종료·슬롯 해제 시에도 전체 동기화가 일어난다. 근거: `src/main.cpp:876`, `:552`, `:1030`.
- **Claim 6 (참조):** decode는 블록 내 첫 토큰일 때만 새 물리 블록을 pop한다(`src/main.cpp:851-873`).

### Exclusions
- 별도 "scheduler" 객체·우선순위·선점은 어떤 주장도 뒷받침하지 않는다. 슬롯 재사용 로직으로만 서술한다.
- Claim 10 Limitation: 매 레이어 전체 테이블(2×16×128×4바이트)을 복사하며, 소스가 TODO로 인지한다(`src/main.cpp:551`).
- Claim 9 Limitation: 종료는 큐가 비고 `num_active_slots==0`일 때뿐이라 모든 슬롯이 끝날 때까지 지속된다(`src/main.cpp:739-746`).

### Visuals
- decode 반복 타임라인: 슬롯 해제 → 여유 슬롯 prefill → 같은 반복 active에 합류.
- 레이어 루프 안 `block_table` 동기화 지점(`src/main.cpp:876`)과 KV 산포 → pagedAttention 호출 순서 도식.

---

## Chapter 9 — online softmax CUDA kernel

### Scope
- `src/kernels.cu`, `tests/test_softmax.cu` (`series.yaml` 9장).

### Claims
- **Claim 8 (주 소유):** FlashAttention 방식 온라인 softmax가 `current_max`/`d`/`acc`를 갱신하고 `acc/d`를 출력한다(`src/kernels.cu:509-519`, `:522`).
- **Claim 8 (참조):** 각 스레드가 Q의 1차원을 담당하고, `__shfl_down_sync` 트리 합으로 dot product를 구한다(`src/kernels.cu:489-493`, `:494-507`).
- **Claim 2 Limitation (참조):** 커널 단위 테스트는 `tests/test_softmax.cu` 하나뿐이다.

### Exclusions
- `tests/test_softmax.cu`의 구체 내용·수치는 Claim 1~10에 근거가 없으므로 서술하지 않는다.
- prefill의 별도 `softmax` 커널은 4·5장 문맥에서만 언급한다.
- Claim 8 Limitation: `dot_products`는 크기 2의 shared 배열로 두 warp 절반을 결합하며, `HEAD_DIM=64`에서 모든 스레드가 유효하다고 가정한다.

### Visuals
- FlashAttention식 온라인 softmax 갱신 도식(running max, rescale, 누적 `d`, 가중합 `acc`).
- warp 트리 합(`__shfl_down_sync`)과 두 warp 절반 결합(`dot_products[2]`) 도식.

---

## Chapter 10 — paged KV cache와 paged attention

### Scope
- `src/main.cpp`, `src/kernels.cu` (`series.yaml` 10장).

### Claims
- **Claim 5:** 2GiB 고정 블록 풀과 CPU/GPU 블록 테이블.
- **Claim 6:** prefill(블록 단위)과 decode(토큰 단위)의 KV 산포.
- **Claim 7:** GQA `kv_head_idx = q_head_id / 4`.
- **Claim 8 (주 소유):** `pagedAttentionKernel`은 그리드 `(num_active_slots, NUM_Q_HEADS)`, 블록 `HEAD_DIM=64`로 실행된다. 블록 테이블(`gpu_active_slots` 매핑)을 따라 물리 블록을 순회하며 K·V를 읽고, warp 내 트리 합으로 dot product를 구하고, 누적 K 토큰만 읽는 방식으로 마스킹을 대신하며, 온라인 softmax로 `current_max`/`d`/`acc`를 갱신해 `acc/d`를 출력한다. 근거: `src/kernels.cu:525-527`, `:461-523`; 호출 `src/main.cpp:878`.
- **Claim 10:** `pagedAttention` 호출 전 매 레이어 `block_table`을 재동기화한다(`src/main.cpp:876`).

### Exclusions
- 배칭 정책은 7·8장으로 넘긴다.
- Claim 8 Limitation: `dot_products` 크기 2의 두 warp 절반 결합은 `HEAD_DIM=64` 가정에 의존한다.
- Claim 6 Limitation: KV 산포는 커널이 아니라 D2D `cudaMemcpy`다.
- vLLM 등 외부 시스템과의 성능 비교는 `guide.md` 범위 밖이다.

### Visuals
- 물리 블록 풀 + `block_table` + `gpu_active_slots` 매핑 도식.
- `pagedAttentionKernel`의 그리드/블록/스레드 배치와 warp 트리 합 → 온라인 softmax 흐름도.
- "누적 K만 읽음 = causal masking" 도식.
