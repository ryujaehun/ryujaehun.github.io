# tiny-vllm 근거 주장 (evidence claims)

기준 커밋: `1896ff5c37a241050dbcd9527caf9dc1d3087a61`
주장은 첨부 소스(`src/main.cpp`, `src/kernels.cu`)와 추적 메모(`evidence/trace.md`)를 근거로 한다. README 기반 설명은 배제했다.

## Claim 1 — 가중치는 safetensors 헤더의 오프셋으로 한 번에 GPU에 올라간다

- **Claim**: `loadWeights`는 `model.safetensors`의 8바이트 헤더 크기와 JSON 헤더를 읽어 각 텐서의 `data_offsets` 종료값 최대치를 `max_offset`으로 잡고, 원시 바이트 전체를 한 번의 H2D `cudaMemcpy`로 GPU에 복사한 뒤, 텐서 포인터들을 그 오프셋 기준으로 배정한다. `__metadata__`는 건너뛴다.
- **Confidence**: 높음 — 소스 직접 확인.
- **Evidence**:
  - `src/main.cpp:96-117` — 헤더 크기 8바이트 읽기(`:96-97`), JSON 파싱(`:99-101`), `data_offsets[1]` 최대치를 `max_offset`으로(`:105-115`), `__metadata__` 제외(`:108-111`).
  - `src/main.cpp:120-127` — `cudaMalloc(model_weights, max_offset)` 후 전체 바이트를 CPU 버퍼로 읽어 H2D 복사.
  - `src/main.cpp:132-145` — `embed_tokens`, `norm`, 레이어별 `input_layernorm`/`mlp_*`/`w_k/w_o/w_q/w_v`를 오프셋 기반 포인터로 설정.
- **Limitation**: 실제 `model.safetensors` 파일을 실행하지 않아 메모리 배치를 런타임 검증하지 못했다(`evidence/recon.md:4`). `json.hpp` 파싱 세부는 서드파티 코드라 검증 범위 밖이다.

## Claim 2 — 모든 GEMM은 cuBLAS "전치 트릭"으로 행-주 메모리를 처리한다

- **Claim**: 소스의 행-주 메모리를 cuBLAS의 열-주 관점으로 해석해, `Q^T = w_q^T * inputs` 형태의 `CUBLAS_OP_T/CUBLAS_OP_N` 호출로 실제 출력 `(num_tok, EMBEDDING_LENGTH)`를 얻는다. 출력을 다시 전치하지 않아도 되는 것이 이 트릭의 핵심이다.
- **Confidence**: 높음 — 주석과 호출 인자가 일치.
- **Evidence**:
  - `src/main.cpp:168-176` — 전치 트릭 원리 주석.
  - `src/main.cpp:177-196` — Q 투영 `m=EMBEDDING_LENGTH, n=prompt_len, k=EMBEDDING_LENGTH`(`CUBLAS_OP_T/CUBLAS_OP_N`).
  - `src/main.cpp:509-527` — 로짓 GEMM `m=VOCAB_SIZE, n=prompt_len, k=EMBEDDING_LENGTH`, embed_tokens를 전치.
- **Limitation**: 전치 트릭의 수치 정확성은 테스트로 확인하지 않았다. 인벤토리상 커널 테스트는 `tests/test_softmax.cu` 하나뿐이다(`evidence/recon.md:12`).

## Claim 3 — prefill은 레이어별로 rmsNorm → Q/K/V 투영 → RoPE → KV 산포 → attention → MLP 순서를 따른다

- **Claim**: `prefill`은 프롬프트 토큰을 임베딩으로 gather한 뒤(`:158`), 16개 레이어 각각에서 입력 `rmsNorm` → Q/K/V GEMM → `rope` → paged KV 산포 → 헤드별 attention 스코어 GEMM → `causalMask` → `softmax` → scores×V → o_proj → residual → post-attn `rmsNorm` → SwiGLU MLP → residual을 수행하고, 최종 `rmsNorm` 후 로짓 GEMM으로 토큰을 생성한다.
- **Confidence**: 높음 — `prefill` 함수와 레이어 루프를 직접 확인.
- **Evidence**:
  - `src/main.cpp:150-163` — 큐에서 프롬프트 pop, `embeddingGather`(`:158`), 임베딩을 `hidden_state`로 복사.
  - `src/main.cpp:164-493` — 레이어 루프 전체 순서(`rmsNorm:166`, Q/K/V:`177-240`, rope:`244-245`, KV 산포:`247-288`, attention:`301-371`, o_proj:`377-396`, residual:`399`, post-attn norm:`401`, MLP:`414-492`).
  - `src/main.cpp:494-548` — 최종 `rmsNorm`(`:494`), 로짓 GEMM(`:509-527`), CPU argmax(`:533-543`), 상태 갱신(`:546-548`).
  - `src/kernels.cu:42-53` — `embeddingGather` 커널은 토큰당 2048 요소를 1024 스레드×2로 수집.
- **Limitation**: `prompt_len > 1024`면 `causalMask`/`softmax` 커널이 실행되지 않고 메시지만 출력된다(`src/kernels.cu:241-244`, `:295-298`). 이 커밋의 프롬프트는 17토큰 이하여서 도달하지 않지만, 일반 prefill 경로의 한계로 남는다.

## Claim 4 — decode는 슬롯 배치로 처리하고 argmax는 CPU에서 수행한다

- **Claim**: decode 루프는 활성 슬롯을 한 번에 묶어 Q/K/V GEMM의 `n=num_active_slots` 배치로 처리하고, 레이어별 K/V를 블록 테이블이 가리키는 KV 캐시 위치에 D2D 산포한다. 로짓은 D2H 복사 후 CPU 루프의 argmax로 토큰을 선택한다.
- **Confidence**: 높음 — decode 루프를 직접 확인.
- **Evidence**:
  - `src/main.cpp:722-737` — `active_slots`/`active_tokens` 재구성, 여유 슬롯이면 큐에서 prefill.
  - `src/main.cpp:759` — `embeddingGatherDecode`(블록=활성 슬롯 수).
  - `src/main.cpp:763-842` — Q/K/V GEMM `n=num_active_slots` 배치 처리.
  - `src/main.cpp:996-1012` — 로짓 D2H 복사 후 CPU argmax.
  - `src/kernels.cu:358-369` — `embeddingGatherDecode` 호출 래퍼.
- **Limitation**: argmax가 CPU라 매 decode 반복마다 D2H 복사가 필요하다. 소스가 TODO로 인지한다(`src/main.cpp:531-532`, `:686`). CPU argmax는 bf16을 float로 캐스팅해 비교한다(`src/main.cpp:1003-1010`).

## Claim 5 — KV 캐시는 2GiB 고정 블록 풀과 CPU/GPU 블록 테이블로 관리된다

- **Claim**: KV 캐시는 `KV_CACHE_SIZE_BYTES=2GiB`로 한 번에 할당되고, `BLOCK_SIZE=16` 토큰×`KV_DIM=512`×bf16(2바이트)를 단위로 한 블록을 `BLOCK_BYTES=32768`바이트로 계산한다. 총 `NUM_BLOCKS=65536`개 블록을 `free_blocks` 스택이 소유하고, 각 시퀀스·레이어의 논리 블록→물리 블록 매핑은 CPU `block_table`(전부 -1 초기화)이 담당한다.
- **Confidence**: 높음 — 상수 정의와 초기화를 직접 확인.
- **Evidence**:
  - `src/main.cpp:32-37` — `BLOCK_SIZE=16`, `V_OFFSET`, `BLOCK_BYTES`, `KV_CACHE_SIZE_BYTES=2GiB`, `MAX_BLOCKS_PER_SEQ=128`, `NUM_BLOCKS=65536`.
  - `src/main.cpp:576-581` — `kv_cache` 2GiB 할당, `free_blocks` 0..65535 채움, `block_table` -1 초기화, `block_table_gpu` 할당.
  - `src/kernels.cu:16-19` — `BLOCK_SIZE`, `V_OFFSET`, `BLOCK_BYTES`, `MAX_BLOCKS_PER_SEQ` 재정의(중복).
- **Limitation**: 상수가 `main.cpp`와 `kernels.cu`에 중복 정의되어 있고 소스가 이 중복을 TODO로 인지한다(`src/kernels.cu:6`, `:8-19`). 이 캐시는 예약(paged) 방식이라 실제 사용량과 무관하게 2GiB 전체가 할당된다.

## Claim 6 — KV 산포는 prefill에서 미리 전체 블록을, decode에서 한 토큰씩 채운다

- **Claim**: prefill은 프롬프트를 `BLOCK_SIZE` 단위로 잘라 논리 블록마다 `free_blocks`에서 물리 블록을 pop해 `block_table`에 기록한 뒤 K(`+0`)와 V(`+V_OFFSET`)를 D2D 복사한다. 블록이 이미 할당돼 있으면 `assert(false)`를 던진다. decode는 `seq_len`의 블록 내 위치(`seq_len % BLOCK_SIZE`)가 0일 때만 새 블록을 pop하고, 그 외에는 기존 블록의 해당 토큰 위치에 K/V를 덮어쓴다.
- **Confidence**: 높음 — 양쪽 산포 경로를 직접 확인.
- **Evidence**:
  - prefill 산포: `src/main.cpp:251-288` — 블록 인덱스 계산(`:264`), `block_table`에서 -1이면 pop 후 기록(`:266-272`), 기존 할당 시 `assert(false)`(`:273-277`), K/V D2D 복사(`:280-287`).
  - decode 산포: `src/main.cpp:851-873` — 논리 블록= `seq_len/BLOCK_SIZE`, 블록 내 토큰= `seq_len%BLOCK_SIZE`(`:856-857`), 첫 토큰일 때만 pop(`:859-865`), K/V 복사(`:866-872`).
  - 블록 테이블 GPU 동기화: `src/main.cpp:876`(매 레이어), `:552`(prefill 종료), `:1030`(슬롯 해제).
- **Limitation**: 산포가 전부 D2D `cudaMemcpy`라 커널이 아니다. `assert(false)` 분기는 정상 경로에서 도달하지 않는다고 가정한다(`src/main.cpp:273-277`).

## Claim 7 — GQA는 4개 Q 헤드가 1개 K/V 헤드를 공유한다

- **Claim**: Q는 32개 헤드, K/V는 8개 헤드이며, prefill과 paged attention 모두 `k_head_idx = q_head_id / GQA_Q_TO_K_RATIO(=4)`로 4개 Q 헤드가 1개의 K·V 헤드를 공유한다.
- **Confidence**: 높음 — 상수와 인덱싱을 직접 확인.
- **Evidence**:
  - `src/main.cpp:20-24` — `NUM_Q_HEADS=32`, `NUM_K_HEADS=NUM_V_HEADS=8`, `GQA_Q_TO_K_RATIO=4`, `GQA_ATTN_SCORES_TO_V_RATIO=4`.
  - `src/main.cpp:303` — attention 스코어 GEMM의 `k_head_idx = i / GQA_Q_TO_K_RATIO`.
  - `src/main.cpp:346` — scores×V GEMM의 `v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO`.
  - `src/kernels.cu:468` — paged attention 커널의 `kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO`.
- **Limitation**: GQA 비율(4)이 상수로 박혀 있어 모델 구조가 바뀌면 재컴파일이 필요하다(`src/main.cpp:15`, `:23`).

## Claim 8 — paged attention 커널은 블록 테이블 순회 + warp 내 트리 합 + 온라인 softmax를 하나로 합친다

- **Claim**: `pagedAttentionKernel`은 그리드 `(num_active_slots, NUM_Q_HEADS)`, 블록 `HEAD_DIM=64`로 실행된다. 각 스레드가 Q의 1차원을 담당하고, 블록 테이블(`gpu_active_slots` 매핑)을 따라 물리 블록을 순회하며 K·V를 읽어 warp 내 `__shfl_down_sync` 트리 합으로 dot product를 구하고, 누적 K 토큰만 읽는 방식으로 마스킹을 대신한다. FlashAttention 온라인 softmax로 `current_max`/`d`/`acc`를 갱신해 `acc/d`를 출력한다.
- **Confidence**: 높음 — 커널 구현을 직접 확인.
- **Evidence**:
  - `src/kernels.cu:525-527` — `pagedAttention` 호출 래퍼, 그리드/블록 구성.
  - `src/kernels.cu:461-523` — 커널 본문: 슬롯·헤드 매핑(`:464-469`), 물리 블록 조회(`:480`), 블록 내 토큰 루프(`:482-520`), warp 트리 합(`:489-493`), online softmax(`:510-519`), 출력(`:522`).
  - `src/main.cpp:878` — decode 루프의 호출 지점.
- **Limitation**: `dot_products`는 크기 2의 shared 배열로 두 warp 절반을 결합한다(`src/kernels.cu:463`, `:494-507`) — `HEAD_DIM=64`에서 두 번째 warp은 블록이 비는지 보장되지 않으나, 시퀀스/헤드 크기가 64라 모든 스레드가 유효하다고 가정한다. 캐시에 이미 쓴 KV만 읽으므로 causal 마스킹은 별도 커널 없이 "읽지 않음"으로 구현된다.

## Claim 9 — 배칭은 2개 슬롯, 여유 슬롯이 생기면 decode 중에도 즉시 prefill을 채운다

- **Claim**: `BATCH_SIZE=2` 고정 슬롯 배칭이다. 초기 루프가 큐의 프롬프트 2개를 슬롯 0, 1에 prefill하고, decode 루프에서 여유 슬롯이 있으면 큐에서 다음 프롬프트를 prefill해 같은 반복의 decode에 즉시 포함시킨다. 종료 조건(EOT/EOT_ID 또는 `MAX_SEQ_LEN-1`)에 도달하면 슬롯을 해제하고 블록을 `free_blocks`에 반환한다.
- **Confidence**: 높음 — 배치/슬롯/해제 로직을 직접 확인.
- **Evidence**:
  - `src/main.cpp:29` — `BATCH_SIZE=2` 주석("just here to have batching").
  - `src/main.cpp:695-708` — 초기 prefill 루프(슬롯 0, 1 채움).
  - `src/main.cpp:726-737` — 여유 슬롯 prefill 후 active에 추가, `num_active_slots` 결정.
  - `src/main.cpp:1015-1037` — EOT/최대 길이 시 슬롯 해제(`:1015-1031`), 아니면 `current_prompt_len` +1(`:1032-1037`).
  - `src/main.cpp:584-594` — 큐에 고정 4개 프롬프트.
- **Limitation**: decode 루프 탈출은 큐가 비고 `num_active_slots==0`일 때뿐이라(`src/main.cpp:739-746`), 큐가 빈 뒤 모든 슬롯이 종료 조건에 도달할 때까지 루프가 지속된다. `MAX_NEW_TOKENS_GENERATED`(`:12`)는 선언만 되고 루프 조건에 쓰이지 않는다.

## Claim 10 — decode는 매 레이어마다 블록 테이블을 CPU→GPU로 전체 재동기화한다

- **Claim**: decode의 각 레이어에서 KV 산포로 `block_table`이 바뀐 뒤, `block_table_gpu` 전체를 H2D로 다시 복사한 다음에야 `pagedAttention`을 호출한다. prefill 종료 시에도 전체 테이블을 한 번 동기화한다. 슬롯 해제 시에도 동기화가 재발생한다.
- **Confidence**: 높음 — 호출 지점을 직접 확인.
- **Evidence**:
  - `src/main.cpp:876` — 레이어 루프 내 `cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES*N_LAYERS*MAX_BLOCKS_PER_SEQ*sizeof(int), H2D)`.
  - `src/main.cpp:552` — prefill 종료 시 전체 동기화.
  - `src/main.cpp:1030` — 슬롯 해제 시 전체 동기화.
- **Limitation**: 매 레이어 전체 테이블(2×16×128×4바이트)을 복사한다. 소스가 TODO로 인지한다("do it more clever and not copy full table unnecessarily", `src/main.cpp:551`).
