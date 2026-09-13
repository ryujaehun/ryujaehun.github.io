# tiny-vllm 실행 경로 추적

고정 커밋: `1896ff5c37a241050dbcd9527caf9dc1d3087a61`
본 문서는 `src/main.cpp`, `src/kernels.cu`, `CMakeLists.txt`만 사용해
모델 로딩 → prefill → decode 루프 → paged attention까지의 실행 경로를 추적한다.
모든 실행 주장은 소스 경로와 줄 범위로 인용한다.

## Entrypoint

- 빌드 단위: `CMakeLists.txt:49-52`가 `src/main.cpp` + `src/kernels.cu`로
  `tiny-vllm` 실행파일을 구성한다. CUDA arch는 `120`(`CMakeLists.txt:24`),
  링크는 `CUDA::cublas`, `CUDA::cudart`(`CMakeLists.txt:71-74`).
- 진입점: `main(int, char**)` — `src/main.cpp:555`.
- `main`의 최상위 순서(`src/main.cpp:557-570`):
  1. `cublasCreate`로 cuBLAS 핸들 생성(`:557-563`). 실패 시 `return 1`.
  2. `loadWeights(weights)`(`:566-569`). 실패 시 `return 1`.
  3. `init_rope_frequencies(HEAD_DIM, MAX_SEQ_LEN, 500000, 32, 1, 4, 8192)`(`:572`).
  4. KV 캐시·블록 테이블 초기화(`:574-581`).
  5. 프롬프트 큐 구성(`:584-594`), 배치 상태 초기화(`:596-601`).
  6. CUDA 버퍼 전부 할당(`:615-693`).
  7. 초기 prefill 루프로 슬롯 채우기(`:695-708`).
  8. 무한 decode 루프(`:720-1039`).
  9. 종료: `"Ok bye!"` 출력, `cublasDestroy`, `cudaDeviceSynchronize`,
     `return 0`(`:1040-1044`).
- 상수: `N_LAYERS=16`, `EMBEDDING_LENGTH=2048`, `HIDDEN_DIM=8192`, `KV_DIM=512`,
  `HEAD_DIM=64`, `NUM_Q_HEADS=32`, `NUM_K_HEADS=NUM_V_HEADS=8`, `GQA_Q_TO_K_RATIO=4`,
  `VOCAB_SIZE=128256`, `MAX_SEQ_LEN=2048`, `BATCH_SIZE=2`, `MAX_PROMPT_LEN=512`,
  `BLOCK_SIZE=16`, `KV_CACHE_SIZE_BYTES=2GiB`(`src/main.cpp:12-38`).

## Ordered steps

### 1. 가중치 로드 — `loadWeights` (`src/main.cpp:79-147`)

1. `checkGPUStatus()`(`:81-84`; 정의 `:40-62`): 디바이스 수 0이면 `1` 반환
   (경로 분기), GPU 프로퍼티/메모리 출력. 실패 시 `loadWeights`가 `1` 반환.
2. `model.safetensors`를 바이너리로 열기(`:87-93`). 실패 시 `1` 반환.
3. 헤더 크기 8바이트를 `uint64_t`로 읽음(`:96-97`).
4. 헤더 JSON 읽기(`:99-101`).
5. 텐서 이름→오프셋 맵 구축, `data_offsets[1]`의 최대값을 `max_offset`으로
   저장(`:103-118`). `__metadata__`는 건너뜀(`:108-111`).
6. `cudaMalloc`으로 `model_weights`를 `max_offset` 크기로 할당(`:120-121`).
7. 원시 가중치를 CPU 벡터로 읽고(`:123-125`) H2D `cudaMemcpy`(`:127`).
8. 각 텐서 포인터를 오프셋 기준으로 설정(`:132-145`):
   `embed_tokens`, `norm`, 그리고 레이어별 `input_layernorm`,
   `mlp_down_proj/gate_proj/up_proj`, `post_attention_layernorm`,
   `w_k/w_o/w_q/w_v`.

### 2. RoPE 주파수 테이블 — `init_rope_frequencies` (`src/kernels.cu:96-152`)

- `main.cpp:572`에서 `head_dim=64, max_seq_len=2048, theta=500000, factor=32,
  low_freq_factor=1, high_freq_factor=4, original_max_len=8192`로 호출.
- `inv_freq` 계산 후 Llama3 NTK 스케일링 적용(`kernels.cu:100-124`).
- 디바이스에 `d_inv_freq` 할당·복사(`:126-127`).
- `cos_table`/`sin_table`은 `[max_seq_len, head_dim]` 크기, 각 (pos, i)에서
  `cos/sin(pos*inv_freq)`를 짝수·홀수 요소에 중복 저장(`:129-144`).
- `d_cos_table`, `d_sin_table` 할당·복사(`:146-151`). 전역 포인터
  `d_inv_freq/d_cos_table/d_sin_table`(`kernels.cu:22-24`)이 이후
  `rope`/`ropeDecode`에서 사용된다.

### 3. KV 캐시·블록 테이블 초기화 (`src/main.cpp:574-581`)

- `kv_cache` 2GiB 할당(`:576`).
- `free_blocks`를 `0..NUM_BLOCKS-1`(65536개)로 채움(`:577-578`).
- `block_table`을 `MAX_SEQUENCES*N_LAYERS*MAX_BLOCKS_PER_SEQ` 크기로 `-1`
  초기화(`:579`), `block_table_gpu` 할당(`:580-581`).

### 4. 배치/슬롯 상태와 버퍼 (`src/main.cpp:584-693`)

- 프롬프트 큐: 4개 문장(길이 17/14/13/14 토큰) 푸시(`:584-594`).
- `is_slot_free` 2칸 모두 `true`(`:597`), `generated_tokens`,
  `last_generated_tokens`, `current_prompt_len` 초기화(`:599-601`).
- 디바이스 버퍼 할당: `gpu_input_tokens`(`:618`), `input_embeddings`(`:620`),
  `hidden_state`(`:623`), `rms_norms`(`:626`), `buf_2048_1`(`:629`),
  `k_proj_temp_buf`/`v_proj_temp_buf`(`:636`,`:639`),
  `prefill_attn_scores`(`:648`), `buf_2048_2`(`:657`), `gate`(`:663`),
  `up`(`:668`), `embed_proj`(`:677`), `embed_proj_cpu`(`:681-682`),
  `gpu_last_tokens`(`:685`), `k_proj_batched_buffer`/`v_proj_batched_buffer`
  (`:690`,`:693`).
- `buf_2048_1`은 q_proj와 attn_scores_v 공유, `buf_2048_2`는 o_proj와 down
  공유(`:628-629`, `:656-657`).

### 5. 초기 prefill — `prefill` (`src/main.cpp:150-553`, 호출 `:695-708`)

`for slot in 0..BATCH_SIZE`, `!queue.empty()` 동안 여유 슬롯에 prefill
(`:695-701`). BATCH_SIZE=2이므로 슬롯 0, 1에 큐의 프롬프트 1, 2가 들어간다.

prefill 내부 순서:

1. 큐에서 프롬프트를 꺼내고(`:152-154`) 슬롯 점유 처리(`:155`).
2. 프롬프트 토큰을 GPU로 복사(`:157`).
3. `embeddingGather`로 입력 임베딩 수집(`:158`;
   `kernels.cu:42-53`, 커널 `:32-40`, 블록=토큰 수, 스레드=1024).
4. 임베딩을 `hidden_state`로 복사(`:160-163`).
5. 레이어 루프 `layer=0..15`(`:164-493`):
   - `rmsNorm`(입력)(`:166`; `kernels.cu:84-94`, 커널 `:55-81`).
   - Q 투영 GEMM(`:177-196`): `CUBLAS_OP_T/CUBLAS_OP_N`,
     `m=EMBEDDING_LENGTH, n=prompt_len, k=EMBEDDING_LENGTH`. 행-주
     메모리를 열-주로 보는 cuBLAS 전치 트릭(`:168-176` 주석).
   - K 투영 GEMM(`:201-219`), V 투영 GEMM(`:222-240`): `m=KV_DIM`.
   - `rope(q_proj, prompt_len, 2048)`(`:244`; `kernels.cu:202-222`,
     커널 `:173-200`) 및 `rope(k_proj_temp_buf, prompt_len, 512)`(`:245`).
   - Paged KV 산포(`:247-288`): `BLOCK_SIZE` 단위로 블록 인덱스 계산
     (`:251-264`), `block_table`에서 논리 블록의 물리 블록을 읽어 `-1`이면
     `free_blocks`에서 pop 후 기록(`:264-272`), `-1`이 아니면
     `assert(false)`(`:275`). K는 `kv_cache + block*BLOCK_BYTES`,
     V는 `+ V_OFFSET`에 D2D 복사(`:280-287`).
   - 헤드별 attention 스코어 GEMM(`:301-327`): `i`번째 Q 헤드는
     `k_head_idx = i/4` K 헤드와 공유(GQA, `:303`),
     `attn_alpha = 1/8`(`:649`)로 스케일.
   - `causalMask`(`:329`; `kernels.cu:239-255`, 커널 `:224-237`) —
     `column > row` 위치를 `-HUGE_VALF`로 마스킹.
   - `softmax`(`:331`; `kernels.cu:293-309`, 커널 `:257-290`) — 트리 결합으로
     최대값+합을 한 번에 계산하는 온라인 softmax.
   - attn_scores × V GEMM(`:343-371`): V 헤드도 `i/4` 매핑(`:346`),
     출력은 `buf_2048_1`(=attn_scores_v)의 각 Q 헤드 슬롯에 기록.
   - o_proj GEMM(`:377-396`) → `residualAdd`(`:399`;
     `kernels.cu:319-329`, 커널 `:311-316`) → post-attn `rmsNorm`(`:401`).
   - MLP: gate GEMM(`:414-432`, `m=HIDDEN_DIM`), up GEMM(`:435-453`),
     `silu`(`:459`; `kernels.cu:341-344`, 커널 `:331-338`, gate를 in-place로
     `SiLU(gate)*up`로 덮어씀), down GEMM(`:470-489`, 입력은 `gate`),
     `residualAdd`(`:492`).
6. 최종 `rmsNorm`(`:494`).
7. 로짓 GEMM: `embed_tokens`와의 전치 곱, `m=VOCAB_SIZE, n=prompt_len`
   (`:509-527`).
8. 로짓을 CPU로 복사(`:529`), 마지막 토큰 행에서 CPU argmax(`:533-543`),
   결과 출력(`:544`).
9. 상태 갱신: `generated_tokens[slot].push_back`,
   `last_generated_tokens[slot]`, `current_prompt_len[slot] = prompt_len`
   (`:546-548`).
10. `block_table` → `block_table_gpu` 전체 복사(`:552`).

### 6. decode 루프 (`src/main.cpp:720-1039`)

1. `active_slots`/`active_tokens` 재구성(`:722-737`): 여유 슬롯이면 큐에서
   prefill(기존 `generated_tokens`는 clear, `:732-733`), 모든 슬롯을
   active에 추가(`:735-736`). 즉 이번 decode 반복에 새로 prefill된 슬롯도
   즉시 포함된다.
2. `num_active_slots == 0`이면 큐가 비었을 때 `break`(`:738-746`).
3. `gpu_last_tokens`, `gpu_active_slots`, `gpu_seq_lens`
   (`current_prompt_len+1`)를 GPU로 복사(`:749-757`).
4. `embeddingGatherDecode`(`:759`; `kernels.cu:358-369`, 커널 `:347-356`) —
   블록=활성 슬롯 수.
5. 레이어 루프(`:760-972`):
   - `rmsNorm`(`:762`).
   - Q 투영 GEMM(`:763-783`), K 투영 GEMM(`:802-821`), V 투영 GEMM
     (`:824-842`): `n=num_active_slots` 배치 처리, K/V는 임시 버퍼에 기록.
   - 슬롯별 `ropeDecode`(`:844-849`; `kernels.cu:388-405`, 커널 `:371-384`) —
     위치는 `current_prompt_len[active_slot]`.
   - Paged KV 산포(`:851-873`): 논리 블록 = `seq_len/BLOCK_SIZE`,
     블록 내 토큰 = `seq_len%BLOCK_SIZE`(`:856-857`). 블록 내 첫 토큰일 때만
     `free_blocks`에서 pop·블록 테이블 기록(`:859-865`). K/V를
     `kv_cache + block*BLOCK_BYTES (+ V_OFFSET) + token*KV_DIM`에 D2D 복사
     (`:866-872`).
   - `block_table` → GPU 동기화(`:876`).
   - `pagedAttention`(`:878`; `kernels.cu:525-527`, 커널 `:461-523`):
     그리드 `(num_active_slots, NUM_Q_HEADS)`, 블록 `HEAD_DIM=64`.
     커널은 블록 테이블을 따라 물리 블록을 순회하며
     (`kernels.cu:478-481`) 각 토큰의 K·V를 읽고(`:484-485`) warp 내
     `__shfl_down_sync` 트리 합으로 dot product를 계산(`:489-505`),
     온라인 softmax로 `d`와 가중합 `acc`를 갱신(`:508-519`),
     `acc/d`를 출력(`:522`). 이 커널 안에 마스킹(누적 K만 읽음)과
     softmax가 모두 포함된다.
   - o_proj GEMM(`:880-900`) → `residualAdd`(`:902`) → post-attn
     `rmsNorm`(`:904`).
   - MLP: gate(`:907-925`), up(`:928-946`), `silu`(`:948`), down(`:950-969`),
     `residualAdd`(`:971`).
6. 최종 `rmsNorm`(`:974`), 로짓 GEMM(`:976-994`).
7. 로짓 CPU 복사(`:996`), 슬롯별 CPU argmax(`:1000-1012`).
8. 종료 판정(`:1015-1037`): argmax가 `END_OF_TEXT_TOKEN_ID`(128001) 또는
   `EOT_ID_TOKEN_ID`(128009)이거나 `current_prompt_len == MAX_SEQ_LEN-1`이면
   슬롯 해제(아래 State changes), 아니면 `last_generated_tokens`,
   `generated_tokens` 갱신 및 `current_prompt_len` +1(`:1034-1036`).
9. 반복. 최종 탈출은 `:739-744`의 `break`.

## State changes

| 상태 | 초기값 | prefill 시 | decode 시 | 종료 시 |
|---|---|---|---|---|
| `queue` | 4개 프롬프트(`:585-594`) | 1개 pop(`:152-154`) | 여유 슬롯에서 pop(`:733`) | 비면 break(`:742-743`) |
| `is_slot_free[slot]` | 모두 true(`:597`) | false(`:155`) | — | true(`:1017`) |
| `free_blocks` | 0..65535(`:577-578`) | 블록 산포에서 pop(`:268-269`) | 새 블록 필요 시 pop(`:861-862`) | 해제된 슬롯의 물리 블록 push(`:1025`) |
| `block_table` | 전부 -1(`:579`) | 논리 블록에 물리 블록 기록(`:271`) | 블록 경계에서 기록(`:864`) | 해제 슬롯은 -1로 재설정(`:1026`) |
| `block_table_gpu` | 할당만(`:581`) | H2D 동기화(`:552`) | 매 레이어 H2D 동기화(`:876`), 해제 시(`:1030`) | — |
| `kv_cache` | 할당만(`:576`) | K/V D2D 산포(`:280-287`) | K/V D2D 산포(`:866-872`) | 해제 슬롯 블록은 free_blocks로 복귀 |
| `generated_tokens[slot]` | 빈 벡터(`:599`) | 토큰 추가(`:546`) | clear 후 prefill(`:732`), 토큰 추가(`:1035`) | — |
| `last_generated_tokens[slot]` | 0(`:600`) | argmax 결과(`:547`) | argmax 결과(`:1034`) | — |
| `current_prompt_len[slot]` | 0(`:601`) | prompt_len(`:548`) | +1(`:1036`) | — |
| `hidden_state` 등 재사용 버퍼 | 할당(`:623-674`) | prefill 크기 사용 | 배치 크기 사용 | — |

- 전역 디바이스 배열: `d_inv_freq`, `d_cos_table`, `d_sin_table`
  (`kernels.cu:22-24`)은 `init_rope_frequencies`에서 할당·초기화
  (`kernels.cu:126-151`), `rope`/`ropeDecode`에서 읽음(`:194-196`).

## Branches and exits

- `cublasCreate` 실패 → `main` `return 1`(`src/main.cpp:559-562`).
- `loadWeights` 실패 → `main` `return 1`(`:566-568`).
- `checkGPUStatus`: 디바이스 0개 → `1`(`:44-48`).
- `model.safetensors` 열기 실패 → `1`(`:88-93`).
- prefill 블록 산포에서 블록이 이미 할당됨 → `assert(false)`
  (`:273-277`); 정상 경로는 항상 `-1`로 가정.
- decode 블록 산포: 블록 내 첫 토큰(`token_in_block_idx==0`)에서만 새 블록
  할당(`:859-865`), 이후는 기존 블록에 덮어씀.
- decode 루프: `num_active_slots==0 && queue.empty()` → `break`(`:739-744`);
  `num_active_slots==0 && !queue.empty()` → `continue`(`:745`).
- argmax 종료 조건(EOT/EOT_ID 또는 `MAX_SEQ_LEN-1`) → 슬롯 해제
  (`:1015-1031`); 아니면 생성 지속(`:1032-1037`).
- 정상 종료: `cublasDestroy` + `cudaDeviceSynchronize` + `return 0`
  (`:1040-1044`).
- 커널 가드: RoPE는 `num_threads = proj_dim / 2 > 1024`일 때 실행을 건너뛰고
  메시지만 출력(`kernels.cu:204-209`). `causalMask`와 `softmax`는
  `num_tokens > 1024`일 때 실행을 건너뛴다(`:241-244`, `:295-298`).
- `siluKernel`은 `a`를 in-place로 덮어씀(`kernels.cu:331-344`).

## Unresolved gaps

- `softmaxDecode`(`kernels.cu:408-458`)가 정의되어 있으나 `main.cpp`에서
  호출되지 않는다. decode의 softmax는 `pagedAttentionKernel` 내부 온라인
  softmax(`kernels.cu:508-519`)가 담당하므로 이 함수는 이 커밋에서 미사용
  (dead code)으로 보인다.
- `free_rope_frequencies`(`kernels.cu:154-171`)도 호출처가 없다.
- `main.cpp`의 종료는 `break` 하나뿐이라, 큐가 빈 뒤 모든 슬롯이 EOT/
  MAX_SEQ_LEN에 도달할 때까지 루프가 계속된다(`:720`, `:739-744`).
- `checkGPUStatus`(`:40-62`)는 `cudaGetDeviceCount` 외 다른 반환값
  검증이 없고 프로퍼티 출력만 한다.
- 상수(`N_LAYERS`, `EMBEDDING_LENGTH` 등)가 `main.cpp:12-38`과
  `kernels.cu:8-19`에 중복 정의되어 있으며, `kernels.cu:6` 주석이 이 중복을
  TODO로 인지하고 있다.
- `prefill_attn_scores`는 `MAX_PROMPT_LEN^2 * NUM_Q_HEADS`로 할당되지만
  실제 사용은 `prompt_len^2 * NUM_Q_HEADS`이고, 커널 스레드 수가
  `prompt_len`을 넘을 수 없어 `prompt_len > 1024`면 실행이 생략된다
  (`main.cpp:648`, `kernels.cu:241-244`, `:295-298`).
- prefill과 decode 모두 argmax를 CPU에서 수행하며 D2H 복사 후 루프를
  거쳐야 한다(`main.cpp:529`, `:533-543`, `:996`, `:1000-1012`); TODO 주석
  참조(`:531-532`, `:686`).
- decode 산포의 논리 블록/토큰 인덱스는 실제 슬롯(`active_slot`) 기준이고
  (`main.cpp:856-858`), `pagedAttentionKernel`은 q·seq_len은
  `blockIdx.x`(활성 슬롯 내 위치)로, 블록 테이블 조회는 `gpu_active_slots`
  매핑을 거쳐 실제 슬롯으로 읽는다(`kernels.cu:464-470`, `:480`) — 두
  인덱싱 체계가 섞여 있어 추적 시 주의가 필요하다.
- `embed_proj_cpu`는 `MAX_BUFFER_SIZE*VOCAB_SIZE`로 할당되지만
  (`main.cpp:681-682`) prefill은 `prompt_len*VOCAB_SIZE`만 복사하므로
  남은 행은 이전 슬롯의 잔여 데이터일 수 있다(`:529`).

## Source paths

- `src/main.cpp` — `main`(`:555`), `loadWeights`(`:79-147`),
  `checkGPUStatus`(`:40-62`), `prefill`(`:150-553`), decode 루프
  (`:720-1039`), 상수(`:12-38`).
- `src/kernels.cu` — `embeddingGather`(`:42-53`)/`embeddingGatherKernel`
  (`:32-40`), `rmsNorm`(`:84-94`)/`rmsNormKernel`(`:55-81`),
  `init_rope_frequencies`(`:96-152`), `rope`(`:202-222`)/`ropeKernel_llama3`
  (`:173-200`), `causalMask`(`:239-255`)/`causalMaskKernel`(`:224-237`),
  `softmax`(`:293-309`)/`softmaxKernel`(`:257-290`), `residualAdd`
  (`:319-329`)/`residualKernel`(`:311-316`), `silu`(`:341-344`)/
  `siluKernel`(`:331-338`), `embeddingGatherDecode`(`:358-369`)/
  `embeddingGatherKernelDecode`(`:347-356`), `ropeDecode`(`:388-405`)/
  `ropeKernelDecode`(`:371-384`), `softmaxDecode`(`:442-458`)/
  `softmaxKernelDecode`(`:408-439`), `pagedAttention`(`:525-527`)/
  `pagedAttentionKernel`(`:461-523`).
- `CMakeLists.txt` — 프로젝트/언어(`:15`), CUDA arch(`:24`), 소스 목록
  (`:49-52`), 링크(`:71-74`).
