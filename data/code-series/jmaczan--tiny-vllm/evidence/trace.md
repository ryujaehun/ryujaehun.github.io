# tiny-vLLM 실행 경로 Trace Memo

이 메모는 `guide.md`, `src/main.cpp`, `src/kernels.cu`, `CMakeLists.txt` 네
첨부만을 근거로 고정 revision
`e25bf1994efa90bc98b721ba7c527402f86fbeaf` 의 실행 경로를 정적 추적한 것이다.
코드는 실행하지 않았고, 런타임 산출물도 만들지 않았다(결정 003,
`guide.md:75-87`). 첨부에 없는 경로의 내용은 추론하지 않으며 `guide.md` 의
서술로만 인용하거나 미해결로 남긴다. 경로 표기는 `파일:시작-끝` 형식이다.

## Entrypoint

- 저장소가 스스로 규정한 실행 경로는 셸 경계다. `guide.md:114-125` 에 따르면
  `full_test.sh` 가 하드코딩된 토큰 ID 열을 표준 입력으로 `./test.sh` 에 넘기고,
  `test.sh` 가 `build.sh`(cmake + ninja)로 `build/tiny-vllm` 을 만들고 `run.sh`
  로 즉시 실행한다. **셸 스크립트 본문은 첨부에 없으므로 이 사실은
  `guide.md` 인용이며 본문 검증은 미해결이다.**
- 제품 바이너리는 두 번역 단위로만 만들어진다. `CMakeLists.txt:49-52` 의
  `add_executable(tiny-vllm src/main.cpp src/kernels.cu)` 가 그 대상을 규정하고,
  `CMakeLists.txt:10-16` 이 `USE_HIP` 분기로 언어(`CUDA`/`HIP`)를 정하며,
  `CMakeLists.txt:24` 가 CUDA 아키텍처 `120` 을, `CMakeLists.txt:62-63` 이
  include 경로 `src`·`include` 를, `CMakeLists.txt:71-74` 가
  `CUDA::cublas`·`CUDA::cudart` 링크를 정한다.
- C++ 진입점은 `main` 이다. `src/main.cpp:555-556` 에서 시작해 첫 행위는
  `cublasCreate`(`src/main.cpp:557-563`), 곧바로 `loadWeights`(가중치 적재)
  호출(`src/main.cpp:566-569`)이다.
- 호스트 함수는 `checkGPUStatus`(`src/main.cpp:40-62`), `loadWeights`
  (`src/main.cpp:79-147`), `prefill`(`src/main.cpp:150-553`), `main`
  (`src/main.cpp:555-1044`) 넷이며, 그중 `main` 이 실행의 주도권을 쥔다
  (`guide.md:103-105`).

## Ordered steps

### 1. 초기화(`main`, `src/main.cpp:555-708`)

1. cuBLAS 핸들 생성: `cublasCreate` — `src/main.cpp:557-563`.
2. 가중치 적재: `loadWeights` — `src/main.cpp:566-569`.
   - GPU 상태 확인: `checkGPUStatus` 호출 — `src/main.cpp:81-84`;
     구현 `src/main.cpp:40-62`.
   - `model.safetensors` 이진 오픈 — `src/main.cpp:87-93`.
   - 헤더 크기 8바이트 읽기 — `src/main.cpp:96-97`; 헤더 읽기 —
     `src/main.cpp:99-101`.
   - 헤더 JSON 파싱 후 텐서별 `data_offsets` 수집, `max_offset` 산정 —
     `src/main.cpp:103-118`.
   - `cudaMalloc` 으로 가중치 바이트 전체를 한 번에 잡기 —
     `src/main.cpp:120-121`; CPU 버퍼에 읽고 H2D 복사 — `src/main.cpp:123-127`.
   - 텐서 이름→포인터 매핑: `weights.embed_tokens`·`norm` 과 레이어별
     layernorm/MLP/attention 가중치 12종 — `src/main.cpp:132-145`.
3. RoPE 주파수 초기화: `init_rope_frequencies(HEAD_DIM, MAX_SEQ_LEN, 500000.0f,
   32.0f, 1.0f, 4.0f, 8192)` — `src/main.cpp:572`; 구현은 `d_inv_freq`,
   `d_cos_table`, `d_sin_table` 세 디바이스 배열을 할당·채움 —
   `src/kernels.cu:96-152`.
4. paged KV cache 할당자:
   - `kv_cache` 2GB `cudaMalloc` — `src/main.cpp:575-576`.
   - `free_blocks` 를 0..65535 로 채우기 — `src/main.cpp:577-578`.
   - `block_table` 전부 `-1` 초기화, `block_table_gpu` `cudaMalloc` —
     `src/main.cpp:579-581`.
5. 요청 큐: 채팅 템플릿 토큰 ID 4개 프롬프트를 `queue` 에 push —
   `src/main.cpp:583-594`.
6. 슬롯·배치 상태:
   - `is_slot_free(BATCH_SIZE, true)` — `src/main.cpp:597`.
   - `generated_tokens`, `last_generated_tokens`, `current_prompt_len` —
     `src/main.cpp:599-601`.
   - `active_slots`, `active_tokens` — `src/main.cpp:604-605`.
   - `gpu_active_slots`, `gpu_seq_lens` `cudaMalloc` — `src/main.cpp:607-610`.
7. 버퍼 할당:
   - prefill 공용: `gpu_input_tokens`(`617-618`), `input_embeddings`(`619-620`),
     `hidden_state`(`622-623`), `rms_norms`(`625-626`), `buf_2048_1`(`628-629`,
     q_proj·attn_scores_v 공유), `k_proj_temp_buf`(`635-636`),
     `v_proj_temp_buf`(`638-639`), `prefill_attn_scores`(`647-648`,
     `MAX_PROMPT_LEN^2 × NUM_Q_HEADS`), `buf_2048_2`(`656-657`,
     o_proj·down 공유), `gate`(`662-663`), `up`(`667-668`),
     `embed_proj`(`676-677`), `embed_proj_cpu`(`681-682`).
   - decode 전용: `gpu_last_tokens`(`684-685`), `k_proj_batched_buffer`(`689-690`),
     `v_proj_batched_buffer`(`692-693`).
8. 초기 prefill 루프 — `src/main.cpp:695-708`: `slot < BATCH_SIZE && !queue.empty()`
   동안 빈 슬롯에 `prefill`. `BATCH_SIZE=2`, 큐 4개이므로 슬롯 0·1 이 채워지고
   프롬프트 2·3 은 큐에 남는다.

### 2. prefill 경로(`prefill`, `src/main.cpp:150-553`)

1. 큐에서 프롬프트를 꺼내 `prompt`/`prompt_len` 설정, `is_slot_free[slot]=false`
   — `src/main.cpp:152-155`.
2. 토큰 H2D 복사 — `src/main.cpp:157`; `embeddingGather`(임베딩) —
   `src/main.cpp:158`; 커널 구현 `src/kernels.cu:32-53`.
3. `hidden_state` = `input_embeddings` D2D 복사 — `src/main.cpp:160-163`.
4. 레이어 루프(16개, `src/main.cpp:164-493`):
   - RMSNorm: `rmsNorm` — `src/main.cpp:166`; 커널 `src/kernels.cu:55-94`.
   - Q 투영 cuBLAS — `src/main.cpp:178-196`; K 투영 — `src/main.cpp:201-219`;
     V 투영 — `src/main.cpp:222-240`.
   - RoPE: `rope(q_proj, ...)`, `rope(k_proj_temp_buf, ...)` — `src/main.cpp:244-245`;
     커널 `src/kernels.cu:173-222`.
   - K/V 를 paged 블록으로 분산: 프롬프트를 `BLOCK_SIZE=16` 단위로 잘라
     `block_table` 이 `-1` 이면 `free_blocks` 에서 물리 블록을 꺼내 기록하고
     (`src/main.cpp:264-277`) K·V 를 `cudaMemcpy`(D2D)로 `kv_cache` 블록에 기록 —
     `src/main.cpp:251-288`.
   - 어텐션 점수: Q 헤드 32개 각각에 cuBLAS `Q_head·K_head^T/sqrt(64)` —
     `src/main.cpp:301-327`.
   - `causalMask` — `src/main.cpp:329`; 커널 `src/kernels.cu:224-255`.
   - `softmax` — `src/main.cpp:331`; 커널 `src/kernels.cu:257-309`.
   - 점수×V: Q 헤드 32개 각각 cuBLAS — `src/main.cpp:343-371`.
   - O 투영 cuBLAS — `src/main.cpp:378-396`.
   - 잔차: `residualAdd(hidden_state, o_proj, ...)` — `src/main.cpp:399`;
     커널 `src/kernels.cu:311-329`.
   - post-attn RMSNorm — `src/main.cpp:401`.
   - SwiGLU: gate cuBLAS — `src/main.cpp:414-432`; up cuBLAS —
     `src/main.cpp:435-453`; `silu`(in-place) — `src/main.cpp:459`;
     커널 `src/kernels.cu:331-344`; down cuBLAS — `src/main.cpp:471-489`;
     잔차 `residualAdd` — `src/main.cpp:492`.
5. 최종 RMSNorm — `src/main.cpp:494`.
6. 로짓: `embed_tokens` cuBLAS — `src/main.cpp:509-527`.
7. `embed_proj` D2H 복사 — `src/main.cpp:529`.
8. CPU argmax(마지막 토큰 행만) — `src/main.cpp:533-543`; 출력 —
   `src/main.cpp:544`.
9. 상태 기록: `generated_tokens[slot]` push, `last_generated_tokens[slot]`,
   `current_prompt_len[slot] = prompt_len` — `src/main.cpp:546-548`.
10. `block_table` → `block_table_gpu` 전체 H2D 동기화 — `src/main.cpp:552`.

### 3. decode 루프(`main`, `src/main.cpp:720-1039`)

`while(true)` 의 각 반복은 다음 순서를 따른다.

1. `active_slots`·`active_tokens` 초기화 — `src/main.cpp:722-723`.
2. 슬롯별 처리 — `src/main.cpp:724-737`: 슬롯이 비어 있으면
   `generated_tokens[slot].clear()` 후 `prefill` 로 큐의 다음 프롬프트를 채우고
   (`src/main.cpp:726-734`), 모든 점유 슬롯을 `active_slots`·`active_tokens` 에
   push(`src/main.cpp:735-736`).
3. `num_active_slots==0` 이면 큐가 빌 때 break, 아니면 continue —
   `src/main.cpp:738-746`.
4. H2D: `gpu_last_tokens`(`749`), `gpu_active_slots`(`750`), `seq_lens`(=현재
   길이+1, `751-757`).
5. 임베딩: `embeddingGatherDecode` — `src/main.cpp:759`; 커널
   `src/kernels.cu:347-369`.
6. 레이어 루프(16개, `src/main.cpp:760-972`):
   - RMSNorm — `src/main.cpp:762`.
   - Q cuBLAS — `src/main.cpp:765-783`; K cuBLAS — `src/main.cpp:802-821`;
     V cuBLAS — `src/main.cpp:824-842`.
   - 슬롯별 `ropeDecode`(현재 위치 하나만 회전) — `src/main.cpp:844-849`;
     커널 `src/kernels.cu:371-405`.
   - 슬롯별 K/V 분산: `token_in_block_idx==0` 일 때만 새 블록 할당
     (`src/main.cpp:859-865`), 그 외 기존 블록 재사용 — `src/main.cpp:851-873`.
   - `block_table_gpu` H2D 동기화 — `src/main.cpp:876`.
   - `pagedAttention` — `src/main.cpp:878`; 커널 `src/kernels.cu:461-523`,
     런치 `src/kernels.cu:525-527`.
   - O cuBLAS — `src/main.cpp:882-900`; `residualAdd` — `src/main.cpp:902`;
     RMSNorm — `src/main.cpp:904`.
   - gate cuBLAS — `src/main.cpp:907-925`; up cuBLAS — `src/main.cpp:928-946`;
     `silu` — `src/main.cpp:948`; down cuBLAS — `src/main.cpp:951-969`;
     `residualAdd` — `src/main.cpp:971`.
7. 최종 RMSNorm — `src/main.cpp:974`; 로짓 cuBLAS — `src/main.cpp:976-994`.
8. `embed_proj` D2H — `src/main.cpp:996`.
9. 활성 슬롯별 CPU argmax·출력 — `src/main.cpp:1000-1014`.
10. 종료 처리 — `src/main.cpp:1015-1037`:
    - `<|end_of_text|>`(128001)·`<|eot_id|>`(128009) 또는
      `current_prompt_len == MAX_SEQ_LEN-1` 이면 슬롯 해제(`is_slot_free=true`),
      소유 블록을 `free_blocks` 로 반납, `block_table` 재초기화 —
      `src/main.cpp:1015-1031`.
    - 아니면 토큰 push, `current_prompt_len+1` — `src/main.cpp:1032-1037`.
11. 루프 반복(`src/main.cpp:720`).

### 4. 종료

- break 는 `src/main.cpp:739-744` 한 곳이며, 큐가 빈 상태에서 활성 슬롯이
  전부 사라졌을 때만 도달한다. 주석(`src/main.cpp:720`)은 상시 운영 서버를
  의도한다.
- break 후 `Ok bye!` 출력, `cublasDestroy`, `cudaDeviceSynchronize`,
  `return 0` — `src/main.cpp:1040-1044`.

## State changes

| 상태 | 초기값 | 변경 지점 |
| --- | --- | --- |
| `is_slot_free` | 전부 `true` — `src/main.cpp:597` | `prefill` 에서 `false` — `src/main.cpp:155`; EOS/EOT/길이한도에서 `true` — `src/main.cpp:1017` |
| `block_table` | 전부 `-1` — `src/main.cpp:579` | prefill 블록 할당 — `src/main.cpp:266-272`; decode 블록 할당 — `src/main.cpp:859-865`; 슬롯 해제 시 `-1` — `src/main.cpp:1022-1027` |
| `free_blocks` | `iota(0..NUM_BLOCKS-1)` — `src/main.cpp:577-578` | 할당 시 `pop_back` — `src/main.cpp:268-269`, `861-862`; 해제 시 `push_back` — `src/main.cpp:1025` |
| `block_table_gpu` | `cudaMalloc` — `src/main.cpp:581` | prefill 후 — `src/main.cpp:552`; decode 레이어마다 — `src/main.cpp:876`; 슬롯 해제 후 — `src/main.cpp:1030` |
| `current_prompt_len[slot]` | 0 — `src/main.cpp:601` | prefill 후 `= prompt_len` — `src/main.cpp:548`; decode 토큰마다 `+1` — `src/main.cpp:1036` |
| `last_generated_tokens[slot]` | 0 — `src/main.cpp:600` | prefill 후 — `src/main.cpp:547`; decode 토큰 추가 시 — `src/main.cpp:1034` |
| `generated_tokens[slot]` | 빈 벡터 — `src/main.cpp:599` | prefill 후 push — `src/main.cpp:546`; 재-prefill 전 clear — `src/main.cpp:732`; decode push — `src/main.cpp:1035` |
| `queue` | 프롬프트 4개 — `src/main.cpp:583-594` | `prefill` 진입부에서 pop — `src/main.cpp:154` |
| `kv_cache` | `cudaMalloc`(2GB) — `src/main.cpp:575-576` | prefill K/V 기록 — `src/main.cpp:280-287`; decode K/V 기록 — `src/main.cpp:866-872` |
| `hidden_state` | prefill 후 임베딩 — `src/main.cpp:160-163` | 레이어마다 잔차 누적 — `src/main.cpp:399`, `492`, `902`, `971`; decode 반복마다 `embeddingGatherDecode` 로 덮어씀 — `src/main.cpp:759` |
| `active_slots`·`active_tokens` | 빈 벡터 — `src/main.cpp:604-605` | decode 반복마다 재구성 — `src/main.cpp:722-737` |
| 버퍼 별칭 | `buf_2048_1` = q_proj·attn_scores_v — `src/main.cpp:628`; `buf_2048_2` = o_proj·down — `src/main.cpp:657` | prefill `src/main.cpp:177`, `343`, `377`, `470`; decode `src/main.cpp:763`, `878`, `880`, `950` |

`pagedAttention` 의 출력은 입력 `q_proj` 와 같은 버퍼(`buf_2048_1`)에 쓰인다.
각 스레드가 읽은 위치와 같은 위치에 쓴다(`src/kernels.cu:469` 대
`src/kernels.cu:522`)는 인덱스가 같아 in-place 가 안전하며, 이는
`src/main.cpp:878`(입력)과 `src/main.cpp:892`(O 투영의 입력)이 같은 버퍼를
가리키는 이유다.

## Branches and exits

- `checkGPUStatus`: `device_count == 0` 이면 오류 출력 후 `return 1` —
  `src/main.cpp:44-48`.
- `loadWeights`: `checkGPUStatus` 실패 시 `return 1` — `src/main.cpp:81-84`;
  `model.safetensors` 오픈 실패 시 `return 1` — `src/main.cpp:88-93`.
- `main`: `cublasCreate` 실패 시 `return 1` — `src/main.cpp:559-563`;
  `loadWeights` 실패 시 `return 1` — `src/main.cpp:566-569`.
- prefill 블록 할당: `block_table` 값이 `-1` 이면 `free_blocks` 에서 할당
  (`src/main.cpp:266-272`), 아니면 `assert(false)` — `src/main.cpp:275-277`.
- decode 블록 할당: `token_in_block_idx == 0` 일 때만 새 블록 할당, 아니면
  기존 블록 재사용 — `src/main.cpp:859-865`.
- 커널 가드(런치 생략): `rope` 스레드 수 1024 초과 시 — `src/kernels.cu:205-209`;
  `causalMask` 1024 초과 시 — `src/kernels.cu:241-245`; `softmax` 1024 초과 시 —
  `src/kernels.cu:295-299`; `softmaxDecode` 1024 초과 시 —
  `src/kernels.cu:444-448`.
- decode 루프: `num_active_slots == 0` 이면 큐가 빌 때 break, 아니면 continue —
  `src/main.cpp:738-746`.
- 슬롯 종료 분기: EOS/EOT 토큰 또는 `MAX_SEQ_LEN-1` 도달이면 슬롯 해제
  (`src/main.cpp:1015-1031`), 아니면 토큰 누적(`src/main.cpp:1032-1037`).
- 정상 종료 경로는 위 break 하나뿐이며, 초기화 실패는 `main` 의 `return 1`
  셋이다.

## Unresolved gaps

- `build.sh`, `run.sh`, `test.sh`, `full_test.sh` 본문이 첨부에 없다.
  `full_test.sh → test.sh → build.sh → run.sh` 파이프라인은 `guide.md:114-125`
  인용이며, 스크립트의 실제 명령·인자는 검증할 수 없다.
- `src/kernels.cuh`(함수 선언)와 `src/cuda_to_hip.h`(bf16·BLAS shim)가 첨부에
  없다. `src/main.cpp:4,8` 와 `CMakeLists.txt:57,62` 가 존재를 시사하지만
  내용은 직접 인용하지 못한다.
- `MAX_NEW_TOKENS_GENERATED = 20`(`src/main.cpp:12`)은 선언만 있고 다른 곳에서
  참조되지 않는다. 생성 종료는 오직 EOS/EOT 토큰 또는 `MAX_SEQ_LEN-1`
  도달(`src/main.cpp:1015`)로만 정해진다.
- `pagedAttentionKernel` 은 `HEAD_DIM=64` 스레드(2 워프)를 가정한다.
  `dot_products[2]` 와 스레드 0·32 협력(`src/kernels.cu:463,494-506`)은
  `HEAD_DIM` 이 64 가 아니면 성립하지 않는다.
- `softmaxKernelDecode` 의 stride 는 `MAX_SEQ_LEN`(`src/kernels.cu:413`)인 반면
  prefill `softmaxKernel` 은 `num_tokens`(`src/kernels.cu:262`)이다. decode
  버퍼의 행간 레이아웃 가정이 여기에 깔려 있다.
- `ropeKernelDecode` 는 호출마다 `theta` 를 재계산하고 상수 `500000.0` 과
  32(헤드 절반)을 하드코딩한다(`src/kernels.cu:376-378`). prefill 계열의
  사전 계산 테이블(`d_cos_table`/`d_sin_table`)을 쓰지 않는다.
- decode 스텝은 매 레이어마다 `block_table` 전체를 H2D 복사한다
  (`src/main.cpp:876`, TODO 주석 `src/main.cpp:551`).
- `embed_proj` 의 argmax 는 CPU 루프로, 슬롯당 매 스텝 `VOCAB_SIZE` 탐색이다
  (`src/main.cpp:1000-1012`; TODO `src/main.cpp:531-532,686`).
- CUDA 툴체인·GPU 부재로 어떤 커널도 실행 검증되지 않았다(`guide.md:75-87`).
  위 모든 순서·상태 주장은 소스 인용으로만 뒷받침되며 런타임 수치가 없다.
- gated 가중치 `model.safetensors`(`meta-llama/Llama-3.2-1B-Instruct`)를
  확보할 수 없어 실행 검증은 후속 과제로 남는다(`guide.md:78-79,87`).

## Source paths

첨부 파일 전부(고정 revision `e25bf1994efa90bc98b721ba7c527402f86fbeaf` 기준):

- `CMakeLists.txt:1-75` — 언어/CUDA 분기(10-16, 24), 번역 단위(49-52),
  include 경로(62-63), 링크(71-74).
- `src/main.cpp:1-1044` — 상수(12-38), `checkGPUStatus`(40-62),
  `Weights`(64-77), `loadWeights`(79-147), `prefill`(150-553), `main`(555-1044).
- `src/kernels.cu:1-528` — prefill 커널(32-344): `embeddingGatherKernel`,
  `rmsNormKernel`, `ropeKernel_llama3`, `causalMaskKernel`, `softmaxKernel`,
  `residualKernel`, `siluKernel`; decode 커널(347-523):
  `embeddingGatherKernelDecode`, `ropeKernelDecode`, `softmaxKernelDecode`,
  `pagedAttentionKernel`; 주파수 초기화(96-152).
- `guide.md:112-125` — 대표 실행 경로; `guide.md:75-87` — 실행 제약(정적
  검증 채택).