# 실행 추적: jmaczan/tiny-vllm

고정 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61`의 실행 경로를 첨부된
`src/main.cpp`, `src/kernels.cu`, `CMakeLists.txt`만으로 따라간다. 각 주장에는
소스 경로와 줄 범위를 인용한다. 이 문서는 저장소를 실행하지 않았고, 첨부되지
않은 파일(`src/kernels.cuh`, `src/cuda_to_hip.h` 등)의 내용은 추정하지 않는다.

## Entrypoint

- 빌드 산출물은 `tiny-vllm` 실행 파일이며 `src/main.cpp`와 `src/kernels.cu`를
  소스로 한다 (`CMakeLists.txt:49-52`).
- C++ 진입점은 `int main(int argc, char *argv[])`이다 (`src/main.cpp:555`).
- `main`의 첫 동작은 cuBLAS 핸들 생성 `cublasCreate(&cublas_handle)`이며,
  실패 시 `1`을 반환한다 (`src/main.cpp:557-563`).
- 이어서 가중치 로딩 `loadWeights(weights)` (`src/main.cpp:566`), RoPE 주파수
  초기화 `init_rope_frequencies(...)` (`src/main.cpp:572`)를 호출한다.
- 그 다음 paged attention용 `kv_cache`, `free_blocks`, `block_table`,
  `block_table_gpu`를 준비한다 (`src/main.cpp:574-581`).
- 이후 하드코딩된 프롬프트 4개를 큐에 넣고 (`src/main.cpp:583-594`), 배치 상태
  벡터와 대형 작업 버퍼를 할당한 뒤 (`src/main.cpp:596-693`), prefill 루프
  (`src/main.cpp:695-708`)와 무한 decode 루프 (`src/main.cpp:720-1039`)를 실행한다.
- 정상 종료 시 `"Ok bye!"` 출력 후 `cublasDestroy`, `cudaDeviceSynchronize`,
  `return 0`을 수행한다 (`src/main.cpp:1040-1044`).

## Ordered steps

1. **cuBLAS 초기화와 가중치 로딩** — `cublasCreate` 성공을 확인한 뒤
   (`src/main.cpp:557-563`) `loadWeights`를 호출한다 (`src/main.cpp:566`).
   `loadWeights`는 먼저 `checkGPUStatus`로 CUDA 장치 유무와 속성을 확인하고
   (`src/main.cpp:79-84`, `src/main.cpp:40-62`), `model.safetensors`를 이진
   모드로 연다 (`src/main.cpp:87`). 8바이트 헤더 크기 (`src/main.cpp:96-97`)와
   JSON 헤더 (`src/main.cpp:99-104`)를 읽어 텐서별 오프셋 맵과 `max_offset`을
   만든다 (`src/main.cpp:105-118`). `cudaMalloc`으로 장치 버퍼를 잡고
   (`src/main.cpp:120-121`), CPU 벡터로 읽은 뒤 (`src/main.cpp:123-125`)
   H2D 복사한다 (`src/main.cpp:127`). 마지막으로 `weights`의 각 포인터를 단일
   할당 내 오프셋으로 설정한다 (`src/main.cpp:132-145`).

2. **RoPE 주파수 사전 계산** — `init_rope_frequencies`가 `inv_freq`를 만들고
   (`src/kernels.cu:100-105`), llama3 방식 저/고주파 스케일링을 적용한
   `inv_freq_llama`를 계산한다 (`src/kernels.cu:106-124`). 위치별 cos/sin 테이블을
   채우고 (`src/kernels.cu:129-144`), `d_inv_freq`, `d_cos_table`, `d_sin_table`을
   장치에 복사한다 (`src/kernels.cu:126-151`). 호출 지점은 `src/main.cpp:572`다.

3. **paged KV-cache 할당자 준비** — `kv_cache`를 `KV_CACHE_SIZE_BYTES`만큼
   `cudaMalloc`하고 (`src/main.cpp:575-576`), `free_blocks`를 0..`NUM_BLOCKS-1`로
   채우며 (`src/main.cpp:577-578`), `block_table`을 전부 `-1`로 초기화하고
   (`src/main.cpp:579`), `block_table_gpu`를 할당한다 (`src/main.cpp:580-581`).

4. **프롬프트 큐 구성** — 길이 17, 14, 13, 14의 토큰 ID 벡터 4개를 `queue`에
   넣는다 (`src/main.cpp:583-594`). 토크나이저 호출은 코드에 없다.

5. **배치 상태와 작업 버퍼 할당** — `is_slot_free`를 `BATCH_SIZE`만큼 `true`로
   (`src/main.cpp:597`), `generated_tokens`, `last_generated_tokens`,
   `current_prompt_len`을 준비한다 (`src/main.cpp:599-601`). decode용
   `active_slots`/`active_tokens` (`src/main.cpp:604-605`)와 GPU 인덱스/시퀀스
   길이 버퍼 (`src/main.cpp:607-610`), 토큰·임베딩·hidden·RMSNorm·Q/K/V·attention
   score·MLP·logit 버퍼를 순서대로 할당한다 (`src/main.cpp:617-693`).
   `buf_2048_1`은 `q_proj`와 `attn_scores_v`가 공유하고 (`src/main.cpp:628`),
   `buf_2048_2`는 `o_proj`와 `down`이 공유한다 (`src/main.cpp:656`).

6. **초기 prefill 루프** — 각 slot에 대해 비어 있으면 `prefill(...)`을 호출한다
   (`src/main.cpp:695-708`). 이 시점에 최대 `BATCH_SIZE`개의 프롬프트가
   소비된다.

7. **prefill 함수 실행** — 큐에서 프롬프트를 꺼내고 slot을 점유 처리한다
   (`src/main.cpp:152-155`). 토큰을 GPU로 복사한 뒤 (`src/main.cpp:157`)
   `embeddingGather`로 임베딩을 모으고 (`src/main.cpp:158`, `src/kernels.cu:42-53`),
   `hidden_state`로 복사한다 (`src/main.cpp:160-163`). 레이어마다 input
   RMSNorm (`src/main.cpp:166`, `src/kernels.cu:84-94`), Q/K/V projection
   (`src/main.cpp:177-240`), RoPE (`src/main.cpp:244-245`, `src/kernels.cu:202-222`),
   K/V 블록 scatter (`src/main.cpp:251-288`), Q head별 attention score
   (`src/main.cpp:301-327`), causal mask (`src/main.cpp:329`,
   `src/kernels.cu:239-255`), softmax (`src/main.cpp:331`, `src/kernels.cu:293-309`),
   score×V (`src/main.cpp:344-371`), output projection (`src/main.cpp:377-396`),
   residual (`src/main.cpp:399`), post-attention RMSNorm (`src/main.cpp:401`),
   gate/up (`src/main.cpp:414-453`), SiLU (`src/main.cpp:459`), down
   (`src/main.cpp:470-489`), residual (`src/main.cpp:492`)을 수행한다. 레이어
   루프 후 최종 RMSNorm (`src/main.cpp:494`)과 logit projection
   (`src/main.cpp:509-527`)을 거친다. logit을 CPU로 복사하고
   (`src/main.cpp:529`) 마지막 토큰 행에 대해 CPU argmax를 수행하며
   (`src/main.cpp:533-543`), 생성 토큰과 상태를 기록한 뒤 (`src/main.cpp:546-548`)
   `block_table`을 GPU로 동기화한다 (`src/main.cpp:552`).

8. **decode 루프 진입** — `while (true)` 루프가 매 반복마다 `active_slots`와
   `active_tokens`를 비우고 (`src/main.cpp:720-723`), 각 slot이 비어 있으면
   큐에서 prefill하고 (`src/main.cpp:724-734`) 활성 slot/마지막 토큰을
   모은다 (`src/main.cpp:735-736`).

9. **decode forward** — 활성 슬롯 수가 0이 아니면 (`src/main.cpp:738-746`)
   마지막 토큰/슬롯/시퀀스 길이를 GPU로 복사한다 (`src/main.cpp:749-757`).
   `embeddingGatherDecode` (`src/main.cpp:759`, `src/kernels.cu:358-369`) 후
   레이어마다 input RMSNorm (`src/main.cpp:762`), Q/K/V projection
   (`src/main.cpp:765-842`), `ropeDecode` (`src/main.cpp:844-849`,
   `src/kernels.cu:388-405`), K/V 블록 scatter (`src/main.cpp:852-873`),
   `block_table` 동기화 (`src/main.cpp:876`), `pagedAttention`
   (`src/main.cpp:878`, `src/kernels.cu:525-528`), output projection
   (`src/main.cpp:882-900`), residual (`src/main.cpp:902`), post-attention
   RMSNorm (`src/main.cpp:904`), gate/up (`src/main.cpp:907-946`), SiLU
   (`src/main.cpp:948`), down (`src/main.cpp:951-969`), residual
   (`src/main.cpp:971`)을 수행한다.

10. **decode 출력과 종료 판정** — 최종 RMSNorm (`src/main.cpp:974`), logit
    projection (`src/main.cpp:976-994`), D2H 복사 (`src/main.cpp:996`) 후 활성
    슬롯별로 CPU argmax를 수행한다 (`src/main.cpp:1000-1014`). 결과가 종료
    토큰이거나 최대 길이면 slot을 해제하고 블록을 반납하며
    (`src/main.cpp:1015-1031`), 아니면 토큰을 append하고 길이를 증가시킨다
    (`src/main.cpp:1032-1037`). 이후 루프 처음으로 돌아간다
    (`src/main.cpp:720`).

## State changes

- **가중치 포인터**: `weights.embed_tokens`, `norm`, 레이어별 포인터가 단일
  `model_weights` 할당의 오프셋으로 설정된다 (`src/main.cpp:132-145`).
- **큐**: `prefill`이 `queue.front()`를 복사하고 `queue.pop()`하므로 호출마다
  큐가 줄어든다 (`src/main.cpp:152-154`).
- **slot 점유**: `is_slot_free[slot]`이 prefill 시작 시 `false`
  (`src/main.cpp:155`), 종료 조건 충족 시 `true`로 바뀐다
  (`src/main.cpp:1017`).
- **블록 테이블**: `block_table`의 `-1` 항목이 물리 블록 인덱스로 채워지고
  (`src/main.cpp:271`, `src/main.cpp:864`), 해제 시 다시 `-1`이 된다
  (`src/main.cpp:1026`).
- **프리 블록 스택**: 할당 시 `free_blocks.back()`을 꺼내고
  (`src/main.cpp:268-269`, `src/main.cpp:861-862`), 해제 시 `push_back`한다
  (`src/main.cpp:1025`).
- **KV 캐시**: 물리 블록 오프셋에 K와 V가 기록된다
  (`src/main.cpp:280-287`, `src/main.cpp:866-872`).
- **생성 이력**: `generated_tokens[slot]`은 슬롯 재사용 시 비워지고
  (`src/main.cpp:732`), 토큰이 push된다 (`src/main.cpp:546`,
  `src/main.cpp:1035`). `last_generated_tokens[slot]`은 argmax 결과로 갱신된다
  (`src/main.cpp:547`, `src/main.cpp:1034`).
- **시퀀스 길이**: `current_prompt_len[slot]`은 prefill 시 프롬프트 길이로
  설정되고 (`src/main.cpp:548`), decode 성공마다 1 증가한다
  (`src/main.cpp:1036`).
- **GPU 블록 테이블**: `block_table_gpu`는 prefill 끝 (`src/main.cpp:552`),
  decode의 매 레이어 (`src/main.cpp:876`), 해제 시 (`src/main.cpp:1030`)에
  호스트 배열로부터 덮어써진다.
- **작업 버퍼**: `hidden_state`, `rms_norms`, `q_proj`, `gate`, `up`, `down`,
  `o_proj`, `attn_scores_v`가 레이어/스텝마다 덮어써진다. `buf_2048_1`
  (`src/main.cpp:628`)과 `buf_2048_2` (`src/main.cpp:656`)는 용도가 겹치는
  두 버퍼다.
- **RoPE 테이블**: `d_inv_freq`, `d_cos_table`, `d_sin_table`은 초기화 시 한 번
  설정된다 (`src/kernels.cu:126-151`).
- **호스트 logit**: `embed_proj_cpu`가 D2H 복사로 채워진다
  (`src/main.cpp:529`, `src/main.cpp:996`).

## Branches and exits

- **CUDA 장치 없음**: `device_count == 0`이면 오류를 출력하고 `1`을 반환한다
  (`src/main.cpp:44-48`).
- **safetensors 열기 실패**: 파일이 열리지 않으면 `1`을 반환한다
  (`src/main.cpp:88-93`).
- **cuBLAS 초기화 실패**: `CUBLAS_STATUS_SUCCESS`가 아니면 `1`을 반환한다
  (`src/main.cpp:559-563`).
- **가중치 로딩 실패**: `loadWeights`가 0이 아니면 `1`을 반환한다
  (`src/main.cpp:566-569`).
- **prefill 중 기존 블록 발견**: prefill에서 `block != -1`이면
  `assert(false)`로 중단한다 (`src/main.cpp:273-277`). decode에서는
  `token_in_block_idx == 0`일 때만 새 블록을 할당한다
  (`src/main.cpp:859-865`).
- **빈 배치 종료**: `num_active_slots == 0`이고 큐도 비면 `while(true)`를
  빠져나간다 (`src/main.cpp:738-746`).
- **생성 종료 조건**: argmax 결과가 `END_OF_TEXT_TOKEN_ID` 또는
  `EOT_ID_TOKEN_ID`이거나 `current_prompt_len == MAX_SEQ_LEN - 1`이면 슬롯을
  해제한다 (`src/main.cpp:1015-1031`). 그 외에는 토큰을 이어붙인다
  (`src/main.cpp:1032-1037`).
- **커널 런치 가드**: 스레드 수가 1024를 넘으면 커널을 실행하지 않고
  메시지만 출력한다 — RoPE prefill (`src/kernels.cu:205-209`), causal mask
  (`src/kernels.cu:241-245`), softmax prefill (`src/kernels.cu:295-299`),
  `ropeDecode` (`src/kernels.cu:391-395`), `softmaxDecode`
  (`src/kernels.cu:444-448`).
- **빌드 분기**: `USE_HIP` 옵션에 따라 CUDA/HIP 언어, 아키텍처, 라이브러리가
  갈린다 (`CMakeLists.txt:3`, `CMakeLists.txt:10-16`,
  `CMakeLists.txt:21-25`, `CMakeLists.txt:32-38`, `CMakeLists.txt:42-47`,
  `CMakeLists.txt:54-60`, `CMakeLists.txt:65-75`).
- **정상 종료**: 큐 소진과 슬롯 해제가 겹치면 루프를 탈출해 `return 0`에
  도달한다 (`src/main.cpp:1040-1044`).

## Unresolved gaps

- `src/main.cpp:8`과 `src/kernels.cu:2`가 `"kernels.cuh"`를 include하지만 이
  파일은 첨부되지 않아 함수 선언과 시그니처의 정본을 확인할 수 없다.
- `src/main.cpp:4`와 `src/kernels.cu:1`이 `"cuda_to_hip.h"`를 include하지만
  첨부되지 않아 CUDA/HIP 매크로가 실행 경로에 미치는 영향을 확정할 수 없다.
- 토크나이저 경로가 없다. 프롬프트는 하드코딩된 토큰 ID 벡터다
  (`src/main.cpp:583-594`).
- 모델 경로가 `"model.safetensors"`로 하드코딩되어 있다 (`src/main.cpp:87`).
- `MAX_NEW_TOKENS_GENERATED`가 선언만 되고 사용되지 않는다
  (`src/main.cpp:12`). 실제 종료는 EOT 또는 `MAX_SEQ_LEN`에만 달려 있다
  (`src/main.cpp:1015`).
- `ropeKernel_llama3`는 `init_rope_frequencies`의 스케일된 cos/sin 테이블을
  사용하지만 (`src/kernels.cu:173-200`, `src/kernels.cu:129-151`),
  `ropeKernelDecode`는 `500000.0`과 `HEAD_DIM`으로 theta를 다시 계산해 llama3
  주파수 스케일링을 반영하지 않는다 (`src/kernels.cu:371-384`). 첨부 파일만으로
  이 불일치의 의도나 검증 여부를 알 수 없다.
- prefill 경로의 `softmax`, `causalMask`, `rope`는 1024 초과 시 조용히
  건너뛴다 (`src/kernels.cu:241-245`, `src/kernels.cu:295-299`,
  `src/kernels.cu:205-209`). `MAX_SEQ_LEN`이 2048이므로 장문에서의 동작은
  첨부 파일로 확정할 수 없다.
- `prefill_attn_scores`는 `MAX_PROMPT_LEN * MAX_PROMPT_LEN * NUM_Q_HEADS`로
  할당되어 프롬프트 길이가 `MAX_PROMPT_LEN`(512)을 넘는 경우를 다루지 않는다
  (`src/main.cpp:647-648`, `src/main.cpp:30`).
- `BATCH_SIZE`가 2로 고정되어 있고 (`src/main.cpp:29`), `MAX_SEQ_LEN`,
  `BLOCK_SIZE`, `MAX_PROMPT_LEN` 등은 TODO로 표시된 하드코딩 값이다
  (`src/main.cpp:28-32`).
- 블록 할당자는 `NUM_BLOCKS`(=65536)개의 블록을 전제로 하지만
  (`src/main.cpp:37`, `src/main.cpp:577`) 실제 수요 대비 고갈 검사가 첨부
  코드에는 없다.
- argmax가 CPU에서 수행되어 매 스텝마다 logit 전체를 D2H 복사한다
  (`src/main.cpp:529`, `src/main.cpp:996`, `src/main.cpp:533-543`,
  `src/main.cpp:1000-1014`).
- paged attention, batching, softmax 커널에 대한 검증 경로는 첨부 파일에 없다.
- `while (true)` 서버 루프는 큐가 비고 모든 슬롯이 해제될 때만 종료된다
  (`src/main.cpp:720`, `src/main.cpp:738-746`).

## Source paths

- `src/main.cpp` — 진입점, 가중치 로딩, prefill/decode forward, 배치 및 블록 관리
  (`src/main.cpp:40-62`, `src/main.cpp:79-147`, `src/main.cpp:149-553`,
  `src/main.cpp:555-1044`)
- `src/kernels.cu` — embedding gather, RMSNorm, RoPE, causal mask, softmax,
  residual, SiLU, paged attention 커널 (`src/kernels.cu:32-528`)
- `CMakeLists.txt` — `tiny-vllm` 빌드 타깃과 CUDA/HIP 분기
  (`CMakeLists.txt:1-75`)
- `guide.md` — 시리즈 범위와 증거 규칙 (`guide.md:3-29`)
