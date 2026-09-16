# tiny-vLLM 코드 읽기 시리즈 챕터 브리프

이 문서는 `series.yaml` 의 7개 장을 `evidence/claims.md` 의 21개 claim 과
`series.yaml` 의 승인된 시각 자료(visuals) 기록만으로 요약한다. 고정 revision 은
`e25bf1994efa90bc98b721ba7c527402f86fbeaf` 이며(`guide.md:6`), 모든 장의
`required_evidence.runtime` 은 0 이다(`guide.md:85`). 실행 결과 파일과
`runtime-claims.json`, `evidence/assets.json` 은 존재하지 않는다. 따라서 보존해야
할 execution ID 는 없고, 어떤 claim 도 측정값을 전제하지 않는다(Claim 21,
`claims.md:389-403`). 커널 실행 시간·처리량·메모리 사용량은 이 시리즈에서
측정하지 않는다(`guide.md:86-87`).

---

## Chapter 1

### Scope
- **핵심 질문:** 이 저장소에서 제품으로 빌드되는 것은 정확히 무엇이고, 한 번의 실행은 어디서 시작하는가?(`series.yaml:8`)
- **code_scopes:** `CMakeLists.txt`, `build.sh`, `run.sh`, `test.sh`, `full_test.sh`, `src/cuda_to_hip.h`(`series.yaml:10-15`)
- **required_evidence:** code 4 / tests 0 / runtime 0(`series.yaml:17-19`)
- 저장소가 스스로 규정한 단 하나의 대표 실행 경로는 `full_test.sh` 이며, 이것이 `test.sh` → `build.sh` → `run.sh` 로 이어진다(`guide.md:112-125`).

### Claims
- **Claim 1 (product-translation-units, High):** 제품 바이너리 `tiny-vllm` 은 `src/main.cpp` 와 `src/kernels.cu` 두 번역 단위로만 빌드된다. `include/json.hpp` 는 제품 코드가 아니라 include 경로로 들어오는 단일 헤더 의존이다(`claims.md:13-29`).
  - 근거: `trace.md:17-22` 가 `CMakeLists.txt:49-52` 의 `add_executable(tiny-vllm src/main.cpp src/kernels.cu)` 와 `CMakeLists.txt:62-63` 의 include 경로를 인용한다. `src/main.cpp:4-8` 이 `cuda_to_hip.h`·`json.hpp`·`kernels.cuh` 를 include 한다.
  - 한계: `CMakeLists.txt` 본문은 첨부에 없어 위 사실은 `trace.md` 인용으로만 접근한다.
- **Claim 2 (dual-backend-branch, Medium):** `USE_HIP` 옵션이 같은 CUDA 소스를 hipcc/hipBLAS 경로로 돌리며, `src/cuda_to_hip.h` 가 bfloat16 타입과 BLAS 호출의 CUDA/HIP 차이를 흡수하는 shim 이다(`claims.md:33-48`).
  - 근거: `trace.md:19` 가 `CMakeLists.txt:10-16` 의 언어 분기를 인용한다. `src/main.cpp:4` 와 `src/kernels.cu:1` 이 `cuda_to_hip.h` 를 첫 include 로 끌어오고, 제품 전역에서 `__nv_bfloat16` 을 쓴다(`src/main.cpp:66-76`, `src/kernels.cu:32-40`).
  - 한계: 분기·shim 의 구체적 코드는 첨부에 없어 `recon.md:33-36` 의 서술에 의존한다.

### Exclusions
- `include/json.hpp` 는 제3자 단일 헤더(`JSON for Modern C++ 3.12.0`, MIT)로 제품 소스에서 제외되며, 분석 대상 소스 바이트의 92% 를 차지해 규모 산정에서도 제외된다(결정 002, `guide.md:48`).
- `check.sh`, `ncu.sh`, `nsys.sh` 는 compute-sanitizer 와 Nsight 프로파일링용으로 GPU·sudo 를 요구하며 이 시리즈는 측정을 수행하지 않아 제외된다(결정 003, 006, `guide.md:49`).
- `.vscode/`, `assets/column-row-major.png` 는 편집기 설정과 그림 자원으로 구현 근거가 아니다(`guide.md:50`).

### Visuals
- **product-translation-units (table):**
  - "제품 바이너리는 `src/main.cpp` 와 `src/kernels.cu` 두 번역 단위로만 만들어진다" → Claim 1 이 직접 뒷받침.
  - "`include/json.hpp` 는 제품 코드가 아니라 include 경로로 들어오는 단일 헤더 의존이다" → Claim 1 이 직접 뒷받침.
- **dual-backend-branch (block-diagram):**
  - "`USE_HIP` 옵션이 같은 CUDA 소스를 hipcc/hipBLAS 경로로 돌린다" → Claim 2 가 뒷받침.
  - "`src/cuda_to_hip.h` 가 bfloat16 타입 차이를 흡수하는 shim 이다" → Claim 2 가 뒷받침.

---

## Chapter 2

### Scope
- **핵심 질문:** safetensors 가중치를 어떻게 읽어 GPU 메모리에 올리고, 어떤 버퍼를 미리 잡아 두는가?(`series.yaml:36`)
- **code_scopes:** `src/main.cpp`(`series.yaml:37-38`)
- **required_evidence:** code 3 / tests 0 / runtime 0(`series.yaml:42-44`)
- 이 장이 버퍼 크기 산정과 별칭 설계의 근거를 유일하게 소유한다(`guide.md:194`).

### Claims
- **Claim 3 (weights-safetensors-load, High):** `loadWeights` 는 `model.safetensors` 를 열어 헤더 JSON 의 `data_offsets` 로 각 텐서의 경계와 `max_offset` 을 산정하고, 가중치 바이트 전체를 한 번의 `cudaMalloc` 으로 잡은 뒤 CPU 버퍼를 거쳐 H2D 로 복사한다. 이후 텐서 이름→오프셋으로 `Weights` 구조체의 포인터를 채운다(`claims.md:54-68`).
  - 근거: `src/main.cpp:87-93`(이진 오픈), `src/main.cpp:96-101`(헤더 크기·헤더 읽기), `src/main.cpp:103-118`(`data_offsets`·`max_offset`), `src/main.cpp:120-127`(`cudaMalloc`·H2D), `src/main.cpp:132-145`(포인터 매핑).
  - 한계: GPU 가 없어 실제 적재는 실행되지 않았고, 가중치 파일은 gated 모델 `meta-llama/Llama-3.2-1B-Instruct` 여서 확보도 제약이다.
- **Claim 4 (buffer-allocation-table, High):** prefill 어텐션 점수 버퍼 `prefill_attn_scores` 는 `MAX_PROMPT_LEN(512) × MAX_PROMPT_LEN × NUM_Q_HEADS(32)` 크기로 잡힌다. decode 전용 버퍼(`gpu_last_tokens`, `k_proj_batched_buffer`, `v_proj_batched_buffer`)와 K/V 투영 임시 버퍼(`k_proj_temp_buf`, `v_proj_temp_buf`)는 별도로 할당되어 재사용된다(`claims.md:74-87`).
  - 근거: `src/main.cpp:647-648`(점수 버퍼), `src/main.cpp:683-693`(decode 전용 3개), `src/main.cpp:635-639`(K/V 임시 버퍼). 크기 상수는 `src/main.cpp:19-31`.
  - 한계: 크기는 컴파일 타임 상수에 의존하며 실행 시 실제 할당은 검증되지 않았다.
- **Claim 5 (buffer-aliasing, High):** `buf_2048_1` 은 q 프로젝션과 어텐션 점수×V 출력이 공유하고, `buf_2048_2` 는 O 프로젝션과 down 프로젝션이 공유한다. 같은 시점에는 한 역할만 쓰는 버퍼 별칭 설계다(`claims.md:93-104`).
  - 근거: `src/main.cpp:628-629`(q_proj·attn_scores_v 공유 주석), `src/main.cpp:656-657`(o_proj·down 공유 주석), `src/main.cpp:177`·`343`, `src/main.cpp:377`·`470`.

### Exclusions
- 실제 파일 적재는 실행 검증 대상이 아니며, gated 가중치 확보는 후속 근거 과제로만 남는다(`guide.md:87`, Claim 3 한계).
- `python/` 은 편의 주제가 아니라 `reference.py`·`rms_norm.py` 처럼 근거로만 쓴다(`guide.md:52-53`). 이 장의 버퍼 설계는 그 근거를 인용하지 않는다.

### Visuals
- **buffer-allocation-table (table):**
  - "prefill 어텐션 점수 버퍼는 `MAX_PROMPT_LEN` 제곱에 쿼리 헤드 수를 곱한 크기로 잡힌다" → Claim 4 가 직접 뒷받침.
  - "decode 전용 버퍼와 K/V 임시 버퍼는 별도로 할당되어 재사용된다" → Claim 4 가 직접 뒷받침.

---

## Chapter 3

### Scope
- **핵심 질문:** 프롬프트 토큰이 임베딩부터 다음 토큰 선택까지 어떤 커널 순서로 흐르는가?(`series.yaml:54`)
- **code_scopes:** `src/main.cpp`, `src/kernels.cu`, `src/kernels.cuh`, `tests/test_softmax.cu`(`series.yaml:56-59`)
- **required_evidence:** code 6 / tests 1 / runtime 0(`series.yaml:63-65`)
- prefill 은 이 장이 소유하며, 병렬 리덕션 설명의 소유 장이기도 하다(`guide.md:195-196`).

### Claims
- **Claim 6 (prefill-kernel-sequence, High):** 한 번의 `prefill` 호출이 프롬프트의 모든 토큰을 함께 처리하며, 레이어마다 embeddingGather → rmsNorm → Q/K/V 투영 → RoPE → K/V 블록 분산 → 헤드별 어텐션 점수 → causalMask → softmax → 점수×V → O 투영 → residual → post-attn rmsNorm → SwiGLU(gate/up/silu/down) → residual 순으로 커널과 cuBLAS 를 호출한다(`claims.md:110-127`).
  - 근거: `src/main.cpp:157-158`(embeddingGather), `src/main.cpp:166`(rmsNorm), `src/main.cpp:178-240`(Q/K/V), `src/main.cpp:244-245`(RoPE), `src/main.cpp:251-288`(K/V 분산), `src/main.cpp:301-327`(점수), `src/main.cpp:329`(causalMask), `src/main.cpp:331`(softmax), `src/main.cpp:343-396`(점수×V, O), `src/main.cpp:399`(residual), `src/main.cpp:401`(post-attn rmsNorm), `src/main.cpp:414-489`(SwiGLU), `src/main.cpp:492`(residual). 커널 구현은 `src/kernels.cu:32-344`.
  - 한계: 커널 호출 순서는 소스 순서 인용이며 실행 검증은 없다.
- **Claim 7 (rmsnorm-reduction, High):** `rmsNormKernel` 은 블록 내 1024 스레드가 shared memory(`rms_vector[1024]`)에 요소 제곱을 누적하고, 트리 리덕션으로 합을 구한 뒤 스레드 0 이 `sqrt(합/2048 + 1e-5)` 로 rms 를 계산해 정규화에 쓴다(`claims.md:133-145`).
  - 근거: `src/kernels.cu:55-81`, 특히 `src/kernels.cu:57`(shared), `src/kernels.cu:61`(제곱 합), `src/kernels.cu:62-71`(트리 리덕션), `src/kernels.cu:72-75`(`sqrt`), `src/kernels.cu:78-79`(정규화·가중치 곱). 런치는 `src/kernels.cu:84-86`.
  - 한계: 수치 교차검증 대상 `python/rms_norm_crosscheck.txt` 는 저장소에 커밋돼 있으나 첨부에 없고 실행도 없다.
- **Claim 8 (prefill-gqa-attention, High):** prefill 어텐션은 Q 헤드 32개 각각에 대해 K 헤드 인덱스 `i/4`(GQA)를 쓰고, `attn_alpha = 1/sqrt(64)` 로 스케일된 `Q_head·K_head^T` 점수를 cuBLAS 로 계산한다. causalMask 로 미래 마스킹한 뒤 행별 softmax 가 적용된다(`claims.md:151-163`).
  - 근거: `src/main.cpp:301-327`(헤드별 cuBLAS, `src/main.cpp:303` `k_head_idx = i / GQA_Q_TO_K_RATIO`), `src/main.cpp:329`(causalMask), `src/main.cpp:331`(softmax), `src/main.cpp:649-650`(`attn_alpha = 1.0f / 8.0f`). 상수는 `src/main.cpp:19-24`, 커널은 `src/kernels.cu:224-309`.
  - 한계: 참조 출력으로 선언된 `python/reference.py` 는 실행되지 않았고 수치 일치 검증도 없다.

### Exclusions
- `tests/test_softmax.cu` 는 `series.yaml` 의 code_scope 이고 `required_evidence.tests` 가 1 이지만(`series.yaml:59,65`), `claims.md` 에 이 테스트를 직접 다루는 claim 은 없다. 이 편의 검증은 커밋된 참조 출력 인용으로 대체한다(`guide.md:84`).
- softmax 의 병렬 리덕션은 이 장의 `rmsnorm-reduction` 으로 설명을 소유하고, 이후 편에서 재설명하지 않는다(`guide.md:196`).

### Visuals
- **prefill-kernel-sequence (mermaid):**
  - "prefill 은 embeddingGather, rmsNorm, RoPE, causalMask, softmax, residual, silu 커널을 순서대로 쓴다" → Claim 6 이 직접 뒷받침.
  - "한 번의 prefill 호출이 프롬프트의 모든 토큰을 동시에 처리한다" → Claim 6 이 직접 뒷받침.
- **rmsnorm-reduction (block-diagram):**
  - "`rmsNormKernel` 은 블록 내 병렬 리덕션으로 제곱 평균을 구한다" → Claim 7 이 직접 뒷받침.

---

## Chapter 4

### Scope
- **핵심 질문:** 왜 가중치 행렬곱을 cuBLAS 에 전치된 형태로 넘기는가?(`series.yaml:80`)
- **code_scopes:** `src/main.cpp`(`series.yaml:81-82`)
- **required_evidence:** code 3 / tests 0 / runtime 0(`series.yaml:85-87`)
- 이 장은 3편에서 처음 등장한 cuBLAS 호출을 되짚는 편이며 3편에 의존하고(`guide.md:144-146`), 열 우선 레이아웃 설명의 소유 장이다(`guide.md:197`).

### Claims
- **Claim 9 (column-row-major-trick, High):** 가중치·활성은 행 우선(row-major)인데 cuBLAS 는 열 우선을 가정하므로, 코드는 데이터를 그대로 두고 전치 플래그(`CUBLAS_OP_T`)와 lda/ldb/ldc 만으로 행렬곱을 호출해 결과를 전치된 채로 해석한다(`claims.md:171-185`).
  - 근거: `src/main.cpp:168-196`(Q 투영, `src/main.cpp:179-180` `CUBLAS_OP_T`/`CUBLAS_OP_N`, `src/main.cpp:184-194` lda·ldb·ldc), 동일 패턴 `src/main.cpp:201-219`(K), `src/main.cpp:378-396`(O), `src/main.cpp:414-432`(gate), `src/main.cpp:509-527`(로그잇).
  - 한계: `series.yaml` 의 시각 자료 `column-row-major-trick` 필수 주장 중 "alpha 와 beta 인자가 잔차 누적을 별도 커널 없이 처리한다" 는 고정 revision 의 코드로는 뒷받침되지 않는다. 모든 beta 는 0.0 이고(예: `src/main.cpp:631-632`), 잔차는 별도 커널 `residualAdd` 가 담당한다(`src/main.cpp:399`·`492`, `src/kernels.cu:311-329`). 유일한 비-1.0 alpha 는 어텐션 스케일 `1/8` 이다(`src/main.cpp:649`).

### Exclusions
- alpha·beta 인자가 잔차 누적을 처리한다는 시각 자료 주장은 **미지원 claim** 이므로 이 편에서 기각해야 한다(Claim 9 한계). 잔차 누적은 3편 소유의 `residualAdd` 커널이 담당한다(`guide.md:204`).
- 이 편은 `src/main.cpp` 만을 code_scope 로 하며, 커널 구현은 3편·5편이 소유한다.

### Visuals
- **column-row-major-trick (block-diagram):**
  - "cuBLAS 는 열 우선 레이아웃을 가정하므로 행 우선 데이터를 전치 플래그로 맞춘다" → Claim 9 가 직접 뒷받침.
  - "alpha 와 beta 인자가 잔차 누적을 별도 커널 없이 처리한다" → **Claim 9 의 한계가 이를 반박한다.** 모든 beta 는 0.0 이고 잔차는 `residualAdd` 커널이 담당하므로 이 필수 주장은 뒷받침 근거가 없다. 작성 시 이 주장을 제거하거나 Claim 9 한계에 따른 수정이 필요하다.

---

## Chapter 5

### Scope
- **핵심 질문:** decode 는 왜 prefill 과 다른 커널 변형을 쓰는가?(`series.yaml:99`)
- **code_scopes:** `src/kernels.cu`, `src/main.cpp`, `python/decode_test.py`(`series.yaml:101-103`)
- **required_evidence:** code 4 / tests 1 / runtime 0(`series.yaml:107-109`)
- decode 스텝은 이 장이 소유하며, KV 저장 구조는 6편으로 넘긴다(`guide.md:198`). 이 장은 prefill 과의 대비로만 성립하므로 3편에 의존한다(`guide.md:146`).

### Claims
- **Claim 10 (prefill-decode-contrast, High):** decode 계열 커널은 시퀀스 길이 1(슬롯당 토큰 1개)을 전제로 인덱싱을 단순화한다. `embeddingGatherKernelDecode` 는 `blockIdx.x` 를 슬롯으로 삼고, `ropeKernelDecode` 는 단일 블록으로 현재 위치 하나만 회전하며, `softmaxKernelDecode` 는 `MAX_SEQ_LEN` 스트라이드 레이아웃을 가정한다(`claims.md:193-207`).
  - 근거: `src/kernels.cu:347-369`(`embeddingGatherKernelDecode`, `src/kernels.cu:349`), `src/kernels.cu:371-405`(`ropeKernelDecode`, `src/kernels.cu:397` `<<<1, num_threads>>>`), `src/kernels.cu:408-458`(`softmaxKernelDecode`, `src/kernels.cu:413` `blockIdx.x * MAX_SEQ_LEN`). 대비되는 prefill 커널은 `src/kernels.cu:32-344`.
  - 한계: `softmaxKernelDecode` 의 `MAX_SEQ_LEN` 스트라이드(`src/kernels.cu:413`)는 prefill softmax 의 `num_tokens` 스트라이드(`src/kernels.cu:262`)와 다른 레이아웃 가정이며 실행 검증되지 않았다.
- **Claim 11 (rope-tables-recompute, High):** prefill 의 RoPE 는 `init_rope_frequencies` 가 사전 계산한 `d_cos_table`/`d_sin_table` 을 커널이 참조한다. 반면 `ropeKernelDecode` 는 호출마다 `theta = 1/pow(500000, 2i/64)` 를 다시 계산하고 상수 `500000.0` 과 `32` 를 하드코딩한다(`claims.md:213-226`).
  - 근거: `src/kernels.cu:96-152`(테이블 생성·H2D), `src/kernels.cu:173-200`(테이블 참조), `src/kernels.cu:371-384`(재계산, `src/kernels.cu:377-378` 상수), `src/main.cpp:572`(초기화 호출).
  - 한계: decode 계열의 재계산은 TODO 로 남아 있고(`trace.md:231-233`), 두 경로의 수치 일치는 실행 검증되지 않았다.
- **Claim 12 (decode-kv-scatter, High):** decode 는 슬롯별 K/V 투영을 `k_proj_batched_buffer`/`v_proj_batched_buffer` 에 모은 뒤, `token_in_block_idx == 0` 일 때만 `free_blocks` 에서 새 물리 블록을 할당하고 그 외에는 기존 블록의 위치에 `cudaMemcpy`(D2D)로 기록한다(`claims.md:231-244`).
  - 근거: `src/main.cpp:851-873`, 특히 `src/main.cpp:856-865`(블록 인덱스·할당 분기), `src/main.cpp:866-872`(캐시 기록). 배치 K/V 투영은 `src/main.cpp:802-842`.
  - 한계: 매 레이어마다 `block_table` 전체를 H2D 동기화하므로(`src/main.cpp:876`, TODO `src/main.cpp:551`) 이 비용은 측정되지 않았다.

### Exclusions
- `python/decode_test.py` 는 `series.yaml` 의 code_scope 이고 `required_evidence.tests` 가 1 이지만(`series.yaml:103,109`), `claims.md` 에 이 테스트를 직접 다루는 claim 은 없다. decode 계열 검증도 실행 없이 정적 인용으로만 한다.
- KV cache 의 블록 레이아웃과 paged attention 은 이 편에서 다루지 않고 6편으로 링크한다(`guide.md:198`).

### Visuals
- **prefill-decode-contrast (table):**
  - "decode 계열 커널은 시퀀스 길이 1 을 가정해 인덱싱을 단순화한다" → Claim 10 이 직접 뒷받침.
  - "`ropeKernelDecode` 는 전체 위치가 아니라 현재 위치 하나만 회전시킨다" → Claim 10·Claim 11 이 뒷받침.

---

## Chapter 6

### Scope
- **핵심 질문:** KV cache 를 블록으로 나누고 block table 로 참조하면 어텐션 커널은 무엇을 어떻게 읽는가?(`series.yaml:120`)
- **code_scopes:** `src/kernels.cu`, `src/main.cpp`(`series.yaml:121-122`)
- **required_evidence:** code 5 / tests 0 / runtime 0(`series.yaml:126-128`)
- block table 과 paged attention 은 KV 저장 구조의 유일한 소유 편이다(`guide.md:199`). 이 장은 5편에 의존한다(`series.yaml:125`).

### Claims
- **Claim 14 (block-table-indexing, High):** `pagedAttentionKernel` 은 `block_table_gpu` 를 거쳐 논리 블록을 물리 블록으로 바꾸고, `kv_cache` 의 비연속 블록을 순회하며 읽는다. 블록 할당은 `free_blocks` 의 `pop_back`, 반납은 `push_back` 으로 관리된다(`claims.md:269-280`).
  - 근거: `src/kernels.cu:478-485`(블록 순회·주소 계산), `src/kernels.cu:480`(block_table_gpu 참조). 할당·반납은 `src/main.cpp:264-277`(prefill), `src/main.cpp:859-865`(decode), `src/main.cpp:1015-1031`(반납). 초기화는 `src/main.cpp:575-581`.
  - 한계: 논리 인덱스 산술(`slot*N_LAYERS*MAX_BLOCKS_PER_SEQ + layer*MAX_BLOCKS_PER_SEQ + block_idx`)의 실행 정합성은 GPU 없이 검증되지 않았다.
- **Claim 15 (kv-block-layout, High):** KV cache 의 블록 하나는 `BLOCK_SIZE(16) × KV_DIM(512)` bf16 요소를 담으며, K 는 블록 시작에, V 는 `V_OFFSET = 16×512×2` 바이트 뒤에 배치된다. 블록당 `BLOCK_BYTES = V_OFFSET×2` 여서 2GB 캐시는 65536 블록으로 나뉜다(`claims.md:288-295`).
  - 근거: `src/main.cpp:33-37`(상수), `src/kernels.cu:17-19`(동일 상수), `src/kernels.cu:484-485`(K·V 주소 계산), `src/main.cpp:280-287`·`866-872`(기록).
  - 한계: 블록 크기·오프셋은 상수이며 실행 시 레이아웃 검증은 없다.
- **Claim 16 (paged-attention-inputs, High):** `pagedAttentionKernel` 은 `kv_cache`, `block_table_gpu`, `gpu_seq_lens`, `gpu_active_slots` 를 함께 받는다. `gpu_active_slots` 로 활성 슬롯을, `gpu_seq_lens` 로 각 슬롯의 현재 길이를 얻으며, 이 커널만 이 네 상태를 함께 쓴다(`claims.md:303-314`).
  - 근거: `src/kernels.cu:461`(시그니처), `src/kernels.cu:464-471`(활성 슬롯·길이 소비), `src/kernels.cu:480`(block_table 참조). 호출부 `src/main.cpp:878`, 인자 준비 `src/main.cpp:749-757`.
  - 한계: decode 계열에서 이 커널만 네 상태를 함께 받는 비대칭은 `guide.md:107-110` 의 서술과 일치하나, 다른 커널의 시그니처는 첨부에 없어 직접 비교되지 않았다.
- **Claim 17 (pagedattention-online-softmax, High):** `pagedAttentionKernel` 은 워프 내 `__shfl_down_sync` 5단계 리덕션으로 32개 스레드(1 워프)의 내적을 모으고, 스레드 0 과 스레드 32 가 `dot_products[2]` 로 두 워프의 합을 더해 `sqrt(HEAD_DIM)` 으로 나눈다. 이후 온라인 소프트맥스(FlashAttention 방식)로 가중 평균을 누적한다(`claims.md:319-332`).
  - 근거: `src/kernels.cu:463`(shared `dot_products[2]`), `src/kernels.cu:486-507`(내적·워프 리덕션·두 워프 결합), `src/kernels.cu:510-519`(온라인 softmax), `src/kernels.cu:522`(출력 기록).
  - 한계: `dot_products[2]` 와 스레드 0·32 협력은 `HEAD_DIM=64` 전제이며 그 외 크기에서는 성립하지 않는다(`trace.md:225-227`). 실행 검증은 없다.
- **Claim 18 (pagedattention-inplace, High):** `pagedAttentionKernel` 의 출력은 입력 q 프로젝션과 같은 버퍼(`buf_2048_1`)에 쓰인다. 읽기 인덱스(`src/kernels.cu:469`)와 쓰기 인덱스(`src/kernels.cu:522`)가 같아 in-place 가 안전하며, 이것이 `pagedAttention` 호출 결과가 곧바로 O 투영 입력으로 쓰이는 이유다(`claims.md:340-350`).
  - 근거: `src/kernels.cu:469`(입력 q 읽기) 대 `src/kernels.cu:522`(출력 기록), 호출 `src/main.cpp:878`, O 투영 입력 `src/main.cpp:892`.
  - 한계: in-place 안전성은 인덱스 동일성이라는 정적 주장이며 동시성·경합의 실행 검증은 없다.

### Exclusions
- 슬롯 종료 시 블록 반납(`free_blocks` 로의 `push_back`)은 Claim 14 가 인용하지만, 슬롯 해제 동작 자체는 Claim 13 이 7장에서 담당한다. 이 장은 block table 읽기·블록 레이아웃에 집중한다.
- `tests` 가 0 이므로 이 장은 실행 테스트 없이 정적 인용으로만 구성한다.

### Visuals
- **block-table-indexing (block-diagram):**
  - "`pagedAttentionKernel` 은 block_table 을 통해 비연속 KV 블록을 읽는다" → Claim 14 가 직접 뒷받침.
  - "`free_blocks` 목록이 블록 할당과 반납을 관리한다" → Claim 14 가 뒷받침(할당 `src/main.cpp:264-277`·`859-865`, 반납 `src/main.cpp:1015-1031`). 슬롯 종료 연쇄의 반납 절차는 7장의 Claim 13 을 링크한다.
- **paged-attention-inputs (table):**
  - "`pagedAttentionKernel` 은 `kv_cache`, `block_table`, `seq_lens`, `active_slots` 를 함께 받는다" → Claim 16 이 직접 뒷받침.

---

## Chapter 7

### Scope
- **핵심 질문:** 여러 프롬프트를 동시에 처리하기 위해 슬롯과 큐를 어떻게 쓰는가?(`series.yaml:144`)
- **code_scopes:** `src/main.cpp`, `python/batching_test_tokens.py`(`series.yaml:145-146`)
- **required_evidence:** code 4 / tests 0 / runtime 0(`series.yaml:149-151`)
- continuous batching 은 개념이 전제이며 이 장은 슬롯 구현만 다룬다(`guide.md:200`). 이 장은 6편에 의존한다(`series.yaml:150`).

### Claims
- **Claim 13 (decode-termination, High):** decode 루프는 생성 토큰이 `<|end_of_text|>`(128001) 또는 `<|eot_id|>`(128009) 이거나 `current_prompt_len == MAX_SEQ_LEN-1` 이면 슬롯을 해제하고, 해당 슬롯이 소유한 모든 블록을 `free_blocks` 로 반납하며 `block_table` 을 `-1` 로 재초기화한다(`claims.md:252-261`).
  - 근거: `src/main.cpp:1015-1031`(종료 분기), 반납 루프 `src/main.cpp:1018-1029`, 동기화 `src/main.cpp:1030`. 상수는 `src/main.cpp:26-28`.
  - 한계: `MAX_NEW_TOKENS_GENERATED = 20`(`src/main.cpp:12`)은 선언만 있고 참조되지 않아 종료 조건으로 동작하지 않는다(`trace.md:222-224`).
- **Claim 19 (slot-lifecycle, High):** `is_slot_free` 벡터가 슬롯의 점유 상태를 관리하고, decode 루프의 매 반복에서 빈 슬롯은 큐의 다음 프롬프트를 `prefill` 로 채운다. 점유 슬롯만 `active_slots`·`active_tokens` 에 모여 `gpu_active_slots` 로 커널에 전달된다(`claims.md:358-367`).
  - 근거: `src/main.cpp:597`(초기화), `src/main.cpp:722-737`(슬롯 재구성·prefill 채움), `src/main.cpp:749-750`(H2D), `src/main.cpp:878`(커널 전달). 초기 채움은 `src/main.cpp:695-708`.
  - 한계: 슬롯 종료는 EOS/EOT 토큰 또는 `MAX_SEQ_LEN-1` 도달뿐이다(`trace.md:209-210`).
- **Claim 20 (batching-queue, High):** 큐에는 채팅 템플릿 토큰 ID 로 된 프롬프트 4개가 들어 있고, `BATCH_SIZE=2` 이므로 초기 prefill 루프가 슬롯 0·1 을 채우고 프롬프트 2·3 은 큐에 남는다. decode 루프가 슬롯 해제 시 큐의 다음 프롬프트로 채우는 흐름이 continuous batching 의 구현이다(`claims.md:375-385`).
  - 근거: `src/main.cpp:583-594`(큐 4개), `src/main.cpp:29`(`BATCH_SIZE=2`), `src/main.cpp:695-708`(초기 prefill), `src/main.cpp:724-737`(재채움), `src/main.cpp:738-746`(빈 큐·비활성 시 break).
  - 한계: `active_slots` 가 0 이고 큐가 비면 break 하므로, 상시 운영 서버를 의도한 주석(`src/main.cpp:720`)과 달리 실제 실행은 유한하다(`trace.md:162-164`).

### Exclusions
- `python/batching_test_tokens.py` 는 `series.yaml` 의 code_scope 이지만(`series.yaml:146`), `claims.md` 에 이 테스트를 직접 다루는 claim 은 없다. 검증은 정적 인용으로만 한다.
- continuous batching 의 개념·동기 설명은 전제로 두고 슬롯·큐 구현에만 분량을 쓴다(`guide.md:200`).
- paged attention 커널의 읽기 동작은 6편이 소유하며, 이 장은 슬롯·큐만 다룬다.

### Visuals
- **slot-lifecycle (mermaid):**
  - "`is_slot_free` 가 슬롯의 점유 상태를 관리하고 큐에서 다음 프롬프트를 채운다" → Claim 19 가 직접 뒷받침.
  - "활성 슬롯 목록이 decode 스텝마다 커널에 전달된다" → Claim 19 가 직접 뒷받침. 슬롯 해제·블록 반납으로 이어지는 종료 절차는 Claim 13 을 함께 사용한다.