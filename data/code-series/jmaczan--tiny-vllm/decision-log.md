# tiny-vLLM guide decision log

## 001 - 시리즈가 다룰 제품 루트

- Observation: `compatibility-report.md` 의 `monorepo-roots` 가 최상위 소스 루트 4개(`include`, `python`, `src`, `tests`)를 임계값 2 초과로 보고했다. inventory 기준 `src/main.cpp`(53,377B)가 유일한 `entrypoint` 이고 `src/kernels.cu`(19,396B)가 CUDA 커널을 담는다. `include/json.hpp` 는 967,314B 로 분석 대상 소스 바이트의 92% 를 차지한다. `python/` 의 모든 파일은 스스로를 "Reference outputs for verifying tiny-vllm kernel correctness" 로 선언하며, `src/main.cpp` 에는 `system(`/`popen`/`exec`/`python` 호출이 없다.
- Question: 4개 최상위 소스 루트 중 이 시리즈가 다룰 제품은 무엇인가?
- Recommendation: `src/` 를 유일한 제품으로 두고 `python/` 은 커널 정확성 검증 근거, `tests/` 는 테스트 근거, `include/` 는 vendored 제외로 분류한다. 근거는 유일한 entrypoint 가 `src/main.cpp` 이고 C++ 런타임이 Python 을 호출하지 않는다는 점이다.
- User decision: `src/` 만 제품으로 한다. `python/` 은 검증 근거, `tests/` 는 테스트 근거, `include/json.hpp` 는 vendored 제외.
- Guide impact: 모든 편의 `code_scopes` 는 `src/` 아래 경로로 한정한다. `python/` 경로는 수치 검증 근거로만 인용하고 편의 주제로 삼지 않는다. 포함/제외 범위 절에 `include/` 제외 사유를 기록한다.
- Compatibility finding: monorepo-roots

## 002 - 분석 범위를 지배하는 단일 파일

- Observation: `dominant-source-file` 이 `include/json.hpp` 를 분석 대상 소스 바이트의 92% 로 보고했다. 파일 헤더가 `JSON for Modern C++ version 3.12.0`, `SPDX-FileCopyrightText: 2013-2026 Niels Lohmann`, `SPDX-License-Identifier: MIT` 로 제3자 단일 헤더 라이브러리임을 명시한다. `_VENDOR_PARTS` 가 `include/` 를 vendor 경로로 보지 않아 자동 제외되지 않았다.
- Question: 이 파일은 제외할 vendored 의존인가, 아니면 실제 범위인가?
- Recommendation: vendored 의존으로 제외한다. SPDX 헤더가 출처와 라이선스를 명시하고, 제품 코드가 아니라 설정 파싱용 유틸리티다.
- User decision: vendored 로 제외한다(001 의 제품 범위 결정과 동일한 답변에서 함께 확정됨).
- Guide impact: 분석 대상 바이트에서 967,314B 를 제외하므로 실제 제품 규모는 약 78KB(`main.cpp` + `kernels.cu` + 헤더)다. 편 수와 분량 산정을 이 규모로 한다. 어떤 편도 `include/` 를 `code_scopes` 에 넣지 않는다.
- Compatibility finding: dominant-source-file

## 003 - CUDA 경로의 실행 검증 방법

- Observation: `execution-environment` 가 "CUDA core source files" 에 대해 "a documented CPU-only validation path" 부재를 보고했다. 대표 실행 경로는 `full_test.sh` 가 하드코딩된 Llama-3 채팅 템플릿 토큰 ID 를 `test.sh`(=`build.sh` + `run.sh`)에 파이프하는 것이다. `src/main.cpp:87` 은 작업 디렉터리의 `model.safetensors` 를 직접 열며(경로 인자화는 코드 내 TODO), 이는 gated 모델 `meta-llama/Llama-3.2-1B-Instruct` 의 가중치다. 이 환경에는 `nvidia-smi` 와 `nvcc` 가 모두 없다. 반면 `reference.txt`(13,573B)와 `python/rms_norm_crosscheck.txt`(3,770B)는 저장소에 커밋된 참조 출력이다.
- Question: GPU 가 없는 환경에서 CUDA 경로를 어떤 근거로 다루는가?
- Recommendation: 정적 근거 전용. 커밋된 참조 출력을 수치 교차검증 근거로 인용하고 모든 편의 `runtime` 근거를 0 으로 두며, 커널 실행 시간과 처리량을 측정하지 않음을 검증 조건에 명시한다. 지금 확보 가능한 근거만 쓰는 가장 좁은 선택이다.
- User decision: 정적 근거 전용으로 한다.
- Guide impact: 모든 편 `required_evidence.runtime: 0`. 모든 기술 주장은 고정 SHA permalink 인용으로 뒷받침한다. 성능 수치는 측정값이 아니라 upstream 인용으로만 쓰고, Plan 02b 의 측정/upstream 구분을 따른다. 검증 조건 절에 "커널 실행 시간·처리량·메모리 사용량은 이 시리즈에서 측정하지 않는다" 를 명시한다. GPU 실행과 gated 가중치 확보는 후속 근거 과제로만 기록한다.
- Compatibility finding: execution-environment

## 004 - 빌드 정의의 범위 포함

- Observation: `unsupported-build-system` 이 `CMakeLists.txt` 를 결정적 의존성 추출기가 없는 빌드 파일로 보고했다. 실제 내용은 `add_executable(tiny-vllm src/main.cpp src/kernels.cu)` 로 제품 번역 단위를 정의하고, `option(USE_HIP ...)` 로 CUDA/HIP 이중 백엔드를 분기하며(`src/cuda_to_hip.h` 가 그 shim), `CMAKE_CUDA_ARCHITECTURES 120` 과 `CUDA::cublas`/`CUDA::cudart` 링크를 지정한다. `target_include_directories(... include)` 가 `json.hpp` 를 의존으로 끌어온다.
- Question: 빌드·실행 도구 계층을 설명 대상 경계에 포함하는가?
- Recommendation: `CMakeLists.txt`, `full_test.sh`, `build.sh`, `run.sh`, `test.sh` 를 포함하고 `check.sh`, `ncu.sh`, `nsys.sh` 는 제외한다. 앞의 것들은 제품 정의와 대표 실행 경로를 입증하고, 뒤의 것들은 GPU·sudo 를 요구해 결정 003 에서 이미 측정 대상에서 빠졌다.
- User decision: 빌드와 실행만 포함한다(`CMakeLists.txt`, `full_test.sh`, `build.sh`, `run.sh`, `test.sh`). 프로파일링·사니타이저 스크립트는 범위 밖.
- Guide impact: 아키텍처 지도 편이 `CMakeLists.txt` 를 근거로 제품 번역 단위와 CUDA/HIP 이중 백엔드를 설명한다. `include/` 가 제품이 아니라 include 의존이라는 결정 002 의 근거로도 이 파일을 인용한다.
- Compatibility finding: unsupported-build-system

## 005 - 미분류 core 파일의 처리

- Observation: `unclassified-core-file` 이 언어 없는 core 파일 1개로 `full_test.sh` 를 보고했다. 내용은 하드코딩된 Llama-3 채팅 템플릿 토큰 ID 열을 `./test.sh` 에 파이프하는 한 줄이며, 결정 003 이 이를 대표 실행 경로로 지목했다.
- Question: 이 미분류 파일이 프로젝트 설계에 필수인가?
- Recommendation: 필수로 보고 범위에 포함한다. 진입점부터 결과까지의 대표 실행 경로를 저장소가 스스로 규정한 유일한 지점이다.
- User decision: 포함한다(004 의 도구 계층 결정과 동일한 답변에서 함께 확정됨).
- Guide impact: 대표 실행 경로 절이 `full_test.sh` 의 토큰 입력에서 시작해 `main.cpp` 의 `prefill` 과 커널을 거쳐 생성 토큰까지를 추적한다. 언어 추출기가 없으므로 이 파일의 주장은 전문 인용으로 뒷받침한다.
- Compatibility finding: unclassified-core-file

## 006 - 미분류 비분석 파일의 처리

- Observation: `unclassified-unanalyzable-file` 이 6개를 보고했다: `build.sh`(cmake+ninja), `run.sh`(바이너리 실행), `test.sh`(둘의 조합), `check.sh`(compute-sanitizer), `ncu.sh`(Nsight Compute, `sudo` 와 CUDA 12.8 경로 하드코딩), `nsys.sh`(Nsight Systems 타임라인).
- Question: 이 중 프로젝트를 설명하는 데 필요한 것은 무엇인가?
- Recommendation: `build.sh`, `run.sh`, `test.sh` 만 포함하고 `check.sh`, `ncu.sh`, `nsys.sh` 는 제외한다. 뒤의 셋은 GPU 와 sudo 를 요구해 결정 003 의 정적 근거 전용 방침에서 산출물을 만들 수 없다.
- User decision: `build.sh`, `run.sh`, `test.sh` 만 포함한다. 프로파일링·사니타이저 3개는 범위 밖.
- Guide impact: 포함 범위 절에 세 스크립트를 빌드·실행 경계로 기록하고, 제외한 셋은 사유(GPU·sudo 필요, 측정 미수행)와 함께 제외 범위 절에 적는다.
- Compatibility finding: unclassified-unanalyzable-file

## 007 - 독자와 선수 지식

- Observation: upstream README 는 선수 지식을 하드웨어·툴체인(NVIDIA GPU, CUDA 13.1, C++17, GCC 15.2.1)으로만 적고, 본문은 `Intro: LLM, vLLM, models, inference servers` 와 `How floating-point numbers work and why we use bfloat16` 부터 시작해 LLM 추론 입문자를 상정한다. 반면 이 블로그의 최근 게시글 20편은 speculative decoding, KV cache 압축, prefill chunking, MoE 라우팅, FP4 FlashAttention, 양자화로 전부 추론 최적화 주제다.
- Question: upstream 처럼 입문자를 상정할 것인가, 블로그 기존 독자처럼 추론 개념을 아는 독자를 상정할 것인가?
- Recommendation: 추론 개념을 아는 독자를 상정하고 CUDA·시스템 구현에 집중한다. 기존 독자에게 새로운 가치는 개념 설명이 아니라 커널과 메모리에서의 실제 구현이다.
- User decision: 추론 개념을 아는 독자를 상정한다. 선수 지식은 C++17, CUDA 기초 문법, 트랜스포머 추론 개념.
- Guide impact: prefill/decode, KV cache, attention 의 *개념* 설명은 각 편에서 한두 문장으로 요약하고 분량을 쓰지 않는다. 분량은 CUDA 커널 구현, cuBLAS 연동, paged KV cache 인덱싱, 버퍼 재사용에 배분한다. 독자·목표 절에 선수 지식 3개를 명시하고, upstream README 의 입문 절들은 링크로 넘긴다.
- Compatibility finding: 해당 없음

## 008 - 편 구성과 깊이

- Observation: `src/main.cpp` 는 최상위 함수가 4개(`checkGPUStatus:40`, `loadWeights:79`, `prefill:150`, `main:555`)뿐이어서 디렉터리 구조로는 분할 축이 없다. `src/kernels.cu` 의 `__global__` 커널 11개는 prefill 계열 7개(`embeddingGatherKernel:32`, `rmsNormKernel:55`, `ropeKernel_llama3:173`, `causalMaskKernel:224`, `softmaxKernel:257`, `residualKernel:311`, `siluKernel:331`)와 decode 계열 4개(`embeddingGatherKernelDecode:347`, `ropeKernelDecode:371`, `softmaxKernelDecode:408`, `pagedAttentionKernel:461`)로 갈린다. `pagedAttentionKernel` 만 `kv_cache`, `block_table_gpu`, `gpu_seq_lens`, `gpu_active_slots` 를 받는다.
- Question: 런타임 경로를 몇 편으로 나누고 prefill 과 decode 를 분리하는가?
- Recommendation: 7편으로 나누고 prefill 과 decode 를 분리한다. 코드가 이미 decode 전용 변형 커널을 따로 두고 decode 만 paged attention 과 K/V 임시 버퍼를 쓴다는 점이 근거다.
- User decision: 7편, prefill 과 decode 분리.
- Guide impact: 편 id 와 순서를 `build-and-entry`, `weights-and-memory`, `prefill-path`, `cublas-and-layout`, `decode-path`, `paged-kv-cache`, `batching-and-slots` 로 고정한다. Plan 04 Task 4 스케치의 단일 `prefill-decode` id 대신 코드 증거를 따르며, 골든 테스트는 이 구성에 맞춰 작성한다.
- Compatibility finding: 해당 없음
