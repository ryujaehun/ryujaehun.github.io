# tiny-vLLM 코드 읽기 시리즈 Guide

## 분석 대상과 고정 revision

- Repository: https://github.com/jmaczan/tiny-vllm
- Commit: `e25bf1994efa90bc98b721ba7c527402f86fbeaf`
- Project slug: `jmaczan--tiny-vllm`

이 시리즈의 모든 주장은 위 커밋의 트리만을 근거로 한다. 기본 브랜치를 다시
해석하거나 이후 revision 으로 대체하지 않는다.

## 독자, 목표와 선수 지식

독자는 LLM 추론 최적화를 이미 개념으로 아는 사람이다(결정 007). prefill 과
decode 의 구분, KV cache 가 존재하는 이유, attention 의 수식, continuous
batching 의 목적은 설명 대상이 아니라 전제다.

목표는 그 개념들이 **실제 CUDA 코드와 GPU 메모리에서 어떤 모양으로 구현되는가**
를 보이는 것이다. 논문 리뷰로는 볼 수 없는 층 — 블록 내 병렬 리덕션, cuBLAS 의
레이아웃 가정, block table 인덱싱, 버퍼 재사용 — 에 분량을 쓴다.

선수 지식은 세 가지다.

- C++17 문법과 표준 라이브러리
- CUDA 기초 문법(`__global__`, 블록/스레드 인덱싱, `cudaMalloc`)
- 트랜스포머 추론의 개념적 흐름

upstream README 의 입문 절(`Intro: LLM, vLLM, models, inference servers`,
`How floating-point numbers work and why we use bfloat16`, `GPU and CPU memory`)
은 이 전제를 갖추지 못한 독자에게 링크로 넘긴다.

## 포함 범위와 제외 범위

포함 범위는 제품과 그 빌드·실행 경계다(결정 001, 004, 005, 006).

| 경로 | 역할 | 포함 사유 |
| --- | --- | --- |
| `src/main.cpp` | entrypoint | 유일한 진입점이며 가중치 적재·prefill·decode·슬롯 관리를 모두 담는다 |
| `src/kernels.cu`, `src/kernels.cuh` | CUDA 커널 | 11개 `__global__` 커널이 연산의 실체다 |
| `src/cuda_to_hip.h` | 이식 shim | CUDA/HIP 이중 백엔드를 성립시킨다 |
| `CMakeLists.txt` | 빌드 정의 | 제품 번역 단위와 이중 백엔드 분기를 규정한다 |
| `full_test.sh`, `build.sh`, `run.sh`, `test.sh` | 실행 경계 | 대표 실행 경로를 저장소가 스스로 규정한 지점이다 |

제외 범위와 사유는 다음과 같다.

| 경로 | 제외 사유 |
| --- | --- |
| `include/json.hpp` | 제3자 단일 헤더(`JSON for Modern C++ 3.12.0`, MIT, Niels Lohmann). 분석 대상 소스 바이트의 92% 를 차지해 규모 산정을 왜곡한다(결정 002) |
| `check.sh`, `ncu.sh`, `nsys.sh` | compute-sanitizer 와 Nsight 프로파일링. GPU 와 `sudo` 를 요구하며 이 시리즈는 측정을 수행하지 않는다(결정 003, 006) |
| `.vscode/`, `assets/column-row-major.png` | 편집기 설정과 그림 자원. 구현 근거가 아니다 |

`python/` 은 편의 주제가 아니라 **근거**로만 쓴다(결정 001). `reference.py`,
`rms_norm.py` 는 스스로를 "Reference outputs for verifying tiny-vllm kernel
correctness" 로 선언하며, `main.cpp` 는 Python 을 호출하지 않는다.
`reference.txt` 와 `python/rms_norm_crosscheck.txt` 는 저장소에 커밋된 참조
출력이므로 실행 없이 인용할 수 있다.

실제 제품 규모는 `include/json.hpp` 를 제외하면 약 78KB(`main.cpp` 53,377B +
`kernels.cu` 19,396B + 헤더)다. 편 수와 분량은 이 규모에 맞춘다.

## 템플릿 적합성 및 실행 제약

compatibility status 는 `needs-review` 이며 finding 6개 전부에 명시적 결정이
있다. 상세 근거와 사용자 결정은 `decision-log.md` 에 기록했다.

| finding | 결정 | 기록 |
| --- | --- | --- |
| `monorepo-roots` | `src/` 만 제품 | 001 |
| `dominant-source-file` | `include/json.hpp` vendored 제외 | 002 |
| `execution-environment` | 정적 근거 전용 | 003 |
| `unsupported-build-system` | `CMakeLists.txt` 포함 | 004 |
| `unclassified-core-file` | `full_test.sh` 포함 | 005 |
| `unclassified-unanalyzable-file` | `build.sh`·`run.sh`·`test.sh` 포함, 나머지 제외 | 006 |

실행 제약은 다음과 같다. upstream README 가 밝힌 개발·검증 환경은 NVIDIA
RTX 5090, CUDA Toolkit 13.1, GCC 15.2.1 이고 `CMakeLists.txt` 는
`CMAKE_CUDA_ARCHITECTURES 120` 을 지정한다. `src/main.cpp` 는 작업 디렉터리의
`model.safetensors` 를 직접 열며, 이는 gated 모델 `meta-llama/Llama-3.2-1B-Instruct`
의 가중치다. 이 시리즈를 쓰는 환경에는 NVIDIA GPU 도 CUDA 툴체인도 없다.

따라서 **정적 대체 검증**을 택한다(결정 003).

- 모든 기술 주장은 고정 revision 의 소스 인용으로 뒷받침한다.
- 수치 교차검증은 커밋된 `reference.txt` 와 `python/rms_norm_crosscheck.txt` 를 인용한다.
- 모든 편의 `required_evidence.runtime` 은 0 이다.
- 커널 실행 시간·처리량·메모리 사용량은 **측정하지 않는다**. 성능 관련 수치는 측정값이 아니라 upstream 인용으로만 쓰고, 그 출처를 문장에서 밝힌다.
- GPU 실행과 gated 가중치 확보는 후속 근거 과제로만 남긴다.

라이선스: 대상 저장소의 `LICENSE` 와 `include/json.hpp` 의 MIT SPDX 헤더를
확인했다. 코드 인용은 출처 permalink 와 함께 제시한다.

## 전체 아키텍처 지도

제품은 단일 실행 파일이며 경계는 넷이다.

| 컴포넌트 | 책임 | 근거 경로 |
| --- | --- | --- |
| 빌드 정의 | 번역 단위 선택과 CUDA/HIP 백엔드 분기 | `CMakeLists.txt` |
| 호스트 오케스트레이션 | 가중치 적재, 버퍼 할당, 커널 호출 순서, 슬롯·큐 관리 | `src/main.cpp` |
| 디바이스 커널 | 임베딩·정규화·회전·마스크·softmax·잔차·활성화·paged attention | `src/kernels.cu`, `src/kernels.cuh` |
| 이식 계층 | bfloat16 타입과 BLAS 호출의 CUDA/HIP 차이 흡수 | `src/cuda_to_hip.h` |

호스트 함수는 네 개뿐이다: `checkGPUStatus`, `loadWeights`, `prefill`, `main`.
모델 forward 전체와 paged KV cache 관리가 `prefill` 과 `main` 안에 들어 있으므로,
편 분할은 디렉터리가 아니라 **런타임 경로**를 따른다(결정 008).

디바이스 커널은 prefill 계열 7개와 decode 계열 4개로 갈린다. decode 계열만
`pagedAttentionKernel` 을 포함하며 이 커널만 `kv_cache`, `block_table_gpu`,
`gpu_seq_lens`, `gpu_active_slots` 를 함께 받는다. 이 비대칭이 3편과 5편을
나누는 근거다.

## 대표 실행 경로

저장소가 스스로 규정한 단 하나의 경로는 `full_test.sh` 다(결정 005). 이 경로를
진입점부터 결과까지 추적한다.

1. `full_test.sh` 가 하드코딩된 Llama-3 채팅 템플릿 토큰 ID 열을 표준 입력으로 `./test.sh` 에 넘긴다.
2. `test.sh` 가 `build.sh`(cmake + ninja)로 `build/tiny-vllm` 을 만들고 `run.sh` 로 즉시 실행한다.
3. `main` 이 cuBLAS 핸들을 만들고 `loadWeights` 로 `model.safetensors` 를 읽는다.
4. prefill 버퍼와 decode 전용 버퍼를 할당하고, 빈 슬롯을 큐의 프롬프트로 채운다.
5. `prefill` 이 프롬프트 전체를 임베딩부터 다음 토큰 선택까지 한 번에 흘린다.
6. decode 스텝이 슬롯별로 토큰을 하나씩 이어 붙이며 paged KV cache 를 읽고 쓴다.
7. 생성된 토큰 ID 가 표준 출력으로 나온다. 사람이 읽을 텍스트로 만드는 것은 `python/tokenizer.py` 의 역방향이며 런타임 밖의 일이다.

이 경로의 각 단계가 어느 편에 속하는지는 다음 절의 의존성으로 대응한다.

## 시리즈 전체 서사와 편별 의존성

서사는 "무엇이 빌드되는가 → 무엇이 메모리에 올라가는가 → 프롬프트가 어떻게
흐르는가 → 왜 그런 행렬곱을 쓰는가 → 토큰 하나는 어떻게 다른가 → KV 를 어떻게
쪼개 담는가 → 여러 요청을 어떻게 겹치는가" 순서다. 학습 흐름과 코드 의존성이
같은 방향이다.

```text
build-and-entry
  └── weights-and-memory
        └── prefill-path
              ├── cublas-and-layout
              └── decode-path
                    └── paged-kv-cache
                          └── batching-and-slots
```

`cublas-and-layout` 은 `prefill-path` 에서 처음 등장한 cuBLAS 호출을 되짚는
편이므로 3편에 의존한다. `decode-path` 도 prefill 과의 대비로만 성립하므로
3편에 의존한다. 두 편은 서로 독립이다.

## 편별 핵심 질문, 코드 범위와 필수 근거

`series.yaml` 과 동일한 내용이다. 근거 수는 최소값이며 `runtime` 은 결정 003 에
따라 전부 0 이다.

| # | id | 핵심 질문 | code_scopes | code / tests / runtime |
| --- | --- | --- | --- | --- |
| 1 | `build-and-entry` | 제품으로 빌드되는 것은 정확히 무엇이고 실행은 어디서 시작하는가 | `CMakeLists.txt`, `build.sh`, `run.sh`, `test.sh`, `full_test.sh`, `src/cuda_to_hip.h` | 4 / 0 / 0 |
| 2 | `weights-and-memory` | safetensors 를 어떻게 읽고 어떤 버퍼를 미리 잡는가 | `src/main.cpp` | 3 / 0 / 0 |
| 3 | `prefill-path` | 프롬프트가 어떤 커널 순서로 흐르는가 | `src/main.cpp`, `src/kernels.cu`, `src/kernels.cuh`, `tests/test_softmax.cu` | 6 / 1 / 0 |
| 4 | `cublas-and-layout` | 왜 전치된 형태로 cuBLAS 를 부르는가 | `src/main.cpp` | 3 / 0 / 0 |
| 5 | `decode-path` | decode 는 왜 다른 커널 변형을 쓰는가 | `src/kernels.cu`, `src/main.cpp`, `python/decode_test.py` | 4 / 1 / 0 |
| 6 | `paged-kv-cache` | block table 로 참조하면 어텐션 커널은 무엇을 읽는가 | `src/kernels.cu`, `src/main.cpp` | 5 / 0 / 0 |
| 7 | `batching-and-slots` | 슬롯과 큐로 여러 요청을 어떻게 겹치는가 | `src/main.cpp`, `python/batching_test_tokens.py` | 4 / 0 / 0 |

코드 인용은 고정 revision 을 가리키는 GitHub permalink 형식
(`/blob/<고정 revision 절의 커밋>/<경로>#L<시작>-L<끝>`)으로 계획한다.

## 필요한 표, 다이어그램과 코드 예제

각 시각 자료는 답할 질문과 근거 경로를 가진다. 장식용 자료는 두지 않는다.

| 편 | id | 형식 | 답할 질문 | 근거 |
| --- | --- | --- | --- | --- |
| 1 | `product-translation-units` | 표 | 무엇이 제품이고 무엇이 의존인가 | `CMakeLists.txt` 의 `add_executable`, `target_include_directories` |
| 1 | `dual-backend-branch` | 블록 다이어그램 | 같은 소스가 어떻게 두 백엔드로 가는가 | `CMakeLists.txt` 의 `USE_HIP` 분기, `src/cuda_to_hip.h` |
| 2 | `buffer-allocation-table` | 표 | 어떤 버퍼가 얼마나 잡히는가 | `src/main.cpp` 의 `cudaMalloc` 구간 |
| 3 | `prefill-kernel-sequence` | Mermaid | 커널 호출 순서는 무엇인가 | `src/main.cpp` 의 `prefill`, `src/kernels.cu` |
| 3 | `rmsnorm-reduction` | 블록 다이어그램 | 블록 내 리덕션이 어떻게 제곱 평균을 구하는가 | `src/kernels.cu` 의 `rmsNormKernel` |
| 4 | `column-row-major-trick` | 블록 다이어그램 | 레이아웃 불일치를 어떻게 넘기는가 | `src/main.cpp` 의 cuBLAS 호출부 |
| 5 | `prefill-decode-contrast` | 표 | 두 계열 커널이 무엇이 다른가 | `src/kernels.cu` 의 `...Decode` 커널 |
| 6 | `block-table-indexing` | 블록 다이어그램 | 비연속 블록을 어떻게 읽는가 | `src/kernels.cu` 의 `pagedAttentionKernel`, `src/main.cpp` 의 `block_table` |
| 6 | `paged-attention-inputs` | 표 | 커널이 받는 상태는 무엇인가 | `pagedAttentionKernel` 시그니처 |
| 7 | `slot-lifecycle` | Mermaid | 슬롯이 어떻게 채워지고 비는가 | `src/main.cpp` 의 `is_slot_free`, 큐 루프 |

코드 예제는 구문 자체가 설명에 필요한 경우에만 넣는다. 구체적으로는 커널
런치 구성(`<<<...>>>`), cuBLAS 호출의 전치 플래그, block table 인덱싱 산술
세 곳이다.

## 공통 용어집과 중복 방지 규칙

용어는 소유 편을 하나만 두고 다른 편은 그 편을 링크한다.

| 용어 | 소유 편 | 비고 |
| --- | --- | --- |
| 번역 단위, 이중 백엔드 | 1 `build-and-entry` | HIP 경로 언급은 모두 여기로 링크 |
| bfloat16 버퍼, 슬롯 | 2 `weights-and-memory` | 크기 산정의 근거를 여기 한 번만 적는다 |
| prefill | 3 `prefill-path` | 개념 설명은 한두 문장, 구현에 분량 |
| 병렬 리덕션 | 3 `prefill-path` | softmax 편에서 재설명하지 않고 링크 |
| 열 우선 레이아웃 | 4 `cublas-and-layout` | 다른 편의 행렬곱 언급은 여기로 링크 |
| decode 스텝 | 5 `decode-path` | prefill 과의 대비만 다루고 KV 구조는 6편으로 |
| block table, paged attention | 6 `paged-kv-cache` | KV 저장 구조의 유일한 소유 편 |
| continuous batching | 7 `batching-and-slots` | 개념은 전제, 슬롯 구현만 다룬다 |

`src/main.cpp` 와 `src/kernels.cu` 는 여러 편이 공유한다. 같은 함수를 두 편이
다룰 때는 **줄 범위를 나누고** 서로를 링크한다. `prefill` 함수는 3편이 커널
순서를, 4편이 cuBLAS 호출부를 담당한다.

## 검증 및 게시 조건

편별 조건:

- 모든 기술 주장이 고정 revision permalink 인용을 가진다.
- `required_evidence` 의 code·tests 최소값을 충족한다. `runtime` 은 0 이며, 측정값을 제시하지 않는다.
- 각 시각 자료가 `series.yaml` 의 `required_claims` 를 실제로 답한다.
- 용어가 소유 편에서만 정의되고 나머지는 링크한다.

시리즈 조건:

- 7편 전부 한국어·영어 쌍이 존재하고 구조가 대응한다.
- 전체 시리즈 검토가 PASS 여야 게시를 시작한다.
- 게시는 `Asia/Seoul` 기준 하루 한 편, 오래된 due 편부터 진행한다.
- 한국어·영어 두 URL 이 `jaehun.me` 에서 HTTP 200 과 기대 제목을 반환한 뒤에만 알림을 보낸다.

성능 주장에 대한 명시적 제약: 이 시리즈는 커널 실행 시간, 처리량, 메모리
사용량을 측정하지 않는다. upstream 이 밝힌 수치를 인용할 때는 출처와 환경을
문장에서 밝히고, 측정값으로 읽히지 않게 쓴다.
