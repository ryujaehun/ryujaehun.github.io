# tiny-vLLM Recon Memo

이 메모의 모든 사실은 `guide.md` 와 `inventory/files.jsonl` 두 첨부만을 근거로
한다. 저장소는 실행하지 않았고, 런타임 산출물도 만들지 않았다. 고정 revision 은
`e25bf1994efa90bc98b721ba7c527402f86fbeaf` 이며 `guide.md` 가 이를 명시한다.

## Observations

- `inventory/files.jsonl` 은 총 31개 경로를 나열한다. 그중 제품 소스로 분류된
  것은 `src/main.cpp`, `src/kernels.cu`, `src/kernels.cuh`, `src/cuda_to_hip.h`,
  `include/json.hpp`, `python/reference.py`, `python/rms_norm.py`,
  `python/tokenizer.py`, `python/batching_test_tokens.py` 다.
- `src/main.cpp` 는 53377B / 1044줄, role `entrypoint` 다. `guide.md` 는 이
  파일이 가중치 적재·prefill·decode·슬롯 관리를 모두 담는 유일한 진입점이라고
  적는다.
- `src/kernels.cu` 는 19396B / 528줄, `src/kernels.cuh` 는 1422B / 28줄,
  `src/cuda_to_hip.h` 는 2008B / 61줄이다. `guide.md` 는 `src/kernels.cu` 에
  11개 `__global__` 커널이 있다고 적는다.
- `guide.md` 는 커널을 prefill 계열 7개와 decode 계열 4개로 나누고,
  `pagedAttentionKernel` 만이 decode 계열에 속한다고 적는다.
- `guide.md` 는 호스트 함수를 `checkGPUStatus`, `loadWeights`, `prefill`,
  `main` 네 개로 한정한다. `src/main.cpp` 에 모델 forward 와 paged KV cache
  관리가 들어 있다는 뜻이다.
- `include/json.hpp` 는 967314B / 25830줄, role `source` 다. `guide.md` 는 이
  파일을 제3자 단일 헤더(`JSON for Modern C++ 3.12.0`, MIT)로 규정하고 분석
  대상 소스 바이트의 92% 를 차지해 규모 산정을 왜곡한다며 제외한다.
- `guide.md` 는 `include/json.hpp` 를 뺀 실제 제품 규모를 약 78KB 로 적는다.
- 빌드·실행 경계는 `CMakeLists.txt`(1940B / 75줄, role `build`),
  `build.sh`(67B), `run.sh`(17B), `test.sh`(19B), `full_test.sh`(228B) 다.
  `inventory/files.jsonl` 은 `build.sh`, `run.sh`, `test.sh` 를 role `other`
  로 두지만 `guide.md` 는 결정 006 으로 이들을 포함 범위에 넣는다.
- `CMakeLists.txt` 는 `guide.md` 에 따르면 `USE_HIP` 분기와
  `CMAKE_CUDA_ARCHITECTURES 120` 을 담고, 제품 번역 단위와 이중 백엔드 분기를
  규정한다. `src/cuda_to_hip.h` 는 bfloat16 타입과 BLAS 호출의 CUDA/HIP 차이를
  흡수하는 이식 계층이다.
- 검증 자산으로 `tests/test_softmax.cu`(4571B / 123줄, role `test`),
  `python/decode_test.py`(434B / 12줄, role `test`),
  `python/reference.py`(4821B / 122줄), `python/rms_norm.py`(2143B / 59줄)가
  있다. `guide.md` 는 `python/reference.py` 와 `python/rms_norm.py` 가 스스로를
  "Reference outputs for verifying tiny-vllm kernel correctness" 로 선언한다고
  적는다.
- 커밋된 참조 출력으로 `reference.txt`(13573B / 613줄)와
  `python/rms_norm_crosscheck.txt`(3770B / 162줄)가 있다. `guide.md` 는 실행
  없이 인용 가능한 근거로 이 둘을 든다.
- 제외 대상은 `check.sh`(35B), `ncu.sh`(74B), `nsys.sh`(142B),
  `.vscode/c_cpp_properties.json`, `.vscode/launch.json`,
  `.vscode/settings.json`, `.vscode/tasks.json`,
  `assets/column-row-major.png`(34867B)다. `guide.md` 는 앞 셋을 GPU·`sudo`
  요구 때문에, 뒤 둘을 편집기 설정·그림 자원이라 제외한다.
- `python/tokenizer.py`(1712B / 50줄)와
  `python/__pycache__/tokenize.cpython-312.pyc`(2696B)가 인벤토리에 있다.
  `guide.md` 는 생성 토큰 ID 를 사람이 읽을 텍스트로 만드는 일이
  `python/tokenizer.py` 의 역방향이며 런타임 밖이라고 적는다.
- `README.md`(94274B / 1304줄, role `documentation`)와 `LICENSE`(11357B /
  201줄)가 있다. `guide.md` 는 `LICENSE` 와 `include/json.hpp` 의 MIT SPDX
  헤더를 확인했다고 적는다.
- `guide.md` 는 실행 환경을 NVIDIA RTX 5090, CUDA Toolkit 13.1, GCC 15.2.1
  로 적고, `src/main.cpp` 가 작업 디렉터리의 `model.safetensors`(gated 모델
  `meta-llama/Llama-3.2-1B-Instruct` 가중치)를 직접 연다고 적는다.

## Execution paths

- 저장소가 스스로 규정한 단일 경로는 `full_test.sh` 다. `guide.md` 는
  `full_test.sh` 가 하드코딩된 Llama-3 채팅 템플릿 토큰 ID 열을 표준 입력으로
  `./test.sh` 에 넘긴다고 적는다.
- `test.sh` 는 `build.sh`(cmake + ninja)로 `build/tiny-vllm` 을 만들고
  `run.sh` 로 즉시 실행한다. 이 두 스크립트 모두 인벤토리에 존재한다.
- `guide.md` 에 따르면 `src/main.cpp` 의 `main` 이 cuBLAS 핸들을 만들고
  `loadWeights` 로 `model.safetensors` 를 읽는다.
- 이어서 `src/main.cpp` 가 prefill 버퍼와 decode 전용 버퍼를 할당하고, 빈
  슬롯을 큐의 프롬프트로 채운다.
- `src/main.cpp` 의 `prefill` 이 프롬프트 전체를 임베딩부터 다음 토큰 선택까지
  한 번에 흘린다. 이때 호출되는 커널들은 `src/kernels.cu` 와
  `src/kernels.cuh` 에 있다.
- decode 스텝은 `src/main.cpp` 에서 슬롯별로 토큰을 하나씩 이어 붙이며
  `src/kernels.cu` 의 `pagedAttentionKernel` 로 paged KV cache 를 읽고 쓴다.
- 생성된 토큰 ID 가 표준 출력으로 나온다. `guide.md` 는 이를 텍스트로 바꾸는
  단계가 `python/tokenizer.py` 의 역방향이며 런타임 밖이라고 적는다.
- `guide.md` 의 편별 의존성 그래프는
  `build-and-entry` → `weights-and-memory` → `prefill-path` →
  (`cublas-and-layout`, `decode-path`) → `paged-kv-cache` →
  `batching-and-slots` 순서다. `cublas-and-layout` 과 `decode-path` 는
  `prefill-path` 에만 의존하고 서로 독립이다.

## Risks and gaps

- 이 시리즈를 쓰는 환경에 NVIDIA GPU 와 CUDA 툴체인이 없다. `guide.md` 는
  결정 003 으로 정적 대체 검증을 택하고 모든 편의 `required_evidence.runtime`
  을 0 으로 둔다. 따라서 `src/kernels.cu` 의 커널과 `src/main.cpp` 의 호출부는
  실행 검증 없이 소스 인용으로만 다룬다.
- gated 가중치 `meta-llama/Llama-3.2-1B-Instruct` 가 필요하다. `guide.md` 는
  GPU 실행과 gated 가중치 확보를 후속 근거 과제로만 남긴다.
- `include/json.hpp` 가 분석 대상 바이트의 92% 를 차지한다. 제외하지 않으면
  `src/main.cpp`·`src/kernels.cu` 중심의 실제 규모가 왜곡된다.
- `tests/test_softmax.cu` 와 `python/decode_test.py` 는 존재하지만 실행할 수
  없다. 편별 `required_evidence` 의 tests 최소값은 인용 근거로만 충족해야 한다.
- `build.sh`, `run.sh`, `test.sh` 는 인벤토리 role 이 `other` 라 포함 여부가
  결정 006 에 의존한다. role 만으로 제품 범위를 판단하면 누락 위험이 있다.
- `check.sh`, `ncu.sh`, `nsys.sh` 는 GPU 와 `sudo` 를 요구한다. 이들을 근거로
  삼으려는 시도는 실행 제약과 충돌한다.
- `guide.md` 는 커널 실행 시간·처리량·메모리 사용량을 측정하지 않는다고
  못박는다. 성능 수치는 upstream 인용으로만 쓰고 출처를 밝혀야 하며, 측정값으로
  읽히면 안 된다.
- `src/main.cpp`(1044줄)와 `src/kernels.cu`(528줄)를 여러 편이 공유한다.
  `guide.md` 는 같은 함수를 두 편이 다룰 때 줄 범위를 나누고 상호 링크하라고
  지시한다. 범위 분할이 없으면 중복 서술 위험이 있다.
- 호환성 상태는 `needs-review` 이고 finding 6개 전부에 결정이 매핑돼 있다.
  결정 기록은 `decision-log.md` 에 있다.
- `python/__pycache__/tokenize.cpython-312.pyc` 는 바이너리이며 근거로 쓰지
  않는다.
- `README.md`(1304줄)는 입문 절을 링크로 넘기는 용도이며 구현 근거가 아니다.
  구현 주장의 출처로 삼으면 안 된다.

## Source paths

제품·빌드·실행 경계:

- `CMakeLists.txt`
- `src/main.cpp`
- `src/kernels.cu`
- `src/kernels.cuh`
- `src/cuda_to_hip.h`
- `build.sh`, `run.sh`, `test.sh`, `full_test.sh`

검증·참조 근거:

- `tests/test_softmax.cu`
- `python/decode_test.py`
- `python/reference.py`, `python/rms_norm.py`
- `python/rms_norm_crosscheck.txt`, `reference.txt`
- `python/batching_test_tokens.py`, `python/tokenizer.py`

제외:

- `include/json.hpp`
- `check.sh`, `ncu.sh`, `nsys.sh`
- `.vscode/c_cpp_properties.json`, `.vscode/launch.json`,
  `.vscode/settings.json`, `.vscode/tasks.json`
- `assets/column-row-major.png`
- `python/__pycache__/tokenize.cpython-312.pyc`

문서·메타:

- `README.md`, `LICENSE`
- `guide.md`, `decision-log.md`, `inventory/files.jsonl`
