# 정찰 메모: jmaczan/tiny-vllm

## Observations

- 시리즈 대상은 `jmaczan/tiny-vllm`의 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61`로 고정되어 있다 (`guide.md`).
- 인벤토리 `inventory/files.jsonl`은 총 31개 파일을 나열하며, 역할(role)은 `source`, `build`, `docs`, `test`로 구분된다.
- 실행 진입점으로 보이는 파일은 `src/main.cpp`(1045줄)이며, 이것이 저장소에서 가장 큰 1차 소스 파일이다 (`inventory/files.jsonl`).
- CUDA 커널 본문은 `src/kernels.cu`(529줄)에 있고, 선언은 `src/kernels.cuh`(28줄)에 분리되어 있다 (`inventory/files.jsonl`).
- `src/cuda_to_hip.h`(62줄)가 있어 CUDA 코드의 HIP 이식 계층이 존재함을 시사한다 (`inventory/files.jsonl`).
- 빌드 관련 파일은 `CMakeLists.txt`(76줄, role `build`), `build.sh`, `run.sh`, `test.sh`, `check.sh`, `full_test.sh`이다 (`inventory/files.jsonl`).
- 가중치/설정 파싱에 쓰일 것으로 보이는 대형 벤더 헤더 `include/json.hpp`(967,314바이트, 25,831줄)가 저장소에 포함되어 있다 (`inventory/files.jsonl`).
- 문서는 `README.md`(94,055바이트, 1305줄, role `docs`)와 `reference.txt`(614줄)가 담당한다 (`inventory/files.jsonl`).
- Python 참조 구현으로 `python/reference.py`(123줄), `python/tokenizer.py`(51줄), `python/rms_norm.py`(60줄)가 있다 (`inventory/files.jsonl`).
- Python 검증 보조 파일로 `python/decode_test.py`(12줄), `python/batching_test_tokens.py`(96줄), `python/rms_norm_crosscheck.txt`(162줄)가 있다 (`inventory/files.jsonl`).
- 자동화 테스트로 분류된 파일은 `tests/test_softmax.cu`(role `test`, 124줄) 하나뿐이다 (`inventory/files.jsonl`).
- 프로파일링 스크립트로 `ncu.sh`, `nsys.sh`가 있다 (`inventory/files.jsonl`).
- `assets/column-row-major.png`가 있어 행/열 우선순위 레이아웃 설명용 도해가 존재한다 (`inventory/files.jsonl`).
- `python/__pycache__/tokenize.cpython-312.pyc`(2,696바이트)가 인벤토리에 포함되어 있어 바이너리 캐시가 저장소에 커밋되어 있다 (`inventory/files.jsonl`).
- 시리즈 범위는 모델 로딩부터 단일 토큰 생성, 배칭, paged KV-cache attention까지이며 학습, 분산 서빙, 저장소 외부 성능 주장은 제외된다 (`guide.md`).
- 증거 규칙상 모든 구현 주장은 고정 소스 경로와 줄 범위를 인용해야 하고, 충돌 시 코드와 테스트가 README 설명보다 우선한다 (`guide.md`).
- 독자 계약상 C++ 기초, CUDA 스레드/블록 용어, 행렬곱 직관이 전제된다 (`guide.md`).

## Execution paths

- `guide.md`의 시리즈 아크는 (1) 실행 가능한 프로그램과 가중치 포맷, (2) 토큰→로짓 forward 경로, (3) 다수 요청을 위한 메모리·스케줄링, (4) 데이터 구조와 커널 설계가 만나는 paged attention 순서를 제시한다.
- 빌드 진입은 `CMakeLists.txt`(role `build`)와 이를 감싸는 `build.sh`가 담당하는 것으로 인벤토리상 배치되어 있다 (`inventory/files.jsonl`).
- 실행 진입은 `run.sh`, 전체 검증은 `full_test.sh`/`test.sh`, 단일 검사는 `check.sh`로 분리되어 있다 (`inventory/files.jsonl`).
- C++ 측 forward/생성 로직은 `src/main.cpp`에, GPU 연산은 `src/kernels.cu`와 `src/kernels.cuh`에 위치한다 (`inventory/files.jsonl`).
- 커널 단위 검증 경로는 `tests/test_softmax.cu`로 시작하며, 소프트맥스 커널이 첫 검증 대상임을 시사한다 (`inventory/files.jsonl`).
- 수치 대조 경로는 `python/reference.py`, `python/rms_norm.py`와 그 출력 `python/rms_norm_crosscheck.txt`로 이어진다 (`inventory/files.jsonl`).
- 토크나이저 경로는 `python/tokenizer.py`와 그 캐시 `python/__pycache__/tokenize.cpython-312.pyc`가 담당한다 (`inventory/files.jsonl`).
- 성능 분석 경로는 `ncu.sh`, `nsys.sh`로 분리되어 있다 (`inventory/files.jsonl`).
- `guide.md`는 배칭과 paged attention을 forward 경로 이후의 확장 단계로 규정한다.

## Risks and gaps

- `README.md`가 1305줄로 매우 크지만 `guide.md`의 증거 규칙은 README를 보조 증거로만 두므로, README 서술을 그대로 기사 근거로 쓰면 안 된다.
- `include/json.hpp`가 25,831줄의 벤더 라이브러리이므로 시리즈 본문에서 직접 분석 대상으로 삼으면 노이즈가 커진다 (`inventory/files.jsonl`).
- `tests/test_softmax.cu` 외에 커널/배칭/paged attention을 검증하는 테스트 파일이 인벤토리에 없어, 해당 경로 주장은 소스 코드 인용에 의존해야 한다 (`inventory/files.jsonl`).
- `guide.md`가 명시한 paged KV-cache attention과 배칭이 어느 소스 파일·줄에 구현되었는지는 인벤토리만으로 확정할 수 없다.
- `src/cuda_to_hip.h`가 존재하므로 CUDA/HIP 이식 분기가 실제 실행 경로에 영향을 줄 수 있으나 인벤토리만으로 분기 범위를 알 수 없다 (`inventory/files.jsonl`).
- 인벤토리는 파일 크기·줄 수·역할만 제공하고 함수·심볼 정보는 없으므로, `guide.md`의 줄 범위 인용 규칙을 채우려면 별도 소스 열람이 필요하다.
- `python/__pycache__/tokenize.cpython-312.pyc` 같은 생성물이 포함되어 있어, "소스"로 분류된 항목 중 실제 편집 대상이 아닌 파일이 섞여 있다 (`inventory/files.jsonl`).
- 이 메모는 저장소를 실행하지 않았으며, 모든 서술은 `guide.md`와 `inventory/files.jsonl` 두 첨부에 한정된다.

## Source paths

- `guide.md`
- `inventory/files.jsonl`
- `src/main.cpp`
- `src/kernels.cu`
- `src/kernels.cuh`
- `src/cuda_to_hip.h`
- `include/json.hpp`
- `CMakeLists.txt`
- `build.sh`, `run.sh`, `test.sh`, `check.sh`, `full_test.sh`
- `ncu.sh`, `nsys.sh`
- `tests/test_softmax.cu`
- `python/reference.py`, `python/tokenizer.py`, `python/rms_norm.py`
- `python/decode_test.py`, `python/batching_test_tokens.py`, `python/rms_norm_crosscheck.txt`
- `python/__pycache__/tokenize.cpython-312.pyc`
- `README.md`, `reference.txt`
- `assets/column-row-major.png`
