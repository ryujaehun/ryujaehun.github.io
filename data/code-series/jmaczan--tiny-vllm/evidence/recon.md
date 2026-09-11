# tiny-vllm 정찰 메모

기준 커밋: `1896ff5c37a241050dbcd9527caf9dc1d3087a61` (`guide.md`에 명시).
이 메모는 `guide.md`와 `inventory/files.jsonl`만 근거로 한다. 저장소를 실행하지 않았고, 빌드/커널 동작을 검증하지 않았다.

## Observations

- 인벤토리는 총 31개 항목을 기록한다(`inventory/files.jsonl`). 이 중 `role`이 `build`인 항목은 `CMakeLists.txt` 1개, `test`인 항목은 `tests/test_softmax.cu` 1개, `docs`인 항목은 `README.md` 1개이고 나머지는 `source`다(`inventory/files.jsonl`).
- 진입점 후보는 `src/main.cpp`이다. 크기 53377바이트, 1044줄, 언어 `cpp`로 인벤토리에 기록된다(`inventory/files.jsonl:29`).
- CUDA 커널 구현은 `src/kernels.cu`(19396바이트, 528줄, 언어 `cuda`)와 선언 헤더 `src/kernels.cuh`(1422바이트, 28줄)에 있다(`inventory/files.jsonl:27`, `inventory/files.jsonl:28`).
- 이식성 헤더 `src/cuda_to_hip.h`(2008바이트, 61줄)가 존재한다(`inventory/files.jsonl:26`). 이름상 CUDA→HIP 매크로 계층으로 추정되나 내용은 확인하지 않았다.
- 테스트는 `tests/test_softmax.cu`(4571바이트, 123줄, 언어 `cuda`) 한 개만 인벤토리에 있다(`inventory/files.jsonl:31`). 소프트맥스 이외 커널 테스트는 목록에 없다.
- 파이썬 참조 구현군은 `python/reference.py`(4821바이트, 122줄), `python/rms_norm.py`(2143바이트, 59줄), `python/tokenizer.py`(1712바이트, 50줄), `python/batching_test_tokens.py`(3025바이트, 96줄), `python/decode_test.py`(434바이트, 12줄)로 구성된다(`inventory/files.jsonl:18`–`inventory/files.jsonl:23`).
- 교차검증 산출물로 `python/rms_norm_crosscheck.txt`(3770바이트, 162줄)와 `reference.txt`(13573바이트, 613줄)가 있다(`inventory/files.jsonl:22`, `inventory/files.jsonl:24`). 둘 다 언어가 `null`로 기록된다.
- 서드파티 단일 헤더 `include/json.hpp`가 967314바이트, 25830줄로 저장소에서 가장 큰 파일이다(`inventory/files.jsonl:14`). 직접 읽기 대상에서 제외하는 것이 합리적이다.
- 빌드/실행 스크립트군은 `build.sh`(3줄), `check.sh`(1줄), `full_test.sh`(1줄), `ncu.sh`(1줄), `nsys.sh`(5줄), `run.sh`(1줄), `test.sh`(2줄)이다(`inventory/files.jsonl:11`–`inventory/files.jsonl:13`, `inventory/files.jsonl:15`, `inventory/files.jsonl:16`, `inventory/files.jsonl:25`, `inventory/files.jsonl:30`). `nsys.sh`와 `ncu.sh`는 프로파일링 도구 호출로 보인다.
- 문서 `README.md`는 94055바이트, 1304줄이다(`inventory/files.jsonl:9`). `guide.md`는 README 설명을 보조 근거로 두고 충돌 시 코드와 테스트를 우선한다고 명시한다.
- `assets/column-row-major.png`(34867바이트)가 있다(`inventory/files.jsonl:10`). 행/열 우선 저장 순서 설명용 자산으로 보인다.
- `python/__pycache__/tokenize.cpython-312.pyc`(2696바이트)는 바이트코드 산출물로 소스가 아니다(`inventory/files.jsonl:17`).
- `.vscode/` 아래 `c_cpp_properties.json`, `launch.json`, `settings.json`, `tasks.json`이 있다(`inventory/files.jsonl:3`–`inventory/files.jsonl:6`). 편집기 설정이며 시리즈 서술 대상은 아니다.
- `guide.md`의 서사 순서는 (1) 실행 가능한 프로그램과 가중치 포맷, (2) 토큰→로짓 순전파, (3) 다수 요청을 위한 메모리/스케줄링, (4) paged attention이다.

## Execution paths

아래는 인벤토리 메타데이터와 `guide.md` 서사로부터의 추정이며, 실행으로 검증한 것이 아니다.

- 빌드 경로: `CMakeLists.txt`가 빌드 정의이고 `build.sh`가 이를 감싼다. 이후 `run.sh`가 산출 실행 파일을 구동하는 형태로 추정된다(`CMakeLists.txt`, `build.sh`, `run.sh`).
- 프로그램 진입 경로: `src/main.cpp`가 모델 로딩과 생성 루프를 담고, 연산은 `src/kernels.cu`/`src/kernels.cuh`로 위임하는 구조로 추정된다(`src/main.cpp`, `src/kernels.cu`, `src/kernels.cuh`). `guide.md`의 "모델 로딩 → 1토큰 생성" 흐름과 일치한다.
- 테스트 경로: `test.sh`가 `tests/test_softmax.cu`를, `check.sh`/`full_test.sh`가 추가 검증을, `python/decode_test.py`와 `python/batching_test_tokens.py`가 파이썬 측 대조를 담당하는 것으로 추정된다(`test.sh`, `tests/test_softmax.cu`, `check.sh`, `full_test.sh`, `python/decode_test.py`, `python/batching_test_tokens.py`).
- 참조 대조 경로: `python/reference.py`, `python/rms_norm.py`, `python/tokenizer.py`가 CUDA 구현과 대조할 참조 계산을 제공하고, `python/rms_norm_crosscheck.txt`와 `reference.txt`가 그 출력 스냅샷으로 보인다(`python/reference.py`, `python/rms_norm.py`, `python/tokenizer.py`, `python/rms_norm_crosscheck.txt`, `reference.txt`).
- 프로파일링 경로: `ncu.sh`와 `nsys.sh`가 `src/main.cpp` 실행을 대상으로 Nsight 도구를 호출하는 형태로 추정된다(`ncu.sh`, `nsys.sh`, `src/main.cpp`).
- 이식 경로: `src/cuda_to_hip.h`가 `src/kernels.cu`의 CUDA API를 HIP으로 치환하는 헤더로 추정된다(`src/cuda_to_hip.h`, `src/kernels.cu`).

## Risks and gaps

- `src/main.cpp` 1044줄 전체를 읽지 않았다. 실제 로딩·생성·배칭·paged attention의 호출 순서와 함수 경계는 미확인이다(`src/main.cpp`).
- `src/kernels.cu` 528줄의 커널 목록을 확인하지 않았다. `guide.md`가 말하는 paged attention 커널이 실제로 이 파일에 있는지, 별도 파일인지 미확인이다(`src/kernels.cu`).
- 테스트가 `tests/test_softmax.cu` 하나뿐이므로 RMSNorm, GEMM, attention 커널의 정확성은 인벤토리상 테스트로 뒷받침되지 않는다(`tests/test_softmax.cu`). `guide.md`의 "코드와 테스트가 충돌을 결정한다" 원칙에 비추어 커널별 검증 근거가 부족하다.
- `include/json.hpp`는 25830줄 서드파티 코드다. 가중치/설정 파싱이 이 헤더에 의존하면 서술 범위에서 제외하고 호출 지점만 인용해야 한다(`include/json.hpp`).
- `python/__pycache__/tokenize.cpython-312.pyc`는 산출물이다. 소스 `python/tokenizer.py`와의 대응은 확인 전까지 단정하지 않는다(`python/__pycache__/tokenize.cpython-312.pyc`, `python/tokenizer.py`).
- `reference.txt`(613줄)와 `python/rms_norm_crosscheck.txt`(162줄)는 언어가 `null`이라 생성 방식과 신뢰도가 불명확하다. 참조 계산의 입력과 커밋 시점을 확인해야 한다(`reference.txt`, `python/rms_norm_crosscheck.txt`).
- `src/cuda_to_hip.h`의 존재는 CUDA/HIP 이중 지원을 시사하나, 어느 경로가 기본 빌드인지 `CMakeLists.txt`를 읽기 전에는 알 수 없다(`src/cuda_to_hip.h`, `CMakeLists.txt`).
- `README.md` 1304줄은 성능 수치와 설계 주장을 포함할 수 있으나, `guide.md`가 "성능 주장은 범위 밖"이라 했으므로 인용 시 코드 근거와 분리해야 한다(`README.md`, `guide.md`).
- `python/rms_norm.py`와 `python/reference.py`의 역할 중복 가능성이 있다. 어느 쪽이 정본 참조인지 미확인이다(`python/rms_norm.py`, `python/reference.py`).

## Source paths

- `CMakeLists.txt`
- `src/main.cpp`
- `src/kernels.cu`
- `src/kernels.cuh`
- `src/cuda_to_hip.h`
- `include/json.hpp`
- `tests/test_softmax.cu`
- `python/reference.py`
- `python/rms_norm.py`
- `python/tokenizer.py`
- `python/batching_test_tokens.py`
- `python/decode_test.py`
- `python/rms_norm_crosscheck.txt`
- `reference.txt`
- `build.sh`
- `run.sh`
- `test.sh`
- `check.sh`
- `full_test.sh`
- `ncu.sh`
- `nsys.sh`
- `README.md`
- `assets/column-row-major.png`
- `python/__pycache__/tokenize.cpython-312.pyc`
- `.vscode/c_cpp_properties.json`
- `.vscode/launch.json`
- `.vscode/settings.json`
- `.vscode/tasks.json`
- `.gitattributes`
- `.gitignore`
- `LICENSE`
