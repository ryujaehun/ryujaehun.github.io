# Mini-SGLang guide decision log

## 001 - 제품 범위: python/ 만 제품으로 본다

- Observation: `compatibility-report.md` 가 top-level 소스 루트 3개(`benchmark`, `python`, `tests`)를 임계값 2개 초과로 보고했다. 규모는 `python/` 10,061줄(`minisgl` 파이썬 8,090줄 + `python/minisgl/kernel/csrc` CUDA/C++ 1,971줄), `tests/` 649줄, `benchmark/` 321줄이다. `pyproject.toml` 의 `[tool.setuptools] package-dir = {"" = "python"}` 은 `python/` 만 패키징한다.
- Question: 이 시리즈가 다룰 제품의 범위를 어디로 정할 것인가?
- Recommendation: `python/minisgl` 만 제품으로 삼고 `tests/` 와 `benchmark/` 는 독립 편을 만들지 않고 주장을 뒷받침하는 인용 근거로만 쓴다.
- User decision: 권장안 채택 — `python/` 만 제품, `tests/` 와 `benchmark/` 는 근거 전용.
- Guide impact: 편 분할은 `python/minisgl` 의 런타임 경로를 따른다. `tests/core/test_cache_allocate.py` 등은 해당 편의 `code_scopes` 에 근거로 포함될 수 있으나 편의 주제가 되지는 않는다. `benchmark/` 는 성능 수치 인용의 출처로만 언급한다.
- Compatibility finding: `monorepo-roots`

## 002 - 실행 검증: 정적 근거 전용

- Observation: `compatibility-report.md` 의 `execution-environment` finding 은 CUDA core source 를 관찰했고 CPU 전용 검증 경로가 문서화되어 있지 않다고 보고했다. `python/minisgl/engine/engine.py` 는 `torch.cuda.set_device` 로 시작하고, `python/minisgl/kernel/store.py` 와 `python/minisgl/kernel/index.py` 는 tvm-ffi 로 `.cu` 를 런타임 JIT 컴파일한다. `pyproject.toml` 의존성에 `sgl_kernel`, `flashinfer-python` 이 있다. README 는 측정 환경을 1xH200 / 4xH200 으로 밝힌다. 이 시리즈를 쓰는 호스트에는 `nvidia-smi` 도 `nvcc` 도 없다.
- Question: GPU 가 없는 환경에서 어떤 검증 프로파일로 주장을 뒷받침할 것인가?
- Recommendation: 정적 근거 전용. 모든 편의 `required_evidence.runtime` 을 0 으로 두고, 성능 수치는 측정하지 않으며 upstream 인용 시 출처와 환경을 문장에서 밝힌다.
- User decision: 권장안 채택 — 정적 근거 전용.
- Guide impact: `series.yaml` 에 `execution` 블록을 두지 않는다. 모든 편의 runtime 근거는 0. `tests/core/test_cache_allocate.py` 는 실행하지 않고 커밋된 소스로만 인용한다. 커널 실행시간·처리량·메모리 사용량은 측정값으로 제시하지 않는다.
- Compatibility finding: `execution-environment`

## 003 - 빌드 매니페스트: 근거로만 포함

- Observation: `compatibility-report.md` 의 `unsupported-build-system` finding 은 `pyproject.toml` 에 결정론적 의존성 추출기가 없다고 보고했다. 실제 내용은 표준 setuptools 매니페스트이며(`build-backend = "setuptools.build_meta"`, `package-dir = {"" = "python"}`), 서술 가치가 있는 부분은 `dependencies` 목록이다. `sgl_kernel`, `flashinfer-python`, `apache-tvm-ffi`, `quack-kernels` 가 "저장소가 직접 구현하지 않고 빌려 쓰는 커널" 의 경계를 규정한다.
- Question: `pyproject.toml` 을 가이드 범위에 넣을 것인가, 넣는다면 어떤 형태로 넣을 것인가?
- Recommendation: 근거로만 포함한다. 의존 커널 경계를 설명하는 편에서 인용하고, 빌드·설치를 주제로 하는 편은 만들지 않는다.
- User decision: 권장안 채택 — 근거로만 포함, 전용 편 없음.
- Guide impact: `pyproject.toml` 은 어텐션 백엔드 편의 `code_scopes` 에 근거로 들어가고, 가이드의 "템플릿 적합성 및 실행 제약" 절이 의존성 경계를 적는다. 빌드 시스템 추출기 확장은 이 프로젝트에서 수행하지 않는다.
- Compatibility finding: `unsupported-build-system`

## 004 - Dockerfile: 실행 환경 근거로만 포함

- Observation: `compatibility-report.md` 의 `unclassified-unanalyzable-file` finding 은 `Dockerfile` 1개를 결정론적 분류 없는 파일로 보고했다. 내용은 `CUDA_VERSION=12.8.1`, `PYTHON_VERSION=3.12`, `EXPOSE 1919`, `ENTRYPOINT ["python", "-m", "minisgl"]` 을 고정하며, 저장소가 실행 환경을 스스로 규정한 유일한 지점이다.
- Question: `Dockerfile` 을 가이드 범위에 넣을 것인가?
- Recommendation: 근거로만 포함한다. 가이드의 실행 제약 절에서 환경을 밝힐 때 인용하고, 컨테이너·배포를 주제로 하는 편은 만들지 않는다.
- User decision: 권장안 채택 — 근거로만 포함, 전용 편 없음.
- Guide impact: `Dockerfile` 은 편의 `code_scopes` 에 들어가지 않는다. 가이드의 "템플릿 적합성 및 실행 제약" 절이 CUDA 버전과 기본 포트를 여기서 인용한다. 나머지 미분류 파일(`.gitignore`, `.pre-commit-config.yaml`, `LICENSE`, `assets/logo.png`)은 제외한다.
- Compatibility finding: `unclassified-unanalyzable-file`

## 005 - 독자 계약: 개념은 전제, 분량은 CPU 측 스케줄링과 메모리 장부

- Observation: 저장소의 구별되는 층은 CPU 쪽에 있다. `scheduler/scheduler.py:83-106` 의 `overlap_loop` 은 두 CUDA 스트림으로 이번 배치 실행과 지난 배치 결과 처리를 겹치고, 다음 배치 입력을 GPU 의 `token_pool` 에서 직접 읽는다(`scheduler/scheduler.py:229-231`). `engine/engine.py:66` 은 page table 이 페이지 번호가 아니라 raw token location 을 담는다고 밝히고 `core.py:103` 은 이 표를 항상 page_size=1 로 취급한다고 못 박는다. `scheduler/prefill.py:23-29` 의 `ChunkedReq` 는 `can_decode` 를 False 로 두어 decode manager 에 들어가지 않게 만든다. `layers/base.py` 는 `torch.nn.Module` 대신 `self.__dict__` 를 걸어 `_` 접두사를 건너뛰는 자체 `state_dict` 를 구현한다. README 와 `docs/features.md` 는 Sarathi-Serve(arXiv 2403.02310), NanoFlow(arXiv 2408.12757), LMSYS radix attention / v0.4 블로그, FlashAttention, FlashInfer 를 이미 인용한다.
- Question: 독자를 누구로 두고 분량을 어디에 쓸 것인가?
- Recommendation: 독자는 추론 최적화 개념을 알지만 서빙 프레임워크 소스를 읽어 본 적 없는 사람으로 두고, 분량은 overlap scheduling, chunked prefill, radix cache 와 page table, CUDA graph 패딩에 쓴다.
- User decision: 권장안 채택 — CPU 측 스케줄링과 메모리 장부에 분량을 쓴다.
- Guide impact: 개념(prefill/decode, KV cache, PagedAttention, continuous batching, TP)은 한 문장 + 저장소가 이미 인용한 출처 링크로 처리하고 삭제하지 않는다. 편 배분은 스케줄러와 캐시 쪽에 무게를 싣고, layers/models/attention 백엔드는 스케줄러가 만든 메타데이터를 소비하는 쪽으로만 다룬다. 멀티프로세스·ZMQ 구조는 진입 편에서 한 번 세우고 반복하지 않는다.
- Compatibility finding: 해당 없음

## 006 - 대표 실행 경로: 온라인 서빙

- Observation: 저장소는 두 실행 경로를 제공한다. `python/minisgl/server/launch.py:40-113` 의 `launch_server` 는 TP 크기만큼 scheduler 프로세스와 tokenizer·detokenizer 를 spawn 하고 ZMQ 로 연결한다. `python/minisgl/llm/llm.py:28-98` 의 `class LLM(Scheduler)` 는 `offline_mode=True` 로 `offline_receive_msg` / `offline_send_result` 를 갈아끼워 단일 프로세스에서 돈다(`scheduler/io.py:30-33`). `docs/structures.md` 는 온라인 경로의 8단계 request lifecycle 을 이미 문서화했고 README 의 첫 실행 명령도 온라인이다.
- Question: 진입점부터 결과까지 추적할 대표 실행 경로를 무엇으로 고정할 것인가?
- Recommendation: 온라인 서빙 경로. 오프라인 `LLM` 경로는 "같은 스케줄러에서 ZMQ 만 뺀 것" 이라는 대조로 한 문단만 쓴다.
- User decision: 권장안 채택 — 온라인 서빙.
- Guide impact: 모든 편은 `python -m minisgl --model Qwen/Qwen3-0.6B` 로 시작하는 하나의 경로 위에 놓인다. 1편이 프로세스 경계와 메시지 흐름을 세우고, 이후 편은 그 경로의 한 구간을 깊게 판다. 오프라인 경로는 1편에서 한 번만 대조로 언급한다.
- Compatibility finding: 해당 없음

## 007 - 시리즈 깊이: 8편

- Observation: 제품(`python/`)은 파이썬 8,090줄과 `python/minisgl/kernel/csrc` 1,971줄이며 inventory 기준 python 101개, c++ 5개, cuda 5개 파일이다. 결정 005 는 분량을 CPU 측 스케줄링과 메모리 장부에 싣기로 했고, 결정 006 은 대표 경로를 온라인 서빙으로 고정했다. 요청이 지나가는 순서는 프로세스 경계 → 요청 상태 → prefill 배치 구성 → KV 슬롯 할당 → 접두사 재사용 → overlap 실행 → 커널 메타데이터와 CUDA graph → TP 로 이어지며, 코드 의존성도 같은 방향이다.
- Question: 시리즈를 몇 편으로 나누고 어떤 분할을 쓸 것인가?
- Recommendation: 8편. 2~6편 다섯 편을 스케줄러와 캐시에 배정해 결정 005 의 무게 중심을 편 수로 구현한다.
- User decision: 권장안 채택 — 8편.
- Guide impact: `series.yaml` 은 8개 chapter 를 가지며 각 편은 직전 편에만 의존하는 선형 사슬이다. 모델 실행 경로(layers·models)와 tvm-ffi JIT 커널 바인딩은 전용 편을 두지 않고 7·8편에서 필요한 만큼만 다룬다. 게시는 하루 한 편, 한국어·영어 쌍이므로 16개 페이지 / 8일이다.
- Compatibility finding: 해당 없음
