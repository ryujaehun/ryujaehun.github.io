# Reconnaissance memo (sgl-project--mini-sglang, `9a91cfafe754aa85daee49998176275667eb58f2`)

본 메모는 `guide.md`와 `inventory/files.jsonl`에 기록된 고정 리비전 정적 정보만을 정리하며, 저장소 코드 실행을 주장하지 않는다 (`python/minisgl/engine/engine.py` 생성자의 CUDA 요구사항 때문에 실행하지 않는다).

## Observations

- 분석 기준은 `https://github.com/sgl-project/mini-sglang`의 고정 커밋이며, 기본 브랜치 재해석을 하지 않는다는 점이 `guide.md`에 고정되어 있고 대상 트리에는 `python/minisgl/__main__.py` 진입점이 포함된다.
- 포함 범위는 `python/minisgl/scheduler/` 중심의 `python/` 하나이며, `tests/`와 `benchmark/`는 독립 주제가 아니라 인용 근거 전용이라는 점이 `guide.md`의 포함·제외 표에 명시되어 있다.
- 규모는 `python/` 10,061줄(파이썬 8,090줄 + `python/minisgl/kernel/csrc` 1,971줄)이며, 인벤토리에 집계된 전체 120개 파일의 언어 분포는 `python/minisgl/` 파이썬 파일군을 중심으로 python 101, c++ 5, cuda 5, markdown 3, 미분류 6이다.
- 제품 프로세스 경계는 다섯이며, 책임 분담은 `python/minisgl/server/api_server.py`의 API 수신, `python/minisgl/tokenizer/server.py`의 토큰 변환, `python/minisgl/scheduler/scheduler.py`의 스케줄링, `python/minisgl/engine/engine.py`의 forward 소유, `python/minisgl/message/utils.py`의 직렬화로 나뉜다.
- 제어 메시지는 ZMQ, TP 텐서는 NCCL 경로이며, rank 간 수신 개수 동기화 방식은 `python/minisgl/scheduler/io.py`의 rank0·rank1 분기 수신 함수에 있다.
- `Scheduler` 내부는 네 매니저 분담이며, 슬롯 관리는 `python/minisgl/scheduler/table.py`의 `TableManager`, 페이지·prefix 관리는 `python/minisgl/scheduler/cache.py`의 `CacheManager`, 대기열·배치 구성은 `python/minisgl/scheduler/prefill.py`의 `PrefillManager`, 진행 집합은 `python/minisgl/scheduler/decode.py`의 `DecodeManager`가 맡는다.
- 요청 상태 중심은 `python/minisgl/core.py`의 `Req`·`Batch`·`Context` 정의이며, 길이 필드 3종(`cached_len`·`device_len`·`max_device_len`)의 정의 소유 편도 `python/minisgl/core.py`이다.
- page table 특이점은 `python/minisgl/engine/engine.py` 주석이 페이지 번호가 아닌 raw token location을 담는다고 밝히고, `python/minisgl/core.py`가 page_size=1 취급을 못 박으며, 페이지 번호 변환은 `python/minisgl/attention/fa.py`의 슬라이싱·나눗셈에서 수행된다.
- chunked prefill 특이점은 별도 큐가 아니라 `python/minisgl/scheduler/prefill.py`의 `ChunkedReq` 서브클래스(`can_decode` False 고정, `append_host` 예외) 형태이다.
- overlap 특이점은 다음 입력 토큰을 CPU로 가져오지 않고 GPU `token_pool`에 쓴 직전 샘플링 결과를 재사용하며, 해당 루프 소유자는 `python/minisgl/scheduler/scheduler.py`이다.
- CUDA graph 특이점은 `python/minisgl/engine/engine.py`가 만든 `table_idx == max_running_req` 더미 요청과 `num_pages + 1` 더미 페이지를 `python/minisgl/engine/graph.py`의 패딩 로직이 캡처 크기까지 채우는 구조이다.
- 레이어 특이점은 `python/minisgl/layers/base.py`의 `BaseOP`가 `torch.nn.Module` 대신 `self.__dict__` 순회 자체 `state_dict`를 구현하며, `python/minisgl/layers/linear.py`의 `LinearOProj._comm` 언더스코어 명명이 그 규칙과 연결된다.
- radix 탐색 특이점은 키 비교가 파이썬이 아니라 `python/minisgl/kernel/radix.py` 바인딩을 통한 `python/minisgl/kernel/csrc/src/radix.cpp`의 AOT C++ 함수 `fast_compare_key`에서 수행된다.
- 해제 특이점은 배치 종료 시 해제를 `python/minisgl/scheduler/cache.py`의 `lazy_free_region` 컨텍스트에서 `_free` 임시 교체 후 `torch.cat` 1회로 모아서 처리한다.
- 외부 개념 출처는 저장소 `docs/features.md`가 이미 인용한 Sarathi-Serve, NanoFlow, LMSYS 블로그, FlashAttention·FlashInfer·TensorRT-LLM fmha이며, 원본 프로젝트 인용은 `README.md` 도입부에 있다.

## Execution paths

- 대표 경로는 온라인 서빙 하나이며, 진입은 `python/minisgl/__main__.py`를 통한 `python -m minisgl` 호출에서 시작해 `python/minisgl/server/launch.py`의 `launch_server`로 이어진다.
- 기동 순서는 `python/minisgl/server/launch.py`가 TP 크기만큼 `python/minisgl/scheduler/scheduler.py`의 scheduler 프로세스와 `python/minisgl/tokenizer/server.py`의 tokenizer·detokenizer를 spawn한 뒤 ack를 모아 `python/minisgl/server/api_server.py`를 연다.
- 요청 유입은 `python/minisgl/server/api_server.py`의 `/v1/chat/completions`·`/generate`가 받은 텍스트가 `python/minisgl/tokenizer/server.py`의 tokenizer 프로세스에서 토큰 ID 열로 바뀌어 rank 0 스케줄러로 전달되는 흐름이며, 토큰 I/O 메시지 정의는 `python/minisgl/message/tokenizer.py`에 있다.
- rank 확散은 rank 0이 받은 메시지를他 rank에 뿌리고 모든 rank의 `python/minisgl/scheduler/prefill.py` 대기열에 동일 요청을 넣으며, 바이트 PUB·개수 broadcast 구현은 `python/minisgl/scheduler/io.py`에 있다.
- 배치 구성은 `python/minisgl/scheduler/scheduler.py`의 `_schedule_next_batch`가 prefill을 먼저 시도하고 없으면 decode를 만들며, 예산·분할 규칙 본체는 `python/minisgl/scheduler/prefill.py`에 있고 설정 상수는 `python/minisgl/scheduler/config.py`에 있다.
- 배치 준비는 `python/minisgl/scheduler/scheduler.py`의 `_prepare_batch`가 `python/minisgl/scheduler/cache.py`의 페이지 할당·page table 기입을 호출한 뒤 `python/minisgl/attention/fa.py` 계열이 소비할 어텐션 메타데이터를 만들며, KV 버퍼 실체는 `python/minisgl/kvcache/mha_pool.py`에 있다.
- 실행은 `python/minisgl/engine/engine.py`가 forward를 수행하고, decode·캡처 범위 내에서는 `python/minisgl/engine/graph.py`의 graph 재생 경로를 타고 그 외에는 모델을 그대로 돌며, 샘플링 구현은 `python/minisgl/engine/sample.py`에 있다.
- 토큰 전달은 샘플링 결과를 GPU `token_pool`에 기록해 다음 배치 입력으로 쓰고 동시에 CPU로 비동기 복사하며, 관련 루프·대기 로직은 `python/minisgl/scheduler/scheduler.py`의 `overlap_loop`·`_process_last_data`와 `python/minisgl/env.py` 스위치에 있다.
- 결과 회신은 다음 루프가 `python/minisgl/scheduler/scheduler.py`에서 직전 CPU 복사 완료를 기다려 종료를 판정한 뒤 rank 0이 `python/minisgl/tokenizer/detokenize.py` 측 detokenizer로 `DetokenizeMsg`를 보내고, `python/minisgl/server/api_server.py`가 스트리밍으로 사용자에게 흘리며, 메시지 직렬화 규칙은 `python/minisgl/message/utils.py`에 있다.
- 오프라인 대조 경로는 `python/minisgl/llm/llm.py`의 `class LLM(Scheduler)`가 동일 스케줄러에서 ZMQ I/O만 걷어낸 형태이며, 이후 편은 온라인 경로를 전제한다.

## Risks and gaps

- 실행 근거가 0이며 정적 인용만 허용되므로, `python/minisgl/engine/engine.py` 생성자의 CUDA 디바이스 요구나 `python/minisgl/kernel/store.py`·`python/minisgl/kernel/index.py`의 tvm-ffi `.cu` JIT 컴파일(`python/minisgl/kernel/csrc/jit/store.cu`, `python/minisgl/kernel/csrc/jit/index.cu`)을 현재 환경에서 검증할 수 없다.
- 의존 커널이 Linux 전용 CUDA를 요구한다는 제약은 `README.md`에 기술되어 있고, 실행 이미지 고정은 `Dockerfile`의 CUDA 12.8.1·Ubuntu 24.04·Python 3.12·기본 포트 1919·`python -m minisgl` 실행 지정에 있으므로, 빌드·설치·컨테이너를 분량 대상으로 삼지 않는다.
- 성능 수치를 측정하지 않으므로, `README.md`의 1xH200 offline·4xH200 online 측정 인용 시에도 출처·하드웨어를 같은 문장에서 밝히고 `benchmark/offline/bench.py`·`benchmark/online/bench_simple.py`·`python/minisgl/benchmark/perf.py`는 수치 출처 언급 용도로만 사용한다.
- `tests/core/test_cache_allocate.py`는 CPU fixture로 돌 수 있는 형태이지만 결정상 실행하지 않고 커밋된 소스 인용만 하며, `tests/core/test_scheduler.py`·`tests/kernel/test_store.py`·`tests/kernel/test_index.py`·`tests/misc/test_serialize.py`도 동일하게 근거 전용으로만 쓴다.
- `pyproject.toml`은 전용 편 없이 `python/minisgl/kernel/utils.py`·`python/minisgl/kernel/pynccl.py` 등이 가리키는 의존 커널 경계 설명 때만 인용하므로, 의존성 해석이 선언 파일 추측으로 흐르지 않게 주의가 필요하다.
- 공유 파일 3종(`python/minisgl/scheduler/scheduler.py`, `python/minisgl/engine/engine.py`, `python/minisgl/scheduler/cache.py`)은 편별 줄 범위 분담이 필요하며, 특히 `python/minisgl/scheduler/scheduler.py`의 `_prepare_batch`와 `python/minisgl/engine/engine.py`의 `__init__`은 편 간 중복 정의가 생기기 쉬운 지점이다.
- C++/CUDA 경계(`python/minisgl/kernel/csrc/src/radix.cpp`, `python/minisgl/kernel/csrc/src/pynccl.cu`, `python/minisgl/kernel/csrc/include/minisgl/tensor.h`)는 파이썬 장부 해석만으로 단정하기 어려우므로, 4·5·8편에서 필요한 최소 인용 범위를 넘지 않게 주의가 필요하다.

## Source paths

- 1편 process-topology 대표 경로: `python/minisgl/server/launch.py`, `python/minisgl/server/api_server.py`, `python/minisgl/tokenizer/server.py`, `python/minisgl/message/utils.py`, `python/minisgl/message/backend.py`, `python/minisgl/scheduler/io.py`, 보조로 `python/minisgl/message/tokenizer.py`와 `tests/misc/test_serialize.py`.
- 2편 request-state-ledger 대표 경로: `python/minisgl/core.py`, `python/minisgl/scheduler/utils.py`, `python/minisgl/scheduler/table.py`, `python/minisgl/scheduler/decode.py`.
- 3편 prefill-scheduling 대표 경로: `python/minisgl/scheduler/prefill.py`, `python/minisgl/scheduler/config.py`, `python/minisgl/scheduler/decode.py`, `python/minisgl/scheduler/scheduler.py`.
- 4편 page-table-and-allocation 대표 경로: `python/minisgl/scheduler/cache.py`, `python/minisgl/engine/engine.py`, `python/minisgl/kvcache/mha_pool.py`, `python/minisgl/kernel/store.py`, `python/minisgl/core.py`, 보조로 `tests/core/test_cache_allocate.py`.
- 5편 radix-prefix-cache 대표 경로: `python/minisgl/kvcache/radix_cache.py`, `python/minisgl/kvcache/base.py`, `python/minisgl/kvcache/naive_cache.py`, `python/minisgl/kernel/radix.py`, `python/minisgl/scheduler/cache.py`, 보조로 `python/minisgl/kernel/csrc/src/radix.cpp`.
- 6편 overlap-scheduling 대표 경로: `python/minisgl/scheduler/scheduler.py`, `python/minisgl/engine/engine.py`, `python/minisgl/env.py`, 보조로 `python/minisgl/engine/sample.py`.
- 7편 attention-metadata-and-cuda-graph 대표 경로: `python/minisgl/attention/base.py`, `python/minisgl/attention/fa.py`, `python/minisgl/attention/__init__.py`, `python/minisgl/engine/graph.py`, `python/minisgl/layers/attention.py`, 대조용으로 `python/minisgl/attention/fi.py`와 `python/minisgl/attention/trtllm.py`.
- 8편 tensor-parallel-ledger 대표 경로: `python/minisgl/distributed/impl.py`, `python/minisgl/distributed/info.py`, `python/minisgl/layers/linear.py`, `python/minisgl/layers/embedding.py`, `python/minisgl/layers/base.py`, `python/minisgl/models/weight.py`, `python/minisgl/scheduler/io.py`, 보조로 `tests/kernel/test_comm.py`와 `python/minisgl/kernel/pynccl.py`.
- 전 편 공통 문서 근거: 개념 출처용 `docs/features.md`, 구조 개요용 `docs/structures.md`, 수치 출처 한정용 `README.md`, 환경 한정용 `Dockerfile`, 라이선스 확인용 `LICENSE`.
