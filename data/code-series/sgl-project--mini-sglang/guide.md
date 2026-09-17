# Mini-SGLang 코드 읽기 시리즈 Guide

## 분석 대상과 고정 revision

- Repository: https://github.com/sgl-project/mini-sglang
- Commit: `9a91cfafe754aa85daee49998176275667eb58f2`
- Project slug: `sgl-project--mini-sglang`

이 시리즈의 모든 주장은 위 커밋의 트리만을 근거로 한다. 기본 브랜치를 다시
해석하거나 이후 revision 으로 대체하지 않는다.

## 독자, 목표와 선수 지식

독자는 LLM 추론 최적화를 개념으로는 아는 사람이다(결정 005). prefill 과 decode
의 구분, KV cache 가 존재하는 이유, PagedAttention 이 KV 를 페이지로 쪼개는
이유, continuous batching 의 목적, tensor parallelism 이 무엇을 나누는지는
설명 대상이 아니라 전제다. 다만 **전제는 설명의 길이를 한 문장으로 묶는 것이지
문장을 없애는 것이 아니다.** 각 개념은 한 문장 + 저장소가 이미 인용한 출처
링크로 처리한다.

목표는 그 개념들이 **동작하는 서빙 프레임워크의 CPU 쪽 코드에서 어떤 모양으로
구현되는가**를 보이는 것이다. 블로그 그림으로만 설명되던 층 — 배치를 짜는
예산 계산, 페이지 장부 기입, radix 트리의 참조 카운트, 두 CUDA 스트림의 겹침,
CUDA graph 를 위한 더미 요청 패딩 — 에 분량을 쓴다.

선수 지식은 세 가지다.

- 파이썬 타입 힌트와 dataclass, 컨텍스트 매니저
- PyTorch 텐서 인덱싱과 `torch.cuda.Stream` 의 존재 정도
- 트랜스포머 추론의 개념적 흐름

### 분량을 어디에 쓰는가

세 곳에 쓴다. 나머지는 이 셋을 설명하기 위한 받침이다.

1. **스케줄러의 배치 구성**: 무엇을 이번 배치에 넣을지 정하는 예산 계산과,
   예산을 넘는 프롬프트를 쪼개는 방법(3편).
2. **KV 메모리 장부**: 페이지 할당, page table 의 표현, 접두사 재사용과
   eviction(4·5편).
3. **overlap scheduling**: 두 스트림으로 CPU 작업을 GPU 계산 뒤에 숨기는
   구현(6편).

빌드·설치·컨테이너 설정은 분량의 대상이 아니다(결정 003, 004). 어느 저장소에서나
인용하기 가장 쉬운 자료지만 사람들이 이 저장소를 읽는 이유가 아니다.

### 독자에게 넘길 외부 출처

저장소가 이미 인용하고 있는 것들이다. 찾을 필요 없이 넘겨주면 된다.

| 주제 | 출처 | 저장소에서 인용된 위치 |
| --- | --- | --- |
| 원본 프로젝트 | SGLang | `README.md` 도입부 |
| chunked prefill | Sarathi-Serve (arXiv 2403.02310) | `docs/features.md` |
| overlap scheduling | NanoFlow (arXiv 2408.12757) | `docs/features.md` |
| radix attention | LMSYS blog 2024-01-17 | `docs/features.md` |
| overlap scheduling 도해 | LMSYS blog 2024-12-04 (SGLang v0.4) | `docs/features.md` |
| 어텐션 커널 | FlashAttention, FlashInfer, TensorRT-LLM fmha | `docs/features.md` |

### 이 저장소가 특별한 점

독자가 동료에게 옮길 만한 사실들이다. 각 편은 아래 중 최소 하나를 실제 코드와
함께 보여야 한다.

- `python/minisgl/engine/engine.py` 의 page table 주석은 이 표가 **페이지 번호가
  아니라 raw token location** 을 담는다고 밝히고, `python/minisgl/core.py` 는 이
  표를 "항상 page_size = 1 로 취급한다" 고 못 박는다. 페이지 번호로의 변환은
  어텐션 백엔드가 `python/minisgl/attention/fa.py` 에서 슬라이싱과 나눗셈으로
  한다.
- chunked prefill 은 별도 큐가 아니라 `can_decode` 가 항상 False 이고
  `append_host` 가 예외를 던지는 `Req` 서브클래스(`ChunkedReq`)로 구현된다
  (`python/minisgl/scheduler/prefill.py`).
- overlap 루프는 다음 배치의 입력 토큰을 CPU 로 가져오지 않는다. 직전 배치의
  샘플링 결과를 GPU 의 `token_pool` 에 써 두고 다음 배치가 거기서 읽는다
  (`python/minisgl/scheduler/scheduler.py`).
- CUDA graph 재생을 위해 엔진은 `table_idx` 가 `max_running_req` 인 더미 요청과
  `num_pages + 1` 번째 더미 페이지를 만들어 두고, 배치를 캡처된 크기까지 이
  요청으로 채운다(`python/minisgl/engine/engine.py`,
  `python/minisgl/engine/graph.py`).
- 모델 레이어는 `torch.nn.Module` 을 쓰지 않는다. `python/minisgl/layers/base.py`
  의 `BaseOP` 가 `self.__dict__` 를 순회하며 `_` 로 시작하는 속성을 건너뛰는
  자체 `state_dict` 를 구현한다. `LinearOProj._comm` 이 언더스코어인 이유가 이것이다.
- radix 트리의 키 비교는 파이썬이 아니라 AOT 로 컴파일된 C++ 함수
  `fast_compare_key` 가 한다(`python/minisgl/kernel/radix.py`,
  `python/minisgl/kernel/csrc/src/radix.cpp`). CPU 쪽 트리 탐색이 hot path 라는
  뜻이다.
- 배치 하나가 끝날 때의 해제는 `lazy_free_region` 컨텍스트 안에서 인스턴스의
  `_free` 메서드를 임시로 교체해 모아 두었다가 한 번의 `torch.cat` 으로
  처리한다(`python/minisgl/scheduler/cache.py`).

## 포함 범위와 제외 범위

포함 범위는 `python/` 하나다(결정 001).

| 경로 | 역할 | 포함 사유 |
| --- | --- | --- |
| `python/minisgl/scheduler/` | 배치 구성, 캐시 관리, rank 간 I/O | 분량의 중심. 2~6편이 여기서 나온다 |
| `python/minisgl/kvcache/` | KV 풀과 prefix cache 구현 | radix cache 와 naive cache 의 대조가 5편의 근거 |
| `python/minisgl/engine/` | 모델·KV·그래프 초기화와 forward | page table 과 CUDA graph 의 소유자 |
| `python/minisgl/attention/` | 어텐션 백엔드와 메타데이터 변환 | 스케줄러 장부가 커널 인자로 바뀌는 지점 |
| `python/minisgl/server/`, `python/minisgl/tokenizer/`, `python/minisgl/message/` | 프로세스 경계와 직렬화 | 1편의 대표 경로 |
| `python/minisgl/distributed/`, `python/minisgl/layers/`, `python/minisgl/models/` | TP 통신과 가중치 샤딩 | 8편의 근거 |
| `python/minisgl/kernel/` | tvm-ffi JIT·AOT 커널 바인딩 | 4·5편에서 필요한 만큼만 인용 |

제외 범위와 사유는 다음과 같다.

| 경로 | 제외 사유 |
| --- | --- |
| `tests/` | 독립 편의 주제로 삼지 않는다. 주장을 뒷받침할 때 인용 근거로만 쓴다(결정 001) |
| `benchmark/` | 이 시리즈는 측정을 수행하지 않는다(결정 002). 성능 수치의 출처로만 언급한다 |
| `pyproject.toml` | 전용 편을 두지 않는다. 의존 커널 경계를 설명할 때만 인용한다(결정 003) |
| `Dockerfile` | 전용 편을 두지 않는다. 실행 환경을 밝힐 때만 인용한다(결정 004) |
| `.gitignore`, `.pre-commit-config.yaml`, `LICENSE`, `assets/logo.png` | 구현 근거가 아니다 |

inventory 기준 분석 대상은 120개 파일이며 언어 분포는 python 101, c++ 5, cuda 5,
markdown 3, 미분류 6이다. 제품 규모는 `python/` 10,061줄(파이썬 8,090줄 +
`python/minisgl/kernel/csrc` 1,971줄)이고 `tests/` 649줄, `benchmark/` 321줄이다.
편 수와 분량은 이 규모에 맞춘다.

## 템플릿 적합성 및 실행 제약

compatibility status 는 `needs-review` 이며 finding 4개 전부에 명시적 결정이
있다. 상세 근거와 사용자 결정은 `decision-log.md` 에 기록했다.

| finding | 결정 | 기록 |
| --- | --- | --- |
| `monorepo-roots` | `python/` 만 제품, `tests/`·`benchmark/` 는 근거 전용 | 001 |
| `execution-environment` | 정적 근거 전용, runtime 근거 0 | 002 |
| `unsupported-build-system` | `pyproject.toml` 은 근거로만 인용 | 003 |
| `unclassified-unanalyzable-file` | `Dockerfile` 은 근거로만 인용 | 004 |

실행 제약은 다음과 같다. README 는 Linux(x86_64/aarch64) 전용이며 `sgl-kernel`
과 `flashinfer` 가 Linux 전용 CUDA 커널을 요구한다고 밝힌다. `Dockerfile` 은
CUDA 12.8.1 / Ubuntu 24.04 / Python 3.12 를 고정하고 기본 포트 1919 로
`python -m minisgl` 을 실행한다. `python/minisgl/engine/engine.py` 는 생성자에서
CUDA 디바이스를 잡고, `python/minisgl/kernel/store.py` 와
`python/minisgl/kernel/index.py` 는 tvm-ffi 로 `.cu` 를 런타임에 JIT 컴파일한다.
README 가 밝힌 측정 환경은 1xH200(offline) 과 NVLink 로 연결된 4xH200(online)이다.
이 시리즈를 쓰는 환경에는 NVIDIA GPU 도 CUDA 툴체인도 없다.

따라서 **정적 근거 전용**을 택한다(결정 002).

- 모든 기술 주장은 고정 revision 의 소스 인용으로 뒷받침한다.
- 모든 편의 `required_evidence.runtime` 은 0 이다. `series.yaml` 에 `execution`
  블록을 두지 않는다.
- 저장소 코드는 실행하지 않는다. `tests/core/test_cache_allocate.py` 는 CPU 에서
  도는 fixture 지만 이 시리즈는 실행하지 않고 커밋된 소스로만 인용한다.
- 처리량·지연시간·메모리 사용량은 **측정하지 않는다**. README 의 수치를 인용할
  때는 출처와 하드웨어를 같은 문장에서 밝히고 측정값으로 읽히지 않게 쓴다.
- GPU 실행과 모델 가중치 확보는 후속 근거 과제로만 남긴다.

라이선스: 저장소 `LICENSE` 는 MIT (Copyright (c) 2026 sgl-project) 이고
`pyproject.toml` 의 `license` 필드도 MIT 다. 코드 인용은 출처 경로와 줄 범위를
함께 제시한다.

## 전체 아키텍처 지도

제품은 여러 프로세스로 나뉘며 경계는 다섯이다.

| 컴포넌트 | 책임 | 근거 경로 |
| --- | --- | --- |
| API server | FastAPI 로 `/generate`, `/v1/chat/completions` 를 받고 스트리밍으로 돌려준다 | `python/minisgl/server/api_server.py`, `python/minisgl/server/launch.py` |
| Tokenizer / Detokenizer | 텍스트↔토큰 변환. 같은 `tokenize_worker` 를 역할만 바꿔 띄운다 | `python/minisgl/tokenizer/server.py`, `python/minisgl/server/launch.py` |
| Scheduler (TP rank 당 1개) | 요청 수신, 배치 구성, 캐시 관리, 결과 회신 | `python/minisgl/scheduler/scheduler.py`, `python/minisgl/scheduler/io.py` |
| Engine | 모델·KV 풀·page table·CUDA graph 를 소유하고 forward 를 수행 | `python/minisgl/engine/engine.py`, `python/minisgl/engine/graph.py` |
| 메시지 계층 | dataclass 를 `__dict__` 순회로 직렬화해 ZMQ 로 보낸다 | `python/minisgl/message/utils.py`, `python/minisgl/message/backend.py` |

제어 메시지는 ZMQ 로, TP rank 사이의 텐서는 NCCL 로 오간다. rank 0 만 tokenizer
와 직접 연결되고 나머지 rank 는 rank 0 이 PUB 으로 흘리는 원본 바이트를 받는다.
받을 개수는 gloo 그룹의 broadcast 로 미리 맞춘다(`python/minisgl/scheduler/io.py`).

`Scheduler` 안의 책임은 네 매니저로 갈린다. `TableManager` 가 요청 슬롯을,
`CacheManager` 가 페이지와 prefix cache 를, `PrefillManager` 가 대기열과 배치
구성을, `DecodeManager` 가 진행 중인 요청 집합을 맡는다. 편 분할은 디렉터리가
아니라 **요청이 이 매니저들을 지나는 순서**를 따른다(결정 007).

## 대표 실행 경로

대표 경로는 온라인 서빙 하나다(결정 006). 진입점부터 결과까지 추적한다.

1. `python -m minisgl --model "Qwen/Qwen3-0.6B"` 가 `launch_server` 를 부른다.
2. `launch_server` 가 TP 크기만큼 scheduler 프로세스와 tokenizer·detokenizer 를
   spawn 하고, 각 프로세스의 ack 를 모두 받은 뒤에야 API server 를 연다.
3. 사용자가 `/v1/chat/completions` 로 보낸 텍스트가 tokenizer 프로세스에서
   토큰 ID 열이 되어 rank 0 scheduler 로 간다.
4. rank 0 이 메시지를 다른 rank 에 뿌리고, 모든 rank 의 `PrefillManager` 가
   같은 요청을 대기열에 넣는다.
5. `_schedule_next_batch` 가 prefill 배치를 먼저 시도하고, 없으면 decode 배치를
   만든다. `_prepare_batch` 가 페이지를 할당하고 page table 에 기입한 뒤 어텐션
   메타데이터를 만든다.
6. 엔진이 forward 를 수행한다. decode 이고 배치 크기가 캡처 범위 안이면 CUDA
   graph 를 재생하고, 아니면 모델을 그대로 돈다.
7. 샘플링 결과는 GPU 의 `token_pool` 에 기록되고(다음 배치의 입력이 된다) 동시에
   CPU 로 비동기 복사된다.
8. 다음 루프가 직전 배치의 CPU 복사가 끝나기를 기다렸다가 종료 조건을 판정하고,
   rank 0 이 `DetokenizeMsg` 를 detokenizer 로 보낸다.
9. detokenizer 가 텍스트 조각을 API server 로 보내고, API server 가 스트리밍으로
   사용자에게 흘린다.

오프라인 경로(`python/minisgl/llm/llm.py` 의 `class LLM(Scheduler)`)는 같은
스케줄러에서 ZMQ I/O 만 메서드 교체로 걷어낸 것이다. 1편에서 한 문단 대조로만
언급하고 이후 편은 온라인 경로를 전제한다.

## 시리즈 전체 서사와 편별 의존성

서사는 "요청이 어디로 들어오는가 → 요청 하나는 무엇으로 표현되는가 → 어떤
배치로 묶이는가 → KV 는 어디에 적히는가 → 접두사는 어떻게 재사용되는가 →
CPU 일은 어떻게 숨는가 → 그 장부는 커널에서 무엇이 되는가 → 여러 GPU 는 어떻게
같은 결정에 도달하는가" 순서다. 학습 흐름과 코드 의존성이 같은 방향이므로
의존성은 선형 사슬이다.

```text
process-topology
  └── request-state-ledger
        └── prefill-scheduling
              └── page-table-and-allocation
                    └── radix-prefix-cache
                          └── overlap-scheduling
                                └── attention-metadata-and-cuda-graph
                                      └── tensor-parallel-ledger
```

4편은 3편이 만든 배치를 받아야 할당을 설명할 수 있고, 5편은 4편의 페이지 표현을
알아야 트리의 값이 무엇인지 말할 수 있다. 6편은 2~5편이 세운 장부 전체가 한
루프 안에서 어떻게 겹치는지 보이는 편이므로 5편 뒤에 온다. 7편은 6편의 루프가
만든 메타데이터를 소비하고, 8편은 그 전부를 rank 수만큼 복제한다.

## 편별 핵심 질문, 코드 범위와 필수 근거

`series.yaml` 과 동일한 내용이다. 근거 수는 최소값이며 `runtime` 은 결정 002 에
따라 전부 0 이다.

| # | id | 핵심 질문 | 주요 code_scopes | code / tests / runtime |
| --- | --- | --- | --- | --- |
| 1 | `process-topology` | 요청 하나는 어떤 프로세스를 지나며 경계에서 무엇이 오가는가 | `server/launch.py`, `server/api_server.py`, `tokenizer/server.py`, `message/`, `scheduler/io.py` | 5 / 1 / 0 |
| 2 | `request-state-ledger` | 요청의 진행 상태를 어떤 필드로 표현하고 누가 갱신하는가 | `core.py`, `scheduler/utils.py`, `scheduler/table.py`, `scheduler/decode.py` | 4 / 0 / 0 |
| 3 | `prefill-scheduling` | 무엇을 배치에 넣을지 어떤 예산으로 정하고 긴 프롬프트는 어떻게 쪼개는가 | `scheduler/prefill.py`, `scheduler/config.py`, `scheduler/decode.py`, `scheduler/scheduler.py` | 4 / 0 / 0 |
| 4 | `page-table-and-allocation` | 페이지로 잡으면서 page table 에는 왜 토큰 위치를 적는가 | `scheduler/cache.py`, `engine/engine.py`, `kvcache/mha_pool.py`, `kernel/store.py`, `core.py` | 5 / 1 / 0 |
| 5 | `radix-prefix-cache` | 접두사를 어떻게 재사용하고 공간이 모자라면 무엇을 버리는가 | `kvcache/radix_cache.py`, `kvcache/base.py`, `kvcache/naive_cache.py`, `kernel/radix.py`, `scheduler/cache.py` | 5 / 2 / 0 |
| 6 | `overlap-scheduling` | GPU 가 도는 동안 CPU 는 무엇을 하고 다음 입력은 어디서 오는가 | `scheduler/scheduler.py`, `engine/engine.py`, `env.py` | 5 / 0 / 0 |
| 7 | `attention-metadata-and-cuda-graph` | 장부는 어떤 텐서가 되어 커널에 들어가고 graph 재생은 무엇을 고정하는가 | `attention/base.py`, `attention/fa.py`, `attention/__init__.py`, `engine/graph.py`, `layers/attention.py` | 5 / 0 / 0 |
| 8 | `tensor-parallel-ledger` | rank 들은 어떻게 같은 결정에 도달하고 가중치는 어떻게 나뉘는가 | `distributed/`, `layers/linear.py`, `layers/embedding.py`, `layers/base.py`, `models/weight.py`, `scheduler/io.py` | 5 / 1 / 0 |

표의 경로는 `python/minisgl/` 를 생략한 축약형이다. `series.yaml` 에는 저장소
루트 기준 전체 경로가 들어 있다.

코드 인용은 **저장소 상대 경로와 줄 범위**(`path:line-range`) 로 적는다.
permalink 는 게시 단계가 고정 revision 기준으로 만들어 준다.

## 필요한 표, 다이어그램과 코드 예제

각 시각 자료는 답할 질문과 근거 경로를 가진다. 장식용 자료는 두지 않는다.

| 편 | id | 형식 | 답할 질문 | 근거 |
| --- | --- | --- | --- | --- |
| 1 | `request-process-flow` | Mermaid | 요청 하나가 프로세스 사이를 어떤 순서로 지나는가 | `server/launch.py` 의 spawn 구간, `scheduler/io.py` 의 송수신 |
| 1 | `process-roles` | 표 | 어떤 프로세스가 몇 개 뜨고 무엇을 맡는가 | `server/launch.py` 의 `mp.Process` 호출부 |
| 2 | `req-length-fields` | 표 | `cached_len`·`device_len`·`max_device_len` 은 각각 무엇을 가리키는가 | `core.py` 의 `Req` 정의와 `__post_init__` 단언 |
| 2 | `req-lifecycle` | Mermaid | 요청이 대기열에서 완료까지 어떤 상태를 지나는가 | `scheduler/prefill.py` 의 `pending_list`, `scheduler/decode.py` 의 `running_reqs` |
| 3 | `prefill-budget-checks` | 표 | 요청 하나를 배치에 넣기 전 어떤 조건을 순서대로 확인하는가 | `scheduler/prefill.py` 의 `_try_allocate_one` |
| 3 | `chunked-req-split` | Mermaid | 예산을 넘는 프롬프트가 여러 배치로 어떻게 나뉘는가 | `scheduler/prefill.py` 의 `_add_one_req` 와 `ChunkedReq` |
| 4 | `page-table-layout` | Mermaid | 요청의 논리적 위치가 KV 버퍼의 어느 줄로 이어지는가 | `scheduler/cache.py` 의 `_write_page_table`, `engine/engine.py` 의 page table 생성 |
| 4 | `kv-buffer-shape` | 표 | KV 버퍼의 각 축이 무엇을 뜻하는가 | `kvcache/mha_pool.py` 의 `_kv_buffer` 생성과 `_storage_shape` |
| 5 | `radix-tree-split` | Mermaid | 부분 일치가 일어나면 노드가 어떻게 쪼개지는가 | `kvcache/radix_cache.py` 의 `_tree_walk` 와 `split_at` |
| 5 | `lock-evict-sizes` | 표 | lock·unlock 이 evictable/protected 크기를 어떻게 바꾸는가 | `kvcache/radix_cache.py` 의 `lock_handle`, `kvcache/base.py` 의 `SizeInfo` |
| 6 | `overlap-timeline` | Mermaid | 한 루프 안에서 두 스트림의 작업이 어떻게 겹치는가 | `scheduler/scheduler.py` 의 `overlap_loop` 와 `_process_last_data` |
| 6 | `normal-vs-overlap` | 표 | `normal_loop` 과 `overlap_loop` 은 무엇이 다른가 | `scheduler/scheduler.py` 의 두 루프, `env.py` 의 스위치 |
| 7 | `metadata-conversion` | 표 | 스케줄러의 어떤 값이 커널의 어떤 인자가 되는가 | `attention/fa.py` 의 `prepare_metadata` |
| 7 | `graph-padding` | Mermaid | 캡처된 배치 크기에 맞추려고 무엇을 채워 넣는가 | `engine/graph.py` 의 `pad_batch`, `engine/engine.py` 의 더미 요청 |
| 8 | `tp-shard-map` | 표 | 가중치 이름별로 어느 축을 어떻게 자르는가 | `models/weight.py` 의 `_shard_tensor` |
| 8 | `rank-sync-protocol` | Mermaid | rank 0 과 나머지 rank 가 메시지 개수를 어떻게 맞추는가 | `scheduler/io.py` 의 `_recv_msg_multi_rank0` 와 `_recv_msg_multi_rank1` |

모든 기술 주장은 **실제 코드 줄을 코드 블록으로** 함께 싣는다. 경로와 줄 범위만
적은 링크는 검토자에게는 증명이지만 독자에게는 아무것도 보여 주지 않는다.
코드 블록은 주장에 필요한 최소 줄만 담고, 생략한 부분은 `...` 로 표시한다.

## 공통 용어집과 중복 방지 규칙

용어는 소유 편을 하나만 두고 다른 편은 그 편을 링크한다.

| 용어 | 소유 편 | 비고 |
| --- | --- | --- |
| 프로세스 경계, ZMQ 메시지 | 1 `process-topology` | 오프라인 경로 대조도 여기 한 번만 |
| `Req`, `Batch`, `Context` | 2 `request-state-ledger` | 길이 필드 세 개의 정의는 여기에만 |
| 토큰 예산, chunked prefill | 3 `prefill-scheduling` | 개념 설명은 한 문장 + Sarathi-Serve 링크 |
| 페이지, page table, `out_loc` | 4 `page-table-and-allocation` | KV 저장 구조의 유일한 소유 편 |
| prefix cache, eviction, 참조 카운트 | 5 `radix-prefix-cache` | radix attention 개념은 LMSYS 글로 링크 |
| CUDA 스트림, overlap | 6 `overlap-scheduling` | 개념 설명은 한 문장 + NanoFlow 링크 |
| 어텐션 메타데이터, CUDA graph | 7 `attention-metadata-and-cuda-graph` | 백엔드 선택 규칙도 여기서 한 번만 |
| TP rank, 샤딩, all-reduce | 8 `tensor-parallel-ledger` | `BaseOP` 의 자체 `state_dict` 도 여기서 |

`python/minisgl/scheduler/scheduler.py`, `python/minisgl/engine/engine.py`,
`python/minisgl/scheduler/cache.py` 는 여러 편이 공유한다. 같은 파일을 두 편이
다룰 때는 **줄 범위를 나누고** 서로를 링크한다. `Scheduler._prepare_batch` 는
3편이 배치 구성 결과로, 4편이 페이지 할당 호출부로, 7편이 메타데이터 생성
호출부로 나눠 다룬다. `Engine.__init__` 은 4편이 KV 풀과 page table 생성을,
7편이 graph runner 초기화를 맡는다.

## 검증 및 게시 조건

편별 조건:

- 모든 기술 주장이 고정 revision 의 `path:line-range` 인용을 가진다.
- 모든 기술 주장이 실제 코드 줄을 담은 코드 블록을 동반한다. Mermaid 가 아닌
  코드 블록이 최소 한 개 이상 있어야 한다.
- `required_evidence` 의 code·tests 최소값을 충족한다. `runtime` 은 0 이며
  측정값을 제시하지 않는다.
- 각 시각 자료가 `series.yaml` 이 선언한 형식(mermaid 또는 표)으로 실제 본문에
  존재한다.
- 전제로 둔 개념도 최소 한 문장으로 설명하고, 저장소가 이미 인용한 출처를
  링크한다.
- 용어가 소유 편에서만 정의되고 나머지 편은 그 편을 링크한다.
- 인용은 분석 대상 저장소만 가리킨다. 가이드·시리즈 정의·근거 파일은 게시되지
  않으므로 독자가 따라갈 수 없다.
- 파이프라인 내부 용어(필수 주장, 근거 수, 소유 편 같은 진행 관리 어휘)는 본문에
  등장하지 않는다.

시리즈 조건:

- 8편 전부 한국어·영어 쌍이 존재하고 구조가 대응한다.
- 전체 시리즈 검토가 PASS 여야 게시를 시작한다.
- 게시는 `Asia/Seoul` 기준 하루 한 편, 오래된 due 편부터 진행한다.
- 한국어·영어 두 URL 이 `jaehun.me` 에서 HTTP 200 과 기대 제목을 반환한 뒤에만
  알림을 보낸다.

성능 주장에 대한 명시적 제약: 이 시리즈는 처리량, 지연시간, 메모리 사용량을
측정하지 않는다. README 가 밝힌 수치를 인용할 때는 출처와 하드웨어(1xH200 또는
4xH200)를 같은 문장에서 밝히고 측정값으로 읽히지 않게 쓴다.
