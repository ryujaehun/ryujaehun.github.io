# Evidence Claims (sgl-project--mini-sglang, `9a91cfafe754aa85daee49998176275667eb58f2`)

> 정적 근거 전용. 저장소 코드를 실행하지 않았다. 모든 `required_evidence.runtime` 은 0 이며 측정값을 제시하지 않는다.

## Claim 1

- ID: `request-process-flow`
- Claim: 온라인 서빙 요청 하나는 FastAPI(`api_server.py`) → Tokenizer 프로세스(`tokenizer/server.py`) → rank 0 Scheduler → (TP 시) 나머지 rank 로 브로드캐스트 → Scheduler 루프 → Detokenizer → Frontend 스트리밍 순서로 지나간다. 경계는 ZMQ 이며, Frontend→Tokenizer 는 `TokenizeMsg`, Tokenizer→Scheduler 는 `UserMsg`, Scheduler→Detokenizer 는 `DetokenizeMsg`, Detokenizer→Frontend 은 `UserReply` 다.
- Confidence: High
- Evidence:
  - `python/minisgl/server/launch.py:40-44` — `launch_server` 가 `parse_args` 후 `run_api_server(server_args, start_subprocess)` 호출.
  - `python/minisgl/server/api_server.py:255-278` — `POST /v1/chat/completions` 가 `new_user()` 로 `uid` 발급 후 `TokenizeMsg(uid, text, sampling_params)` 를 `send_one` 으로 송신.
  - `python/minisgl/server/api_server.py:431-443` — Frontend→Tokenizer 큐가 `ZmqAsyncPullQueue(zmq_frontend_addr)` / `ZmqAsyncPushQueue(zmq_tokenizer_addr)` 로 결선됨.
  - `python/minisgl/tokenizer/server.py:87-101` — `TokenizeMsg` 를 `tokenize_manager.tokenize` 로 토큰열로 바꿔 `UserMsg(uid, input_ids, sampling_params)` 를 `BatchBackendMsg` 로 backend 에 송신.
  - `python/minisgl/message/tokenizer.py:27-38` — `DetokenizeMsg(uid, next_token, finished)` 와 `TokenizeMsg(uid, text, sampling_params)` 정의.
  - `python/minisgl/message/backend.py:32-36` — `UserMsg(uid, input_ids: CPU 1D int32 tensor, sampling_params)` 정의.
  - `python/minisgl/scheduler/io.py:88-122` — rank 0 은 blocking 원시 바이트를 PUB 후 gloo `broadcast` 로 개수 동기화, rank 1+ 는 개수만큼 SUB 에서 수신.
  - `python/minisgl/tokenizer/server.py:71-85` — `DetokenizeMsg` 묶음을 `detokenize` 후 `UserReply(uid, incremental_output, finished)` 로 frontend 에 송신.
  - `python/minisgl/server/api_server.py:116-123` — `FrontendManager.listen` 이 `recv_tokenizer.get()` 을 풀어 `ack_map[uid]` 에 적재하고 `event.set()`.
- Limitation: 정적 인용만이며 ZMQ 송수신을 실행하지 않았다. `message/frontend.py` 원문 미첨부로 `UserReply` 정의 원문은 사용처로만 추적했다. 런타임 근거 0.

## Claim 2

- ID: `process-roles`
- Claim: 프로세스 구성은 TP 크기만큼 Scheduler(`_run_scheduler`), Detokenizer 1개, Tokenizer `num_tokenizers` 개이며, API server 는 모든 ack(`num_tokenizers + 2` 개, 스케줄러는 primary 만 ack)를 받은 뒤에야 uvicorn 을 연다. 역할 구분은 `tokenizer_id == num_tokenizers` 가 detokenizer 라는 점만으로 한다.
- Confidence: High
- Evidence:
  - `python/minisgl/server/launch.py:54-69` — `world_size = tp_info.size` 만큼 `_run_scheduler` 프로세스 기동, `DistributedInfo(i, world_size)` 로 rank 부여.
  - `python/minisgl/server/launch.py:72-87` — detokenizer 1개 기동 (`tokenizer_id=num_tokenizers`).
  - `python/minisgl/server/launch.py:88-103` — `num_tokenizers` 개 tokenizer 기동.
  - `python/minisgl/server/launch.py:105-111` — `num_tokenizers + 2` 개 ack 대기. 주석에 primary rank 만 ack하므로 `1 + num_tokenizers + 1` 임이 명시됨.
  - `python/minisgl/server/launch.py:16-25` — `_run_scheduler` 는 `Scheduler(args)` 생성·`sync_all_ranks()` 후 primary 만 ack, 이후 `run_forever()`.
  - `python/minisgl/server/api_server.py:411-452` — `run_api_server` 가 `FrontendManager` 생성·`start_backend()` 호출 후 `uvicorn.run(app)` 또는 `asyncio.run(shell())`.
  - `python/minisgl/tokenizer/server.py:43-57` — `tokenize_worker` 가 backend/frontend/addr 큐 3개를 열고 tokenizer 로드 후 `TokenizeManager/DetokenizeManager` 생성, 준비되면 ack.
- Limitation: `server/args.py` 미첨부로 `num_tokenizer`, ZMQ 주소 기본값은 `launch.py` 사용처로만 확인했다. 실행하지 않았으므로 spawn 순서를 측정하지 않았다. 런타임 근거 0.

## Claim 3

- ID: `req-length-fields`
- Claim: 요청 진행 상태는 `Req.cached_len`(커널이 읽고 쓸 수 있는 확정 prefix 길이), `Req.device_len`(호스트 `input_ids` 길이), `Req.max_device_len`(`input_ids + output_len`) 세 길이 필드로 표현된다. `remain_len = max - device`, `extend_len = device - cached` 이며, `complete_one()` 은 `cached=device; device+=1`, `append_host()` 는 `torch.cat` 으로 호스트 길이를 늘린다.
- Confidence: High
- Evidence:
  - `python/minisgl/core.py:28-42` — `Req(input_ids, table_idx, cached_len, output_len, uid, sampling_params, cache_handle)` 정의와 `__post_init__` 의 `device_len=len(input_ids)`, `max_device_len=len+output_len`, `0<=cached_len<device_len<=max` 단언.
  - `python/minisgl/core.py:44-50` — `remain_len`, `extend_len` 파생치 정의.
  - `python/minisgl/core.py:52-57` — `complete_one()` 과 `append_host(next_token)` 구현.
  - `python/minisgl/core.py:59-61` — `can_decode` 는 `remain_len > 0`.
  - `python/minisgl/scheduler/utils.py:14-27` — `PendingReq(uid, input_ids, sampling_params, chunked_req)` 와 `input_len/output_len(max_tokens)` 파생치.
- Limitation: 길이 필드의 런타임 전이는 실행하지 않고 코드로만 인용했다. 런타임 근거 0.

## Claim 4

- ID: `req-lifecycle`
- Claim: 요청은 `PrefillManager.pending_list` 대기 → `PrefillAdder` 로 `Req/ChunkedReq` 생성 → `_prepare_batch` → forward → `DecodeManager.running_reqs(Set[Req])` 등록 → `_process_last_data` 에서 `append_host`·종료 판정·해제 또는 prefix 삽입 순서로 상태가 갱신된다. 갱신 주체는 Scheduler 루프(`_process_one_msg`, `_forward`, `_process_last_data`)와 Engine(`complete_one`)이다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/prefill.py:123-124` — `add_one_req` 가 `PendingReq` 로 `pending_list` 에 append.
  - `python/minisgl/scheduler/scheduler.py:169-189` — `_process_one_msg` 가 `UserMsg` 길이 검증·`max_tokens` 하향 후 `prefill_manager.add_one_req`.
  - `python/minisgl/scheduler/scheduler.py:219-225` — `_schedule_next_batch` 가 prefill 우선, 없으면 decode 생성.
  - `python/minisgl/scheduler/decode.py:14-15` — `filter_reqs` 가 `can_decode` 만 남김.
  - `python/minisgl/engine/engine.py:199-200` — `forward_batch` 후 `req.complete_one()` 으로 길이 전진.
  - `python/minisgl/scheduler/scheduler.py:138-167` — `_process_last_data` 가 `copy_done.synchronize()` 후 `ChunkedReq` skip, `append_host`, `can_decode`·eos 로 종료 판정, 종료면 `remove_req + _free_req_resources`, prefill 비종료면 `cache_req(finished=False)`, 마지막에 `send_result(reply)`.
  - `python/minisgl/scheduler/scheduler.py:190-195` — `AbortBackendMsg` 는 prefill→decode 순 탐색 후 `_free_req_resources`.
  - `python/minisgl/scheduler/decode.py:32-35` — decode 배치는 `running_reqs` 를 `uid` 순 정렬한 `Batch(phase="decode")`.
- Limitation: 상태 전이를 실행으로 관찰하지 않았다. 오프라인 `LLM(Scheduler)` 교체 구현은 미첨부로 1편 대조 한 문단 수준으로만 다룬다. 런타임 근거 0.

## Claim 5

- ID: `prefill-budget-checks`
- Claim: 이번 prefill 배치에 넣을지는 `PrefillAdder._try_allocate_one` 이 테이블 슬롯 → `match_req` → `extend_len/estimated_len` 계산 → `available_size` 확인 → `lock` → 재확인 → `table.allocate` 순서로 검사하며, 예산은 `token_budget=prefill_budget(max_extend_tokens)` 와 `reserved_size=decode_manager.inflight_tokens` 두 카운터로 관리된다. 검사를 통과하지 못하면 그 요청에서 배치가 끊긴다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/prefill.py:39-56` — `_try_allocate_one` 의 슬롯 확인, `match_req`, `extend_len = input_len - cached_len`, `estimated_len = extend_len + output_len`, `estimated_len + reserved_size > available_size` 면 `None`, `lock` 후 재확인, `table.allocate`.
  - `python/minisgl/scheduler/prefill.py:57-62` — `cached_len>0` 이면 적중분을 `token_pool`·`page_table` 앞부분에 미리 복사.
  - `python/minisgl/scheduler/prefill.py:92-113` — `try_add_one` 은 예산 소진 시 `None`, `chunked_req` 재진입이면 기존 핸들·슬롯으로 계속.
  - `python/minisgl/scheduler/prefill.py:126-147` — `pending_list` 비면 `None`, `reserved_size=inflight_tokens` 로 `PrefillAdder` 생성, 실패 시 `break`.
  - `python/minisgl/scheduler/config.py:14-41` — `max_extend_tokens=8192`, `max_forward_len=max_extend_tokens`.
  - `python/minisgl/scheduler/decode.py:27-30` — `inflight_tokens = Σremain + (page_size-1)*N`.
  - `python/minisgl/scheduler/cache.py:32-34` — `available_size = evictable + free*page_size`.
- Limitation: 예산 정책의 효과(처리량·지연)는 측정하지 않았다. `benchmark/` 수치는 출처 언급용으로만 사용한다. 런타임 근거 0.

## Claim 6

- ID: `chunked-req-split`
- Claim: 예산을 넘는 긴 프롬프트는 별도 큐가 아니라 `chunk_size = min(token_budget, remain_len)` 으로 잘라 남으면 `ChunkedReq`, 아니면 `Req` 로 만든다. `ChunkedReq` 는 `can_decode=False` 고정·`append_host` 예외이므로 decode 집합·샘플링·결과 회신에서 제외되고, 남은 분량은 `pending_list` 맨 앞에 `[chunked...] + 미처리` 로 다시 넣어 다음 루프에 잇는다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/prefill.py:23-29` — `ChunkedReq(Req)` 의 `append_host` 예외와 `can_decode=False`.
  - `python/minisgl/scheduler/prefill.py:65-90` — `_add_one_req` 의 `chunk_size` 절단, `token_budget` 차감·`reserved_size` 증가, 이번 청크의 `token_pool` 복사, `input_ids[:cached_len+chunk_size]` 로 `CLS` 생성.
  - `python/minisgl/scheduler/prefill.py:139-150` — 루프에서 `ChunkedReq` 는 `pending_req.chunked_req` 로 보관해 `chunked_list` 로 모으고 `pending_list = chunked_list + pending_list[len(reqs):]` 로 재구성.
  - `python/minisgl/scheduler/scheduler.py:148-149` — `_process_last_data` 가 `ChunkedReq` 를 건너뜀.
  - `python/minisgl/scheduler/scheduler.py:232` — `_forward` 후 `filter_reqs` 가 `can_decode` 만 남기므로 `ChunkedReq` 는 decode 집합에 못 들어감 (`python/minisgl/scheduler/decode.py:14-15`).
- Limitation: 분할 실행을 재현하지 않았다. 개념 설명은 한 문장 + Sarathi-Serve(arXiv 2403.02310, `docs/features.md` 인용) 링크로 처리한다. 런타임 근거 0.

## Claim 7

- ID: `page-table-layout`
- Claim: 페이지는 `div_ceil(cached_len/device_len, page_size)` 로 새로 필요한 구간만 일괄 할당하고, page table 에는 페이지 번호가 아니라 raw token location(토큰 위치)을 적는다. 전역 표는 항상 `page_size=1` 로 취급되며, 페이지 번호로의 변환은 어텐션 백엔드가 슬라이싱(`:max_seqlen_k:page_size`)과 나눗셈(`div_(page_size)`)으로 한다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/cache.py:42-53` — `allocate_paged` 가 `first_page=div_ceil(cached_len)`, `last_page=div_ceil(device_len)` 사이 새 페이지만 모아 `_allocate` 후 `_write_page_table`.
  - `python/minisgl/scheduler/cache.py:127-146` — `_write_page_table` 이 `(table_idx, first_pos:last_pos)` 위치에 할당된 raw location 을 기입.
  - `python/minisgl/engine/engine.py:65-73` — page table 을 `(max_running_req+1, aligned_max_seq_len) int32 GPU 0` 으로 생성, 주석에 raw locations 저장·128바이트 정렬 명시.
  - `python/minisgl/core.py:100-108` — `Context` 의 `page_table` 에 항상 `page_size=1` 취급 주석 (`python/minisgl/core.py:102-103`).
  - `python/minisgl/scheduler/scheduler.py:204-217` — `_prepare_batch` 순서 `pad_batch → allocate_paged → positions/input/write 매핑 → out_loc = page_table[input_mapping] → prepare_metadata`.
  - `python/minisgl/attention/fa.py:92-97` — 전역 표를 `page_table[table_idx, :max_seqlen_k:page_size]` 로 슬라이싱 후 `page_size>1` 이면 `div_(page_size, floor)`.
  - `python/minisgl/scheduler/cache.py:119-124` — `_page_to_token` 은 `page_size==1` 이면 그대로, 아니면 오프셋 전개.
- Limitation: GPU page table 기입을 실행하지 않았다. `tests/core/test_cache_allocate.py:58-200` 은 근거 전용 인용이며 실행하지 않았다. 런타임 근거 0.

## Claim 8

- ID: `kv-buffer-shape`
- Claim: KV 버퍼 실체는 `(2, layers, pages, page_size, local_kv_heads, head_dim)` 형상의 단일 `_kv_buffer` 이며, 커널에 줄 때는 레이어별 `k/v` 를 `(num_pages*page_size, local_kv_heads, head_dim)` 으로 view 해서 `out_loc` 인덱스와 함께 `store_cache` 로 기록한다. 페이지 수는 모델 메모리 차감 후 남은 메모리를 페이지당 바이트로 나눠 정하고 `+1` 더미 페이지를 둔다.
- Confidence: High
- Evidence:
  - `python/minisgl/kvcache/mha_pool.py:28-37` — `_kv_buffer` 생성 형상과 `_storage_shape = (num_pages*page_size, local_kv_heads, head_dim)`.
  - `python/minisgl/kvcache/mha_pool.py:45-56` — `store_kv` 가 view 한 버퍼와 `out_loc` 을 `store_cache` 에 전달.
  - `python/minisgl/kernel/store.py:30-42` — `store_cache` 가 `view(num_tokens, -1)` 후 `element_size` 로 JIT 모듈을 골라 `launch`.
  - `python/minisgl/engine/engine.py:55-63` — KV 풀을 `num_pages + 1` 로 생성 (`+1` 더미 페이지).
  - `python/minisgl/engine/engine.py:148-168` — 페이지당 바이트(`2*head_dim*local_kv_heads*page_size*itemsize*layers`)로 `num_pages` 결정.
- Limitation: `store.cu` JIT 컴파일과 GPU 기록을 실행하지 않았다. 메모리 수치는 측정하지 않았다. 런타임 근거 0.

## Claim 9

- ID: `radix-tree-split`
- Claim: 접두사 재사용 자료구조는 radix tree 이며, 탐색은 `_tree_walk` 으로 자식을 키 함수로 타고 `fast_compare_key`(AOT C++)로 첫 불일치를 찾아 `page_size` 내림한 뒤, 부분 일치이면 `split_at(pos)` 으로 노드를 쪼갠다. 삽입은 `page_size` 내림 후 새 노드만 추가하고, 대조군 naive 캐시는 항상 miss·no-op 이다.
- Confidence: High
- Evidence:
  - `python/minisgl/kvcache/radix_cache.py:205-231` — `_tree_walk` 의 자식 조회, `get_match_len`, `align_down(match_len, page_size)`, 불완전 일치 시 `split_at(match_len)` 후 반환, 접근 시각 갱신.
  - `python/minisgl/kvcache/radix_cache.py:69-81` — `split_at` 이 부모에 앞부분 새 노드, 자신은 뒷부분으로 재설정하고 `ref_count` 승계.
  - `python/minisgl/kvcache/radix_cache.py:63-67` — `get_match_len` 이 `fast_compare_key(self._key, input_ids)` 호출.
  - `python/minisgl/kernel/radix.py:18-20` — `fast_compare_key(x, y)` 는 1-D int CPU 텐서 비교.
  - `python/minisgl/kvcache/radix_cache.py:136-146` — `insert_prefix` 가 `align_down(len, page_size)` 후 미적중 꼬리만 새 노드로 추가, `evictable_size` 증가.
  - `python/minisgl/kvcache/naive_cache.py:23-42` — `match` 항상 miss, `insert` no-op, `evict` 시 예외, `size_info` 0.
  - `python/minisgl/kvcache/radix_cache.py:234-237` — 키 함수는 `page_size==1` 이면 첫 토큰, 아니면 첫 페이지 튜플.
- Limitation: C++ `radix.cpp` 본문은 최소 인용 범위로만 다루며 파이썬 장부 해석으로 단정하지 않는다. 개념 설명은 한 문장 + LMSYS blog 2024-01-17(`docs/features.md` 인용) 링크로 처리한다. 런타임 근거 0.

## Claim 10

- ID: `lock-evict-sizes`
- Claim: 캐시 크기는 `SizeInfo(evictable_size, protected_size)` 로 관리되며, `lock` 은 `ref_count==0` 노드를 evictable→protected 로, `unlock` 은 `ref_count==0` 이 되면 protected→evictable 로 옮긴다. 공간이 모자라면 `_allocate` 가 `evict` 로 메우고, `evict` 는 ref 0 리프를 timestamp 힙 순서(LRU 근사)로 버린다. 종료 요청 꼬리는 free, 미종료는 핸들 갱신+lock 이다.
- Confidence: High
- Evidence:
  - `python/minisgl/kvcache/base.py:48-54` — `SizeInfo` 와 `total_size` 정의.
  - `python/minisgl/kvcache/radix_cache.py:113-130` — `lock_handle` 의 lock/unlock 분기와 `evictable_size/protected_size` 이동, 루트까지 `ref_count` 증감.
  - `python/minisgl/kvcache/radix_cache.py:148-175` — `evict(size)` 의 0 조기반환, 리프 수집·heapify, `ref_count==0 and is_leaf` 단언 후 `value` 회수·`evictable_size` 차감.
  - `python/minisgl/scheduler/cache.py:106-113` — `_allocate` 가 부족분을 `evict((needed-free)*page_size)` 로 메운 뒤 `evicted[::page_size]` 로 free refill.
  - `python/minisgl/scheduler/cache.py:55-79` — `cache_req` 의 `insert_prefix`·`unlock(old)`·`_free(page_indices[old_cached:cached])`, finished 면 꼬리 free, 아니면 핸들 갱신+lock.
  - `python/minisgl/scheduler/cache.py:93-104` — `lazy_free_region` 이 `_free` 를 모아 `torch.cat` 1회로 합침.
  - `python/minisgl/scheduler/cache.py:27-30` — `match_req` 는 마지막 토큰 제외하고 매칭.
- Limitation: eviction 순서를 실행으로 검증하지 않았다. `tests/core/test_cache_allocate.py:58-200` 은 페이지 정렬 fixture 근거 전용이며 실행하지 않았다. 런타임 근거 0.

## Claim 11

- ID: `overlap-timeline`
- Claim: overlap 루프는 이번 배치 forward 를 엔진 스트림에서 돌리는 동안 직전 배치의 CPU 후처리(`_process_last_data`: `copy_done.synchronize()` → `append_host`·종료 판정·`send_result`)를 겹친다. 다음 배치 입력 토큰을 CPU 로 가져오지 않고, 직전 샘플링 결과를 GPU `token_pool` 에 써 두고 `_forward` 시작 시 `token_pool[input_mapping]` 으로 읽는다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/scheduler.py:83-106` — `overlap_loop` 순서: blocking 결정 → 수신 → `_schedule_next_batch` → 엔진 스트림에서 `wait_stream` 후 forward → `_process_last_data(last_data)`.
  - `python/minisgl/scheduler/scheduler.py:227-233` — `_forward` 가 `batch.input_ids = token_pool[input_mapping]` 으로 GPU 풀에서 읽고, `token_pool[output_mapping] = next_tokens_gpu` 로 다음 입력 보관, `filter_reqs` 로 decode 등록.
  - `python/minisgl/engine/engine.py:191-206` — `forward_batch` 가 graph 가능하면 replay 아니면 `model.forward`, `complete_one`, 샘플링→`int32`→CPU 비동기 복사→`copy_done_event.record`.
  - `python/minisgl/scheduler/scheduler.py:138-143` — `_process_last_data` 가 `copy_done.synchronize()` 로 CPU 복사 완료 대기.
  - `python/minisgl/scheduler/prefill.py:57-62` — 적중분 `token_pool` 미리 복사도 같은 풀 재사용 예시.
- Limitation: 두 CUDA 스트림 겹침을 실행·계측하지 않았다. 개념 설명은 한 문장 + NanoFlow(arXiv 2408.12757, `docs/features.md` 인용) 링크로 처리한다. 런타임 근거 0.

## Claim 12

- ID: `normal-vs-overlap`
- Claim: `normal_loop` 은 같은 배치를 바로 처리(`_process_last_data(ongoing_data)`)하고, `overlap_loop(data)` 는 직전 배치 결과와 현재 배치 실행을 1단 파이프라인으로 겹친다. 선택 스위치는 `ENV.DISABLE_OVERLAP_SCHEDULING` 이며, idle 시 `run_when_idle()` 이 `check_integrity()` 를 수행한다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/scheduler.py:108-118` — `normal_loop` 의 수신→스케줄→forward→즉시 후처리.
  - `python/minisgl/scheduler/scheduler.py:120-131` — `run_forever` 가 `DISABLE_OVERLAP_SCHEDULING` 이면 `normal_loop`, 아니면 `overlap_loop(data)` 체인. overlap 진입 전 `current_stream == self.stream` 단언.
  - `python/minisgl/scheduler/scheduler.py:51-55` — Scheduler 가 엔진 스트림과 별도 `stream` 을 만들고 현재 스트림을 overlap 용으로 전환.
  - `python/minisgl/env.py:62-69` — `DISABLE_OVERLAP_SCHEDULING = EnvBool(False)` 스위치 정의.
  - `python/minisgl/scheduler/scheduler.py:78-81` — `run_when_idle` 이 idle 로그 후 `cache_manager.check_integrity()`.
  - `python/minisgl/scheduler/cache.py:81-91` — `check_integrity` 가 `free + cache_pages == num_pages` 검증.
- Limitation: 두 루프의 성능 차이는 측정하지 않았다. 런타임 근거 0.

## Claim 13

- ID: `metadata-conversion`
- Claim: 스케줄러 장부는 FA 백엔드 `prepare_metadata` 에서 `extend_len/device_len/cached_len` → `cu_seqlens_k/q, cache_seqlens, max_seqlen_k/q, page_table(페이지 번호 표)` 텐서로 바뀐다. `max_seqlen_q==1` 이면 decode 고정식, 캐시 미적중 prefill 이면 `cu_q = cu_k`, 부분 적중이면 별도 cumsum 이며, 전역 표 슬라이싱+나눗셈이 페이지 번호 변환을 담당한다. prefill/decode 분리 하이브리드(`fa,fi` 등)는 `HybridBackend` 가 배치 phase 로 위임한다.
- Confidence: High
- Evidence:
  - `python/minisgl/attention/fa.py:67-105` — `prepare_metadata`: `seqlens_q=extend`, `seqlens_k=device`, `cached`, CPU pin-memory 생성 후 GPU `non_blocking` 전송, `max_seqlen_q==1` 분기, 미적중 시 `cu_q=cu_k`, 부분 적중 시 별도 cumsum, 전역 표 슬라이싱+`div_(page_size)`, `FAMetadata` 생성.
  - `python/minisgl/attention/base.py:37-63` — `HybridBackend` 가 `batch.is_prefill` 로 prefill/decode 백엔드에 위임 (`forward`, `prepare_metadata`, capture/replay).
  - `python/minisgl/attention/__init__.py:52-68` — `a,b` 하이브리드 생성 규칙 (콤마 1개만 허용, 다르면 Hybrid, 같으면 단일).
  - `python/minisgl/layers/attention.py:47-57` — 레이어가 `qkv split → RoPE(positions) → attn_backend.forward(q,k,v,layer_id,batch)` 호출.
  - `python/minisgl/scheduler/scheduler.py:236-267` — `_make_positions/_make_input_tuple/_make_write_tuple` 의 pin-memory 호스트 버퍼 생성과 `non_blocking` 업로드, write 불가면 `-1`.
- Limitation: `fi.py/trtllm.py` 커널 인자 완전 대응은 추적하지 않고 FA 경로만 확인했다. 커널 실행은 하지 않았다. 런타임 근거 0.

## Claim 14

- ID: `graph-padding`
- Claim: CUDA graph 재생을 위해 엔진은 `table_idx == max_running_req` 인 더미 요청과 `num_pages` 값(더미 페이지 가리킴)으로 채운 더미 행을 만들어 두고, `pad_batch` 가 캡처된 크기 중 첫 충족값까지 이 더미로 채운다. `replay` 는 버퍼 복사→`prepare_for_replay`→`g.replay()` 이며, `can_use` 조건은 `is_decode and size<=max_graph_bs` 다.
- Confidence: High
- Evidence:
  - `python/minisgl/engine/engine.py:88-98` — `dummy_req(table_idx=max_running_req)` 생성과 `page_table[dummy_row].fill_(num_tokens)` (더미 페이지 지시).
  - `python/minisgl/engine/engine.py:99-110` — `GraphRunner` 초기화 인자에 `dummy_req`, `cuda_graph_bs`, `max_seq_len`, `vocab_size` 전달.
  - `python/minisgl/engine/graph.py:105-144` — 캡처가 더미 배치(`[dummy_req]*bs`, `padded_reqs=reqs`)로 `prepare_for_capture`→버퍼 결선→graph 캡처, 풀 재사용.
  - `python/minisgl/engine/graph.py:149-158` — `can_use_cuda_graph` 와 `replay` 의 버퍼 복사·`prepare_for_replay`·`g.replay()`.
  - `python/minisgl/engine/graph.py:160-166` — `pad_batch` 가 첫 충족 캡처 크기로 더미 패딩, 불가면 그대로.
  - `python/minisgl/engine/graph.py:21-47` — `GraphCaptureBuffer(input_ids, out_loc, positions, logits)` 와 `set_batch/copy_from` 슬라이싱.
- Limitation: CUDA graph 캡처·재생을 실행하지 않았다. 캡처 크기 목록 결정(`_determine_cuda_graph_bs`)의 메모리 효과는 측정하지 않았다. 런타임 근거 0.

## Claim 15

- ID: `tp-shard-map`
- Claim: 가중치는 이름 규칙으로 나뉜다: `q/k/v/gate/up_proj` 는 dim 0 분할(단 `k/v` 는 `num_kv_heads < n` 이면 복제 허용 head 단위 선택), `o/down_proj` 는 dim 1 분할, `lm_head/embed_tokens` 는 vocab dim ceil 분할, 그 외는 복제다. QKV·gate-up 은 샤딩 후 fused(`qkv_proj`, `gate_up_proj`)로 합치고, expert 키는 packed 키로 stack 한다. 레이어는 `torch.nn.Module` 이 아니라 `BaseOP` 자체 `state_dict`(`_` 시작 속성 skip)를 쓰므로 `LinearOProj._comm` 같은 통신자는 저장 대상이 아니다.
- Confidence: High
- Evidence:
  - `python/minisgl/models/weight.py:34-52` — `_shard_tensor` 의 `_SPLIT_DIM_0/_SPLIT_DIM_1/vocab` 분기와 kv 복제 분기.
  - `python/minisgl/models/weight.py:55-60` — `_get_merge_info` 의 fused 매핑.
  - `python/minisgl/models/weight.py:75-121` — `load_weight` 의 샤딩→merge 버퍼→`torch.cat`→expert stack→yield 순서.
  - `python/minisgl/layers/linear.py:71-88` — `LinearQKVMerged` 의 `local_num_qo/kv` 와 `local_osize` 계산.
  - `python/minisgl/layers/linear.py:91-106` — `LinearOProj` 의 입력 dim 분할 후 `all_reduce`.
  - `python/minisgl/layers/linear.py:109-127` — `LinearRowParallel` 의 입력 분할 후 `all_reduce`.
  - `python/minisgl/layers/embedding.py:14-42` — `VocabParallelEmbedding` 의 `div_ceil` vocab 분할과 `vocab_range`, `all_reduce`.
  - `python/minisgl/layers/embedding.py:45-110` — `ParallelLMHead` 의 tied 분기, prefill `get_last_indices` 절단, `all_gather` 후 vocab 절단.
  - `python/minisgl/layers/base.py:19-54` — `BaseOP.state_dict/load_state_dict` 의 `__dict__` 순회와 `_` skip.
- Limitation: 가중치 로딩·NCCL 통신을 실행하지 않았다. MoE·분산 초기화 내부는 `engine.py:112-137` 호출부 외 실체를 추적하지 않았다. 런타임 근거 0.

## Claim 16

- ID: `rank-sync-protocol`
- Claim: rank 들은 같은 결정을 CPU 동기화로 강제한다: 기동 시 `tp_cpu_group.barrier()`, 메시지 개수 `broadcast`, 텐서는 NCCL(PyNCCL 우선, 폴백 torch) `all_reduce/all_gather` 로 합친다. rank 0 만 tokenizer 와 직결(PULL/PUSH)되고 나머지는 rank 0 이 PUB 한 원본 바이트를 받아 역직렬화 없이 공유하며, 회신도 rank 0 만 하고 나머지는 no-op 이다.
- Confidence: High
- Evidence:
  - `python/minisgl/scheduler/io.py:30-45` — `offline_mode` early return, 온라인 rank 0 만 tokenizer PULL/PUSH 개설.
  - `python/minisgl/scheduler/io.py:47-65` — `size>1` 이면 rank 0 은 PUB+`_recv_msg_multi_rank0`, rank 1+ 는 SUB+`_recv_msg_multi_rank1`, 회신 함수도 분리.
  - `python/minisgl/scheduler/io.py:76-77` — `sync_all_ranks` 는 `tp_cpu_group.barrier().wait()`.
  - `python/minisgl/scheduler/io.py:100-102` — 개수 `broadcast(src_tensor, root=0)`.
  - `python/minisgl/scheduler/io.py:109-122` — rank 1+ 가 개수 broadcast 를 받아 그만큼 SUB 수신.
  - `python/minisgl/scheduler/io.py:124-133` — rank 0 만 단건/`BatchTokenizerMsg` 회신, rank 1+ 는 no-op.
  - `python/minisgl/distributed/info.py:6-16` — `DistributedInfo(rank, size)` 와 `is_primary() == rank==0`.
  - `python/minisgl/distributed/impl.py:24-42` — torch `all_reduce/all_gather`, 단일 rank 조기반환.
  - `python/minisgl/distributed/impl.py:73-90` — `enable_pynccl_distributed` 가 플러그인 append 로 PyNCCL 우선화.
  - `python/minisgl/engine/engine.py:112-137` — `_init_communication` 의 gloo/nccl 그룹 분기와 `enable_pynccl_distributed` 호출.
  - `tests/kernel/test_comm.py:137-150` — PyNCCL `all_reduce/all_gather` 예시 (근거 전용, 본 경로 미사용, 실행하지 않음).
- Limitation: 다중 rank 실행을 하지 않았으므로 동기화·집합 통신을 측정하지 않았다. `tests/kernel/test_comm.py:14-150` 은 근거 전용 인용이며 실행하지 않았다. 런타임 근거 0.
