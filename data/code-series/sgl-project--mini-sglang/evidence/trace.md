# mini-sglang 실행 경로 추적 (9a91cfa)

> 고정 revision `9a91cfafe754aa85daee49998176275667eb58f2` 의 첨부 파일만을 근거로 한다. 실행하지 않았다.

## Entrypoint

진입점은 `python -m minisgl` 로 `launch_server` 를 호출하는 온라인 서빙 경로다.

- `python/minisgl/server/launch.py:40-44` 에서 `launch_server` 는 `parse_args(sys.argv[1:], run_shell)` 로 `ServerArgs` 를 만들고 `run_api_server(server_args, start_subprocess, run_shell=run_shell)` 를 호출한다 (`python/minisgl/server/launch.py:113`).
- `python/minisgl/server/launch.py:47-52` 에서 `start_subprocess` 는 `mp.set_start_method("spawn", force=True)` 를 강제하고 `ack_queue` 를 만든다.
- `python/minisgl/server/api_server.py:411-452` 의 `run_api_server` 는 `FrontendManager` 를 만들고(`python/minisgl/server/api_server.py:430-443`), `start_backend()` 를 호출한 뒤(`python/minisgl/server/api_server.py:446`) `run_shell` 이 아니면 `uvicorn.run(app, host=host, port=port)` 로 FastAPI 를 연다(`python/minisgl/server/api_server.py:449-450`). `run_shell` 이면 `asyncio.run(shell())` 이다(`python/minisgl/server/api_server.py:452`).
- 백엔드 기동은 `world_size = server_args.tp_info.size` 만큼 스케줄러 프로세스를 띄운다(`python/minisgl/server/launch.py:54-69`). 각 프로세스는 `_run_scheduler` 를 실행하고, `DistributedInfo(i, world_size)` 로 rank 를 부여한다(`python/minisgl/server/launch.py:59-69`).
- 같은 함수에서 detokenizer 1개(`python/minisgl/server/launch.py:72-87`)와 `num_tokenizers` 개 tokenizer(`python/minisgl/server/launch.py:88-103`)를 `tokenize_worker` 로 띄운다. 역할 구분은 `tokenizer_id == num_tokenizers` 가 detokenizer라는 점만으로 한다.
- `python/minisgl/server/launch.py:105-111` 에서 `num_tokenizers + 2` 개의 ack 를 기다린다. 주석이 명시하듯 스케줄러는 primary rank 만 ack 를 보내므로 `1 + num_tokenizers + 1` 이다.
- `_run_scheduler` 는 `torch.inference_mode()` 안에서 `Scheduler(args)` 를 만들고 `scheduler.sync_all_ranks()` 를 한 뒤 primary 만 `ack_queue.put("Scheduler is ready")` 한다(`python/minisgl/server/launch.py:16-25`). 이후 `scheduler.run_forever()` 를 돈다(`python/minisgl/server/launch.py:30-37`).

## Ordered steps

1. **API 수신 → `TokenizeMsg` 생성**
   - `POST /v1/chat/completions` 는 `python/minisgl/server/api_server.py:255-278` 에서 처리한다. `messages` 가 있으면 `model_dump()` 리스트를, 없으면 `prompt` 를 사용한다(`python/minisgl/server/api_server.py:258-262`).
   - `FrontendManager.new_user()` 로 `uid` 를 발급하고(`python/minisgl/server/api_server.py:109-114`, 호출은 `python/minisgl/server/api_server.py:265-266`), `TokenizeMsg(uid, text, sampling_params)` 를 `send_one` 으로 보낸다(`python/minisgl/server/api_server.py:266-277`).
   - `TokenizeMsg` 형상은 `python/minisgl/message/tokenizer.py:34-38` 에 `uid, text: str | List[Dict[str,str]], sampling_params` 로 정의된다.
   - `POST /generate` 도 같은 구조다(`python/minisgl/server/api_server.py:228-247`). `stream=True` 이면 `StreamingResponse(stream_chat_completions)` 를, 아니면 `wait_for_ack` 를 모아 단일 JSON 을 반환한다(`python/minisgl/server/api_server.py:280-310`).

2. **Frontend → Tokenizer (ZMQ, 비동기 PUSH/PULL)**
   - `FrontendManager.send_one` 은 최초 1회 `listen()` 태스크를 만들고 큐에 넣는다(`python/minisgl/server/api_server.py:130-132`, `python/minisgl/server/api_server.py:125-128`).
   - 큐 생성은 `run_api_server` 에서 한다: 수신은 `ZmqAsyncPullQueue(zmq_frontend_addr, decoder=BaseFrontendMsg.decoder)`, 송신은 `ZmqAsyncPushQueue(zmq_tokenizer_addr, encoder=BaseTokenizerMsg.encoder)` 이다(`python/minisgl/server/api_server.py:431-443`).
   - 직렬화는 dataclass `__dict__` 순회다. `serialize_type` 은 `__type__` + 필드 dict 를 만들고 1D `torch.Tensor` 는 `numpy().tobytes()` + dtype 으로 바꾼다(`python/minisgl/message/utils.py:20-35`). 역직렬화는 `cls_map[type_name](**kwargs)` 다(`python/minisgl/message/utils.py:52-69`).

3. **Tokenizer 프로세스 분기**
   - `tokenize_worker` 는 `backend/frontend/addr` 큐 3개를 열고(`python/minisgl/tokenizer/server.py:43-45`), `load_tokenizer` 후 `TokenizeManager/DetokenizeManager` 를 만든다(`python/minisgl/tokenizer/server.py:47-54`). 준비되면 `ack_queue.put` 한다(`python/minisgl/tokenizer/server.py:56-57`).
   - 수신은 `BatchTokenizerMsg.decoder` 로 하고(`python/minisgl/tokenizer/server.py:45`), `_unwrap_msg` 로 배치/단건을 푼다(`python/minisgl/tokenizer/server.py:24-27`, 루프는 `python/minisgl/tokenizer/server.py:60-63`).
   - `DetokenizeMsg/TokenizeMsg/AbortMsg` 로 분류하고 합이 입력 길이와 같은지 단언한다(`python/minisgl/tokenizer/server.py:67-70`).
   - `TokenizeMsg` 는 `tokenize_manager.tokenize` 로 텐서열을 만들고 `UserMsg(uid, input_ids, sampling_params)` 를 담아 `BatchBackendMsg` 로 backend 에 보낸다(`python/minisgl/tokenizer/server.py:87-101`). 단건이면 언랩한다(`python/minisgl/tokenizer/server.py:99-101`).
   - `UserMsg` 형상은 `python/minisgl/message/backend.py:32-36` 에 `uid, input_ids: CPU 1D int32 tensor, sampling_params` 로 정의된다.
   - `DetokenizeMsg` 는 `detokenize_manager.detokenize` 후 `UserReply(uid, incremental_output, finished)` 로 바꿔 frontend 에 보낸다(`python/minisgl/tokenizer/server.py:71-85`). `AbortMsg` 는 `AbortBackendMsg(uid)` 로 바꿔 backend 에 보낸다(`python/minisgl/tokenizer/server.py:102-108`).

4. **Scheduler 수신 (rank 0 / 나머지 분기)**
   - `SchedulerIOMixin.__init__` 은 `offline_mode` 면 early return 하고 온라인이면 rank 0 만 tokenizer PULL/PUSH 를 연다(`python/minisgl/scheduler/io.py:30-45`).
   - `TP size > 1` 이면 rank 0 은 `_recv_msg_multi_rank0` + PUB(`zmq_scheduler_broadcast_addr`), rank 1+ 는 `_recv_msg_multi_rank1` + SUB 로 갈라진다(`python/minisgl/scheduler/io.py:47-65`).
   - 단일 rank 는 blocking 시 `run_when_idle()` 후 1개를 받고 비어 있을 때까지 긁는다(`python/minisgl/scheduler/io.py:79-86`).
   - 멀티 rank 0 은 blocking 원시 바이트를 먼저 PUB 하고(`python/minisgl/scheduler/io.py:90-94`), 남은 원시 메시지 개수를 gloo `broadcast` 로 알린 뒤 바이트를 그대로 뿌린다(`python/minisgl/scheduler/io.py:96-107`). rank 1+ 는 blocking 1개를 SUB 에서 받고, 개수 broadcast 를 받아 그만큼 더 받는다(`python/minisgl/scheduler/io.py:109-122`). 즉 rank 0 만 역직렬화하고 나머지는 원본 바이트를 받는다.
   - `sync_all_ranks` 는 `tp_cpu_group.barrier().wait()` 다(`python/minisgl/scheduler/io.py:76-77`).

5. **메시지 → 대기열 (`_process_one_msg`)**
   - `Scheduler.overlap_loop/normal_loop` 는 `receive_msg(blocking)` 으로 받은 메시지를 `_process_one_msg` 으로 처리한다(`python/minisgl/scheduler/scheduler.py:90-96`, `python/minisgl/scheduler/scheduler.py:108-111`).
   - `BatchBackendMsg` 는 재귀로 푼다(`python/minisgl/scheduler/scheduler.py:170-172`). `ExitMsg` 는 `KeyboardInterrupt` 를 던진다(`python/minisgl/scheduler/scheduler.py:173-174`).
   - `UserMsg` 는 `input_len, max_seq_len` 을 비교해 넘치면 drop 하고(`python/minisgl/scheduler/scheduler.py:177-183`), `max_tokens` 을 `max_seq_len - input_len` 으로 하향 조정 뒤(`python/minisgl/scheduler/scheduler.py:184-188`) `prefill_manager.add_one_req(msg)` 한다(`python/minisgl/scheduler/scheduler.py:189`).
   - `add_one_req` 는 `PendingReq(uid, input_ids, sampling_params)` 로 `pending_list` 에 append 한다(`python/minisgl/scheduler/prefill.py:123-124`, 정의는 `python/minisgl/scheduler/utils.py:14-28`).
   - `AbortBackendMsg` 는 prefill→decode 순으로 찾아 `_free_req_resources` 한다(`python/minisgl/scheduler/scheduler.py:190-195`).

6. **배치 구성 (prefill 우선)**
   - `_schedule_next_batch` 는 `prefill.schedule_next_batch(prefill_budget) or decode.schedule_next_batch()` 다(`python/minisgl/scheduler/scheduler.py:219-225`). TODO 주석대로 DECODE 우선 등 다른 정책은 없다.
   - `PrefillManager.schedule_next_batch` 는 `pending_list` 가 비면 `None` 이다(`python/minisgl/scheduler/prefill.py:126-128`). `decode_manager.inflight_tokens` 을 `reserved_size` 로 넣어 `PrefillAdder` 를 만든다(`python/minisgl/scheduler/prefill.py:130-136`).
   - `PrefillAdder._try_allocate_one` 순서는 테이블 슬롯 확인 → `cache_manager.match_req` → `extend_len/estimated_len` 계산 → `available_size` 확인 → `lock` → 재확인 → `table.allocate` 다(`python/minisgl/scheduler/prefill.py:39-56`). `cached_len>0` 이면 `token_pool` 과 `page_table` 의 앞부분에 cached 구간을 복사한다(`python/minisgl/scheduler/prefill.py:57-62`).
   - `_add_one_req` 는 `chunk_size = min(token_budget, remain_len)` 으로 자르고(`python/minisgl/scheduler/prefill.py:72-74`), 남으면 `ChunkedReq`, 아니면 `Req` 를 만든다(`python/minisgl/scheduler/prefill.py:75`). `token_budget` 차감과 `reserved_size` 증가, 이번 청크 토큰의 `token_pool` 복사가 함께 일어난다(`python/minisgl/scheduler/prefill.py:76-81`). `input_ids` 는 `cached_len+chunk_size` 까지만 잘라 `Req` 에 넣는다(`python/minisgl/scheduler/prefill.py:82-90`).
   - `try_add_one` 은 예산 소진 시 `None`, `chunked_req` 재진입이면 기존 핸들/슬롯으로 이어서 넣는다(`python/minisgl/scheduler/prefill.py:92-113`).
   - 루프는 `try_add_one` 실패 시 `break` 한다(`python/minisgl/scheduler/prefill.py:139-147`). 남은 `pending_list` 는 `[chunked...] + 미처리` 로 재구성된다(`python/minisgl/scheduler/prefill.py:150`). `ChunkedReq` 는 `can_decode=False` 이고 `append_host` 가 예외이므로 decode 집합에 못 들어간다(`python/minisgl/scheduler/prefill.py:23-29`).
   - `DecodeManager.schedule_next_batch` 는 `running_reqs` 를 `uid` 순 정렬해 `Batch(phase="decode")` 로 만든다(`python/minisgl/scheduler/decode.py:32-35`). `runnable` 은 집합 비어 있음 여부다(`python/minisgl/scheduler/decode.py:37-39`).

7. **배치 준비 (`_prepare_batch`) → forward**
   - `_prepare_batch` 순서는 `graph_runner.pad_batch` → `cache_manager.allocate_paged` → `positions/input/write` 매핑 → `batch.out_loc = page_table[input_mapping]` → `attn_backend.prepare_metadata` 다(`python/minisgl/scheduler/scheduler.py:204-217`).
   - `allocate_paged` 는 `cached_len→device_len` 사이 새 페이지만 `div_ceil` 로 계산해 일괄 할당하고 `_write_page_table` 한다(`python/minisgl/scheduler/cache.py:42-53`, `python/minisgl/scheduler/cache.py:127-146`). `_page_to_token` 은 `page_size==1` 이면 그대로, 아니면 오프셋展开한다(`python/minisgl/scheduler/cache.py:119-124`).
   - `positions/input/write` 는 pin-memory 호스트 버퍼에 만들고 `non_blocking` 으로 올린다(`python/minisgl/scheduler/scheduler.py:236-267`). write 쪽은 decode 불가면 `-1` 이다(`python/minisgl/scheduler/scheduler.py:265-266`).
   - `_forward` 는 `batch.input_ids = token_pool[input_mapping]` 으로 CPU 복사 없이 GPU 풀에서 읽고(`python/minisgl/scheduler/scheduler.py:227-230`), `engine.forward_batch` 후 `token_pool[output_mapping] = next_tokens_gpu` 로 다음 배치 입력을 GPU 에 남긴다(`python/minisgl/scheduler/scheduler.py:231`). 이어서 `decode_manager.filter_reqs` 로 `can_decode` 만 남긴다(`python/minisgl/scheduler/scheduler.py:232`, 정의는 `python/minisgl/scheduler/decode.py:14-15`).
   - `Engine.forward_batch` 는 `ctx.forward_batch` 안에서 graph 가능하면 `replay`, 아니면 `model.forward` 를 하고(`python/minisgl/engine/engine.py:191-197`), `req.complete_one()` 으로 길이를 전진시킨 뒤(`python/minisgl/engine/engine.py:199-200`) 샘플링→`int32`→CPU 비동기 복사→`copy_done_event.record` 한다(`python/minisgl/engine/engine.py:202-206`).

8. **overlap 실행과 결과 처리**
   - `run_forever` 는 `ENV.DISABLE_OVERLAP_SCHEDULING` 이면 `normal_loop`, 아니면 `overlap_loop(data)` 체인이다(`python/minisgl/scheduler/scheduler.py:120-131`, 스위치는 `python/minisgl/env.py:62-69`).
   - `overlap_loop` 는 blocking 여부 결정 → 수신 → `_schedule_next_batch` → 엔진 스트림에서 `wait_stream` 후 forward → `_process_last_data(last_data)` 순이다(`python/minisgl/scheduler/scheduler.py:83-106`). 현재 배치 실행과 직전 배치 CPU 후처리가 겹친다. `normal_loop` 은 같은 배치를 바로 처리한다(`python/minisgl/scheduler/scheduler.py:108-118`).
   - `_process_last_data` 는 `copy_done.synchronize()` 로 CPU 복사를 기다린 뒤(`python/minisgl/scheduler/scheduler.py:138-143`), `lazy_free_region` 안에서 요청별로 처리한다(`python/minisgl/scheduler/scheduler.py:146-164`).
   - `ChunkedReq` 는 건너뛴다(`python/minisgl/scheduler/scheduler.py:148-149`). 일반 요청은 `append_host(next_token)` 로 호스트 길이를 늘리고(`python/minisgl/core.py:56-57` 호출), `can_decode` 와 `eos` 로 종료를 판정해 `DetokenizeMsg(uid, next_token, finished)` 를 만든다(`python/minisgl/scheduler/scheduler.py:150-156`).
   - 종료면 `decode_manager.remove_req` + `_free_req_resources(table.free + cache_req(finished=True))` 하고, prefill 비종료면 `cache_req(finished=False)` 로 prefix 를 넣는다(`python/minisgl/scheduler/scheduler.py:158-164`, 해제는 `python/minisgl/scheduler/scheduler.py:200-202`). 중복 free 방지를 위해 `finished_reqs` 로 두 번째 해제를 skip 한다(`python/minisgl/scheduler/scheduler.py:158-159`). 마지막에 `send_result(reply)` 한다(`python/minisgl/scheduler/scheduler.py:166-167`).
   - `DetokenizeMsg` 형상은 `python/minisgl/message/tokenizer.py:27-31` 이다. rank 0 만 단건/`BatchTokenizerMsg` 로 detokenizer 주소에 보내고 나머지는 no-op 이다(`python/minisgl/scheduler/io.py:124-133`).

9. **Detokenize → Frontend 스트리밍**
   - detokenizer는 `DetokenizeMsg` 묶음을 `detokenize` 해 `UserReply` 묶음으로 frontend 에 보낸다(`python/minisgl/tokenizer/server.py:71-85`).
   - `FrontendManager.listen` 은 `recv_tokenizer.get()` 을 풀어 `ack_map[uid].append` + `event.set()` 한다(`python/minisgl/server/api_server.py:116-123`, 언랩은 `python/minisgl/server/api_server.py:42-50`). 모르는 `uid` 는 버린다(`python/minisgl/server/api_server.py:120-121`).
   - `wait_for_ack` 는 event 대기→맵 비우기→yield→`finished` 면 정리 순서다(`python/minisgl/server/api_server.py:134-150`).
   - `/generate` 스트림은 `data: {incremental_output}\n` + `data: [DONE]\n` 이다(`python/minisgl/server/api_server.py:152-158`). chat 스트림은 OpenAI chunk JSON + `finish_reason: stop` + `[DONE]` 이다(`python/minisgl/server/api_server.py:160-188`). 클라이언트 단절은 `is_disconnected` 로 감지해 `abort_user(AbortMsg)` 를 예약한다(`python/minisgl/server/api_server.py:190-209`).

## State changes

- `Req`: `input_ids(CPU) / table_idx / cached_len / output_len / uid / sampling_params / cache_handle` 를 들고(`python/minisgl/core.py:28-37`), `__post_init__` 에서 `device_len=len(input_ids)`, `max_device_len=len+output_len`, `0<=cached_len<device_len<=max` 를 단언한다(`python/minisgl/core.py:38-42`). `remain=max-device`, `extend=device-cached` 다(`python/minisgl/core.py:44-50`). `complete_one` 은 `cached=device; device+=1` 이다(`python/minisgl/core.py:52-54`). `append_host` 는 `torch.cat` 이다(`python/minisgl/core.py:56-57`). `can_decode` 는 `remain>0` 이다(`python/minisgl/core.py:59-61`).
- `Batch`: `reqs/phase` + 스케줄러가 채우는 `input_ids/positions/out_loc/padded_reqs` + 백엔드가 채우는 `attn_metadata` 다(`python/minisgl/core.py:71-81`). `size/padded_size`, `is_prefill/is_decode` 로 분기한다(`python/minisgl/core.py:83-97`).
- `Context`: `page_size` 와 `page_table/attn_backend/moe_backend/kv_cache` 를 들고(`python/minisgl/core.py:100-108`), “항상 page_size=1 취급” 주석이 있다(`python/minisgl/core.py:102-103`). `forward_batch` 는 중첩 금지 컨텍스트다(`python/minisgl/core.py:115-122`).
- `PendingReq`: `uid/input_ids/sampling_params/chunked_req` 다(`python/minisgl/scheduler/utils.py:14-19`). `input_len/output_len(max_tokens)` 파생치가 예산 계산에 쓰인다(`python/minisgl/scheduler/utils.py:21-27`).
- `TableManager`: `max_running_reqs` 슬롯 풀과 GPU `page_table` 별칭, `int32` 0 초기화 `token_pool` 을 가진다(`python/minisgl/scheduler/table.py:4-11`). `allocate/pop`, `free/append` 다(`python/minisgl/scheduler/table.py:13-21`).
- `CacheManager`: `free_slots(페이지 시작점 토큰 위치)` + `prefix_cache` + `num_pages/page_size/page_table` 이다(`python/minisgl/scheduler/cache.py:15-25`). `available_size = evictable + free*page_size` 다(`python/minisgl/scheduler/cache.py:32-34`). `match_req` 는 마지막 토큰 제외하고 매칭한다(`python/minisgl/scheduler/cache.py:27-30`).
- `PrefillManager.pending_list`, `DecodeManager.running_reqs(Set[Req])` 가 대기/진행 상태를 나눈다(`python/minisgl/scheduler/prefill.py:116-124`, `python/minisgl/scheduler/decode.py:9-12`). `inflight_tokens = Σremain + (page_size-1)*N` 으로 예약 오프셋을 둔다(`python/minisgl/scheduler/decode.py:27-30`).
- `FrontendManager`: `uid_counter/ack_map/event_map` 으로 요청별 스트림 상태를 들고(`python/minisgl/server/api_server.py:99-107`), `new_user` 생성·`wait_for_ack` 소멸이 쌍을 이룬다(`python/minisgl/server/api_server.py:109-114`, `python/minisgl/server/api_server.py:134-150`).
- `Engine`: KV 풀을 `num_pages+1` 로 만들고(`python/minisgl/engine/engine.py:57-63`), page table 을 `(max_running_req+1, aligned_max_seq_len) int32 GPU 0` 으로 만들고(`python/minisgl/engine/engine.py:65-73`), `dummy_req(table_idx=max_running_req)` 의 행을 `num_tokens`(더미 페이지)로 채운다(`python/minisgl/engine/engine.py:89-98`). KV 버퍼 형상은 `(2, layers, pages, page_size, local_kv_heads, head_dim)` 이다(`python/minisgl/kvcache/mha_pool.py:28-37`).
- `Scheduler`: 엔진 스트림과 별도 `stream` 을 만들고 현재 스트림을 overlap 용으로 바꾼다(`python/minisgl/scheduler/scheduler.py:51-55`). `prefill_budget=max_extend_tokens`, `token_pool` 별칭, `finished_reqs` 집합을 둔다(`python/minisgl/scheduler/scheduler.py:68-73`). `ForwardInput(batch/sample_args/input_tuple/write_tuple)` 과 `last_data/ongoing_data` 가 루프 사이 상태를 잇는다(`python/minisgl/scheduler/scheduler.py:35-42`, `python/minisgl/scheduler/scheduler.py:98-106`).

## Branches and exits

- `prefill` 우선, 없으면 `decode`: `_schedule_next_batch` 의 `or` 체인이다(`python/minisgl/scheduler/scheduler.py:219-225`). 둘 다 없으면 `None` → forward 없이 `_process_last_data(None)` (no-op)이다(`python/minisgl/scheduler/scheduler.py:98-106`, `python/minisgl/scheduler/scheduler.py:138-140`).
- `ChunkedReq` vs `Req`: 예산 초과분은 `ChunkedReq` 로 다음 루프에 이어진다(`python/minisgl/scheduler/prefill.py:72-90`, `python/minisgl/scheduler/prefill.py:139-150`). `ChunkedReq` 는 샘플링·decode 등록·결과 회신에서 제외된다(`python/minisgl/scheduler/prefill.py:23-29`, `python/minisgl/scheduler/scheduler.py:148-149`).
- prefix 적중 vs 미적중: `match_req` 의 `cached_len` 이 `extend_len/estimated_len` 을 바꾸고, 적중분은 `token_pool/page_table` 에 미리 복사된다(`python/minisgl/scheduler/prefill.py:43-62`). radix 삽입은 `page_size` 내림 후 새 노드만 추가한다(`python/minisgl/kvcache/radix_cache.py:136-146`). naive 캐시는 항상 miss·no-op이다(`python/minisgl/kvcache/naive_cache.py:23-42`).
- 메모리 부족: `available_size` 초과면 `_try_allocate_one` 이 `None` 이고 배치가 끊긴다(`python/minisgl/scheduler/prefill.py:50-54`). 실제 페이지 할당 시 부족하면 `evict` 로 메운다(`python/minisgl/scheduler/cache.py:106-113`). 종료 요청의 꼬리는 free, 미종료는 핸들 갱신+lock 이다(`python/minisgl/scheduler/cache.py:55-79`). 배치 종료 시 `lazy_free_region` 으로 `_free` 를 모아 `torch.cat` 1회로 합친다(`python/minisgl/scheduler/cache.py:93-104`).
- CUDA graph vs eager: `can_use_cuda_graph = is_decode and size<=max_graph_bs` 다(`python/minisgl/engine/graph.py:149-150`). `pad_batch` 는 캡처 크기 중 첫 충족값으로 더미를 채우고 아니면 그대로 둔다(`python/minisgl/engine/graph.py:160-166`). `replay` 는 버퍼 복사→`prepare_for_replay`→`g.replay()` 다(`python/minisgl/engine/graph.py:152-158`).
- 어텐션 백엔드: `fa/fi/trtllm` 레지스트리와 `a,b` 하이브리드(prefill/decode 분리)가 있다(`python/minisgl/attention/__init__.py:19-68`). FA `prepare_metadata` 는 `extend/device/cached` 에서 `cu_seqlens/cache_seqlens` 를 만들고 전역 page table(page_size=1 취급)을 슬라이싱+나눗셈으로 페이지 번호 표로 바꾼다(`python/minisgl/attention/fa.py:67-105`).
- TP 단일 vs 멀티, primary vs 비primary: 위 4번과 같다(`python/minisgl/scheduler/io.py:47-65`). rank 1+ 회신은 no-op 이다(`python/minisgl/scheduler/io.py:132-133`). CPU 동기 `barrier`, 텐서 개수 `broadcast` 가 같은 결정을 강제한다(`python/minisgl/scheduler/io.py:76-77`, `python/minisgl/scheduler/io.py:100-102`).
- 스트리밍 vs 비스트리밍, shell vs 서버: chat completion 분기(`python/minisgl/server/api_server.py:280-310`), `run_shell` 분기(`python/minisgl/server/api_server.py:449-452`), shell 명령(`/exit//reset`)과 `ENV.SHELL_*` 기본값(`python/minisgl/server/api_server.py:351-409`, `python/minisgl/env.py:61-65`).
- 입력 검증·취소·종료: 과장 입력 drop·`max_tokens` 하향(`python/minisgl/scheduler/scheduler.py:177-188`), `AbortMsg→AbortBackendMsg`→prefill/decode 탐색 후 해제(`python/minisgl/scheduler/scheduler.py:190-195`, `python/minisgl/scheduler/prefill.py:153-158`, `python/minisgl/scheduler/decode.py:20-25`), `ExitMsg→KeyboardInterrupt→shutdown(cuda 동기+랭크 동기+그래프 파괴)`(`python/minisgl/scheduler/scheduler.py:133-136`, `python/minisgl/engine/engine.py:208-211`, `python/minisgl/engine/graph.py:168-171`).
- 실행 모드: `DISABLE_OVERLAP_SCHEDULING` (`python/minisgl/env.py:69`)에 따라 `normal_loop`/`overlap_loop` 가 갈린다(`python/minisgl/scheduler/scheduler.py:120-131`). idle 시 `check_integrity` (`python/minisgl/scheduler/scheduler.py:78-81`, `python/minisgl/scheduler/cache.py:81-91`).

## Unresolved gaps

- `server/args.py`(`parse_args`, `ServerArgs` 기본값: `num_tokenizer`, ZMQ 주소, `frontend_create_tokenizer_link` 등)와 `python -m minisgl` 패키지 진입점(`__main__`)은 첨부되지 않아 인용 불가. `launch.py` 가 쓰는 필드 존재만 확인된다(`python/minisgl/server/launch.py:44-113`).
- `message/frontend.py`·`message/__init__.py` 미첨부로 `BaseFrontendMsg/UserReply/BatchFrontendMsg` 정의 원문은 미확인. 사용처(`python/minisgl/server/api_server.py:15-22`, `python/minisgl/tokenizer/server.py:7-20`, `python/minisgl/server/api_server.py:42-50`)로 역할만 추적했다.
- `tokenizer/tokenize.py`·`tokenizer/detokenize.py`, `utils.py`(ZMQ 큐, `load_tokenizer`, 로거), `llm/llm.py`(오프라인 `class LLM(Scheduler)`) 미첨부로 토큰화 세부·오프라인 I/O 교체 구현은 미확인. 가이드 서술(1편 한 문단 대조)은 코드로 검증하지 못했다.
- `engine/config.py`, `engine/sample.py`, `models/*`, `moe/*`, `distributed` 통신 초기화(`engine/engine.py:112-137` 호출부 제외한 NCCL/gloo 실체), `kvcache/__init__.py` 팩토리는 미첨부라 모델·샘플러 내부는 추적하지 않았다. TP 가중치 샤딩은 `models/weight.py:34-52`·`distributed/impl.py:24-90`·`distributed/info.py:6-16` 범위만 확인했다.
- `attention/fi.py`·`trtllm.py`, `layers/*`(임베딩·리니어 분기 일부 제외), `kernel/*`(JIT/AOT·`store_cache` 호출부 제외) 미첨부/부분 첨부라 커널 인자 완전 대응은 FA 경로(`python/minisgl/attention/fa.py:48-65`)만 확인했다.
- `tests/misc/test_serialize.py:23-35` 는 직렬화 왕복 예시, `tests/core/test_cache_allocate.py:58-200` 은 eviction 페이지 정렬 fixture, `tests/kernel/test_comm.py:14-150` 은 PyNCCL all-reduce/all-gather 예시이며, 본 추적의 온라인 경로 주장 근거로는 쓰지 않았다.
- 런타임 근거 없음: GPU·CUDA·가중치 없이 정적 인용만 했다. 처리량·지연·메모리 수치는 측정하지 않았고 `pyproject.toml:24-39` 의존성·`Dockerfile` 언급은 범위 밖으로 두었다.

## Source paths

- `python/minisgl/server/launch.py:16-37,40-113,116-117`
- `python/minisgl/server/api_server.py:36-50,99-158,160-209,228-310,313-452`
- `python/minisgl/tokenizer/server.py:24-27,31-57,60-108`
- `python/minisgl/message/backend.py:12-41`
- `python/minisgl/message/tokenizer.py:11-43`
- `python/minisgl/message/utils.py:9-35,38-69`
- `python/minisgl/scheduler/io.py:27-65,76-133`
- `python/minisgl/core.py:15-25,28-68,71-97,100-136`
- `python/minisgl/scheduler/utils.py:14-33`
- `python/minisgl/scheduler/table.py:4-21`
- `python/minisgl/scheduler/decode.py:9-39`
- `python/minisgl/scheduler/prefill.py:23-29,32-63,65-113,116-162`
- `python/minisgl/scheduler/config.py:14-41`
- `python/minisgl/scheduler/scheduler.py:35-42,45-77,78-106,108-167,169-233,236-267`
- `python/minisgl/scheduler/cache.py:15-34,42-53,55-124,127-146`
- `python/minisgl/engine/engine.py:29-110,112-206,208-233`
- `python/minisgl/engine/graph.py:20-47,49-76,78-171`
- `python/minisgl/kvcache/mha_pool.py:16-37`
- `python/minisgl/kvcache/base.py:40-65,67-135`
- `python/minisgl/kvcache/radix_cache.py:17-146,148-237`
- `python/minisgl/kvcache/naive_cache.py:6-45`
- `python/minisgl/kernel/store.py:15-42`
- `python/minisgl/kernel/radix.py:13-20`
- `python/minisgl/env.py:58-87`
- `python/minisgl/attention/base.py:12-63`
- `python/minisgl/attention/fa.py:22-52,67-137,139-182`
- `python/minisgl/attention/__init__.py:19-69`
- `python/minisgl/layers/base.py:15-54`
- `python/minisgl/layers/linear.py:13-127`
- `python/minisgl/layers/embedding.py:14-42,45-110`
- `python/minisgl/layers/attention.py:18-57`
- `python/minisgl/distributed/info.py:6-36`
- `python/minisgl/distributed/impl.py:15-97`
- `python/minisgl/models/weight.py:13-60,75-124`
- `tests/misc/test_serialize.py:1-35` (근거 전용, 본 경로 미사용)
- `tests/core/test_cache_allocate.py:1-203` (근거 전용, 본 경로 미사용)
- `tests/kernel/test_comm.py:1-172` (근거 전용, 본 경로 미사용)
- `pyproject.toml:1-39` (의존 경계 인용용)
