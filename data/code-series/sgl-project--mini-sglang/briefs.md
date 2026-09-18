# Mini-SGLang 편별 브리프

`guide.md`, `series.yaml`, `evidence/claims.md`, `evidence/trace.md` 만을 근거로
쓴 집필 지시서다. 저장소는 실행하지 않았고 모든 편의 `required_evidence.runtime`
은 0 이므로 측정값을 제시하는 문장은 어느 편에도 두지 않는다. 인용은 저장소
상대 경로와 줄 범위로 적고, 주장마다 실제 코드 줄을 코드 블록으로 함께 싣는다.
고정 revision 은 `9a91cfafe754aa85daee49998176275667eb58f2` 다.

## Chapter 1

Scope: 온라인 서빙 요청 하나가 지나는 프로세스와 그 경계에서 오가는 메시지를
따라간다. `server/launch.py` 의 spawn 구간에서 어떤 프로세스가 몇 개 뜨는지
세고, `server/api_server.py` 가 uid 를 발급해 `TokenizeMsg` 를 보내는 지점,
`tokenizer/server.py` 가 토큰열로 바꿔 `UserMsg` 를 backend 로 넘기는 지점,
`scheduler/io.py` 가 rank 0 에서 받은 것을 나머지 rank 로 퍼뜨리는 지점을
차례로 보인다. 메시지 타입 정의는 `message/` 아래에서 인용한다. 이 편은
프로세스 경계와 ZMQ 메시지라는 용어의 소유 편이므로, 뒤 편들이 "스케줄러
프로세스"라고 말할 때 가리키는 대상을 여기서 확정한다.

Claims: `request-process-flow`, `process-roles`.

Exclusions: 스케줄러 내부에서 무엇이 일어나는지는 2편부터다. 배치 구성, KV
저장, 커널 호출은 전부 제외한다. TP rank 사이의 동기화는 여기서 "rank 0 이
받아 퍼뜨린다"는 사실까지만 말하고 규약 자체는 8편이 맡는다.

Visuals: `request-process-flow`(mermaid) 는 요청 하나가 프로세스를 지나는
순서를, `process-roles`(table) 은 어떤 프로세스가 몇 개 뜨고 무엇을 맡는지를
답한다. 두 자료 모두 `server/launch.py` 의 실제 spawn 호출부를 근거로 한다.

## Chapter 2

Scope: 요청 하나의 진행 상태가 어떤 필드로 표현되고 누가 언제 갱신하는지를
`core.py` 의 `Req` 정의에서 시작해 설명한다. `cached_len`, `device_len`,
`max_device_len` 세 길이와 거기서 파생되는 `remain_len`, `extend_len` 이 무엇을
가리키는지, `complete_one()` 과 `append_host()` 가 어느 필드를 어떻게 움직이는지
보인다. `scheduler/table.py` 의 슬롯 관리와 `scheduler/decode.py` 의 진행 중
요청 집합은 그 필드를 읽고 쓰는 쪽으로 다룬다. `Req`, `Batch`, `Context` 와 길이
필드 세 개의 정의는 이 편에만 둔다.

Claims: `req-length-fields`, `req-lifecycle`.

Exclusions: 어떤 요청을 언제 배치에 넣을지 고르는 규칙은 3편이다. 페이지 할당과
`out_loc` 은 4편이 소유하므로 여기서는 "확정된 prefix 는 커널이 읽을 수 있다"
정도로만 말하고 저장 구조는 말하지 않는다.

Visuals: `req-length-fields`(table) 는 세 길이 필드가 각각 무엇을 가리키는지를
`core.py` 의 정의와 `__post_init__` 단언을 근거로 답한다. `req-lifecycle`
(mermaid) 은 요청이 대기열에서 완료까지 지나는 상태를 `scheduler/prefill.py` 의
`pending_list` 와 `scheduler/decode.py` 의 `running_reqs` 를 근거로 그린다.

## Chapter 3

Scope: 대기 중인 요청 가운데 무엇을 이번 prefill 배치에 넣을지 정하는 예산과,
예산을 넘는 긴 프롬프트를 쪼개는 방법을 다룬다. `scheduler/prefill.py` 의
`_try_allocate_one` 이 요청 하나를 받아들이기 전에 확인하는 조건을 코드 순서
그대로 보이고, `scheduler/config.py` 의 예산 값이 어디서 오는지 밝힌다.
`_add_one_req` 와 `ChunkedReq` 로 긴 프롬프트가 여러 배치에 나뉘어 들어가는
과정을 보이되, `ChunkedReq` 가 별도 큐가 아니라 `Req` 의 서브클래스로 구현돼
`can_decode` 와 `append_host` 만 다르게 동작한다는 점을 짚는다. 토큰 예산과
chunked prefill 은 이 편이 소유하고, 개념 설명은 한 문장에 Sarathi-Serve 링크를
붙이는 선에서 끝낸다.

Claims: `prefill-budget-checks`, `chunked-req-split`.

Exclusions: 조건 확인에 쓰이는 여유 페이지 수가 어디서 나오는지는 4편이 맡는다.
접두사 재사용으로 예산이 줄어드는 경로는 5편이다. `Scheduler._prepare_batch` 는
이 편이 배치 구성 결과까지만 다루고, 할당 호출부는 4편, 메타데이터 생성
호출부는 7편이 나눠 가진다.

Visuals: `prefill-budget-checks`(table) 는 요청 하나를 배치에 넣기 전 확인하는
조건을 순서대로, `chunked-req-split`(mermaid) 은 예산을 넘는 프롬프트가 여러
배치로 나뉘는 흐름을 답한다.

## Chapter 4

Scope: KV cache 공간을 페이지 단위로 잡으면서 page table 에는 왜 페이지 번호가
아니라 토큰 위치를 적는지 답한다. `scheduler/cache.py` 의 `_write_page_table`,
`engine/engine.py` 의 page table 생성, `kvcache/mha_pool.py` 의 `_kv_buffer` 와
`_storage_shape`, `kernel/store.py` 의 저장 경로를 이어 붙여 요청의 논리적
위치가 버퍼의 어느 줄로 이어지는지 보인다. `core.py` 에서 page_size 를 항상 1 로
취급한다는 정의도 여기서 인용한다. 페이지, page table, `out_loc` 은 이 편이
유일한 소유 편이다.

Claims: `page-table-layout`, `kv-buffer-shape`.

Exclusions: 같은 접두사를 여러 요청이 나눠 쓰는 이야기는 5편이다. 이 편은 할당
자체와 그 표현까지만 다루고 재사용·회수는 다루지 않는다. `Engine.__init__` 중
graph runner 초기화 부분은 7편이 맡는다.

Visuals: `page-table-layout`(mermaid) 은 논리적 위치에서 KV 버퍼 줄까지의 연결을,
`kv-buffer-shape`(table) 은 버퍼의 각 축이 무엇을 뜻하는지를 답한다.

## Chapter 5

Scope: 여러 요청이 같은 접두사를 공유할 때 KV 를 재사용하는 자료구조와, 공간이
모자랄 때 무엇을 먼저 버리는지를 다룬다. `kvcache/radix_cache.py` 의 `_tree_walk`
와 `split_at` 으로 부분 일치가 노드를 쪼개는 과정을 보이고, `lock_handle` 과
`kvcache/base.py` 의 `SizeInfo` 로 lock·unlock 이 evictable 과 protected 크기를
어떻게 바꾸는지 보인다. 키 비교가 파이썬이 아니라 `kernel/radix.py` 를 통해
AOT 로 빌드된 C++ 함수에서 일어난다는 점, `kvcache/naive_cache.py` 라는 대조군이
같은 인터페이스를 구현한다는 점을 함께 짚는다. prefix cache, eviction, 참조
카운트는 이 편이 소유하고 radix attention 개념은 LMSYS 글로 링크한다.

Claims: `radix-tree-split`, `lock-evict-sizes`.

Exclusions: 페이지와 page table 의 정의는 4편에서 이미 정해졌으므로 다시 세우지
않고 링크한다. 배치가 언제 재시도되는지는 3편, 해제 시점이 루프의 어디에
걸리는지는 6편이 맡는다.

Visuals: `radix-tree-split`(mermaid) 은 부분 일치 시 노드 분할을,
`lock-evict-sizes`(table) 은 lock·unlock 이 크기 정보를 바꾸는 방식을 답한다.

## Chapter 6

Scope: 이번 배치를 GPU 가 도는 동안 CPU 가 무엇을 하는지, 다음 배치의 입력
토큰이 어디서 오는지를 `scheduler/scheduler.py` 의 `overlap_loop` 과
`_process_last_data` 로 보인다. `normal_loop` 과 나란히 놓고 무엇이 다른지
밝히고, `env.py` 의 스위치가 어느 쪽을 고르는지 인용한다. overlap 루프가 다음
입력을 호스트로 되가져오지 않고 GPU 의 `token_pool` 을 재사용한다는 점이 이
편의 핵심이다. CUDA 스트림과 overlap 은 이 편이 소유하고 개념 설명은 한 문장에
NanoFlow 링크를 붙인다.

Claims: `overlap-timeline`, `normal-vs-overlap`.

Exclusions: 배치에 무엇이 들어갔는지는 3편, 그 배치가 커널 인자로 바뀌는 과정은
7편이다. 여기서는 두 스트림의 시간 관계와 루프 구조까지만 다룬다.

Visuals: `overlap-timeline`(mermaid) 은 한 루프 안에서 두 스트림의 작업이 겹치는
모습을, `normal-vs-overlap`(table) 은 두 루프의 차이를 답한다.

## Chapter 7

Scope: 스케줄러가 만든 장부가 어떤 텐서로 바뀌어 어텐션 커널에 들어가는지,
CUDA graph 재생이 그 텐서를 어떻게 고정하는지 다룬다. `attention/fa.py` 의
`prepare_metadata` 에서 스케줄러의 어떤 값이 커널의 어떤 인자가 되는지 하나씩
대응시키고, `attention/base.py` 와 `attention/__init__.py` 로 백엔드 선택 규칙을
한 번만 정리한다. `engine/graph.py` 의 `pad_batch` 와 `engine/engine.py` 의 더미
요청·더미 페이지로 캡처된 배치 크기에 맞추는 과정을 보인다. 어텐션 메타데이터와
CUDA graph 는 이 편이 소유한다.

Claims: `metadata-conversion`, `graph-padding`.

Exclusions: 그 값들이 어떻게 만들어졌는지는 2~4편이 이미 설명했으므로 링크만
한다. rank 마다 이 과정이 반복된다는 사실은 8편이 맡는다.

Visuals: `metadata-conversion`(table) 은 스케줄러의 값과 커널 인자의 대응을,
`graph-padding`(mermaid) 은 캡처 크기를 맞추려고 채워 넣는 것을 답한다.

## Chapter 8

Scope: TP rank 들이 각자 스케줄러를 돌리면서 어떻게 같은 결정에 도달하는지,
가중치가 어떤 규칙으로 나뉘는지 다룬다. `scheduler/io.py` 의
`_recv_msg_multi_rank0` 와 `_recv_msg_multi_rank1` 로 rank 0 이 받은 메시지
개수를 나머지 rank 와 맞추는 규약을 보이고, `models/weight.py` 의 `_shard_tensor`
로 가중치 이름별 분할 축을 표로 정리한다. `layers/linear.py` 와
`layers/embedding.py` 의 샤딩된 레이어, `layers/base.py` 의 `BaseOP` 가
`torch.nn.Module` 대신 자체 `state_dict` 를 쓰는 이유도 여기서 한 번만 다룬다.
TP rank, 샤딩, all-reduce 는 이 편이 소유한다.

Claims: `tp-shard-map`, `rank-sync-protocol`.

Exclusions: 1편이 정한 프로세스 구성을 다시 세우지 않고 링크한다. 스케줄러가
무엇을 결정하는지는 3편, 그 결정이 텐서가 되는 과정은 7편이다.

Visuals: `tp-shard-map`(table) 은 가중치 이름별로 어느 축을 자르는지를,
`rank-sync-protocol`(mermaid) 은 rank 0 과 나머지 rank 가 메시지 개수를 맞추는
절차를 답한다.
