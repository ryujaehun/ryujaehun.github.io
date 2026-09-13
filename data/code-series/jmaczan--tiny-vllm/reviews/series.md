# tiny-vllm 시리즈 전체 리뷰

고정 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61`. 기준 문서: `guide.md`,
`series.yaml`, `briefs.md`, `evidence/claims.md`, `evidence/trace.md`, 그리고
`articles/ko-01.md`~`ko-10.md`, `articles/en-01.md`~`en-10.md`.

## Decision

**PASS**

## Coverage

- 10개 챕터가 ko/en 모두 존재하고, 번호·제목이 `series.yaml`과 정확히 일치하며, 각
  챕터의 대상 파일(scope)도 `series.yaml`과 일치한다. 시리즈 서사(실행 경로·가중치 →
  토큰→로짓 순전파 → 메모리·스케줄링 → paged attention)는 `guide.md:24-29`의 아크를
  그대로 따른다. 리더 계약(각 장이 구현을 따라가기 전에 필요한 연산을 먼저 설명)도
  모든 장에서 충족된다(safetensors 레이아웃, 임베딩/argmax, 전치 트릭·rmsNorm·RoPE,
  블록 기하, 온라인 softmax의 이유, 물리 블록 풀 등).
- Claim 소유권이 `briefs.md`와 정확히 일치한다. ch1=참조만, ch2=Claim 1, ch3=Claim
  3/4/9(일부), ch4=Claim 2/3/7, ch5=Claim 5/6/7, ch6=Claim 3/4, ch7=Claim 9/4,
  ch8=Claim 9/10/6, ch9=Claim 8, ch10=Claim 8(주)/5/6/7/10. 각 챕터의 Exclusions도
  소유 챕터로 정확히 넘긴다(예: ch4의 attention 상세→ch5, ch7의 즉시 재사용→ch8,
  ch9의 블록 테이블 순회·그리드→ch10).
- Claim 1~10이 전부 시리즈 전체에 걸쳐 커버된다. Claim 1→ch2, Claim 2→ch4,
  Claim 3→ch3/4/6, Claim 4→ch3/6/7, Claim 5→ch1(참조)/5/6/10, Claim 6→ch5/6/8/10,
  Claim 7→ch4/5/10, Claim 8→ch9/10, Claim 9→ch1(참조)/3/7/8, Claim 10→ch8/10.
- ko/en 쌍이 10개 모두 완비되어 있고, 각 쌍이 동일한 섹션 구조·주장·인용을 가진다
  (en-07은 ko-07보다 압축적이나 섹션·표·도식·인용이 모두 대응).

## Cross-chapter consistency

- 상수 일관성: `N_LAYERS=16`, `EMBEDDING_LENGTH=2048`, `HIDDEN_DIM=8192`,
  `KV_DIM=512`, `HEAD_DIM=64`, Q/KV 헤드 32/8, `GQA_Q_TO_K_RATIO=4`,
  `VOCAB_SIZE=128256`, `MAX_SEQ_LEN=2048`, `MAX_PROMPT_LEN=512`, `BATCH_SIZE=2`,
  `BLOCK_SIZE=16`, `V_OFFSET=16384`, `BLOCK_BYTES=32768`, 2GiB, `NUM_BLOCKS=65536`,
  `MAX_BLOCKS_PER_SEQ=128` — ch1/5/10의 표·본문 전부 동일.
- 프롬프트 길이 17/14/13/14와 EOT 토큰 ID 128001/128009, `MAX_SEQ_LEN-1=2047`이
  ch1/3/7/8에서 일치.
- prefill attention 순서(스코어 GEMM → causalMask → softmax → scores×V)가 ch4/5/6에서
  동일. `attn_alpha = 1/8 = 1/sqrt(64)`도 ch4(`:649`)/ch5(`:298` 주석)/ch9
  (`SQRT_HEAD_DIM=8`, `kernels.cu:12`)가 일치.
- KV 산포(prefill 블록 단위 D2D, decode 토큰 단위 D2D, 블록 경계에서만 pop)가
  ch5/6/8/10에서 동일. `block_table` 인덱스 공식(`slot*N_LAYERS*MAX_BLOCKS_PER_SEQ +
  layer*MAX_BLOCKS_PER_SEQ + block_idx`)도 동일.
- `block_table` 동기화 지점 `:876`/`:552`/`:1030`과 복사 크기 2×16×128×4=16KiB가
  ch8/10에서 일치(ch5는 존재만 언급). 소스의 TODO 인지(`src/main.cpp:551`)도 동일.
- GQA 매핑 `kv_head_idx = q_head_id/4`가 ch4/5/10에서 일치.
- paged attention 그리드 `(num_active_slots, NUM_Q_HEADS)`·블록 `HEAD_DIM=64`
  (`kernels.cu:525-527`)가 ch9/10에서 일치. `dot_products[2]`, thread 0/32 결합,
  `WARP_FULL_MASK`(`cuda_to_hip.h:50/:59`)도 ch9/10 일치.
- 커널 런치 가드 조건과 미사용 함수(`softmaxDecode`) 서술이 ch1/9에서 일치하고, 양쪽
  모두 "잠재적 경로/현재 상수 구성에서는 미발동"으로 동일하게 프레이밍된다.
- CPU argmax(매 decode 반복 D2H, bf16→float 캐스팅)가 ch3/6에서 일치.
- `project.yaml:1-2` 인용은 해당 파일(줄 1-2)과 실제로 일치하여 유효하다.
- 수치·사실 수준의 모순은 발견되지 않았다.

## Problems

1. (경계, 사소) ch1의 "커널 런치 가드" 절이 `src/kernels.cu`(`:204-209`, `:241-244`,
   `:295-298`, `:390-395`, `:444-448`)와 `src/cuda_to_hip.h`(`:50`, `:59`)의 내용을
   다룬다. ch1의 선언된 scope 파일은 `CMakeLists.txt`/`src/main.cpp`/`test.sh`이고
   `briefs.md` ch1의 참조 목록에도 없다. 내용은 소스 인용으로 정확하고 Claim 소유권과
   충돌하지 않지만("실행 맥락", "잠재적 경로"로 프레이밍됨), scope 경계를 벗어난다는
   점은 남는다. `series.yaml`/`briefs.md` ch1 scope에 kernels.cu(가드)를 추가하거나
   해당 절을 cross-scope 문맥으로 명시하면 해소된다.
2. (표기, 사소) ch5(ko/en)의 주소 공식이 `token_in_block_idx*KV_DIM*bf16`로 쓰이면서
   바로 다음 문장에서는 같은 양을 `KV_DIM * sizeof(bf16)`로 쓴다. 표기 통일이 필요하다.
3. (유지보수, 사소) ko-08은 `ko-07.md:140-141`을, en-08은 `en-07.md:94`를 같은
   교차 인용(배치 폭 상한)에 쓴다. 현재는 모두 정확하지만 언어 쌍의 줄 범위가 달라
   챕터 편집 시 조용히 깨질 수 있다.

## Required fixes

- 블로킹 항목 없음.
- 권장: (1) ch1 커널 가드 절의 scope 경계를 scope 문서에 반영하거나 명시적으로
  cross-scope 문맥으로 표기; (2) ch5의 bf16 크기 표기(`KV_DIM*bf16` →
  `KV_DIM * sizeof(bf16)`) 통일; (3) ko-08/en-08의 교차 챕터 줄 인용을 언어 쌍에서
  유지.