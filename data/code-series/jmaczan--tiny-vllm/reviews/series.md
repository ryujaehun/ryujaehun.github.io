# 시리즈 전체 리뷰 — tiny-vLLM 7편

고정 revision `e25bf1994efa90bc98b721ba7c527402f86fbeaf` 의 체크아웃 트리와 첨부
증거(`guide.md`, `series.yaml`, `briefs.md`, `evidence/claims.md`,
`evidence/trace.md`, `evidence/recon.md`, `decision-log.md`,
`compatibility-report.md`)를 기준으로 `articles/` 의 한국어·영어 14개 파일 전체와
`reviews/` 의 편별 리뷰 14건을 대조했다. 편별 리뷰가 소스 대비 사실 정확성을, 이
리뷰는 편 간 일관성·범위·용어 소유·한영 대응·인용 정합을 담당한다.

## Decision

**PASS**

7편 전부에 한국어·영어 쌍이 존재하고, 순서·의존성·근거 최소값·시각 자료·용어
소유·한계·성능 서술이 `series.yaml` 과 `guide.md` 에 정확히 정렬된다. 편 간 기술적
모순은 찾지 못했다. 21개 claim 전부가 소유 편을 가지며, 필수 근거
(`code` 최소값, `tests`, `runtime 0`)와 10개 시각 자료가 양 언어로 충족된다. 측정값으로
읽힐 성능 수치도 전무하다. 수정이 필요한 편 간 모순은 없고, 아래 Required fixes 의
2건(한글 오타, 영문 용어 통일)은 게시 전 교정만 하면 되는 편집 사항이다. 편별
리뷰에서 지적된 선택 관찰은 모두 비차단이다.

## Coverage

- **편 구성·순서·의존성.** `series.yaml:4-152` 의 7장(`build-and-entry`,
  `weights-and-memory`, `prefill-path`, `cublas-and-layout`, `decode-path`,
  `paged-kv-cache`, `batching-and-slots`)이 `guide.md:134-146` 의 의존 트리
  (build→weights→prefill→{cublas,decode}→paged→batching, cublas·decode 는 서로
  독립)와 일치한다. 모든 편의 `depends_on` 이 준수되고, 편이 스스로 밝히는 의존
  (4편→3편 `guide.md:144-146`, 5편→3편 `guide.md:146`, 6편→5편, 7편→6편)도
  동일하다.
- **한영 대응.** ko-01..en-07 까지 14개 파일이 모두 존재하고, en 리뷰 7건이 각각
  "완전·충실한 번역"으로 PASS 했다. 구조(절 순서·개수)와 인용 토큰이 ko/en 쌍에서
  동일하고, 시각 자료 마커 10개(`product-translation-units`,
  `dual-backend-branch`, `buffer-allocation-table`, `prefill-kernel-sequence`,
  `rmsnorm-reduction`, `column-row-major-trick`, `prefill-decode-contrast`,
  `block-table-indexing`, `paged-attention-inputs`, `slot-lifecycle`)가 양 언어로
  존재한다. `series.yaml` 의 kind(table 4, mermaid 6)와 마커의 `supports:` claim
  ID 가 실제 표·다이어그램과 일치한다.
- **근거 최소값.** 각 편이 스스로 밝힌 근거 수가 `series.yaml` 과 일치한다: 1편
  code 4 / tests 0, 2편 code 3 / tests 0, 3편 code 6 / tests 1, 4편 code 3 /
  tests 0, 5편 code 4 / tests 1, 6편 code 5 / tests 0, 7편 code 4 / tests 0.
  모든 편 `runtime 0`(`guide.md:85`). `tests` 1건(3편)은 커밋된
  `tests/test_softmax.cu` 와 참조 출력(`reference.txt`,
  `python/rms_norm_crosscheck.txt`) 인용으로, 5편의 `tests` 1건은 `code_scope` 의
  `python/decode_test.py` 를 claim 없이 범위로 두는 `briefs.md:150` 방침으로
  충족된다. 커널 실행 시간·처리량·메모리 사용량 수치가 어느 편에도 없어 결정
  003(`guide.md:75-87`)을 지킨다.
- **claim 소유권.** `claims.md` 의 21개 claim 이 전부 한 편에 귀속된다: Claim
  1-2(1편), 3-5(2편), 6-8(3편), 9(4편), 10-12(5편), 14-18(6편), 13·19-20(7편),
  21(시리즈 공통). 6편의 `block-table-indexing` 다이어그램이 `kv-block-layout`
  (Claim 15)까지 함께 근거하고, 7편의 `slot-lifecycle` 이 `decode-termination`
  (Claim 13)까지 함께 근거하는 것은 시각 자료가 필요한 claim 을 초과해 답하는
  것으로 결함이 아니다.
- **코드 커버리지.** 커널 11개(`guide.md:107-110`, 결정 008)가 1편(분류)·3편
  (prefill 7개)·5편(decode 4개)·6편(`pagedAttentionKernel`)에 걸쳐 전부 다뤄진다.
  호스트 함수 4개(`checkGPUStatus`, `loadWeights`, `prefill`, `main`)도 1·2·3편이
  나눠 소유한다.

## Cross-chapter consistency

- **상수·수치 일관성.** 편 간 공유 상수가 전부 일치한다: `MAX_SEQ_LEN=2048`,
  `BATCH_SIZE=2`, `MAX_PROMPT_LEN=512`, `BLOCK_SIZE=16`, `V_OFFSET=16384`,
  `BLOCK_BYTES=32768`, `NUM_BLOCKS=65536`, block_table 4096 항목(2×16×128),
  `KV_CACHE_SIZE_BYTES=2GB`, `HEAD_DIM=64`, `KV_DIM=512`, `EMBEDDING_LENGTH=2048`,
  `HIDDEN_DIM=8192`, `VOCAB_SIZE=128256`, GQA 비율 4(Q 32·K/V 8),
  `attn_alpha=1/8=1/sqrt(64)`, 종료 토큰 128001·128009, 프롬프트 길이
  17/14/13/14. 어느 편도 다른 편의 수치와 어긋나지 않는다.
- **공유 파일의 줄 범위 분할(`guide.md:202-204`).** `prefill` 함수는 3편이 커널
  순서(`src/main.cpp:164-493`), 4편이 cuBLAS 호출부(`178-196` 등)를 맡고
  상호 링크한다. `main` 초기화 구간은 1·2·7편이 목적별로 나눠 쓰고, 같은
  인용(`555-1044`, `695-708`, `720-737`, `878`, `1015-1037`)이 모든 편에서 동일한
  의미로 쓰인다. decode 의 K/V 분산(`851-873`)은 5편(배치·분산 절차)과
  6편(블록 인덱스·레이아웃)이 줄 범위를 나눠 소유한다.
- **용어 소유(`guide.md:189-204`).** 번역 단위·이중 백엔드(1편), bfloat16
  버퍼·슬롯(2편), prefill·병렬 리덕션(3편), 열 우선 레이아웃(4편), decode
  스텝(5편), block table·paged attention(6편), continuous batching 의 슬롯
  구현(7편)이 각각 소유 편에서만 정의되고 나머지는 링크한다. 6편의 워프 셔플
  리덕션이 prefill 의 트리 리덕션과 "다르다"고만 짚고 재설명하지 않는 것도 규약을
  따른다. `WARP_FULL_MASK` 소비는 1편이 정의 지점까지만 다루고 6편으로 넘긴다.
- **미지원 claim 의 일관된 기각.** `series.yaml` 의 `column-row-major-trick`
  필수 주장 "alpha·beta 가 잔차 누적을 처리"는 4편이 명시적으로 기각하고
  (beta 전부 0.0, 잔차는 `residualAdd` 커널), 다이어그램에 넣지 않는다. 이
  기각이 `claims.md:180-185`(Claim 9 한계)·`briefs.md:120` 과 정렬된다. 3편도
  "잔차는 beta 인자가 아닌 별도 커널"로 같은 결론을 반복하지 않고 4편을 링크한다.
- **상태·순서 서술.** block_table H2D 동기화 3지점(2편 552, 6편 552·876·1030,
  7편 876·1030)과 `+1` 해석(`seq_lens = current_prompt_len + 1`, 5편·7편)이
  편 간 동일하다. `gpu_last_tokens`(임베딩 입력)와 `gpu_active_slots`/`gpu_seq_lens`
  (paged attention 입력)의 분리는 6·7편이 같은 시그니처(`src/kernels.cu:461`,
  `src/main.cpp:878`)로 일치시킨다.
- **제약·한계 서술.** GPU·CUDA 툴체인 부재, gated 가중치
  `meta-llama/Llama-3.2-1B-Instruct` 확보 제약, `MAX_NEW_TOKENS_GENERATED=20`
  미참조, `softmaxDecode` 미호출, 매 레이어 block_table 전체 H2D TODO 가 모든
  관련 편에서 동일하게 서술된다.

## Problems

편 간 기술 모순은 없다. 아래는 교정·문서 정합 관찰이다.

- **P1 (권장, 한글 오타).** ko-07 `slot-lifecycle` 다이어그램의 break 엣지 라벨이
  `"0 + 큐 빔"`(`articles/ko-07.md:188`)이다. 의도는 "활성 0 + 큐 비면"이며, en-07
  의 대응 라벨은 `"0 + queue empty"` 로 정확하다(`articles/en-07.md:194`). 편별
  리뷰 `reviews/ko-07.md` O1 이 게시 전 교정을 권장한다.
- **P2 (권장, 영문 용어 통일).** en-07 만 시리즈의 장을 "Part" 가 아니라
  "episode" 로 표기한다(`articles/en-07.md:16,18,20,93,95` 등 13곳). en-01..en-06
  은 전부 "Part"/"part" 이므로, 같은 용어를 두 가지로 쓰는 편 간 불일치다. 한글
  ko-07 은 "편" 으로 다른 편과 일관된다.
- **P3 (문서 계층, 근거 문서 정합).** `guide.md:174,177,178,180` 과
  `briefs.md:39` 는 `dual-backend-branch`, `rmsnorm-reduction`,
  `column-row-major-trick`, `block-table-indexing` 의 형식을 "블록 다이어그램"으로
  적지만, `series.yaml:27,76,92,130` 은 `mermaid` 다. 초안들은 `series.yaml` 을
  따라 mermaid 로 작성되었으므로 초안 결함이 아니며, 게시 전 근거 문서 두 곳의
  표기를 일치시킬 것을 남긴다(`reviews/ko-01.md` O5 와 동일).
- **P4 (선택, 표현 정밀도).** ko-03 의 "모델 forward 전체가 `prefill` 안에 들어
  있다"(`articles/ko-03.md:29-30`)는 `guide.md:103-105` 의 "모델 forward 전체와
  paged KV cache 관리가 `prefill` 과 `main` 안에"와 1편의 서술
  (`articles/ko-01.md:160-162`)을 좁힌 표현이다. prefill 장의 문맥에서는 사실과
  어긋나지 않고 3편 리뷰 O2 도 인정했지만, 1편 문구와 정렬하려면 "prefill 경로의
  모델 forward 가" 로 다듬을 수 있다.
- **P5 (선택, 다이어그램 완성도).** ko-07 다이어그램은 `num_active_slots == 0`
  분기에서 break 만 그리고 `continue` 경로(`src/main.cpp:745`)를 생략한다. 본문이
  정확히 서술하므로 오류는 아니고(`reviews/ko-07.md` O3), `continue → D` 화살표를
  추가하면 본문과 완전히 대응한다.
- **P6 (선택, 기타).** 편별 리뷰의 비차단 관찰 — ko-01 O1·O2·O4, ko-02 O1·O2,
  ko-04 O1·O2, ko-05 O1~O4, ko-06 O1, en-03·en-04·en-07 의 영문 표현 관찰 — 은
  모두 인용·표현 정밀도 수준이며 기술 오류가 아니다. 편 간 일관성에 영향을 주지
  않아 이 리뷰에서 그대로 계승한다.

## Required fixes

1. **P1.** `articles/ko-07.md:188` 의 mermaid 라벨 `"0 + 큐 빔"` 을
   `"0 + 큐 비면"` 으로 교정한다(필수 시각 자료의 한글 오타, 게시 전).
2. **P2.** `articles/en-07.md` 의 장 표기를 en-01..en-06 과 동일한 "Part" 로
   통일한다(영문 용어 일관성, 게시 전).
3. **P3.** `guide.md:174,177,178,180` 과 `briefs.md:39` 의 형식 표기를
   `series.yaml` 과 일치하게 "mermaid" 로 정정한다(근거 문서 정합, 초안 수정
   아님).

위 3건은 모두 편집·문서 정합 사항이며 기술 내용, 인용, claim 소유, 게시 조건
(`guide.md:206-220`)을 건드리지 않는다. 적용 후 재검토는 편집 diff 만으로
충분하다.