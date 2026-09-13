# tiny-vllm 시리즈 전체 리뷰

고정 커밋 `1896ff5c37a241050dbcd9527caf9dc1d3087a61` 기준. 검토 대상은
`guide.md`, `series.yaml`, `briefs.md`, `evidence/claims.md`, `evidence/trace.md`,
`articles/ko-01..10.md`, `articles/en-01..10.md`.

## Decision

REVISE

## Coverage

- **챕터 구성·순서**: `series.yaml`의 10개 챕터가 모두 ko/en 양쪽에 존재하고
  (`articles/ko-01..10.md`, `articles/en-01..10.md`), 각 문서의 제목·Scope가
  `series.yaml`의 챕터 제목·대상 파일과 정확히 일치한다. 서사 순서(실행 경로 →
  가중치 포맷 → 순전파 → 배칭/스케줄링 → paged attention)도 `guide.md:24-29`의
  arc를 따른다.
- **Claim 1~10 커버리지**: 전 Claim이 한 번 이상 소유·참조된다. Claim 1(2장),
  Claim 2(4장, 9장 한계 참조), Claim 3(3·4·6장), Claim 4(3·6·7장 참조),
  Claim 5(1·5·6·10장), Claim 6(5·6·8·10장), Claim 7(4·5·10장),
  Claim 8(9·10장), Claim 9(1·3·7·8장), Claim 10(8·10장). 각 챕터가 사용한
  주장·소유 표기는 `briefs.md`의 챕터 브리프와 일치한다.
- **Claim 한계 커버리지**: 각 Claim의 Limitation이 소유·참조 챕터에서 모두
  노출된다. 실행 미검증(`recon.md:4`), 유일 테스트 `tests/test_softmax.cu`,
  `prompt_len>1024` 가드, 상수 중복(`kernels.cu:6`), D2D `cudaMemcpy` 산포,
  GQA 하드코딩, `dot_products[2]`의 `HEAD_DIM=64` 의존, 전체 블록 테이블
  재복사 등이 챕터 간 반복 인용되며 수치가 어긋나지 않는다.
- **챕터 간 경계**: 브리프가 정한 넘김(5장→커널/배칭, 6장→7·8·10장,
  7장→연속 배칭, 8장→커널 내부, 9장→블록 순회)이 문서 본문의 "다루지 않는 것"
  절에 일관되게 반영되어 있다. 소유·참조 구분도 본문 표기와 브리프가 일치한다.

## Cross-chapter consistency

- **상수**: `BLOCK_SIZE=16`, `KV_DIM=512`, `V_OFFSET=16384`, `BLOCK_BYTES=32768`,
  `KV_CACHE_SIZE_BYTES=2GiB`, `NUM_BLOCKS=65536`, `MAX_BLOCKS_PER_SEQ=128`,
  `BATCH_SIZE=2`, GQA 32/8/4, `VOCAB_SIZE=128256`, `MAX_SEQ_LEN=2048`,
  `MAX_PROMPT_LEN=512`, `HEAD_DIM=64`이 1·5·7·10장과 근거 문서에서 동일하다.
- **줄 범위**: KV 산포 prefill `:247-288`/decode `:851-873`, `block_table` 동기화
  `:876/:552/:1030`, paged attention `src/kernels.cu:461-523`·래퍼 `:525-527`,
  GQA `:303/:346/:468`, 로짓 argmax `:1000-1012`가 관련 챕터에서 일치한다.
  `block_table_gpu` 크기 2×16×128×4바이트(16KiB)도 8·10장이 같다.
- **용어**: "슬롯/물리 블록/논리 블록/산포/전치 트릭/온라인 softmax/warp 트리
  합/활성 슬롯/여유 슬롯/재동기화"의 ko/en 대응이 전 챕터에서 일관되다.
  "block"과 "page"도 `BLOCK_SIZE` TODO 인용(`src/main.cpp:32`)에서만 page가
  쓰이고 본문은 block으로 통일된다.
- **attention 스케일**: `attn_alpha=1/8`(`:649`), 주석 `Q*K^T/sqrt(64)`(`:298`),
  `SQRT_HEAD_DIM=8`(`src/kernels.cu:12`)이 4·5·9장에서 상호 일치한다.
- **커널 가드**: `causalMask`/`softmax`(`src/kernels.cu:241-244`, `:295-298`)와
  `softmaxDecode` 미호출(dead code) 사실이 1·3·4·6·9장에서 동일하게 서술된다.
- **Korean-English pairing**: 10쌍 전부 동일한 섹션 구조와 주장을 담고 있으며,
  핵심 수치·인용이 어긋나지 않는다. 챕터 간 상호 참조(`ko-07.md:140-141` 등)의
  대상 위치도 실제 내용과 일치한다.

## Problems

1. **6장 prefill 순서 오류 (교차 모순)** — 6장 시퀀스 다이어그램의 prefill 열이
   attention(스코어 GEMM `:301-327` → `causalMask` `:329` → `softmax` `:331` →
   scores×V `:343-371`)을 KV 산포(`:247-288`) **앞에** 놓는다
   (`ko-06.md:119-122`, `en-06.md:138-141`). 단계 비교표도 attention 행을
   KV 산포 행 앞에 둔다(`ko-06.md:105-106`, `en-06.md:124-125`). 그러나 소스
   실행 순서는 RoPE(`:244-245`) → KV 산포(`:247-288`) → attention(`:301-371`)이고,
   `claims.md` Claim 3, `trace.md:94-121`, 4장 파이프라인, 5장 본문이 모두
   산포를 attention 앞에 둔다. 같은 다이어그램의 decode 열은 KV 산포(`:851-873`)
   → 동기화(`:876`) → pagedAttention(`:878`) 순으로 옳아서, prefill 열의
   도식 순서와 그 자체의 줄 번호가 서로 모순된다. 6장 내부 모순이자 4·5장 및
   근거 문서와의 교차 모순이다.
2. **9장 경계 중첩 (경미)** — "이 커널은 어디서 호출되는가" 절
   (`ko-09.md:148-153`, `en-09.md:153-158`)이 `num_blocks`(`:471`)와
   `tokens_in_block`(`:481-482`)을 인용한다. 이는 브리프상 10장이 소유하는
   블록 순회·마스킹 근거다. Claim 8의 "읽지 않음 = 마스킹" 문맥으로 소개되어
   모순은 아니지만, 10장의 "누적 K만 읽음 = causal masking" 절과 부분 중복된다.
3. **근거 문서에 없는 소스 직접 관찰 (경미)** — `CUBLAS_COMPUTE_32F`
   (`ko-04.md:118-120`), `SQRT_HEAD_DIM=8`(`ko-09.md:122`) 등이 Claim 1~10이나
   `trace.md`에는 없고 소스 줄만 인용한 채 등장한다. ko/en 쌍 간에는 일관되고
   인용 형식도 `guide.md:11-16`의 소스 인용 규칙을 만족하므로 오류는 아니지만,
   브리프의 "Claim 1~10만 사용" 규정과의 긴장이 있다.

## Required fixes

1. **6장 prefill 열의 순서 교정** — 시퀀스 다이어그램(`ko-06.md:119-122`,
   `en-06.md:138-141`)과 단계 비교표(`ko-06.md:105-106`, `en-06.md:124-125`)에서
   KV 산포(블록 단위, `:247-288`)를 attention 블록(스코어 GEMM `:301-327` →
   `causalMask` `:329` → `softmax` `:331` → scores×V `:343-371`)보다 **앞에**
   놓아, 소스 순서 및 4·5장·Claim 3과 일치시킨다.
2. **(선택) 9장의 순회 근거 인용 정리** — `num_blocks`/`tokens_in_block` 인용을
   10장 소유로 명시하거나 10장으로의 명시적 넘김 문구를 추가해 블록 순회 근거의
   소유를 10장에 한정한다.
3. **(선택) 소스 직접 관찰의 근거 처리** — 4·9장의 소스 직접 관찰
   (`CUBLAS_COMPUTE_32F`, `SQRT_HEAD_DIM`)을 `trace.md`/`claims.md`에 보강
   근거로 추가하거나, Claim 1~10 밖의 소스 관찰임을 명시적으로 표기한다.