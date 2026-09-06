# 논문 수집·선별·초안 파이프라인 설계

작성일: 2026-09-06

## 1. 목적

LLM 추론 최적화 논문을 매일 손으로 찾는 수고를 없앤다. arXiv 신규 논문과
HuggingFace Daily Papers에서 후보를 모으고, 관심사 부합도로 점수를 매겨,
기준을 넘은 논문의 PDF를 받아 Hugo 초안까지 만든다. 요약 본문은 opencode에
위임하되, 완전 자동과 반자동 두 경로를 모두 지원한다.

## 2. 범위

**한다**

- HuggingFace Daily Papers + arXiv 신규 피드에서 논문 수집
- 규칙 기반 스코어링으로 후보 선별
- 임계값과 상한을 넘긴 논문의 PDF 다운로드
- Hugo 초안(`draft: true`) 생성
- opencode 서브프로세스로 요약 본문 채우기(선택)

**하지 않는다**

- LLM 재랭킹, 임베딩 유사도 검색 — 외부 큐레이션 신호와 규칙으로 충분하다
- 데이터베이스 — JSONL 파일과 기존 글 스캔으로 상태를 관리한다
- 영어 번역본(`*.en.md`) 생성 — 기존 워크플로 그대로 둔다
- 자동 커밋·푸시 — 초안 생성까지만 하고 사람이 검토한다
- GitHub Actions 크론 — 로컬 CLI가 안정화된 뒤 별도 작업으로 올린다

## 3. 아키텍처

네 단계가 파일로만 통신한다. 각 단계는 독립 실행 가능하다.

```
   HF Daily Papers API ─┐
                        ├─→ fetch ─→ data/papers/raw/YYYY-MM-DD.jsonl
   arXiv Atom API ──────┘             정규화 레코드, ID 기준 병합
                                          │
   data/paper-filter.yaml ────────────→ score ─→ data/papers/scored/YYYY-MM-DD.jsonl
   content/posts/ (기존 리뷰 ID) ─────→   │      data/papers/reports/YYYY-MM-DD.md
                                          │
                                     materialize ─→ .cache/papers/<id>.pdf
                                          │         content/posts/<날짜>-paper-<id>.md
                                          │         data/papers/state.json
                                          │
                                      summarize ─→ 초안 본문 채움 (백엔드별)
```

단계를 나눈 이유는 `score`를 네트워크 없이 재실행할 수 있게 하기 위함이다.
가중치와 임계값은 저장된 raw JSONL로 초 단위 반복 보정한다.

## 4. 데이터 모델

`fetch`가 만드는 정규화 레코드. 한 줄에 하나씩 JSONL로 쓴다.

```json
{
  "id": "2609.03430",
  "version": "v1",
  "title": "Random Attention: Rethinking KV Cache Eviction ...",
  "abstract": "...",
  "authors": ["...", "..."],
  "primary_category": "cs.LG",
  "categories": ["cs.LG", "cs.CL"],
  "published": "2026-09-03T00:00:00Z",
  "sources": ["hf", "arxiv"],
  "hf": {
    "upvotes": 161,
    "num_comments": 4,
    "ai_keywords": ["KV cache compression", "vLLM throughput"],
    "github_repo": "https://github.com/...",
    "github_stars": 31,
    "organization": "...",
    "submitted_at": "2026-09-04T01:22:00Z"
  },
  "fetched_at": "2026-09-06T08:00:00Z"
}
```

`hf` 필드는 HF에 없는 논문이면 `null`. `sources`가 병합 출처를 기록하며
스코어링의 게이트 선택에 쓰인다.

**ID 정규화** — arXiv ID에서 버전 접미사를 떼고 `2609.03430` 형태로 통일한다.
버전은 `version`에 따로 담는다. 중복 판정은 항상 버전 없는 ID로 한다.

## 5. 소스 어댑터

### 5.1 HuggingFace Daily Papers

- `GET https://huggingface.co/api/daily_papers?date=YYYY-MM-DD`
- 인증 불필요. 응답은 배열, 각 원소의 `paper` 객체에 필요한 필드가 모두 있다.
- `--since 7d`이면 날짜 7개를 각각 호출한다. 요청 간 1초 간격.
- 하루 20~50편.

### 5.2 arXiv 신규 피드

- `GET https://export.arxiv.org/api/query?search_query=cat:<cat>&sortBy=submittedDate&sortOrder=descending&max_results=<n>`
- **`https` 필수** — `http`는 301을 반환하고 본문이 비어 있다.
- 수집 카테고리: `cs.LG`, `cs.CL`, `cs.DC`, `cs.AR`, `cs.PF`.
  §7의 `preferred_categories`와는 다른 목록이다 — 이쪽은 "무엇을 긁어올까",
  저쪽은 "긁어온 것 중 무엇에 감점하지 않을까"다. 저쪽이 더 넓다
- 요청 간 3초 간격(arXiv 권장).
- `--since` 범위 밖의 `published`는 버린다.

### 5.3 병합

정규화 ID를 키로 합친다. 양쪽에 있으면 HF 메타데이터를 채우고
`sources: ["hf", "arxiv"]`로 기록한다. 제목·초록은 arXiv 쪽을 신뢰한다.

## 6. 스코어링 모델

### 6.1 최종 점수

```
score = (0.55·topic + 0.25·buzz + 0.12·impl + 0.08·venue) × category_mult
```

`category_mult`는 `primary_category`가 선호 목록에 있으면 `1.0`, 아니면 `0.9`.

### 6.2 topic (0~1) — 지배적 항

어휘집의 각 용어를 세 필드에서 찾는다. 한 용어는 **필드당 최대 1회**만
집계한다(초록에 열 번 나와도 한 번).

```
raw   = Σ  tier_weight(term) × field_weight(field)
topic = 1 − exp(−raw / K)          K = 6.0
```

| 필드 | 가중 |
|---|---|
| `title` | 2.0 |
| `hf.ai_keywords` | 1.5 |
| `abstract` | 1.0 |

티어 가중은 설정 파일이 정한다(§7).

**매칭 규칙**

- 소문자 정규화 후 **단어 경계 기준**으로 찾는다 — `int8`이 `print8`에
  걸리지 않아야 한다. 여러 단어로 된 용어(`kv cache`)는 구(phrase) 전체를
  경계로 감싸 찾고, 용어 안의 공백은 공백·하이픈·언더스코어를 모두 받는다
  (`kv cache`, `kv-cache`, `kv_cache`가 같이 걸린다)
- **겹치는 매치는 가장 무거운 것 하나만 센다.** `sparse attention`이 걸린
  자리에서 `attention`(adjacent)을 또 세지 않는다. 구현은 각 필드에서
  매치 구간을 수집한 뒤 가중 내림차순으로 훑으며 이미 점유된 구간과 겹치는
  매치를 버린다. 이 규칙이 없으면 긴 용어일수록 짧은 용어를 품어 점수가
  부풀고, 어휘를 추가할수록 기존 점수가 흔들려 임계값 보정이 무의미해진다
- 한 용어는 **필드당 최대 1회**만 집계한다(초록에 열 번 나와도 한 번)

`core_hits`(core 티어에서 매치된 서로 다른 용어 수)를 함께 기록한다.
게이트와 하드 제외 판정에 쓴다.

### 6.3 buzz (0~1)

```
buzz = min(1, log1p(upvotes) / log1p(150))
```

HF에 없으면 `0`.

### 6.4 impl (0~1)

```
impl = min(1, has_repo·0.5 + 0.5 · min(1, log1p(stars) / log1p(500)))
```

추론 최적화 논문은 구현체 유무가 실질 신호라 별도 항으로 둔다.

### 6.5 venue (0~1)

`hf.organization` 이름이 화이트리스트에 있으면 `1.0`, 없으면 `0`.
가중이 8%라 없어도 큰 손해는 아니다.

### 6.6 제외 규칙

점수 계산 **이전에** 적용한다.

1. 기존 글에서 복원한 arXiv ID — 버전 무시하고 제외
2. `state.json`에 `status: "materialized"`로 기록된 ID — 제외
3. `state.json`에 실패로 3회 이상 기록된 ID — 제외(재시도 폭주 방지)
4. 하드 제외 어휘가 제목에 있고 **`core_hits == 0`**일 때만 제외

4번의 조건부 설계가 핵심이다. `Radial Attention: ... for Long Video
Generation`처럼 도메인은 비디오지만 sparse attention이 본체인 논문을
살리기 위함이다. 실제로 이 블로그가 리뷰한 논문이다.

### 6.7 게이트 — 소스별 분리

arXiv에만 있는 논문은 `buzz = 0`이라 HF 출신과 같은 임계값으로는 영원히
통과하지 못한다. 그래서 게이트를 나눈다.

모든 후보는 먼저 **실질 용어 관문**을 넘어야 한다. `core` / `named_systems`
/ `systems` 티어에서 최소 1개가 걸려야 한다(`min_substantive_hits`).

이 규칙은 구현 중 실제 데이터에서 발견해 추가했다. 없으면 upvote 140 의
`SolarWM: Long-Horizon Video World Models` 가 `autoregressive` +
`distillation` 만 걸고도 topic 0.714 로 통과한다. 인기와 약한 용어 누적이
주제 관련성을 대신해서는 안 된다.

관문을 넘은 뒤 출처별 게이트를 적용한다.

| 출처 | 통과 조건 |
|---|---|
| `sources`에 `hf` 포함 | `score ≥ 0.45` |
| arXiv 단독 | `topic ≥ 0.70` **또는** `core_hits ≥ 2` |

통과분을 `score` 내림차순 정렬하고 **`top_k`(기본 5)** 로 자른다. 임계값이
잘못 잡혀도 한 번에 5편을 넘지 않는다.

## 7. 설정 파일 — `data/paper-filter.yaml`

티어 개수를 코드에 박지 않는다. `tiers`는 순서 있는 목록이고 각 항목이
가중과 용어를 갖는다.

```yaml
version: 1

weights:
  topic: 0.55
  buzz: 0.25
  impl: 0.12
  venue: 0.08

field_weights:
  title: 2.0
  keywords: 1.5
  abstract: 1.0

saturation_k: 6.0
buzz_reference: 150      # 이 upvote 수를 1.0으로 본다
stars_reference: 500

gates:
  hf_min_score: 0.45
  arxiv_min_topic: 0.70
  arxiv_min_core_hits: 2
  top_k: 5

preferred_categories: [cs.LG, cs.CL, cs.DC, cs.AR, cs.PF, cs.SE, cs.MS, cs.OS]
non_preferred_multiplier: 0.9

tiers:
  - name: core
    weight: 3.0
    terms:
      # 추론·서빙 고유 개념
      - kv cache
      - kv-cache
      - key-value cache
      - kv compression
      - cache compression
      - prefix caching
      - prefix cache
      - speculative decoding
      - speculative sampling
      - paged attention
      - pagedattention
      - continuous batching
      - prefill
      - decode phase
      - decoding phase
      - ttft
      - tpot
      - time to first token
      - time per output token
      - llm serving
      - llm inference
      - inference serving
      - model serving
      - sparse attention
      - flash attention
      - flashattention
      - disaggregated
      - disaggregation
      - slo
      - kv eviction
      - cache eviction

  - name: named_systems
    weight: 2.5
    terms:
      # 고유명사 자체가 강한 신호
      - vllm
      - sglang
      - flashinfer
      - tensorrt
      - qserve
      - lserve
      - xgrammar
      - deepspeed
      - orca
      - sarathi
      - cutlass
      - triton

  - name: systems
    weight: 2.0
    terms:
      # MLSys 축 — 2023-2025 채택 논문에서 추출
      - quantization
      - post-training quantization
      - w4a8
      - w4a16
      - w8a8
      - fp8
      - int4
      - int8
      - offloading
      - cpu offload
      - attention kernel
      - attention engine
      - fused attention
      - context parallelism
      - pipeline parallelism
      - tensor parallelism
      - communication overlap
      - computation-communication
      - overlapping
      - scheduling
      - structured generation
      - re-sharding
      - resharding
      - gpu memory
      - on-device
      - edge llm

  - name: adjacent
    weight: 1.5
    terms:
      - mixture-of-experts
      - mixture of experts
      - moe
      - long context
      - throughput
      - latency
      - memory efficient
      - compression
      - attention
      - distillation
      - pruning
      - sparsity
      - batching
      - cache
      - kernel

  - name: weak
    weight: 0.5
    terms:
      - efficient
      - efficiency
      - scaling
      - transformer
      - llm
      - large language model
      - acceleration
      - optimization

exclude_terms:
  # 제목에 있고 core_hits == 0 일 때만 제외한다
  - text-to-image
  - image generation
  - video generation
  - protein
  - molecule
  - drug discovery
  - recommendation
  - federated learning
  - autonomous driving
  - robot
  - speech synthesis
  - medical imaging
  - time series
  - graph neural network

organizations:
  # venue 항 화이트리스트 (소문자 부분 일치)
  - deepseek
  - qwen
  - alibaba
  - moonshot
  - zhipu
  - bytedance
  - nvidia
  - microsoft
  - google
  - deepmind
  - meta
  - tsinghua
  - berkeley
  - stanford
  - carnegie mellon
  - lmsys
  - hazy research
  - han lab

summarize:
  model: opencode-go/glm-5.3
  variant: null            # --variant 값, 없으면 생략
  timeout_seconds: 900
```

`federated learning`을 제외 어휘에 넣은 건 의도적이다. MLSys 3년치에서 10회
등장할 만큼 큰 축이지만 이 블로그의 관심사가 아니다.

## 8. `materialize`

1. `scored.jsonl`에서 게이트 통과분을 `top_k`까지 읽는다
2. `https://arxiv.org/pdf/<id>` → `.cache/papers/<id>.pdf` (요청 간 3초)
3. `create_post.py`의 `get_arxiv_title()`로 제목을 확인한다
4. `create_post.py`의 `front_matter()`로 초안 머리말을 만든다
5. 초안을 `content/posts/<오늘>-paper-<id>v<n>.md`에 쓴다
6. `state.json`을 갱신한다

**기존 코드 재사용** — `create_post.py`는 이미 `get_arxiv_title()`,
`front_matter()`, `yaml_quote()`, `PROMPT_TEMPLATE`이 모듈 수준에 분리되어
있다. `if __name__ == "__main__"` 가드도 있어 import에 부작용이 없다.
그대로 import해서 쓰고, 로직을 복제하지 않는다.

`PROMPT_TEMPLATE`만 `data/paper-prompts/paper-review.md`로 옮기고
`create_post.py`는 그 파일을 읽도록 고친다. 세 백엔드가 같은 프롬프트를
공유해야 하므로 코드가 아니라 데이터여야 한다. `create_post.py`의 기존 CLI
동작은 바뀌지 않는다.

파일 쓰기는 임시 파일 → `os.replace()`로 원자적으로 처리한다.

## 9. `summarize` — 백엔드 3종

```
papers.py summarize --backend {none|opencode|task} [--id <arxiv-id>]
```

### 9.1 `none` (기본)

현행 `create_post.py`와 동일. 초안 본문에 질의 프롬프트 묶음만 넣는다.
네트워크도 API 키도 쓰지 않는다.

### 9.2 `opencode` — 완전 자동

```
opencode run --format json --auto \
             --dir .cache/papers/workdir \
             --model <summarize.model> \
             [--variant <v>] \
             --file .cache/papers/<id>.pdf \
             "<프롬프트 본문>"
```

**`--auto` 가 필요하다.** 없이 돌리면 비대화형 실행에서 권한 프롬프트를
기다리다 그대로 멈춘다(opencode 1.18.27 에서 확인). opencode 자체 도움말이
"dangerous" 로 표시하는 플래그라, `--dir` 을 리포 밖 전용 작업 폴더
(`.cache/papers/workdir`)로 좁혀 파일 조작을 가둔다.

출력은 **줄 단위 JSON 이벤트**다. 본문은 `{"type":"text","part":{"type":
"text","text":...}}` 이벤트에 조각으로 실려 오고, `step_finish` 이벤트에
`cost`(USD)와 토큰 수가 담긴다.

**stdout 을 본문으로 쓰면 안 된다.** opencode 는 완성 응답을 내는 LLM 이
아니라 파일을 쓰고 결과를 보고하는 에이전트다. 실제로 28KB 리뷰를
`workdir/2609.03430-review.md` 에 쓰고 stdout 으로는 953 바이트짜리
"완료했습니다…" 보고만 냈다. 그 보고문이 초안 본문이 되어 원본을 덮어썼다.

그래서 **출력 파일 계약**을 쓴다.

1. 프롬프트 끝에 쓸 경로를 명시한다 (`build_review_prompt`)
2. 실행 전에 그 경로의 낡은 파일을 지운다 — 이전 실행 결과를 새 결과로
   착각하지 않기 위해
3. 실행 후 그 파일을 읽는다. 없으면 stdout 본문으로 물러선다
4. 어느 쪽이든 `MIN_BODY_CHARS`(2000자)에 못 미치면 실패로 보고 초안을
   건드리지 않는다 — 작업 보고문이 본문 자리에 들어가는 걸 막는 마지막 방어

- 논문마다 **새 세션**(`--continue` 미사용) — 컨텍스트 오염 방지
- `--format json`으로 받아 파싱, 실패하면 stderr를 보존하고 초안은
  `none` 수준으로 남긴다. 빈 본문 파일을 만들지 않는다
- `timeout_seconds` 초과 시 프로세스를 죽이고 실패로 기록
- 성공 시 초안 카테고리에 `with-<model-slug>` 추가
  (`opencode-go/glm-5.3` → `with-glm-5-3`). 기존 `with-gpt-5.2` 관례를 잇는다

API 키는 opencode가 `~/.local/share/opencode/auth.json`에서 관리한다.
파이프라인 코드는 자격 증명을 다루지 않는다.

### 9.3 `task` — 반자동

초안과 함께 작업 지시서를 쓴다.

```json
{
  "id": "2609.03430",
  "title": "...",
  "pdf_path": ".cache/papers/2609.03430.pdf",
  "draft_path": "content/posts/2026-09-06-paper-2609.03430v1.md",
  "prompt_path": "data/paper-prompts/paper-review.md",
  "score": 0.71,
  "matched_terms": {"core": ["kv cache", "cache eviction"], "systems": []}
}
```

`data/papers/tasks/<id>.json`에 쓴다. opencode TUI나 Claude Code가 이 파일을
읽고 대화형으로 본문을 채운다.

## 10. 상태·중복 관리

### 10.1 이미 리뷰한 논문

`content/posts/`를 매 실행 스캔한다. 두 경로로 ID를 복원한다.

- 파일명: `*-paper-<id>*.md` → 304개
- 본문: `arxiv.org/(abs|pdf)/<id>` → 367개
- 합집합 368개

캐시하지 않는다. 글을 지우거나 옮겨도 다음 실행에 자동으로 반영된다.

### 10.2 `data/papers/state.json`

```json
{
  "2609.03430": {
    "first_seen": "2026-09-06",
    "materialized_at": "2026-09-06",
    "score": 0.71,
    "draft_path": "content/posts/2026-09-06-paper-2609.03430v1.md",
    "status": "materialized",
    "failures": 0
  }
}
```

`status`는 `materialized` | `failed` | `skipped`. `failures`가 3 이상이면
이후 실행에서 제외한다.

## 11. CLI

```
papers.py fetch       [--date YYYY-MM-DD] [--since 7d] [--source hf,arxiv] [--json]
papers.py score       [--input FILE] [--date YYYY-MM-DD] [--config PATH] [--json]
papers.py materialize [--input FILE] [--top-k N] [--dry-run] [--json]
papers.py summarize   [--backend none|opencode|task] [--id ID] [--model M] [--json]
papers.py run         [--since 7d] [--backend ...]      # 네 단계 연속 실행
```

**에이전트 친화 규약** — `task` 백엔드가 성립하려면 CLI가 기계 판독 가능해야
한다.

- 모든 서브커맨드가 `--json`을 받는다. 켜면 stdout은 JSON만 낸다
- 사람이 읽을 로그는 전부 stderr로 보낸다
- 종료 코드로 결과를 구분한다

| 코드 | 뜻 |
|---|---|
| 0 | 성공 |
| 2 | 정상 동작했으나 통과한 후보가 없음 |
| 3 | 네트워크 실패 |
| 4 | 설정 오류 |
| 5 | 요약 백엔드 실패 |

`--dry-run`은 `materialize`에서 점수 분포와 통과 예정 목록만 출력하고
파일을 쓰지 않는다. 임계값 보정용이다.

## 12. 에러 처리

- **`fetch`** — 소스 하나가 실패해도 나머지로 진행한다. 경고를 stderr에
  쓰고 리포트에 실패한 소스를 명시한다. 전부 실패하면 코드 3
- **`score`** — 네트워크를 쓰지 않으므로 실패 경로는 설정 오류(코드 4)뿐.
  알 수 없는 티어 이름, 음수 가중, 빈 어휘집은 시작 시 검증해서 거른다
- **`materialize`** — PDF 다운로드 실패는 해당 논문만 건너뛰고 `state.json`에
  `failures`를 올린다. 나머지는 계속 처리한다
- **`summarize`** — opencode 실패 시 초안을 `none` 수준으로 남기고 코드 5.
  이미 만들어진 초안을 지우지 않는다

## 13. 테스트

`tests/test_scoring.py` — `score`가 순수 함수라 대부분 여기서 검증된다.

1. **단위** — 필드 가중, 티어 가중, 중복 용어 1회 집계, 단어 경계 매칭
   (`int8`이 `print8`에 안 걸림), 포화 함수 단조성
2. **게이트** — HF 출신과 arXiv 단독이 서로 다른 조건으로 판정되는지,
   `top_k` 절단
3. **제외 규칙** — 하드 제외 어휘가 `core_hits > 0`이면 무력화되는지.
   `Radial Attention ... Long Video Generation`을 실제 픽스처로 쓴다
4. **골든셋 회귀** — 과거 리뷰한 논문 중 표본 12편을 raw 레코드로 재구성해
   "이건 통과해야 한다"는 기준선을 만든다. 어휘집·가중치를 만질 때 회귀를
   잡는 안전망이다

픽스처는 `tests/fixtures/`에 실제 API 응답을 저장해 쓴다. 테스트는 네트워크를
타지 않는다.

## 14. 파일 구성

```
scripts/papers.py                CLI 진입점
scripts/paperlib/__init__.py
scripts/paperlib/sources.py      HF + arXiv 어댑터, 병합
scripts/paperlib/scoring.py      순수 함수
scripts/paperlib/render.py       create_post.py 재사용, 초안 쓰기
scripts/paperlib/summarize.py    백엔드 3종
scripts/paperlib/state.py        state.json, 기존 글 스캔
scripts/create_post.py           기존 파일 — 프롬프트 외부화만 수정
data/paper-filter.yaml           어휘집·가중치·임계값
data/paper-prompts/paper-review.md
data/papers/raw/                 (gitignore)
data/papers/scored/              (gitignore)
data/papers/reports/             커밋한다 — 무엇이 왜 걸렸는지 남는다
data/papers/tasks/               (gitignore)
data/papers/state.json           커밋한다
.cache/papers/                   (gitignore)
tests/test_scoring.py
tests/fixtures/
```

## 15. 의존성

`PyYAML`만 새로 필요하다(이미 설치되어 있다). 어휘집은 사람이 손보는
파일이라 주석이 필수라서 JSON으로는 부족하다.

HTTP는 표준 라이브러리 `urllib`을 쓴다. `create_post.py`가 이미 그렇게 하고
있고 `requests`를 더할 이유가 없다.

`requirements.txt`를 새로 만들어 `PyYAML`을 적는다.
