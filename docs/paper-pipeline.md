# 논문 파이프라인 사용법

`scripts/papers.py` 는 LLM 추론 최적화 논문을 찾아 읽을 만한 것만 골라
Hugo 초안까지 만든다. 매일 arXiv 를 뒤지는 수고를 없애는 게 목적이다.

설계 배경과 근거는
[스펙 문서](superpowers/specs/2026-09-06-paper-pipeline-design.md)에 있다.
이 문서는 쓰는 법만 다룬다.

## 준비

```bash
pip install -r requirements.txt
```

`PyYAML` 과 `defusedxml` 두 개뿐이다. 요약 자동화까지 쓰려면
[opencode](https://opencode.ai) 가 설치되어 있고 프로바이더 인증이
끝나 있어야 한다(`opencode providers list` 로 확인).

## 한눈에

```bash
python3 scripts/papers.py fetch --since 5d        # 1. 후보 수집
python3 scripts/papers.py score                   # 2. 채점  ← 여기를 반복한다
python3 scripts/papers.py materialize --dry-run   # 3. 미리보기
python3 scripts/papers.py materialize             # 3. PDF + 초안
python3 scripts/papers.py summarize --backend task  # 4. 요약
```

네 단계를 이어서 돌리려면:

```bash
python3 scripts/papers.py run --since 5d --backend task
```

## 단계별로 무슨 일이 일어나나

### 1. `fetch` — 후보 수집

HuggingFace Daily Papers 와 arXiv 신규 피드를 읽어 하나로 합친다.

```
data/papers/raw/<날짜>.jsonl
```

HF 는 날짜별 API 라 `--since 5d` 면 5일치를 각각 부른다. **주말에는 HF 에
논문이 올라오지 않는다** — `0편` 이 떠도 고장이 아니다. arXiv 도 주말에는
신규가 없다.

**`--since` 는 게시일이 아니라 arXiv 제출일(v1)로 거른다.** 그런데 HF
데일리는 제출된 지 2~4일 지난 논문을 올린다. 창을 좁게 잡으면 소스가
멀쩡히 읽혀도 최종 0편이 된다.

```
--since 3d → HF 19편 + arXiv 1000편 → 창(09-05~09-07) 밖 947편 제외 → 0편
--since 5d → HF 86편 + arXiv 1000편 → 창(09-03~09-07) 밖 600편 제외 → 400편
```

**`--since 5d` 아래로는 내리지 마라.** 월요일에 `5d` 로 도는 게 기준선이다.

날짜 범위 밖이라 버린 편수를 로그에 찍는다. `1000편 수집 → 0편` 같은
일이 조용히 벌어지지 않게 하기 위함이다.

```bash
python3 scripts/papers.py fetch --date 2026-09-04            # 하루만
python3 scripts/papers.py fetch --since 7d --source hf       # HF 만
```

### 2. `score` — 채점

**이 단계는 네트워크를 타지 않는다.** 저장된 raw 파일만 읽으므로 몇
십 밀리초면 끝난다. 어휘집이나 임계값을 고치고 이 명령만 다시 돌리면
된다 — 파이프라인을 4단계로 나눈 이유가 이것이다.

```
data/papers/scored/<날짜>.jsonl   기계용
data/papers/reports/<날짜>.md     사람용  ← 이걸 본다
```

이미 글로 쓴 논문은 자동으로 빠진다. `content/posts/` 를 매번 훑어
파일명과 본문 arXiv 링크에서 ID 를 복원한다(현재 369편). 별도 DB 가
없으니 글을 지우거나 옮겨도 다음 실행에 반영된다.

### 3. `materialize` — PDF + 초안

게이트를 통과한 논문 중 상위 `top_k` 편의 PDF 를 받아
`content/posts/<날짜>-paper-<id><버전>.md` 에 `draft: true` 초안을 만든다.

```bash
python3 scripts/papers.py materialize --dry-run    # 무엇이 만들어질지만
python3 scripts/papers.py materialize --top-k 2    # 설정값 덮어쓰기
```

`--dry-run` 을 먼저 보는 습관을 들이는 게 좋다. 임계값을 잘못 잡아도
`top_k` 가 상한이라 한 번에 5편을 넘지 않지만, 안 쓸 초안이 쌓이면
`content/posts/` 가 지저분해진다.

PDF 는 `.cache/papers/` 에 받는다(git 무시). arXiv 예의상 요청 간 3초를 쉰다.

### 4. `summarize` — 본문 채우기

세 가지 방식이 있다.

| 백엔드 | 하는 일 | 언제 |
|---|---|---|
| `none` (기본) | 초안에 질의 프롬프트만 둔다 | 직접 정리할 때 |
| `task` | 작업 지시서(JSON)를 남긴다 | **반자동** — Claude Code / opencode TUI 가 이어받음 |
| `opencode` | opencode 를 돌려 본문까지 채운다 | **완전 자동** |

```bash
python3 scripts/papers.py summarize --backend task
python3 scripts/papers.py summarize --backend opencode
python3 scripts/papers.py summarize --backend opencode --id 2609.03430 \
        --model opencode-go/kimi-k3
```

#### 모델과 폴백

`data/paper-filter.yaml` 의 `summarize` 에서 정한다.

```yaml
summarize:
  model: opencode-go/deepseek-v4-pro                    # 1차
  fallback_model: opencode/muse-spark-1.3-contributor-free  # 2차 (무료)
```

1차가 `SummarizeError` 로 죽으면 2차로 넘어간다. 프로바이더 한도나 장애로
한 모델이 통째로 막혔을 때 그날 수집분을 전부 버리지 않으려는 장치다.
**둘 다 실패해야 실패**이고, 그때 두 모델의 사유가 함께 남는다.

폴백 모델의 `contributor-free` 는 프로바이더가 `opencode-go` 가 아니라
`opencode` 다. 접두사를 옮겨 쓰면 모델을 못 찾는다.

한 번만 바꿔 볼 때는 CLI 로 덮어쓴다. 둘은 독립이다.

```bash
python3 scripts/papers.py summarize --backend opencode \
        --model opencode-go/glm-5.3 --fallback-model opencode-go/qwen3.8-flash
```

`task` 가 만드는 지시서는 이렇게 생겼다. 에이전트에게 이 파일 경로만
알려주면 된다.

```json
{
  "id": "2609.03430",
  "title": "Random Attention: Rethinking KV Cache Eviction ...",
  "pdf_path": "/…/.cache/papers/2609.03430.pdf",
  "draft_path": "/…/content/posts/2026-09-06-paper-2609.03430v1.md",
  "prompt_path": "/…/data/paper-prompts/paper-review.md",
  "score": 0.892,
  "matched": {"core": ["kv cache", "cache eviction"], "named_systems": ["vllm"]}
}
```

`opencode` 백엔드가 성공하면 초안 카테고리에 `with-<모델>` 이 붙는다
(`opencode-go/glm-5.3` → `with-glm-5-3`). 기존 `with-gpt-5.2` 관례를 잇는 것이다.
**폴백이 돌았으면 폴백 모델 이름이 붙는다** — 나중에 어느 글을 어느 모델이
썼는지 카테고리만 보면 된다.

**실패하면 초안을 건드리지 않는다.** 프롬프트만 있는 상태로 남고 종료
코드 5 를 낸다. 반쯤 채워진 글이 생기지 않게 하기 위함이다.

## 선별 규칙 읽는 법

점수는 네 항의 가중합이다.

```
score = (0.55·topic + 0.25·buzz + 0.12·impl + 0.08·venue) × 카테고리 보정
```

| 항 | 뜻 |
|---|---|
| `topic` | 어휘집이 제목·HF 키워드·초록에서 찾은 용어의 가중합. 지배적인 항이다 |
| `buzz` | HF upvote. **10개 이하는 0**, 150개에서 1.0 (그 사이는 선형) |
| `impl` | GitHub 레포 유무 + 스타 수. 추론 최적화는 구현체가 실질 신호다 |
| `venue` | 조직 화이트리스트(DeepSeek, Qwen, NVIDIA …) |

점수를 넘기 전에 두 관문이 있다.

1. **실질 용어 관문** — `core` / `named_systems` / `systems` 티어에서
   최소 1개가 걸려야 한다. 이게 없으면 `efficient`, `attention`,
   `distillation` 같은 약한 용어가 쌓인 인기 논문이 통과한다.
2. **출처별 게이트** — HF 에 올라온 논문은 `score ≥ 0.45`,
   arXiv 단독은 `topic ≥ 0.70` 또는 `core_hits ≥ 2`.
   arXiv 단독은 upvote 가 구조적으로 0 이라 총점으로 재면 영원히 못 넘는다.

통과분을 점수 내림차순으로 `top_k`(기본 5)까지 자른다.

## 튜닝하는 법

리포트(`data/papers/reports/<날짜>.md`)에 **선정분과 탈락분이 같이** 나온다.
점수·topic·core·upvote·출처가 열로 붙어 있어 왜 걸리고 왜 떨어졌는지
바로 보인다.

```
| | ID | 점수 | topic | core | upvote | 출처 | 제목 |
| **선정** | 2609.03430 | 0.892 | 0.997 | 3 | 161 | hf+arxiv | Random Attention: … |
|  | 2609.02886 | 0.668 | 0.714 | 0 | 140 | hf | SolarWM: … Video World Models |
```

(실제 리포트에서 ID 는 arXiv 링크로 걸려 있다. 위는 줄여 쓴 것이다.)

반복 절차는 이렇다.

1. 리포트에서 잘못 걸린 것 / 놓친 것을 찾는다
2. `matched` 열(리포트 아래쪽 "선정된 논문의 매칭 근거")로 어떤 용어가
   걸렸는지 본다
3. `data/paper-filter.yaml` 을 고친다
4. **`score` 만 다시 돌린다** — 네트워크를 안 타서 즉시 끝난다
5. `python3 -m pytest tests/` 로 회귀를 확인한다

5번이 중요하다. `tests/test_golden_set.py` 가 **과거에 실제로 리뷰한 논문
12편**을 들고 있어서, 어휘를 잘못 건드려 표적을 놓치게 되면 바로 잡힌다.
반대로 관심 밖 논문 5편이 통과하면 그것도 잡힌다.

### 어휘집 구조

```yaml
tiers:
  - name: core            # 3.0 — 추론·서빙 고유 개념 (kv cache, prefill, TTFT …)
  - name: named_systems   # 2.5 — 고유명사 (vllm, sglang, flashinfer …)
  - name: systems         # 2.0 — MLSys 축 (quantization, offloading, overlap …)
  - name: adjacent        # 1.5 — 인접 (moe, long context, throughput …)
  - name: weak            # 0.5 — 범용 (efficient, scaling, transformer …)
```

- 티어 개수는 코드에 박혀 있지 않다. 늘리거나 줄여도 된다
- `substantive_tiers` 가 실질 용어 관문에 쓰일 티어를 정한다
- 매칭은 단어 경계 기준이라 `int8` 이 `print8` 에 걸리지 않는다
- 용어 안의 공백/하이픈/언더스코어는 서로 같게 본다.
  `kv cache` 하나로 `kv-cache`, `kv_cache` 를 다 잡는다
- **겹치는 매치는 무거운 것만 센다.** `sparse attention` 이 걸린 자리에서
  `attention` 을 또 세지 않는다. 이게 없으면 어휘를 추가할수록 기존 점수가
  흔들려 임계값 보정이 무의미해진다

### 제외 어휘

`exclude_terms` 는 **제목에 있고 core 용어가 하나도 없을 때만** 작동한다.

조건부인 이유가 있다. `Radial Attention: … for Long Video Generation` 은
도메인이 비디오지만 sparse attention 이 본체이고, 실제로 이 블로그가
리뷰한 논문이다. 무조건 제외하면 이런 걸 놓친다.

## 알아둘 함정

만들면서 실제로 밟은 것들이다.

- **주말엔 논문이 없다.** HF Daily Papers 도 arXiv 도 그렇다. `0편` 은 정상
- **`--since 3d` 는 좁다.** HF 데일리가 올리는 논문은 제출된 지 2~4일
  지난 것들이라 3일 창에는 하나도 안 들어온다. 최소 `5d`
- **arXiv API 는 `https` 여야 한다.** `http` 는 301 을 주고 본문이 비어 있다
- **opencode 는 완성 응답을 내는 LLM 이 아니라 에이전트다.** 그냥 두면
  본문을 파일에 쓰고 stdout 으로는 "완료했습니다" 만 낸다. 그래서
  프롬프트에 쓸 경로를 명시하고 그 파일을 읽는다. 2000자에 못 미치면
  실패로 본다
- **`opencode run` 은 `--auto` 없이 비대화형에서 멈춘다.** 권한 프롬프트를
  기다린다. 대신 `--dir` 을 `.cache/papers/workdir` 로 좁혀 파일 조작을
  리포 밖에 가둔다
- **`--dir` 이 작업 디렉터리를 바꾸므로** `--file` 은 절대 경로여야 한다
- **`opencode --file` 은 배열 옵션이다.** 뒤에 프롬프트를 두면 그것까지
  파일로 삼켜 `File not found: <프롬프트>` 로 죽는다

## 비용

요약 자동화만 돈이 든다. 수집·채점·초안 생성은 무료다.

폴백(`opencode/muse-spark-1.3-contributor-free`)은 무료라 1차가 막혔을 때
비용이 늘지 않는다. 1차 `opencode-go/deepseek-v4-pro` 만 과금된다.

`opencode-go/qwen3.8-flash` 로 논문 한 편을 끝까지 정리하는 데 약 6분,
`step_finish` 이벤트가 보고한 비용은 센트 단위였다. 더 큰 모델을 쓰면
비례해 오른다. `data/paper-filter.yaml` 의 `summarize.model` 로 바꾼다.

주 3~5편이 목표라면 월 비용은 크지 않다. 정확한 수치는 `opencode stats`
로 확인할 수 있다.

## 종료 코드

에이전트가 스크립트를 호출해 쓰라고 나눠 뒀다. `--json` 을 주면 stdout 은
JSON 만, 사람이 읽을 로그는 전부 stderr 로 간다.

| 코드 | 뜻 |
|---|---|
| 0 | 성공 |
| 2 | 정상 동작했으나 통과한 후보가 없음 |
| 3 | 네트워크 실패 |
| 4 | 설정 오류 |
| 5 | 요약 백엔드 실패 |

## 파일 배치

```
scripts/papers.py              CLI
scripts/paperlib/              라이브러리
  scoring.py                     어휘 매칭·게이트 (순수 함수)
  sources.py                     HF / arXiv / PDF
  state.py                       중복 관리
  render.py                      초안·리포트
  summarize.py                   요약 백엔드 3종
scripts/create_post.py         PDF 폴더 → 초안 (기존 도구, 그대로 동작)

data/paper-filter.yaml         어휘집·가중치·임계값   ← 튜닝은 여기
data/paper-prompts/            질의 프롬프트 (create_post.py 와 공유)
data/papers/
  raw/        수집 원본       (git 무시)
  scored/     채점 결과       (git 무시)
  reports/    사람용 리포트   커밋한다 — 무엇이 왜 걸렸는지 남는다
  tasks/      작업 지시서     (git 무시)
  state.json  materialize 이력  커밋한다
.cache/papers/                 PDF, opencode 작업 폴더 (git 무시)

tests/                         74개
  test_golden_set.py             과거 리뷰 논문으로 회귀 방지  ← 튜닝 안전망
```

## 어휘집은 어디서 왔나

두 곳에서 뽑았다.

- 이 블로그가 2025~2026 에 리뷰한 arXiv 논문 **149편의 제목 n-gram**
- **MLSys 2023~2025 채택 논문 144편**의 제목 n-gram

MLSys 는 federated learning(3년간 10회)·GNN·클러스터 스케줄링 비중이
커서 통째로 넣지 않았다. LLM 추론과 겹치는 축만 가져왔고, `federated
learning` 은 오히려 제외 어휘에 넣었다.

MLSys 2025 가 채워준 구멍들: `SLO`, `scheduling`, `offloading`,
`context parallelism`, `compute-communication overlap`, `attention
kernel`/`engine`, `W4A8KV4`, `structured generation`, `on-device`.
`FlashInfer`·`QServe`·`LServe`·`Marconi`·`XGrammar` 처럼 고유명사 자체가
신호인 것들은 별도 티어로 뒀다.
