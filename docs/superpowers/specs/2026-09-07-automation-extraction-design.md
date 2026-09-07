# 자동화 코드를 `blog-automation` submodule 로 분리한다

- 날짜: 2026-09-07
- 상태: 설계 승인 대기 → 구현 계획서로
- 범위: 서브프로젝트 1 (전체 분해안은 아래 "이 스펙이 다루지 않는 것" 참고)

## 왜

논문 파이프라인이 블로그 리포 안에서 자랐다. 코드(`scripts/`, 1621행)와
테스트(`tests/`, 11개 파일)가 Hugo 콘텐츠와 같은 리포에 있고, 경로는
`__file__` 에서 리포 루트를 역산한다.

```python
# scripts/papers.py:31
REPO = Path(__file__).resolve().parent.parent
# scripts/paperlib/render.py:14
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
```

즉 "스크립트는 블로그 리포 바로 아래에 있다" 가 코드에 박혀 있다. 앞으로
논문 외의 자동화 작업(첫 후속은 오픈소스 분석)을 계속 붙일 예정이라,
지금 경계를 긋지 않으면 작업이 늘어날 때마다 같은 가정이 복제된다.

## 무엇을

`ryujaehun/blog-automation` (**private**) 리포를 새로 만들고, 블로그의
`automation/` 에 submodule 로 마운트한다. 코드와 기본값은 새 리포로, 운영
설정·상태·산출물은 블로그에 남긴다.

## 리포 경계

| 대상 | 위치 | 근거 |
|---|---|---|
| 파이프라인 코드 | `blog-automation` | 도구 |
| 기본 설정·프롬프트 (`defaults/`) | `blog-automation` | submodule 단독 테스트가 가능해야 한다 |
| 테스트·CI | `blog-automation` | 코드와 같이 있어야 한다 |
| 라이브 설정 `data/paper-filter.yaml` (227행) | 블로그 | 사람이 자주 손보는 운영 파라미터 |
| 라이브 프롬프트 `data/paper-prompts/paper-review.md` (227행) | 블로그 | 동일 |
| 상태·리포트 `data/papers/**` | 블로그 | 매 실행 갱신된다. 여기 있으면 실행 후 커밋이 1개다 |
| 산출물 `content/posts/**` | 블로그 | 결과물 |
| PDF 캐시 `.cache/papers/` | 블로그 | 이미 gitignore 되어 있다 |

상태를 submodule 에 두면 실행마다 submodule 커밋 + 블로그 포인터 커밋으로
커밋이 2개가 된다. 게다가 `state.py:reviewed_ids()` 가 이미 `content/posts/`
를 스캔해 진짜 원본을 블로그에서 읽고 있으므로, 상태를 블로그에 두는 것이
데이터 소유권과도 맞다.

## 디렉터리 구조

```
blog-automation/
├─ README.md                  구조 · workspace 개념 · 새 작업 추가법
├─ requirements.txt           블로그에서 이동
├─ .github/workflows/ci.yaml  pytest + ruff
├─ core/
│   ├─ workspace.py           블로그 루트 해석
│   ├─ http.py                URL 읽기 (UA · timeout · 에러 래핑)
│   ├─ jsonl.py               원자적 read/write + latest_file
│   └─ log.py                 stderr 로그 · stdout JSON · 종료코드 상수
├─ tasks/
│   └─ papers/
│       ├─ cli.py             ← scripts/papers.py
│       ├─ layout.py          workspace 상대경로 표 (신규)
│       ├─ sources.py scoring.py render.py state.py summarize.py
│       ├─ create_post.py     ← scripts/create_post.py (그대로)
│       ├─ defaults/
│       │   ├─ paper-filter.yaml    ← data/paper-filter.yaml 의 복사본
│       │   └─ paper-review.md      ← data/paper-prompts/paper-review.md 의 복사본
│       └─ README.md          ← docs/paper-pipeline.md
├─ tests/
│   ├─ conftest.py
│   ├─ fixtures/{arxiv_query.xml, golden_reviewed.json, …}
│   └─ papers/test_*.py
└─ docs/specs/2026-09-06-paper-pipeline-design.md   ← 블로그에서 이동
```

블로그에서 사라지는 것: `scripts/`, `tests/`, `requirements.txt`,
`docs/paper-pipeline.md`, `docs/superpowers/specs/2026-09-06-paper-pipeline-design.md`.
블로그에 추가되는 것: `automation/` submodule, `.gitmodules` 항목.

## core 의 계약

네 모듈만 둔다. 작업이 2개가 되기 전에 단계(stage) 프레임워크는 만들지 않는다.

### `core/workspace.py`

작업이 읽고 쓸 대상 리포(=블로그)의 루트를 정한다.

```python
class Workspace:
    def __init__(self, root: Path)
    @classmethod
    def resolve(cls, explicit: str | None = None) -> "Workspace"
    def path(self, relative: str) -> Path        # root / relative
    def ensure_dir(self, relative: str) -> Path  # mkdir -p 후 반환
    def rel(self, path: Path) -> str             # 로그·state.json 용 상대 경로
```

`resolve()` 우선순위:

1. `--workspace PATH` (명령행)
2. `$AUTO_WORKSPACE`
3. CWD 에서 위로 올라가며 `hugo.yaml` 을 찾는다
4. 못 찾으면 `WorkspaceError` — 종료코드 4 (설정 오류)로 죽는다

4번이 중요하다. 못 찾았을 때 submodule 루트로 조용히 폴백하면 산출물이
`automation/content/posts/` 에 생기고 사람은 원인을 못 찾는다. 명시적으로
실패한다.

3번 덕분에 submodule 이 블로그 안에 있는 평소 상황에서는 플래그가 필요
없다. `python automation/tasks/papers/cli.py score` 가 그대로 동작한다.

### `core/http.py`

`sources.py:_read` 와 `USER_AGENT` 를 옮긴다. `HttpError(RuntimeError)` 를
정의하고, `sources.py` 가 이를 잡아 기존 `SourceError` 로 감싼다 — 호출부와
테스트의 예외 계약은 변하지 않는다.

**재시도는 넣지 않는다.** 현재 코드에 재시도가 없으므로 추가하면 순수
이동이 아니라 동작 변경이다. 예의용 `delay`/`sleep` 도 지금처럼 호출부
(`cli.py`, `fetch_arxiv`)에 남긴다. 재시도는 후속 항목(아래)으로 둔다.

### `core/jsonl.py`

`papers.py` 의 `read_jsonl`, `write_jsonl`(tmp → replace), `latest_file` 을
그대로 옮긴다.

### `core/log.py`

```python
def log(message)                # stderr. stdout 은 --json 전용
def emit(payload, as_json)      # stdout 에 JSON
EXIT_OK = 0; EXIT_NO_CANDIDATES = 2; EXIT_NETWORK = 3
EXIT_CONFIG = 4; EXIT_SUMMARIZE = 5
```

종료코드는 작업 간 계약이다(cron 이 이걸로 분기한다). 30행 남짓이지만
작업마다 복사하면 값이 갈라진다.

## 작업(task)의 계약

새 작업을 추가할 때 지켜야 할 최소 규약. README 에 그대로 적는다.

```
tasks/<name>/
  cli.py       argparse. main(argv) -> int(종료코드).
               공통 플래그: --workspace, --json
  layout.py    workspace 상대경로를 전부 여기 모은다. 다른 파일에서
               경로 문자열을 조립하지 않는다.
  defaults/    기본 설정·프롬프트. 테스트는 이것만 쓴다.
  README.md    사용법
tests/<name>/  해당 작업 테스트
```

## 경로 하드코딩 제거

현재 `papers.py` 에 리포 상대 경로가 10곳(31–42행 상수 + `relative_to(REPO)`
호출부), `render.py` 에 1곳(14–16행), `create_post.py` 에 1곳(77행) 흩어져
있다. 이를 `layout.py` 한 곳으로 모은다.

```python
# tasks/papers/layout.py
CONFIG   = "data/paper-filter.yaml"
PROMPTS  = "data/paper-prompts"
STATE    = "data/papers/state.json"
RAW      = "data/papers/raw"
SCORED   = "data/papers/scored"
REPORTS  = "data/papers/reports"
TASKS    = "data/papers/tasks"
POSTS    = "content/posts"
PDFCACHE = ".cache/papers"
WORKDIR  = ".cache/papers/workdir"
```

`cli.py` 는 `ws = Workspace.resolve(args.workspace)` 로 시작해
`ws.path(layout.STATE)` 형태로 쓴다. 모듈 임포트 시점에 경로를 계산하는
현재 방식(모듈 상수)을 버리고 런타임에 해석한다 — 그래야 테스트가
`tmp_path` 를 workspace 로 주입할 수 있다.

`render.py` 의 모듈 상수 `PROMPT_DIR`/`REVIEW_PROMPT` 와 `create_post.py`
의 임포트 시점 `open(PROMPT_PATH)` 도 같이 없앤다. 지금은 `create_post`
를 임포트하는 것만으로 프롬프트 파일이 있어야 한다 — 테스트에서 걸림돌이다.

## 기본값 / 오버라이드

설정과 프롬프트는 **workspace 에 있으면 그것, 없으면 `defaults/`** 를 쓴다.
`--config PATH` 로 강제 지정도 가능하게 한다.

```python
def resolve_config(ws, explicit=None):
    if explicit: return Path(explicit)
    live = ws.path(layout.CONFIG)
    return live if live.exists() else DEFAULTS / "paper-filter.yaml"
```

블로그에 두 파일이 이미 있으므로 **운영 동작은 변하지 않는다.**

이게 필요한 이유: 현재 `tests/test_golden_set.py:15` 와
`tests/test_scoring_pipeline.py:198,209` 가 블로그의
`../data/paper-filter.yaml` 을 직접 읽는다. 이대로 옮기면 automation 리포
단독 CI 에서 파일이 없어 깨진다. 테스트는 `defaults/` 를 쓰도록 바꾼다.

`defaults/` 는 현재 블로그 파일의 복사본으로 시작한다. 이후 두 파일은
갈라질 수 있고, 그래도 된다 — `defaults/` 는 "새 workspace 를 위한 출발점",
블로그 파일은 "지금 운영 중인 값" 이다.

## 임포트 방식

설치(`pip install -e`)를 요구하지 않는다. `pyproject.toml` 을 두지 않는다.

- `tasks/__init__.py`, `tasks/papers/__init__.py`, `core/__init__.py` 를 둔다.
- `cli.py` 상단에서 automation 루트를 `sys.path` 에 넣는다 (현재
  `papers.py:29` 가 이미 하는 것과 같은 방식).
- 그 뒤 `from core import http, jsonl, log, workspace`,
  `from tasks.papers import scoring, sources, render, state, summarize` 로
  임포트한다.
- `tests/conftest.py` 는 automation 루트 하나만 `sys.path` 에 넣는다.
- `render.py` 의 `import create_post` 는
  `from tasks.papers import create_post` 가 된다.

## 히스토리 이전

`filter-repo` 의 `--path-rename` 매핑은 실수하기 쉽다. 필터로 경로만
남기고, 재배치는 새 리포에서 일반 `git mv` 커밋으로 한다. `git log --follow`
가 rename 을 넘어 추적하므로 히스토리는 온전하다.

```bash
# 1) 임시 클론 (원본을 건드리지 않는다)
git clone /home/jaehun/blog "$WORK/blog-automation-src"
cd "$WORK/blog-automation-src"

# 2) 옮길 경로만 남긴다
git filter-repo \
  --path scripts \
  --path tests \
  --path requirements.txt \
  --path docs/paper-pipeline.md \
  --path docs/superpowers/specs/2026-09-06-paper-pipeline-design.md

# 3) private 리포 생성 후 push
gh repo create ryujaehun/blog-automation --private
git remote add origin git@github.com:ryujaehun/blog-automation.git
git push -u origin main

# 4) 새 리포에서 재배치 + 코드 수정 (커밋을 나눈다)
#    a. git mv 로 구조 재배치
#    b. core/ 추출
#    c. layout.py + Workspace 도입, 경로 하드코딩 제거
#    d. defaults/ 추가 + 테스트가 defaults 를 쓰게 수정
#    e. README.md, CI

# 5) 블로그에서 제거 + submodule 등록
cd /home/jaehun/blog
git rm -r scripts tests
git rm requirements.txt docs/paper-pipeline.md \
       docs/superpowers/specs/2026-09-06-paper-pipeline-design.md
git submodule add git@github.com:ryujaehun/blog-automation.git automation
```

`blog-automation` 이 private 이므로, 블로그를 clone 하는 제3자는 submodule
체크아웃에 실패한다. 블로그 배포 workflow 는 `GITHUB_TOKEN` 으로는 private
submodule 을 읽지 못하므로 **4번 항목(아래 "CI")을 반드시 같이 처리한다.**

## CI

### 블로그 배포 workflow

`.github/workflows/main.yaml` 은 이미 `submodules: recursive` 다. 그런데
`blog-automation` 이 private 이면 기본 `GITHUB_TOKEN` 으로는 체크아웃이
**실패한다** — 지금까지의 submodule(`themes/hugo-narrow`)이 public 이라
문제가 없었을 뿐이다.

Hugo 빌드는 `automation/` 을 전혀 읽지 않는다. 그래서 배포 workflow 에서는
automation submodule 을 체크아웃하지 않는 것이 맞다:

```yaml
- uses: actions/checkout@v7
  with:
    submodules: false
    fetch-depth: 0
- run: git submodule update --init --recursive themes/hugo-narrow
```

`submodules: recursive` 를 `false` + 테마만 명시 초기화로 바꾼다. 토큰
설정이 필요 없고, 실패 지점이 줄어든다. 서브프로젝트 2(cron 실행)에서
automation 을 체크아웃해야 할 때 그때 PAT 를 secret 으로 넣는다.

### automation 리포 CI

`.github/workflows/ci.yaml` 신설: `python -m pytest` + `ruff check`.

## 검증

구현 완료의 기준. 계획서의 각 단계는 이 중 해당 항목을 통과해야 한다.

1. `python -m pytest` 가 automation 리포 **단독으로** 전부 통과 (블로그
   경로에 의존하지 않음).
2. `ruff check` 통과.
3. **회귀 확인** — 보관된 raw 로 재채점해 기존 리포트와 바이트 단위로 같은
   산출이 나오는지 본다:
   ```bash
   python automation/tasks/papers/cli.py score --input data/papers/raw/2026-09-07.jsonl
   git diff --stat data/papers/reports/2026-09-07.md   # 변경 없어야 한다
   ```
   `score` 는 네트워크를 타지 않으므로 이 비교가 결정적이다.
4. `cli.py materialize --dry-run` 이 이전과 같은 목록을 낸다.
5. `--workspace` 없이 블로그 안에서 실행했을 때 경로가 블로그를 가리킨다
   (`.cache/papers/`, `data/papers/` 가 `automation/` 아래에 생기지 않는다).
6. workspace 를 못 찾는 위치(예: `/tmp`)에서 실행하면 종료코드 4 로 명확한
   메시지와 함께 죽는다.
7. `hugo --gc --minify` 빌드가 이전과 같은 페이지 수를 낸다 (KO 998 / EN 191).
8. 블로그 배포 workflow 가 성공한다.

## 이 스펙이 다루지 않는 것

전체 분해안 중 이 스펙은 1번만 다룬다.

| # | 서브프로젝트 | 상태 |
|---|---|---|
| 0 | 초안 6편 게시 | **완료** (커밋 `3d7f8ad`) |
| 1 | automation 리포 추출 + submodule | 이 스펙 |
| 2 | 파이프라인 cron 실행 + 자동 커밋/배포 | 미착수 (1 의존) |
| 3 | 영어 번역 단계 (deepseek-4-flash / muse-spark-1.3-free) | 미착수 (1, 2 의존) |
| 4 | PDF 표·그림 추출 (표부터) | 미착수 (1 의존) |
| 5 | 템플릿 일원화 + 기존 중복 25쌍 처리 | 미착수, 독립 |

`create_post.py` 의 front matter 생성은 **그대로 옮기기만** 한다.
`archetypes/posts.md` 와의 일원화, 그리고 확인된 `<h1>` 중복(테마가
front matter `title` 로 h1 을 만드는데 본문도 `# 제목` 으로 시작한다)은
서브프로젝트 5 소관이다. 추출 커밋에 리팩터링을 섞으면 회귀 확인이
불가능해진다.

`core/http.py` 의 재시도, `sources.py` 의 백오프도 이 스펙 밖이다.
