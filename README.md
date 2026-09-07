# jaehun.me

개인 블로그(<https://jaehun.me>) 소스입니다. [Hugo](https://gohugo.io) + [hugo-narrow](https://github.com/tom2almighty/hugo-narrow) 테마로 만들었습니다.

## 요구사항

- Hugo **extended** 0.165.0 이상 (`hugo.yaml`의 `module.hugoVersion.min`과 GitHub Actions가 같은 버전을 사용합니다)
- 별도 Node/Sass 툴체인은 필요 없습니다. 테마가 컴파일된 CSS(`assets/css/compiled.css`)를 커밋해 배포합니다.

## 시작하기

```bash
git clone https://github.com/ryujaehun/ryujaehun.github.io.git
cd ryujaehun.github.io
git submodule update --init themes/hugo-narrow
hugo server -D          # http://localhost:1313
```

submodule이 둘 있습니다.

| submodule | 경로 | 공개 | 빌드에 필요 |
|---|---|---|---|
| [hugo-narrow](https://github.com/tom2almighty/hugo-narrow) | `themes/hugo-narrow` | public | **필요** |
| blog-automation | `automation` | private | 불필요 |

`automation`은 콘텐츠 자동화 도구라 사이트 빌드와 무관합니다. private
이므로 권한이 없으면 체크아웃되지 않고, 그래도 빌드는 됩니다. 그래서
`--recurse-submodules` 대신 테마만 초기화하는 것을 권합니다.

## 자주 쓰는 명령

```bash
hugo server -D                  # 초안(draft) 포함 로컬 미리보기
hugo --gc --minify              # 프로덕션 빌드 (public/)
hugo new content posts/글-제목.md  # archetypes/posts.md 기반 새 글
```

## 구조

```
content/
  posts/          한국어 글 (*.md) + 영어 번역 (*.en.md)
  archives/       연도별 아카이브 섹션
  about.md        소개 페이지 (about.en.md = 영어)
static/           그대로 배포되는 파일 (CNAME, robots.txt, favicon, 이미지)
themes/hugo-narrow/  테마 (submodule)
hugo.yaml         사이트 설정

automation/       콘텐츠 자동화 도구 (submodule, private)
data/             자동화 설정·프롬프트·산출물   ← 이 리포가 소유합니다
docs/             문서
```

자동화 **코드**는 `automation/` submodule에, 자동화가 읽고 쓰는
**설정과 산출물**은 이 리포에 있습니다. 코드는 버전을 매겨 배포하는
것이고 산출물은 매 실행 바뀌는 것이라, 한 리포에 두면 코드 히스토리가
산출물 커밋에 묻힙니다.

다국어는 `content/` 한 곳에서 파일명 접미사로 관리합니다. 한국어는 기본 언어라 `/`, 영어는 `/en/` 아래에 배포됩니다.

## 자동화

콘텐츠 자동화는 [blog-automation](https://github.com/ryujaehun/blog-automation)
(private submodule)에 있습니다. 구조와 새 작업 추가 방법은 그쪽
`README.md`를 보세요.

### 논문 파이프라인

LLM 추론 최적화 논문을 찾아 읽을 만한 것만 골라 Hugo 초안까지 만듭니다.
자세한 사용법은 `automation/tasks/papers/README.md`에 있습니다.

```bash
git submodule update --init automation
pip install -r automation/requirements.txt

# 이 리포 안에서 실행하면 대상(workspace)을 알아서 찾습니다
python3 automation/tasks/papers/cli.py fetch --since 5d        # 수집
python3 automation/tasks/papers/cli.py score                   # 채점 (네트워크 안 씀)
python3 automation/tasks/papers/cli.py materialize --dry-run   # 무엇이 초안이 될지
python3 automation/tasks/papers/cli.py materialize             # PDF + 초안 생성
python3 automation/tasks/papers/cli.py summarize --backend task      # 반자동
python3 automation/tasks/papers/cli.py summarize --backend opencode  # 완전 자동
```

무엇을 고를지는 `data/paper-filter.yaml`이 정합니다 — 이 리포에 있고,
튜닝은 여기서 합니다. 어휘집이나 가중치를 고친 뒤 `score`만 다시 돌리면
됩니다. 네트워크를 타지 않아 즉시 끝납니다.

선정·탈락 이유는 `data/papers/reports/<날짜>.md`에 남습니다. 이미 글로
쓴 논문은 `content/posts/`를 스캔해 자동으로 제외됩니다.

초안은 `draft: true`로 생성됩니다. 프로덕션 빌드는 `-D` 없이 돌기
때문에, 검토한 뒤 `draft: false`로 바꿔야 사이트에 나옵니다.

## 배포

`main`에 push하면 `.github/workflows/main.yaml`이 빌드해 GitHub Pages로 배포합니다.
도메인은 `static/CNAME`(jaehun.me)으로 연결되어 있습니다.

## Acknowledgement

- 테마: <https://github.com/tom2almighty/hugo-narrow> (GPL-3.0)
- 일부 글 작성에 LLM(Gemini, GPT)의 도움을 받았습니다.
