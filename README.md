# jaehun.me

개인 블로그(<https://jaehun.me>) 소스입니다. [Hugo](https://gohugo.io) + [hugo-narrow](https://github.com/tom2almighty/hugo-narrow) 테마로 만들었습니다.

## 요구사항

- Hugo **extended** 0.165.0 이상 (`hugo.yaml`의 `module.hugoVersion.min`과 GitHub Actions가 같은 버전을 사용합니다)
- 별도 Node/Sass 툴체인은 필요 없습니다. 테마가 컴파일된 CSS(`assets/css/compiled.css`)를 커밋해 배포합니다.

## 시작하기

```bash
git clone --recurse-submodules https://github.com/ryujaehun/blog.git
cd blog
hugo server -D          # http://localhost:1313
```

테마는 git submodule입니다. 이미 클론한 저장소라면:

```bash
git submodule update --init --recursive
```

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
```

다국어는 `content/` 한 곳에서 파일명 접미사로 관리합니다. 한국어는 기본 언어라 `/`, 영어는 `/en/` 아래에 배포됩니다.

## 배포

`main`에 push하면 `.github/workflows/main.yaml`이 빌드해 GitHub Pages로 배포합니다.
도메인은 `static/CNAME`(jaehun.me)으로 연결되어 있습니다.

## Acknowledgement

- 테마: <https://github.com/tom2almighty/hugo-narrow> (GPL-3.0)
- 일부 글 작성에 LLM(Gemini, GPT)의 도움을 받았습니다.
