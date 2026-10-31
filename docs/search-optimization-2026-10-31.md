# 검색 최적화 점검 — 2026-10-31

## 적용 범위

기존 Hugo 테마와 URL 구조를 유지한다. 새 플러그인은 추가하지 않았다.
Google·Bing·네이버와 AI 검색이 읽는 HTML, canonical, 구조화 데이터,
사이트맵, RSS의 일관성을 개선했다. 검색 순위나 AI 답변 인용을 보장하는 작업은 아니다.

## 변경

- 목록 2쪽 이후의 canonical, og:url, CollectionPage URL과 @id가 실제 쪽 주소를 가리킨다.
  이전에는 이 값들이 모두 1쪽을 가리켰다.
- 한국어·영어 목록의 글 구성과 페이지 경계는 다르므로 2쪽 이후 hreflang을 생략한다.
  번역 글과 목록 1쪽의 hreflang은 유지한다.
- robots 결정을 하나의 partial로 통합했다. 명시적 noindex/none은 사이트맵과 RSS에서도 제외한다.
  기본적으로 색인 가능한 페이지는 큰 이미지 미리보기를 허용한다.
- 한국어·영어 메인 RSS와 게시글 RSS는 최신 30편의 본문 전체를 제공한다.
  태그·카테고리 RSS는 전송량을 줄이기 위해 요약을 유지한다.
- 공유 기본 이미지를 테마 기본 이미지에서 실제 작성자의 프로필 이미지로 변경했다.
- 사이트 설명과 소개 페이지를 LLM 추론·컴파일러·C++ 중심으로 구체화했다.
  소개에는 주제별 입구와 논문 실험 결과를 읽는 주의점을 추가했다.
- 새 글 archetype에 description 입력란을 추가했다.
- 배포 CI에서 검색 메타·공유 이미지·JSON-LD·사이트맵·전체 본문 RSS를 검사한다.

기존 네이버 HTML 소유 확인 파일은 보존했다. robots.txt의 `User-agent: * / Allow: /`는
Yeti를 포함한 검색 크롤러의 접근을 허용한다. 특정 봇용 Allow 규칙을 중복 추가할 필요는 없다.

## 검증 방법

```sh
hugo --minify --printPathWarnings --destination .hugo_seo_final
python3 -m unittest discover -s scripts -p 'test_check*.py'
python3 scripts/check_seo.py .hugo_seo_final
python3 scripts/check_local_links.py .hugo_seo_final
```

검사기는 meta만 존재하는지 보지 않고 파일 경로와 canonical·OG·CollectionPage 주소가
맞는지 확인한다. RSS 본문은 실제 게시글 article의 텍스트 전체와 비교한다.
네이버에 제출할 두 메인 RSS는 각각 10MB 미만인지도 확인한다.
소유 확인 페이지와 Hugo 리다이렉트 alias는 일반 문서 검사에서 제외한다.

## 검증 결과

- 변경 전: 검색 메타·RSS 검사 오류 1,842건. 목록 594페이지의 주소 신호 3종 불일치
  1,782건과 메인 RSS의 부분 본문 60건을 재현했다.
- 변경 후: 일반 HTML 1,748페이지, 사이트맵 URL 852개에서 검사 오류 0건.
- 검사기와 내부 링크 검사 단위 테스트 14개 통과.
- HTML 파일 2,202개에서 깨진 내부 링크·이미지 0건.
- 로컬 Hugo 0.169.0과 배포용 0.165.0에서 빌드 성공, 중복 경로 경고 없음.
- 명시적 index/noindex/none은 작은 Hugo fixture에서도 HTML·사이트맵·RSS 일치 확인.
- 메인 RSS 크기: 한국어 약 2.18MiB, 영어 약 1.92MiB.

## 네이버에서 배포 후 확인할 일

이 저장소의 파일만으로는 네이버 계정의 등록·인증 만료·수집 상태를 알 수 없다.
아래 작업은 Search Advisor 계정에서 수행한다.

1. [Search Advisor](https://searchadvisor.naver.com/) 웹마스터 도구에
   `https://jaehun.me`가 등록되어 있는지와 소유 확인 상태를 확인한다.
   HTML 파일은 이미 있으므로 현재 계정에 연결된 토큰과 같은지 확인한다.
2. 요청 → 사이트맵 제출에 `https://jaehun.me/sitemap.xml`을 제출한다.
   한국어·영어 사이트맵을 가리키는 인덱스다.
3. 요청 → RSS 제출에 `https://jaehun.me/index.xml`을 제출한다.
   영어 피드는 `https://jaehun.me/en/index.xml`이다.
4. 검증 → robots.txt에서 Yeti의 홈페이지·게시글 접근이 허용되는지 확인한다.
5. URL 검사에서 홈, 최신 글, 주요 주제 허브를 확인하고 필요한 URL의 수집을 요청한다.
6. 수집·색인·노출 보고서에서 반영 추이를 확인한다. 로컬 검증과 실제 색인 판정은 다르다.

공식 근거:
[RSS·사이트맵 제출](https://searchadvisor.naver.com/guide/request-feed),
[대표 URL·robots](https://searchadvisor.naver.com/guide/markup-structure),
[URL 검사](https://searchadvisor.naver.com/guide/url-inspection).

## Google·Bing·AI 검색 운영

- Search Console에 사이트맵을 제출하고 대표 URL과 실제 색인 상태를 확인한다.
- Bing Webmaster Tools의 색인 보고서와
  [AI Performance](https://www.bing.com/webmasters/help/ai-performance-9f8e7d6c)에서
  Copilot 등의 인용 현황을 확인한다. 이 작업에서 계정 보고서에 접근하지는 않았다.
- `llms.txt`나 별도 AI 전용 스키마를 순위 개선 수단으로 가정하지 않는다.
  Google은 AI 검색에도 일반 SEO 기본 원칙과 접근 가능한 본문을 요구한다.
  [Google AI 검색 가이드](https://developers.google.com/search/docs/appearance/ai-features).
- 크롤러가 읽을 수 있는 HTML 본문과 내부 링크를 유지한다. 현재 서버 렌더링 구조를 유지했다.

## 콘텐츠 운영 권고

1. 새 글의 description에는 대상 문제, 핵심 방법, 적용 조건을 1~2문장으로 적는다.
   프롬프트나 제목을 반복하기보다 독자가 얻을 답을 요약한다.
2. 논문 리뷰는 첫머리에 원문 링크·핵심 결론·실험 조건을 두고,
   논문이 보고한 사실과 작성자의 해석·제안을 구분한다.
3. 수치에는 모델·하드웨어·데이터셋·비교 기준을 같이 적는다.
   실제 재현하지 않은 결과는 직접 검증한 수치로 표현하지 않는다.
4. KV 캐시·추론 최적화처럼 글이 많은 주제부터 관련 글 사이의 문맥 링크를 보강한다.
5. 오래된 설치 가이드는 실제 재검증 후 수정한다. 수정하지 않은 글의 날짜만 갱신하지 않는다.
6. AI 도구 사용 여부와 검증 범위를 공개할 때는 실제 제작 과정에 맞게 적는다.
   이번 작업에서 개별 논문 리뷰의 기술적 정확성 전체를 재검증하지는 않았다.
