# 논문 파이프라인 무인 실행과 자동 게시

- 날짜: 2026-09-07
- 상태: 구현 완료 (`blog-automation` `2aec01f`)
- 범위: 서브프로젝트 2. 사용법은 `automation/README.md` 의 "자동 실행" 을 본다.
  이 문서는 **왜 그렇게 했는지**만 남긴다.

## 문제

파이프라인은 있는데 스케줄이 없었다. 손으로 돌리고 손으로 커밋해야만
배포됐고, 실제로 초안 6편이 `draft: true` 로 며칠 방치됐다. 배포
workflow 는 정상이었지만 프로덕션 빌드가 `-D` 없이 돌아 사이트에 나오지
않았다.

## workflow 를 블로그가 아니라 automation 리포에 둔다

opencode API 키가 필요한데 블로그는 공개 리포다. workflow 와 secret 을
private 인 `blog-automation` 에 두고, 블로그는 체크아웃 대상(workspace)
으로만 다룬다.

```
blog-automation (private)          blog (public)
  daily.yaml + secrets  ──push──►  content/posts, data/papers
                                       │
                                       ▼
                                   Deploy Hugo (기존 workflow, 손대지 않음)
```

## PAT 대신 deploy key

이유가 둘이다.

1. **범위** — 리포 하나에만 유효하고 만료 관리가 없다. PAT 는 계정 전체
   범위를 잘못 주기 쉽다.
2. **트리거** — `GITHUB_TOKEN` 으로 push 한 커밋은 다른 workflow 를
   **트리거하지 않는다.** 그걸로 밀면 블로그 배포가 돌지 않아 글이
   사이트에 나오지 않는다. 애초에 고치려던 증상이 그대로 재발한다.
   deploy key push 는 트리거한다.

공개키는 블로그 리포 Deploy keys 에 read-write 로 등록(id 162544513),
개인키는 `blog-automation` 의 `BLOG_DEPLOY_KEY` secret. 로컬 사본은
생성 직후 `shred` 했다.

## 게시는 요약 성공에만 묶는다

`materialize` 는 프롬프트만 든 초안을 만든다. 그게 게시되면 사이트에
질의 프롬프트가 그대로 나간다.

```
materialize          항상 draft: true
summarize --publish  본문이 채워진 것만 draft: false
```

`build_draft(publish=True, body=None)` 은 `ValueError` 로 거부한다.
플래그 하나로 사고가 나지 않게 코드 수준에서 막았다. 요약이 실패한
논문은 초안으로 남아 **자동으로 비공개**가 된다 — 실패가 조용히 게시로
이어지지 않는다.

## exit 2 를 단계별로 다르게 처리한다

여기가 함정이었다. `score` 의 exit 2 는 "정상 동작했으나 후보 없음"
이고, 주말엔 arXiv 신규 발표가 없어 정상적으로 자주 나온다. 실패로
잡으면 매주 실패 메일이 오고, 곧 알림을 무시하게 된다.

| 단계 | exit 2 | 근거 |
|---|---|---|
| `score` | **성공** (커밋 없이 종료) | 후보 없는 날은 정상이다 |
| `materialize` | **실패** | 뽑아 놓고 초안이 하나도 없으면 PDF 다운로드가 막힌 것이다. 조용히 넘기면 며칠이 지나도 모른다 |

## 비용을 남긴다

`extract_run_cost` 가 구현돼 있는데 **아무도 호출하지 않고 있었다.**
무인으로 매일 돈을 쓰는 이상, 얼마 썼는지 남지 않으면 청구서를 보고
나서야 알게 된다. `Summary(body, model, cost)` 로 돌려주고 합계를
로그·`--json`(`cost_usd`)·job summary 에 적는다.

## top_k 5 → 2

매일 실행에 `top_k: 5` 면 주 25편이다. 이 블로그의 과거 실적은 주 4편
남짓(2년간 414편)이라 6배다. 2 로 내려 평일 기준 주 10편 상한으로 뒀다.
탈락분은 `data/papers/reports/` 에 남아 무엇을 놓쳤는지 보인다.

블로그의 `data/paper-filter.yaml` 과 패키지 기본값을 함께 내렸다.
분량 조정은 블로그 쪽 파일 한 줄이다.

## 남은 위험

**arXiv/HF 가 데이터센터 IP 를 조일 수 있다.** 코드에 재시도가 없어
`fetch` 만 workflow 레벨에서 3회 재시도한다. 그래도 실패하면 exit 3 으로
알림이 온다. 재시도를 `core/http.py` 로 내리는 건 서브프로젝트 1 에서
의도적으로 미룬 항목이다(순수 이동을 유지하려고).

**요약 품질을 사람이 보지 않는다.** `--publish` 를 켠 것은 사용자의
명시적 선택이다. 되돌리기는 후속 커밋으로 `draft: true` 로 바꾸면 된다.
품질 게이트가 필요해지면 서브프로젝트 5(템플릿) 와 함께 다룬다.

## 이 문서가 다루지 않는 것

서브프로젝트 3(영어 번역), 4(표·그림 추출), 5(템플릿 일원화 + 기존 중복
25쌍). 전체 분해안은
[2026-09-07-automation-extraction-design.md](2026-09-07-automation-extraction-design.md)
를 본다.
