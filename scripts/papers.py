#!/usr/bin/env python3
"""LLM 추론 최적화 논문을 모아 걸러 초안까지 만든다.

네 단계가 파일로만 통신한다. 각 단계는 따로 돌릴 수 있다.

    papers.py fetch --since 7d      수집  -> data/papers/raw/<날짜>.jsonl
    papers.py score                 채점  -> data/papers/scored/<날짜>.jsonl + 리포트
    papers.py materialize           PDF + 초안 생성
    papers.py summarize --backend … 요약 본문 채우기
    papers.py run --since 7d        위 넷을 이어서

score 가 네트워크를 타지 않는 게 요점이다. 저장된 raw 로 가중치와 임계값을
초 단위로 다시 계산할 수 있다.

설계: docs/superpowers/specs/2026-09-06-paper-pipeline-design.md
"""

import argparse
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperlib import render, scoring, sources, summarize as summarize_mod  # noqa: E402
from paperlib.state import State, repo_relative, reviewed_ids  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "data" / "paper-filter.yaml"
POSTS_DIR = REPO / "content" / "posts"
DATA_DIR = REPO / "data" / "papers"
RAW_DIR = DATA_DIR / "raw"
SCORED_DIR = DATA_DIR / "scored"
REPORTS_DIR = DATA_DIR / "reports"
TASKS_DIR = DATA_DIR / "tasks"
STATE_PATH = DATA_DIR / "state.json"
PDF_DIR = REPO / ".cache" / "papers"
# opencode 는 --auto 로 돌아야 해서, 작업 범위를 리포 밖 전용 폴더로 가둔다.
WORKDIR = PDF_DIR / "workdir"

EXIT_OK = 0
EXIT_NO_CANDIDATES = 2
EXIT_NETWORK = 3
EXIT_CONFIG = 4
EXIT_SUMMARIZE = 5


def log(message):
    """사람이 읽을 로그는 stderr 로. stdout 은 --json 전용이다."""
    print(message, file=sys.stderr)


def emit(payload, as_json):
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=1)
        sys.stdout.write("\n")


def parse_since(text):
    m = re.fullmatch(r"(\d+)d", (text or "").strip())
    if not m:
        raise ValueError(f"--since 는 '7d' 같은 형식이어야 합니다: {text!r}")
    return int(m.group(1))


def date_window(args):
    """수집할 날짜 목록을 만든다. 최신 날짜가 앞에 온다."""
    if args.date:
        return [args.date]
    days = parse_since(args.since)
    today = date.today()
    return [(today - timedelta(days=i)).isoformat() for i in range(days)]


def load_config():
    try:
        return scoring.FilterConfig.load(CONFIG_PATH)
    except (OSError, ValueError, KeyError) as exc:
        log(f"설정을 읽지 못했습니다 ({CONFIG_PATH}): {exc}")
        raise SystemExit(EXIT_CONFIG)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def latest_file(directory, suffix):
    files = sorted(Path(directory).glob(f"*{suffix}"))
    return files[-1] if files else None


# --------------------------------------------------------------------------
# fetch


def cmd_fetch(args):
    config = load_config()
    dates = date_window(args)
    wanted = {s.strip() for s in args.source.split(",") if s.strip()}
    records, failed = [], []

    if "hf" in wanted:
        for index, day in enumerate(dates):
            if index:
                time.sleep(args.hf_delay)
            try:
                got = sources.fetch_hf_daily(day)
            except sources.SourceError as exc:
                log(f"  ! HF {day}: {exc}")
                failed.append(f"hf:{day}")
                continue
            log(f"  + HF {day}: {len(got)}편")
            records.extend(got)

    if "arxiv" in wanted:
        categories = list(config.fetch_categories) or [
            "cs.LG",
            "cs.CL",
            "cs.DC",
            "cs.AR",
            "cs.PF",
        ]
        try:
            got = sources.fetch_arxiv(categories, max_results=args.arxiv_max)
            log(f"  + arXiv {','.join(categories)}: {len(got)}편")
            records.extend(got)
        except sources.SourceError as exc:
            log(f"  ! arXiv: {exc}")
            failed.append("arxiv")

    if not records:
        log("아무 소스도 읽지 못했습니다.")
        emit({"fetched": 0, "failed_sources": failed}, args.json)
        return EXIT_NETWORK

    merged = sources.merge(records)
    before = len(merged)
    merged = [r for r in merged if sources.within_window(r, dates)]
    dropped = before - len(merged)
    if dropped:
        # 1000 편이 조용히 0 편이 되면 원인을 못 찾는다.
        log(f"  - 날짜 범위({min(dates)}~{max(dates)}) 밖이라 {dropped}편 제외")
    merged.sort(key=lambda r: r["id"], reverse=True)

    out = RAW_DIR / f"{date.today().isoformat()}.jsonl"
    write_jsonl(out, merged)
    if failed:
        (RAW_DIR / f"{date.today().isoformat()}.failed.json").write_text(
            json.dumps(failed, ensure_ascii=False), encoding="utf-8"
        )

    log(f"수집 {len(merged)}편 -> {out.relative_to(REPO)}")
    emit(
        {"fetched": len(merged), "path": str(out), "failed_sources": failed}, args.json
    )
    return EXIT_OK


# --------------------------------------------------------------------------
# score


def cmd_score(args):
    config = load_config()
    path = Path(args.input) if args.input else latest_file(RAW_DIR, ".jsonl")
    if not path or not path.exists():
        log(f"채점할 raw 파일이 없습니다. 먼저 fetch 하세요. ({RAW_DIR})")
        return EXIT_NO_CANDIDATES

    records = read_jsonl(path)
    already = reviewed_ids(POSTS_DIR)
    state = State.load(STATE_PATH)
    excluded = already | state.excluded_ids()
    log(f"제외 대상: 기존 글 {len(already)}편 + 이력 {len(state.excluded_ids())}건")

    scored = [scoring.score_paper(r, config) for r in records]
    selected = scoring.select(scored, config, excluded_ids=excluded)
    selected_ids = {s.id for s in selected}

    stamp = path.stem
    failed_path = path.with_name(f"{stamp}.failed.json")
    failed_sources = (
        json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else []
    )

    rows = [
        {
            **s.record,
            "score": round(s.score, 4),
            "components": {k: round(v, 4) for k, v in s.components.items()},
            "core_hits": s.core_hits,
            "matched": s.matched,
            "selected": s.id in selected_ids,
        }
        for s in sorted(scored, key=lambda x: -x.score)
    ]
    out = SCORED_DIR / f"{stamp}.jsonl"
    write_jsonl(out, rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{stamp}.md"
    report_path.write_text(
        render.render_report(stamp, scored, selected_ids, failed_sources),
        encoding="utf-8",
    )

    log(f"채점 {len(scored)}편 / 선정 {len(selected)}편 -> {report_path.relative_to(REPO)}")
    for s in selected:
        log(f"  * {s.score:.3f} {s.id} {s.record.get('title', '')[:64]}")

    emit(
        {
            "scored": len(scored),
            "selected": [s.id for s in selected],
            "scored_path": str(out),
            "report_path": str(report_path),
        },
        args.json,
    )
    return EXIT_OK if selected else EXIT_NO_CANDIDATES


# --------------------------------------------------------------------------
# materialize


def cmd_materialize(args):
    config = load_config()
    path = Path(args.input) if args.input else latest_file(SCORED_DIR, ".jsonl")
    if not path or not path.exists():
        log(f"채점 결과가 없습니다. 먼저 score 하세요. ({SCORED_DIR})")
        return EXIT_NO_CANDIDATES

    rows = [r for r in read_jsonl(path) if r.get("selected")]
    top_k = args.top_k or config.gates["top_k"]
    rows = rows[:top_k]
    if not rows:
        log("선정된 논문이 없습니다.")
        emit({"materialized": []}, args.json)
        return EXIT_NO_CANDIDATES

    if args.dry_run:
        log(f"[dry-run] {len(rows)}편이 초안이 됩니다:")
        for r in rows:
            log(
                f"  {r['score']:.3f} topic={r['components']['topic']:.3f} "
                f"core={r['core_hits']} {r['id']} {r.get('title', '')[:60]}"
            )
        emit({"dry_run": True, "would_materialize": [r["id"] for r in rows]}, args.json)
        return EXIT_OK

    state = State.load(STATE_PATH)
    today = date.today().isoformat()
    made = []

    for index, row in enumerate(rows):
        arxiv_id = row["id"]
        if index:
            time.sleep(args.pdf_delay)  # arXiv 예의
        try:
            pdf = sources.download_pdf(arxiv_id, PDF_DIR)
        except sources.SourceError as exc:
            log(f"  ! {arxiv_id}: PDF 실패 — {exc}")
            state.mark_failed(arxiv_id, reason=str(exc))
            continue

        title = row.get("title") or arxiv_id
        version = row.get("version")
        draft_path = POSTS_DIR / render.draft_filename(arxiv_id, version, today)
        text = render.build_draft(
            title=title,
            arxiv_id=arxiv_id,
            version=version,
            date=today,
            categories=["paper-review"],
            body=None,
        )
        render.write_draft(draft_path, text)
        relative = repo_relative(draft_path, REPO)
        state.mark_materialized(
            arxiv_id,
            score=row["score"],
            draft_path=relative,
            title=title,
            version=version,
            matched=row.get("matched", {}),
        )
        made.append({"id": arxiv_id, "draft": relative, "pdf": repo_relative(pdf, REPO)})
        log(f"  + {arxiv_id} -> {draft_path.relative_to(REPO)}")

    state.save()
    emit({"materialized": made}, args.json)
    return EXIT_OK if made else EXIT_NO_CANDIDATES


# --------------------------------------------------------------------------
# summarize


def cmd_summarize(args):
    config = load_config()
    settings = dict(config.summarize)
    try:
        models = summarize_mod.model_chain(
            args.model or settings.get("model"),
            getattr(args, "fallback_model", None) or settings.get("fallback_model"),
        )
    except summarize_mod.SummarizeError as exc:
        log(str(exc))
        return EXIT_CONFIG
    state = State.load(STATE_PATH)

    targets = _summarize_targets(args, state)
    if not targets:
        log("요약할 초안이 없습니다.")
        emit({"summarized": []}, args.json)
        return EXIT_NO_CANDIDATES

    if args.backend == "none":
        log(f"backend=none — {len(targets)}편의 초안에 프롬프트만 둡니다.")
        emit({"summarized": [], "backend": "none"}, args.json)
        return EXIT_OK

    prompt = render.load_review_prompt()
    done, failed = [], []

    for arxiv_id, entry in targets:
        pdf = PDF_DIR / f"{arxiv_id}.pdf"
        draft_path = REPO / entry["draft_path"]
        if not pdf.exists():
            log(f"  ! {arxiv_id}: PDF 가 없습니다 ({pdf})")
            failed.append(arxiv_id)
            continue

        if args.backend == "task":
            task = summarize_mod.write_task_file(
                tasks_dir=TASKS_DIR,
                arxiv_id=arxiv_id,
                title=entry.get("title", ""),
                pdf_path=pdf,
                draft_path=draft_path,
                prompt_path=render.REVIEW_PROMPT,
                score=entry.get("score"),
                matched=entry.get("matched", {}),
            )
            log(f"  + {arxiv_id} -> {task.relative_to(REPO)}")
            done.append(arxiv_id)
            continue

        WORKDIR.mkdir(parents=True, exist_ok=True)
        try:
            body, used_model = summarize_mod.summarize_with_models(
                models=models,
                pdf_path=pdf,
                prompt=prompt,
                workdir=WORKDIR,
                output_path=WORKDIR / f"{arxiv_id}-review.md",
                variant=settings.get("variant"),
                timeout=settings.get("timeout_seconds", 900),
            )
        except summarize_mod.SummarizeError as exc:
            # 초안은 프롬프트만 있는 상태로 그대로 둔다. 지우지 않는다.
            log(f"  ! {arxiv_id}: 요약 실패 — {exc}")
            state.mark_failed(arxiv_id, reason=f"summarize: {exc}")
            failed.append(arxiv_id)
            continue

        category = render.model_category(used_model)
        text = render.build_draft(
            title=entry.get("title") or arxiv_id,
            arxiv_id=arxiv_id,
            version=entry.get("version"),
            date=date.today().isoformat(),
            categories=["paper-review"] + ([category] if category else []),
            body=body,
        )
        render.write_draft(draft_path, text)
        log(f"  + {arxiv_id} 요약 완료 -> {draft_path.relative_to(REPO)}")
        done.append(arxiv_id)

    state.save()
    emit({"summarized": done, "failed": failed, "backend": args.backend}, args.json)
    if failed and not done:
        return EXIT_SUMMARIZE
    return EXIT_OK


def _summarize_targets(args, state):
    if args.id:
        entry = state.entries.get(args.id)
        return [(args.id, entry)] if entry else []
    return [
        (paper_id, entry)
        for paper_id, entry in sorted(state.entries.items())
        if entry.get("status") == "materialized"
        and entry.get("materialized_at") == date.today().isoformat()
    ]


# --------------------------------------------------------------------------
# run


def cmd_run(args):
    for step in (cmd_fetch, cmd_score, cmd_materialize):
        code = step(args)
        if code != EXIT_OK:
            return code
    return cmd_summarize(args)


# --------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="LLM 추론 최적화 논문 수집·선별·초안 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="종료 코드: 0 성공 / 2 후보 없음 / 3 네트워크 실패 / 4 설정 오류 / 5 요약 실패",
    )
    parser.add_argument("--json", action="store_true", help="stdout 에 JSON 만 출력")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_window(p):
        p.add_argument("--date", help="이 날짜 하루만 (YYYY-MM-DD)")
        p.add_argument("--since", default="3d", help="최근 N일 (기본 3d)")
        p.add_argument("--source", default="hf,arxiv", help="hf,arxiv 중 골라서")
        p.add_argument("--hf-delay", type=float, default=1.0)
        p.add_argument("--arxiv-max", type=int, default=200)

    p_fetch = sub.add_parser("fetch", help="논문 후보 수집")
    add_window(p_fetch)
    p_fetch.set_defaults(func=cmd_fetch)

    p_score = sub.add_parser("score", help="후보 채점 (네트워크 안 씀)")
    p_score.add_argument("--input", help="raw jsonl 경로 (기본: 가장 최근)")
    p_score.set_defaults(func=cmd_score)

    p_mat = sub.add_parser("materialize", help="PDF 다운로드 + 초안 생성")
    p_mat.add_argument("--input", help="scored jsonl 경로 (기본: 가장 최근)")
    p_mat.add_argument("--top-k", type=int, help="설정의 top_k 를 덮어씀")
    p_mat.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 목록만")
    p_mat.add_argument("--pdf-delay", type=float, default=3.0)
    p_mat.set_defaults(func=cmd_materialize)

    p_sum = sub.add_parser("summarize", help="요약 본문 채우기")
    p_sum.add_argument("--backend", choices=("none", "opencode", "task"), default="none")
    p_sum.add_argument("--id", help="이 arXiv ID 하나만")
    p_sum.add_argument("--model", help="설정의 summarize.model 을 덮어씀")
    p_sum.add_argument(
        "--fallback-model", help="설정의 summarize.fallback_model 을 덮어씀"
    )
    p_sum.set_defaults(func=cmd_summarize)

    p_run = sub.add_parser("run", help="fetch → score → materialize → summarize")
    add_window(p_run)
    p_run.add_argument("--input", default=None, help=argparse.SUPPRESS)
    p_run.add_argument("--top-k", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--pdf-delay", type=float, default=3.0)
    p_run.add_argument("--backend", choices=("none", "opencode", "task"), default="none")
    p_run.add_argument("--id", default=None, help=argparse.SUPPRESS)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--fallback-model", default=None)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        log(str(exc))
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
