"""Hugo 초안을 만든다.

머리말 생성은 기존 `create_post.py` 의 함수를 그대로 쓴다. 로직을 복제하지
않는다. 질의 프롬프트는 세 요약 백엔드가 공유해야 하므로 코드가 아니라
`data/paper-prompts/` 아래 데이터로 둔다.
"""

import os
import re
from pathlib import Path

import create_post

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = REPO_ROOT / "data" / "paper-prompts"
REVIEW_PROMPT = PROMPT_DIR / "paper-review.md"


def load_review_prompt(path=None):
    """논문 리뷰 질의 프롬프트 묶음을 읽는다."""
    return Path(path or REVIEW_PROMPT).read_text(encoding="utf-8")


def draft_filename(arxiv_id, version, date):
    """기존 글 이름 규칙을 그대로 따른다: <날짜>-paper-<id><버전>.md"""
    return f"{date}-paper-{arxiv_id}{version or ''}.md"


def model_category(model):
    """모델 이름을 카테고리 슬러그로 바꾼다.

    기존 글의 `with-gpt-5.2` 관례를 잇는다.
    'opencode-go/glm-5.3' -> 'with-glm-5-3'
    """
    if not model:
        return None
    name = model.split("/")[-1]
    return "with-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_draft(title, arxiv_id, version, date, categories, body=None, prompt_path=None):
    """초안 전문을 만든다. body 가 없으면 질의 프롬프트를 넣는다."""
    extra = [c for c in categories if c != "paper-review"]
    head = create_post.front_matter(title, f"{arxiv_id}{version or ''}", date, extra)
    tail = body if body is not None else load_review_prompt(prompt_path)
    return head + tail


def write_draft(path, text):
    """임시 파일에 쓰고 바꿔치기해 부분 상태를 남기지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def render_report(date, scored, selected_ids, failed_sources=()):
    """무엇이 왜 걸렸는지 남기는 마크다운 리포트.

    선정분과 탈락분을 함께 보여야 임계값을 데이터 보고 잡을 수 있다.
    """
    ordered = sorted(scored, key=lambda s: -s.score)
    lines = [
        f"# 논문 후보 리포트 {date}",
        "",
        f"수집 {len(scored)}편 / 선정 {len(selected_ids)}편",
        "",
    ]

    if failed_sources:
        lines += [
            f"> 읽지 못한 소스: {', '.join(failed_sources)} — 이 리포트는 불완전하다.",
            "",
        ]

    lines += [
        "| | ID | 점수 | topic | core | upvote | 출처 | 제목 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for s in ordered:
        picked = "**선정**" if s.id in selected_ids else ""
        upvotes = (s.record.get("hf") or {}).get("upvotes") or 0
        title = (s.record.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| {picked} | [{s.id}](https://arxiv.org/abs/{s.id}) "
            f"| {s.score:.3f} | {s.components['topic']:.3f} | {s.core_hits} "
            f"| {upvotes} | {'+'.join(s.sources)} | {title[:70]} |"
        )

    lines += ["", "## 선정된 논문의 매칭 근거", ""]
    for s in ordered:
        if s.id not in selected_ids:
            continue
        terms = "; ".join(
            f"{tier}: {', '.join(words)}" for tier, words in sorted(s.matched.items())
        )
        lines.append(f"- `{s.id}` — {terms or '(매칭 없음)'}")

    return "\n".join(lines) + "\n"
