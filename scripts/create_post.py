#!/usr/bin/env python3
"""arXiv PDF 파일명(= arXiv ID)으로부터 Hugo 논문 리뷰 초안을 만든다.

사용법:
    python3 scripts/create_post.py <pdf 폴더> [-o content/posts]

<pdf 폴더> 안의 `2501.17811v1.pdf` 같은 파일마다 arXiv API 에서 제목을 받아
`content/posts/<날짜>-paper-<arXiv ID>.md` 초안을 만든다. 본문에는 논문을 LLM 에
질의할 때 쓰는 프롬프트 모음(data/paper-prompts/paper-review.md)이 함께 들어간다.

표준 라이브러리만 사용한다.
"""

import argparse
import datetime
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query?id_list={}"
ATOM = "{http://www.w3.org/2005/Atom}"


def get_arxiv_title(arxiv_id, timeout=20):
    """arXiv Atom 피드에서 논문 제목을 가져온다. 실패하면 None."""
    try:
        with urllib.request.urlopen(ARXIV_API.format(arxiv_id), timeout=timeout) as r:
            body = r.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  ! arXiv 요청 실패 ({arxiv_id}): {e}", file=sys.stderr)
        return None
    try:
        feed = ET.fromstring(body)
    except ET.ParseError as e:
        print(f"  ! 응답 파싱 실패 ({arxiv_id}): {e}", file=sys.stderr)
        return None
    # 피드의 첫 <title> 은 질의 자체이므로 <entry> 안의 제목을 쓴다.
    entry = feed.find(f"{ATOM}entry")
    if entry is None:
        return None
    title = entry.findtext(f"{ATOM}title") or ""
    title = " ".join(title.split())  # 줄바꿈/중복 공백 정리
    return title or None


def yaml_quote(s):
    """YAML 스칼라로 안전하게 감싼다."""
    return "'" + s.replace("'", "''") + "'"


def front_matter(title, arxiv_id, date, extra_categories=()):
    cats = ["paper-review", *extra_categories]
    lines = [
        "---",
        f"title: {yaml_quote(title)}",
        f'date: "{date}"',
        "draft: true",
        'description: ""',
        "categories:",
        *(f"- {c}" for c in cats),
        "tags:",
        f"- {arxiv_id}",
        "---",
        "",
        f"[논문 링크](https://arxiv.org/abs/{arxiv_id})",
        "",
    ]
    return "\n".join(lines)


# 질의 프롬프트는 papers.py 의 요약 백엔드들과 공유해야 하므로 코드가
# 아니라 데이터로 둔다.
PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "paper-prompts", "paper-review.md",
)

with open(PROMPT_PATH, encoding="utf-8") as _f:
    PROMPT_TEMPLATE = _f.read()


def main():
    parser = argparse.ArgumentParser(
        description="arXiv PDF 파일명에서 Hugo 논문 리뷰 초안을 생성합니다."
    )
    parser.add_argument("input_folder", help="arXiv ID 이름의 PDF 가 있는 폴더")
    parser.add_argument(
        "-o", "--output-folder", default="content/posts",
        help="마크다운을 저장할 폴더 (기본: content/posts)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="이미 있는 파일도 덮어씁니다.",
    )
    parser.add_argument(
        "-c", "--category", action="append", default=[], metavar="NAME",
        help="paper-review 외에 추가할 카테고리. 여러 번 쓸 수 있습니다 "
             "(예: -c with-gpt-5.2). 어떤 LLM 으로 정리했는지 기록하는 용도.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        parser.error(f"입력 폴더가 없습니다: {args.input_folder}")
    os.makedirs(args.output_folder, exist_ok=True)

    today = datetime.date.today().isoformat()
    pdfs = sorted(f for f in os.listdir(args.input_folder) if f.lower().endswith(".pdf"))
    if not pdfs:
        print(f"{args.input_folder} 에 PDF 가 없습니다.")
        return 0

    created = skipped = failed = 0
    for filename in pdfs:
        arxiv_id = os.path.splitext(filename)[0]
        if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id):
            print(f"- {filename}: arXiv ID 형식이 아니라 건너뜁니다.")
            skipped += 1
            continue

        out_path = os.path.join(args.output_folder, f"{today}-paper-{arxiv_id}.md")
        if os.path.exists(out_path) and not args.force:
            print(f"- {arxiv_id}: 이미 있습니다 ({out_path}). --force 로 덮어쓸 수 있습니다.")
            skipped += 1
            continue

        title = get_arxiv_title(arxiv_id)
        if not title:
            print(f"- {arxiv_id}: 제목을 가져오지 못해 건너뜁니다.")
            failed += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(front_matter(title, arxiv_id, today, args.category))
            f.write(PROMPT_TEMPLATE)
        print(f"+ {arxiv_id}: {out_path}")
        created += 1

    print(f"\n생성 {created} / 건너뜀 {skipped} / 실패 {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
