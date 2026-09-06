"""논문 소스 어댑터. HuggingFace Daily Papers 와 arXiv 신규 피드를 읽는다.

파싱과 병합은 순수 함수라 네트워크 없이 테스트한다. 실제 요청은
`fetch_hf_daily` / `fetch_arxiv` 가 맡고, opener 를 주입할 수 있다.
"""

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# 표준 ElementTree 는 entity 확장 공격에 취약하다. arXiv 는 신뢰할 만한
# 소스지만 방어 비용이 없으므로 defusedxml 을 쓴다.
from defusedxml import ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

HF_DAILY_URL = "https://huggingface.co/api/daily_papers?date={date}"
# http 는 301 을 돌려주고 본문이 비어 있다. https 를 써야 한다.
ARXIV_QUERY_URL = (
    "https://export.arxiv.org/api/query"
    "?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending"
    "&max_results={max_results}"
)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

USER_AGENT = "jaehun.me paper-pipeline (+https://jaehun.me)"


class SourceError(RuntimeError):
    """소스 한 곳을 읽지 못했을 때. 나머지 소스는 계속 진행한다."""


def normalize_arxiv_id(raw):
    """어떤 형태의 arXiv 참조에서든 (ID, 버전) 을 뽑는다.

    URL, 버전 접미사, 맨 ID 를 모두 받는다. arXiv ID 가 아니면 (None, None).
    """
    if not raw:
        return (None, None)
    m = ARXIV_ID_RE.search(str(raw))
    if not m:
        return (None, None)
    return (m.group(1), m.group(2))


def _blank_record(paper_id, version):
    return {
        "id": paper_id,
        "version": version,
        "title": "",
        "abstract": "",
        "authors": [],
        "primary_category": None,
        "categories": [],
        "published": None,
        "sources": [],
        "hf": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def parse_hf_daily(payload):
    """HF Daily Papers 응답을 정규화 레코드 목록으로 바꾼다."""
    records = []
    for entry in payload or []:
        paper = entry.get("paper") or {}
        paper_id, version = normalize_arxiv_id(paper.get("id"))
        if not paper_id:
            continue

        record = _blank_record(paper_id, version)
        record.update(
            {
                "title": " ".join((paper.get("title") or "").split()),
                "abstract": " ".join((paper.get("summary") or "").split()),
                "authors": [a.get("name", "") for a in paper.get("authors") or []],
                "published": paper.get("publishedAt"),
                "sources": ["hf"],
                "hf": {
                    "upvotes": paper.get("upvotes") or 0,
                    "num_comments": entry.get("numComments") or 0,
                    "ai_keywords": paper.get("ai_keywords") or [],
                    "github_repo": paper.get("githubRepo"),
                    "github_stars": paper.get("githubStars") or 0,
                    "organization": paper.get("organization"),
                    "submitted_at": paper.get("submittedOnDailyAt"),
                },
            }
        )
        records.append(record)
    return records


def parse_arxiv_atom(xml_bytes):
    """arXiv Atom 응답을 정규화 레코드 목록으로 바꾼다."""
    feed = ET.fromstring(xml_bytes)
    records = []
    for entry in feed.findall(f"{ATOM}entry"):
        paper_id, version = normalize_arxiv_id(entry.findtext(f"{ATOM}id"))
        if not paper_id:
            continue

        categories = [
            c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")
        ]
        primary = entry.find(f"{ARXIV_NS}primary_category")

        record = _blank_record(paper_id, version)
        record.update(
            {
                "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
                "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
                "authors": [
                    (a.findtext(f"{ATOM}name") or "").strip()
                    for a in entry.findall(f"{ATOM}author")
                ],
                "primary_category": (
                    primary.get("term") if primary is not None else (categories[0] if categories else None)
                ),
                "categories": categories,
                "published": entry.findtext(f"{ATOM}published"),
                "sources": ["arxiv"],
            }
        )
        records.append(record)
    return records


def merge(records):
    """같은 논문의 레코드를 하나로 합친다.

    제목·초록·카테고리는 arXiv 를 신뢰하고, 큐레이션 신호는 HF 에서 가져온다.
    """
    merged = {}
    for record in records:
        existing = merged.get(record["id"])
        if existing is None:
            merged[record["id"]] = dict(record)
            continue

        for source in record["sources"]:
            if source not in existing["sources"]:
                existing["sources"].append(source)

        if record.get("hf"):
            existing["hf"] = record["hf"]

        if "arxiv" in record["sources"]:
            # arXiv 쪽 서지 정보가 더 정확하다.
            for key in ("title", "abstract", "authors", "primary_category",
                        "categories", "published", "version"):
                if record.get(key):
                    existing[key] = record[key]
        else:
            for key in ("title", "abstract", "published"):
                if not existing.get(key) and record.get(key):
                    existing[key] = record[key]

    return list(merged.values())


def _read(url, opener=None, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceError(f"{url} 요청 실패: {exc}") from exc


def fetch_hf_daily(date, opener=None, timeout=30):
    """하루치 HF Daily Papers 를 읽는다. date 는 YYYY-MM-DD."""
    body = _read(HF_DAILY_URL.format(date=date), opener=opener, timeout=timeout)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SourceError(f"HF 응답 파싱 실패 ({date}): {exc}") from exc
    return parse_hf_daily(payload)


def fetch_arxiv(categories, max_results=200, opener=None, timeout=30, delay=3.0, sleep=time.sleep):
    """카테고리별 arXiv 신규 논문을 읽는다. 요청 간 delay 초를 쉰다."""
    records = []
    for index, category in enumerate(categories):
        if index:
            sleep(delay)
        body = _read(
            ARXIV_QUERY_URL.format(category=category, max_results=max_results),
            opener=opener,
            timeout=timeout,
        )
        records.extend(parse_arxiv_atom(body))
    return records
