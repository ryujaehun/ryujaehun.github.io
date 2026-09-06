import pytest
import json
from pathlib import Path

from paperlib.sources import merge, normalize_arxiv_id, parse_arxiv_atom, parse_hf_daily

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_arxiv_id_splits_the_version_suffix():
    assert normalize_arxiv_id("2609.03430v2") == ("2609.03430", "v2")
    assert normalize_arxiv_id("2609.03430") == ("2609.03430", None)
    assert normalize_arxiv_id("http://arxiv.org/abs/2506.19852v1") == ("2506.19852", "v1")


def test_normalize_arxiv_id_rejects_non_arxiv_strings():
    assert normalize_arxiv_id("not-an-id") == (None, None)


def test_parse_hf_daily_extracts_curation_signals():
    payload = json.loads((FIXTURES / "hf_daily.json").read_text(encoding="utf-8"))

    records = {r["id"]: r for r in parse_hf_daily(payload)}

    kv = records["2609.03430"]
    assert kv["sources"] == ["hf"]
    assert kv["title"].startswith("Random Attention")
    assert kv["hf"]["upvotes"] == 161
    assert kv["hf"]["github_stars"] == 31
    assert "KV cache compression" in kv["hf"]["ai_keywords"]
    assert kv["abstract"]


def test_parse_arxiv_atom_extracts_categories_and_dates():
    xml = (FIXTURES / "arxiv_query.xml").read_bytes()

    records = parse_arxiv_atom(xml)

    assert records
    first = records[0]
    assert first["id"] == "2609.04199"
    assert first["version"] == "v1"
    assert first["sources"] == ["arxiv"]
    assert first["primary_category"]
    assert first["published"].startswith("2026-09-03")


def test_merge_combines_both_sources_and_keeps_hf_signals():
    hf = [
        {
            "id": "2609.03430",
            "version": None,
            "title": "HF title",
            "abstract": "hf abstract",
            "authors": [],
            "primary_category": None,
            "categories": [],
            "published": "2026-09-03T00:00:00Z",
            "sources": ["hf"],
            "hf": {"upvotes": 161},
        }
    ]
    arxiv = [
        {
            "id": "2609.03430",
            "version": "v1",
            "title": "arXiv title",
            "abstract": "arxiv abstract",
            "authors": ["A"],
            "primary_category": "cs.LG",
            "categories": ["cs.LG"],
            "published": "2026-09-03T00:00:00Z",
            "sources": ["arxiv"],
            "hf": None,
        }
    ]

    merged = merge(hf + arxiv)

    assert len(merged) == 1
    record = merged[0]
    assert sorted(record["sources"]) == ["arxiv", "hf"]
    assert record["hf"]["upvotes"] == 161
    # 제목과 초록은 arXiv 를 신뢰한다
    assert record["title"] == "arXiv title"
    assert record["primary_category"] == "cs.LG"
    assert record["version"] == "v1"


class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_pdf_writes_the_file_and_returns_its_path(tmp_path):
    from paperlib.sources import download_pdf

    requested = []

    def opener(request, timeout=None):
        requested.append(request.full_url)
        return _FakeResponse(b"%PDF-1.7 fake")

    path = download_pdf("2506.19852", dest_dir=tmp_path, opener=opener)

    assert path.read_bytes() == b"%PDF-1.7 fake"
    assert path.name == "2506.19852.pdf"
    assert requested == ["https://arxiv.org/pdf/2506.19852"]


def test_download_pdf_rejects_a_response_that_is_not_a_pdf(tmp_path):
    from paperlib.sources import SourceError, download_pdf

    def opener(request, timeout=None):
        return _FakeResponse(b"<html>404 not found</html>")

    with pytest.raises(SourceError):
        download_pdf("2506.19852", dest_dir=tmp_path, opener=opener)

    assert list(tmp_path.iterdir()) == []


def test_within_window_accepts_a_record_published_inside_the_range():
    from paperlib.sources import within_window

    assert within_window({"published": "2026-09-04T12:00:00Z"}, ["2026-09-04"])


def test_within_window_rejects_a_record_published_after_a_single_day_window():
    from paperlib.sources import within_window

    # --date 2026-09-04 는 그 하루만 뜻해야 한다. 그 뒤 날짜까지 통과하면 안 된다.
    assert not within_window({"published": "2026-09-05T00:30:00Z"}, ["2026-09-04"])


def test_within_window_rejects_a_record_older_than_the_range():
    from paperlib.sources import within_window

    assert not within_window(
        {"published": "2026-09-01T00:00:00Z"}, ["2026-09-06", "2026-09-05"]
    )


def test_within_window_keeps_records_with_no_publication_date():
    from paperlib.sources import within_window

    assert within_window({"published": None}, ["2026-09-06"])
    assert within_window({"published": "garbage"}, ["2026-09-06"])
