from pathlib import Path

from paperlib.render import build_draft, draft_filename, load_review_prompt, model_category

REPO = Path(__file__).parent.parent


def test_review_prompt_is_loaded_from_the_data_directory():
    prompt = load_review_prompt()

    # create_post.py 가 쓰던 프롬프트 자산이 그대로 살아 있어야 한다
    assert "프롬프트 1.1.1" in prompt
    assert "마스터 프롬프트" in prompt
    assert "Evidence Tagging" in prompt


def test_draft_filename_keeps_the_existing_naming_convention():
    assert (
        draft_filename("2506.19852", "v1", "2026-09-06")
        == "2026-09-06-paper-2506.19852v1.md"
    )
    assert (
        draft_filename("2506.19852", None, "2026-09-06")
        == "2026-09-06-paper-2506.19852.md"
    )


def test_model_category_follows_the_with_gpt_convention():
    assert model_category("opencode-go/glm-5.3") == "with-glm-5-3"
    assert model_category("opencode-go/kimi-k3") == "with-kimi-k3"
    assert model_category(None) is None


def test_build_draft_writes_front_matter_link_and_prompt():
    text = build_draft(
        title="Radial Attention: Sparse Attention",
        arxiv_id="2506.19852",
        version="v1",
        date="2026-09-06",
        categories=["paper-review"],
        body=None,
    )

    assert text.startswith("---\n")
    assert "title: 'Radial Attention: Sparse Attention'" in text
    assert "draft: true" in text
    assert "- paper-review" in text
    assert "https://arxiv.org/abs/2506.19852v1" in text
    # body 가 없으면 질의 프롬프트가 들어간다
    assert "프롬프트 1.1.1" in text


def test_build_draft_replaces_the_prompt_when_a_body_is_given():
    text = build_draft(
        title="X",
        arxiv_id="2506.19852",
        version="v1",
        date="2026-09-06",
        categories=["paper-review", "with-glm-5-3"],
        body="## 한 줄 요약\n\n요약 본문입니다.\n",
    )

    assert "요약 본문입니다." in text
    assert "프롬프트 1.1.1" not in text
    assert "- with-glm-5-3" in text


def test_create_post_reads_its_prompt_from_the_shared_data_file():
    """프롬프트가 코드에 박혀 있으면 세 백엔드가 갈라진다."""
    import create_post

    source = (REPO / "scripts" / "create_post.py").read_text(encoding="utf-8")

    assert "프롬프트 1.1.1" not in source, "프롬프트가 아직 코드에 박혀 있다"
    assert create_post.PROMPT_TEMPLATE == load_review_prompt()


def _fake_scored(paper_id, score, topic, core_hits, title, sources=("hf",), upvotes=0):
    from paperlib.scoring import ScoredPaper

    return ScoredPaper(
        id=paper_id,
        score=score,
        components={"topic": topic, "buzz": 0.0, "impl": 0.0, "venue": 0.0},
        core_hits=core_hits,
        substantive_hits=core_hits,
        matched={"core": ["kv cache"]} if core_hits else {},
        sources=tuple(sources),
        record={"title": title, "hf": {"upvotes": upvotes}, "version": "v1"},
    )


def test_report_marks_selected_papers_and_lists_near_misses():
    from paperlib.render import render_report

    picked = _fake_scored("2609.001", 0.72, 0.90, 2, "KV cache eviction done right")
    missed = _fake_scored("2609.002", 0.31, 0.40, 0, "Something only mildly related")

    report = render_report(
        date="2026-09-06",
        scored=[picked, missed],
        selected_ids={"2609.001"},
        failed_sources=["arxiv"],
    )

    assert "2026-09-06" in report
    assert "2609.001" in report and "2609.002" in report
    # 선정된 것과 아닌 것이 구분되어야 사람이 임계값을 감으로 잡을 수 있다
    assert report.index("2609.001") < report.index("2609.002")
    assert "arxiv" in report  # 실패한 소스를 숨기지 않는다
