import copy

from paperlib.scoring import FilterConfig, score_paper, select

BASE_CONFIG = {
    "weights": {"topic": 0.55, "buzz": 0.25, "impl": 0.12, "venue": 0.08},
    "field_weights": {"title": 2.0, "keywords": 1.5, "abstract": 1.0},
    "saturation_k": 6.0,
    "buzz_reference": 150,
    "stars_reference": 500,
    "gates": {
        "hf_min_score": 0.45,
        "arxiv_min_topic": 0.70,
        "arxiv_min_core_hits": 2,
        "top_k": 5,
    },
    "preferred_categories": ["cs.LG", "cs.CL", "cs.DC"],
    "non_preferred_multiplier": 0.9,
    "tiers": [
        {"name": "core", "weight": 3.0, "terms": ["kv cache", "sparse attention"]},
        {"name": "adjacent", "weight": 1.5, "terms": ["attention"]},
    ],
    "exclude_terms": ["video generation"],
    "organizations": ["deepseek"],
    "summarize": {"model": "opencode-go/glm-5.3", "timeout_seconds": 900},
}


def config(**overrides):
    raw = copy.deepcopy(BASE_CONFIG)
    raw.update(overrides)
    return FilterConfig.from_dict(raw)


def paper(**overrides):
    record = {
        "id": "2609.03430",
        "version": "v1",
        "title": "A paper about nothing",
        "abstract": "",
        "authors": [],
        "primary_category": "cs.LG",
        "categories": ["cs.LG"],
        "published": "2026-09-03T00:00:00Z",
        "sources": ["arxiv"],
        "hf": None,
    }
    record.update(overrides)
    return record


def test_buzz_saturates_at_the_reference_upvote_count():
    at_reference = score_paper(
        paper(sources=["hf"], hf={"upvotes": 150, "github_repo": None, "github_stars": 0}),
        config(),
    )
    above_reference = score_paper(
        paper(sources=["hf"], hf={"upvotes": 900, "github_repo": None, "github_stars": 0}),
        config(),
    )

    assert at_reference.components["buzz"] == 1.0
    assert above_reference.components["buzz"] == 1.0


def test_impl_rewards_a_repository_and_its_stars():
    no_repo = score_paper(paper(sources=["hf"], hf={"upvotes": 0}), config())
    bare_repo = score_paper(
        paper(sources=["hf"], hf={"upvotes": 0, "github_repo": "https://x", "github_stars": 0}),
        config(),
    )
    starred = score_paper(
        paper(sources=["hf"], hf={"upvotes": 0, "github_repo": "https://x", "github_stars": 500}),
        config(),
    )

    assert no_repo.components["impl"] == 0.0
    assert bare_repo.components["impl"] == 0.5
    assert starred.components["impl"] == 1.0


def test_venue_matches_organization_whitelist_case_insensitively():
    hit = score_paper(
        paper(sources=["hf"], hf={"upvotes": 0, "organization": "DeepSeek-AI"}), config()
    )
    miss = score_paper(
        paper(sources=["hf"], hf={"upvotes": 0, "organization": "Some Lab"}), config()
    )

    assert hit.components["venue"] == 1.0
    assert miss.components["venue"] == 0.0


def test_non_preferred_primary_category_is_penalised():
    preferred = score_paper(paper(title="KV cache eviction"), config())
    other = score_paper(
        paper(title="KV cache eviction", primary_category="eess.SP"), config()
    )

    assert other.score == preferred.score * 0.9


def scored(**overrides):
    return score_paper(paper(**overrides), config())


def test_hf_paper_passes_on_total_score():
    strong = scored(
        title="Sparse Attention for KV cache eviction",
        sources=["hf"],
        hf={"upvotes": 150, "github_repo": "https://x", "github_stars": 500},
    )
    weak = scored(title="A paper about nothing", sources=["hf"], hf={"upvotes": 1})

    passed = select([strong, weak], config())

    assert [p.id for p in passed] == [strong.id]


def test_arxiv_only_paper_passes_on_core_hits_despite_failing_the_score_gate():
    # 본문에만 core 용어가 있어 총점은 낮지만, buzz 가 없는 arXiv 단독 논문을
    # 놓치지 않기 위해 core_hits 로 통과시켜야 한다.
    candidate = scored(
        title="Revisiting long sequence serving",
        abstract="We combine sparse attention with KV cache reuse.",
        sources=["arxiv"],
    )

    assert candidate.score < config().gates["hf_min_score"]
    assert candidate.core_hits == 2
    assert [p.id for p in select([candidate], config())] == [candidate.id]


def test_already_reviewed_ids_are_excluded():
    candidate = scored(
        title="Sparse Attention for KV cache eviction",
        sources=["hf"],
        hf={"upvotes": 150},
    )

    assert select([candidate], config(), excluded_ids={"2609.03430"}) == []


def test_hard_exclude_term_is_disarmed_when_a_core_term_is_present():
    off_topic = scored(title="Fast video generation with diffusion", sources=["hf"],
                       hf={"upvotes": 300})
    on_topic = scored(
        title="Sparse Attention for KV cache in video generation",
        sources=["hf"],
        hf={"upvotes": 300},
    )
    on_topic = score_paper({**on_topic.record, "id": "2609.99999"}, config())

    passed = {p.id for p in select([off_topic, on_topic], config())}

    assert passed == {"2609.99999"}


def test_top_k_caps_the_number_of_selected_papers():
    candidates = [
        score_paper(
            paper(
                id=f"2609.0{i}",
                title="Sparse Attention for KV cache eviction",
                sources=["hf"],
                hf={"upvotes": 100 + i},
            ),
            config(),
        )
        for i in range(9)
    ]

    passed = select(candidates, config())

    assert len(passed) == 5
    # 점수 내림차순이라 upvote 가 가장 높은 것이 앞선다
    assert passed[0].id == "2609.08"
