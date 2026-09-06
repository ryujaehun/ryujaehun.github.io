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
        "min_substantive_hits": 1,
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


def test_config_exposes_fetch_categories_separately_from_preferred():
    cfg = config(fetch_categories=["cs.LG", "cs.DC"])

    # 무엇을 긁어올까(fetch)와 무엇에 감점하지 않을까(preferred)는 다른 목록이다
    assert cfg.fetch_categories == ("cs.LG", "cs.DC")
    assert cfg.preferred_categories == ("cs.LG", "cs.CL", "cs.DC")


def test_config_falls_back_to_preferred_when_fetch_categories_absent():
    assert config().fetch_categories == ("cs.LG", "cs.CL", "cs.DC")


def test_real_config_file_loads():
    from pathlib import Path

    from paperlib.scoring import FilterConfig

    cfg = FilterConfig.load(Path(__file__).parent.parent / "data" / "paper-filter.yaml")

    assert cfg.gates["top_k"] == 5
    assert cfg.fetch_categories
    assert cfg.summarize["model"].startswith("opencode-go/")



def _real_config():
    from pathlib import Path

    return FilterConfig.load(Path(__file__).parent.parent / "data" / "paper-filter.yaml")


def test_paper_without_a_substantive_term_is_rejected():
    """약한 용어가 쌓여 topic 이 높아져도, 실질 용어가 없으면 주제 밖이다.

    실제로 걸렸던 사례: 'SolarWM: Long-Horizon Video World Models' 가
    upvote 140 + topic 0.714 로 통과했다. 매칭은 autoregressive/distillation
    같은 adjacent 뿐이었다. 실제 어휘집으로 재현해야 의미가 있다.
    """
    cfg = _real_config()
    popular_off_topic = score_paper(
        paper(
            title="SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models",
            abstract=(
                "We introduce SolarWM, a fully open foundation for building interactive "
                "video world models from data preparation through long-horizon inference. "
                "Training across heterogeneous data sources and video backbones is "
                "challenging. We use autoregressive distillation."
            ),
            sources=["hf"],
            hf={"upvotes": 140, "github_repo": "https://x", "github_stars": 200},
        ),
        cfg,
    )

    assert popular_off_topic.components["buzz"] > 0.9, "인기 신호는 실제로 높다"
    assert popular_off_topic.substantive_hits == 0, "실질 용어가 하나도 없다"
    assert select([popular_off_topic], cfg) == []


def test_substantive_rule_does_not_block_a_relevant_popular_paper():
    cfg = _real_config()
    relevant = score_paper(
        paper(
            title="Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning",
            abstract="We revisit sparse attention and cache compression for vLLM serving.",
            sources=["hf"],
            hf={"upvotes": 140},
        ),
        cfg,
    )

    assert [p.id for p in select([relevant], cfg)] == [relevant.id]



def test_substantive_hits_count_core_named_and_systems_but_not_adjacent():
    cfg = _real_config()

    only_adjacent = score_paper(
        paper(title="Attention and caching in autoregressive models", sources=["arxiv"]), cfg
    )
    with_systems = score_paper(
        paper(title="NVFP4 quantization for language models", sources=["arxiv"]), cfg
    )

    assert only_adjacent.substantive_hits == 0
    assert with_systems.substantive_hits >= 1


def test_zero_upvote_paper_with_core_terms_beats_a_popular_off_topic_one():
    """실제 순위 뒤집힘 재현: upvote 0 인 표적 논문이 밀려나면 안 된다."""
    cfg = _real_config()

    target = score_paper(
        paper(
            id="2609.aaaa",
            title="SGD-KV: Summarization Guided KV Cache Compression",
            abstract="We compress the KV cache during decoding for LLM serving.",
            sources=["hf"],
            hf={"upvotes": 0},
        ),
        cfg,
    )
    popular = score_paper(
        paper(
            id="2609.bbbb",
            title="SolarWM: Scalable Training for Long-Horizon Video World Models",
            abstract="Interactive video world models with autoregressive distillation.",
            sources=["hf"],
            hf={"upvotes": 140},
        ),
        cfg,
    )

    assert [p.id for p in select([popular, target], cfg)] == ["2609.aaaa"]
