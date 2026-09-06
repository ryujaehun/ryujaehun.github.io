from paperlib.scoring import Tier, Vocabulary, topic_score


def test_term_matches_only_at_word_boundaries():
    vocab = Vocabulary([Tier(name="core", weight=3.0, terms=["int8"])])

    assert vocab.match("print8 and sprint8 quantization") == []
    assert [m.term for m in vocab.match("an int8 kernel")] == ["int8"]


def test_overlapping_match_keeps_only_the_heaviest_term():
    vocab = Vocabulary(
        [
            Tier(name="core", weight=3.0, terms=["sparse attention"]),
            Tier(name="adjacent", weight=1.5, terms=["attention"]),
        ]
    )

    matches = vocab.match("Sparse Attention for long context")

    assert [m.term for m in matches] == ["sparse attention"]


def test_phrase_term_matches_space_hyphen_and_underscore_forms():
    vocab = Vocabulary([Tier(name="core", weight=3.0, terms=["kv cache"])])

    for text in ("KV cache eviction", "kv-cache eviction", "kv_cache eviction"):
        assert [m.term for m in vocab.match(text)] == ["kv cache"], text


def test_hyphenated_vocabulary_term_matches_spaced_form():
    vocab = Vocabulary([Tier(name="adjacent", weight=1.5, terms=["mixture-of-experts"])])

    assert [m.term for m in vocab.match("Mixture of Experts at scale")] == [
        "mixture-of-experts"
    ]


CORE_ONLY = Vocabulary([Tier(name="core", weight=3.0, terms=["kv cache"])])
FIELD_WEIGHTS = {"title": 2.0, "keywords": 1.5, "abstract": 1.0}


def test_topic_score_weights_title_higher_than_abstract():
    in_title = topic_score(
        title="KV cache eviction",
        keywords=[],
        abstract="nothing relevant here",
        vocab=CORE_ONLY,
        field_weights=FIELD_WEIGHTS,
        saturation_k=6.0,
    )
    in_abstract = topic_score(
        title="nothing relevant here",
        keywords=[],
        abstract="we revisit KV cache eviction",
        vocab=CORE_ONLY,
        field_weights=FIELD_WEIGHTS,
        saturation_k=6.0,
    )

    assert in_title.raw == 6.0          # tier 3.0 x title 2.0
    assert in_abstract.raw == 3.0       # tier 3.0 x abstract 1.0
    assert in_title.score > in_abstract.score


def test_topic_score_counts_core_hits_across_fields_once_per_term():
    result = topic_score(
        title="KV cache eviction",
        keywords=["kv cache"],
        abstract="kv cache again and kv cache again",
        vocab=CORE_ONLY,
        field_weights=FIELD_WEIGHTS,
        saturation_k=6.0,
    )

    # 필드마다 1회씩 걸리므로 raw 는 세 필드 가중의 합
    assert result.raw == 3.0 * (2.0 + 1.5 + 1.0)
    # 서로 다른 core 용어는 하나뿐이다
    assert result.core_hits == 1
