"""실제로 리뷰했던 논문이 이 규칙을 통과하는지 확인한다.

어휘집이나 가중치를 만질 때 회귀를 잡는 안전망이다. 픽스처는 arXiv API
응답이라 HF 큐레이션 신호가 없다 — 즉 arXiv 단독 게이트로 판정된다.
"""

import json
from pathlib import Path

import pytest

from paperlib.scoring import FilterConfig, passes_gate, score_paper

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = Path(__file__).parent.parent / "data" / "paper-filter.yaml"

# 이 블로그의 표적 그 자체인 논문들. 하나라도 떨어지면 규칙이 망가진 것이다.
MUST_PASS = {
    "2411.19379": "Marconi: Prefix Caching",
    "2411.02820": "DroidSpeak: KV Cache Sharing",
    "2312.07104": "SGLang",
    "2506.19852": "Radial Attention (비디오지만 sparse attention 이 본체)",
    "2505.21487": "Hardware-Efficient Attention for Fast Decoding",
    "2508.08448": "GPU Multitasking for LLM serving",
}


@pytest.fixture(scope="module")
def config():
    return FilterConfig.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def scored(config):
    records = json.loads((FIXTURES / "golden_reviewed.json").read_text(encoding="utf-8"))
    return {r["id"]: score_paper(r, config) for r in records}


@pytest.mark.parametrize("paper_id", sorted(MUST_PASS))
def test_core_target_paper_passes_the_gate(paper_id, scored, config):
    result = scored[paper_id]

    assert passes_gate(result, config), (
        f"{MUST_PASS[paper_id]} 가 떨어졌다 — "
        f"topic={result.components['topic']:.3f} core_hits={result.core_hits} "
        f"matched={result.matched}"
    )


def test_hard_exclude_does_not_drop_the_video_sparse_attention_paper(scored, config):
    from paperlib.scoring import select

    radial = scored["2506.19852"]

    assert "video generation" in radial.record["abstract"].lower()
    assert radial.core_hits > 0, "core 용어가 있어야 제외 규칙이 무력화된다"
    assert [p.id for p in select([radial], config)] == ["2506.19852"]


def test_off_topic_papers_are_rejected(config):
    """반대 방향 — 규칙이 헐거워지면 여기서 잡힌다."""
    off_topic = [
        ("Photon: Federated LLM Pre-Training", "Federated pre-training across silos."),
        ("A Risk-cognizant Imitation Agent for vCPU Oversubscription", "VM allocation."),
        ("LLaDA-Image: Building Strong Image Generators", "Image generation with diffusion."),
        ("Supply-Chain Attacks in Machine Learning Frameworks", "We audit ML supply chains."),
        ("SwiftVI: Time-Efficient Planning with MDPs", "Value iteration for MDPs."),
    ]

    for index, (title, abstract) in enumerate(off_topic):
        result = score_paper(
            {
                "id": f"off-{index}",
                "title": title,
                "abstract": abstract,
                "primary_category": "cs.LG",
                "sources": ["arxiv"],
                "hf": None,
            },
            config,
        )
        assert not passes_gate(result, config), (
            f"관심 밖 논문이 통과했다: {title} "
            f"(topic={result.components['topic']:.3f} matched={result.matched})"
        )


def test_most_of_the_golden_set_survives(scored, config):
    passed = [s for s in scored.values() if passes_gate(s, config)]

    assert len(passed) >= 10, (
        "12 편 중 10 편 미만 통과 — 규칙이 너무 빡빡하다.\n"
        + "\n".join(
            f"  {'PASS' if passes_gate(s, config) else 'FAIL'} {s.id} "
            f"topic={s.components['topic']:.3f} core={s.core_hits} "
            f"{s.record['title'][:50]}"
            for s in sorted(scored.values(), key=lambda x: -x.components["topic"])
        )
    )
