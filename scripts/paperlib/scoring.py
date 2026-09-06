"""논문 후보 스코어링. 네트워크를 타지 않는 순수 함수 모음."""

import math
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    weight: float
    terms: list = field(default_factory=list)


@dataclass(frozen=True)
class Match:
    term: str
    tier: str
    weight: float
    span: tuple


def _term_pattern(term):
    """용어를 단어 경계로 감싼 정규식으로 바꾼다.

    용어와 본문 양쪽에서 공백/하이픈/언더스코어를 같은 구분자로 본다.
    'kv cache' 하나로 'kv-cache', 'kv_cache' 를 잡고,
    'mixture-of-experts' 하나로 'mixture of experts' 를 잡는다.
    """
    parts = [re.escape(p) for p in re.split(r"[\s\-_]+", term.lower()) if p]
    return re.compile(r"\b" + r"[\s\-_]+".join(parts) + r"\b")


class Vocabulary:
    def __init__(self, tiers):
        self.tiers = list(tiers)

    def match(self, text):
        """겹치지 않는 매치를 가중 내림차순으로 고른다. 용어당 최대 1회."""
        lowered = text.lower()
        candidates = []
        for tier in self.tiers:
            for term in tier.terms:
                for m in _term_pattern(term).finditer(lowered):
                    candidates.append(
                        Match(term=term, tier=tier.name, weight=tier.weight, span=m.span())
                    )

        # 무거운 것 먼저, 같은 가중이면 긴 것 먼저 — 짧은 용어가 긴 용어를
        # 밀어내지 않게 한다.
        candidates.sort(key=lambda m: (-m.weight, -(m.span[1] - m.span[0]), m.span[0]))

        taken_spans = []
        seen_terms = set()
        accepted = []
        for cand in candidates:
            if cand.term in seen_terms:
                continue
            start, end = cand.span
            if any(start < s_end and s_start < end for s_start, s_end in taken_spans):
                continue
            taken_spans.append(cand.span)
            seen_terms.add(cand.term)
            accepted.append(cand)
        return accepted


@dataclass(frozen=True)
class TopicResult:
    score: float
    raw: float
    core_hits: int
    matched: dict


def topic_score(title, keywords, abstract, vocab, field_weights, saturation_k):
    """제목/키워드/초록에서 어휘를 찾아 포화된 0~1 점수를 낸다."""
    fields = {
        "title": title or "",
        "keywords": " . ".join(keywords or []),
        "abstract": abstract or "",
    }

    raw = 0.0
    matched = {}
    for name, text in fields.items():
        weight = field_weights.get(name, 0.0)
        if not weight or not text:
            continue
        for m in vocab.match(text):
            raw += m.weight * weight
            matched.setdefault(m.tier, set()).add(m.term)

    matched = {tier: sorted(terms) for tier, terms in matched.items()}
    return TopicResult(
        score=1.0 - math.exp(-raw / saturation_k),
        raw=raw,
        core_hits=len(matched.get("core", [])),
        matched=matched,
    )


@dataclass(frozen=True)
class FilterConfig:
    weights: dict
    field_weights: dict
    saturation_k: float
    buzz_reference: float
    stars_reference: float
    gates: dict
    preferred_categories: tuple
    non_preferred_multiplier: float
    vocabulary: Vocabulary
    exclude_terms: tuple
    organizations: tuple
    summarize: dict

    @classmethod
    def from_dict(cls, raw):
        tiers = [
            Tier(name=t["name"], weight=float(t["weight"]), terms=list(t.get("terms", [])))
            for t in raw.get("tiers", [])
        ]
        if not tiers:
            raise ValueError("설정에 tiers 가 비어 있습니다.")
        for tier in tiers:
            if tier.weight <= 0:
                raise ValueError(f"티어 '{tier.name}' 의 가중이 0 이하입니다.")
            if not tier.terms:
                raise ValueError(f"티어 '{tier.name}' 에 용어가 없습니다.")

        return cls(
            weights=dict(raw["weights"]),
            field_weights=dict(raw["field_weights"]),
            saturation_k=float(raw["saturation_k"]),
            buzz_reference=float(raw["buzz_reference"]),
            stars_reference=float(raw["stars_reference"]),
            gates=dict(raw["gates"]),
            preferred_categories=tuple(raw.get("preferred_categories", [])),
            non_preferred_multiplier=float(raw.get("non_preferred_multiplier", 1.0)),
            vocabulary=Vocabulary(tiers),
            exclude_terms=tuple(raw.get("exclude_terms", [])),
            organizations=tuple(o.lower() for o in raw.get("organizations", [])),
            summarize=dict(raw.get("summarize", {})),
        )

    @classmethod
    def load(cls, path):
        import yaml

        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass(frozen=True)
class ScoredPaper:
    id: str
    score: float
    components: dict
    core_hits: int
    matched: dict
    sources: tuple
    record: dict


def _log_ratio(value, reference):
    if value <= 0 or reference <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(reference))


def _organization_name(hf):
    org = (hf or {}).get("organization")
    if isinstance(org, dict):
        org = org.get("name") or org.get("fullname") or ""
    return (org or "").lower()


def score_paper(record, config):
    hf = record.get("hf") or {}

    topic = topic_score(
        title=record.get("title"),
        keywords=hf.get("ai_keywords") or [],
        abstract=record.get("abstract"),
        vocab=config.vocabulary,
        field_weights=config.field_weights,
        saturation_k=config.saturation_k,
    )

    buzz = _log_ratio(hf.get("upvotes") or 0, config.buzz_reference)

    has_repo = 0.5 if hf.get("github_repo") else 0.0
    impl = min(
        1.0, has_repo + 0.5 * _log_ratio(hf.get("github_stars") or 0, config.stars_reference)
    )

    org = _organization_name(hf)
    venue = 1.0 if org and any(w in org for w in config.organizations) else 0.0

    components = {"topic": topic.score, "buzz": buzz, "impl": impl, "venue": venue}
    weighted = sum(config.weights[k] * v for k, v in components.items())

    if record.get("primary_category") not in config.preferred_categories:
        weighted *= config.non_preferred_multiplier

    return ScoredPaper(
        id=record["id"],
        score=weighted,
        components=components,
        core_hits=topic.core_hits,
        matched=topic.matched,
        sources=tuple(record.get("sources", [])),
        record=record,
    )


def _is_hard_excluded(scored, config):
    """제외 어휘가 제목에 있고 core 용어가 하나도 없을 때만 제외한다.

    도메인은 비디오지만 sparse attention 이 본체인 논문을 살리기 위한
    조건부 규칙이다.
    """
    if scored.core_hits > 0:
        return False
    title = (scored.record.get("title") or "").lower()
    return any(_term_pattern(term).search(title) for term in config.exclude_terms)


def passes_gate(scored, config):
    """출처별로 다른 게이트를 적용한다.

    arXiv 단독 논문은 buzz 가 0 이라 총점 게이트로는 영원히 통과하지 못한다.
    그래서 토픽만으로 판정한다.
    """
    gates = config.gates
    if "hf" in scored.sources:
        return scored.score >= gates["hf_min_score"]
    topic = scored.components["topic"]
    return topic >= gates["arxiv_min_topic"] or scored.core_hits >= gates["arxiv_min_core_hits"]


def select(scored_papers, config, excluded_ids=frozenset()):
    """게이트를 통과한 논문을 점수 내림차순으로 top_k 까지 고른다."""
    kept = [
        s
        for s in scored_papers
        if s.id not in excluded_ids
        and not _is_hard_excluded(s, config)
        and passes_gate(s, config)
    ]
    kept.sort(key=lambda s: (-s.score, s.id))
    return kept[: config.gates["top_k"]]
