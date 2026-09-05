#!/usr/bin/env python3
"""arXiv PDF 파일명(= arXiv ID)으로부터 Hugo 논문 리뷰 초안을 만든다.

사용법:
    python3 scripts/create_post.py <pdf 폴더> [-o content/posts]

<pdf 폴더> 안의 `2501.17811v1.pdf` 같은 파일마다 arXiv API 에서 제목을 받아
`content/posts/<날짜>-paper-<arXiv ID>.md` 초안을 만든다. 본문에는 논문을 LLM 에
질의할 때 쓰는 프롬프트 모음이 함께 들어간다.

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


PROMPT_TEMPLATE = r"""


### 토글을 클릭하면 논문에 대한 자세한 LLM 질의응답 내용을 확인할 수 있습니다.

<details markdown="block">
<summary>▶️<strong>클릭하여 펼치기</strong></summary>


</details>
---


### [글로벌 규칙: Evidence Tagging]
- 모든 사실 주장 뒤에 (근거: §/Fig./Tab./Alg./Appx) 표기
- 직접 인용 ≤ 20단어, 나머지는 **숫자 중심** 요약
- 모든 수치에 단위 표기 (M/B Params, B Tokens, (P)FLOPs, ms, tokens/s, GB, $/1M tok)
- 마크다운을 사용시에 bold를 사용할때 뒤에있는 * 공백을 추가해야함
- katex는 $$이 있어야 rendering되는걸 유의해야함
---

## Key Numbers (요약)
- Params: __ B | Context: __ tokens | Vocab: __
- Architecture: Dense/MoE(Experts=__ / Active=__)
- Positional: RoPE/ALiBi/YaRN | Attention: Flash/GQA/Sliding/Long
- Pretrain Data: __ B tokens (mix: __% code/__% web/__% books/…)
- Train Compute: __ PF-days (est. FLOPs: __e__)
- **Serving**: TTFT __ ms | **TPOT** (ms/token) __ | Throughput __ tok/s
- KV-Cache: __ GB @ __ tokens | Peak VRAM: __ GB (bs=__, tp=__, pp=__)
- Cost: $__ / 1M tokens (bs=__, ctx=__) | Energy: __ kWh / 1M tok (PUE=__)

$$
\text{KV-Cache(GB)} \approx 
\frac{2 \cdot L \cdot H \cdot d_\text{head} \cdot \text{seq} \cdot \text{batch} \cdot \text{bytes/elt}}{10^9}
$$

> 용어: **TPOT = Time Per Output Token**. (필요 시 “TBT=TPOT”로 병기)

---

## SOTA 비교 (동일 세팅 필수)
| Benchmark | Metric | Setting (ctx/temp/top-p/stop/beam) | Model | Size | OOM | Result | Δ vs Best |
|---|---:|---|---|---:|---|---:|---:|
| MMLU | acc | 8k / 0.7 / 0.95 / ["

"] / 1 | Ours | 7B | Dense | 72.4 | -1.1 |
| GSM8K-CoT | acc | 32k / 0.2 / 0.95 / … | Ours | 7B | Dense | 82.1 | -3.4 |

---

## Compute & Cost (공통)
- Train: steps __ × bs __ × seq __ → tokens __ B; est. FLOPs = __e__  
- HW: __×GPU(__GB) | time __ days | PF-days __  
- Inference cost: $__ / 1M tok (provider __, region __, batch __, ctx __)

---

## Repro Checklist (공통)
- [ ] 코드/Commit/License  
- [ ] 데이터 스냅샷·필터링 규칙·라이선스  
- [ ] 하이퍼파라미터/시드/스케줄  
- [ ] 하드웨어/드라이버/라이브러리 버전  
- [ ] 평가 스크립트 & exact prompts

---

이 논문을 자주 자세하게 읽고 논문에 대한 자세한 리뷰를 작성하기 위해서 질의 응답을 한 후 이를 바탕으로 블로그 포스트를 작성할 것이다 이러한 목적과 github를 blog hugo의 narrow(https://github.com/tom2almighty/hugo-narrow)테마를 사용하고 있다는 점을 염두해서 markdown 형식으로 작성한다. 직접적인 블로그 작성 팁을 답변에 포함하지 않아야한다.
그리고 답변이 블로그에 포스팅의 일부라는 점을 염두에 두고 작성한다.
마크다운 작성시에 구조에 맞게 마크업을 잘 사용하고 Katex Mermaid 를 사용할 수 있다는점을 고려해서 답변을 해주었으면 하고 특별한 지시가 없다면 한글로 작성바란다.


## 프롬프트 1.1.1 (연구의 공백)

```
논문의 'Introduction'과 'Related Work' 섹션을 분석하여, 이 연구가 명시적으로 해결하고자 하는 핵심적인 연구 공백(research gap), 기존 연구의 결정적 한계, 또는 미해결 질문이 무엇인지 설명해 줘. 저자들이 설명하는, 이 논문 출판 시점의 '최신 기술(state of the art)'은 어떤 상태였는지 요약해 줘.
```

## 프롬프트 1.1.2 (핵심 가설)

```
이 논문의 중심 가설(central hypothesis) 또는 핵심 주장은 무엇인가? '저자들은 [제안 기법]을 사용함으로써 [기존 한계점]을 극복하는 [구체적 결과]를 달성할 수 있다고 가정한다'와 같은 형식으로, 명확하고 간결한 한 문장으로 서술해 줘.
```

## 프롬프트 1.2.1 (독창성 식별)

```
논문 전체를 바탕으로, 가장 중요하고 독창적인 기여(contribution) 1~3가지를 구별되는 항목으로 나열해 줘. 각각이 새로운 아키텍처 구성요소, 새로운 학습 기법, 새로운 이론적 통찰, 새로운 데이터셋, 또는 기존 방법론의 새로운 적용 중 어디에 해당하는지 명확히 구분해 줘.
```

## 프롬프트 1.2.2 (저자 관점에서의 강점)

```
저자들의 관점에서, 자신들의 접근법이 이전 방법들보다 우월한 이유는 무엇인가? 그들이 자신들의 연구가 지닌 독창성과 강점을 뒷받침하기 위해 사용하는 핵심 논거를 인용하거나 알기 쉽게 설명해 줘.
```

## 프롬프트 1.3.1 (알고리즘 단계별 설명)

```
핵심 알고리즘, 모델 아키텍처, 또는 주요 방법론을 단계별(step-by-step)로 설명해 줘. 독자는 AI 분야의 대학원생 수준이라고 가정해. 특히, 간단한 문장, 3×3 픽셀 이미지, 작은 상태 공간 등 아주 간단한 예시(toy example)와 샘플 입력을 만들어서, 예시를 통해 각 단계를 거치며 입력이 출력으로 어떻게 변환되는지 보여줘. 등장하는 모든 핵심 용어와 변수는 그 즉시 정의해 줘.
```

## 프롬프트 1.3.2 ('비밀 병기' 식별)

```
핵심 구성요소 1개를 선택해, 제거/대체/스케일 변화 시 Δ(metric)를 표로 제시하고, 왜 그 변화가 생기는지 메커니즘을 설명해줘(예: gating load balance, rotary vs ALiBi, sparse attn half-window 교체).
```

## 프롬프트 1.4.1 (핵심 결과 분석)

```
'Experiments' 또는 'Results'의 표/그림을 포함한 주요 결과를 분석해 줘. 핵심 성능 지표는 무엇인가? 어떤 벤치마크에서 보고되었는가? 저자들이 성공 증거로 가장 강조하는 결과를 요약해 줘.
```

## 프롬프트 1.4.2 (비판적 비교)

```
제안된 방법론은 논문에서 언급된 주요 베이스라인 및 SOTA 모델들과 비교하여 어떤 성능을 보이는가? 우월성 주장을 가장 강력하게 뒷받침하는 특정 비교 지점을 식별해 줘. 반대로, 능가하지 못했거나 개선이 미미했던 결과가 있다면 이유를 정리해 줘.
```

## 프롬프트 1.5.1 (언급된 한계와 잠재적 한계)

```
저자들이 명시적으로 인정한 한계/약점/실패 사례는 무엇인가? 분석을 바탕으로 잠재적 한계(강한 가정, 확장성, 연산 비용, 일반화 한계, 사회적 영향 등)는 무엇이라고 보나?
```


## 프롬프트 1.5.2 (미래 연구 궤적)

```
저자들이 제안하는 향후 연구 방향은 무엇인가? 한계에 비추어 합리적인 다음 단계나 대안적 방향을 제안해 줘.
```


# 주제별 추가 질문

## 모듈 A: 컴퓨터 비전 (cs.CV) 논문용

## 프롬프트 데이터 및 전처리

```
학습 및 추론에 사용된 이미지 해상도는 얼마인가? 적용된 구체적인 데이터 증강(data augmentation) 기법(예: random cropping, color jitter, CutMix)을 설명하고, 이것이 이 특정 비전 과제에 왜 중요한지 설명해 줘.
```

## 프롬프트 모델 아키텍처

```
사용된 백본 아키텍처는 무엇인가(예: ResNet, ViT, ConvNeXt)? 공간적 특징(spatial features)은 어떻게 추출되고 융합되는가? 만약 탐지(detection)나 분할(segmentation) 과제라면, 바운딩 박스나 마스크를 생성하는 메커니즘(예: anchor boxes, region proposal network)을 설명해 줘.
```

## 프롬프트 평가 및 지표

```
정확도(accuracy) 외에 어떤 다른 지표가 사용되었는가(예: 탐지를 위한 mAP, 분할을 위한 IoU)? 시각적 결과에 대한 정성적 분석이 있는가? 있다면, 모델이 인상적으로 성공한 예시와 실패한 예시를 하나씩 설명해 줘.
```

## 모듈 B: 자연어 처리 (cs.CL) 논문용

## 프롬프트 데이터 및 전처리

```
어떤 토큰화(tokenization) 전략이 사용되었는가(예: BPE, WordPiece, SentencePiece)? 어휘에 없는 단어(Out-of-vocabulary words)는 어떻게 처리되는가? 텍스트 정제 및 정규화 단계를 설명해 줘.
```

## 프롬프트 모델 아키텍처

```
트랜스포머를 사용했다면, 어텐션 메커니즘의 구성(예: 헤드 수, 레이어 수)을 상세히 설명해 줘. 위치 인코딩(positional encodings)은 어떻게 처리되는가? Seq2Seq 모델이라면, 인코더-디코더 상호작용을 설명해 줘.
```

## 프롬프트 학습 및 최적화

```
구체적인 언어 모델링 목표(objective)는 무엇인가(예: Causal LM, Masked LM, Prefix LM)? 모델은 어떤 코퍼스로 사전학습(pre-trained)되었는가? 다운스트림 과제를 위한 파인튜닝(fine-tuning) 전략을 설명해 줘.
```

## 모듈 C: 강화 학습 (cs.LG/cs.AI) 논문용

## 프롬프트 모델 아키텍처/알고리즘

```
상태 공간(State Space), 행동 공간(Action Space), 보상 함수(Reward Function)를 정의해 줘. 모델은 온-폴리시(on-policy)인가 오프-폴리시(off-policy)인가? 가치 기반(예: DQN), 정책 기반(예: REINFORCE), 또는 액터-크리틱(예: A2C, PPO) 중 어느 유형인가? 정책 및/또는 가치 함수에 대한 핵심 업데이트 규칙을 설명해 줘.
```

## 프롬프트 학습 및 최적화

```
탐험(exploration)과 활용(exploitation)의 트레이드오프는 어떻게 관리되는가(예: epsilon-greedy, entropy regularization)? 리플레이 버퍼(replay buffer)가 사용되었는가? 학습에 사용된 시뮬레이션 환경이나 실제 환경 설정을 설명해 줘.
```

## 프롬프트 평가 및 지표

```
성능은 어떻게 측정되는가(예: 누적 보상, 에피소드 길이, 성공률)? 평가에 얼마나 많은 에피소드나 타임스텝이 사용되었는가? 시간에 따른 성능을 보여주는 학습 곡선(learning curves)이 있는가?
```

## 모듈 D: 시스템 및 구현 (cs.DC, cs.AR 등) 논문용

## 프롬프트 구현 및 자원

```
"핵심 소프트웨어 의존성(예: CUDA, MPI, 특정 라이브러리)은 무엇인가? 학습 및 추론 중 예상되는 메모리 점유량(GPU의 경우 VRAM, CPU의 경우 RAM)은 얼마인가? 명시된 하드웨어에서의 처리량(throughput)은 얼마인가(예: images/sec, tokens/sec)? 총 연산 비용(예: 총 FLOPs 또는 Petaflop-days)에 대한 상세한 내역을 제공해 줘."
```

## 프롬프트 평가 및 지표

```
"평가를 위한 주요 지표는 무엇인가: 지연 시간(latency), 처리량(throughput), 전력 소비, 또는 비용 대비 성능? 시스템은 더 많은 데이터, 사용자, 또는 컴퓨팅 노드에 따라 어떻게 확장되는가?"
```

# 정리를 위한 마스터 프롬프트

"당신은 저명한 AI 블로그의 전문 AI 연구원이자 기술 작가입니다. 당신의 독자는 AI 실무자, 연구원, 학생들로 구성되어 있습니다. 답변은 제공받은 논문의 내용과 질의응 답을 기반으로 작성하면 됩니다. 당신의 임무는 이 모든 정보를 하나의 일관되고 잘 구조화된 블로그 포스트로 종합하는 것입니다. 포스트는 반드시 마크다운으로 작성해야 합니다.
아래의 구조를 정확히 따르세요:

포스팅의 제목
한 줄 요약 (TL;DR)
핵심 아이디어
배경: 그들이 해결한 문제
새로운 접근법: Method Name
작동 원리: 구체적인 예시로 살펴보기
성능 검증: 주요 결과
우리의 관점: 강점, 한계, 그리고 이 연구가 중요한 이유
다음 단계는?: 앞으로의 길
비판적이면서도 공정한 어조를 유지하세요. 복잡한 개념을 명확하고 간결하게 설명하세요. 섹션 간의 전환이 자연스럽도록 하세요. 단순히 답변을 복사-붙여넣기 하지 말고, 매력적인 서사로 엮어내세요.



"""


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
