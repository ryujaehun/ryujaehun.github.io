---
categories:
- paper-review
- with-gpt
date: '2024-10-31'
tags:
- Natural Language Processing
title: Prompt Cache Modular Attention Reuse for Low-Latency Inference
lastmod: '2026-10-25T00:54:46+09:00'
description: Prompt Cache의 PML·위치 ID·모듈 간 attention 근사를 설명하고, 2048토큰 예시의 FLOPs 오류와
  실험 조건별 TTFT를 원문에 맞게 정리한다.
---

논문: [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934v2) · [저자 구현](https://github.com/yale-sys/prompt-cache)

> 2026-10-25 원문 재검토: v2를 기준으로 기존 GPT 대화의 위치 제약 설명과 FLOPs 계산을 수정했다. 실험 결과와 설명용 계산 예시를 구분했다.

## KV 캐시의 재사용 범위를 모듈로 넓힌다

일반적인 KV 캐시는 한 요청의 autoregressive decoding 중에 이미 처리한 토큰의 K·V를 재사용한다. Prompt Cache는 자주 반복되는 시스템 메시지·문서·템플릿의 K·V를 미리 저장하고, 여러 요청에서 해당 **프롬프트 모듈**을 불러와 첫 토큰까지의 시간(TTFT)을 줄인다. 모든 층의 K·V를 재사용하는 것이며 완성된 답변을 캐시하는 방식은 아니다. [§2–3](https://arxiv.org/html/2311.04934v2#S2)

Prefill을 줄이는 것과 이후 토큰의 생성 속도를 높이는 것은 다르다. 논문은 첫 토큰 이후 decoding의 계산 복잡도와 지연이 일반 KV cache 방식과 같다고 설명한다. [§3.4, §5](https://arxiv.org/html/2311.04934v2#S3.SS4)

## PML과 위치 ID가 해결하는 문제

K·V는 토큰 내용뿐 아니라 위치 정보와 앞선 문맥의 영향을 받는다. 따라서 같은 문장이라는 이유만으로 임의의 위치에 캐시를 옮겨 쓰면 안 된다.

Prompt Cache는 PML(Prompt Markup Language) 스키마에 재사용할 모듈과 위치를 선언한다. 모듈을 선택적으로 가져오더라도 **스키마에서 부여한 위치 ID를 유지**하며, 연속되지 않는 위치 ID를 지원하도록 모델의 위치 인코딩 구현을 조정한다. 이 설계 덕분에 모든 모듈이 프롬프트의 접두사일 필요는 없다. 그렇다고 아무 텍스트의 캐시를 아무 위치에 자유롭게 재배치할 수 있다는 뜻도 아니다. [§3.1–3.3, §4.2](https://arxiv.org/html/2311.04934v2#S3.SS1)

다음은 논문의 문법을 따라 만든 설명용 예시다. 성능 측정에 사용한 실제 입력은 아니다.

```xml
<schema name="legal-advice">
  <module name="case-summary">사건 요약 문서</module>
  <module name="court-ruling">판결 문서</module>
</schema>
```

```xml
<prompt schema="legal-advice">
  <case-summary />
  <court-ruling />
  두 문서를 바탕으로 항소 쟁점을 정리해 주세요.
</prompt>
```

두 문서의 상태는 스키마에서 캐시하고 마지막 질문은 새로 계산한다. 스키마는 파라미터 자리도 예약할 수 있지만, 요청마다 들어오는 실제 파라미터 값의 attention 상태는 캐시된 것으로 취급하지 않는다. [§3.2–3.4](https://arxiv.org/html/2311.04934v2#S3.SS2)

## 같은 위치 ID여도 full prefill과 동일하지 않을 수 있다

위치 문제와 문맥 문제는 별개다. 모듈을 독립적으로 인코딩하면 각 모듈은 다른 모듈의 토큰을 보지 않는다. 새로 계산하는 질문 토큰이 결합된 캐시에 attention할 수는 있지만, 앞서 저장한 문서 토큰의 내부 표현이 다른 문서를 읽고 갱신되는 것은 아니다.

원문은 이를 모듈 범위로 attention을 제한하는 근사로 설명한다. 서로 독립적인 모듈에는 적합할 수 있지만, 강하게 의존하는 문서 조합에서는 품질을 낮출 수 있다. 필요한 모듈 집합을 함께 인코딩해 별도로 저장하는 **scaffolding**은 이런 문제를 줄이는 대신 추가 메모리가 든다. 따라서 “품질 손실이 없다”는 보편적인 보장으로 해석하면 안 된다. [§3.3](https://arxiv.org/html/2311.04934v2#S3.SS3)

## 2048토큰 중 절반을 캐시하면 연산도 절반인가

기존 글은 `2048² = 4,194,304 FLOPs`라고 계산했다. **토큰 수의 제곱은 attention 행렬의 원소 수 규모이며 FLOPs가 아니다.** 실제 연산량에는 head dimension, head 수, 층 수, QKV projection, MLP 등이 들어간다. FLOPs는 연산량이고 FLOP/s는 초당 연산 처리율이다.

아래는 Prompt Cache 실험의 재현이 아니라, 같은 접두사를 정확히 재사용하는 causal attention의 단순 계산 예시다. 전체 길이 $n=2048$, 캐시된 접두사 길이 $m=1024$라고 가정한다. 토큰이 자기 자신과 앞선 토큰을 보는 유효한 query-key 쌍 수는 다음과 같다.

$$
P_{\mathrm{full}} = \frac{n(n+1)}{2} = 2{,}098{,}176.
$$

접두사의 query 행은 재계산하지 않아도 되지만, 남은 토큰은 앞선 캐시 토큰도 읽어야 한다.

$$
P_{\mathrm{remaining}}
= \sum_{i=m+1}^{n} i
= \frac{n(n+1)-m(m+1)}{2}
= 1{,}573{,}376.
$$

따라서 이 예시에서 유효 쌍 수의 절감은 약 25.01%다. 남은 1024토큰끼리만 attention한다고 보고 길이의 제곱을 다시 계산하면, 접두사로 향하는 attention을 빠뜨리게 된다.

이것도 GPU 커널이 실제 실행한 FLOPs나 전체 모델의 시간 절감률은 아니다. 토큰별 projection·MLP의 절감 양상은 attention의 쌍 수와 다르고, 메모리 복사 비용과 커널 구현도 영향을 준다. Llama2의 실제 TTFT를 알고 싶다면 모델 크기·자료형·하드웨어·캐시 저장 위치를 정하고 측정해야 한다. **캐시된 토큰의 비율 하나로 전체 연산량이나 지연 감소율을 단정할 수 없다.**

## 논문은 실제로 얼마나 빨라졌는가

논문 v2 §5.2의 Llama 7B TTFT 비교는 8개 LongBench 데이터셋을 본문에 제시하며, GPU 실험의 평균 입력 길이는 약 5K토큰이다. GPU와 캐시 저장 위치에 따라 결과가 달랐다.

| 추론 장치 | 모듈 캐시 위치 | §5.2의 TTFT 가속 결과 |
| --- | --- | --- |
| GPU: RTX 4090, A40, A100 | CPU 메모리 | 1.5–3배 |
| GPU: RTX 4090, A40, A100 | GPU 메모리 | 5–10배 |
| CPU: Intel i9-13900K | CPU 메모리 | 최대 70배 |
| CPU: AMD Ryzen 9 7950X | CPU 메모리 | 최대 20배 |

초록은 GPU 8배·CPU 60배로 요약하지만, 위 표는 본문의 조건별 결과다. 서로 다른 요약 수치를 합쳐 하나의 보편적인 가속 배수로 제시하면 안 된다. CPU 캐시는 더 큰 저장 공간을 제공하는 대신 GPU로 복사하는 비용이 발생한다. [§5.1–5.2, Figures 3–4](https://arxiv.org/html/2311.04934v2#S5.SS2)

품질 평가도 평균적인 유사성에 대한 결과이지 모든 점수가 같은 것은 아니다. 예를 들어 Table 1의 Llama2 7B에서 2Wiki Multi-Hop QA F1은 16.63에서 13.95로 낮아지고, MuSiQue F1은 7.31에서 8.57로 높아진다. 이 표는 scaffolding 없이 수행한 평가다. [§5.3, Table 1](https://arxiv.org/html/2311.04934v2#S5.SS3)

## 메모리와 적용 한계

캐시 저장 비용은 모듈 길이에 비례한다. 일반적인 K·V payload 크기를 계산하려면 다음 요소가 필요하다.

$$
M_{\mathrm{KV}} = 2 L N H_{\mathrm{KV}} d_h b.
$$

여기서 $L$은 층 수, $N$은 저장하는 토큰 수, $H_{\mathrm{KV}}$는 KV head 수, $d_h$는 head dimension, $b$는 원소당 바이트 수다. 앞의 2는 K와 V를 뜻한다. 이 식은 텐서 payload만 세므로 메타데이터·정렬·중복 캐시·scaffold 비용은 별도로 고려해야 한다. 이는 저장량을 설명하는 일반식이며 논문의 모델별 메모리 측정값을 대신하지 않는다.

개인적으로는 반복되는 시스템 메시지와 고정 문서가 많은 서비스에서 효과적인 방향이라고 본다. 다만 스키마를 설계하고, 모델의 위치 ID 처리를 맞추며, 모듈 사이 의존성과 저장 공간을 점검해야 한다. [CacheBlend](https://arxiv.org/html/2405.16444v3)는 독립 청크 캐시를 현재 입력에 맞춰 일부 재계산한다는 점에서 비교할 만하다. 둘 중 하나가 모든 워크로드에서 우월하다는 결론은 각 서비스의 품질·적중률·TTFT 측정이 있어야 내릴 수 있다.
