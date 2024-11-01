---
title: Prompt Cache Modular Attention Reuse for Low-Latency Inference
categories:
 - paper-review
 - with-gpt
tags:
---

논문 : https://arxiv.org/abs/2311.04934

개인의견 : 일부 반복되는 입력이 같은 위치에 들어오는 task에서(prompt에서 전처리가 있을시에 더욱) prefill시에 효과적으로 caching이 될거 같아 간단하게 효과적인 caching을 할 수 있는 방법일것 같으나 효과적이지 않은 상황이 생길 수 도 있을것 같습니다. 때문에 PML이라는 언어를 기술하여 효과적으로 재사용을 하려고 하지 않나 생각됩니다.

아래글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점

# Answer : 
네, 물론입니다. 다음은 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점에 대한 설명입니다.

### 논문의 강점 및 독창성
1. **지연 시간 감소**: Prompt Cache 접근법의 주요 강점은 자주 사용되는 프롬프트 모듈의 주의(attention) 상태를 재사용하여 첫 번째 토큰을 생성하는 데 걸리는 시간(TTFT)을 대폭 줄이는 것입니다. 이 방식은 특히 반복적인 구조를 가진 프롬프트에서 계산 오버헤드를 줄이고 응답 속도를 높이는 데 매우 유리합니다.
   
2. **모듈화된 재사용과 프롬프트 마크업 언어(PML)**: Prompt Cache는 프롬프트 모듈을 구조화하여 재사용성을 높이기 위해 '프롬프트 마크업 언어(PML)'를 도입했습니다. 사용자는 PML을 통해 자주 반복되는 프롬프트 부분을 명시적으로 정의하고, 이를 통해 주의 상태를 효율적으로 재사용할 수 있습니다. 이는 법률 문서나 템플릿이 자주 반복되는 시스템에서 매우 유용합니다.

3. **확장성과 다양성**: Prompt Cache는 다양한 아키텍처와 하드웨어 환경에서 사용될 수 있도록 설계되어 있습니다. CPU와 GPU 모두에서 작동 가능하고, 여러 모델에서 유연하게 적용할 수 있어 확장성이 뛰어납니다.

4. **경험적 검증**: 다양한 모델(예: Llama2, Falcon)에서 Prompt Cache가 지연 시간을 줄이면서도 정확도 손실이 거의 없음을 실험을 통해 증명하여 실제 환경에서도 높은 실용성을 보장합니다.

### 핵심 알고리즘 설명 (예시 포함)
Prompt Cache 알고리즘은 LLM 프롬프트에 대한 모듈화된 주의 상태 캐싱을 통해 작동합니다. 다음은 전체 프로세스와 예시입니다:

1. **스키마 및 모듈 정의**: 법률 보조 LLM의 경우 공통 프롬프트를 정의할 수 있습니다. 예를 들어,
   ```xml
   <schema name="legal-advice">
     <module name="case-summary">...</module>
     <module name="court-ruling">...</module>
     <module name="user-question">...</module>
   </schema>
   ```
   여기서 각 모듈은 반복적으로 나타나는 프롬프트의 한 부분입니다. 예를 들어 법적 요약이나 판결과 같은 부분은 자주 나타날 수 있습니다.

2. **주의 상태의 사전 계산**: 각 모듈(`case-summary` 또는 `court-ruling`)에 대해 주의 상태를 한 번 계산하고 이를 캐시에 저장합니다. 이러한 사전 계산된 상태는 각 모듈이 프롬프트에서 동일한 위치에 나타날 때마다 재사용될 수 있도록 고유한 위치 ID와 함께 저장됩니다.

3. **캐시 기반 프롬프트 조립**: 새 프롬프트가 들어오면 Prompt Cache는 사전에 정의된 모듈에 대한 주의 상태를 메모리에서 불러와 다시 계산하지 않고 바로 사용합니다. 예를 들어:
   ```xml
   <prompt schema="legal-advice">
     <case-summary />
     <court-ruling />
     <user-question>항소 성공 가능성은 어떻게 되나요?</user-question>
   </prompt>
   ```
   여기서 `case-summary`와 `court-ruling` 모듈은 캐시에서 재사용되고, `user-question`만 동적으로 계산됩니다.

4. **결과 생성**: `case-summary`와 `court-ruling` 모듈에 대한 사전 계산된 상태와 `user-question`에 대한 동적 상태가 결합되어 전체 프롬프트의 응답을 생성합니다. 이로 인해 첫 번째 토큰을 생성하는 데 필요한 계산량이 크게 줄어듭니다.

### 한계점
1. **위치 의존성**: Transformer 구조의 위치 인코딩 때문에 텍스트 세그먼트의 주의 상태를 재사용하려면 같은 위치에 나타나야 한다는 제약이 있습니다. 이로 인해 일부 애플리케이션에서 유연성이 떨어질 수 있습니다.

2. **메모리 오버헤드**: 캐싱은 계산 시간을 줄여주지만, 메모리 사용량이 늘어납니다. 특히 대형 LLM의 경우 여러 프롬프트 모듈에 대한 주의 상태를 저장하면 상당한 메모리가 필요합니다.

3. **최적화 요구 사항**: 스키마 설계 및 사전 계산 단계가 필요하므로 다소 추가적인 설정이 필요합니다. 이는 구조가 고정되지 않은 프롬프트를 사용하는 애플리케이션에 제약이 될 수 있습니다.

결론적으로 Prompt Cache는 자주 사용되는 프롬프트 모듈의 주의 상태를 재사용하여 LLM의 추론 효율을 높이기 위한 강력한 도구이지만, 위치 제약과 메모리 사용량 같은 최적화 필요성을 고려해야 할 부분도 있습니다.


# Q : 실제로 캐싱이 되어서 저장되는 값은 KV cache와 같은 값인걸까? 그리고 prefill과정에서 앞부분의 일부 값이 캐싱 되는게 얼마나 효과적일지 input length가 2048일때 LLAMA2같은 모델에서 수치적으로 얼마나 될지 설명해줘

# Answer

### 1. 캐싱된 값과 KV Cache 값의 유사성

Prompt Cache에서 저장되는 값은 **기존의 KV Cache**와 유사한 방식으로, 각 토큰에 대한 **Key와 Value의 attention 상태**입니다. Prompt Cache는 일반적인 KV Cache와 마찬가지로 주어진 프롬프트에 대한 attention 상태를 캐싱하여 반복적인 계산을 피할 수 있습니다. 그러나 Prompt Cache는 단일 프롬프트 내에서만 재사용 가능한 기존 KV Cache와 달리, **여러 프롬프트에 걸쳐 재사용할 수 있도록 설계**되어 있다는 점에서 차이가 있습니다. 

따라서 캐싱되는 값은 일반적인 KV Cache와 구조상 동일하지만, Prompt Cache에서는 이를 **모듈별로 사전 정의된 위치에 맞춰 사용할 수 있도록 PML 구조**로 저장해 두는 것이 특징입니다. 이러한 방식을 통해, 동일한 모듈이 여러 프롬프트에서 반복될 경우 해당 attention 상태를 재활용하여 효율을 높이는 것이 가능해집니다.

### 2. Input Length가 2048일 때, LLAMA2에서 캐싱 효율 수치 분석

#### KV Cache와 Prompt Cache의 계산 복잡도 차이
Transformer 모델에서 각 토큰에 대한 **attention 계산의 복잡도**는 시퀀스 길이에 따라 **O(n^2)**으로 증가합니다. 예를 들어, 시퀀스 길이가 \(n\)일 때 계산 복잡도는 **O(n^2 \times d)** (여기서 \(d\)는 hidden dimension)입니다. 따라서 시퀀스 길이가 길어질수록 attention 계산의 비용은 급격히 증가하게 됩니다.

Prompt Cache는 특정 모듈을 캐싱하여, **전체 input 길이가 길어지더라도 사전 계산된 모듈에 대해서는 추가적인 attention 계산이 필요하지 않도록 합니다**. 이를 통해 **최초 입력 단계에서의 attention 연산을 선형으로 줄일 수 있어, 큰 시퀀스에서도 상당한 시간 절약 효과**를 기대할 수 있습니다.

#### LLAMA2 모델에서의 구체적 효율 비교
* **시퀀스 길이 (Input Length) 2048**에서 LLAMA2 모델의 attention 계산에 필요한 FLOPs (Floating Point Operations per Second)는 대략적으로 **O(n^2)** 형태이므로, 일반적인 KV Cache 대비 Prompt Cache는 시퀀스 길이가 길어질수록 효율성이 증가합니다.
* 예를 들어, 시퀀스 길이 2048에서, **전체 attention 계산을 매번 수행할 경우** 필요한 계산량을 **2048^2 = 4,194,304 FLOPs**로 볼 수 있습니다. 하지만 Prompt Cache가 적용된 경우, **사전 캐싱된 모듈을 제외한 나머지 부분에 대해서만 계산**이 필요하므로, 만약 전체 프롬프트의 50% 정도가 캐싱된 모듈로 구성된 경우, 이 계산량은 약 절반으로 줄어들어 **2,097,152 FLOPs**로 감소할 수 있습니다.

따라서 **LLAMA2 모델의 input 길이가 2048일 때** Prompt Cache를 통해 일부 프롬프트를 사전에 캐싱하면, 전체적인 초기 토큰 생성 과정에서 최대 **50% 이상의 연산량 절감** 효과를 볼 수 있습니다.

