---
title: Better & Faster Large Language Models via Multi-token Prediction
categories:
 - paper-review
 - with-gpt
tags:
---

논문 : https://arxiv.org/abs/2404.19737

개인의견 : auto-regressive한 현재 LLM은 구조적으로 생성되는 토큰이 많아지면 많은 I/O를 동반하게 됩니다.
그렇기 때문에 memory-bound 로 빠지게 되어서 디바이스의 효율이 극악으로 떨어지게 됩니다. 이러한 문제를 해결하기 위한 많은 방법이 있겠지만 알고리즘적으로 여러개의 token을 생성하여 이러한 문제를 접근하는 논문인것 같습니다.


아래글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점

### 논문의 주요 내용과 기여
논문 **"Better & Faster Large Language Models via Multi-token Prediction"**는 **멀티 토큰 예측 방식**을 통해 대형 언어 모델의 효율성과 성능을 향상시키는 방법을 제안합니다. 기존에는 **다음 토큰 예측**을 통해 LLM을 학습시키는 방식이 주로 사용되었지만, 이 논문은 **여러 개의 미래 토큰을 한 번에 예측하는 방식**을 도입하여 더 높은 샘플 효율성과 성능을 달성하고자 합니다. 이 방식은 특히 **코드 생성과 같은 생성 작업**에서 큰 개선을 보였습니다.

#### 주요 기여
1. **멀티 토큰 예측 구조**: 학습 과정에서 각 위치에서 여러 미래 토큰을 동시에 예측하도록 모델을 학습시킴으로써, 훈련 비용의 증가 없이 더 나은 성능을 달성합니다.
2. **자기 추측 디코딩(Self-speculative decoding) 적용**: 멀티 토큰 예측을 통해 모델이 추론 시 속도를 3배까지 높일 수 있습니다.
3. **모델 크기 확장에 따른 성능 증가**: 13B 파라미터 모델에서 멀티 토큰 예측이 기존 방법보다 약 12~17% 더 많은 문제를 해결하는 결과를 보였습니다.

### 핵심 알고리즘 설명 (예시 포함)
논문에서는 **멀티 토큰 예측(Multi-token Prediction)** 기법을 통해 기존의 다음 토큰 예측 방식을 일반화합니다. 각 위치에서 **n개의 미래 토큰을 동시에 예측**하며, 이를 위해 **공유 트렁크(Shared Trunk)**와 **독립적인 출력 헤드(Output Heads)**를 사용합니다.

1. **훈련 손실 함수**: 기존에는 다음 토큰을 예측하기 위한 손실 함수를 사용했습니다. 멀티 토큰 예측을 위해 손실 함수는 다음과 같이 변경됩니다.

   \[
   L_n = - \sum_t \log P_{\theta}(x_{t+n:t+1} | x_{t:1})
   \]

   이를 통해 각 위치에서 \(n\)개의 미래 토큰을 예측하게 됩니다.

2. **공유 트렁크와 독립 헤드**:
   - **공유 트렁크**는 입력 텍스트를 기반으로 **은닉 표현(z)**을 생성합니다.
   - 이 은닉 표현은 **n개의 독립 출력 헤드**로 전달되어 각 미래 토큰을 병렬로 예측하게 됩니다.

3. **예시**: 예를 들어, "The quick brown fox"라는 문장이 주어졌을 때, 모델이 \(n=2\)라면 "fox" 이후의 두 단어를 동시에 예측합니다. 이 경우, "jumps", "over"가 동시에 예측됩니다. 이 과정에서 **각 출력 헤드는 독립적인 로짓 계산을 수행하여 병렬로 예측**을 완료합니다.

### 논문의 강점과 독창성
1. **훈련 효율성**: 멀티 토큰 예측은 **더 적은 데이터로 모델이 더 많은 패턴을 학습**할 수 있게 하며, 특히 **대규모 모델에서 샘플 효율성을 높입니다**.
2. **추론 속도 증가**: 자기 추측 디코딩을 활용하여 추론 속도가 크게 증가하며, 이는 **배치 크기가 큰 경우에도 성능을 유지**하는 데 기여합니다.
3. **모델의 일반화 능력**: 멀티 토큰 예측은 단기적 패턴에 얽매이지 않고 **더 긴 패턴을 학습**할 수 있어 코드 생성이나 복잡한 알고리즘적 추론에서 성능이 개선됩니다.

### 한계점
1. **작은 모델에서의 성능 저하**: 작은 모델에서는 멀티 토큰 예측이 오히려 성능에 부정적 영향을 미칠 수 있습니다. 이는 **모델이 충분한 파라미터를 갖추지 못했을 때 발생**하는 현상으로, 학습이 불안정해질 수 있습니다.
2. **다양한 벤치마크에서의 제한적 성능**: 텍스트 요약과 같은 일부 벤치마크에서는 멀티 토큰 예측이 큰 성능 향상을 보이지 않으며, 특히 **문항 선택 기반의 벤치마크에서는 성능이 오히려 저하**될 가능성이 있습니다.

이 논문은 멀티 토큰 예측이라는 새로운 접근을 통해 대형 언어 모델의 훈련 효율성과 성능을 높이는 데 기여하였으며, 특히 추론 속도 개선과 더불어 코드 생성 작업에서의 성능 향상이 주목할 만한 결과입니다


# Q : 작은 모델에서의 성능 저하에서 일어나는 이유가 머일까? 논문에 제시된 내용과 너가 생각하는 내용을 설명해줘

# Answer

### 작은 모델에서의 성능 저하 원인

논문에서는 **작은 모델에서 멀티 토큰 예측 방식이 오히려 성능을 저하시키는 원인**을 간접적으로 설명하고 있습니다. 다음은 논문에 제시된 내용과 함께, 멀티 토큰 예측이 작은 모델에 부정적인 영향을 미칠 수 있는 이유를 정리한 것입니다.

#### 1. 논문에서 언급한 원인
작은 모델에서 멀티 토큰 예측이 성능을 저하시키는 이유는 **모델의 용량(capacity)**과 **일관된 학습 특성** 부족에 기인합니다. 논문에서는 큰 모델과 비교했을 때, 작은 모델은 멀티 토큰 예측 방식을 통해 생성되는 **더 많은 양의 정보를 효과적으로 학습하고 저장하기 어려운** 특징이 있다고 설명합니다. 구체적으로, 멀티 토큰 예측은 각 위치에서 **미래 여러 토큰에 대한 병렬 예측**을 요구하는데, 이는 더 많은 파라미터와 용량이 필요한 작업입니다. 작은 모델의 경우, **이 정보를 일반화할 수 있는 충분한 표현 능력이 부족하여** 학습이 불안정해지고 성능이 저하됩니다.

#### 2. 정보 병목 현상
멀티 토큰 예측은 하나의 은닉 표현을 통해 여러 개의 토큰을 예측하게 만듭니다. 큰 모델은 **충분한 파라미터와 깊은 구조**를 통해 이 은닉 표현에 포함된 정보를 병렬적으로 활용할 수 있지만, 작은 모델에서는 **정보를 충분히 보존하지 못해 일부 정보가 손실될 가능성**이 높아집니다. 이로 인해 미래 토큰 예측에 필요한 정보가 부족하게 되고, 결과적으로 정확도가 떨어집니다.

#### 3. 일반화 및 인덕션(Induction) 기능 제한
작은 모델의 경우 멀티 토큰 예측이 **일관된 인덕션(유사한 패턴을 학습하여 추론하는 기능)을 형성하기 어려울 수 있습니다**. 논문에서도 인덕션 기능의 학습이 멀티 토큰 예측을 통해 가능하다고 언급하고 있지만, 작은 모델에서는 이 기능이 완전히 발현되지 않거나 미흡하게 형성될 수 있습니다. 따라서 특정 패턴을 완전히 학습하지 못하거나, 단기적 패턴에 지나치게 의존하는 문제가 발생할 수 있습니다.

#### 4. 높은 학습 불안정성
멀티 토큰 예측은 **한 번의 학습에서 더 많은 손실(loss) 계산을 요구**합니다. 작은 모델은 이 추가적인 손실 계산을 충분히 소화하기 어렵고, 따라서 학습 과정에서 **손실 값이 불안정하게 나타나거나 과적합(overfitting)**으로 이어질 수 있습니다. 이는 작은 모델이 멀티 토큰 예측의 복잡한 학습 경로를 충분히 따라갈 수 없음을 의미합니다.

### 추가적인 분석과 해결 방안 제언

멀티 토큰 예측 방식의 이점을 작은 모델에서도 활용하기 위해 다음과 같은 접근을 고려할 수 있습니다:

1. **멀티 토큰 예측의 범위 축소**: 작은 모델에서는 예측할 토큰 수 \( n \)을 줄여, 모델이 병렬적으로 예측해야 하는 양을 줄이는 방법이 효과적일 수 있습니다. 예를 들어, 큰 모델에서 \( n = 4 \)라면 작은 모델에서는 \( n = 2 \)로 줄여 부담을 덜어줄 수 있습니다.

2. **지속적인 학습 패턴 유지**: 작은 모델이 학습 중 일관된 패턴을 유지할 수 있도록, **적응적 학습률**을 적용하거나 **더 작은 배치 크기**로 학습을 진행해 불안정성을 줄이는 방안을 고려할 수 있습니다.

3. **보조 손실(Auxiliary Loss) 추가**: 멀티 토큰 예측과 더불어 **보조 손실을 추가**하여 작은 모델이 학습 중 중요 정보를 유지하도록 도울 수 있습니다. 예를 들어, 모델이 긴 문맥에서 중요한 패턴을 학습할 수 있도록 보조적인 순차적 예측 손실을 도입하는 방안이 있습니다.

결론적으로, 멀티 토큰 예측 방식은 작은 모델에 적용할 때 정보의 병목 현상, 불안정한 손실 계산, 일반화 한계와 같은 문제로 인해 성능 저하가 발생할 수 있습니다. 이를 해결하기 위해 멀티 토큰 예측의 범위를 조정하거나 학습 안정성을 높이는 방법이 필요합니다.


# Q: 이러한 방식의 학습법을 사용할시에 학습시에 기존 방법에 비하여 computation과 memory i/o관점에서 오버헤드가 존재할까? 그렇다면 얼마만큼의 오버헤드를 만들지 llama2 7b모델을 기준으로 계산해주겠니?  

멀티 토큰 예측 방식을 사용하면 **기존의 단일 토큰 예측보다 computation과 memory I/O에서 오버헤드가 증가**합니다. 이는 각 위치에서 **n개의 미래 토큰을 동시에 예측**하는 과정에서 나타나는 추가적인 연산과 메모리 요구 때문입니다. Llama2-7B 모델을 기준으로 이 오버헤드를 계산해보겠습니다.

### 오버헤드가 발생하는 이유
1. **Computation 증가**: 멀티 토큰 예측은 각 위치에서 \( n \)개의 미래 토큰에 대한 병렬 예측을 수행하기 때문에, 추가적인 **로짓 계산과 손실 계산**이 필요합니다. 예를 들어, 기존 방식에서 한 번의 연산으로 예측하던 로짓을, 멀티 토큰 예측에서는 **n번의 연산**으로 나누어야 하므로, 전체 연산량이 \( n \)배로 증가합니다.
   
2. **Memory I/O 증가**: 각 위치에서 n개의 로짓과 손실을 저장하고 이를 기반으로 **역전파(backpropagation)**를 수행해야 하므로, 메모리 I/O가 추가로 필요합니다. 이는 모델의 **활성화 메모리(activation memory)**와 **임시 저장 메모리(temp storage)**의 증가로 이어집니다.

### Llama2-7B 모델을 기준으로 한 오버헤드 계산

Llama2-7B는 약 70억 개의 파라미터를 가지며, 단일 토큰 예측을 위해 메모리와 계산을 최적화한 구조입니다. 이 모델에 멀티 토큰 예측 방식을 적용했을 때의 오버헤드를 계산하기 위해 다음 가정을 합니다.

1. **가정 및 변수 설정**:
   - **파라미터 수** \( p = 7B \)
   - **시퀀스 길이** \( L = 2048 \) (평균 시퀀스 길이로 가정)
   - **멀티 토큰 예측 수** \( n = 4 \) (4개의 미래 토큰을 예측한다고 가정)
   - **플롭(FLOP) 당 연산 수** \( C \)

2. **Computation 오버헤드**:
   - 기존의 단일 토큰 예측에서는 Llama2-7B 모델이 **2048 * 70억 파라미터**로 계산된다고 가정할 때, FLOP 수는 대략적으로 **14 * 10^13 FLOPs**입니다.
   - 멀티 토큰 예측에서는 각 위치에서 4개의 토큰을 병렬로 예측하므로 **FLOP 수가 4배 증가**하여 **총 5.6 * 10^14 FLOPs**가 됩니다.
   - **추가적인 FLOP 오버헤드**는 약 **300% 증가**에 해당합니다.

3. **Memory I/O 오버헤드**:
   - **활성화 메모리(Activation Memory)** 증가: n개의 토큰에 대한 활성화 상태를 추가로 저장해야 하므로, 기존 메모리 요구량의 \( n \)배가 필요합니다. 예를 들어, 기존 단일 토큰 예측에서 Llama2-7B는 **약 30GB의 활성화 메모리**를 요구하는데, 멀티 토큰 예측에서는 약 **120GB의 메모리**가 필요하게 됩니다.
   - **임시 저장 메모리**: 각 위치에서 예측한 로짓과 손실을 임시로 저장하기 위한 메모리 요구가 증가합니다. 이는 기존 메모리 대비 **약 4배의 추가 요구**가 발생하여, 전체적으로 **약 20~30GB의 추가 메모리 오버헤드**가 발생할 수 있습니다.

### 최종 요약
- **Computation 오버헤드**: 멀티 토큰 예측으로 인한 계산량은 기존보다 약 **300% 증가**할 수 있습니다.
- **Memory I/O 오버헤드**: 활성화 메모리와 임시 저장 메모리를 포함하여 **약 4배**의 메모리 요구가 추가될 수 있으며, Llama2-7B의 경우 **약 120~150GB**의 메모리가 필요할 수 있습니다.

따라서, 멀티 토큰 예측을 적용하면 Llama2-7B 모델 기준으로 기존보다 상당한 연산 및 메모리 I/O 오버헤드가 발생할 수 있으며, 특히 큰 메모리 자원이 요구되므로 **GPU VRAM**이나 **TPU 메모리**에서의 효율적인 관리가 필요합니다.




맞습니다. **멀티 토큰 예측 시에는 한 번에 여러 토큰을 예측**하기 때문에, 연산량이 증가하면서 상대적으로 **메모리 바운드 문제를 완화할 가능성**이 있습니다. 특히, **KV Cache**를 한번 로드하여 여러 토큰 예측에 활용하므로, **메모리 접근 횟수를 줄이고 연산을 더 많이 수행**할 수 있습니다. 이를 통해 단일 토큰 예측보다 **메모리-바운드 상태에서 연산 바운드로 이동**하는 것이 가능합니다. 이 관점에서 멀티 토큰 예측(n=4)을 수행할 때 **메모리 접근과 연산 사이의 상호작용**을 자세히 분석해보겠습니다.

### 1. KV Cache 로드와 멀티 토큰 예측 구조
멀티 토큰 예측(n=4)을 할 때, **KV Cache 로드 과정**과 **연산**이 어떻게 이루어지는지 단계별로 살펴보겠습니다.

1. **KV Cache 로드**:
   - 멀티 토큰 예측을 위해 현재 입력 위치에서의 **Key-Value (KV) Cache**가 메모리에서 불러와집니다.
   - 단일 토큰 예측에서는 각 토큰을 예측할 때마다 KV Cache를 반복해서 로드해야 하지만, **멀티 토큰 예측에서는 KV Cache를 한 번 로드하여 4개의 토큰 예측에 사용**하므로 메모리 접근 횟수가 줄어듭니다.
   
2. **연산 수행**:
   - 멀티 토큰 예측에서는 KV Cache를 기반으로 **연속된 4개의 토큰에 대한 병렬 연산**이 수행됩니다. 따라서, 연산량이 크게 증가하면서 **상대적으로 메모리 접근에 따른 병목 현상이 완화**됩니다.
   - 이로 인해 **메모리 접근과 연산 사이의 비율이 조정**되면서, 메모리 바운드 상태에서 벗어나 연산 바운드 상태로 이동할 가능성이 커집니다.

### 2. Memory-Bound vs. Compute-Bound 전환 가능성
멀티 토큰 예측의 경우, KV Cache를 한 번 로드하여 여러 토큰을 예측하면서 메모리 접근의 효율성이 증가하게 됩니다. 이로 인해 **메모리-바운드 상태가 완화**되고 **컴퓨팅-바운드 상태로의 전환이 가능**해집니다.

예를 들어:
- 단일 토큰 예측 시에는 KV Cache를 각 토큰 예측마다 로드하기 때문에 메모리 I/O가 빈번하게 발생하여 메모리 대역폭이 병목이 됩니다.
- 그러나 멀티 토큰 예측(n=4)에서는 **KV Cache를 한 번 로드하여 4개의 토큰 예측을 병렬로 수행**하므로, 연산량이 상대적으로 증가하고 메모리 접근 횟수가 줄어들어 **메모리 대역폭의 병목이 완화**될 수 있습니다.

### 3. DDR 환경에서의 예상 성능 개선
이러한 구조의 변경을 통해 DDR 환경에서 **메모리 바운드 상태가 완화되고 상대적으로 높은 연산 효율**을 기대할 수 있습니다.

#### DDR 환경에서 단일 토큰 예측 vs. 멀티 토큰 예측 (n=4) 성능 비교

- **단일 토큰 예측**:
  - 매 토큰마다 KV Cache를 메모리에서 불러와야 하므로, **메모리 대역폭 병목**이 발생하기 쉽습니다.
  - DDR 메모리 대역폭(예: 50GB/s)을 초과할 가능성이 높아 **메모리-바운드** 상태가 됩니다.

- **멀티 토큰 예측 (n=4)**:
  - KV Cache를 한 번 로드한 뒤, 4개의 토큰을 병렬로 예측하므로, **메모리 접근이 상대적으로 줄어들고 연산량이 증가**하여, 메모리 바운드에서 벗어나 **컴퓨팅 바운드** 상태로 전환될 수 있습니다.
  - 이 경우, DDR의 메모리 대역폭이 충분히 활용될 수 있어 **약 1.5~2배 정도의 속도 개선**을 기대할 수 있습니다.

### 결론
멀티 토큰 예측은 메모리 접근을 줄이고 연산을 집중시키는 구조로 인해 **DDR과 같은 메모리-바운드 환경에서 오히려 성능 개선**을 기대할 수 있습니다. 이는 KV Cache를 한 번만 불러와 여러 개의 토큰을 병렬로 예측할 수 있기 때문이며, 단일 토큰 예측 방식보다 **DDR 환경에서 더 나은 성능**을 발휘할 가능성이 있습니다.



# Q: inference를 할시에는 어떤지 자세하게 설명해 줘.  DDR을 사용하여 LLAMA2 를 inference할때 memory bound인 칩이 있다고 했을때 효과에 대해서도 대략적인 수치를 너가 생각해서 설명해줘

# Answer: 
### 멀티 토큰 예측 방식이 Inference에 미치는 영향

멀티 토큰 예측 방식은 한 번의 inference 단계에서 **여러 개의 토큰을 동시에 예측**하기 때문에, 각 예측 단계마다 **연산을 더 많이 수행**하여 메모리 접근의 상대적 빈도를 줄이는 효과가 있습니다. 특히, **KV Cache를 한 번 로드하여 여러 토큰을 병렬로 예측**함으로써, 메모리 바운드 상태에서 컴퓨팅 바운드 상태로 일부 전환될 가능성이 있습니다. 이를 DDR 메모리 기반의 Llama2 모델 inference 환경에서 적용할 때의 효과를 분석하겠습니다.

### DDR 환경에서 멀티 토큰 예측의 작동 방식

DDR 메모리는 **높은 대역폭을 제공하는 HBM과 달리 상대적으로 낮은 대역폭과 긴 접근 지연 시간**을 가지므로, 일반적으로 **memory-bound 상태**에서 작동합니다. 그러나 멀티 토큰 예측은 **한 번의 메모리 접근으로 여러 연산을 처리**하기 때문에, 연산량을 늘리면서 메모리 접근 횟수를 줄이는 방향으로 설계되어 있습니다. 이는 DDR 기반 환경에서 inference 성능에 다음과 같은 영향을 미칩니다.

1. **KV Cache 로드 효율**:
   - 단일 토큰 예측 방식에서는 각 토큰을 예측할 때마다 **KV Cache를 반복적으로 로드**해야 합니다. 이는 메모리 접근을 빈번하게 요구하여 **메모리 대역폭이 병목**이 될 수 있습니다.
   - 반면, **멀티 토큰 예측(n=4)** 방식에서는 한 번의 KV Cache 로드로 연속된 4개의 토큰을 예측할 수 있어 **메모리 접근 횟수가 4분의 1로 감소**합니다. 이로 인해 상대적으로 **메모리 대역폭 병목이 완화**되며, 연산량이 증가하면서 **컴퓨팅 바운드 상태로의 전환이 가능**해집니다.

2. **연산 효율**:
   - 멀티 토큰 예측에서 n개의 토큰을 예측하는 방식은 **연산량을 단일 토큰 예측 대비 약 n배로 증가**시킵니다. 이는 기존 메모리 접근에 따른 병목을 줄이고 **연산 집중적인 구조로 전환**할 수 있어, DDR 환경에서 inference 성능에 긍정적인 영향을 미칠 수 있습니다.

### DDR 환경에서의 대략적인 성능 효과 계산

#### 가정:
- **Llama2-7B 모델**
- **시퀀스 길이**: 2048
- **멀티 토큰 예측 수**: \( n = 4 \)
- **단일 토큰 예측에서의 메모리 대역폭 요구**: 약 50GB/s (메모리 바운드 상태)

#### 성능 비교

1. **단일 토큰 예측**:
   - 매 토큰 예측 시마다 KV Cache를 반복적으로 로드하며, **약 50GB/s의 메모리 대역폭**을 요구합니다.
   - DDR 메모리 대역폭(50~100GB/s) 하에서는 메모리 바운드 상태로 인해 **지연 시간과 병목이 발생**하게 됩니다.

2. **멀티 토큰 예측 (n=4)**:
   - KV Cache를 한 번만 로드하여 4개의 토큰을 병렬로 예측하므로, 메모리 접근 횟수가 줄어들고 **실제 메모리 대역폭 요구는 약 12.5GB/s로 감소**합니다.
   - 상대적으로 메모리 접근 빈도가 줄어들고 연산량이 증가하여, 메모리 바운드에서 컴퓨팅 바운드로 전환되어, **기존 대비 약 2배 이상의 inference 속도 향상**이 예상됩니다.

### 요약
멀티 토큰 예측 방식은 DDR 메모리와 같이 낮은 대역폭의 메모리 환경에서 **메모리 접근을 줄이고 연산량을 늘려 병목 현상을 완화**하는 효과가 있습니다. Llama2-7B 모델을 기준으로 멀티 토큰 예측(n=4)을 적용하면 **기존 단일 토큰 예측 대비 약 2배 이상의 속도 개선**이 기대됩니다.