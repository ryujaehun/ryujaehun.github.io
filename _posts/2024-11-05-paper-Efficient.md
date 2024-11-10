---
title: Efficient Generative LLM Inference Using Phase Splitting
categories:
 - paper-review
 - with-gpt
tags:
---


논문 : https://arxiv.org/abs/2311.18677v2


아래글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점

# Answer

논문 **"Splitwise: Efficient Generative LLM Inference Using Phase Splitting"**는 대형 언어 모델(LLM)의 추론을 효율적으로 수행하기 위해서 **추론 단계를 분리하는 Splitwise**라는 기법을 제안합니다. 이 논문은 주로 대형 모델의 추론 요청 시 발생하는 비효율성을 개선하고자 합니다. Splitwise는 추론 과정의 **프롬프트 계산 단계**와 **토큰 생성 단계**를 분리하여 각 단계에 맞는 최적의 하드웨어 자원을 할당함으로써 성능과 비용을 동시에 개선하는 시스템입니다. 이 논문의 주요 강점, 독창적인 지점, 핵심 알고리즘의 예시, 그리고 한계점은 다음과 같습니다.

### 강점 및 독창적인 지점

1. **단계별 하드웨어 최적화**:
   - LLM 추론의 프롬프트 계산 단계와 토큰 생성 단계는 각각 서로 다른 하드웨어 요구 사항을 가집니다. Splitwise는 이 점을 활용하여, **프롬프트 계산은 높은 계산 성능이 필요한 최신 GPU에서**, **토큰 생성은 메모리 집약적인 작업이므로 상대적으로 저렴한 GPU에서** 수행하도록 분리합니다.
   - 이로 인해 최신 GPU를 과다 사용하지 않으면서도 비용 효율적인 운영이 가능합니다.

2. **데이터 전송 최적화**:
   - Splitwise는 프롬프트 계산 단계에서 생성된 **키-값(KV) 캐시를 토큰 생성 단계로 전달하는 과정을 최적화**하여 전송 지연을 줄입니다. Infiniband와 같은 고속 네트워크를 활용해 캐시 데이터를 효율적으로 전송합니다.
   - 이러한 최적화를 통해 두 단계 간의 데이터 통신에서 발생하는 병목 현상을 완화하고 전체 시스템 성능을 높입니다.

3. **이질적인 GPU 클러스터 구성**:
   - Splitwise는 최신 고성능 GPU와 이전 세대의 저렴한 GPU를 혼합하여 사용하는 **이질적 클러스터를 구성**할 수 있어, 사용자의 다양한 성능 및 비용 요구를 충족할 수 있습니다.
   - 이를 통해 기존 단일 GPU 클러스터 대비 최대 2.35배의 처리량을 제공하거나, 같은 처리량을 20% 더 낮은 비용으로 달성할 수 있습니다.

### 핵심 알고리즘과 예시

Splitwise의 핵심 알고리즘은 **프롬프트 계산 단계와 토큰 생성 단계를 분리하여 별도의 머신에서 실행**하는 것입니다. 이를 통해 각 단계에 적합한 하드웨어를 사용하고, **효율적으로 데이터 전송**을 관리합니다.

#### 예시 입력 및 전체 과정:

1. **프롬프트 계산 단계**:
   - 사용자가 입력 프롬프트를 제공하면, Splitwise는 이를 먼저 프롬프트 계산 머신으로 보냅니다.
   - 예를 들어, 질문이 "Is tomato a fruit?"라면, 프롬프트 머신이 입력을 처리하고 첫 번째 토큰을 생성합니다. 이때, **프롬프트 계산 단계는 높은 계산 성능이 필요한 GPU에서 수행**됩니다.
   - 프롬프트 계산 단계에서 생성된 캐시 데이터는 **키-값(KV) 캐시**로 저장되며, 이후 토큰 생성에 반복적으로 사용될 데이터를 포함하고 있습니다.

2. **KV 캐시 전송**:
   - 프롬프트 계산 단계가 완료되면, **KV 캐시가 토큰 생성 머신으로 전송**됩니다.
   - 이때, Splitwise는 KV 캐시의 전송을 최적화하여 레이어 단위로 비동기 전송을 수행하고, 프롬프트 머신의 다음 계산과 병렬로 진행하여 전송 지연을 최소화합니다.

3. **토큰 생성 단계**:
   - 토큰 생성 단계에서는 이전에 전송된 KV 캐시를 사용하여 **순차적으로 토큰을 생성**합니다. 예를 들어, 첫 번째 응답 토큰으로 "Yes"가 생성되면, 이후 토큰은 이 답변에 추가됩니다.
   - 토큰 생성 단계는 메모리 요구량이 높기 때문에 상대적으로 저렴한 GPU에서 수행되며, 이로써 비용 효율성을 높입니다.

이러한 단계를 통해 Splitwise는 두 단계의 서로 다른 리소스 요구 사항을 고려하여 **최적의 하드웨어 구성을 적용**하고, 데이터 전송 지연을 최소화하여 효율적인 LLM 추론을 수행합니다.

### 한계점

1. **네트워크 의존성**:
   - Splitwise는 프롬프트 계산과 토큰 생성 사이의 데이터 전송을 위해 고속 네트워크(Infiniband 등)에 의존합니다. 만약 네트워크 속도가 낮거나 대역폭이 제한되면 데이터 전송에서 병목이 발생하여 성능이 저하될 수 있습니다.

2. **이질적인 GPU 간 통합의 어려움**:
   - 이 논문에서 제안하는 이질적 클러스터 구성이 모든 데이터 센터 환경에서 적용되기 어려울 수 있습니다. 예를 들어, 최신 고성능 GPU와 이전 세대 GPU가 혼합된 클러스터를 구성하려면 추가적인 인프라 관리와 지원이 필요합니다.

3. **실시간 추론 환경에서의 제한**:
   - Splitwise는 일반적인 배치(batch) 작업에 적합하지만, 실시간 반응이 필요한 추론 환경에서는 단계 분리가 지연을 초래할 수 있습니다.


# Q : 제시해준 한계점에서 "실시간 추론 환경에서의 제한"이 있는데 서비스 레벨에서 decoding이 real-time이 아닌 이런 환경에서 진행이 될 경우가 어떤 경우일까? 그리고 그때 이러한 방법이 얼마나 효과적일까? 

# Answer : 
LLM 추론이 **실시간 반응성을 필수적으로 요구하지 않는 서비스 레벨**에서 이루어질 수 있는 경우에는 Splitwise와 같은 단계 분리 방식을 더욱 효과적으로 적용할 수 있습니다. 대표적으로 이러한 환경은 다음과 같은 경우가 있을 수 있습니다.

### 1. **비동기적 처리**가 가능한 경우

예를 들어, 이메일 요약, 문서 분석, 대규모 텍스트 분류 또는 추천 시스템 등에서는 사용자 입력이 이루어지고 나서 결과를 **즉시 반환할 필요가 없으며, 몇 초에서 몇 분의 지연이 허용**될 수 있습니다. 이러한 서비스는 실시간으로 사용자와의 상호작용이 없기 때문에 응답 지연 시간이 엄격하지 않습니다. 

이러한 비동기적 작업에서는 **높은 처리량과 비용 효율성이 더 중요한 목표**이므로, Splitwise의 분리된 단계 처리가 매우 유리하게 작용할 수 있습니다. Splitwise를 사용하면 최신 GPU를 프롬프트 계산에만 집중할 수 있고, 메모리 집약적인 토큰 생성은 상대적으로 저렴한 하드웨어에서 처리할 수 있어 **비용 절감과 자원 최적화 효과가 극대화**됩니다.

### 2. **대량의 요청을 처리해야 하는 경우**

고객 서비스에서의 FAQ 응답 생성, SNS에서 트렌드 분석을 위한 데이터 처리 등 대규모의 텍스트 데이터를 비동기적으로 처리해야 하는 경우도 해당합니다. 이런 서비스들은 실시간으로 개별 응답을 처리할 필요가 없으며, **많은 양의 데이터를 주기적으로 또는 배치(batch) 방식으로 처리**하는 것이 목표입니다.

이 경우 Splitwise는 **추론 파이프라인의 각 단계를 최대한 병렬화**하여 처리량을 극대화하는 데 유리합니다. Splitwise는 프롬프트 계산과 토큰 생성을 다른 기기에서 수행함으로써, 요청을 효율적으로 배치하고 하드웨어 리소스를 최적 활용하여 **단위 시간당 더 많은 요청을 처리**할 수 있게 합니다. 특히, 토큰 생성이 시간이 걸리더라도 전체적인 대기 시간이 문제가 되지 않기 때문에 Splitwise 방식이 매우 효과적일 수 있습니다.

### 3. **장시간 처리 작업**이 필요한 경우

또한, 분산된 대규모 데이터 세트에 대해 텍스트 기반 분석을 수행하거나, 법률 문서 분석, 기술 문서 작성 지원 등에서 **장시간 처리 작업이 필요한 경우**도 있습니다. 이러한 작업은 실시간성이 필수가 아니고, 대규모 데이터 처리와 메모리 요구가 크기 때문에 Splitwise의 분리 방식이 유용할 수 있습니다.

Splitwise가 적용되는 경우, **프롬프트 계산은 최신 GPU에서 빠르게 진행**하고, 이후 **토큰 생성은 메모리 용량이 큰 하드웨어에서 느리게라도 처리**할 수 있기 때문에 전체적으로 리소스 비용 절감 효과가 큽니다.

---

### Splitwise의 효과성 평가

이러한 **비실시간 환경**에서 Splitwise는 **비용 효율성, 하드웨어 자원 활용도, 그리고 처리량을 극대화**하는 데 매우 효과적일 수 있습니다. 실시간 응답이 요구되지 않는 경우 Splitwise의 프롬프트-토큰 분리 방식은 대규모 요청을 안정적으로 처리할 수 있게 하며, 더 저렴한 GPU와 다양한 하드웨어 구성을 통해 비용을 절감하는 장점이 두드러집니다.

따라서, Splitwise는 실시간성이 덜 요구되는 작업에서 높은 성능과 비용 효율을 제공하여 대규모 비동기적 추론 작업에 최적화된 방식으로 적용될 수 있습니다.


# Q : 그리고 이논문에서 사용하는 고속 네트워크는 nvlink일까 아니면 infiniband일까 어떤 네크워크를 이용하는데 토플로지 및 스펙을 자세하게 알려줘

# Answer : 

논문에서 Splitwise 시스템은 **InfiniBand**를 사용하여 GPU 인스턴스 간 고속 데이터 전송을 수행합니다. Splitwise 클러스터의 각 머신은 주로 8개의 NVIDIA GPU(A100 또는 H100)를 포함하며, 이러한 GPU들은 Mellanox InfiniBand 네트워크로 연결되어 **고대역폭 통신**을 지원합니다. 특히, A100 GPU는 **200 Gbps**의 대역폭을, H100 GPU는 **400 Gbps**의 대역폭을 제공합니다【24:8†source】【24:16†source】【24:10†source】.

Splitwise의 InfiniBand 활용 방식 및 네트워크 토폴로지는 다음과 같습니다.

1. **레이어별 KV 캐시 전송**: KV 캐시 데이터를 레이어 단위로 비동기 전송하여 프롬프트 머신에서 계산을 수행하는 동안 InfiniBand를 통해 데이터 전송을 겹치게 합니다. 이를 통해 데이터 전송 지연을 줄이고 전체 성능을 최적화합니다.

2. **MSCCL++를 활용한 제로 카피 데이터 전송**: 프롬프트 머신은 InfiniBand의 **제로 카피(one-sided put) 전송**을 사용하여 직접 데이터 전송을 수행하며, 이때 토큰 머신에서 별도의 수신 명령을 수행하지 않아도 됩니다. 동기화는 InfiniBand 상의 세마포어를 통해 이루어지므로, 데이터 전송 중 발생할 수 있는 지연이 최소화됩니다【24:4†source】【24:16†source】.

이러한 고속 InfiniBand 네트워크는 Splitwise가 프롬프트 및 토큰 생성 단계를 분리하더라도 실시간에 가까운 데이터 동기화를 가능하게 해줍니다. 특히 대규모 데이터 전송이 필요한 상황에서도 성능 저하 없이 데이터를 효율적으로 처리할 수 있게 합니다.

# Q : Splitwise에 대한 질문 하나만 더할께 이논문에서는 모두 HBM을 탑재한 H100/A100이 대상인것 같은데 예를들어 A100과 GDDR을 탑재한 RTX 2080TI에서도 이논문에서 제시하는 방법이 유효할까?

# Answer :
이 논문에서 제안하는 **Splitwise 방법**은 HBM을 탑재한 A100 또는 H100 같은 고성능 GPU를 활용하여 **프롬프트 계산과 토큰 생성의 단계를 분리**하고 각 단계의 하드웨어 요구 사항에 맞게 최적화하는 것이 핵심입니다. 그러나 A100/H100과 달리 GDDR 메모리를 사용하는 RTX 2080 Ti 같은 GPU에서는 몇 가지 기술적 한계로 인해 Splitwise의 효율성이 제한될 수 있습니다.

### 1. **메모리 대역폭의 한계**

A100과 H100은 **HBM 메모리**를 사용하여 매우 높은 메모리 대역폭을 제공하며, 이는 대용량의 KV 캐시 전송이나 연산이 필요한 LLM 추론에서 특히 유리합니다. 반면 RTX 2080 Ti의 **GDDR6 메모리**는 HBM 메모리보다 대역폭이 낮아, 대규모 모델을 위한 **캐시 데이터 전송 속도가 느려질 가능성**이 큽니다. 이는 Splitwise가 데이터 전송을 겹치며 단계 간 효율적으로 진행되도록 설계된 점에서 병목을 초래할 수 있습니다.

### 2. **메모리 용량의 제약**

RTX 2080 Ti는 A100/H100과 비교할 때 메모리 용량이 훨씬 적습니다(11GB GDDR6 vs. 40GB HBM). Splitwise는 토큰 생성 단계에서 많은 메모리를 요구하므로, RTX 2080 Ti 같은 제한된 메모리를 가진 GPU에서는 대형 모델을 전부 메모리에 올리지 못해 **토큰 생성 단계에서 효율이 낮아지거나 추론이 불가능할 수** 있습니다. 

### 3. **데이터 전송 및 지연 시간**

Splitwise는 프롬프트 계산과 토큰 생성 단계 간 **인피니밴드(Infiniband)와 같은 고속 네트워크**를 통해 데이터를 전송하여 전송 지연을 줄입니다. RTX 2080 Ti 기반 시스템에서는 이러한 고속 네트워크가 없거나, PCIe를 통한 데이터 전송이 주로 이루어져 전송 속도가 떨어질 수 있습니다. 이는 **캐시 데이터를 빠르게 전송하지 못해 단계 간 지연이 발생**할 가능성을 높입니다.

### 결론적으로

RTX 2080 Ti와 같은 GDDR 메모리 기반 GPU에서는 Splitwise의 성능이 크게 제한될 수 있으며, 특히 **메모리 대역폭, 용량, 네트워크 속도**가 중요한 Splitwise 방식의 효율성을 충분히 발휘하기 어려울 것입니다. 따라서 Splitwise 방식은 **HBM과 고속 네트워크 환경에서 더욱 효과적**으로 동작하며, GDDR 메모리 기반 GPU에서는 더 작은 규모의 모델이나, 적은 연산 요구를 가진 작업에만 제한적으로 적용할 수 있을 것으로 보입니다.