---
categories:
- paper-review
- with-gpt
date: "2024-11-05"
tags: null
title: SpotServe Serving Generative Large Language Models on Preemptible Instances
---


논문 : https://arxiv.org/abs/2311.15566


아래글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점

# Answer

논문 **"SpotServe: Serving Generative Large Language Models on Preemptible Instances"**는 대형 언어 모델(LLM)을 경제적인 프리엠티브 인스턴스에서 효율적으로 서빙할 수 있는 새로운 분산 시스템 SpotServe를 제안합니다. 이 논문의 주요 강점과 독창적인 지점, 핵심 알고리즘 예시, 그리고 한계점을 다음과 같이 설명드리겠습니다.

### 강점 및 독창적인 지점

1. **비용 절감**: SpotServe는 LLM을 프리엠티브 GPU 인스턴스에서 운영하여 운영 비용을 54%까지 절감할 수 있습니다. 기존의 LLM 서빙 방식은 고가의 온디맨드 인스턴스에 의존해 비용이 높았지만, SpotServe는 이를 획기적으로 낮췄습니다.

2. **동적 재병렬화**: SpotServe는 인스턴스 가용성 변동에 따라 실시간으로 병렬 구성(데이터, 텐서, 파이프라인 병렬성)을 최적화합니다. 이를 통해 인스턴스가 프리엠트되거나 새로운 인스턴스가 추가될 때 성능 저하를 최소화합니다.

3. **이중 그래프 매칭 및 상태 기반 복구**: SpotServe는 쿠흐-먼커스(Kuhn-Munkres) 알고리즘을 사용하여 프리엠트된 인스턴스와 새로 할당된 인스턴스 간의 최적 매핑을 도출하고, 토큰 단위로 계산 상태를 복구하여 연산의 중복을 줄입니다.

4. **실험적 검증**: 다양한 대형 언어 모델(OPT-6.7B, GPT-20B, LLaMA-30B)을 대상으로 실험을 수행하여 SpotServe의 효율성을 입증하였으며, 기존 시스템 대비 P99 지연 시간을 최대 9배까지 단축했습니다.

### 핵심 알고리즘과 예시

SpotServe의 핵심 알고리즘은 다음과 같습니다.

1. **동적 재병렬화**: 인스턴스의 가용성에 따라 병렬화 전략을 조정하는 이 알고리즘은 성능(처리량, 지연 시간)을 극대화합니다. 예를 들어, 인스턴스 가용성이 높을 때는 `(D=3, P=3, M=4)`와 같은 구성을 통해 LLM 서빙 성능을 높이며, 가용 인스턴스가 줄어들면 `(D=1, P=2, M=8)`로 재구성하여 비용을 최적화합니다.

2. **기기 매핑 최적화**: 새로운 병렬 구성으로의 전환 시 발생하는 데이터를 최적화하기 위해 바이파트 매칭을 사용합니다. 예를 들어, 기존의 인스턴스가 갖고 있는 모델 파라미터와 캐시 상태를 최대한 재활용할 수 있도록, 새로운 인스턴스와 기존 인스턴스 간 최적의 매핑을 찾습니다.

3. **상태 기반 복구**: SpotServe는 토큰 단위로 계산 상태를 저장하여 인스턴스 프리엠트 시 복구 시간을 단축합니다. 예를 들어, 프리엠트가 발생하기 전에 진행된 토큰 계산 결과를 저장해 두고, 인스턴스 복구 시 이전에 저장된 상태에서 연산을 이어갈 수 있게 합니다.

### 한계점

1. **클라우드 제공자의 프리엠티브 인스턴스 의존성**: SpotServe는 클라우드 플랫폼의 프리엠티브 인스턴스 및 그들이 제공하는 유예 시간(grace period)에 크게 의존하고 있습니다. 이는 특정 클라우드 서비스 제공자에서만 구현이 가능할 수 있다는 한계를 가지며, 유예 시간이 예기치 않게 변동되거나 짧아지면 성능이 저하될 수 있습니다.

2. **이종 인스턴스와의 호환성**: 현재는 동일한 GPU 인스턴스 유형에서만 효율적으로 동작하도록 설계되어 있습니다. SpotServe를 이종 인스턴스 또는 다른 클라우드 플랫폼 간 통합하기 위해서는 추가적인 조정이 필요합니다.

3. **추가적인 최적화 목표 부족**: 현재 SpotServe는 지연 시간 최적화에 초점을 맞추고 있지만, 높은 처리량이나 서비스 수준 목표(SLO) 충족을 위한 다양한 목표를 지원하지 않습니다.

# Q : 동적 재병렬화와 기기매핑 최적화에 대해서 자세하게 설명해 줄래? 

# Answer : 
### 1. 동적 재병렬화 (Dynamic Reparallelization)

**동적 재병렬화**는 SpotServe가 인스턴스 가용성 변화에 따라 실시간으로 LLM 병렬화 구성을 최적화하는 핵심 알고리즘입니다. 이를 통해 성능을 극대화하면서도 비용을 절감할 수 있습니다. SpotServe는 다음과 같은 병렬화 요소를 조정합니다:

- **데이터 병렬화** (\(D\)) : 데이터를 여러 인스턴스에 나누어 병렬로 처리하여 효율성을 높입니다.
- **파이프라인 모델 병렬화** (\(P\)) : 모델을 여러 단계로 나누어 파이프라인처럼 운영하여, 서로 다른 단계가 동시에 실행될 수 있도록 합니다.
- **텐서 모델 병렬화** (\(M\)) : 모델의 텐서 연산을 여러 GPU에 분산하여 병렬로 계산합니다.

이 알고리즘은 다음과 같은 방식으로 병렬화 구성을 동적으로 최적화합니다.

1. **가용 인스턴스와 작업량 파악**: SpotServe는 주기적으로 현재 가용한 인스턴스의 수와 작업량(예: 요청 도착률)을 파악합니다. 예를 들어, 작업량이 증가하거나 인스턴스가 줄어드는 경우가 있을 수 있습니다.

2. **적절한 병렬 구성 선택**: 각 시점의 병렬 구성(\(C\))에 대해 시스템의 처리량(throughput)과 지연 시간(latency)을 측정하고, 주어진 작업량에 가장 최적화된 구성을 찾습니다. 예를 들어, 요청 도착률이 증가하면, 시스템은 높은 처리량을 위해 \(D\)와 \(P\)를 높이고 \(M\)을 줄이는 구성을 선택할 수 있습니다.

3. **최적 구성 적용**: SpotServe는 사용 가능한 인스턴스 내에서 최적의 병렬화 구성으로 전환합니다. 이때 쿠흐-먼커스 알고리즘(기기 매핑 최적화)과 같은 최적화 기술을 사용하여 전환 과정에서의 데이터 이동과 통신 비용을 최소화합니다.

예시:
   - 초기 구성: 가용 인스턴스가 많아 병렬화 구성 \(C = (D=2, P=2, M=8)\)을 사용하여 처리량을 높였다면, 
   - 가용 인스턴스 감소 시: 요청이 많지 않고 가용 인스턴스가 줄어드는 상황에서는 \(C = (D=1, P=2, M=8)\)로 전환하여 비용을 줄입니다.

이를 통해 SpotServe는 **지연 시간과 처리량을 동적으로 최적화**하고, 시스템 비용을 효율적으로 관리할 수 있습니다.

### 2. 기기 매핑 최적화 (Device Mapping Optimization)

**기기 매핑 최적화**는 새로운 병렬 구성으로 전환할 때 기존의 인스턴스에 있는 데이터(모델 파라미터와 중간 계산 상태)를 최대한 재활용하여 데이터 이동과 통신 비용을 최소화하는 과정입니다. SpotServe는 **쿠흐-먼커스(Kuhn-Munkres) 알고리즘**을 이용해 최적의 매핑을 찾습니다.

이 과정은 다음 단계로 이루어집니다.

1. **바이파트 매칭 문제로 모델링**:
   - SpotServe는 기존의 GPU 인스턴스와 새로운 병렬 구성의 기기 매핑을 바이파트 그래프로 모델링합니다.
   - 그래프의 각 노드는 GPU 인스턴스(현재 상태)와 파이프라인-스테이지-셰어드의 논리적 위치(새 구성)를 나타내며, 각 엣지의 가중치는 재사용 가능한 모델 파라미터와 캐시 데이터의 양을 나타냅니다.

2. **쿠흐-먼커스 알고리즘을 통한 최적 매칭**:
   - SpotServe는 가중치가 가장 높은 매칭을 찾아 인스턴스와 논리적 위치를 연결합니다. 이를 통해 가장 많은 데이터가 재사용되도록 하고, 데이터 전송을 최소화합니다.
   - 예를 들어, 이전에 저장된 캐시 상태가 많이 포함된 인스턴스와 새로운 구성에서 해당 스테이지를 담당할 인스턴스를 연결합니다.

3. **데이터 전송 최소화**: SpotServe는 새로운 구성에서 요구되는 파라미터와 캐시 상태의 재활용 가능성을 극대화하여 **불필요한 데이터 이동을 줄이고** 전환 지연을 줄입니다.

예시:
   - 이전 병렬 구성에서 인스턴스 \(A\)가 첫 번째 파이프라인 단계의 데이터를 가지고 있다면, 새로운 병렬 구성에서도 이 데이터를 유지하기 위해 인스턴스 \(A\)를 첫 번째 단계의 노드로 매핑합니다.

**결과적으로** 기기 매핑 최적화를 통해 SpotServe는 **데이터 전송과 초기화 시간**을 줄일 수 있으며, 시스템의 전반적인 효율성을 높이는 데 기여합니다.

# Q : Dynamic Reparallelization에서 D/M/P가 바뀌면서 각각이 유리해지는 지점에 대해서 조금더 자세하게 예시를 들어서 설명해줄수 있니?

동적 재병렬화에서 SpotServe가 각 병렬화 요소(데이터 병렬화 \(D\), 파이프라인 모델 병렬화 \(P\), 텐서 모델 병렬화 \(M\))의 구성을 상황에 맞게 조정하여 성능을 최적화하는 원리를 예시와 함께 자세히 설명해드리겠습니다.

### 1. 데이터 병렬화 (\(D\))의 역할과 유리한 경우

**데이터 병렬화**는 동일한 모델을 여러 인스턴스에 복제하여 병렬로 작업을 수행하는 방식입니다. 이는 각 인스턴스가 다른 데이터 청크(batch)를 처리하는 것을 의미하므로, **처리량(throughput)을 높이기에 적합**합니다.

#### 예시:
- **상황**: 작업량(즉, 요청의 도착률)이 증가하여 인스턴스당 처리해야 하는 요청이 많아짐.
- **해결**: 데이터를 여러 인스턴스로 나누어 처리할 수 있는 \(D\)를 증가시킵니다. 예를 들어, 기존 구성에서 \(D=1\)인 경우에는 하나의 인스턴스만 사용했지만, 이를 \(D=2\)로 조정하면 각 인스턴스가 다른 배치를 동시에 처리하여 **처리량을 두 배로** 늘릴 수 있습니다.

이 경우 \(D\)가 커지면서 처리량이 증가하지만, 인스턴스 메모리 사용량도 증가하므로 메모리 제약이 없는 경우에 유리합니다.

### 2. 파이프라인 모델 병렬화 (\(P\))의 역할과 유리한 경우

**파이프라인 모델 병렬화**는 모델을 여러 단계로 나누고, 각 인스턴스가 특정 단계를 담당하여 연속적인 요청을 파이프라인처럼 처리하는 방식입니다. 이를 통해 **대기 시간을 줄이고, GPU 활용도를 높일 수 있습니다.**

#### 예시:
- **상황**: 각 요청에 대한 응답 속도가 중요하고, 인스턴스가 여러 단계의 연산을 병렬로 처리할 여유가 있음.
- **해결**: 모델을 두 단계로 나누고 각 단계가 하나의 GPU에 할당된다고 가정해봅니다. 이를 \(P=2\)로 구성하면, 첫 번째 요청이 첫 번째 단계를 처리하는 동안 두 번째 요청이 두 번째 단계를 동시에 처리할 수 있습니다.

이 방식은 특히 인스턴스 간의 통신 지연이 적을 때 유리하며, 응답 지연 시간을 줄이기 위해 유용합니다. 그러나 단계 수가 많아지면 오히려 통신 비용이 늘어날 수 있으므로, 최적의 \(P\) 값을 설정하는 것이 중요합니다.

### 3. 텐서 모델 병렬화 (\(M\))의 역할과 유리한 경우

**텐서 모델 병렬화**는 모델의 텐서 연산을 여러 GPU에 나누어 병렬로 처리하는 방식입니다. 이 방식은 특히 **모델이 매우 크고 GPU 메모리 한 개에 담기 어려운 경우**에 유리합니다.

#### 예시:
- **상황**: LLM 모델이 매우 크고, 단일 GPU 메모리로는 전체 모델을 처리할 수 없음.
- **해결**: 모델의 파라미터를 여러 인스턴스에 분할하여 나누고, 각각의 인스턴스가 모델의 일부만을 저장하고 계산하도록 설정합니다. 예를 들어, \(M=2\)로 설정하면 모델을 두 개의 GPU에 걸쳐 분할해 메모리 문제를 해결할 수 있습니다.

이 경우 **모델 크기와 GPU 메모리 용량에 따라 적절한 \(M\) 값**을 선택하는 것이 중요합니다. \(M\)이 커질수록 GPU 간 통신이 더 많이 필요하기 때문에, 통신 지연을 최소화할 수 있는 최적의 \(M\) 값을 찾아야 합니다.

---

### 예시 상황에서의 동적 재병렬화 조정

1. **요청 도착률이 높고 GPU 메모리 여유가 있는 경우**: SpotServe는 \(D\) 값을 높여 여러 인스턴스가 각기 다른 배치를 동시에 처리하게 하여 처리량을 극대화합니다. 예를 들어, 기존 구성 \(C = (D=1, P=2, M=4)\)에서 \(D\)를 2로 늘려 \(C = (D=2, P=2, M=4)\)로 전환하면, 처리량이 증가합니다.

2. **짧은 지연 시간이 중요한 경우**: \(P\) 값을 높여 각 파이프라인 단계에서 동시에 처리가 이루어지게 합니다. 예를 들어, \(P\)를 2에서 3으로 늘리면 파이프라인의 단계가 더 세분화되므로, 한 단계가 완료되기 전에 다음 단계가 시작될 수 있어 전체 지연 시간이 줄어듭니다.

3. **모델이 매우 크고 GPU 메모리가 제한된 경우**: SpotServe는 \(M\)을 증가시켜 모델 파라미터를 여러 GPU에 나누어 저장하도록 합니다. 예를 들어, \(M=4\)에서 \(M=8\)로 늘리면 각 GPU가 처리해야 할 모델 크기가 줄어 메모리 문제를 해결할 수 있습니다.

SpotServe는 이러한 동적 조정을 통해 최적의 성능을 달성하며, 각각의 경우에 맞는 \(D\), \(P\), \(M\) 구성을 실시간으로 설정하여 비용 절감과 성능 향상을 동시에 이루는 것이 목표입니다.

# Q : 쿠흐-먼커스 알고리즘에 대해서 예시를 들어 자세하게 설명해주고 이게 어떻게 이 논문에서 사용되는지도 구체적으로 알려줘

# Answer : 

쿠흐-먼커스 알고리즘(Kuhn-Munkres Algorithm), 또는 헝가리안 알고리즘은 이진 그래프에서 최대 가중치 매칭을 찾기 위한 최적화 알고리즘입니다. 이 알고리즘은 그래프의 노드 간 연결된 엣지의 가중치를 기반으로 **최적의 매칭을 찾는 데** 사용됩니다. SpotServe 논문에서는 이 알고리즘을 통해 프리엠티브 인스턴스의 재활용과 최적화된 자원 매핑을 구현하여, **불필요한 데이터 이동을 최소화하고 재활용을 극대화**하는 데 활용됩니다.

### 1. 쿠흐-먼커스 알고리즘의 원리

쿠흐-먼커스 알고리즘은 두 집합 \(A\)와 \(B\)에 있는 노드들 간의 이진 그래프에서, \(A\)와 \(B\) 간 최적의 매칭을 찾아 **전체 가중치의 합이 최대가 되는 매칭을 구하는** 방법입니다. 각 노드들은 서로 다른 가중치를 가진 엣지로 연결되며, 이 가중치가 매칭의 기준이 됩니다.

#### 예시로 설명:
다음과 같은 예시로 쿠흐-먼커스 알고리즘을 이해할 수 있습니다.

- **상황**: 네 명의 직원이 네 개의 작업을 수행해야 하고, 각 직원이 특정 작업을 수행했을 때 얻을 수 있는 성과(가중치)가 다음과 같다고 합시다.
  
  | 직원\작업 | 작업 1 | 작업 2 | 작업 3 | 작업 4 |
  | --------- | ------ | ------ | ------ | ------ |
  | 직원 A    | 5      | 8      | 7      | 6      |
  | 직원 B    | 7      | 6      | 9      | 5      |
  | 직원 C    | 8      | 7      | 6      | 9      |
  | 직원 D    | 6      | 9      | 5      | 8      |

- **목표**: 각 작업을 한 명의 직원에게 할당하면서, 모든 작업의 성과(가중치) 합이 최대가 되도록 하는 것입니다.

쿠흐-먼커스 알고리즘은 다음의 단계로 이루어집니다.

1. **초기 매칭 구성**: 행렬의 각 행과 열에서 가장 큰 값을 선택하여 초기 매칭을 만듭니다.
2. **레이블링 및 재매칭**: 이 매칭이 최적임을 보장할 수 없는 경우, **레이블을 조정**하여 더 나은 매칭을 찾습니다.
3. **매칭 갱신**: 레이블이 조정되면 새롭게 매칭을 갱신하여 최대 가중치 매칭을 찾을 때까지 반복합니다.

위 예시의 경우, 알고리즘은 최종적으로 다음과 같은 매칭을 찾을 수 있습니다:

- 직원 A → 작업 2 (가중치 8)
- 직원 B → 작업 3 (가중치 9)
- 직원 C → 작업 4 (가중치 9)
- 직원 D → 작업 1 (가중치 6)

이때, 전체 가중치 합은 \(8 + 9 + 9 + 6 = 32\)로, 이는 가능한 최대 값입니다.

### 2. SpotServe에서의 쿠흐-먼커스 알고리즘 활용

SpotServe 논문에서는 쿠흐-먼커스 알고리즘을 사용하여 **프리엠티브 GPU 인스턴스를 최적으로 매핑**합니다. SpotServe가 인스턴스의 가용성이 변화할 때, 기존 인스턴스에 남아 있는 데이터(모델 파라미터와 캐시 상태)를 최대한 재활용하여 **불필요한 데이터 이동과 통신 비용을 줄이는 것**이 목표입니다.

SpotServe에서 이 알고리즘이 적용되는 방식은 다음과 같습니다:

1. **기기 매핑 문제의 모델링**:
   - SpotServe는 기존 인스턴스에 저장된 파라미터와 캐시 상태를 재활용하기 위해, 현재 가용한 인스턴스와 새로운 병렬 구성의 논리적 위치(파이프라인 단계 및 셰어드 위치)를 **바이파트 그래프로 모델링**합니다.
   - 각 노드는 GPU 인스턴스와 새로운 구성에서 필요한 파이프라인 위치로 나타나며, 엣지의 가중치는 **재활용 가능한 데이터의 양**을 나타냅니다. 이 가중치는 재활용 가능성이 높을수록 높아집니다.

2. **최대 가중치 매칭을 위한 쿠흐-먼커스 알고리즘 적용**:
   - SpotServe는 이 매핑 문제를 **최대 가중치 매칭 문제로 변환**하여 쿠흐-먼커스 알고리즘을 통해 최적의 매칭을 찾습니다.
   - 예를 들어, 이전에 모델의 첫 번째 파이프라인 스테이지를 처리하던 인스턴스가 현재 캐시 데이터를 많이 가지고 있다면, 새 병렬 구성에서도 이 인스턴스가 첫 번째 스테이지를 담당하도록 매칭됩니다.

3. **데이터 전송 최소화**:
   - SpotServe는 이 최적 매칭을 통해 **데이터 재활용을 최대화하고, 필요하지 않은 데이터 이동을 줄임**으로써 전환 시의 초기화 시간과 통신 비용을 줄일 수 있습니다.
   - 최종 매칭이 적용되면, 각 인스턴스는 새로 할당된 역할에 맞춰 기존 데이터를 재활용해 빠르게 작업을 재개할 수 있습니다.

### 요약

쿠흐-먼커스 알고리즘은 SpotServe에서 **프리엠티브 인스턴스의 최적 매핑**을 위해 사용됩니다. 이를 통해 SpotServe는 **데이터 이동을 최소화하고, 기존 데이터의 재활용을 극대화**하여, 재병렬화 시 발생할 수 있는 비효율을 줄이고 성능을 극대화합니다. 이 최적 매핑 덕분에 SpotServe는 동적 자원 환경에서 비용 효율적인 LLM 서빙이 가능해집니다.
