---
categories:
- paper-review
- with-gpt
date: "2024-10-31"
tags: null
title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
---

논문 : https://arxiv.org/abs/2403.02310

개인의견 : 서비스를 하는 수준에서 생각하면 continuous batching같은 테크닉을 사용하면 prefill을 스케줄을 어떠한 방식으로 할지가 궁금했는데 이러한 방법으로 stall을 줄이다니 재밋는 논문이네요. 다만 prefill의 stall을 줄이기 위해서 decode는 불필요하게 sequence length를 늘려야해서 서비스 관점에서는 필요한 테크닉이지면 생각해볼 점은 있을거 같습니다.

아래글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점


논문 **"Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"**는 대형 언어 모델(LLM) 추론에서 발생하는 처리량-지연 시간 간의 트레이드오프 문제를 해결하기 위해 Sarathi-Serve라는 혁신적인 접근 방식을 제안합니다. 이 논문은 Sarathi-Serve의 강점, 독창성, 핵심 알고리즘, 예시와 한계점을 중심으로 설명드립니다.

### 강점과 독창성
1. **처리량-지연 시간 최적화**: Sarathi-Serve는 LLM 추론에서 처리량과 지연 시간 간의 균형을 맞추기 위한 기술을 도입하여, 기존 시스템에서 흔히 발생하는 성능 상의 손실을 최소화합니다. 이를 통해 처리량을 높이면서도 지연 시간은 낮추는 *chunked-prefills*와 *stall-free batching* 방식을 제시한 점에서 독창적입니다.

2. **정지 없는 스케줄링(Stall-Free Scheduling)**: 기존 스케줄러들은 입력 프리필(prefill) 또는 디코드(decode) 단계 중 하나에 우선순위를 두어 처리량 또는 지연 시간 중 하나를 희생해야 했습니다. Sarathi-Serve의 stall-free batching은 기존 디코드를 방해하지 않고 새로운 요청을 추가하여, vLLM이나 Orca 같은 반복 레벨 스케줄러의 비효율성을 해결합니다.

3. **광범위한 평가와 확장성**: Sarathi-Serve는 다양한 모델과 GPU 설정에서 테스트되었으며, 단일 GPU부터 다중 GPU 파이프라인 병렬 구성을 포함해 높은 확장성을 보여주었습니다. 특히, 8개의 A100 GPU를 사용한 Falcon-180B 모델에서 최대 5.6배의 성능 향상을 보여, 대규모 환경에서도 높은 적응력을 보입니다.

### 핵심 알고리즘 단계 (예시와 함께)
Sarathi-Serve는 다음 두 가지 주요 요소로 구성됩니다:
1. **Chunked-Prefills**: 프리필 단계를 작고 관리 가능한 청크로 나누어, 높은 계산 활용도를 유지하면서도 지연 시간을 크게 늘리지 않습니다.
   
2. **Stall-Free Batching**: Sarathi-Serve는 프리필과 디코드 요청을 함께 배치하여, 디코드 요청이 중단 없이 진행될 수 있도록 합니다. 이를 통해 처리량과 지연 시간의 최적화를 동시에 달성합니다.

**예시로 설명**:
세 개의 추론 요청 A, B, C가 있는 배치를 예로 들면:
- **초기 배치**: A와 B 요청이 디코드 단계에 있고, C 요청은 프리필 단계로 새롭게 들어옴.
- **C의 Chunked Prefill**: Sarathi-Serve는 C의 프리필 단계를 두 개의 청크(C1, C2)로 나눕니다. 첫 번째 청크 C1은 진행 중인 디코드 요청과 동시에 실행되며 지연을 발생시키지 않습니다.
- **Stall-Free Scheduling**: C의 두 번째 청크 C2는 C1이 완료된 후에 배치되며, 디코드 프로세스를 중단시키지 않고 새 요청을 배치에 추가합니다.

이는 vLLM이나 Orca가 프리필 전체를 우선순위로 처리해 디코드 요청이 지연되는 방식과 대조됩니다.

### 한계점
1. **Chunked-Prefill 오버헤드**: Sarathi-Serve의 청크 나누기 방식은 지연 시간을 줄여주지만, 작은 청크 크기는 추가적인 오버헤드를 유발할 수 있습니다. 특히, 많은 토큰을 처리하는 대형 모델에서는 캐시 접근이 반복되어 메모리 병목 현상이 발생할 가능성이 있습니다.

2. **대규모 모델에서 파이프라인 버블**: 다중 노드 구성에서 Falcon-180B와 같은 대규모 모델이 배포될 경우, 파이프라인 병렬 처리 중 연산 시간 차이로 인해 GPU가 비활성화되는 *파이프라인 버블*이 여전히 발생할 수 있습니다. Sarathi-Serve가 이를 완화하긴 했으나, 매우 대규모 설정에서는 전체 처리량이 감소하는 한계가 존재합니다.

### 결론
Sarathi-Serve는 LLM 추론에서 처리량-지연 시간 문제를 해결하기 위해 고안된 혁신적인 시스템으로, 효과적인 스케줄러 설계를 통해 높은 확장성과 성능을 제공합니다. 이를 통해 프리필을 청크 단위로 나누고 디코드가 중단되지 않도록 하는 기술을 도입해 처리량과 지연 시간 모두에서 탁월한 성능을 보입니다.


# Q : 알고리즘 1과 2에 대해서 예시를 포함해서 자세하게 설명해 줘

논문에 제시된 알고리즘 1과 2는 LLM 추론 시스템에서 요청을 배치하고 실행하는 방법을 다룹니다. 이 두 알고리즘은 프리필(prefill)과 디코드(decode) 단계의 스케줄링에 따라 처리량과 지연 시간에 큰 영향을 미칩니다. 각 알고리즘을 예시를 통해 상세히 설명드리겠습니다.

---

### 알고리즘 1: 요청 수준 배치 (Request-Level Batching)
이 알고리즘은 한 번에 전체 요청을 배치로 처리하며, 디코드가 모두 끝날 때까지 새로운 프리필 요청을 시작하지 않습니다. 이는 지연 시간(Time-Between-Tokens, TBT)을 줄이는 데 효과적이지만, 디코드 단계에서 작은 배치 크기로 인해 GPU 리소스가 낭비될 가능성이 있습니다.

#### 알고리즘 1 과정
1. **배치 초기화**: 현재 배치 `B`를 빈 상태로 초기화합니다.
2. **프리필 요청 추가**: 만약 현재 배치 `B`에 디코드 요청이 없다면, 새로운 프리필 요청을 추가합니다.
   - `get_next_request()`를 통해 새로운 요청을 가져오고, 메모리 할당이 가능한 동안 요청을 배치에 추가합니다.
3. **프리필 단계 실행**: 배치 `B`에 프리필 요청이 존재하면 프리필 작업을 실행합니다.
4. **디코드 단계 실행**: 프리필 요청이 완료된 배치는 디코드 단계를 시작합니다.
5. **완료된 요청 필터링**: 디코드 단계가 완료된 요청은 배치에서 제거됩니다.

#### 예시
가령 A, B, C 세 개의 요청이 도착했다고 가정해보겠습니다. A, B는 프리필 단계에 있고, C는 대기 중입니다.

- **초기 배치(B)**: A, B가 프리필 단계에 있습니다. `B = {A, B}`
  - `get_next_request()`로 A와 B를 순서대로 가져와 배치에 추가합니다.
  - 두 요청 모두 메모리 할당이 가능하므로 배치에 추가됩니다.

- **프리필 실행**: 배치의 프리필 단계를 실행하여 A와 B 요청의 첫 번째 토큰이 생성됩니다.

- **디코드 실행**: 프리필이 끝난 후 A와 B의 디코드가 시작됩니다. 여기서 디코드가 끝날 때까지 C 요청은 대기해야 합니다.
  - 이로 인해 A나 B가 먼저 완료되어도, 배치 내에서 남은 요청이 완료될 때까지 기다려야 합니다.

- **완료된 요청 제거**: A와 B의 디코드가 모두 끝나면 C가 배치에 추가됩니다.

이 방식은 프리필을 기다리지 않고 디코드를 먼저 처리함으로써 TBT를 최적화하지만, 디코드만 남은 배치에서 GPU 리소스가 소규모 배치로 인해 낭비될 수 있습니다.

---

### 알고리즘 2: 반복 수준 배치 (Iteration-Level Batching, vLLM)
반복 수준 배치에서는 프리필 요청을 우선 처리하되, 각 반복마다 요청을 배치에 동적으로 추가하거나 제거할 수 있습니다. 이는 요청이 길어질 경우 지연 시간을 증가시킬 수 있지만, 높은 처리량을 얻을 수 있습니다.

#### 알고리즘 2 과정
1. **배치 초기화**: 현재 배치 `B`를 빈 상태로 초기화합니다.
2. **새로운 요청 추가**: 새로운 요청 `Rnew`를 가져와 메모리 할당이 가능한 동안 `Bnew` 배치에 추가합니다.
3. **프리필 우선 처리**: `Bnew`에 프리필 요청이 있으면, 해당 요청들을 먼저 처리합니다.
   - 프리필이 완료되지 않은 요청이 있다면, 해당 요청의 프리필 단계를 실행합니다.
4. **디코드 단계 실행**: 프리필 요청이 없을 경우, 배치 내 디코드 단계를 진행합니다.
5. **완료된 요청 필터링**: 디코드 단계가 완료된 요청을 배치에서 제거하여 업데이트합니다.

#### 예시
A와 B가 먼저 도착하고, 이어서 C와 D 요청이 들어왔다고 가정해봅시다.

- **초기 배치(B)**: A와 B가 프리필 단계에 있습니다. `B = {A, B}`
  - 프리필이 우선 순위이므로, A와 B의 프리필 단계를 먼저 처리합니다.

- **새로운 요청 추가**: C와 D가 도착하면 `Bnew`에 추가합니다. 이 요청들도 프리필이 우선으로 처리됩니다.
  - 프리필 단계는 A와 B의 디코드 진행을 일시적으로 멈추게 하므로 `generation stall`이 발생할 수 있습니다.

- **프리필과 디코드 단계 병행**: 이후 C와 D의 프리필이 완료되면, `B`에 존재하는 모든 요청의 디코드 단계가 진행됩니다.

이 방법은 디코드가 진행되는 도중에도 새로운 프리필 요청이 계속 추가되므로 GPU 활용을 극대화하여 처리량을 증가시킬 수 있습니다. 하지만 프리필이 길어질 경우 디코드가 정체되며 지연 시간이 늘어날 수 있습니다.

# Q : 알고리즘 3 에 대해서도 설명해줘

# Answer: 
알고리즘 3은 Sarathi-Serve에서 제안된 *stall-free batching* 기법을 사용하여 높은 처리량과 낮은 지연 시간을 동시에 달성하는 방식입니다. 이 알고리즘은 Sarathi-Serve의 핵심 개념인 **청크 프리필(chunked-prefill)**과 **정지 없는 배치(stall-free batching)**을 활용하여, 디코드 단계가 방해받지 않으면서 새로운 프리필 요청을 효율적으로 추가합니다.

### 알고리즘 3: 정지 없는 배치 (Stall-Free Batching with Sarathi-Serve)
이 알고리즘은 기존 디코드 요청을 중단하지 않으면서 새로운 프리필 요청을 추가할 수 있도록 설계되었습니다. 이를 통해 프리필과 디코드 단계를 효율적으로 결합하여, 프리필 요청이 디코드 작업에 미치는 영향을 최소화합니다.

#### 알고리즘 3의 주요 과정
1. **최대 토큰 용량 계산**: 사용자 지정 서비스 레벨 목표(SLO)에 맞는 최대 토큰 용량을 계산합니다. 이 토큰 용량은 한 번의 배치에서 처리할 수 있는 최대 토큰 수를 의미하며, 과도한 지연 없이 처리량을 최적화하는 데 중요한 역할을 합니다.

2. **디코드 요청 배치**: 먼저, 현재 배치에 존재하는 디코드 요청을 최대 용량만큼 채웁니다. 이미 실행 중인 디코드 요청은 중단 없이 다음 배치에 포함됩니다.

3. **진행 중인 프리필 요청 추가**: 만약 프리필 단계가 완료되지 않은 요청이 있다면, 이 요청을 청크 단위로 나누어 배치에 추가합니다. 이를 통해 디코드 작업이 진행 중인 동안에도 프리필 요청이 처리될 수 있습니다.

4. **새로운 프리필 요청 추가**: 기존 요청을 처리한 후에 남은 토큰 용량이 있으면, 새로운 프리필 요청을 추가합니다. 이때도 새로운 프리필 요청을 청크 단위로 나누어, 디코드 작업에 영향을 미치지 않도록 합니다.

5. **하이브리드 배치 처리**: 마지막으로, 디코드와 프리필 요청이 포함된 하이브리드 배치를 실행합니다. 이를 통해 GPU의 처리 능력을 최대한 활용하여 효율적으로 요청을 처리할 수 있습니다.

6. **완료된 요청 제거**: 배치 처리가 끝나면 완료된 요청을 배치에서 제거하고 다음 배치를 준비합니다.

#### 예시로 설명
다음은 A, B, C, D 네 개의 요청이 들어온 상황을 예시로 든 설명입니다.

- **단계 1 - 디코드 요청 배치**: A와 B 요청이 이미 디코드 단계에 있는 상황입니다. 이 디코드 요청들은 토큰 용량이 허용하는 한도 내에서 배치에 추가됩니다. `B = {A(d), B(d)}`

- **단계 2 - 진행 중인 프리필 요청 추가**: 만약 C 요청이 프리필 단계에 있다면, 이를 청크로 나누어 배치에 추가합니다. 예를 들어, C 요청을 `C1`, `C2` 두 개의 청크로 나눌 수 있습니다. `B = {A(d), B(d), C1(p)}`

- **단계 3 - 새로운 프리필 요청 추가**: 만약 D 요청이 새로 도착했다면, 남아 있는 토큰 용량을 확인하여 D를 추가합니다. D 역시 청크로 나누어 추가됩니다. `B = {A(d), B(d), C1(p), D1(p)}`

- **단계 4 - 하이브리드 배치 처리**: 이렇게 완성된 하이브리드 배치를 GPU에 제출하여 디코드와 프리필이 동시에 실행됩니다. Sarathi-Serve는 청크 프리필을 통해 디코드가 정지되지 않고 프리필이 동시에 처리될 수 있도록 설계되어 있습니다.

- **단계 5 - 완료된 요청 제거**: 모든 요청이 처리되고 나면 배치에서 완료된 요청을 제거합니다. A 또는 B가 먼저 완료되면 그 요청은 배치에서 제외되고, 이후 새로운 요청을 배치에 추가할 수 있습니다.

### Sarathi-Serve의 장점
Sarathi-Serve의 이 정지 없는 배치 방식은 다음과 같은 장점을 가집니다:
- **높은 처리량과 낮은 지연 시간 동시 확보**: 디코드와 프리필을 청크 단위로 나누어 동시에 실행함으로써, 처리량을 높이면서도 지연 시간을 최소화할 수 있습니다.
- **효율적인 GPU 활용**: 하이브리드 배치를 통해 GPU 자원을 효율적으로 사용할 수 있어, 단일 요청 또는 디코드만으로 인한 GPU 낭비를 줄입니다.

### 결론
알고리즘 3은 프리필 요청을 청크 단위로 나누어, 디코드 작업을 중단하지 않고 효율적으로 처리하는 기법입니다. 이를 통해 Sarathi-Serve는 LLM 추론에서 발생하는 처리량-지연 시간 간의 트레이드오프를 해결하여, 높은 효율성을 보장합니다.


# Q:  이 논문이 pipeline parallelism 과 관련이 있을까? 관련성이 있으면 해당 내용에 대해서 설명해줘

# Answer : 
네, 이 논문은 **파이프라인 병렬 처리(Pipeline Parallelism)**와 관련이 있습니다. Sarathi-Serve가 파이프라인 병렬 처리를 어떻게 활용하고 최적화하는지에 대해 논문에서는 다루고 있습니다. 파이프라인 병렬 처리는 Sarathi-Serve의 핵심 구성 요소 중 하나로, 특히 대규모 모델 추론에서 높은 처리량과 효율성을 달성하는 데 기여합니다.

### 파이프라인 병렬 처리의 개념
파이프라인 병렬 처리는 대형 모델을 여러 GPU에 걸쳐 나누어 각 GPU가 특정 레이어 또는 작업을 담당하도록 하는 방식입니다. 각 GPU가 작업을 처리하는 동안 다른 GPU는 다음 작업을 준비하는 식으로 *파이프라인*처럼 구성되어, 연속적인 데이터 흐름을 통해 GPU 자원을 최대한 활용할 수 있습니다.

### Sarathi-Serve와 파이프라인 병렬 처리의 관련성
Sarathi-Serve는 대형 모델을 추론할 때 파이프라인 병렬 처리와 결합하여 성능을 최적화합니다. Sarathi-Serve가 파이프라인 병렬 처리와 관련하여 주목할 만한 두 가지 개선 사항은 다음과 같습니다:

1. **청크 프리필(chunked-prefill)과 균일한 배치 구성**:
   - Sarathi-Serve는 프리필 단계에서 입력을 작은 청크로 나누어 처리하는 방식을 통해, 각 GPU가 최대한 효율적으로 활용되도록 합니다. 파이프라인 병렬 처리 환경에서는 각 마이크로배치가 GPU 파이프라인 내에서 비슷한 처리 시간을 가져야만 불필요한 GPU 대기 시간을 줄일 수 있습니다.
   - Sarathi-Serve는 디코드와 프리필 작업을 청크 단위로 결합하여 균일한 작업 부하를 배치에 구성합니다. 이를 통해 각 GPU의 작업 시간이 비슷해져, 파이프라인 내의 *버블 현상(pipeline bubbles)*을 줄이고 GPU 자원을 더 효과적으로 사용할 수 있습니다.

2. **정지 없는 배치(Stall-Free Batching)**:
   - 파이프라인 병렬 처리에서 디코드와 프리필 작업을 효율적으로 스케줄링하기 위해 Sarathi-Serve는 정지 없는 배치 기법을 사용합니다. 이 기법은 GPU 자원을 지속적으로 활용할 수 있도록, 디코드 단계가 중단되지 않으면서 프리필 요청이 자연스럽게 이어지도록 배치를 구성합니다.
   - 파이프라인 병렬 처리 환경에서, 프리필 단계가 디코드에 간섭하여 지연이 발생하지 않도록 청크 프리필과 stall-free batching을 결합해 배치의 균형을 맞춥니다. 이로 인해 Sarathi-Serve는 여러 GPU에 걸쳐 모델을 파이프라인으로 나누어 처리할 때 지연 시간을 최소화하면서 높은 처리량을 유지할 수 있습니다.

### 논문에서 Sarathi-Serve의 파이프라인 병렬 처리 성능
실험 결과에 따르면 Sarathi-Serve는 **Falcon-180B**와 같은 대형 모델에서 파이프라인 병렬 처리를 활용하여 기존 시스템보다 최대 5.6배 높은 성능 향상을 달성했습니다. 이는 Sarathi-Serve가 파이프라인 병렬 환경에서 GPU 리소스를 균일하게 활용하고, 디코드 단계의 정체를 방지하여 발생한 결과입니다.

### 요약
Sarathi-Serve는 파이프라인 병렬 처리를 통해 GPU 리소스를 효율적으로 분산하여 대규모 모델의 추론 성능을 극대화합니다. 이 논문에서 Sarathi-Serve는 파이프라인 병렬 처리를 청크 프리필과 결합하여, 불균형한 작업 시간으로 인한 지연을 줄이고 처리량을 높이는 방식으로 설계되었습니다.

# Q :  decode와 함께 돌리기 위해서 prefill을 sequence size=1로  돌리는거야?

Sarathi-Serve에서는 **decode 단계와 함께 prefill을 동시에 처리**하기 위해 prefill 단계를 작은 청크로 나눠서 실행하지만, 각 청크의 크기를 `sequence size = 1`로 설정하지는 않습니다. 대신 **청크(prefill chunk)** 크기를 적절하게 조절하여, decode 단계와 함께 효율적으로 실행되도록 합니다. 

### 청크 프리필 (Chunked Prefill) 방식의 주요 개념
Sarathi-Serve에서 prefill 단계의 청크 크기는 `sequence size = 1`보다는 더 큽니다. 청크 크기를 지나치게 작게(예: sequence size = 1) 설정하면, 각 청크 실행 시 GPU에 반복적인 메모리 접근이 필요해져 오버헤드가 증가할 수 있습니다. 따라서 청크 크기는 적절한 수준으로 조절하여 **decode 작업에 지장을 주지 않으면서도 GPU의 처리 성능을 최대한 활용**할 수 있도록 합니다.

### 실제 Sarathi-Serve의 청크 프리필 전략
1. **청크 크기 조절**: Sarathi-Serve는 시스템의 GPU 성능과 처리량 요구에 맞춰 prefill 단계의 청크 크기를 결정합니다. 예를 들어, GPU에서 효율적인 병렬 처리가 가능한 **최소한의 크기**로 청크를 구성하지만, 이 청크가 `sequence size = 1`이 되도록 설정하지는 않습니다. 일반적으로 GPU의 **연산 집약도(arithmetic intensity)**와 **메모리 대역폭**을 고려해 적절한 청크 크기를 선택합니다.

2. **decode와의 병행 실행**: Sarathi-Serve는 청크 단위로 prefill을 나누어 decode 작업과 병행으로 수행할 수 있습니다. 이를 통해, decode 작업의 진행을 방해하지 않으면서도 prefill 작업을 GPU의 유휴 시간을 활용하여 처리합니다.

3. **토큰 용량에 따른 청크 크기 결정**: Sarathi-Serve는 특정 지연 시간 요구 사항(SLO)에 따라 한 배치에서 처리할 수 있는 최대 토큰 용량을 사전에 계산합니다. 이 토큰 용량 내에서 decode 작업과 prefill 청크가 동시에 실행되도록 하여, **stall-free batching**을 구현합니다. 예를 들어, 청크 크기를 `sequence size = 256` 또는 `512`와 같은 크기로 설정하여 GPU의 계산 자원을 효과적으로 활용합니다.

### Sarathi-Serve가 `sequence size = 1`을 사용하지 않는 이유
1. **오버헤드 증가**: `sequence size = 1`로 prefill을 돌리게 되면, 작은 단위로 처리하기 때문에 메모리 접근과 GPU 커널 호출이 반복되어 오버헤드가 증가합니다. 이는 GPU의 효율적인 계산 자원 사용을 저해할 수 있습니다.
  
2. **처리 속도 저하**: 지나치게 작은 청크 크기는 전체 처리 속도를 떨어뜨리며, Sarathi-Serve의 목적(효율적인 처리량 및 지연 시간 최적화)에 맞지 않습니다.

### 결론
Sarathi-Serve는 prefill을 `sequence size = 1`로 나누기보다는, **decode와 병행 처리하기에 적절한 크기의 청크로 나누어** 효율성을 극대화합니다. 이를 통해 GPU 자원을 최대로 활용하면서도 decode 작업을 방해하지 않는 방식으로 처리량과 지연 시간을 최적화합니다.

# Q : 디코더는 seq=1 로 돌아야하는거 아니야?

# Answer :

맞습니다. **디코드(decode) 단계**는 **자동 회귀 방식(autoregressive)**으로 작동하므로, 각 토큰을 순차적으로 생성해야 하며 **sequence size = 1**로 동작합니다. 디코딩 과정에서는 이전에 생성된 토큰을 기반으로 다음 토큰을 하나씩 예측하여 생성해야 하므로, 한 번에 하나의 토큰을 처리하는 방식이 필수적입니다. 

### 디코드 단계에서의 시퀀스 크기 = 1의 이유
디코드 단계에서는 이전 토큰을 모델에 입력으로 제공하고 다음 토큰을 예측해야 하므로, 매번 새로운 토큰을 생성할 때마다 **단일 토큰(sequence size = 1)**을 입력으로 받습니다. 이 방식은 다음과 같은 이유로 인해 필수적입니다:

1. **순차적 생성**: 디코드 단계는 이전에 생성된 토큰의 정보를 활용하여 다음 토큰을 생성하는 방식이므로, 각 단계에서 한 번에 하나의 토큰만을 예측할 수 있습니다.
   
2. **키-값 캐시(KV-cache) 활용**: 각 디코드 단계에서 이전에 생성된 토큰 정보를 메모리에 저장하고, 이를 통해 효율적인 토큰 예측이 가능합니다. 그러나 이 캐시 구조 역시 각 디코드 단계가 하나의 토큰을 생성할 때마다 업데이트되기 때문에, 시퀀스 크기는 1로 유지됩니다.

### Sarathi-Serve의 디코드와 청크 프리필 동시 실행
Sarathi-Serve의 핵심은 **디코드와 프리필을 함께 실행**하여 GPU 자원을 최대한 활용하는 것입니다. 디코드 단계는 `sequence size = 1`로 돌아가야 하지만, **프리필 단계는 여러 토큰을 동시에 처리할 수 있는 청크 단위로 나누어** 디코드와 병행 실행할 수 있습니다.

### Sarathi-Serve에서의 디코드와 프리필 병행 처리 방법
1. **디코드 단계의 순차 처리**: 디코드는 항상 `sequence size = 1`로 처리하며, 새로운 토큰을 하나씩 생성합니다.
2. **프리필의 청크 처리**: 프리필 단계는 디코드와 병행하여 처리할 수 있도록 적절한 크기의 청크로 나뉩니다. Sarathi-Serve는 이 청크들이 디코드 작업에 지장을 주지 않도록 최적의 청크 크기를 설정합니다.
3. **정지 없는 배치(Stall-Free Batching)**: Sarathi-Serve는 디코드가 진행되는 동안 청크 단위의 프리필이 중단 없이 진행되도록 스케줄링하여, 디코드와 프리필이 동시에 실행될 수 있게 합니다.

### 요약
디코드 단계는 `sequence size = 1`로 돌아가는 것이 맞습니다. Sarathi-Serve에서는 이 디코드 작업과 프리필을 적절히 병행 처리하여 GPU 효율성을 높이며, 청크 단위의 프리필을 통해 디코드 작업에 방해가 되지 않도록 설계되었습니다.

# Q : 최적의 청크는 decoder는 손해는 보지만 encoder의 크기에 맞춘다는걸까?

# Answer :
Sarathi-Serve에서 **최적의 청크 크기**는 **디코더가 최적 성능을 발휘하면서도 프리필(prefill) 단계가 효율적으로 실행될 수 있도록** 설정됩니다. 따라서 Sarathi-Serve의 청크 크기 설정은 **디코더 단계의 지연 시간에 큰 손해를 주지 않으면서도 인코더(prefill) 단계가 높은 처리량을 유지할 수 있는 균형을 찾는 것**이 목표입니다.

### Sarathi-Serve에서 최적의 청크 크기를 결정하는 원리
1. **디코더의 손해 최소화**: 디코더 단계는 `sequence size = 1`로 실행되어야 하므로, Sarathi-Serve는 디코더가 지연 없이 매번 토큰을 생성할 수 있도록 합니다. 프리필 청크가 지나치게 크면 디코더의 지연 시간(TBT)이 증가할 수 있기 때문에, 이를 방지하기 위해 청크 크기를 제한합니다.

2. **프리필의 효율 극대화**: 프리필 단계는 여러 토큰을 병렬로 처리할 수 있기 때문에, 지나치게 작은 청크로 나누면 오버헤드가 증가하여 전체 시스템의 효율이 떨어질 수 있습니다. 따라서 Sarathi-Serve는 프리필 단계가 효율적으로 돌아가도록 일정 크기 이상의 청크를 유지하면서도, 디코더에 최소한의 영향을 주는 수준에서 청크 크기를 조절합니다.

3. **최적의 청크 크기 설정**: Sarathi-Serve는 다음과 같은 기준으로 최적의 청크 크기를 설정합니다.
   - **디코더의 지연 시간 제한(TBT SLO)**: 시스템에서 허용 가능한 최대 TBT에 맞춰 청크 크기를 조절하여, 디코더의 지연이 제한을 넘지 않도록 합니다.
   - **프리필의 처리량 유지**: 프리필이 지나치게 세분화되지 않도록, 적정 수준 이상의 청크 크기를 설정하여 GPU 자원의 효율적 사용을 돕습니다.
   
   Sarathi-Serve는 이 두 가지 요소를 고려하여 *정지 없는 배치(stall-free batching)*를 통해 프리필과 디코더 단계가 함께 실행되도록 최적의 청크 크기를 찾습니다.

### Sarathi-Serve의 최적 청크 크기 결정 예시
예를 들어, 디코더가 매 토큰 생성에 10ms가 걸리고, 프리필을 통해 500토큰을 동시에 처리할 수 있다고 가정할 때:
- **디코더 손해 최소화**: 청크 크기를 500토큰 단위로 유지하면 디코더의 대기 시간이 길어질 수 있으므로, Sarathi-Serve는 프리필을 더 작은 청크(예: 256토큰)로 나누어 디코더 지연을 최소화합니다.
- **프리필 효율성 유지**: 청크 크기를 지나치게 줄이면 프리필 단계에서 반복적인 메모리 접근 오버헤드가 발생하므로, 디코더의 지연 허용 한도 내에서 가능한 큰 청크 크기를 선택합니다.

### 결론
Sarathi-Serve에서 최적의 청크 크기는 디코더의 성능 손해를 최소화하면서 프리필이 효율적으로 처리될 수 있는 균형을 찾아 설정됩니다. 이로써 디코더와 프리필 간의 상호 간섭을 줄이고, 시스템의 전반적인 처리량과 지연 시간을 최적화할 수 있게 됩니다.