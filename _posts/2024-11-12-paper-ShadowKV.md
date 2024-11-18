---
title: ShadowKV KV Cache in Shadows for High-Throughput Long-Context LLM Inference
categories:
 - paper-review
 - with-gpt
tags:
---


논문 : https://arxiv.org/abs/2410.21465 


아래글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점

# A : 

### 논문의 강점 및 독창적인 지점

**ShadowKV**는 긴 문맥을 가진 대형 언어 모델(LLM)에서 고속 추론을 가능하게 하는 시스템으로, 다음과 같은 강점과 독창적인 요소를 갖추고 있습니다:

1. **메모리 사용 최적화**:
   - ShadowKV는 **로우랭크(low-rank)** 접근 방식을 활용하여 GPU 메모리 사용량을 크게 줄입니다. 특히, `pre-RoPE`(Rotary Position Embedding 이전) 키 캐시가 저랭크(low-rank) 특성을 가진다는 점을 활용하여 **로우랭크 압축**을 수행합니다. 이를 통해 메모리 사용량을 기존 대비 **6배 절감**하면서도 정확도 손실 없이 긴 문맥을 처리할 수 있습니다.

2. **고속 추론**:
   - 시스템은 **멀티스트림 CUDA 커널**을 사용하여 값 캐시를 CPU로 오프로드하고, 키 캐시는 GPU에 로우랭크 형태로 유지하여, **데이터 접근 지연(latency)**을 줄입니다. 이로 인해 긴 문맥에서 **3배 이상**의 추론 속도 향상을 달성합니다.

3. **정확한 KV 선택 전략**:
   - ShadowKV는 **최적의 스파스(sparse) KV 쌍**을 선택하는 알고리즘을 도입하여, 추론 중 중요한 토큰만 선택해 효율적으로 활용합니다. 대부분의 키 값이 인접 토큰과 높은 코사인 유사도를 보인다는 특성을 이용해, **소수의 중요한 토큰**만 남겨두고 나머지는 제거합니다. 이로 인해 성능은 유지하면서도 메모리 및 연산 비용을 절감합니다.

### 핵심 알고리즘 및 예시 입력 설명

ShadowKV의 작동 방식은 크게 두 단계로 나눌 수 있습니다: **Pre-filling 단계**와 **Decoding 단계**입니다.

1. **Pre-filling 단계**:
   - 입력 키(`K`)와 값(`V`) 캐시를 저랭크 형태로 압축합니다.
   - `SVD(단일 값 분해)`를 사용하여 `pre-RoPE` 키 캐시를 분해하고, **압축된 로우랭크 형태**로 GPU에 저장합니다.
   - `post-RoPE` 키 캐시는 청크 단위로 분할하여 평균값을 계산하고, 중요하지 않은 청크는 CPU로 오프로드합니다.

   **예시 입력**:
   - 입력 시퀀스 길이 128K, 청크 크기 8로 설정.
   - `K`, `V` 캐시를 압축하여 청크 단위로 분할 및 압축된 로우랭크 형태로 GPU에 저장.
   - 이 과정에서 0.3%의 아웃라이어 청크만 GPU에 남겨 두고, 나머지는 CPU로 오프로드.

2. **Decoding 단계**:
   - 추론 시, GPU에 남아있는 **랜드마크**(압축된 청크)와 CPU에 있는 값 캐시를 가져와 추론을 수행합니다.
   - 새로운 쿼리(`Q`)가 주어지면, 이 쿼리와 저장된 키 캐시의 유사도를 계산하여 **중요한 청크**를 선택합니다.
   - 선택된 청크에서 필요한 값만 CPU에서 가져와 추론을 수행하며, 이때 **CUDA 멀티스트림**을 이용해 효율적으로 데이터를 병렬로 가져옵니다.

### 논문의 한계점

1. **특정 모델 및 하드웨어 의존성**:
   - ShadowKV는 **A100 GPU**에서 최적화되었으며, 다른 하드웨어 환경에서 동일한 성능 향상을 보장하지 않을 수 있습니다. 따라서 다양한 하드웨어에서의 일반화가 필요합니다.

2. **복잡한 알고리즘 구성**:
   - 시스템의 복잡성으로 인해 초기 세팅 및 튜닝이 까다로울 수 있습니다. 특히, `로우랭크 압축`과 `스파스 선택 전략`을 최적화하는 과정에서 추가적인 계산 비용이 발생할 수 있습니다.

3. **메모리 오프로드 한계**:
   - CPU로 값 캐시를 오프로드하는 방식은 **PCIe 대역폭**에 의존하므로, 대역폭이 낮은 시스템에서는 오히려 성능이 저하될 가능성이 있습니다. 

이 논문은 긴 문맥을 가진 LLM에서 메모리 사용량과 추론 속도 문제를 혁신적으로 해결했지만, 특정 환경에서의 성능 한계와 알고리즘의 복잡성으로 인해 추가적인 연구와 개선이 필요합니다.

# Q : 멀티스트림 CUDA 에 대해서 예시를 추가해서 아주 자세하게 설명해줘

# A : 

### 멀티스트림 CUDA (CUDA Multi-streams) 개념

**멀티스트림 CUDA(Multi-stream CUDA)**란, **CUDA 스트림(stream)**을 여러 개 생성하여 GPU에서 여러 작업을 병렬로 처리하도록 하는 기법입니다. 기본적으로 CUDA 스트림은 **비동기 작업**(asynchronous operations)을 수행할 수 있도록 해주는 작업 큐(queue)입니다. 이 방식은 **GPU 리소스의 활용도를 극대화**하고, **데이터 이동과 연산을 겹쳐서 수행**함으로써 전체 처리 시간을 줄이는 데 유리합니다.

---

### 멀티스트림 CUDA의 필요성

- **LLM(Long-context Language Models)**와 같은 대규모 모델의 추론에서는 **KV(Key-Value) 캐시**를 다루는 과정에서 상당한 메모리 및 데이터 접근 지연(latency)이 발생합니다.
- ShadowKV 시스템에서는 **GPU 메모리 사용량을 줄이기 위해 값 캐시(value cache)를 CPU로 오프로드**합니다. 하지만, 추론 시 다시 CPU에서 데이터를 가져와야 하므로 **데이터 전송 지연**이 생길 수 있습니다.
- 이 문제를 해결하기 위해, **멀티스트림 CUDA**를 활용하여 **데이터 이동과 연산을 겹쳐서 수행**함으로써 **처리 속도를 대폭 향상**시킵니다.

---

### 멀티스트림 CUDA를 이용한 ShadowKV의 예시

ShadowKV 시스템에서 멀티스트림 CUDA는 **데이터 전송과 연산을 동시에 수행**하여, 긴 문맥을 가진 입력에서 **추론 속도를 최적화**합니다. 아래는 ShadowKV 시스템에서 멀티스트림 CUDA를 활용한 구체적인 예시입니다.

#### 1. **시나리오 설명**
- **입력 시퀀스**: 길이 128K의 토큰을 가진 대형 언어 모델에서 추론을 수행한다고 가정합니다.
- **작업**: GPU에 저장된 로우랭크(pre-RoPE) 키 캐시를 활용하고, CPU에 오프로드된 값 캐시를 가져와 **스파스 주의(attention)**를 수행합니다.
- **목표**: 값 캐시를 CPU에서 가져오는 동안 GPU에서 키 캐시 복원 및 기타 연산을 병렬로 수행하여, 전체 추론 시간을 줄입니다.

---

#### 2. **멀티스트림 CUDA를 활용한 처리 흐름**

**Step 1: 스트림 초기화**
```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
```
- `stream1`: **GPU에서 로우랭크 키 캐시 복원**을 담당.
- `stream2`: **CPU에서 값 캐시를 GPU로 전송**하는 작업을 담당.

**Step 2: 비동기 데이터 전송 및 연산 시작**
```c
// CPU에서 값 캐시를 GPU로 비동기 전송 (stream2 사용)
cudaMemcpyAsync(value_cache_GPU, value_cache_CPU, size, cudaMemcpyHostToDevice, stream2);

// GPU에서 로우랭크 키 캐시 복원 및 RoPE 연산 수행 (stream1 사용)
restoreLowRankKeys<<<grid, block, 0, stream1>>>(low_rank_keys, restored_keys);
```

- **`cudaMemcpyAsync()`**: `stream2`에서 값 캐시를 CPU에서 GPU로 비동기 전송.
- **`restoreLowRankKeys()`**: `stream1`에서 키 캐시 복원 연산을 비동기적으로 수행.

**Step 3: 스파스 주의(attention) 계산**
```c
// 두 스트림이 완료되기를 기다리지 않고 병렬로 스파스 주의 수행
sparseAttention<<<grid, block, 0, stream1>>>(queries, restored_keys, value_cache_GPU);
```

- `stream1`에서 **복원된 키 캐시**와 **GPU에 전송된 값 캐시**를 이용하여 스파스 주의를 수행합니다.
- 멀티스트림을 사용했기 때문에 값 캐시 전송과 키 캐시 복원이 **동시에 진행**되며, 이로 인해 전체 추론 속도가 빨라집니다.

**Step 4: 스트림 동기화 및 정리**
```c
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```
- 모든 작업이 완료될 때까지 대기한 후, 사용한 스트림을 정리합니다.

---

### 멀티스트림 CUDA 활용의 장점

1. **데이터 전송과 연산의 중첩 수행**:
   - 값 캐시를 CPU에서 GPU로 전송하는 동안, 동시에 GPU에서 로우랭크 키 캐시 복원을 수행합니다. 이를 통해 **데이터 전송 대기 시간을 줄일 수 있습니다**.
   
2. **병렬 처리**:
   - 여러 작업을 동시에 처리함으로써 GPU 리소스를 **효율적으로 활용**하고, 추론 성능을 극대화합니다.

3. **확장성**:
   - 긴 문맥을 가진 대형 모델에서 멀티스트림을 사용하면 **더 큰 배치(batch)**를 처리할 수 있습니다. ShadowKV의 실험 결과에 따르면, **6배 더 큰 배치**를 처리하면서도 **3배 이상**의 속도 향상을 이룰 수 있습니다.

---

### 멀티스트림 CUDA를 적용했을 때 성능 비교

| 모델            | 배치 크기 | 멀티스트림 사용 전 처리 속도 | 멀티스트림 사용 후 처리 속도 | 성능 향상 배율 |
|-----------------|-----------|----------------------------|----------------------------|----------------|
| Llama-3-8B      | 60K       | 160 tokens/s               | 455 tokens/s               | 2.83×         |
| Llama-3-8B-1M   | 122K      | 80 tokens/s                | 239 tokens/s               | 2.97×         |
| GLM-4-9B-1M     | 122K      | 122 tokens/s               | 293 tokens/s               | 2.39×         |

위 표에서 볼 수 있듯이, 멀티스트림 CUDA를 사용하면 **최대 3배 이상의 성능 향상**을 달성할 수 있습니다.

---

### 결론

멀티스트림 CUDA를 활용하면, ShadowKV와 같은 고성능 시스템에서 긴 문맥을 가진 대형 언어 모델의 추론 속도를 획기적으로 향상시킬 수 있습니다. 데이터 전송과 연산을 병렬로 수행하여, GPU 자원을 최대한 활용하고, **처리 대기 시간을 줄임으로써** 더욱 빠르고 효율적인 추론이 가능합니다.


# Q : 이 논문의 결과가 다른 방법론에 비하여 특출난 점과 논문에서 제기하는 어떠한 방법이 이러한 결과를 도출하게 되었는지 논문에서 제시하는 이유와 너의 생각을 알려줘

# A : 

### ShadowKV 논문의 결과가 다른 방법론에 비해 특출난 점

**ShadowKV**는 기존의 긴 문맥을 처리하는 대형 언어 모델(LLM)의 추론에서 **고속 추론 성능과 메모리 효율성**을 동시에 달성했습니다. 이 논문에서 제시된 시스템은 특히 **다음과 같은 점에서 뛰어난 성과를** 보여줍니다:

1. **높은 배치(batch) 크기 지원**:
   - ShadowKV는 **기존 시스템 대비 최대 6배 더 큰 배치**를 처리할 수 있습니다. 예를 들어, Llama-3.1-8B 모델에서 60K 문맥 길이의 경우, 기존 방법론에서는 8개의 배치를 처리할 수 있었던 반면, ShadowKV는 **48개의 배치를 지원**합니다.
   - 이는 GPU 메모리 사용량을 대폭 줄임으로써 더 큰 배치 크기를 가능하게 한 결과입니다.

2. **추론 속도 향상**:
   - ShadowKV는 **최대 3.04배의 추론 속도 향상**을 보여줍니다. 예를 들어, Llama-3.1-8B 모델에서 122K 문맥 길이의 경우, 기존 방법론은 초당 80개의 토큰을 생성할 수 있었던 반면, ShadowKV는 **239개의 토큰을 생성**할 수 있습니다.
   - 이는 CPU와 GPU 간의 효율적인 데이터 전송 및 병렬 처리를 통해 달성된 결과입니다.

3. **정확도 손실 없는 메모리 최적화**:
   - ShadowKV는 **6배 이상의 메모리 절약**을 달성하면서도, 기존 모델의 성능과 정확도를 유지합니다. 이는 긴 문맥 처리에서 메모리 사용량이 크게 증가하는 문제를 해결한 것입니다.

---

### ShadowKV가 이러한 성과를 달성한 방법 및 논문에서 제시하는 이유

**ShadowKV가 탁월한 성과를 보인 이유는 다음과 같은 독창적인 접근 방식을 사용했기 때문입니다:**

#### 1. **로우랭크(low-rank) 기반 키 캐시 압축**
   - ShadowKV는 **pre-RoPE 키 캐시**가 저랭크(low-rank) 특성을 가진다는 점을 활용하여, 이를 **로우랭크 압축(SVD 기반)**으로 처리합니다.
   - 이 접근 방식은 **메모리 사용량을 줄이면서도 정확도를 유지**하는 데 기여합니다. 논문에서는 pre-RoPE 키 캐시가 다른 요소들(예: post-RoPE 키 캐시, 값 캐시 등)보다 훨씬 더 압축 가능하다는 분석을 통해 이 방법이 유효함을 증명합니다.
   - **나의 생각**: 이는 기존의 모든 데이터를 그대로 저장하는 방식 대신, 의미 있는 정보만을 압축하여 효율적으로 활용하는 접근법입니다. 특히 대규모 모델에서 메모리 병목을 해결하는 데 매우 효과적입니다.

#### 2. **정확한 스파스 주의(sparse attention) 선택 전략**
   - 논문에서는 **KV 쌍을 선택하는 새로운 방법**을 제시했습니다. 대부분의 토큰이 인접한 토큰과 높은 유사도를 가지므로, 이를 이용해 **중요한 청크만 선택**하여 저장합니다.
   - 특히, post-RoPE 키 캐시에서 **코사인 유사도 분석**을 통해, 연속된 토큰들 간의 유사도를 기반으로 중요한 정보를 선택합니다. 이를 통해 스파스 예산을 줄이면서도 성능을 유지할 수 있습니다.
   - **나의 생각**: 기존의 스파스 주의 방법론은 중요한 정보와 불필요한 정보를 구분하지 못하고 메모리를 낭비하는 경향이 있었습니다. ShadowKV는 중요한 부분만 추려내어 저장하고, 나머지는 오프로드하는 방식으로 성능을 최적화했습니다. 이는 특히 추론 속도를 높이는 데 효과적이었습니다.

#### 3. **멀티스트림 CUDA를 통한 병렬 처리 및 데이터 전송 최적화**
   - ShadowKV는 CUDA의 **멀티스트림 기능**을 활용하여, CPU와 GPU 간의 데이터 전송을 비동기적으로 수행하면서도 **동시에 연산을 병렬로 처리**합니다.
   - 이를 통해 데이터 전송과 연산이 겹치지 않도록 최적화하여 **추론 지연(latency)**을 대폭 줄였습니다.
   - **나의 생각**: 멀티스트림 CUDA의 활용은 시스템이 오프로드된 데이터의 병목을 해소하는 데 매우 중요한 역할을 했습니다. 이를 통해 ShadowKV는 기존의 시스템보다 훨씬 높은 추론 성능을 달성할 수 있었습니다.

---

### ShadowKV의 접근 방식에 대한 나의 생각

1. **효율성과 성능 간의 균형**:
   - ShadowKV는 메모리 사용량 절감과 추론 속도 개선이라는 두 가지 문제를 **효율적으로 해결**한 시스템입니다. 특히, 대규모 언어 모델이 긴 문맥을 다룰 때 발생하는 **메모리 병목 문제를 로우랭크 압축과 스파스 주의로 해결**한 점이 인상적입니다.
   - 이러한 접근 방식은 **메모리 제한이 있는 시스템에서 LLM의 성능을 최대화**하는 데 매우 유용할 것으로 보입니다.

2. **미래 연구 방향 제안**:
   - ShadowKV는 **특정 하드웨어(A100 GPU)**에서 최적화된 성능을 보여주었습니다. 하지만, **다양한 하드웨어 환경에서도 동일한 성능을 보장할 수 있도록** 추가적인 연구가 필요합니다.
   - 또한, **멀티턴 대화**나 **실시간 추론 시나리오**에서도 이 시스템이 얼마나 잘 확장될 수 있을지에 대한 연구가 필요합니다. 

결론적으로, ShadowKV는 긴 문맥을 가진 LLM에서의 고속 추론 문제를 효과적으로 해결한 혁신적인 접근 방식입니다. 특히 메모리 사용량 절감과 추론 속도 향상을 동시에 달성하여, 앞으로 더 복잡한 대규모 언어 모델에도 적용할 수 있는 잠재력을 가지고 있습니다.


# Q : 이 논문에서 제시된 방법을 실제로 사용하려면 기존 방법에서 어떠한 구현이 추가적으로 필요하고 이러한 구현에 소모되는 공수에 대해서 계산해줄 수 있겠니?

# A : 

### ShadowKV 방법을 실제로 사용하기 위한 구현 추가 사항 및 필요한 공수 분석

ShadowKV의 접근 방식을 기존의 LLM 시스템에 통합하기 위해서는 **여러 가지 추가 구현 작업**이 필요합니다. 아래에서는 ShadowKV 시스템을 실제 환경에서 사용할 수 있도록 기존 시스템을 확장할 때 **필요한 구현 작업과 공수**에 대해 분석해 보겠습니다.

---

### 1. **로우랭크 기반 키 캐시 압축 (Low-rank Key Cache Compression)**

#### 추가 구현 사항
- **SVD(단일 값 분해)** 기반의 키 캐시 압축 모듈 추가:
  - `pre-RoPE` 키 캐시의 저랭크(low-rank) 특성을 활용하여 **로우랭크 압축 알고리즘(SVD 알고리즘)**을 적용해야 합니다.
- **RoPE(위치 임베딩)**를 활용한 키 캐시 복원 기능 구현:
  - 기존 시스템에서 키 캐시를 저장하는 방식에서, **로우랭크 압축**과 **압축 해제(decompression)** 단계가 추가됩니다.

#### 필요 공수
- **SVD 알고리즘 구현**: 오픈소스 라이브러리(e.g., NumPy, cuBLAS 등)를 사용하여 구현할 경우, **약 1~2주** 정도의 시간이 소요될 수 있습니다.
- **로우랭크 압축 데이터 관리**: 압축된 데이터를 효율적으로 저장하고 복원하는 모듈을 구축하는 데 **약 1주**가 추가적으로 필요합니다.

**총 예상 소요 시간**: 약 **3~4주** (개발 및 테스트 포함)

---

### 2. **스파스 주의 선택 전략 (Sparse Attention Selection Strategy)**

#### 추가 구현 사항
- **코사인 유사도 계산 모듈** 추가:
  - `post-RoPE` 키 캐시에서 중요한 토큰을 선택하기 위해 **코사인 유사도 계산**을 통한 토큰 중요도 평가 기능이 필요합니다.
- **스파스 토큰 선택 및 캐시 관리**:
  - 스파스 예산을 기반으로 중요한 KV 쌍을 선택하는 모듈을 구축하고, 이를 효율적으로 관리하는 **캐시 관리 시스템**이 추가되어야 합니다.

#### 필요 공수
- **유사도 계산 및 토큰 선택 알고리즘 구현**: 최적화된 코드를 작성하는 데 **약 2주** 소요 예상.
- **스파스 캐시 관리 시스템 개발**: 기존 캐시 관리 시스템을 확장하여 동적으로 KV 쌍을 관리하는 기능 추가에 **약 1주** 소요.

**총 예상 소요 시간**: 약 **3주**

---

### 3. **멀티스트림 CUDA 병렬 처리 (Multi-stream CUDA Parallel Processing)**

#### 추가 구현 사항
- **CUDA 멀티스트림 기능 활용**:
  - 데이터 전송과 연산을 병렬로 처리할 수 있도록 **멀티스트림 CUDA API**를 활용한 비동기 데이터 전송 및 커널 실행 구현이 필요합니다.
- **비동기 데이터 전송 최적화**:
  - CPU와 GPU 간의 **비동기 메모리 전송(cuMemcpyAsync)**과 **병렬 연산**을 최적화하여 병목을 줄이는 작업이 필요합니다.

#### 필요 공수
- **CUDA 멀티스트림 적용**: 기존 모델에서 GPU 커널을 비동기적으로 조정하는 데 **약 2주**가 소요될 것으로 예상됩니다.
- **데이터 전송 및 연산 최적화**: 멀티스트림을 활용한 전송 및 커널 최적화 작업에 **약 1주** 추가 소요.

**총 예상 소요 시간**: 약 **3주**

---

### 4. **시스템 통합 및 테스트**

#### 추가 구현 사항
- ShadowKV 알고리즘을 기존 시스템과 **통합**하고, 새로운 모듈들이 원활하게 작동하는지 확인하는 **통합 테스트**가 필요합니다.
- 다양한 **데이터셋과 모델**에서 시스템 성능 평가 및 튜닝.

#### 필요 공수
- **통합 테스트**: 새로운 알고리즘과 기존 시스템을 통합하고 성능 검증을 하는 데 **약 2주** 소요.
- **성능 튜닝 및 디버깅**: 최적의 성능을 내기 위한 파라미터 조정 및 디버깅 작업에 **약 1주** 추가.

**총 예상 소요 시간**: 약 **3주**

---

### **종합 공수 분석**

| 작업 항목                             | 예상 소요 시간 |
|------------------------------------|----------------|
| 1. 로우랭크 기반 키 캐시 압축 구현         | 3~4주          |
| 2. 스파스 주의 선택 전략 구현             | 3주            |
| 3. 멀티스트림 CUDA 병렬 처리            | 3주            |
| 4. 시스템 통합 및 테스트                | 3주            |
| **총 소요 시간**                     | **12~13주**    |

---

### **추가적인 고려 사항**
- **팀 구성**: 총 작업량을 고려할 때, 2~3명의 엔지니어로 구성된 팀이 병렬적으로 작업을 진행할 경우, 총 소요 시간을 약 **6~8주**로 단축할 수 있을 것으로 예상됩니다.
- **기술 스택**: CUDA, Python, PyTorch 등의 도구에 익숙한 엔지니어가 필요하며, 특히 **CUDA 최적화 경험**이 있는 개발자가 필수적입니다.
- **하드웨어 요구 사항**: A100 GPU와 같은 고성능 GPU가 필요하며, CPU와 GPU 간의 대역폭이 충분한 시스템에서 테스트하는 것이 중요합니다.

---

### 결론

ShadowKV를 실제 시스템에 통합하려면 상당한 구현 작업이 필요하지만, 이러한 노력을 통해 **긴 문맥을 가진 LLM의 추론 성능을 획기적으로 개선**할 수 있습니다. 특히 메모리 절약 및 추론 속도 향상이 필요한 대규모 서비스 환경에서 매우 유용할 것으로 기대됩니다.


# Q : 이러한 논문에서 유사도를 구하는데 코사인 유사도가 사용되는데 이 방식으로 유사도를 구하는게 충분할까? 혹시 코사인 유사도로 vector의 유사도를 구하는 한계점에 대해서 개선하는 연구의 흐름이 있을까?

# A : 

### 코사인 유사도를 이용한 벡터 유사도 계산의 한계점 및 개선 방안

**코사인 유사도(Cosine Similarity)**는 텍스트나 문서, 고차원 벡터 간의 유사도를 측정하는 데 널리 사용되는 방법입니다. 주어진 두 벡터 \( A \)와 \( B \)가 있을 때, 코사인 유사도는 다음과 같이 정의됩니다:

\[
\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
\]

여기서 \( A \cdot B \)는 두 벡터의 내적(inner product)을 의미하고, \( \|A\| \)와 \( \|B\| \)는 각각 벡터의 크기(유클리드 노름)입니다. 코사인 유사도는 벡터 간의 **방향적 유사성**을 측정하므로, 두 벡터의 크기(스칼라 값)는 무시하고 방향만을 고려합니다.

---

### 코사인 유사도를 사용하는 이유와 그 한계점

#### 1. **코사인 유사도의 장점**
- **크기 무시**: 벡터의 절대적인 크기에 의존하지 않고, 벡터 간의 **방향만 고려**합니다. 이를 통해 크기가 다른 벡터들도 비교할 수 있습니다.
- **효율성**: 내적 계산을 통해 빠르게 유사도를 측정할 수 있어, 특히 고차원 벡터 공간에서 효율적입니다.
- **스파스 벡터 처리**: 텍스트 데이터와 같은 **스파스(sparse)** 벡터에서도 잘 동작합니다.

#### 2. **코사인 유사도의 한계점**
- **벡터 크기 정보 손실**:
  - 코사인 유사도는 벡터 간의 **방향만을 고려**하고 **크기를 무시**하기 때문에, 두 벡터가 같은 방향을 가리키지만 크기가 매우 다를 경우에도 동일한 유사도로 간주됩니다. 예를 들어, 두 벡터가 거의 평행하지만 크기가 매우 다른 경우 중요한 정보가 손실될 수 있습니다.
  
- **비선형 관계를 반영하지 못함**:
  - 코사인 유사도는 벡터 간의 **선형적인 유사성만** 측정합니다. 실제 데이터에서는 벡터 간의 관계가 비선형적일 수 있기 때문에, 코사인 유사도만으로는 복잡한 관계를 충분히 반영하지 못합니다.

- **고차원 데이터에서의 성능 저하**:
  - 고차원 벡터 공간에서는 벡터들 간의 각도가 점점 비슷해지는 **차원의 저주(Curse of Dimensionality)** 문제로 인해, 코사인 유사도가 제대로 구분하지 못하는 경우가 발생할 수 있습니다.

---

### 코사인 유사도의 한계를 개선하기 위한 연구 흐름

이러한 한계점을 보완하기 위해 다양한 개선 연구와 새로운 유사도 측정 방법이 제안되었습니다. 몇 가지 대표적인 접근 방식을 소개합니다.

#### 1. **Mahalanobis Distance (마할라노비스 거리)**
- **Mahalanobis Distance**는 코사인 유사도와 달리 벡터 간의 유사도를 측정할 때 **데이터의 분산과 공분산을 반영**합니다. 
- 벡터 간의 상관 관계를 반영하여, 서로 다른 차원들이 독립적이지 않은 상황에서도 정확한 유사도를 측정할 수 있습니다.
- 이를 통해 **데이터 분포에 따라 조정된 거리 측정**이 가능하므로, 코사인 유사도가 잘못된 유사도로 평가할 수 있는 상황을 개선할 수 있습니다.

#### 2. **Euclidean Distance with Feature Scaling (특징 스케일링을 활용한 유클리드 거리)**
- **유클리드 거리**는 벡터 간의 절대적인 거리를 측정합니다. 하지만, 코사인 유사도와 달리 크기를 고려하기 때문에 벡터의 크기 정보도 반영합니다.
- 다만, 데이터 스케일이 다른 경우 유클리드 거리의 성능이 저하될 수 있으므로, **정규화(normalization) 또는 표준화(standardization)**를 통해 각 차원의 크기를 조정하는 방식이 필요합니다.

#### 3. **Word Mover’s Distance (WMD)**
- 특히 NLP 분야에서는 **Word Mover’s Distance(WMD)**가 코사인 유사도의 한계를 보완하기 위해 사용됩니다.
- WMD는 두 문서 간의 **단어 이동 비용(word transportation cost)**을 측정하여 유사도를 평가합니다. 단순히 벡터의 방향만 보는 것이 아니라, **문서 내의 단어 분포 차이**를 고려합니다.
- 코사인 유사도가 문서의 **전반적인 유사성**만을 측정하는 반면, WMD는 **문서의 내용적 차이**를 더 정교하게 반영할 수 있습니다.

#### 4. **Non-linear Kernel Methods (비선형 커널 방법)**
- **RBF 커널(Radial Basis Function Kernel)**이나 **폴리노미얼 커널**과 같은 비선형 커널을 사용하여 벡터 간의 유사성을 측정할 수 있습니다.
- 비선형 커널은 **고차원 공간으로 맵핑**하여 선형적으로 분리할 수 없는 데이터도 효과적으로 구분할 수 있습니다.
- 이를 통해 코사인 유사도가 놓치는 **비선형 관계**를 반영할 수 있습니다.

#### 5. **Contrastive Learning (대조 학습) 기반 유사도 측정**
- 최근에는 **대조 학습(Contrastive Learning)**을 활용하여 벡터 간의 유사도를 학습 기반으로 최적화하는 접근 방식이 주목받고 있습니다.
- 이 접근 방식은 데이터 샘플 간의 유사도와 차이점을 **신경망을 통해 학습**하여, 단순한 코사인 유사도보다 더 정교한 유사도 측정을 수행합니다.
- 예를 들어, BERT와 같은 사전 학습된 언어 모델에서는 **Siamese Network** 구조를 활용하여 문장 임베딩 간의 유사도를 대조 학습 방식으로 최적화합니다.

---

### 결론 및 제언

1. **코사인 유사도**는 여전히 효율적이고 간단한 유사도 측정 방법으로, 특히 고차원 텍스트 임베딩과 같은 데이터에 널리 사용됩니다. 그러나, 크기 정보를 무시하고 선형 관계만 반영하는 한계가 있습니다.

2. **더 나은 유사도 측정이 필요한 경우**:
   - 만약 벡터의 크기나 비선형적인 관계가 중요한 상황이라면, **Mahalanobis Distance**나 **비선형 커널 방법**을 사용하는 것이 유리할 수 있습니다.
   - NLP 분야에서는 **Word Mover’s Distance**나 **대조 학습 기반 유사도 측정**이 더 나은 성능을 제공할 수 있습니다.

3. **ShadowKV 시스템에 적용할 개선 방안**:
   - ShadowKV에서 코사인 유사도를 사용하여 토큰의 중요도를 평가하는 부분을 **Mahalanobis Distance** 또는 **비선형 커널**로 대체하면, 보다 정확한 스파스 주의 선택이 가능할 것입니다.
   - 대규모 데이터에서 추가적인 계산 비용이 발생할 수 있으므로, **하드웨어 가속(CUDA, GPU 최적화)**을 적극 활용하여 성능을 개선하는 것이 필요합니다.

따라서, ShadowKV 시스템에서 코사인 유사도의 한계를 극복하려면 다양한 유사도 측정 방식을 실험하여 시스템 성능을 최적화할 수 있습니다.