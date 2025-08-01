---
categories:
- paper-review
- with-gemini-2.5-pro(preview)
- MLSYS2025
date: "2025-06-05"
title: 'SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware
  Scheduling'
---

[논문 링크](https://openreview.net/pdf?id=ubIvpetAd6)

# Q : 이 논문을 아주 자세하게 읽고 논문의 강점과 독창적인 지점을 설명해주고 핵심 알고리즘을 예시 입력을 들어서 전체적인 과정을 설명해줘 추가적으로 논문의 한계점에 대해서도 알려줘



이 SOLA 논문은 대규모 언어 모델(LLM) 서비스에서 서비스 수준 목표(SLO), 특히 첫 토큰까지의 시간(TTFT)과 출력 토큰당 시간(TPOT) 달성률을 최적화하기 위한 새로운 스케줄링 프레임워크를 제안합니다. 기존 시스템의 고정된 원칙에 따른 스케줄링 방식이 야기하는 TTFT와 TPOT 간의 지연 시간 분포 편향 및 요청 간 분포 분산 문제를 해결하고자 합니다. [cite: 3]

**논문의 강점 및 독창적인 지점**

1.  **세분화된 스케줄링 프레임워크 공식화**:
    * 기존의 고정된 스케줄링 정책과 달리, SOLA는 반복(iteration) 단위로 요청 실행 순서와 각 반복의 작업량을 유연하게 제어할 수 있는 세분화된 스케줄링 프레임워크를 제시합니다. [cite: 4, 5, 64, 65] 이를 통해 매 반복마다 스케줄링 전략을 동적으로 변경할 수 있습니다. [cite: 52, 136]

2.  **상태 인식(State-Aware) 스케줄링 전략**:
    * SOLA는 개별 요청 관점의 상태(실시간 TTFT/TPOT, 남은 출력 길이 등)와 시스템 관점의 상태(메모리 사용률, 전체 요청의 TTFT/TPOT SLO 달성률 통계 등) 두 가지를 모두 모니터링합니다. [cite: 6, 59, 66, 186]
    * 이러한 상태 정보를 기반으로 시스템은 TTFT와 TPOT 간의 트레이드오프, 그리고 서로 다른 요청들 간의 트레이드오프를 효과적으로 관리하여 전체 SLO 달성률을 극대화합니다. [cite: 6, 60, 62, 67]

3.  **제약 조건 최적화(Constrained Optimization) 기반 전략 생성**:
    * 매 반복마다 SOLA는 현재 시스템 및 요청 상태, 비용 모델 예측, 주어진 SLO를 바탕으로 제약 조건 최적화 문제를 해결하여 최적의 스케줄링 전략(실행 순서 $F_i$, 작업량 $n_i, k_i$)을 동적으로 생성합니다. [cite: 61, 144, 197]
    * 특히, TTFT와 TPOT 중 어떤 것을 우선 최적화할지, 다른 하나는 SLO를 만족하는 제약 조건으로 설정하는 방식을 동적으로 전환합니다 (Table 2 참고). [cite: 201, 202, 203]

4.  **두 가지 핵심 트레이드오프 관리**:
    * **TTFT와 TPOT 간의 트레이드오프 (그림 2a)**: 한 요청에 대해 TTFT가 SLO 여유가 있다면 TPOT를 개선하거나, 그 반대의 경우를 통해 두 지표 모두 SLO를 만족하도록 조정합니다. [cite: 25, 111]
    * **서로 다른 요청 간의 트레이드오프 (그림 2b)**: 지연 시간이 긴 요청에 더 많은 리소스를 할당하고, 짧은 요청은 SLO를 만족하는 선에서 조율하여 전체적인 SLO 달성률을 높입니다. [cite: 25, 112]

5.  **실증적 성능 향상**:
    * 실험 결과, SOLA는 기존 최첨단 시스템 대비 SLO 달성률을 현저히 개선(예: 45.5%에서 99.4%로)하고, 평균적으로 1.04배에서 1.27배 더 많은 요청을 처리할 수 있음을 보여줍니다. [cite: 7, 69, 287, 288] 스케줄링 오버헤드는 0.40%-0.45%로 미미합니다. [cite: 259, 306]

**핵심 알고리즘 (상태 인식 스케줄링) 과정 예시**

SOLA의 핵심은 매 반복(iteration)마다 **상태 모니터(State Monitor)**가 시스템과 요청들의 상태를 업데이트하고, 이를 바탕으로 **전략 생성기(Strategy Generator)**가 다음 반복을 위한 스케줄링 전략을 최적화하는 것입니다 (그림 4 참고 [cite: 157]).

**입력 예시**:
* **SLOs**: $T^{TTFT} = 500ms$, $T^{TPOT} = 200ms$ [cite: 16]
* **요청 큐 (Waiting Queue, $Q_{i}^{wait}$)**:
    * Req1: (도착, 아직 처리 안됨, 입력 길이 $l^{in}_{Req1}$)
    * Req2: (도착, 아직 처리 안됨, 입력 길이 $l^{in}_{Req2}$)
* **실행 큐 (Running Queue, $Q_{i-1}^{run}$)**:
    * Req0: (현재 디코딩 중, 현재까지 생성된 토큰 수 $l^{out}_{i-1,Req0}$, 현재까지 평균 $t^{TPOT}_{i-1,Req0}$, 남은 예측 토큰 수 $l^{left}_{i-1,Req0}$)
* **시스템 상태**: $m_{i-1}^{ratio}$ (KV 캐시 사용률), $p_{i-1}^{TTFT}$, $p_{i-1}^{TPOT}$ (이전 반복까지의 시스템 전체 TTFT/TPOT의 SLO 대비 비율)
* **비용 모델**: $C^p, C^d$ (오프라인 프로파일링 및 온라인 튜닝된 지연시간 예측 모델) [cite: 245, 246, 254]

**전체 과정 (i번째 반복)**:

1.  **상태 모니터 (State Monitor - Sec 4.3)**:
    * (i-1)번째 반복의 실행 결과를 바탕으로 Req0의 $t^{TPOT}_{i,Req0}$, $l^{out}_{i,Req0}$ 등을 업데이트합니다. [cite: 187]
    * 새로 도착한 요청이 있다면 $Q_{i}^{wait}$에 추가합니다 (여기서는 Req1, Req2가 이미 대기 중이라고 가정).
    * 시스템 전체의 $p_{i}^{TTFT}$와 $p_{i}^{TPOT}$를 계산합니다. [cite: 188, 189] 예를 들어, 현재 많은 요청들이 TPOT SLO를 겨우 만족하거나 초과하고 있고 ($p_{i}^{TPOT} \approx 1$ 또는 $>1$), TTFT는 여유가 있다면 ($p_{i}^{TTFT} < 1$), 시스템은 TPOT 개선에 더 중점을 둘 필요가 있다고 판단할 수 있습니다.

2.  **전략 생성기 (Strategy Generator - Sec 4.4)**:
    * **제약 조건 최적화 문제 선택 (Sec 4.4.1, Table 2)**: [cite: 184]
        * 상태 모니터의 $p_{i}^{TTFT}$와 $p_{i}^{TPOT}$ 값을 비교합니다.
        * 예시: 만약 $p_{i}^{TPOT} > p_{i}^{TTFT}$ 이고 $p_{i}^{TPOT} > 1$ (TPOT가 SLO를 더 많이 위반)이라면, "TPOT 최적화, TTFT는 SLO 만족"을 목표로 설정합니다. 즉, `min max_r($t_{i,r}^{TPOT}$)`  s.t. `max_r($t_{i,r}^{TTFT}$) $\le T^{TTFT}$`. [cite: 203]
    * **계층적 우선순위 결정 ($F_i$) (Sec 4.4.2, Fig 6)**: [cite: 219]
        * **단계 수준 우선순위**: TPOT 최적화가 목표이므로 디코딩 단계 요청(Req0)을 프리필 단계 요청(Req1, Req2)보다 우선 고려합니다. [cite: 210]
        * **요청 수준 우선순위**:
            * Req0 (디코딩): 현재 $t^{TPOT}_{i,Req0}$와 예측된 남은 디코딩 시간을 고려하여 예상 완료 시점의 $TPOT_{Req0}$를 계산합니다. [cite: 216]
            * Req1, Req2 (프리필): 각 요청의 예상 $TTFT$ (대기시간 + $C^p(Req)$)를 계산합니다. [cite: 214]
            * $F_i$는 이 계산된 값들을 기준으로 정렬 순서를 결정합니다. (예: Req0 -> Req1 -> Req2 순서)
    * **제약된 작업량 결정 ($n_i, k_i$) (Sec 4.4.3, Fig 6)**: [cite: 219]
        * TPOT 최적화가 목표이므로, 먼저 디코딩 요청(Req0)을 실행 큐에 넣습니다. [cite: 235]
        * 그 다음, 프리필 요청(Req1, Req2)을 추가할지 결정합니다. 이때, 이들을 추가함으로써 어떤 요청이라도 TTFT SLO($T^{TTFT}$)를 위반하지 않도록 하는 최대 프리필 요청 수 $n_i$를 결정합니다 (Eq. 2 참고). [cite: 235, 236] 예를 들어, Req1을 추가해도 모든 실행 중/예정 요청의 예상 TTFT가 SLO 내에 있지만, Req2까지 추가하면 Req2의 예상 TTFT가 SLO를 초과한다면 $n_i$는 Req0을 포함하여 (기존 실행큐 크기 + 1)이 될 수 있습니다. $k_i$는 무효화됩니다.

3.  **스케줄링 실행 (Algorithm 1 - Sec 4.1.2)**: [cite: 156]
    * 입력: $Q_{i}^{wait}$=[Req1, Req2], $Q_{i-1}^{run}$=[Req0], 결정된 $F_i, n_i, k_i$.
    * $Q_{i}^{run}$을 초기화합니다 (또는 Req0으로 시작).
    * $F_i$에 따라 정렬된 $Q_{i}^{wait}$ (예: Req1, Req2 순)에서 요청을 하나씩 가져옵니다.
    * **Req1 처리 시도**:
        * Req1을 $Q_{i}^{run}$에 추가했을 때의 예상 최대 KV 캐시 메모리 사용량($m^{peak}$)을 예측합니다. [cite: 172]
        * 만약 $m^{peak}$이 가용 메모리($M \times (1-m_i^{ratio})$)를 초과하지 않고, $Q_{i}^{run}$의 요청 수가 $n_i$를 넘지 않으면 Req1을 $Q_{i}^{run}$으로 옮기고 $Q_{i}^{wait}$에서 제거합니다. [cite: 173] (여기서는 $k_i$는 무효화되었으므로 토큰 수 제한은 고려 안 함)
    * **Req2 처리 시도**: (Req1이 성공적으로 추가되었다면)
        * 유사하게 메모리 및 $n_i$ 제약조건을 확인하여 Req2를 $Q_{i}^{run}$에 추가할지 결정합니다.
    * 최종적으로 $Q_{i}^{run}$에 포함된 요청들이 현재 반복(i-th iteration)에서 실행됩니다.

4.  **실행 및 다음 반복 준비**:
    * 선택된 요청들이 GPU에서 실행됩니다.
    * 실행 후, 각 요청의 실제 지연 시간, 토큰 생성 수 등의 상태가 업데이트되고, 이는 다음 (i+1)번째 반복의 상태 모니터링 단계에 입력으로 사용됩니다. 비용 모델도 이 실제 지연 시간 정보를 바탕으로 온라인 튜닝될 수 있습니다. [cite: 255]

이 과정이 매 반복마다 동적으로 수행되면서, SOLA는 변화하는 시스템 상황과 다양한 요청들의 요구사항에 맞춰 SLO 달성률을 지속적으로 최적화합니다.

**논문의 한계점**

1.  **비용 모델의 정확성 및 일반화**:
    * SOLA의 성능은 프리필 및 디코딩 단계의 지연 시간을 예측하는 비용 모델($C^p, C^d$)의 정확성에 크게 의존합니다. [cite: 244] 논문에서는 다항식 모델을 사용하고 온라인 튜닝을 수행하지만[cite: 247, 254, 255], 매우 동적인 워크로드나 새로운 모델 아키텍처에 대해서는 예측 오차가 커질 수 있습니다. 실제로 arXiv 데이터셋에서 8.91%의 평균 절대 오차를 보였습니다. [cite: 302]
2.  **출력 길이 예측의 어려움**:
    * TPOT 관련 의사결정, 특히 디코딩 요청의 우선순위 결정 및 남은 작업량 예측은 출력 길이에 대한 예측($l_{i,r}^{left}$)을 필요로 합니다. [cite: 192, 216] SOLA는 과거 요청들의 출력 길이 분포를 기반으로 이를 예측하지만[cite: 193, 194], 정확한 예측은 본질적으로 어렵고, 특히 이전에 보지 못한 유형의 요청에 대해서는 부정확할 수 있습니다. 논문에서도 길이 예측 자체는 주된 연구 초점이 아니라고 언급합니다. [cite: 218]
3.  **상태 공간 및 최적화 문제의 복잡성**:
    * 다양한 상태 변수들을 모니터링하고 [cite: 152] 매 반복마다 제약 조건 최적화 문제를 푸는 것은 계산 오버헤드를 유발할 수 있습니다. [cite: 61] 현재 구현에서는 오버헤드가 미미하다고 보고되었지만($0.40\%-0.45\%$)[cite: 306], 요청 수가 매우 많아지거나 상태 변화가 극심한 환경에서도 낮은 오버헤드를 유지할 수 있을지는 추가 검증이 필요할 수 있습니다.
4.  **하이퍼파라미터 민감성**:
    * 비용 모델 튜닝 시 사용되는 신뢰도 계수 $\alpha$ [cite: 256] 나 제약 조건 완화 시 사용되는 $a\%$ 같은 하이퍼파라미터들의 설정에 따라 성능이 영향을 받을 수 있으나, 이에 대한 민감도 분석은 자세히 다뤄지지 않았습니다.
5.  **동일 GPU 내 프리필/디코딩 집중**:
    * 논문은 프리필과 디코딩 단계가 동일 GPU에서 처리되는 시스템에 주로 초점을 맞추고 있다고 명시합니다. [cite: 106] 프리필과 디코딩이 물리적으로 분리된 GPU에서 처리되는 최근의 이종(disaggregated) 시스템 아키텍처에서의 SOLA의 효과는 추가적인 연구가 필요할 수 있습니다.

# Q : 논문의 핵심알고리즘을 예시를 들어 아주 자세하게 설명해주고 정리해 줘



SOLA의 핵심 알고리즘은 **상태 인식 스케줄링**으로, 매 반복(iteration)마다 시스템과 개별 요청의 상태를 면밀히 모니터링하고, 이를 바탕으로 최적의 스케줄링 전략을 동적으로 생성하여 서비스 수준 목표(SLO) 달성률을 극대화합니다.

---
## SOLA 핵심 알고리즘 상세 설명 (예시 포함)

SOLA의 스케줄링 과정은 크게 **상태 모니터링(State Monitor)**, **전략 생성기(Strategy Generator)**, 그리고 **스케줄링 실행(Scheduling Execution)**의 세 단계로 나눌 수 있습니다[cite: 157, 179].

**시나리오 예시**:

* **SLOs**: TTFT $\le$ 500ms, TPOT $\le$ 200ms[cite: 16].
* **초기 상태**:
    * **실행 큐 ($Q_{run}$)**: `Req0` (디코딩 중, 현재 TPOT 180ms, 남은 예측 토큰 50개).
    * **대기 큐 ($Q_{wait}$)**: `Req1` (새로 도착, 프리필 대기), `Req2` (새로 도착, 프리필 대기).
    * **시스템 상태**: KV 캐시 사용률 60%, 이전 반복까지의 시스템 전체 TTFT SLO 만족률($p^{TTFT}$) 0.8 (양호), TPOT SLO 만족률($p^{TPOT}$) 0.95 (주의).
* **비용 모델 ($C^p, C^d$)**: 현재 워크로드에 맞게 튜닝된 지연 시간 예측 모델.

**i번째 반복에서의 알고리즘 수행 과정**:

1.  **상태 모니터링 (State Monitor - Sec 4.3)**
    * `Req0`의 디코딩이 한 번 완료되어 TPOT, 생성된 토큰 수 등이 업데이트됩니다.
    * 시스템 전체의 $p^{TTFT}$와 $p^{TPOT}$가 갱신됩니다. 예시에서는 $p^{TPOT}$가 0.95로 SLO에 근접하고 있으므로, TPOT 관리가 중요하다고 인식합니다[cite: 188].

2.  **전략 생성기 (Strategy Generator - Sec 4.4)**
    * **제약 조건 최적화 문제 선택 (Sec 4.4.1, Table 2)**:
        * 현재 $p^{TPOT}$ (0.95)이 $p^{TTFT}$ (0.8)보다 높고, SLO에 더 가깝거나 초과할 가능성이 있으므로 "TPOT 최적화, TTFT는 SLO 만족"을 목표로 설정합니다[cite: 202, 203].
            * 목표: $\min \max_{r}(t_{i,r}^{TPOT})$
            * 제약: $\max_{r}(t_{i,r}^{TTFT}) \le T^{TTFT}$
    * **계층적 우선순위 결정 ($F_i$) (Sec 4.4.2, Fig 6)**:
        * **단계 수준 우선순위**: TPOT 최적화가 목표이므로 디코딩 요청(`Req0`)을 프리필 요청(`Req1`, `Req2`)보다 우선적으로 고려합니다[cite: 210].
        * **요청 수준 우선순위**:
            * `Req0` (디코딩): 현재 $t^{TPOT}$와 남은 토큰 수를 기반으로 예상 최종 TPOT를 계산합니다[cite: 216].
            * `Req1`, `Req2` (프리필): 각 요청의 예상 TTFT(대기 시간 + $C^p(\text{요청})$)를 계산합니다[cite: 214].
            * $F_i$는 이 계산된 값들을 기준으로 정렬 순서를 정합니다. 예를 들어, 디코딩 중인 `Req0`가 가장 높은 우선순위를 갖고, 프리필 요청 중에서는 예상 TTFT가 더 긴 `Req2`가 `Req1`보다 높은 우선순위를 가질 수 있습니다 (긴급도). (예: `Req0` $\rightarrow$ `Req2` $\rightarrow$ `Req1`)
    * **제약된 작업량 결정 ($n_i, k_i$) (Sec 4.4.3, Fig 6)**:
        * TPOT 최적화가 목표이므로, 우선 디코딩 요청(`Req0`)을 실행 큐에 포함시킵니다[cite: 235].
        * 그 다음, 프리필 요청(`Req1`, `Req2`)을 추가할지 결정합니다. 이때, 어떤 프리필 요청을 추가하더라도 기존 요청(`Req0`) 및 추가되는 프리필 요청들의 예상 TTFT가 SLO($T^{TTFT}$)를 위반하지 않도록 하는 최대 프리필 요청 수 $n_i$를 설정합니다[cite: 235]. 예를 들어, `Req2`를 추가하면 `Req0`의 예상 TTFT가 SLO를 초과한다면, `Req1`만 추가하는 식으로 $n_i$가 결정됩니다. $k_i$는 이 경우 무효화됩니다[cite: 236].

3.  **스케줄링 실행 (Algorithm 1 - Sec 4.1.2)**
    * 입력: $Q_{i}^{wait}$=[`Req1`, `Req2`], $Q_{i-1}^{run}$=[`Req0`], 그리고 위에서 결정된 전략 $F_i, n_i, k_i$.
    * $Q_{i}^{run}$은 `Req0`으로 시작합니다.
    * $F_i$에 따라 정렬된 $Q_{i}^{wait}$ (예: `Req2` $\rightarrow$ `Req1`)에서 요청을 하나씩 가져옵니다.
    * **`Req2` 처리 시도**:
        * `Req2`를 $Q_{i}^{run}$에 추가했을 때의 예상 최대 KV 캐시 메모리 사용량($m^{peak}$)을 예측합니다[cite: 172].
        * 만약 $m^{peak}$이 가용 메모리를 초과하지 않고[cite: 172], $Q_{i}^{run}$의 요청 수가 $n_i$를 넘지 않으면 [cite: 173] `Req2`를 $Q_{i}^{run}$으로 옮기고 $Q_{i}^{wait}$에서 제거합니다.
    * **`Req1` 처리 시도**: (`Req2`가 성공적으로 추가되었다고 가정)
        * 유사하게 메모리 및 $n_i$ 제약조건을 확인하여 `Req1`을 $Q_{i}^{run}$에 추가할지 결정합니다.
    * 최종적으로 $Q_{i}^{run}$에 포함된 요청들(예: `Req0`, `Req2`)이 현재 반복(i-th iteration)에서 실행됩니다.

4.  **실행 및 다음 반복 준비**:
    * 선택된 요청들이 GPU에서 처리됩니다.
    * 실행 후, 각 요청의 실제 지연 시간, 생성된 토큰 수 등의 상태가 업데이트되고[cite: 187], 이는 다음 (i+1)번째 반복의 상태 모니터링 단계에 사용됩니다. 비용 모델도 이 실제 측정값을 사용하여 지속적으로 튜닝됩니다[cite: 255, 256].

---
## 정리

SOLA의 핵심 알고리즘은 다음과 같이 요약할 수 있습니다:

1.  **지속적인 상태 모니터링**: 개별 요청(지연 시간, 남은 길이 등)과 시스템 전체(메모리, SLO 달성률 통계)의 상태를 매 반복마다 업데이트합니다[cite: 66, 180].
2.  **동적 전략 생성**:
    * 모니터링된 상태를 기반으로 **제약 조건 최적화 문제**를 설정하여 TTFT와 TPOT 중 어떤 지표를 우선적으로 최적화할지, 다른 지표는 어떤 제약 조건으로 둘지 결정합니다[cite: 197, 201].
    * 결정된 최적화 목표에 따라 **계층적 우선순위($F_i$)**를 사용하여 요청 실행 순서를 정합니다 (단계 수준, 요청 수준)[cite: 208, 209].
    * 설정된 제약 조건(예: TTFT SLO)을 만족하는 범위 내에서 실행할 최대 **작업량($n_i$ 또는 $k_i$)**을 결정합니다[cite: 226].
3.  **선택적 스케줄링 실행**: 생성된 전략($F_i, n_i, k_i$)과 메모리 제약 등을 고려하여 대기 큐에서 실행 큐로 요청을 선별적으로 옮겨 실행합니다[cite: 167, 175].
4.  **반복적 개선**: 실행 결과를 다시 상태 모니터링에 반영하고, 비용 모델을 튜닝하여 다음 반복의 스케줄링 정확도를 높입니다[cite: 181].

이러한 반복적인 상태 인식과 동적 전략 최적화를 통해 SOLA는 고정된 규칙에 기반한 스케줄러보다 훨씬 유연하게 다양한 상황에 대처하며 SLO 달성률을 효과적으로 높입니다[cite: 62, 312].


# Q : 이 논문에서 제시한 결과를 자세하게 보고 다른 방법론에 비하여 특출난 점과 논문에서 제기하는 어떠한 방법이 이러한 결과를 도출하게 되었는지 논문에서 제시하는 이유와 너의 생각을 알려줘



SOLA는 다른 방법론에 비해 서비스 수준 목표(SLO) 달성률과 처리량에서 눈에 띄는 향상을 보입니다. 이는 상태 인식 스케줄링과 세분화된 제어를 통해 요청 및 시스템 상태에 동적으로 적응하여 TTFT와 TPOT 간, 그리고 요청들 간의 트레이드오프를 효과적으로 관리하기 때문입니다.

---
## SOLA 결과 분석 및 성공 요인

### 결과 요약 📊

SOLA는 다양한 벤치마크에서 기존 방법론들(vLLM-S, vLLM-D, SJF) 대비 우수한 성능을 보였습니다[cite: 286].

* **SLO 달성률 향상**: 가장 두드러지는 결과는 SLO 달성률의 대폭적인 개선입니다. 예를 들어, 특정 조건에서 기존 시스템의 SLO 달성률이 45.5%였던 것을 SOLA는 99.4%까지 끌어올렸습니다[cite: 7]. 전반적으로 다양한 벤치마크에서 90% SLO 달성률 기준 평균 1.08배~1.27배, 99% SLO 달성률 기준 평균 1.04배~1.11배 더 많은 요청을 처리했습니다[cite: 258, 287, 288].
* **처리량 (Goodput) 증가**: SLO를 만족하면서 처리할 수 있는 요청의 비율인 goodput이 크게 향상되었습니다. 그림 8은 90% 및 99% SLO 달성률에서 SOLA가 다른 방법론들보다 높은 정규화된 goodput을 달성했음을 보여줍니다[cite: 276, 277].
* **지연 시간 분포 개선**: 그림 1과 그림 9에서 볼 수 있듯이, SOLA는 TTFT와 TPOT 간의 분포 편향을 줄이고 요청 간 지연 시간 분산을 낮춰 요청 대부분이 SLO 경계 내에 집중되도록 합니다[cite: 1, 15, 280, 281, 293, 295]. 이는 TTFT와 TPOT 간의 효과적인 트레이드오프를 의미하며, 한쪽 지표에 치우치지 않고 균형 잡힌 성능을 제공합니다[cite: 293].
* **낮은 오버헤드**: 이러한 성능 향상에도 불구하고 SOLA의 스케줄링 오버헤드는 0.40%~0.45% 증가에 그쳐 매우 낮습니다[cite: 259, 279, 306].

---
### SOLA의 특출난 점 (대비 우위) ✨

1.  **동적 및 적응적 스케줄링**: 기존 방법론들이 고정된 우선순위(예: 프리필 우선 또는 디코딩 우선)나 단순한 규칙(예: SJF)에 의존하는 반면[cite: 3, 39, 41, 46, 48], SOLA는 매 반복마다 시스템과 요청의 현재 상태를 분석하여 스케줄링 전략(요청 실행 순서, 작업량)을 동적으로 조정합니다[cite: 6, 52, 65, 311]. 이는 정적인 환경에서는 잘 작동할 수 있지만 변화하는 워크로드에는 최적이 아닌 기존 방식과의 큰 차별점입니다.
2.  **SLO 직접 인식 및 최적화**: 많은 기존 방법론들은 SLO를 직접적으로 고려하지 않고 전체 지연 시간 최소화나 처리량 극대화에 초점을 맞춥니다[cite: 50]. 반면 SOLA는 TTFT와 TPOT라는 두 가지 SLO 지표를 명시적으로 인지하고, 이 둘 사이의 균형을 맞추는 제약 조건 최적화 문제를 해결하여 SLO 달성률 자체를 극대화하려고 시도합니다[cite: 60, 67, 197].
3.  **종합적인 트레이드오프 관리**: SOLA는 그림 2에서 강조된 것처럼 (1) 단일 요청 내 TTFT와 TPOT 간의 트레이드오프와 (2) 서로 다른 요청들 간의 지연 시간 트레이드오프를 모두 고려합니다[cite: 6, 60, 110, 293, 295]. 이는 기존 방법론들이 간과하거나 단편적으로만 다루던 문제입니다[cite: 46]. 예를 들어, vLLM의 기본 전략은 TTFT를 개선하지만 TPOT를 악화시키는 경향이 있고[cite: 39], SplitFuse는 이를 완화하려 하지만 SOLA만큼 정교한 균형을 맞추지는 못합니다[cite: 99, 100].

---
### 결과 도출의 핵심 방법 (논문 주장 및 개인 의견) 💡

논문에서 제시된 SOLA의 성공 요인은 다음과 같습니다:

1.  **세분화된 스케줄링 프레임워크 (Fine-grained Scheduling Framework)**: 논문은 기존 스케줄링 방식이 반복 중 고정된 원칙을 따르는 "coarse-grained"라고 지적하며[cite: 3], SOLA는 반복(iteration) 단위로 요청 실행 순서($\mathcal{F}_i$)와 작업량($n_i, k_i$)을 유연하게 제어할 수 있는 "fine-grained" 프레임워크를 제안합니다[cite: 4, 52, 64, 65, 311].
    * **개인 의견**: 이러한 세분성은 시스템이 매우 짧은 시간 간격으로 변화하는 상황에 민첩하게 대응할 수 있게 하는 핵심 요소입니다. 고정된 정책으로는 다양한 요청 길이와 도착 패턴, 시스템 부하 변화에 최적으로 대응하기 어렵습니다.

2.  **상태 인식 (State-Awareness)**: SOLA는 개별 요청의 상태(실시간 TTFT/TPOT, 남은 출력 길이 등)와 시스템 전체의 상태(KV 캐시 사용률, 전반적인 SLO 달성률 통계 등)를 지속적으로 모니터링합니다[cite: 6, 59, 66, 180].
    * **개인 의견**: "아는 것이 힘이다"라는 말처럼, 현재 상태를 정확히 알아야 최적의 결정을 내릴 수 있습니다. SOLA는 이 정보를 바탕으로 어떤 요청에 자원을 더 할당하고, 어떤 요청을 잠시 대기시킬지 등을 결정하여 전반적인 SLO를 개선합니다. 특히 $p^{TTFT}$ 와 $p^{TPOT}$ 같은 시스템 수준의 SLO 달성률 지표는 매크로 레벨에서의 전략 방향을 설정하는 데 중요하게 작용합니다[cite: 188, 190].

3.  **제약 조건 최적화 기반 전략 생성 (Constrained Optimization)**: 단순히 휴리스틱에 의존하기보다, SOLA는 매 반복마다 모니터링된 상태와 비용 모델 예측치를 사용하여 제약 조건 최적화 문제를 공식화하고 해결함으로써 스케줄링 전략을 도출합니다[cite: 61, 67, 144, 148, 197]. Table 2는 SLO 상태에 따라 최적화 목표와 제약 조건을 동적으로 전환하는 로직을 보여줍니다[cite: 183, 184, 202].
    * **개인 의견**: 이 접근 방식은 스케줄링 결정을 좀 더 원칙적이고 수학적인 기반 위에서 내릴 수 있게 합니다. 예를 들어, TPOT가 SLO를 초과할 위험이 크면 TPOT를 최소화하는 것을 목표로 삼되, TTFT는 SLO를 넘지 않도록 제약하는 방식은 매우 합리적입니다.

4.  **계층적 우선순위 및 제약된 작업량 관리**: 전략 생성 시, 단계 수준(프리필 vs. 디코딩) 우선순위를 먼저 정하고, 같은 단계 내 요청들 간 우선순위를 정합니다[cite: 208, 209, 210, 212]. 이후, 최적화 목표와 제약 조건에 따라 실행할 요청의 수($n_i$)나 토큰의 수($k_i$)를 제한합니다[cite: 226, 227]. 예를 들어, TTFT 최적화 시에는 TPOT SLO를 넘지 않도록 프리필 토큰 수($k_i$)를 제한하고(Eq. 1)[cite: 229, 232], TPOT 최적화 시에는 TTFT SLO를 넘지 않도록 프리필 요청 수($n_i$)를 제한합니다(Eq. 2)[cite: 235, 236].
    * **개인 의견**: 이 두 가지 메커니즘(우선순위, 작업량 제한)은 SOLA가 트레이드오프를 실제로 구현하는 구체적인 수단입니다. 우선순위는 어떤 요청을 먼저 고려할지 정하고, 작업량 제한은 그 과정에서 다른 SLO 지표가 과도하게 희생되지 않도록 제어하는 역할을 합니다.

5.  **정확한 비용 모델 및 피크 메모리 예측**: SOLA는 지연 시간 예측을 위한 비용 모델($C^p, C^d$)을 사용하며, 이를 온라인으로 튜닝하여 정확도를 유지합니다[cite: 154, 181, 244, 254, 255]. 또한, 선제적인 피크 메모리 예측을 통해 불필요한 선점(preemption) 및 이로 인한 지연 시간 증가를 방지합니다[cite: 129, 172, 240, 242, 243].
    * **개인 의견**: 정확한 예측은 미래를 대비한 의사결정의 질을 높입니다. 비용 모델은 "만약 이 요청들을 실행한다면 지연 시간이 얼마나 될까?"라는 질문에 답을 주고, 피크 메모리 예측은 "실행 중에 메모리가 부족해지지는 않을까?"라는 위험을 사전에 관리하게 해줍니다. 이 두 가지가 없다면 SOLA의 동적 결정은 불안정해질 수 있습니다.

결론적으로, SOLA는 LLM 서비스 환경의 복잡성과 동적인 특성을 깊이 이해하고, 이에 대응하기 위한 정교하고 적응적인 메커니즘들을 통합적으로 설계했기 때문에 다른 방법론들보다 우수한 결과를 달성할 수 있었다고 판단됩니다.

# Q : 이 논문의 입력데이터와 추론 과정에 대해서 예시를 들어 아주 자세하게 설명해 주겠니? 추가적으로 모델아키텍처의 구성 및 모델을 구성하는 연산과 메모리 요구량 컴퓨팅 요구량 대해서도 설명해줘



제공된 문서는 SOLA라는 LLM 서비스 스케줄링 프레임워크에 대한 논문으로, LLM 모델 자체의 아키텍처, 연산량, 메모리 요구량, 컴퓨팅 요구량에 대한 상세 정보를 포함하고 있지 않습니다. 논문은 주로 LLM 서비스를 효율적으로 운영하기 위한 스케줄링 전략에 초점을 맞추고 있으며, 특정 LLM 모델의 내부 구조보다는 해당 모델을 어떻게 효과적으로 사용자에게 제공할지에 대한 방법론을 다룹니다.

따라서 SOLA 프레임워크의 입력 데이터와 추론 과정에 대한 예시는 설명드릴 수 있지만, LLM 모델 아키텍처, 구성 연산, 메모리/컴퓨팅 요구량에 대한 자세한 설명은 이 논문의 범위를 벗어납니다.

---
## SOLA 프레임워크의 입력 데이터 및 추론 과정 예시

SOLA 스케줄링 프레임워크는 LLM 서비스 요청들을 받아, 각 요청의 상태와 시스템 전체의 상태를 고려하여 어떤 요청을 언제, 얼마나 처리할지 결정합니다. 이 결정 과정을 '추론'이라고 표현할 수 있습니다.

**입력 데이터 (매 스케줄링 반복마다 고려되는 정보)**:

* **시스템 설정 (Setting) [cite: 124, 150]**:
    * 최대 KV 캐시 메모리 ($M$) [cite: 124]
    * TTFT의 SLO ($T^{TTFT}$) [cite: 124]
    * TPOT의 SLO ($T^{TPOT}$) [cite: 124]
* **시스템 상태 (System State) [cite: 124, 152]**:
    * 현재 KV 캐시 메모리 사용 비율 ($m_i^{ratio}$) [cite: 124]
    * 대기 중인 요청 큐 ($Q_i^{wait}$) [cite: 124]
    * 현재 실행 중인(또는 이번 반복에 실행될) 요청 큐 ($Q_i^{run}$) [cite: 124]
    * 시스템 전체 요청들의 실시간 TTFT 대비 SLO 비율 ($p^{TTFT}$) [cite: 124]
    * 시스템 전체 요청들의 실시간 TPOT 대비 SLO 비율 ($p^{TPOT}$) [cite: 124]
    * 이전 요청들의 출력 길이 분포 ($D_i$) [cite: 124]
* **개별 요청 상태 (Request State - 요청 $r$에 대해) [cite: 124, 152]**:
    * 요청 $r$의 현재까지의 실시간 TTFT ($t_{i,r}^{TTFT}$) [cite: 124]
    * 요청 $r$의 현재까지의 실시간 TPOT ($t_{i,r}^{TPOT}$) [cite: 124]
    * 요청 $r$에 대해 현재까지 생성된 출력 토큰 길이 ($l_{i,r}^{out}$) [cite: 124]
    * 요청 $r$에 대해 앞으로 생성될 것으로 예측되는 남은 토큰 길이 ($l_{i,r}^{left}$) [cite: 124]
    * 요청 $r$이 프리필 단계일 경우, 이번 반복에 처리될 토큰 수 ($k_{i,r}^{new}$) [cite: 124, 165]
    * 요청 $r$의 입력 토큰 길이 ($l_r^{in}$) [cite: 124]
* **비용 모델 (Cost Model) [cite: 124, 154]**:
    * 프리필 요청 지연 시간 예측 모델 ($C_i^p$) [cite: 124]
    * 디코딩 요청 지연 시간 예측 모델 ($C_i^d$) [cite: 124]

**SOLA의 스케줄링 추론 과정 (예시 시나리오)**:

가정:
* SLOs: $T^{TTFT} = 500ms$, $T^{TPOT} = 100ms$.
* 시스템 상태: $p^{TTFT} = 0.7$ (TTFT는 SLO 대비 70% 수준으로 양호), $p^{TPOT} = 0.98$ (TPOT는 SLO 대비 98% 수준으로 거의 한계). KV 캐시 여유 있음.
* 실행 큐 ($Q_{i-1}^{run}$):
    * `ReqA` (디코딩 중, $t^{TPOT}_{ReqA} = 95ms$, $l^{left}_{ReqA} = 20$ 토큰)
* 대기 큐 ($Q_i^{wait}$):
    * `ReqB` (새 요청, $l^{in}_{ReqB} = 300$ 토큰)
    * `ReqC` (새 요청, $l^{in}_{ReqC} = 50$ 토큰)

**i번째 반복에서의 SOLA 추론 단계**:

1.  **상태 모니터링 업데이트 (Sec 4.3)**:
    * (i-1)번째 반복에서 `ReqA`의 디코딩 실행 결과를 반영하여 $t^{TPOT}_{ReqA}$ 등의 상태를 갱신합니다.
    * 새 요청 `ReqB`, `ReqC`가 $Q_i^{wait}$에 있음을 확인합니다.
    * 시스템 전체의 $p^{TTFT}$와 $p^{TPOT}$를 최신 상태로 계산합니다. (위 예시값 사용)

2.  **전략 생성 (Strategy Generator - Sec 4.4)**:
    * **제약 조건 최적화 문제 결정 (Sec 4.4.1, Table 2)**:
        * 현재 $p^{TPOT} (0.98) > p^{TTFT} (0.7)$ 이고 $p^{TPOT}$가 1에 매우 가까우므로, TPOT를 최적화하고 TTFT를 제약 조건으로 설정합니다[cite: 184, 203].
            * **목표**: $\min \max_r (t_{i,r}^{TPOT})$ (가장 오래 걸리는 요청의 TPOT 최소화)
            * **제약**: $\max_r (t_{i,r}^{TTFT}) \le T^{TTFT}$ (모든 요청의 TTFT가 SLO 내에 있도록)
    * **계층적 우선순위 $\mathcal{F}_i$ 결정 (Sec 4.4.2, Fig 6)**:
        * **단계 수준 우선순위**: TPOT 최적화가 목표이므로 디코딩 단계 요청(`ReqA`)을 프리필 단계 요청(`ReqB`, `ReqC`)보다 우선합니다[cite: 210].
        * **요청 수준 우선순위**:
            * `ReqA`는 현재 디코딩 중이므로 가장 높은 우선순위를 갖습니다.
            * 프리필 요청 `ReqB`와 `ReqC`에 대해서는 예상 TTFT ($t_{i,r}^{TTFT} + C_i^p(r)$)를 계산합니다[cite: 214]. 입력 길이가 짧은 `ReqC`가 $C_i^p(ReqC)$가 더 작아 예상 TTFT가 짧다면 `ReqC`가 `ReqB`보다 높은 우선순위를 가질 수 있습니다. (또는 다른 정렬 기준 적용 가능)
            * 결정된 정렬 함수 $\mathcal{F}_i$: 예) `ReqA` $\rightarrow$ `ReqC` $\rightarrow$ `ReqB`.
    * **제약된 작업량 ($n_i, k_i$) 결정 (Sec 4.4.3, Fig 6)**:
        * TPOT 최적화 목표이므로, 디코딩 요청(`ReqA`)은 우선적으로 실행 큐에 포함됩니다.
        * 프리필 요청(`ReqC`, `ReqB` 순으로 고려)을 추가할 때, 해당 요청의 예상 TTFT($t_{i,r}^{TTFT} + C_i^p(Q_i^{run}) + C_i^d(Q_i^{run}) + C_i^p(r)$)가 $T^{TTFT}$를 넘지 않도록 하는 최대 프리필 요청 수 $n_i$를 설정합니다 (Eq. 2)[cite: 235].
            * 예를 들어, `ReqC`를 추가해도 모든 요청의 TTFT가 SLO 내에 있지만, `ReqB`까지 추가하면 `ReqB`의 TTFT가 $500ms$를 초과한다고 예측되면, 이번 반복에는 `ReqC`만 프리필 대상으로 추가하고 `ReqB`는 대기시킵니다. 이때 $n_i$는 (실행 중이던 `ReqA` 수) + (`ReqC` 수) = 2가 될 수 있습니다. $k_i$는 무관하게 설정됩니다.

3.  **스케줄링 실행 (Algorithm 1 - Sec 4.1.2)**:
    * $Q_i^{run}$은 `ReqA`로 시작합니다.
    * $\mathcal{F}_i$에 따라 정렬된 $Q_i^{wait}$ (`ReqC` $\rightarrow$ `ReqB`)에서 `ReqC`를 먼저 고려합니다.
    * **`ReqC` 처리**:
        * `ReqC`를 $Q_i^{run}$에 추가했을 때의 피크 메모리 사용량을 예측합니다 (Sec 5.1)[cite: 243].
        * 메모리가 충분하고, $Q_i^{run}$의 요청 수가 위에서 결정된 $n_i$ (예: 2)를 넘지 않으면, `ReqC`를 $Q_i^{run}$으로 옮기고 $Q_i^{wait}$에서 제거합니다.
    * **`ReqB` 처리**:
        * 만약 `ReqC`가 추가되어 $Q_i^{run}$의 크기가 이미 $n_i$에 도달했다면, `ReqB`는 이번 반복에 추가되지 않습니다.
    * 최종 $Q_i^{run}$: [`ReqA`, `ReqC`]

4.  **LLM 실행**:
    * 결정된 $Q_i^{run}$에 포함된 요청들 (`ReqA`의 디코딩 1토큰, `ReqC`의 프리필)이 GPU에서 배치로 실행됩니다.

5.  **다음 반복 준비**:
    * 실행 결과를 바탕으로 `ReqA`의 $t^{TPOT}$, `ReqC`의 $t^{TTFT}$ 등 상태가 업데이트되고, 비용 모델도 실제 측정값으로 튜닝됩니다 (Sec 5.2)[cite: 255]. 이는 다음 스케줄링 결정에 사용됩니다.

이러한 과정을 매 반복마다 거치면서 SOLA는 시스템의 현재 상태와 SLO 목표에 맞춰 동적으로 스케줄링 결정을 내립니다.

---
## LLM 모델 아키텍처, 연산, 메모리, 컴퓨팅 요구량

앞서 언급했듯이, SOLA 논문은 특정 LLM 모델의 상세 아키텍처나 그에 따른 구체적인 요구량을 기술하고 있지 않습니다. 다만, LLM 일반에 대한 배경 지식과 논문에서 간접적으로 언급되는 내용들을 통해 일반적인 사항을 추론해 볼 수 있습니다.

* **모델 아키텍처**:
    * 논문은 주류 LLM들이 트랜스포머(Transformer) 아키텍처를 사용한다고 언급합니다[cite: 71].
    * 트랜스포머의 핵심은 셀프 어텐션(self-attention) 메커니즘이며, 이전 토큰들의 키(key)와 값(value) 벡터들(KV 캐시)이 다음 토큰 생성에 재사용됩니다[cite: 72, 73].
    * LLM 처리 과정은 크게 **프리필(Prefill) 단계**와 **디코딩(Decode) 단계**로 나뉩니다[cite: 11, 74].
        * 프리필: 입력 프롬프트를 처리하여 첫 번째 출력 토큰을 생성하고 KV 캐시를 채웁니다[cite: 12, 76, 77].
        * 디코딩: 이전 단계에서 생성된 토큰과 KV 캐시를 사용하여 다음 토큰을 순차적으로 생성합니다[cite: 12, 78].
* **주요 연산**:
    * 트랜스포머 모델의 주요 연산은 행렬 곱셈(GEMM)입니다. 이는 주로 선형 계층(linear projection)과 어텐션 스코어 계산에 사용됩니다[cite: 72].
    * 프리필 단계는 입력 시퀀스 전체에 대한 병렬 처리가 가능하여 계산 집약적입니다.
    * 디코딩 단계는 한 번에 하나의 토큰을 생성하며, KV 캐시를 읽어오는 과정 때문에 메모리 대역폭에 민감할 수 있습니다 (메모리 집약적).
* **메모리 요구량**:
    * 가장 큰 부분은 **모델 파라미터**입니다. 수십억~수백억 개의 파라미터를 가진 모델(예: Llama3-70B [cite: 15])은 그 자체로 수십~수백 GB의 메모리를 차지합니다.
    * **KV 캐시** 또한 상당한 메모리를 요구합니다. 이는 (배치 크기) x (시퀀스 길이) x (레이어 수) x (히든 차원)에 비례하며, 특히 긴 시퀀스를 처리할 때 커집니다. SOLA는 vLLM의 PagedAttention을 언급하며 KV 캐시 관리를 중요하게 다룹니다[cite: 87].
    * 그 외에 활성화 값(activations), 옵티마이저 상태(학습 시) 등의 메모리가 필요합니다.
* **컴퓨팅 요구량 (FLOPs)**:
    * LLM의 컴퓨팅 요구량은 모델 크기(파라미터 수)와 처리하는 토큰 수에 따라 크게 달라집니다.
    * 프리필 단계는 입력 시퀀스 길이에 거의 제곱으로 비례하는 연산량을 가질 수 있고, 디코딩 단계는 생성하는 각 토큰마다 입력 시퀀스 길이에 비례하는 연산량을 가집니다.
    * SOLA의 비용 모델 $C^p$ (Eq. 3)는 입력 길이와 KV 캐시 길이의 곱, 입력 길이의 제곱 등에 비례하는 항을 포함하며, 이는 프리필 단계의 FLOPs 특성을 반영합니다[cite: 247, 249]. $C^d$ (Eq. 4)는 배치 크기, KV 캐시 길이에 비례하는 항을 포함하여 디코딩 단계의 메모리 접근 특성을 반영합니다[cite: 247, 249].

특정 LLM (예: 논문에 언급된 Llama3-8B/70B, Qwen1.5-14B/72B 등 [cite: 253])의 정확한 파라미터 수, 레이어 수, 히든 차원 등을 알아야 더 구체적인 메모리 및 컴퓨팅 요구량을 계산할 수 있지만, 해당 정보는 이 논문에 포함되어 있지 않습니다.

# Q : 이 논문의 한계를 극복하기 위한 방법으로 어떤 연구흐름이 있는지 정리해서 자세하게 설명해 줘



SOLA 논문의 한계를 극복하기 위한 연구는 주로 **비용 모델의 정확성 향상**, **출력 길이 예측 정교화**, 그리고 **더욱 동적이고 세밀한 자원 관리 및 스케줄링 최적화** 방향으로 진행될 수 있습니다. 🧐

---
## SOLA 한계 극복을 위한 연구 흐름

SOLA는 LLM 서비스의 SLO 달성률을 크게 향상시키는 인상적인 결과를 보여주었지만, 몇 가지 한계점도 가지고 있습니다. 이러한 한계들을 극복하기 위한 잠재적인 연구 흐름은 다음과 같습니다:

### 1. 비용 모델(Cost Model)의 정확성 및 일반화 능력 향상 🎯

* **한계**: SOLA는 다항식 기반의 비용 모델을 사용하며 온라인 튜닝을 하지만, 복잡하고 동적인 실제 환경, 특히 새로운 유형의 쿼리나 모델 아키텍처에 대해서는 예측 정확도가 떨어질 수 있습니다[cite: 300, 302].
* **연구 방향**:
    * **머신러닝 기반 비용 모델**: 더 복잡한 패턴을 학습할 수 있는 신경망이나 트리 기반 모델(예: Gradient Boosting, Random Forest)을 사용하여 비용 예측 모델을 구축합니다. 이러한 모델은 더 많은 특징(feature)을 고려하고 비선형 관계를 더 잘 포착할 수 있습니다.
    * **컨텍스트 인지(Context-Aware) 모델링**: 현재 시스템 부하, 다른 요청들의 특성, 하드웨어 상태 등 더 넓은 컨텍스트 정보를 비용 모델에 통합하여 예측 정확도를 높입니다.
    * **온라인 학습 강화**: 단순한 스케일링 팩터 조정[cite: 254, 255, 256]을 넘어, 온라인으로 모델 파라미터 자체를 지속적으로 업데이트하는 정교한 온라인 학습 기법(예: Online Gradient Descent, Bayesian Online Learning)을 적용합니다.
    * **모델 아키텍처별 특화 모델**: LLM 모델 아키텍처(예: Transformer 변형, Mixture-of-Experts)의 특성을 반영한 맞춤형 비용 모델을 개발하여 일반화 성능을 높입니다.

---
### 2. 출력 길이 예측(Output Length Prediction) 정교화 🔮

* **한계**: SOLA는 과거 요청들의 출력 길이 분포에 기반하여 남은 출력 길이를 예측합니다[cite: 192, 193]. 이는 새로운 유형의 프롬프트나 사용자에 대해서는 부정확할 수 있으며, 이는 TPOT 관련 의사결정에 영향을 미칩니다.
* **연구 방향**:
    * **프롬프트 기반 예측**: 입력 프롬프트의 내용(의도, 주제, 길이 등)을 분석하여 출력 길이를 예측하는 모델을 개발합니다. 자연어 처리(NLP) 기술을 활용하여 프롬프트 임베딩을 생성하고 이를 예측 모델의 입력으로 사용할 수 있습니다.
    * **사용자별/세션별 예측**: 사용자 또는 현재 대화 세션의 과거 기록을 바탕으로 출력 길이를 예측하여 개인화된 예측을 제공합니다.
    * **점진적 예측 및 신뢰도 기반 조정**: 토큰이 생성됨에 따라 남은 출력 길이에 대한 예측을 점진적으로 업데이트하고, 예측의 신뢰도를 함께 추정하여 스케줄링 결정에 활용합니다. 신뢰도가 낮을 경우 더 보수적인 스케줄링을 수행할 수 있습니다.

---
### 3. 스케줄링 전략 및 자원 관리 고도화 ⚙️

* **한계**: SOLA는 상태 인식 및 제약 조건 최적화를 사용하지만, 매우 많은 요청 수나 극도로 동적인 환경에서는 현재의 계층적 우선순위 규칙이나 제약 조건 완화 방식이 최적이 아닐 수 있습니다. 또한, 주로 동일 GPU 내 프리필/디코딩에 초점을 맞추고 있습니다.
* **연구 방향**:
    * **강화학습(Reinforcement Learning) 기반 스케줄링**: 스케줄링 문제를 순차적 의사결정 문제로 정의하고, 강화학습 에이전트가 시행착오를 통해 최적의 스케줄링 정책을 학습하도록 합니다. 이는 복잡한 상태 공간과 동적인 환경 변화에 더 잘 적응할 수 있습니다.
    * **다중 목표 최적화(Multi-Objective Optimization)**: TTFT, TPOT, 처리량, 공정성 등 여러 목표를 동시에 고려하는 다중 목표 최적화 기법을 도입하여 더 균형 잡힌 스케줄링 결정을 내립니다.
    * **선점(Preemption) 및 마이그레이션(Migration) 전략 고도화**: SOLA는 선점을 피하는 방향으로 설계되었지만[cite: 169, 240, 242], 불가피한 경우를 대비해 선점 비용을 최소화하고, 요청을 다른 자원(예: 다른 GPU, CPU 오프로딩)으로 동적으로 마이그레이션하는 더 정교한 전략을 연구합니다.
    * **이종 시스템(Disaggregated System) 및 분산 환경 지원**: 프리필과 디코딩이 서로 다른 GPU나 클러스터에서 수행되는 이종 시스템 환경에 SOLA의 원칙을 확장 적용하는 연구가 필요합니다. 이는 자원 할당 및 통신 오버헤드 관리 등 새로운 도전 과제를 야기합니다. 논문에서도 DistServe, TetriInfer와 같은 시스템을 언급하며 이종 시스템 디자인을 소개하고 있습니다[cite: 103].
    * **에너지 효율성 고려**: 성능뿐만 아니라 에너지 소비까지 고려하는 스케줄링 전략을 개발하여 지속 가능한 LLM 서비스를 지향합니다.

---
### 4. 하이퍼파라미터 자동 튜닝 및 민감도 분석 🔧

* **한계**: SOLA에는 비용 모델 튜닝 시 신뢰도 계수 $\alpha$[cite: 256], 제약 조건 완화 시 사용되는 $a\%$[cite: 206] 등 여러 하이퍼파라미터가 존재하며, 이들의 최적값은 워크로드나 시스템에 따라 달라질 수 있습니다.
* **연구 방향**:
    * **자동화된 하이퍼파라미터 최적화(Automated Hyperparameter Optimization, HPO)**: 베이지안 최적화(Bayesian Optimization), 유전 알고리즘(Genetic Algorithms) 등의 HPO 기법을 사용하여 특정 환경에 맞는 최적의 하이퍼파라미터 조합을 자동으로 탐색합니다.
    * **민감도 분석 및 적응형 파라미터**: 주요 하이퍼파라미터 변화에 따른 시스템 성능 민감도를 분석하고, 실시간 시스템 상태에 따라 하이퍼파라미터를 동적으로 조정하는 적응형 메커니즘을 연구합니다.

이러한 연구 흐름들은 SOLA가 제시한 세분화된 상태 인식 스케줄링의 개념을 더욱 발전시켜, 미래의 LLM 서비스가 더욱 효율적이고 안정적으로 운영될 수 있도록 기여할 것입니다.


# Q : SOLA의 상태 인식 스케줄링에서 모니터링하는 '개별 요청 상태'와 '시스템 상태'의 구체적인 예시는 무엇이며, 이 두 가지 유형의 상태 정보가 어떻게 상호작용하여 스케줄링 결정에 영향을 미칩니까? 

 

SOLA의 상태 인식 스케줄링은 **개별 요청 상태**와 **시스템 상태**를 모니터링하여 스케줄링 결정을 내립니다[cite: 6, 59, 66].

## 모니터링 상태 예시

**개별 요청 상태 (Request State - 특정 요청 r에 대해) [cite: 124]**:
* **실시간 TTFT ($t_{i,r}^{TTFT}$)**: 해당 요청이 도착한 후 첫 번째 토큰이 생성될 때까지 걸린 시간입니다.
* **실시간 TPOT ($t_{i,r}^{TPOT}$)**: 해당 요청의 디코딩 단계에서 토큰당 평균 생성 시간입니다.
* **생성된 출력 길이 ($l_{i,r}^{out}$)**: 현재까지 생성된 출력 토큰의 수입니다.
* **예측된 남은 길이 ($l_{i,r}^{left}$)**: 앞으로 생성될 것으로 예상되는 토큰의 수입니다.
* **새로 추가될 토큰 수 ($k_{i,r}^{new}$)**: 이번 반복에서 처리될 토큰 수 (주로 프리필 단계의 청크 크기)입니다.
* **입력 길이 ($l_r^{in}$)**: 해당 요청의 입력 프롬프트 길이입니다.

**시스템 상태 (System State) [cite: 124]**:
* **KV 캐시 메모리 사용 비율 ($m_i^{ratio}$)**: 현재 사용 중인 KV 캐시 메모리의 비율입니다.
* **대기 요청 큐 ($Q_i^{wait}$)**: 처리를 기다리고 있는 요청들의 목록입니다.
* **실행 요청 큐 ($Q_i^{run}$)**: 현재 처리 중이거나 이번 반복에 처리될 요청들의 목록입니다.
* **실시간 TTFT 대 SLO 비율 ($p^{TTFT}$)**: 시스템 내 모든 요청들의 현재 TTFT 값들을 SLO 값과 비교하여 산출한 비율 (예: 가장 높은 TTFT / TTFT SLO)입니다[cite: 188].
* **실시간 TPOT 대 SLO 비율 ($p^{TPOT}$)**: 시스템 내 모든 요청들의 현재 TPOT 값들을 SLO 값과 비교하여 산출한 비율 (예: 가장 높은 TPOT / TPOT SLO)입니다[cite: 188].
* **출력 길이 분포 ($D_i$)**: 과거에 처리된 요청들의 출력 길이 통계 정보입니다.

---
## 상태 정보 상호작용 및 스케줄링 결정 영향

개별 요청 상태와 시스템 상태 정보는 SOLA의 스케줄링 결정 과정에서 긴밀하게 상호작용합니다.

1.  **최적화 목표 설정**:
    * 시스템 상태인 $p^{TTFT}$와 $p^{TPOT}$를 비교하여 현재 어떤 SLO 지표가 더 위태로운지 판단합니다[cite: 202, 203].
    * 예를 들어, $p^{TPOT}$가 $p^{TTFT}$보다 높고 SLO 기준치(1)에 가깝거나 초과한다면, 시스템은 TPOT를 우선적으로 최적화하는 목표를 설정합니다[cite: 184]. 이 경우, 개별 요청들의 $t_{i,r}^{TTFT}$는 SLO를 만족하는 제약 조건으로 작용합니다.

2.  **요청 우선순위 결정**:
    * 설정된 최적화 목표에 따라 요청 실행 순서($\mathcal{F}_i$)가 결정됩니다[cite: 209, 210].
    * 만약 TPOT 최적화가 목표라면, 디코딩 단계에 있는 요청들이 프리필 단계에 있는 요청들보다 높은 우선순위를 갖습니다[cite: 210].
    * 같은 단계 내에서는 개별 요청 상태(예: 예상 완료 시점의 TPOT, 예상 TTFT)를 기반으로 세부적인 우선순위가 정해집니다[cite: 214, 216]. 예를 들어, 프리필 요청들은 예상 TTFT가 긴 순서대로 정렬될 수 있습니다[cite: 214].

3.  **작업량 제어**:
    * 개별 요청 상태(예: $t_{i,r}^{TPOT}$, $l_{i,r}^{out}$, $l_{i,r}^{left}$)와 시스템의 비용 모델($\mathcal{C}_i^p, \mathcal{C}_i^d$)을 사용하여, 특정 요청을 추가했을 때 다른 요청들의 SLO(특히 제약 조건으로 설정된 SLO)가 위반되지 않도록 실행할 요청 수($n_i$)나 토큰 수($k_i$)를 제한합니다[cite: 228, 229, 235].
    * 예를 들어, TTFT를 최적화하면서 TPOT를 제약 조건으로 둘 때, 새로운 프리필 요청을 처리함으로써 기존 디코딩 중인 요청들의 TPOT가 SLO를 초과하지 않도록 프리필 토큰 수($k_i$)를 조절합니다 (Eq. 1)[cite: 229, 230].

4.  **메모리 관리**:
    * 시스템 상태인 $m_i^{ratio}$ (KV 캐시 사용률)와 개별 요청의 예상 메모리 사용량을 고려하여, 새로운 요청을 실행 큐에 추가할지 여부를 결정합니다[cite: 172]. 이는 Algorithm 1의 메모리 확인 단계에서 이루어집니다[cite: 156].

이처럼 SOLA는 개별 요청의 미시적인 상태와 시스템 전체의 거시적인 상태를 종합적으로 판단하여, 매 반복마다 SLO 달성률을 극대화하는 방향으로 스케줄링 결정을 동적으로 내립니다.

# Q : SOLA가 TTFT와 TPOT 간의 트레이드오프, 그리고 서로 다른 요청들 간의 트레이드오프를 관리하기 위해 사용하는 '제약 조건 최적화 문제'는 구체적으로 어떻게 정의되며, 시스템 상태 변화에 따라 이 최적화 목표와 제약 조건이 어떻게 동적으로 변경됩니까? 

 

SOLA는 TTFT(첫 토큰까지의 시간)와 TPOT(출력 토큰당 시간) 간의 트레이드오프 및 서로 다른 요청들 간의 트레이드오프를 관리하기 위해, 시스템의 실시간 상태에 따라 동적으로 변경되는 '제약 조건 최적화 문제'를 사용합니다[cite: 60, 197].

**제약 조건 최적화 문제 정의 및 동적 변경**

SOLA의 핵심 아이디어는 두 가지 SLO 지표(TTFT, TPOT) 중 현재 더 만족스럽지 못한 지표를 최적화 대상으로 삼고, 다른 지표는 SLO를 만족하는 제약 조건으로 설정하는 것입니다[cite: 201, 203]. 이러한 전환은 시스템의 실시간 TTFT/TPOT 상태($p_i^{TTFT}$, $p_i^{TPOT}$)를 기반으로 이루어집니다[cite: 202]. $p_i^{TTFT}$와 $p_i^{TPOT}$는 각각 시스템 내 요청들의 최대 실시간 TTFT와 TPOT를 해당 SLO 값으로 나눈 비율을 의미합니다[cite: 188].

Table 2는 이 변환 과정을 요약하여 보여줍니다[cite: 183].

1.  **TPOT 최적화, TTFT 제약 (Optimize TPOT subject to TTFT)**:
    * **조건**:
        * $1 > p_i^{TPOT} > p_i^{TTFT}$ (두 지표 모두 SLO 만족, 그러나 TPOT가 TTFT보다 SLO에 더 근접)
        * $p_i^{TPOT} > 1 > p_i^{TTFT}$ (TPOT는 SLO 위반, TTFT는 SLO 만족)
    * **최적화 문제**:
        * 목표: $\min \max_r (t_{i,r}^{TPOT})$ (가장 오래 걸리는 요청의 TPOT 최소화)
        * 제약: $\max_r (t_{i,r}^{TTFT}) \le T^{TTFT}$ (모든 요청의 TTFT가 SLO 이내여야 함) [cite: 184]

2.  **TTFT 최적화, TPOT 제약 (Optimize TTFT subject to TPOT)**:
    * **조건**:
        * $1 > p_i^{TTFT} > p_i^{TPOT}$ (두 지표 모두 SLO 만족, 그러나 TTFT가 TPOT보다 SLO에 더 근접)
        * $p_i^{TTFT} > 1 > p_i^{TPOT}$ (TTFT는 SLO 위반, TPOT는 SLO 만족)
    * **최적화 문제**:
        * 목표: $\min \max_r (t_{i,r}^{TTFT})$ (가장 오래 걸리는 요청의 TTFT 최소화)
        * 제약: $\max_r (t_{i,r}^{TPOT}) \le T^{TPOT}$ (모든 요청의 TPOT가 SLO 이내여야 함) [cite: 184]

3.  **두 지표 모두 SLO 위반 시 제약 조건 완화**:
    * **조건**: $p_i^{TTFT} > 1$ 이고 $p_i^{TPOT} > 1$ (두 지표 모두 SLO 위반)
    * **처리**: 이 경우, 제약 조건을 만족하는 요청이 없으므로 최적화가 어렵습니다. SOLA는 제약 조건을 점진적으로 완화합니다[cite: 204]. 예를 들어, 상위 a% (a<100)의 요청만이 제약 조건을 만족하도록 목표를 낮춥니다[cite: 206]. 이렇게 제약 조건을 완화하여 한쪽 지표라도 SLO를 (완화된 기준으로) 만족하게 되면, 다시 위의 두 가지 최적화 문제 중 하나로 전환하여 해결을 시도합니다[cite: 205]. Figure 5는 이러한 상태 전환을 시각적으로 보여줍니다[cite: 185].

이처럼 SOLA는 시스템의 전반적인 SLO 달성 현황을 나타내는 $p_i^{TTFT}$와 $p_i^{TPOT}$ 값을 지속적으로 모니터링하고, 이 값들의 변화에 따라 최적화할 대상과 제약 조건을 동적으로 바꿔가며 스케줄링 전략을 수립합니다. 이를 통해 한쪽 지표에 치우치지 않고 두 SLO 사이의 균형을 맞추며, 동시에 여러 요청들 간의 자원 경쟁을 조율하여 시스템 전체의 SLO 달성률을 극대화합니다.

# Q : 기존의 고정된 스케줄링 방식(예: 프리필 우선)과 비교했을 때, SOLA의 '세분화된 반복 단위 스케줄링'(요청 실행 순서 및 작업량 제어)이 SLO 달성률을 향상시키는 핵심적인 이유는 무엇이며, 실제 서비스 환경에서 발생할 수 있는 다양한 요청 패턴과 시스템 부하 변화에 어떻게 더 효과적으로 대응할 수 있습니까? 

 

SOLA의 '세분화된 반복 단위 스케줄링'은 고정된 방식과 달리, 매 반복마다 요청 실행 순서와 작업량을 동적으로 조절하여 SLO 달성률을 높입니다. 이는 변화하는 요청 패턴과 시스템 부하에 실시간으로 적응할 수 있게 해줍니다.

---
## SOLA의 세분화된 스케줄링의 SLO 달성률 향상 이유 및 적응성

### SLO 달성률 향상의 핵심 이유

1.  **동적 트레이드오프 관리**: 고정된 스케줄링(예: 프리필 우선)은 TTFT에 유리하지만 TPOT를 악화시키거나, 그 반대의 경향을 보입니다[cite: 39, 41]. SOLA는 매 반복마다 시스템 및 개별 요청의 상태(예: $p^{TTFT}$, $p^{TPOT}$, $t_{i,r}^{TTFT}$, $t_{i,r}^{TPOT}$)를 모니터링하여, 어떤 SLO 지표가 더 시급한지에 따라 최적화 목표와 제약 조건을 동적으로 변경합니다 (Table 2). 이를 통해 TTFT와 TPOT 간의 균형을 맞추고, 한쪽 지표의 과도한 희생을 막아 전반적인 SLO 달성률을 높입니다. 예를 들어, 많은 요청이 TPOT SLO를 위반할 위기에 처하면, SOLA는 TPOT를 우선적으로 최적화하면서 TTFT는 SLO를 만족하는 선에서 제어합니다.
2.  **요청 간 자원 분배 최적화**: SOLA는 단순히 선착순(FCFS)이나 고정된 우선순위로 요청을 처리하지 않습니다. 대신, 각 요청의 SLO 달성 가능성과 시스템 전체에 미치는 영향을 고려하여 실행 순서($\mathcal{F}_i$)를 정합니다. 예를 들어, 어떤 요청이 SLO를 곧 위반할 것 같으면 해당 요청에 우선순위를 더 높게 부여하고, 다른 요청은 SLO를 만족하는 범위 내에서 잠시 대기시킬 수 있습니다. 이는 그림 2(b)의 "서로 다른 요청 간의 트레이드오프" 개념으로 설명됩니다.
3.  **정교한 작업량 제어**: SOLA는 매 반복마다 실행할 요청의 수($n_i$)나 토큰의 수($k_i$)를 제어합니다[cite: 164]. 이는 설정된 최적화 목표와 제약 조건(예: 특정 SLO 지표 위반 방지)을 만족시키기 위함입니다. 예를 들어, 프리필 요청을 처리할 때, 이로 인해 기존 디코딩 중인 요청들의 TPOT가 SLO를 넘지 않도록 프리필 작업량(토큰 수)을 세밀하게 조절합니다(Eq. 1). 이러한 정교한 제어는 고정된 배치 크기나 청크 크기를 사용하는 방식보다 더 유연합니다.

---
### 다양한 요청 패턴 및 시스템 부하 변화에 대한 효과적 대응

1.  **실시간 상태 기반 적응**: SOLA는 "상태 인식(state-aware)" 스케줄링을 수행합니다[cite: 6, 66]. 이는 시스템 부하가 낮을 때는 더 많은 요청을 공격적으로 처리하여 처리량을 높이고, 부하가 높아져 SLO 위반 위험이 커지면 요청 처리 속도를 조절하거나 우선순위를 변경하여 안정성을 확보하는 등 실시간 상황 변화에 유연하게 대응할 수 있게 합니다. 예를 들어, 갑자기 긴 입력 길이를 가진 요청들이 몰려오면, 비용 모델과 상태 모니터링을 통해 이를 인지하고 프리필 단계의 작업량을 조절하거나, 다른 짧은 요청들과의 실행 순서를 변경하여 시스템 과부하를 막고 SLO를 최대한 준수하려 합니다.
2.  **예측 기반 선제적 조치**: 비용 모델($\mathcal{C}_i^p, \mathcal{C}_i^d$)과 피크 메모리 예측 메커니즘을 통해 특정 스케줄링 결정이 미래에 미칠 영향을 예측합니다[cite: 154, 240]. 이를 통해 잠재적인 SLO 위반이나 메모리 부족 문제를 사전에 방지할 수 있습니다. 예를 들어, 특정 요청을 추가했을 때 메모리 부족이 예상되면 해당 요청의 실행을 보류하거나, 다른 요청의 KV 캐시를 선점하는 대신 작업량을 줄이는 방식으로 대응합니다.
3.  **지속적인 학습 및 튜닝**: SOLA의 비용 모델은 오프라인 프로파일링 데이터로 초기화된 후, 실제 서빙 중 발생하는 지연 시간 데이터를 통해 온라인으로 튜닝됩니다(Sec 5.2, Eq. 5). 이는 요청 패턴이나 시스템 특성이 시간에 따라 변하더라도 비용 모델의 예측 정확도를 유지하고, 결과적으로 스케줄링 결정의 질을 지속적으로 개선할 수 있게 합니다.

이처럼 SOLA의 세분화된 반복 단위 스케줄링은 고정된 규칙의 한계를 벗어나, 실제 서비스 환경의 복잡성과 가변성에 효과적으로 적응하며 SLO 달성률을 극대화합니다.