---
categories:
- NAS
- ML
- paper-review
date: "2021-02-12"
tags: null
title: 간단논문 정리 DARTS DIFFERENTIABLE ARCHITECTURE SEARCH (ICLR 2019)
---

## 제목
DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH


## 저자
Hanxiao Liu,Karen Simonyan,Yiming Yang

## Motivation
기존 NAS가 상당수의 시간 혹은 cost가 필요(2000 GPU days of reinforcement learning, 3150 GPU days of evolution)이러한 원인 중 하나가  discrete domain, which leads to a large number of architecture evaluations required 때문이라고 분석. 물론 이전에도 filter size와 같은 것들을 연속적으로 학습 했으나 해당 논문은 블록, 그래프 토플로지 까지 학습하는 것을 목표로 함

## Contribution
- 기존의 discrete and non-differentiable search space에서 RL혹은 GA를 이용하던 NAS를 아키텍쳐의 표현을 bilevel optimization을 사용하여 gradient descent로 학습 하게 함.
-  extensive experiments on image classification and language modeling (좋은 결과)
- 기존 방법에 비하여 학습 시간을 줄임
- CNN,RNN에서 transferable 함을 보임


## CONTINUOUS RELAXATION AND OPTIMIZATION

![DARTS 개요 그림. (a) 간선의 연산이 아직 정해지지 않은 상태, (b) 각 간선에 후보 연산을 섞어 탐색 공간을 연속화한 상태, (c) 이중 수준 최적화로 혼합 확률과 가중치를 함께 학습하는 상태, (d) 학습된 혼합 확률에서 최종 구조를 뽑아낸 상태](/assets/images/darts1.PNG)

위 그림과 아래 수식을 통해서 어떠한 방식을 통하여 연속적으로 연산을 정의 하는지 알 수 있다. 
node$i$,$j$연산의 종류를 선택하는 방법은 아래 식처럼 $\alpha$의 softmax를 이용하는 것이고 이는 위 그림을 통하여 직관적으로 알 수 있다. 

![간선 (i, j) 의 혼합 연산 정의 수식. 후보 연산 o 의 출력에 alpha 의 softmax 를 가중치로 곱해 더한다](/assets/images/darts2.PNG)


building block을 위에서 정의 했으니 weight를 학습하며 final architecture를 정해야 한다. 
이는 아래와 같이 bilevel optimization을 사용한다. 
![DARTS 의 이중 수준 최적화 수식. 검증 손실을 alpha 에 대해 최소화하되, 가중치 w 는 학습 손실을 최소화하는 값으로 제약한다](/assets/images/dart3.PNG)

![Algorithm 1: DARTS 의사코드. 간선마다 alpha 로 매개화된 혼합 연산을 만들고, 수렴할 때까지 검증 손실로 alpha 를, 학습 손실로 w 를 번갈아 갱신한 뒤 최종 구조를 뽑는다](/assets/images/darts4.PNG)

## APPROXIMATE ARCHITECTURE GRADIENT 
개인적으로는 design choice로 보이며 관련 후속논문이 있기때문에 크게 중요한 내용은 아닌것 같다. 
위 bilevel optimization form을 보면 MAML의 수식이 떠오른다. 이 논문에서도. First-order Approximation을 포함하여 연산량 감소를 위하여 수식을 변형 하였다.(trade-off가 있기 때문에 상황에 맞춰야) 
## DERIVING DISCRETE ARCHITECTURES
discrete architecture를 만들기 위해서 top-k strongest operations만 선택 (zero는 예외)


## Results

NASNET-A(2000 GPU days),AmoebaNet-A(3150 GPU days) ENAS (0.5 GPU day)에 비하여 동일 파라라미터를 맞췄을때 상당하게 시간 측면에서 효율적인 결과를 보여줌
![Figure 3: CIFAR-10 합성곱 셀과 Penn Treebank 순환 셀에 대한 DARTS 탐색 진행 그래프 네 개. GPU 시간에 따른 검증 오차와 퍼플렉서티가 NASNet-A, AmoebaNet-A, ENAS 기준선에 수렴해 가는 모습](/assets/images/darts5.PNG)

### cifar10

![Table 1: CIFAR-10 에서 최신 이미지 분류기와 비교한 표. DARTS(2차)가 테스트 오차 2.76%, 파라미터 3.3M, 탐색 비용 4 GPU-day 로 NASNet-A(2000 GPU-day), AmoebaNet-A(3150 GPU-day)와 견줄 성능을 훨씬 적은 비용에 낸다](/assets/images/darts6.PNG)
### PTB
![Table 2: Penn Treebank 에서 최신 언어 모델과 비교한 표. DARTS(2차)가 검증 퍼플렉서티 58.1, 테스트 55.7 로 ENAS 와 LSTM 계열을 앞서면서 탐색 비용은 1 GPU-day 다](/assets/images/darts7.PNG)
### ImageNet in the mobile setting
![Table 3: 모바일 환경 ImageNet 비교표. CIFAR-10 에서 탐색한 DARTS 셀이 top-1 오차 26.7%, 파라미터 4.7M, 연산 574M 으로 NASNet-A·AmoebaNet 계열과 비슷한 수준을 4 GPU-day 에 낸다](/assets/images/darts8.PNG)

## references
[paper](https://openreview.net/pdf?id=S1eYHoC5FX)
[official code](https://github.com/quark0/darts.git)


