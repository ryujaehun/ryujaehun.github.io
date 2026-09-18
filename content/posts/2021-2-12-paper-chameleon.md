---
categories:
- compiler
- ML
- paper-review
date: "2021-02-12"
tags: null
title: 논문 정리 Chameleon Adaptive Code Optimization for Expedited Deep Neural Network
  Compilation(ICLR 2020)
---
![CHAMELEON 의 전체 설계와 컴파일 흐름 도식의 축소판](/assets/images/chameleon1.jpg)
## 제목
Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation

## 저자
Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, Hadi Esmaeilzadeh

## Motivation
The current approaches are oblivious to the patterns in the design space of schedules that are available for exploitation, and causes inefficient search or even converges to solutions that may even be suboptimal.
Current solutions that rely on greedy sampling lead to significant fractions of the candidate configurations being redundant over iterations(long compilation time)
## Contribution
- Devising an __Adaptive Exploration__ module that utilizes reinforcement learning to adapt to unseen design space of new networks to reduce search time yet achieve better performance.
- Proposing an __Adaptive Sampling__ algorithm that utilizes clustering to adaptively reduce the number of costly hardware measurements
## 개인적인 생각
RL을 이용하여 exploration을 잘하고 sampling을 효율적으로 해서 time을 줄이고자하는 목적이 참 깔끔하고 좋은 논문.
## Overall design
![Figure 3: CHAMELEON 의 전체 설계. 코드 템플릿과 설계 공간이 Adaptive Exploration Module 로 들어가 후보 설정을 만들고, Adaptive Sampling Module 이 표본을 골라 Code Generator 가 코드를 내며, 하드웨어에서 잰 실행 시간이 Cost Model 로 되돌아간다](/assets/images/chameleon2.png)

## Adaptive Exploration
- TVM leverages simulated annealing relies on the stochastic guarantees of its random walks(required numerous iterations of exploration) thus insufficient to enable disruptive innovations in neural networks
- Adaptive Exploration, based Reinforcement Learning ,is concerned with learning to maximize reward given an environment by making good exploration and exploitation tradeoffs
- These networks not only learn the dependencies among the different knobs of the design space (which are interrelated) that helps our module navigate through the design space but also lean the potential gains of the modifications to the configurations.
  
## Learning procedure
![Figure 4: Adaptive Exploration Module 의 동작. Policy Network 와 Config Updater 가 번갈아 돌며 한 에피소드 안에서 1차·2차·n차 설정을 만들어 간다](/assets/images/chameleon3.png)

## Adaptive Sampling : Reducing number of costly hardware measurements
![Figure 5: 후보 설정의 군집. VGG-16 4번째 층과 ResNet-18 11번째 층의 tile_x, tile_y 산점도에 군집이 원으로 표시되어 있다](/assets/images/chameleon4.png)
- we observe that the candidate configurations are clustered in subregions of the design space
- Our Adaptive Sampling iterates over a different number of clusters for their respective centroids and the L2 loss.(k-means)
- Selecting the number of centroids for clustering entails making the important tradeoff (using L2-performance degradation graph of knee of the curve
## Improving candidate configurations using sample synthesis
- Many of the automated approaches for black-box optimization are prone to invalid configurations
- These invalid configurations not only blow the chances for better exploration but also leads to an extra optimization time overhead to reset the physical hardware for the subsequent hardware measurement
- When our compiler runs into redundant samples, the proposed synthesis method analyzes the candidate samples to determine the most probable (most frequent = mode function) non-invalid choice for each knob to come up with a new configuration
## Improving candidate configurations using sample synthesis
- During training, some of the programs took a long time to compile, mainly when the agent was trying to vectorize more than plausible
- giving a penalty reward of −9 (equivalent to assuming it takes ten times the execution time of the baseline) so that the agent will learn not to overestimate the vectorization and avoid it
![Algorithm 1: Adaptive Sampling 의사코드. k 를 8에서 64까지 늘리며 K-means 를 돌리고 손실 곡선의 무릎에서 멈춘 뒤, 이미 방문한 설정은 최빈값으로 바꿔 돌려준다](/assets/images/chameleon5.png)
the most probable (most frequent = mode function)
## Evaluation
![Table 5: 평가에 쓴 층 목록. AlexNet, VGG-16, ResNet-18 의 합성곱 층 여덟 개가 L1~L8 과 태스크 인덱스로 정리되어 있다](/assets/images/chameleon6.png)
Task Index => layer order
![Figure 7: 구성 요소별 평가. (a) 수렴까지의 스텝 수가 AutoTVM 대비 최대 3.85배 줄고, (b) 하드웨어 측정 횟수가 최대 2.84배 줄며, (c) 반복 한 번의 시간이 어떻게 짧아지는지 보여 준다](/assets/images/chameleon7.png)
Overall, observation is that CHAMELEON’s Adaptive Exploration requires 2.88 less search steps compared to simulated annealing to find good solution.
![하드웨어 측정 횟수에 따른 TFLOPS 곡선. CHAMELEON 이 측정 800회를 392회로 줄이면서 출력 코드 성능은 4.71 에서 5.26 으로 올린다](/assets/images/chameleon8.png)
![(a) 층 단위 평가와 (b) 종단간 평가 막대그래프. 최적화 시간은 AutoTVM 대비 기하평균 4.82배 빠르고 출력 성능은 1.17배 좋다](/assets/images/chameleon9.png)
![Table 2·3: 심층 신경망 종단간 평가. AlexNet, VGG-16, ResNet-18 의 최적화 시간이 AutoTVM 대비 각각 4.31→1.20시간, 11.18→1.95시간, 9.13→2.13시간으로 줄고 출력 성능도 소폭 개선된다](/assets/images/chameleon10.png)


## references
https://openreview.net/forum?id=rygG4AVFvH
## Project Page
https://github.com/anony-sub/chameleon
