---
categories:
- compiler
- ML
- paper-review
date: "2021-02-12"
tags:
- Machine Learning
- Programming Languages
title: 간단논문 정리 Fast and Effective Orchestration of Compiler Optimizations(Zhelong
  Pan,Rudolf Eigenmann;Purdue University ;CGO’06)
---
## 제목
Fast and Effective Orchestration of Compiler Optimizations

## 저자
Zhelong Pan,Rudolf Eigenmann

## Motivation
compile-time optimizations 은 전반적으로 프로그램 성능을 향상시키지만 일부 기법은 성능 하락을 야기한다. 
입력프로그램와 target architecture에 대한 불충분한 정보는 컴파일 시간에 정확도 향상을 향상 시키는 모델의 한계를 만든다. 
## Contribution
- 기존에 존재하는 Batch Elimination(BE)와 Iterative Elimination(IE)의 장점을 섞어 Combined Elimination(CE) 알고리즘을 제안한다. 
- OptimizationSpace Exploration (OSE),Statistical Selection (SS)에 비하여도 향상된 성능을 보여준다. 
- large set of realistic programs에서 평가하여 현실적인 결과를 제시한다. 
## Content
- Exhaustive Search =>O(2^n) 
- Batch Elimination => O(n) 
  - Relative Improvement Percentage(RIP)을 기준으로 RIP하락시 optimizations 제거 
- Iterative Elimination => O(n^2) 
  - RIP을 기준으로 부정적인 결과를 보이는 하나의 optimization을 제거하는 방법을 반복 
- Combined Elimination => O(n^2)  
  - RIP을 기준으로 부정적인 결과를 보이는 optimization을 모두 제거하는 방법을 반복 
- Optimization Space Exploration(OSE) => O(n^3) 
  - The basic idea of the pruning algorithm is to iteratively find better optimization combinations by merging the beneficial ones. 
- Statistical Selection (SS) =>O(n^2) 
  - It uses a statistical method to identify the performance effect of the optimization options. The options with positive effects are turned on, while the ones with negative effects are turned off in the final version, in an iterative fashion 
## Results

![SPEC CPU2000 부동소수점 벤치마크(Pentium IV)에서 BE, IE, OSE, SS, CE 다섯 조합 알고리즘의 정규화된 튜닝 시간을 비교한 막대그래프. CE 가 전반적으로 가장 짧다](/assets/images/fe1.png)
![같은 벤치마크에서 각 조합 알고리즘이 -O3 대비 낸 성능 향상률 막대그래프. art 에서 약 60%, sixtrack 에서 약 17~19% 로 크고 기하평균은 약 12% 다](/assets/images/fe2.png)
## references

https://arxiv.org/abs/1802.04799