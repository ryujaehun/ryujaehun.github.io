---
categories:
- compiler
- ML
- paper-review
date: "2021-02-12"
tags: null
title: 간단논문 정리 End-to-End Deep Learning of Optimization Heuristics (PACT 17)
description: '사람이 설계한 특징 없이 원시 소스 코드를 언어 모델로 학습해 컴파일러 최적화 휴리스틱을 예측하는 PACT 2017 논문을 정리한 글입니다.'
---
## 제목
End-to-End Deep Learning of Optimization Heuristics

## 저자
Chris Cummins ; Pavlos Petoumenos ; Zheng Wang ; Hugh Leather

## Motivation
 기존 머신러닝을 이용한 compiler optimizaion 방법에서는  human experts를 이용한 feature engineering 이 필요 

## Contribution
논문에서 제안하는 Source Rewriter & Language model을 이용하여 RAW PROGRAM CODE을 직접 이용하여 compiler optimizaion을 수행 아울어 transfer learning 을 이용하여 small number of program 에서도 학습을 수행 

## references
https://ieeexplore.ieee.org/document/1281665