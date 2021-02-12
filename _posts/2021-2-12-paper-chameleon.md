---
title: 논문 정리 Chameleon Adaptive Code Optimization for Expedited Deep Neural Network Compilation(ICLR 2020)
categories:
 - compiler
 - ML
 - paper-review
tags:
---
![](/assets/images/chameleon1.jpg)
# 제목
Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation

# 저자
Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, Hadi Esmaeilzadeh

# Motivation
The current approaches are oblivious to the patterns in the design space of schedules that are available for exploitation, and causes inefficient search or even converges to solutions that may even be suboptimal.
Current solutions that rely on greedy sampling lead to significant fractions of the candidate configurations being redundant over iterations(long compilation time)
# Contribution
- Devising an __Adaptive Exploration__ module that utilizes reinforcement learning to adapt to unseen design space of new networks to reduce search time yet achieve better performance.
- Proposing an __Adaptive Sampling__ algorithm that utilizes clustering to adaptively reduce the number of costly hardware measurements
# 개인적인 생각
RL을 이용하여 exploration을 잘하고 sampling을 효율적으로 해서 time을 줄이고자하는 목적이 참 깔끔하고 좋은 논문.
# Overall design
![](/assets/images/chameleon2.png)

# Adaptive Exploration
- TVM leverages simulated annealing relies on the stochastic guarantees of its random walks(required numerous iterations of exploration) thus insufficient to enable disruptive innovations in neural networks
- Adaptive Exploration, based Reinforcement Learning ,is concerned with learning to maximize reward given an environment by making good exploration and exploitation tradeoffs
- These networks not only learn the dependencies among the different knobs of the design space (which are interrelated) that helps our module navigate through the design space but also lean the potential gains of the modifications to the configurations.
  
# Learning procedure
![](/assets/images/chameleon3.png)

# Adaptive Sampling : Reducing number of costly hardware measurements
![](/assets/images/chameleon4.png)
- we observe that the candidate configurations are clustered in subregions of the design space
- Our Adaptive Sampling iterates over a different number of clusters for their respective centroids and the L2 loss.(k-means)
- Selecting the number of centroids for clustering entails making the important tradeoff (using L2-performance degradation graph of knee of the curve
# Improving candidate configurations using sample synthesis
- Many of the automated approaches for black-box optimization are prone to invalid configurations
- These invalid configurations not only blow the chances for better exploration but also leads to an extra optimization time overhead to reset the physical hardware for the subsequent hardware measurement
- When our compiler runs into redundant samples, the proposed synthesis method analyzes the candidate samples to determine the most probable (most frequent = mode function) non-invalid choice for each knob to come up with a new configuration
# Improving candidate configurations using sample synthesis
- During training, some of the programs took a long time to compile, mainly when the agent was trying to vectorize more than plausible
- giving a penalty reward of −9 (equivalent to assuming it takes ten times the execution time of the baseline) so that the agent will learn not to overestimate the vectorization and avoid it
![](/assets/images/chameleon5.png)
the most probable (most frequent = mode function)
# Evaluation
![](/assets/images/chameleon6.png)
Task Index => layer order
![](/assets/images/chameleon7.png)
Overall, observation is that CHAMELEON’s Adaptive Exploration requires 2.88 less search steps compared to simulated annealing to find good solution.
![](/assets/images/chameleon8.png)
![](/assets/images/chameleon9.png)
![](/assets/images/chameleon10.png)


# references
https://openreview.net/forum?id=rygG4AVFvH
# Project Page
https://github.com/anony-sub/chameleon
