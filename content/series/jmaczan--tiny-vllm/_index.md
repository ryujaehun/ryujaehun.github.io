---
title: "tiny-vLLM: CUDA로 직접 쓴 LLM 추론 엔진 읽기"
description: "jmaczan/tiny-vllm 의 CUDA 1,500줄을 7편으로 읽습니다. 가중치 적재와 GPU 버퍼부터 block table KV 캐시와 continuous batching 까지."
---

[jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) 은 CUDA 로 직접
쓴 1,500 줄짜리 추론 엔진입니다. 커밋 하나에 고정해 놓고 일곱 편에 걸쳐
읽습니다.

가중치를 올리고 GPU 버퍼를 잡는 일부터 시작해, 프롬프트 전체를 한 번에
흘리는 prefill 경로, cuBLAS 에 행렬을 뒤집어 넘기는 이유, 토큰 하나를
만드는 decode 경로와 전용 커널, KV 캐시를 16토큰 블록으로 쪼개 block
table 로 읽는 방법, 슬롯과 큐로 여러 요청을 겹치는 continuous batching
까지 따라갑니다.
