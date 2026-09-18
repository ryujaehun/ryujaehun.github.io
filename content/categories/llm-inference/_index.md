---
title: "LLM-Inference"
aliases:
  - /categories/llm-infernce/
description: "LLM 추론을 빠르게, 싸게 만드는 방법을 다룬 논문 리뷰입니다. 추측 디코딩, KV 캐시 압축, 프롬프트 압축, 희소성 활용, 이기종 파이프라인 등을 다룹니다."
---

학습이 끝난 모델을 **어떻게 더 빠르고 싸게 돌릴 것인가** 를 다룬 논문
리뷰를 모았습니다. 되풀이해서 나오는 축들:

- 토큰을 미리 지어 놓고 검증하는 **추측 디코딩(speculative decoding)**
- KV 캐시를 줄이는 방향 — 양자화, 희소성 기반 축출, 동적 메모리 압축
- 입력 쪽을 줄이는 방향 — 프롬프트 압축
- 하드웨어를 나눠 쓰는 방향 — 이기종 파이프라인, 소비자용 GPU 서빙

`aliases` 로 옛 오타 주소(`/categories/llm-infernce/`)에서 이 페이지로
넘어옵니다.
