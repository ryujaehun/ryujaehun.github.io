---
title: "code-series"
description: "오픈소스 추론 엔진을 한 커밋에 고정해 놓고 처음부터 끝까지 읽는 연재입니다. 실행 경로를 따라가며 각 편이 코드의 특정 구간과 그 설계 이유를 다룹니다."
---

작은 추론 엔진 저장소를 하나 고르고, **커밋 하나에 고정한 뒤** 처음부터
끝까지 읽어 나가는 연재입니다. 인용한 코드는 모두 그 커밋의 permalink
로 연결되므로 저장소가 바뀌어도 글과 코드가 어긋나지 않습니다.

지금까지 다룬 저장소:

- **[jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm)** (7편) —
  CUDA 1,500 줄짜리 추론 엔진. 가중치 적재와 GPU 버퍼, prefill 경로,
  cuBLAS 전치 트릭, decode 커널, 16토큰 블록 KV 캐시와 block table,
  슬롯·큐로 요청을 겹치는 continuous batching 까지.
- **[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)** (8편) —
  요청 하나가 지나는 프로세스 경계에서 시작해 Req/Batch 상태 장부,
  chunked prefill 예산, 페이지 할당, radix cache 와 eviction, 두 스트림
  겹치기까지.

각 편은 앞 편을 읽었다고 가정하므로 1편부터 순서대로 읽는 편이 낫습니다.
