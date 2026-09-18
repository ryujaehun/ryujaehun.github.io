---
title: "Mini-SGLang: LLM 서빙 프레임워크의 스케줄러 읽기"
description: "sgl-project/mini-sglang 을 커밋 하나에 고정해 8편으로 읽습니다. 요청이 지나는 프로세스 경계부터 radix cache, 두 스트림 겹치기까지."
---

[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) 을
커밋 `9a91cfa` 에 고정해 놓고 여덟 편에 걸쳐 읽습니다. 인용한 코드는 모두
그 커밋의 permalink 로 이어집니다.

HTTP 요청 하나가 답으로 돌아오기까지 지나는 프로세스 경계에서 시작해,
Req/Batch 상태 장부, chunked prefill 예산, 페이지 할당, 접두사 재사용을
위한 radix cache 와 eviction, 두 스트림으로 CPU 스케줄링을 감추는 방법,
그리고 여러 TP rank 가 같은 결정에 도달하는 구조까지 따라갑니다.

각 편은 앞 편을 읽었다고 가정합니다. 1편부터 순서대로 읽는 편이 낫습니다.
