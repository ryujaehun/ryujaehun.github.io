---
title: "Mini-SGLang: reading the scheduler of an LLM serving framework"
description: "Reading sgl-project/mini-sglang in eight parts, pinned to a single commit — from the process boundaries a request crosses to the radix cache and two-stream overlap."
---

Eight parts reading [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang),
pinned to commit `9a91cfa`. Every quoted snippet links to a permalink at that commit.

It starts from the process boundaries one HTTP request crosses, then works
through the Req/Batch state ledger, the chunked-prefill budget, page allocation,
the radix cache and eviction that make prefix reuse possible, hiding CPU
scheduling behind two streams, and how several TP ranks reach the same decision.

Each part assumes the previous one, so reading from part 1 works best.
