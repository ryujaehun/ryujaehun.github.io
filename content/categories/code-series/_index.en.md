---
title: "code-series"
description: "Reading an open-source inference engine end to end, pinned to one commit. Each part follows the execution path through a specific region of the code and the reasoning behind its design."
---

Pick a small inference-engine repository, **pin it to a single commit**, and
read it from front to back. Every quoted snippet links to a permalink at that
commit, so the writing and the code never drift apart.

Repositories covered so far:

- **[jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm)** (7 parts) —
  a 1,500-line CUDA inference engine: weight loading and GPU buffers, the
  prefill path, the cuBLAS transpose trick, the decode kernel, a 16-token
  block KV cache with a block table, and continuous batching over slots and
  queues.
- **[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)** (8 parts) —
  starting from the process boundaries one request crosses: the Req/Batch
  state ledger, the chunked-prefill budget, page allocation, the radix cache
  and eviction, and overlapping work across two streams.

Each part assumes the previous one, so reading from part 1 works best.
