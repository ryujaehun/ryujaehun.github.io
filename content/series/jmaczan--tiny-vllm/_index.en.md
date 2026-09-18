---
title: "tiny-vLLM: reading an LLM inference engine written in CUDA"
description: "Reading the 1,500 lines of CUDA in jmaczan/tiny-vllm across seven parts — from weight loading and GPU buffers to a block-table KV cache and continuous batching."
---

[jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) is a 1,500-line
inference engine written directly in CUDA. Seven parts read it end to end,
pinned to a single commit.

It begins with loading weights and laying out GPU buffers, then the prefill
path that streams a whole prompt at once, why matrices are handed to cuBLAS
transposed, the decode path that produces one token and the kernel written for
it, splitting the KV cache into 16-token blocks read through a block table, and
continuous batching over slots and queues.
