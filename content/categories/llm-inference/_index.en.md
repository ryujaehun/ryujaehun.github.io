---
title: "LLM-Inference"
aliases:
  - /en/categories/llm-infernce/
description: "Paper reviews on making LLM inference faster and cheaper: speculative decoding, KV cache compression, prompt compression, sparsity, and heterogeneous pipelines."
---

Paper reviews about **running a trained model faster and cheaper**. The axes
that keep coming back:

- Drafting tokens ahead and verifying them — **speculative decoding**
- Shrinking the KV cache — quantization, sparsity-aware eviction, dynamic memory compression
- Shrinking the input side — prompt compression
- Splitting the hardware — heterogeneous pipelines, serving on consumer-grade GPUs
