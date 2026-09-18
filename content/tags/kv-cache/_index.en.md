---
title: "KV Cache"
description: "Paper reviews on shrinking, evicting and reusing the KV cache: compression, quantization, eviction signals, offloading and prefix reuse."
---

The problem is always the same — the KV cache grows with the decode length —
but the ways out differ. The axes these posts split along:

- **Choosing what to drop** — attention-score eviction, query-agnostic
  compression, eviction signals the model emits on its own
- **How precisely to keep it** — 4-bit and FP4 quantization, and the error that follows
- **Where to keep it** — pushing it off the GPU and reading back only what is needed
- **Whether it can be reused** — prefix sharing, cache reuse across models
