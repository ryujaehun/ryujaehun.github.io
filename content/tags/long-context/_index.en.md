---
title: "Long Context"
description: "Paper reviews on coping with long context: sparse attention, context compression and recompute, state-space models, and million-token serving."
---

As context grows, attention cost and the KV cache grow with it. These posts
generally take one of four routes:

- **Make attention sparse** — hardware-aligned, natively trainable sparsity
- **Compress, then recompute** — shrink the context and rebuild what is needed
- **Change the architecture** — state-space models and their relation to Transformers
- **Absorb it in serving** — parallelism for million-token requests
