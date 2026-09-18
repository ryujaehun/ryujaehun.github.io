---
title: "Distributed Computing"
description: "Paper reviews on LLM serving and training across GPUs and nodes: prefill/decode disaggregation, pipeline parallelism, overlapping communication with compute, scheduling and energy."
---

What happens once you go past a single GPU. The recurring axes:

- **Splitting the phases** — disaggregating prefill and decode to cut interference
- **Pipelines** — graph pipeline parallelism, max-flow placement over heterogeneous GPUs
- **Hiding communication** — overlapping transfer and compute through kernel fusion
- **Scheduling and cost** — throughput-latency tradeoffs, preemptible instances, energy efficiency
