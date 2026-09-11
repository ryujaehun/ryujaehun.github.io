# tiny-vllm: CUDA로 읽는 LLM 추론 엔진

## Scope

This series reads commit `1896ff5c37a241050dbcd9527caf9dc1d3087a61` of
`jmaczan/tiny-vllm`. It follows the executable path from model loading through
one-token generation, batching, and paged KV-cache attention. Training,
distributed serving, and performance claims outside this repository are out of
scope.

## Evidence rules

Every implementation claim must cite the pinned source path and a line range.
README explanations are supporting evidence; code and tests decide conflicts.
The series distinguishes the repository's teaching implementation from vLLM's
production behavior.

## Reader contract

Readers need C++ basics, CUDA thread/block terminology, and matrix
multiplication intuition. Each chapter explains the required transformer
operation before following its implementation.

## Series arc

Start with the runnable program and weight format, establish the token-to-logit
forward path, then explain how memory and scheduling change that path for many
requests. End with paged attention as the point where the data structure and
kernel design meet.
