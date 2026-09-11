# Decision log

## 2026-09-11 — Audience and emphasis

The repository is a C++/CUDA teaching implementation of an LLM inference
engine. The series can either explain the end-to-end implementation for readers
who know basic CUDA, or focus on the scheduling and memory choices that connect
it to production inference servers.

- Status: decided
- Question: Which is the primary audience: CUDA beginners building an engine,
  or experienced ML-systems readers comparing implementation trade-offs?
- Decision: CUDA beginners who know basic C++ and linear algebra. Explain the
  implementation in execution order; use production systems only to clarify
  trade-offs, not as the series' primary subject.
