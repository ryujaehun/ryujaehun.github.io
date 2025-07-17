---
categories:
- compiler
- paper-review
date: "2021-02-12T00:00:00Z"
tags: null
title: 논문 정리 LLVM A Compilation Framework for Lifelong Program Analysis & Transformation(CGO
  04)
---
# 제목
LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation

# 저자
Chris Lattner Vikram Adve

# 개인적으로 느끼는 논문의 insight 
Lifelong Program Analysis개념을 도입하여 Front-end를 제외한 부분에서 전체적인 최적화를 수행,SSA,machine-independent optimization
논문에서 제시된 개념이 지금의 llvm과 정확하게 일치하지는 않지만 대단하다.. 

# Motivation
- Multiple-stages of analysis & transformation
- compile-time, link-time, install-time, run-time, idle-time
- Use aggressive interprocedural optimizations
- Gather and exploit end-user profile information
- Tune the application to the user’s hardware
# Contributions
- A persistent, rich code representation
  - Enables analysis & optimization throughout lifetime
- Offline native code generation
  - Must be able to generate high-quality code statically
- Profiling & optimization in the field
  - Adapt to the end-user’s usage patterns
- Language independence
  - No runtime, object model, or exception semantics
- Uniform whole-program optimization
  - Allow optimization across languages and runtime
# Instruction Set
- Avoids machine specific constraints
- Infinite set of typed virtual registers
  - In SSA form
  - Includes support for phi functions
  - This allows flow insensitive algorithm to gain benefits of flow sensitive without expensive Data Flow analysis
- Avoids same code for multiple instructions (overloaded opcodes)
- Exceptions mechanism based on two instructions invoke and unwind
# LLVM Compiler Architecture
![](/assets/images/llvm1.png)
__This strategy provides the 5 benefits__
- Some limitations
  - Language specific optimizations must be performed on frontend
  - Benefit to languages like Java(JVM) requiring sophisticated runtime systems?
- Front-end compiler
  - Translate source code to LLVM representation
  - Perform language specific optimizations
  - Need not perform SSA construction at this time
  - Invoke LLVM passes for global inter procedural optimization at module level
- Linker/Interprocedure Optimizer
  - Various analyses occur
    - Points-to analysis
    - Mod/Ref analysis
    - Dead global elimination, dead argument elimination, constant, propagation, array bounds check, etc
    - Can be speeded up by adding inter-procedural summaries
- Native Code Generation
  - JIT or Offline
  - Currently supports Sparc V9 and x86 architectures
- Reoptimizers
  - Identifies frequently run code and ‘hotspots’
  - Performs additional optimizations, thus native code generation can be performed ahead of time
  - Idle-time reoptimizer

# Results:How do high-level features map onto LLVM?
![](/assets/images/llvm2.png)
The table shows that many of these programs (164, 176,
179, 181, 183, 186, & 256) are surprisingly type-safe, despite
the fact that the programming language does not enforce
type-safety.
![](/assets/images/llvm3.png)
The figure shows that LLVM code is about the same size
as native executables for SPARC, and is roughly 25% larger
on average for x86
![](/assets/images/llvm4.png)
DGE (aggressive10 Dead
global variable and function Elimination), DAE (an aggressive Dead Argument Elimination), inline (a function integration pass), DSA (Data Structure Analysis), and GCC
(time to compile the programs with the gcc 3.3 compiler at –
O3, provided as a reference point)

# references
https://ieeexplore.ieee.org/document/1281665