---
title: "abseil"
description: "Korean renderings of Abseil's Tip of the Week — string_view, unique_ptr, temporaries and lifetimes, and the other C++ questions that keep coming back."
---

Korean renderings of the **Tip of the Week** series that Google circulated
internally for its C++ library [Abseil](https://abseil.io/). Each post covers
one tip, keeping the original code examples and rewriting the prose in Korean.

Recurring themes:

- How to take a string parameter — `const char*` vs `const std::string&` vs `absl::string_view`
- Ownership and lifetime — `unique_ptr`, return values and references, reference lifetime extension
- When temporaries, moves and copies actually happen
- Notation choices such as `enum class` and raw string literals
- When a test fixture earns its place, and how data flows through one

The originals live at [abseil.io/tips](https://abseil.io/tips/).
