---
title: "abseil"
description: "Abseil Tip of the Week 를 한국어로 옮긴 글 모음입니다. string_view, unique_ptr, 임시 객체와 수명처럼 C++ 를 쓰면서 매번 다시 확인하게 되는 주제를 한 편에 하나씩 다룹니다."
---

구글의 C++ 라이브러리 [Abseil](https://abseil.io/) 이 사내에 돌리던
**Tip of the Week** 를 한국어로 옮긴 글들입니다. 한 편이 팁 하나에
대응하고, 원문의 코드 예시를 그대로 쓰되 설명은 우리말로 다시
썼습니다.

자주 찾게 되는 주제들:

- 문자열을 인자로 받는 방법 — `const char*` / `const std::string&` / `absl::string_view`
- 소유권과 수명 — `unique_ptr`, 반환 값과 참조, 참조 수명 연장
- 임시 객체, 이동, 복사가 실제로 언제 일어나는가
- `enum class`, raw 문자열 리터럴 같은 표기 선택
- 테스트 픽스처를 두는 기준과 데이터 흐름

원문은 [abseil.io/tips](https://abseil.io/tips/) 에 있습니다.
