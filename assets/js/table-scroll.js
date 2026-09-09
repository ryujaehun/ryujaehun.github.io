// 열이 많은 표를 가로로 밀어 볼 수 있게 한다.
//
// CSS(`overflow-x: auto`)만으로도 스크롤바·터치 드래그·트랙패드 가로
// 제스처·Shift+휠은 이미 동작한다. 여기서 더하는 것은 두 가지다.
//
//  1. 넘치는 표에만 표시를 붙인다. 스크롤할 것이 없는 표까지 스크롤바
//     자리를 잡거나 키보드 탭 순서에 끼어들면 방해만 된다. 넘치는지는
//     렌더 후에야 알 수 있어서 CSS 로는 판단할 수 없다.
//  2. 세로 휠을 가로 이동으로 바꾼다. 단, 표의 끝에 닿으면 그때부터는
//     페이지에 넘긴다 — 그러지 않으면 표 위에서 페이지가 멈춘다.
(function () {
  "use strict";

  function sync(el) {
    // 소수점 오차로 1px 넘치는 것을 스크롤로 보지 않는다.
    var scrollable = el.scrollWidth - el.clientWidth > 1;
    el.classList.toggle("can-scroll", scrollable);
    if (scrollable) {
      // 키보드만 쓰는 사람도 좌우 키로 볼 수 있어야 한다.
      el.setAttribute("tabindex", "0");
      el.setAttribute("role", "region");
      el.setAttribute("aria-label", el.dataset.cols + "열 표 (가로 스크롤)");
    } else {
      el.removeAttribute("tabindex");
      el.removeAttribute("role");
      el.removeAttribute("aria-label");
    }
  }

  function onWheel(event) {
    var el = event.currentTarget;
    // 트랙패드의 가로 제스처와 확대(Ctrl+휠)는 브라우저에 맡긴다.
    if (event.deltaX !== 0 || event.ctrlKey) return;
    var max = el.scrollWidth - el.clientWidth;
    if (max <= 1) return;

    // **이미 끝에 있을 때만** 페이지에 넘긴다. "한 번의 휠이 끝을
    // 넘어가면 넘긴다" 로 짰다가 실측에서 걸렸다 — 남은 거리가 휠
    // 한 칸보다 짧으면 표가 끝에 닿기 전에 페이지가 먼저 움직였다
    // (709px 짜리 표가 500px 에서 멈췄다). 남은 만큼은 밀고, 끝에
    // 닿은 다음 휠부터 페이지로 보낸다.
    if (event.deltaY > 0 && el.scrollLeft >= max - 1) return;
    if (event.deltaY < 0 && el.scrollLeft <= 0) return;

    el.scrollLeft = Math.max(0, Math.min(max, el.scrollLeft + event.deltaY));
    event.preventDefault();
  }

  function init() {
    var tables = document.querySelectorAll(".table-scroll");
    if (!tables.length) return;

    Array.prototype.forEach.call(tables, function (el) {
      sync(el);
      el.addEventListener("wheel", onWheel, { passive: false });
    });

    // 넘치는지는 폭에 따라 바뀐다. 창 크기, 폰트 로딩, 접힌 섹션이
    // 펼쳐지는 것까지 모두 영향을 준다.
    if (typeof ResizeObserver === "function") {
      var observer = new ResizeObserver(function (entries) {
        entries.forEach(function (entry) {
          sync(entry.target);
        });
      });
      Array.prototype.forEach.call(tables, function (el) {
        observer.observe(el);
      });
    } else {
      window.addEventListener("resize", function () {
        Array.prototype.forEach.call(tables, sync);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
