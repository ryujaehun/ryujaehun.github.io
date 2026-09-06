import json
import subprocess
from pathlib import Path

import pytest

from paperlib.summarize import (
    SummarizeError,
    build_opencode_command,
    extract_assistant_text,
    summarize_with_opencode,
    write_task_file,
)


def test_build_opencode_command_attaches_the_pdf_and_pins_the_model():
    cmd = build_opencode_command(
        pdf_path="/tmp/2609.03430.pdf",
        prompt="요약해줘",
        model="opencode-go/glm-5.3",
        workdir="/tmp/paper-workdir",
    )

    assert cmd[0] == "opencode"
    assert cmd[1] == "run"
    assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "json"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "opencode-go/glm-5.3"
    assert "--file" in cmd and cmd[cmd.index("--file") + 1] == "/tmp/2609.03430.pdf"
    # --file 은 배열 옵션이라 뒤에 프롬프트를 두면 그것까지 파일로 삼킨다.
    # (실제 오류: "File not found: Read the attached PDF...")
    # 그래서 메시지를 run 바로 뒤 위치 인자로 둔다.
    assert cmd[2] == "요약해줘"
    assert cmd[cmd.index("--file") + 1] == "/tmp/2609.03430.pdf"
    assert cmd[-1] == "/tmp/2609.03430.pdf", "--file 값이 명령의 끝이어야 한다"
    # 세션을 이어붙이면 이전 논문 맥락이 섞인다
    assert "--continue" not in cmd
    # 비대화형에서는 --auto 없이 권한 프롬프트에 걸려 멈춘다.
    # 대신 --dir 로 작업 범위를 리포 밖으로 가둔다.
    assert "--auto" in cmd
    assert cmd[cmd.index("--dir") + 1] == "/tmp/paper-workdir"


def test_build_opencode_command_omits_variant_when_not_configured():
    without = build_opencode_command("/a.pdf", "p", "m", "/w", variant=None)
    with_variant = build_opencode_command("/a.pdf", "p", "m", "/w", variant="high")

    assert "--variant" not in without
    assert with_variant[with_variant.index("--variant") + 1] == "high"


# opencode 1.18.27 이 --format json 으로 실제로 내는 모양.
# 줄 단위 JSON 이벤트이고 본문은 type=="text" 이벤트의 part.text 에 있다.
def _event(kind, part):
    return json.dumps({"type": kind, "timestamp": 0, "sessionID": "s", "part": part})


def test_extract_assistant_text_joins_text_events_in_order():
    stdout = "\n".join(
        [
            _event("step_start", {"type": "step-start"}),
            _event("text", {"type": "text", "text": "## 요약\n"}),
            _event("text", {"type": "text", "text": "본문입니다.\n"}),
            _event("step_finish", {"type": "step-finish", "reason": "stop", "cost": 0.02}),
        ]
    )

    assert extract_assistant_text(stdout) == "## 요약\n본문입니다.\n"


def test_extract_assistant_text_ignores_non_text_events():
    stdout = "\n".join(
        [
            _event("tool", {"type": "tool", "tool": "read"}),
            _event("text", {"type": "text", "text": "진짜 본문"}),
        ]
    )

    assert extract_assistant_text(stdout) == "진짜 본문"


def test_extract_run_cost_reads_the_step_finish_event():
    from paperlib.summarize import extract_run_cost

    stdout = "\n".join(
        [
            _event("text", {"type": "text", "text": "x"}),
            _event("step_finish", {"type": "step-finish", "cost": 0.00223775}),
        ]
    )

    assert extract_run_cost(stdout) == pytest.approx(0.00223775)


def test_extract_assistant_text_refuses_to_invent_a_body():
    with pytest.raises(SummarizeError):
        extract_assistant_text("")
    with pytest.raises(SummarizeError):
        extract_assistant_text("총체적 난국\n아무 JSON 도 아님")


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _agent_writing(path, article):
    """에이전트처럼 실행 중에 결과 파일을 쓰는 가짜 runner."""

    def runner(*args, **kwargs):
        Path(path).write_text(article, encoding="utf-8")
        return _Completed(stdout=_event("text", {"type": "text", "text": "완료했습니다."}))

    return runner


def test_summarize_with_opencode_returns_the_generated_body(tmp_path):
    article = "# 생성된 본문\n\n" + ("내용 " * 2000)
    out = tmp_path / "review.md"

    body = summarize_with_opencode(
        pdf_path="/a.pdf",
        prompt="p",
        model="m",
        workdir=tmp_path,
        output_path=out,
        runner=_agent_writing(out, article),
    )

    assert body == article


def test_summarize_with_opencode_preserves_stderr_on_failure(tmp_path):
    def runner(*args, **kwargs):
        return _Completed(returncode=1, stderr="provider auth failed")

    with pytest.raises(SummarizeError, match="provider auth failed"):
        summarize_with_opencode(
            pdf_path="/a.pdf",
            prompt="p",
            model="m",
            workdir=tmp_path,
            output_path=tmp_path / "review.md",
            runner=runner,
        )


def test_summarize_with_opencode_turns_a_timeout_into_a_summarize_error(tmp_path):
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="opencode", timeout=5)

    with pytest.raises(SummarizeError, match="끝나지 않"):
        summarize_with_opencode(
            pdf_path="/a.pdf",
            prompt="p",
            model="m",
            workdir=tmp_path,
            output_path=tmp_path / "review.md",
            timeout=5,
            runner=runner,
        )


def test_write_task_file_records_everything_an_agent_needs(tmp_path):
    path = write_task_file(
        tasks_dir=tmp_path,
        arxiv_id="2609.03430",
        title="Random Attention",
        pdf_path="/cache/2609.03430.pdf",
        draft_path="content/posts/x.md",
        prompt_path="data/paper-prompts/paper-review.md",
        score=0.71,
        matched={"core": ["kv cache"]},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["id"] == "2609.03430"
    assert payload["pdf_path"] == "/cache/2609.03430.pdf"
    assert payload["draft_path"] == "content/posts/x.md"
    assert payload["prompt_path"].endswith("paper-review.md")
    assert payload["matched"] == {"core": ["kv cache"]}


def test_build_opencode_command_absolutises_paths():
    """--dir 로 작업 디렉터리가 바뀌므로 상대 경로는 그 기준으로 풀린다.

    실제 오류: --dir .cache/papers/workdir 인데 --file .cache/papers/x.pdf 를
    주면 "File not found: .cache/papers/x.pdf" 로 죽는다.
    """
    import os

    cmd = build_opencode_command("rel/x.pdf", "p", "m", "rel/workdir")

    assert cmd[cmd.index("--file") + 1] == os.path.abspath("rel/x.pdf")
    assert cmd[cmd.index("--dir") + 1] == os.path.abspath("rel/workdir")


# opencode 는 완성 응답을 내는 LLM 이 아니라 파일을 쓰고 보고하는 에이전트다.
# 실제로 겪은 일: 28KB 리뷰를 파일에 쓰고 stdout 으로는 953 바이트짜리
# 작업 보고를 냈다. 그 보고문이 초안 본문으로 들어가 원본을 덮어썼다.
AGENT_STATUS_REPORT = (
    "완료했습니다. /w/2609.03430-review.md 에 블로그 포스트 전체를 저장했습니다.\n"
    "- 논문: Random Attention — 프롬프트만 보호하고 KV 캐시를 무작위 eviction\n"
    "Hugo narrow 테마에 그대로 넣으시면 됩니다.\n"
)


def test_prompt_carries_an_explicit_output_file_contract():
    from paperlib.summarize import build_review_prompt

    prompt = build_review_prompt("원래 프롬프트 본문", "/w/2609.03430-review.md")

    assert "원래 프롬프트 본문" in prompt
    assert "/w/2609.03430-review.md" in prompt


def test_summarize_reads_the_output_file_not_the_status_message(tmp_path):
    out = tmp_path / "review.md"
    article = "# 제목\n\n" + ("본문 " * 2000)

    def runner(*args, **kwargs):
        out.write_text(article, encoding="utf-8")
        return _Completed(
            stdout=_event("text", {"type": "text", "text": AGENT_STATUS_REPORT})
        )

    body = summarize_with_opencode(
        pdf_path="/a.pdf",
        prompt="p",
        model="m",
        workdir=tmp_path,
        output_path=out,
        runner=runner,
    )

    assert body == article
    assert "완료했습니다" not in body


def test_summarize_rejects_a_status_message_when_no_file_was_written(tmp_path):
    """파일도 없고 stdout 도 짧으면 실패다. 초안을 덮어써선 안 된다."""
    with pytest.raises(SummarizeError, match="너무 짧"):
        summarize_with_opencode(
            pdf_path="/a.pdf",
            prompt="p",
            model="m",
            workdir=tmp_path,
            output_path=tmp_path / "missing.md",
            runner=lambda *a, **kw: _Completed(
                stdout=_event("text", {"type": "text", "text": AGENT_STATUS_REPORT})
            ),
        )


def test_summarize_falls_back_to_stdout_when_it_is_a_full_article(tmp_path):
    article = "# 제목\n\n" + ("본문 " * 2000)

    body = summarize_with_opencode(
        pdf_path="/a.pdf",
        prompt="p",
        model="m",
        workdir=tmp_path,
        output_path=tmp_path / "missing.md",
        runner=lambda *a, **kw: _Completed(
            stdout=_event("text", {"type": "text", "text": article})
        ),
    )

    assert body == article


def test_stale_output_file_is_removed_before_the_run(tmp_path):
    """이전 실행이 남긴 파일을 새 결과로 착각하면 안 된다."""
    out = tmp_path / "review.md"
    out.write_text("# 예전 실행 결과\n" + ("낡은 내용 " * 2000), encoding="utf-8")

    seen = {}

    def runner(*args, **kwargs):
        # 에이전트가 돌기 시작하는 시점에 낡은 파일이 남아 있으면 안 된다
        seen["existed"] = out.exists()
        return _Completed(stdout="")

    with pytest.raises(SummarizeError):
        summarize_with_opencode(
            pdf_path="/a.pdf",
            prompt="p",
            model="m",
            workdir=tmp_path,
            output_path=out,
            runner=runner,
        )

    assert seen["existed"] is False
