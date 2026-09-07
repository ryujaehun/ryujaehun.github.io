"""요약 백엔드.

세 가지를 지원한다.

- none     : 초안에 질의 프롬프트만 넣는다 (기존 create_post.py 동작)
- opencode : opencode 를 서브프로세스로 돌려 본문까지 채운다 (완전 자동)
- task     : 작업 지시서를 남기고 사람/에이전트가 이어받는다 (반자동)

API 키는 opencode 가 관리한다. 이 모듈은 자격 증명을 다루지 않는다.
"""

import json
import subprocess
from pathlib import Path


class SummarizeError(RuntimeError):
    """요약에 실패했을 때. 초안은 프롬프트만 있는 상태로 남긴다."""


def build_opencode_command(pdf_path, prompt, model, workdir, variant=None):
    """`opencode run` 명령을 만든다.

    --auto 가 필요한 이유: 비대화형 실행에서 이게 없으면 권한 프롬프트를
    기다리다 그대로 멈춘다(확인함). 대신 --dir 로 작업 범위를 리포 밖
    전용 디렉터리로 좁혀 파일 조작을 가둔다.

    --continue 를 쓰지 않는 이유: 논문마다 새 세션이어야 이전 논문의
    맥락이 섞이지 않는다.

    인자 순서가 중요하다. --file 은 배열 옵션이라 그 뒤에 프롬프트를 두면
    프롬프트까지 파일 경로로 삼켜 "File not found: <프롬프트>" 로 죽는다.
    그래서 프롬프트를 run 바로 뒤 위치 인자로 두고, --file 을 맨 끝에 둔다.

    경로는 절대 경로로 넘긴다. --dir 이 작업 디렉터리를 바꾸므로 상대
    경로는 그 기준으로 풀려 "File not found" 가 난다.
    """
    workdir = Path(workdir).absolute()
    pdf_path = Path(pdf_path).absolute()
    cmd = [
        "opencode",
        "run",
        prompt,
        "--format",
        "json",
        "--auto",
        "--dir",
        str(workdir),
        "--model",
        model,
    ]
    if variant:
        cmd += ["--variant", variant]
    cmd += ["--file", str(pdf_path)]
    return cmd


def _events(stdout):
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def extract_assistant_text(stdout):
    """`opencode run --format json` 출력에서 본문을 뽑는다.

    출력은 줄 단위 JSON 이벤트다. 본문은 type=="text" 이벤트의
    part.text 에 조각으로 실려 온다.
    """
    chunks = [
        event["part"]["text"]
        for event in _events(stdout)
        if event.get("type") == "text" and isinstance(event.get("part"), dict)
        and isinstance(event["part"].get("text"), str)
    ]
    if not chunks:
        raise SummarizeError(
            "opencode 출력에서 본문을 찾지 못했습니다. 원문 앞부분:\n"
            + (stdout or "")[:500]
        )
    return "".join(chunks)


def extract_run_cost(stdout):
    """step_finish 이벤트에 실려 오는 실행 비용(USD). 없으면 None."""
    for event in _events(stdout):
        part = event.get("part") or {}
        if event.get("type") == "step_finish" and "cost" in part:
            return part["cost"]
    return None


# 본문이라고 볼 최소 길이. 에이전트의 작업 보고문은 보통 1KB 안쪽이다.
MIN_BODY_CHARS = 2000

OUTPUT_CONTRACT = """

---

## 출력 규칙 (반드시 지킬 것)

위 지시에 따라 작성한 **블로그 포스트 전문**을 아래 경로 파일 하나에만 쓴다.

    {output_path}

- 이 파일에는 포스트 본문만 넣는다. 작업 요약이나 설명을 섞지 않는다.
- YAML front matter 는 쓰지 않는다. 파이프라인이 따로 붙인다.
- 다른 파일은 만들지 않는다.
"""


def build_review_prompt(prompt, output_path):
    """프롬프트에 출력 파일 계약을 덧붙인다.

    opencode 는 완성 응답을 내는 LLM 이 아니라 파일을 쓰고 결과를 보고하는
    에이전트다. 그대로 두면 본문을 제 맘대로 파일에 쓰고 stdout 으로는
    "완료했습니다" 같은 보고만 낸다. 그러면 그 보고문이 초안 본문이 된다.
    그래서 쓸 경로를 명시하고, 그 파일을 읽는다.
    """
    return prompt + OUTPUT_CONTRACT.format(output_path=Path(output_path).absolute())


def summarize_with_opencode(
    pdf_path,
    prompt,
    model,
    workdir,
    output_path,
    variant=None,
    timeout=900,
    min_chars=MIN_BODY_CHARS,
    runner=subprocess.run,
):
    """opencode 로 논문 요약 본문을 만든다. 실패하면 SummarizeError."""
    output_path = Path(output_path).absolute()
    # 이전 실행이 남긴 파일을 새 결과로 착각하지 않게 먼저 치운다.
    output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_opencode_command(
        pdf_path, build_review_prompt(prompt, output_path), model, workdir, variant=variant
    )
    try:
        completed = runner(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise SummarizeError(f"opencode 가 {timeout}초 안에 끝나지 않았습니다.") from exc
    except FileNotFoundError as exc:
        raise SummarizeError("opencode 실행 파일을 찾지 못했습니다.") from exc

    if completed.returncode != 0:
        raise SummarizeError(
            f"opencode 가 코드 {completed.returncode} 로 끝났습니다: "
            f"{(completed.stderr or '').strip()[:500]}"
        )

    # 약속한 파일이 우선이다. 없으면 stdout 이 통짜 본문인 경우만 받는다.
    if output_path.exists():
        body = output_path.read_text(encoding="utf-8")
        if len(body) >= min_chars:
            return body

    try:
        body = extract_assistant_text(completed.stdout)
    except SummarizeError:
        body = ""

    if len(body) >= min_chars:
        return body

    raise SummarizeError(
        f"본문이 너무 짧습니다({len(body)}자, 최소 {min_chars}자). "
        f"에이전트가 작업 보고만 냈을 수 있습니다. 앞부분:\n{body[:300]}"
    )


def write_task_file(
    tasks_dir, arxiv_id, title, pdf_path, draft_path, prompt_path, score, matched
):
    """반자동 경로용 작업 지시서. opencode TUI 나 Claude Code 가 읽는다."""
    tasks_dir = Path(tasks_dir)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    path = tasks_dir / f"{arxiv_id}.json"
    payload = {
        "id": arxiv_id,
        "title": title,
        "pdf_path": str(pdf_path),
        "draft_path": str(draft_path),
        "prompt_path": str(prompt_path),
        "score": score,
        "matched": matched,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return path


def model_chain(primary, fallback):
    """시도할 모델 순서. 앞에서부터 쓰고 실패하면 다음으로 넘어간다.

    같은 모델을 두 번 부르면 같은 이유로 두 번 죽으므로 중복은 지운다.
    """
    chain = []
    for model in (primary, fallback):
        model = (model or "").strip()
        if model and model not in chain:
            chain.append(model)
    if not chain:
        raise SummarizeError(
            "요약할 모델이 없습니다. data/paper-filter.yaml 의 summarize.model "
            "을 채우거나 --model 을 주세요."
        )
    return chain


def summarize_with_models(models, **kwargs):
    """모델을 순서대로 시도한다. `(본문, 실제로 쓴 모델)` 을 준다.

    폴백이 필요한 이유: 프로바이더 쪽 한도·장애로 한 모델이 통째로 막히는
    일이 있다. 그때 그날 수집분을 전부 버리는 대신 값싼 모델로라도 본문을
    남긴다. 어느 모델이 썼는지는 초안 카테고리에 남으므로 나중에 구분된다.
    """
    reasons = []
    for model in models:
        try:
            return summarize_with_opencode(model=model, **kwargs), model
        except SummarizeError as exc:
            reasons.append(f"{model}: {exc}")
    raise SummarizeError("모든 모델이 실패했습니다.\n" + "\n".join(reasons))
