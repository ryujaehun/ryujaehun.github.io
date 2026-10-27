"""Standalone failure monitor; mirrored in the public blog without dependencies.

Only trusted default-branch code runs. Downloaded reports/logs are read as data.
"""
import argparse
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BAD = {"failure", "timed_out", "action_required", "startup_failure", "stale"}
REPORTS = {"fetch.json", "summarize.json", "translate.json", "assets.json", "notify.json",
           "series-work.json", "series-notify.json"}


def sanitize(text, secrets=()):
    secrets = list(secrets) + [value for key, value in os.environ.items()
                              if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|CHAT_ID", key)]
    text = str(text)
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?:bot)?\d{6,}:[A-Za-z0-9_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"\b(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+|basic\s+)?\S+",
                  r"\1[REDACTED]", text)
    text = re.sub(r'''(?ix)((?:api[_-]?key|token|password|secret|chat[_-]?id)["']?\s*[:=]\s*)["']?[^\s,"'&]+''',
                  r"\1[REDACTED]", text)
    text = re.sub(r"https?://[^\s/@]+:[^\s/@]+@", "https://[REDACTED]@", text)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    return text


def gh_api(path):
    result = subprocess.run(["gh", "api", path], capture_output=True, timeout=60)
    if result.returncode:
        # Do not include subprocess stderr, which may contain credential URLs.
        raise RuntimeError("GitHub API 응답을 가져오지 못했습니다")
    if path.endswith(("/logs", "/zip")):
        return result.stdout
    return json.loads(result.stdout)


def list_pages(api, path, key):
    items = []
    for page in range(1, 21):
        batch = api(f"{path}?per_page=100&page={page}").get(key, [])
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


def report_failures(data):
    failures = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.filename not in REPORTS or info.file_size > 1_000_000:
                continue
            report = json.loads(archive.read(info))
            if not isinstance(report, dict):
                continue
            if report.get("status") == "notify-pending":
                failures.append(f"{info.filename}: {sanitize(report)[:600]}")
            for task in report.get("tasks", [])[:20]:
                if isinstance(task, dict) and task.get("status") not in {"complete", "idle"}:
                    failures.append(f"{info.filename}: {sanitize(task)[:600]}")
            for key in ("failed", "failed_sources", "pending"):
                values = report.get(key, [])
                if isinstance(values, list):
                    for value in values[:20]:
                        failures.append(f"{info.filename} {key}: {sanitize(value)[:240]}")
            source_errors = report.get("source_errors", {})
            if isinstance(source_errors, dict):
                for source, error in list(source_errors.items())[:20]:
                    failures.append(f"{info.filename}: {sanitize(source)[:120]}: {sanitize(error)[:600]}")
    return failures


def error_excerpt(raw):
    text = sanitize(raw.decode("utf-8", errors="replace"))
    lines = text.splitlines()
    errors = [line for line in lines if re.search(
        r"(?i)(##\[error\]|traceback|error:|exception:|timed?\s*out|실패|오류|\s!\s)", line)]
    # Do not send arbitrary log tails (which can include environment dumps).
    return "\n".join(errors[-6:])[:1600] or "명시적 오류 줄이 없습니다. 아래 실행 로그를 확인하세요."


def format_message(run, failures, details, warnings):
    attempt = run.get("run_attempt", 1)
    link = f"{run['html_url']}/attempts/{attempt}"
    header = (f"🚨 블로그 작업 {'실패' if run.get('conclusion') in BAD else '부분 실패'}\n"
              f"작업: {run.get('name', 'unknown')}\n"
              f"브랜치/커밋: {run.get('head_branch', '?')} / {run.get('head_sha', '?')[:7]}\n"
              f"상태: {run.get('conclusion', '?')} · 실행 시도: {attempt}\n"
              "재시도: 이 알림은 자동 재실행을 시작하지 않습니다. 단계 내부 재시도는 로그에서 확인하세요.\n")
    header = sanitize(header)
    body = sanitize("\n".join(failures + details + warnings))
    footer = f"\n실행 로그·수동 재실행:\n{link}"
    # Telegram limits UTF-16 units, not Python code points. Keep links intact.
    available = 3900 - len((header + footer).encode("utf-16-le")) // 2
    clipped = body.encode("utf-16-le")[:max(0, available) * 2].decode("utf-16-le", errors="ignore")
    return header + clipped + footer


def collect(run, repo, api=gh_api):
    if run.get("conclusion") == "cancelled":
        return None
    attempt = run.get("run_attempt", 1)
    prefix = f"repos/{repo}/actions"
    failures, details, warnings = [], [], []
    jobs = []
    try:
        jobs = list_pages(api, f"{prefix}/runs/{run['id']}/attempts/{attempt}/jobs", "jobs")
        for job in jobs:
            steps = [s for s in job.get("steps", []) if s.get("conclusion") in BAD]
            if job.get("conclusion") not in BAD and not steps:
                continue
            failures.append(f"단계: {job['name']} → " +
                            (", ".join(s['name'] for s in steps) or job.get("conclusion", "failure")))
            details.append(f"작업 로그: {run['html_url']}/job/{job['id']}")
            try:
                details.append(error_excerpt(api(f"{prefix}/jobs/{job['id']}/logs")))
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired):
                warnings.append("로그 요약을 가져오지 못했습니다. 실행 로그 링크를 확인하세요.")
        artifacts = list_pages(api, f"{prefix}/runs/{run['id']}/artifacts", "artifacts")
        for artifact in artifacts:
            if artifact['name'] in {f"paper-alert-reports-{run['id']}-{attempt}",
                                    f"series-alert-reports-{run['id']}-{attempt}"}:
                failures.extend(report_failures(api(f"{prefix}/artifacts/{artifact['id']}/zip")))
    except (RuntimeError, ValueError, OSError, zipfile.BadZipFile, subprocess.TimeoutExpired):
        warnings.append("일부 진단 자료를 가져오지 못했습니다. 실행 로그 링크를 확인하세요.")
    if failures and jobs and not details:
        try:
            details.append(error_excerpt(api(f"{prefix}/jobs/{jobs[0]['id']}/logs")))
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired):
            warnings.append("부분 실패의 오류 로그를 가져오지 못했습니다.")
    if not failures and run.get("conclusion") not in BAD:
        if warnings:
            raise RuntimeError("진단 자료 확인 실패: 알림 워크플로를 재실행하세요")
        return None
    return format_message(run, failures, details, warnings)


def send(message):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("Telegram secrets 설정이 필요합니다")
    values = [value for key, value in os.environ.items()
              if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|CHAT_ID", key)]
    message = sanitize(message, values)
    data = json.dumps({"chat_id": chat, "text": message, "disable_web_page_preview": True}).encode()
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                     data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if not json.load(response).get("ok"):
                raise RuntimeError("Telegram이 알림을 수락하지 않았습니다")
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Telegram 전송 실패: 알림 워크플로를 재실행하세요") from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-message", action="store_true")
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repo = os.environ["GITHUB_REPOSITORY"]
    if args.test_message:
        message = ("🧪 오류 알림 연결 테스트 — 실제 장애가 아닙니다.\n"
                   "실패 단계: 예시 / 수식 렌더링\n대상: 예시 글\n"
                   "오류 요약: 예시 명령 미지원\n재시도: 자동 재실행 없음\n"
                   f"로그: https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    else:
        run = event["workflow_run"]
        if (run.get("head_repository", {}).get("full_name") != repo
                or run.get("head_branch") != "main"
                or run.get("event") not in {"push", "schedule", "workflow_dispatch"}):
            print("Skipped untrusted or non-production run")
            return
        message = collect(run, repo)
    if message:
        send(message)
        print("Telegram detailed failure alert delivered")
    else:
        print("No failure; no Telegram alert sent")
    Path(".workflow-alert-sent").write_text("delivered-or-clean\n")


if __name__ == "__main__":
    main()
