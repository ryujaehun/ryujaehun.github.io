#!/usr/bin/env python3
"""Observe NAS liveness from GitHub-hosted infrastructure without private data."""

from __future__ import annotations

import http.client
import json
import os
import ssl
from datetime import UTC, datetime
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPOSITORY = "ryujaehun/ryujaehun.github.io"
MAX_AGE_SECONDS = 30 * 60


def fresh(value: str, now: datetime) -> bool:
    age = (now - datetime.fromisoformat(value.replace("Z", "+00:00"))).total_seconds()
    return 0 <= age <= MAX_AGE_SECONDS


def heartbeat_valid(run: dict, jobs: list[dict], now: datetime) -> bool:
    """Require a recent dispatch and a real successful self-hosted job."""
    try:
        return (
            run["status"] == "completed"
            and run["conclusion"] == "success"
            and run["head_branch"] == "main"
            and run["event"] == "workflow_dispatch"
            and run["display_title"].startswith("NAS heartbeat ")
            and fresh(run["display_title"].removeprefix("NAS heartbeat "), now)
            and fresh(run["created_at"], now)
            and any(
                job["name"] == "heartbeat"
                and job["status"] == "completed"
                and job["conclusion"] == "success"
                and fresh(job["completed_at"], now)
                and {"self-hosted", "local-ryujaehun.github.io"} <= set(job["labels"])
                for job in jobs
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def protected(status: int, location: str) -> bool:
    target = urlsplit(location)
    return status in (401, 403) or (
        status == 302
        and target.path == "/outpost.goauthentik.io/start"
        and target.hostname in (None, "status.jaehun.me")
        and target.scheme in ("", "https")
        and target.port in (None, 443)
        and target.username is None
    )


def api(path: str) -> dict:
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "external-ci-watchdog",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def request(host: str, path: str) -> tuple[int, str]:
    connection = http.client.HTTPSConnection(host, timeout=12, context=ssl.create_default_context())
    try:
        connection.request("GET", path, headers={"User-Agent": "external-ci-watchdog"})
        response = connection.getresponse()
        return response.status, response.getheader("Location", "")
    finally:
        connection.close()  # Do not read/log bodies, cookies or redirect query strings.


def main() -> int:
    checks = {}
    try:
        runs = api(
            "actions/workflows/local-ci-heartbeat.yaml/runs?branch=main&event=workflow_dispatch&status=success&per_page=5"
        )["workflow_runs"]
        now = datetime.now(UTC)
        checks["heartbeat"] = any(
            heartbeat_valid(run, api(f"actions/runs/{int(run['id'])}/jobs")["jobs"], now)
            for run in runs
        )
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
        checks["heartbeat"] = False
    for name, host, path in (
        ("status_auth_boundary", "status.jaehun.me", "/api/status/current"),
        ("auth_login_tls", "auth.jaehun.me", "/if/flow/default-authentication-flow/"),
    ):
        try:
            status, location = request(host, path)
            checks[name] = (
                protected(status, location) if name == "status_auth_boundary" else status == 200
            )
        except (OSError, ValueError, http.client.HTTPException):
            checks[name] = False
    # Only booleans and observation time are published. No NAS snapshot or private API is read.
    print(json.dumps({"observed_at": datetime.now(UTC).isoformat(), "checks": checks}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
