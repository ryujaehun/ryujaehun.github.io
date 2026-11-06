"""Run the publication checker pinned by the blog's staged automation gitlink."""

import argparse
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def run_checker(workspace, paths, *, fix=True):
    workspace = Path(workspace).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "--stage", "--", "automation"],
            capture_output=True,
            text=True,
            check=True,
        )
        fields = result.stdout.split()
        if (
            len(fields) != 4
            or fields[0] != "160000"
            or fields[2:] != ["0", "automation"]
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", fields[1])
        ):
            raise ValueError("automation must have one staged gitlink without conflicts")
        revision = fields[1]
        local_names = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--local-env-vars"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        archive_env = os.environ.copy()
        for name in local_names:
            archive_env.pop(name, None)
        with tempfile.TemporaryDirectory(prefix="blog-pinned-check-") as temporary:
            with tempfile.TemporaryFile() as archive:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(workspace / "automation"),
                        "archive",
                        "--format=tar",
                        revision,
                    ],
                    stdout=archive,
                    stderr=subprocess.PIPE,
                    check=True,
                    env=archive_env,
                )
                archive.seek(0)
                with tarfile.open(fileobj=archive, mode="r:") as content:
                    content.extractall(temporary, filter="data")
            checker = Path(temporary) / "tasks/papers/cli.py"
            if not checker.is_file():
                raise ValueError("pinned publication checker is missing")
            command = [sys.executable, str(checker), "check", "--workspace", str(workspace)]
            if fix:
                command.append("--fix")
            command += ["--path", *map(str, paths)]
            return subprocess.run(command, cwd=workspace, check=False).returncode
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        print("Pinned publication checker unavailable: " + type(exc).__name__, file=sys.stderr)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Read-only validation without fixes")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    return run_checker(args.workspace, args.paths, fix=not args.check)


if __name__ == "__main__":
    raise SystemExit(main())
