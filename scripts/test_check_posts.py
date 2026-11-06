import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PinnedCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "blog"
        self.tool = Path(self.temp.name) / "tool"
        for repo in (self.root, self.tool):
            repo.mkdir()
            self.git(repo, "init", "-b", "main")
            self.git(repo, "config", "user.name", "Test")
            self.git(repo, "config", "user.email", "test@example.invalid")
        self.cli = self.tool / "tasks/papers/cli.py"
        self.cli.parent.mkdir(parents=True)
        self.write_checker("pinned")
        self.git(self.tool, "add", ".")
        self.git(self.tool, "commit", "-m", "checker")
        self.git(
            self.root,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(self.tool),
            "automation",
        )
        self.git(self.root, "commit", "-m", "pin")
        self.post = self.root / "post.md"
        self.post.write_text("valid")

    def git(self, repo, *args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    def write_checker(self, label):
        self.cli.write_text(
            "import pathlib,sys\n"
            f"print({label!r})\n"
            'p=pathlib.Path(sys.argv[sys.argv.index("--path")+1])\n'
            'raise SystemExit(7 if p.read_text()=="bad" else 0)\n'
        )

    def run_checker(self):
        from check_posts import run_checker

        return run_checker(self.root, [str(self.post)], fix=False)

    def test_dirty_tooling_is_ignored_without_modifying_it(self):
        dirty = self.root / "automation/tasks/papers/cli.py"
        dirty.write_text('raise RuntimeError("unfinished other work")\n')
        self.assertEqual(self.run_checker(), 0)
        self.assertEqual(dirty.read_text(), 'raise RuntimeError("unfinished other work")\n')

    def test_checker_rejection_is_preserved(self):
        self.post.write_text("bad")
        self.assertEqual(self.run_checker(), 7)

    def test_staged_gitlink_selects_upcoming_checker_commit(self):
        live = self.root / "automation"
        (live / "tasks/papers/cli.py").write_text("raise SystemExit(9)\n")
        self.git(live, "config", "user.name", "Test")
        self.git(live, "config", "user.email", "test@example.invalid")
        self.git(live, "add", ".")
        self.git(live, "commit", "-m", "new checker")
        self.git(self.root, "add", "automation")
        self.assertEqual(self.run_checker(), 9)

    def test_missing_pinned_object_fails_without_running_dirty_tooling(self):
        self.git(self.root, "update-index", "--cacheinfo", "160000," + "a" * 40 + ",automation")
        self.assertEqual(self.run_checker(), 2)

    def test_missing_gitlink_fails_closed(self):
        self.git(self.root, "rm", "--cached", "automation")
        self.assertEqual(self.run_checker(), 2)

    def test_git_hook_environment_does_not_redirect_the_submodule_object_lookup(self):
        with patch.dict(
            os.environ,
            {
                "GIT_DIR": str(self.root / ".git"),
                "GIT_WORK_TREE": str(self.root),
                "GIT_INDEX_FILE": str(self.root / ".git/index"),
            },
        ):
            self.assertEqual(self.run_checker(), 0)


if __name__ == "__main__":
    unittest.main()
