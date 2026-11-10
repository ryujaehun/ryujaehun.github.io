import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('publish-pages.sh')
SOURCE_SHA = 'a' * 40


class PublishPagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / 'remote.git'
        self.site = self.root / 'public'
        self.checkout = self.root / 'published'
        self.site.mkdir()
        (self.site / 'index.html').write_text('<h1>Current site</h1>')
        (self.site / 'CNAME').write_text('jaehun.me\n')
        self.git(self.root, 'init', '--bare', str(self.remote))
        self.git(self.root, 'clone', str(self.remote), str(self.checkout))
        self.git(self.checkout, 'switch', '--orphan', 'gh-pages')
        self.git(self.checkout, 'config', 'user.name', 'Test')
        self.git(self.checkout, 'config', 'user.email', 'test@example.com')
        (self.checkout / 'obsolete.html').write_text('old')
        self.git(self.checkout, 'add', '.')
        self.git(self.checkout, 'commit', '-m', 'previous publication')
        self.git(self.checkout, 'push', 'origin', 'gh-pages')
        self.previous = self.git(self.checkout, 'rev-parse', 'HEAD').stdout.strip()

    def git(self, directory, *args):
        return subprocess.run(['git', '-C', str(directory), *args],
                              capture_output=True, text=True, check=True)

    def publish(self):
        return subprocess.run(['bash', str(SCRIPT), str(self.site), str(self.checkout)],
                              env=dict(os.environ, GITHUB_SHA=SOURCE_SHA),
                              capture_output=True, text=True)

    def test_publishes_only_built_files_preserving_branch_history(self):
        (self.root / 'private.env').write_text('must not be published')
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD^').stdout.strip(),
                         self.previous)
        files = self.git(self.remote, 'ls-tree', '-r', '--name-only', 'gh-pages').stdout.splitlines()
        self.assertEqual(sorted(files), ['.nojekyll', 'CNAME', 'build-info.json', 'index.html'])
        marker = json.loads(self.git(self.remote, 'show', 'gh-pages:build-info.json').stdout)
        self.assertEqual(marker, {'commit': SOURCE_SHA})
        self.assertEqual(self.git(self.remote, 'show', 'gh-pages:CNAME').stdout, 'jaehun.me\n')

    def test_unchanged_build_does_not_create_another_commit(self):
        first = self.publish()
        self.assertEqual(first.returncode, 0, first.stderr)
        head = self.git(self.checkout, 'rev-parse', 'HEAD').stdout
        second = self.publish()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD').stdout, head)

    def test_refuses_to_overwrite_a_source_branch(self):
        self.git(self.checkout, 'switch', '-c', 'main')
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.checkout / 'obsolete.html').exists())
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD').stdout.strip(),
                         self.previous)

    def test_refuses_a_directory_without_a_built_homepage(self):
        (self.site / 'index.html').unlink()
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.checkout / 'obsolete.html').exists())


if __name__ == '__main__':
    unittest.main()
