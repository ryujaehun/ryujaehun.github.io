import tempfile
import unittest
from pathlib import Path

from check_local_links import check_links


class LocalLinksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content=''):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_missing_article_image_and_relative_link_are_reported(self):
        self.write('posts/review/index.html', '''
            <img src="/assets/missing.webp">
            <a href="protobuf">Protocol Buffers</a>
        ''')
        count, problems = check_links(self.root)
        self.assertEqual(count, 1)
        self.assertEqual(problems, [
            ('posts/review/index.html', 'img', '/assets/missing.webp'),
            ('posts/review/index.html', 'a', 'protobuf'),
        ])

    def test_existing_unicode_query_fragment_and_relative_targets_pass(self):
        self.write('posts/review/index.html', '''
            <img src="../../assets/plot.svg?version=1&amp;theme=light">
            <a href="/posts/%ED%95%9C%EA%B8%80/#section">Korean</a>
            <a href="#section">Same page</a>
            <a href="?page=2">Query</a>
            <a href="/index.xml">RSS</a>
        ''')
        self.write('assets/plot.svg')
        self.write('posts/한글/index.html')
        self.write('index.xml')
        self.assertEqual(check_links(self.root), (2, []))

    def test_remote_and_embedded_urls_are_not_fetched(self):
        self.write('index.html', '''
            <a href="https://example.com/missing">External</a>
            <img src="//cdn.example.com/missing.png">
            <img src="data:image/png;base64,AA==">
            <a href="mailto:test@example.com">Mail</a>
            <a href="javascript:void(0)">Button</a>
        ''')
        self.assertEqual(check_links(self.root), (1, []))

    def test_same_origin_absolute_urls_are_checked(self):
        self.write('index.html', '''
            <a href="https://jaehun.me/missing/">Missing</a>
            <img src="//jaehun.me/assets/ok.svg">
        ''')
        self.write('assets/ok.svg')
        self.assertEqual(check_links(self.root)[1], [
            ('index.html', 'a', 'https://jaehun.me/missing/'),
        ])

    def test_directory_without_index_is_not_a_working_page(self):
        self.write('index.html', '<a href="/assets/">Empty directory</a>')
        self.write('assets/plot.svg')
        self.assertEqual(check_links(self.root)[1], [('index.html', 'a', '/assets/')])

    def test_site_deployed_under_a_base_path(self):
        self.write('index.html', '''
            <img src="/blog/assets/plot.svg">
            <a href="https://example.com/blog/posts/entry/">Post</a>
        ''')
        self.write('assets/plot.svg')
        self.write('posts/entry/index.html')
        self.assertEqual(check_links(self.root, 'https://example.com/blog/'), (2, []))

    def test_escaped_example_markup_is_not_a_real_link(self):
        self.write('index.html', '<pre>&lt;img src="/not-an-image"&gt;</pre>')
        self.assertEqual(check_links(self.root), (1, []))

    def test_missing_or_empty_build_does_not_pass(self):
        for root in (self.root, self.root / 'missing'):
            with self.subTest(root=root), self.assertRaises(ValueError):
                check_links(root)


if __name__ == '__main__':
    unittest.main()
