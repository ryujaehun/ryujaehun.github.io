import tempfile
import unittest
from pathlib import Path

from check_seo import check


class SearchMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'avatar.webp').write_bytes(b'image')
        self.page('posts/example/', '<h2>Heading</h2><p>Opening.</p><p>Final finding.</p>')
        self.page('posts/page/2/', '')
        (self.root / 'sitemap.xml').write_text('''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.com/posts/example/</loc></url></urlset>''')
        for lang in ('', 'en/'):
            path = self.root / lang / 'index.xml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('''<rss><channel><item><link>https://example.com/posts/example/</link><description>&lt;h2&gt;Heading&lt;/h2&gt;&lt;p&gt;Opening.&lt;/p&gt;&lt;p&gt;Final finding.&lt;/p&gt;</description></item></channel></rss>''')

    def page(self, route, body, robots='index, follow'):
        path = self.root / route / 'index.html'
        path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://example.com/' + route
        path.write_text(f'''<link rel="canonical" href="{url}"><meta property="og:url" content="{url}"><meta name="description" content="Example"><meta name="robots" content="{robots}"><meta property="og:image" content="https://example.com/avatar.webp"><meta name="twitter:image" content="https://example.com/avatar.webp"><script type="application/ld+json">{{"@graph":[{{"@type":"CollectionPage","url":"{url}"}}]}}</script><article class="prose">{body}</article>''')
        return path

    def errors(self):
        return check(self.root, 'https://example.com/')[2]

    def test_valid_build(self):
        self.assertEqual([], self.errors())

    def test_later_page_canonical_must_not_point_to_page_one(self):
        path = self.root / 'posts/page/2/index.html'
        path.write_text(path.read_text().replace('https://example.com/posts/page/2/', 'https://example.com/posts/'))
        errors = self.errors()
        self.assertTrue(any('canonical' in error for error in errors))
        self.assertTrue(any('og:url' in error for error in errors))
        self.assertTrue(any('CollectionPage' in error for error in errors))

    def test_full_body_required_even_when_summary_contains_html(self):
        path = self.root / 'index.xml'
        path.write_text(path.read_text().replace('&lt;p&gt;Final finding.&lt;/p&gt;', ''))
        self.assertTrue(any('full HTML body' in error for error in self.errors()))

    def test_noindex_and_none_must_not_be_in_sitemap(self):
        for directive in ('NOINDEX, follow', 'none'):
            with self.subTest(directive=directive):
                self.page('posts/example/', '<p>body</p>', robots=directive)
                self.assertTrue(any('noindex sitemap' in error for error in self.errors()))

    def test_missing_share_image_is_detected(self):
        (self.root / 'avatar.webp').unlink()
        self.assertTrue(any('missing local file' in error for error in self.errors()))

    def test_empty_build_fails(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertIn('No rendered pages found', check(empty, 'https://example.com/')[2])


if __name__ == '__main__':
    unittest.main()
