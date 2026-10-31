#!/usr/bin/env python3
"""Validate rendered search metadata, sitemap targets and full-content RSS offline."""
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET


class Head(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canonicals = []
        self.meta = {}
        self.alternates = []
        self.scripts = []
        self.script = None
        self.refresh = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[attrs.get('name', attrs.get('property', ''))] = attrs.get('content', '')
            self.refresh |= attrs.get('http-equiv', '').lower() == 'refresh'
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonicals.append(attrs.get('href', ''))
        if tag == 'link' and 'hreflang' in attrs:
            self.alternates.append(attrs)
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.script = ''

    def handle_data(self, data):
        if self.script is not None:
            self.script += data

    def handle_endtag(self, tag):
        if tag == 'script' and self.script is not None:
            self.scripts.append(self.script)
            self.script = None


class BodyText(HTMLParser):
    def __init__(self, article_only=False):
        super().__init__()
        self.article_only = article_only
        self.inside = not article_only
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'article' and 'prose' in dict(attrs).get('class', '').split():
            self.inside = True
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag == 'article' and self.article_only:
            self.inside = False
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if self.inside and not self.hidden:
            self.parts.append(data)

    def text(self):
        return re.sub(r'\s+', '', ''.join(self.parts))


def check(root, base_url):
    root = Path(root)
    base_url = base_url.rstrip('/') + '/'
    errors = []
    documents = {}
    count = 0
    for path in sorted(root.rglob('*.html')):
        relative = path.relative_to(root).as_posix()
        parser = Head()
        parser.feed(path.read_text())
        if parser.refresh or relative.startswith('naver'):
            continue  # Hugo aliases and the Naver ownership verification file.
        count += 1
        route = relative[:-10] if relative.endswith('index.html') else relative
        expected = urljoin(base_url, quote(route, safe='/'))
        documents[unquote(expected)] = parser
        if len(parser.canonicals) != 1 or unquote(parser.canonicals[0]) != unquote(expected):
            errors.append(f'{relative}: canonical must identify this page ({expected})')
        if unquote(parser.meta.get('og:url', '')) != unquote(expected):
            errors.append(f'{relative}: og:url does not identify this page')
        if not parser.meta.get('description', '').strip():
            errors.append(f'{relative}: missing description')
        for key in ('og:image', 'twitter:image'):
            image = parser.meta.get(key)
            if not image:
                errors.append(f'{relative}: missing {key}')
            elif urlsplit(image).netloc == urlsplit(base_url).netloc:
                prefix = urlsplit(base_url).path
                image_path = unquote(urlsplit(image).path)
                if not image_path.startswith(prefix) or not (root / image_path[len(prefix):]).is_file():
                    errors.append(f'{relative}: {key} points to a missing local file: {image}')
        try:
            graphs = [json.loads(s) for s in parser.scripts]
            nodes = [node for graph in graphs for node in graph.get('@graph', [])]
            if not nodes:
                errors.append(f'{relative}: missing JSON-LD graph')
            for node in nodes:
                if node.get('@type') == 'CollectionPage' and unquote(node.get('url', '')) != unquote(expected):
                    errors.append(f'{relative}: CollectionPage URL does not identify this page')
        except (ValueError, AttributeError, TypeError):
            errors.append(f'{relative}: invalid JSON-LD')
    if not count:
        errors.append('No rendered pages found')
    sitemap_count = 0
    for path in root.rglob('sitemap.xml'):
        tree = ET.parse(path)
        for loc in tree.findall('{*}url/{*}loc'):
            sitemap_count += 1
            doc = documents.get(unquote(loc.text or ''))
            if doc is None:
                errors.append(f'{path.relative_to(root)}: missing sitemap target {loc.text}')
            elif {'noindex', 'none'} & set(doc.meta.get('robots', '').lower().replace(',', ' ').split()):
                errors.append(f'{path.relative_to(root)}: noindex sitemap target {loc.text}')
    if not sitemap_count:
        errors.append('No sitemap URLs found')
    for language in ('', 'en/'):
        feed = root / language / 'index.xml'
        if not feed.exists():
            errors.append(f'Missing RSS: {language}index.xml')
            continue
        if feed.stat().st_size >= 10 * 1024 * 1024:
            errors.append(f'{language}index.xml: exceeds Naver 10 MB feed limit')
        items = ET.parse(feed).findall('channel/item')
        if not 1 <= len(items) <= 30:
            errors.append(f'{language}index.xml: expected 1–30 recent posts')
        for item in items:
            description = item.findtext('description', '')
            link = item.findtext('link', '')
            origin = urlsplit(base_url)
            target = urlsplit(link)
            if target.netloc != origin.netloc or not target.path.startswith(origin.path):
                errors.append(f'{language}index.xml: unexpected item URL {link}')
                continue
            article = root / unquote(target.path[len(origin.path):]) / 'index.html'
            if not article.is_file():
                errors.append(f'{language}index.xml: missing item target {link}')
                continue
            actual = BodyText()
            actual.feed(description)
            expected_body = BodyText(article_only=True)
            expected_body.feed(article.read_text())
            if not expected_body.text() or actual.text() != expected_body.text():
                errors.append(f'{language}index.xml: full HTML body missing or differs for {link}')
    return count, sitemap_count, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_dir', type=Path)
    parser.add_argument('--base-url', default='https://jaehun.me/')
    args = parser.parse_args()
    try:
        pages, urls, errors = check(args.build_dir, args.base_url)
    except (OSError, ValueError, ET.ParseError) as exc:
        parser.exit(1, f'{exc}\n')
    for error in errors[:30]:
        print(error)
    print(f'Checked {pages} pages and {urls} sitemap URLs; {len(errors)} SEO errors')
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
