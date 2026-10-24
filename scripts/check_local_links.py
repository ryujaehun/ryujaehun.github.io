#!/usr/bin/env python3
"""Check rendered links and image sources against a Hugo build, without network I/O."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        key = {'a': 'href', 'img': 'src'}.get(tag)
        value = dict(attrs).get(key)
        if value:
            self.links.append((tag, value))


def check_links(root, base_url='https://jaehun.me/'):
    root = Path(root).resolve()
    pages = sorted(root.rglob('*.html')) if root.is_dir() else []
    if not pages:
        raise ValueError(f'No rendered HTML files in {root}')
    base_url = base_url.rstrip('/') + '/'
    origin = urlsplit(base_url)
    prefix = origin.path.rstrip('/') + '/'
    problems = []
    for page in pages:
        relative = page.relative_to(root).as_posix()
        parser = Links()
        parser.feed(page.read_text(encoding='utf-8'))
        for tag, raw in parser.links:
            try:
                url = urlsplit(urljoin(base_url + relative, raw))
            except ValueError:
                problems.append((relative, tag, raw))
                continue
            if url.scheme not in ('http', 'https') or url.netloc.lower() != origin.netloc.lower():
                continue
            path = unquote(url.path)
            if not path.startswith(prefix):
                problems.append((relative, tag, raw))
                continue
            target = (root / path[len(prefix):]).resolve()
            exists = target.is_relative_to(root) and (
                target.is_file() or
                (tag == 'a' and (target / 'index.html').is_file())
            )
            if not exists:
                problems.append((relative, tag, raw))
    return len(pages), problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_dir', type=Path)
    parser.add_argument('--base-url', default='https://jaehun.me/')
    args = parser.parse_args()
    try:
        count, problems = check_links(args.build_dir, args.base_url)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    for page, tag, url in problems:
        print(f'{page}: broken {tag}: {url}')
    print(f'Checked {count} HTML files; {len(problems)} broken local links/images')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
