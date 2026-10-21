import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readingPaths } from './reading-paths.mjs';

test('a new Korean post is checked while its translation is pending', () => {
  const index = [1, 2, 3, 4, 5].map(n => ({ section: 'posts', permalink: `/posts/${n}/`, categories: [] }));
  const english = [{ permalink: '/en/posts/2/' }];
  assert.deepEqual(readingPaths(index, english).paths,
    ['/', '/en/', '/posts/1/', '/posts/2/', '/posts/3/', '/posts/4/', '/posts/5/', '/en/posts/2/']);
});

test('existing translations remain subject to the full browser checks', () => {
  const index = [1, 2, 3, 4, 5].map(n => ({ section: 'posts', permalink: `/posts/${n}/`, categories: [] }));
  const english = index.map(p => ({ permalink: `/en${p.permalink}` }));
  assert.equal(readingPaths(index, english).paths.length, 12);
});
