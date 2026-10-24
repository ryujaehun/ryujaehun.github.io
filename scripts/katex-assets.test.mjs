import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const root = new URL('../static/vendor/katex@0.16.22/', import.meta.url);

test('vendored KaTeX files match the verified upstream manifest', async () => {
  const manifest = await readFile(new URL('SHA256SUMS', root), 'utf8');
  const entries = manifest.trim().split('\n');
  assert.equal(entries.length, 64);
  for (const entry of entries) {
    const [expected, name] = entry.split('  ');
    const actual = createHash('sha256').update(await readFile(new URL(name, root))).digest('hex');
    assert.equal(actual, expected, name);
  }
});

test('every font referenced by the unmodified KaTeX CSS is shipped', async () => {
  const css = await readFile(new URL('katex.min.css', root), 'utf8');
  const fonts = [...css.matchAll(/url\(["']?([^"')]+)["']?\)/g)].map(m => m[1]);
  assert.equal(fonts.length, 60);
  for (const font of fonts) {
    assert(font.startsWith('fonts/'), font);
    await access(new URL(font, root));
  }
});
