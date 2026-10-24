import test from 'node:test';
import assert from 'node:assert/strict';
import { rendererLoadExitCode, retryRenderer } from './renderer-retry.mjs';

const network = new Error(JSON.stringify({exceptionDetails:{text:'Uncaught (in promise) TypeError: Failed to fetch dynamically imported module: https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs'}}));

test('only renderer fetch errors during initialization are retryable', () => {
  assert.equal(rendererLoadExitCode(network, true), 75);
  assert.equal(rendererLoadExitCode(network, false), 1);
  assert.equal(rendererLoadExitCode(new Error('Failed to load https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js'), true), 75);
  for (const message of ['ParseError: unexpected token', 'CDP timeout: Runtime.evaluate', 'SyntaxError: bad module', 'Failed to load https://example.test/code.js']) {
    assert.equal(rendererLoadExitCode(new Error(message), true), 1);
  }
});

test('successful check runs once', async () => {
  let calls = 0;
  assert.equal(await retryRenderer(async () => { calls++; return 0; }), 0);
  assert.equal(calls, 1);
});

test('temporary fetch failure retries the whole browser check', async () => {
  const codes = [75, 0];
  let calls = 0;
  const waits = [], messages = [];
  const code = await retryRenderer(async () => { calls++; return codes.shift(); }, {
    sleep: async ms => waits.push(ms), log: message => messages.push(message),
  });
  assert.equal(code, 0);
  assert.equal(calls, 2);
  assert.deepEqual(waits, [1000]);
  assert.equal(messages.length, 1);
});

test('persistent fetch failure stays failed after three attempts', async () => {
  let calls = 0;
  const waits = [];
  const code = await retryRenderer(async () => { calls++; return 75; }, {
    sleep: async ms => waits.push(ms), log: () => {},
  });
  assert.equal(code, 75);
  assert.equal(calls, 3);
  assert.deepEqual(waits, [1000, 2000]);
});

test('content and browser failures are never retried', async () => {
  for (const failure of [1, 2]) {
    let calls = 0;
    assert.equal(await retryRenderer(async () => { calls++; return failure; }), failure);
    assert.equal(calls, 1);
  }
});

async function mathFixture(t) {
  const { mkdtemp, cp, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const fixture = await mkdtemp(join(tmpdir(), 'blog-local-katex-'));
  t.after(() => rm(fixture, {recursive:true, force:true}));
  await cp(new URL('./fixtures/renderer-invalid-math/', import.meta.url), fixture, {recursive:true});
  await cp(new URL('../static/vendor/', import.meta.url), join(fixture,'vendor'), {recursive:true});
  return fixture;
}

async function checkMath(fixture) {
  const { spawn } = await import('node:child_process');
  const { fileURLToPath } = await import('node:url');
  const checker = fileURLToPath(new URL('./check-math.mjs', import.meta.url));
  const wrapper = fileURLToPath(new URL('./renderer-retry.mjs', import.meta.url));
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [wrapper, checker, fixture]);
    let stderr = '', stdout = '';
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.once('error', reject);
    child.once('exit', code => resolve({code, stderr, stdout}));
  });
}

test('the real local KaTeX gate rejects unsupported commands despite permissive page options', async t => {
  const result = await checkMath(await mathFixture(t));
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Undefined control sequence/);
});

test('the real local KaTeX gate renders valid math without a CDN', async t => {
  const { readFile, writeFile } = await import('node:fs/promises');
  const { join } = await import('node:path');
  const fixture = await mathFixture(t);
  const page=join(fixture,'index.html');
  await writeFile(page,(await readFile(page,'utf8')).replace('undefinedCommand','alpha'));
  const result=await checkMath(fixture);
  assert.equal(result.code,0,result.stderr);
  assert.match(result.stdout,/Checked 1 formulas/);
});

test('missing local KaTeX assets fail rather than falling back to a CDN', async t => {
  const { unlink } = await import('node:fs/promises');
  const { join } = await import('node:path');
  const fixture = await mathFixture(t);
  await unlink(join(fixture,'vendor/katex@0.16.22/katex.min.js'));
  const result=await checkMath(fixture);
  assert.equal(result.code,1);
  assert.match(result.stderr,/ENOENT.*katex\.min\.js/s);
});
