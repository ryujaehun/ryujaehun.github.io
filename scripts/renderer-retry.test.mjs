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

test('the real KaTeX gate rejects unsupported commands despite permissive page options', async () => {
  const { spawn } = await import('node:child_process');
  const { fileURLToPath } = await import('node:url');
  const checker = fileURLToPath(new URL('./check-math.mjs', import.meta.url));
  const wrapper = fileURLToPath(new URL('./renderer-retry.mjs', import.meta.url));
  const fixture = fileURLToPath(new URL('./fixtures/renderer-invalid-math', import.meta.url));
  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [wrapper, checker, fixture]);
    let stderr = '';
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.stdout.resume();
    child.once('error', reject);
    child.once('exit', code => resolve({code, stderr}));
  });
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Undefined control sequence/);
});
