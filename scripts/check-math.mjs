// Requires Chrome and Node 22+; checks the renderer and code sources in the actual Hugo output.
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { rendererLoadExitCode } from './renderer-retry.mjs';
// Render all math and check that copyable code survives minification and math rendering.
// Usage: node scripts/check-math.mjs public
const directory = resolve(process.argv[2] || 'public');
const pages = [];
async function collect(folder) {
  for (const item of await readdir(folder, { withFileTypes: true })) {
    const path = join(folder, item.name);
    if (item.isDirectory()) await collect(path);
    else if (item.isFile() && item.name.endsWith('.html')) {
      const html = await readFile(path, 'utf8');
      if (html.includes('renderMathInElement(document.body,') || html.includes('data-code-source')) {
        pages.push({ path: relative(directory, path), html });
      }
    }
  }
}
await collect(directory);
if (!pages.length) {
  console.log('Checked 0 formulas and 0 code sources');
  process.exit(0);
}
const rendererUrls = html => [...html.matchAll(/<script\b[^>]*\bsrc=["']?([^"'\s>]+)/gi)]
  .map(m => m[1]).filter(url => /\/(?:katex|auto-render)\.min\.js$/.test(url));
const mathPages = pages.filter(p => p.html.includes('renderMathInElement(document.body,'));
const urls = mathPages.length ? rendererUrls(mathPages[0].html) : [];
assert(!mathPages.length || urls.length === 2, 'Expected the KaTeX and auto-render URLs in the build');
for (const page of mathPages) assert.deepEqual(rendererUrls(page.html), urls, `Different KaTeX renderer: ${page.path}`);
// Read the actual published local assets. Never silently substitute another renderer
// if an asset is missing. CDN fixtures remain supported by the existing fetch gate.
const renderers = await Promise.all(urls.map(async url => {
  if (url.startsWith('https://cdn.jsdelivr.net/npm/katex@')) return {url};
  const path = new URL(url, 'https://build.invalid').pathname;
  const asset = path.match(/\/vendor\/katex@[0-9.]+\/(?:contrib\/)?(?:katex|auto-render)\.min\.js$/)?.[0];
  assert(asset, `Unexpected local KaTeX asset: ${url}`);
  return {url, code:await readFile(join(directory, asset.slice(1)), 'utf8')};
}));
const profile = await mkdtemp(join(tmpdir(), 'blog-math-check-'));
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let socket;
let loadingRenderer = false;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    const timeout = setTimeout(() => reject(new Error('Chrome startup timed out')), 20000);
    chrome.once('error', error => { clearTimeout(timeout); reject(error); });
    chrome.stderr.on('data', chunk => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
    chrome.once('exit', code => { clearTimeout(timeout); reject(new Error(`Chrome exited: ${code}`)); });
  });
  const port = new URL(endpoint).port;
  const target = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' }).then(r => r.json());
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let sequence = 0;
  const pending = new Map();
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    const request = pending.get(message.id);
    if (!request) return;
    clearTimeout(request.timeout);
    pending.delete(message.id);
    if (message.error) request.reject(new Error(JSON.stringify(message.error)));
    else request.resolve(message.result);
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
    pending.set(id, { resolve, reject, timeout });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await send('Page.enable');
  const {frameTree}=await send('Page.getFrameTree');
  await send('Page.setDocumentContent',{frameId:frameTree.frame.id,html:'<!doctype html><html><head></head><body></body></html>'});
  assert.equal(await evaluate('document.compatMode'),'CSS1Compat');
  loadingRenderer = true;
  await evaluate(`(async () => {
    for (const renderer of ${JSON.stringify(renderers)}) await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      if (renderer.code !== undefined) {
        script.textContent = renderer.code;
        document.head.append(script);
        resolve();
      } else {
        script.src = renderer.url;
        script.onload = resolve;
        script.onerror = () => reject(new Error('Failed to load ' + renderer.url));
        document.head.append(script);
      }
    });
  })()`);
  loadingRenderer = false;
  let total = 0, codeSources = 0;
  const failures = [];
  for (const page of pages) {
    const options = page.html.match(/renderMathInElement\(document\.body,\s*(\{[\s\S]*?\})\)/)?.[1];
    assert(!page.html.includes('renderMathInElement(document.body,') || options, `Missing KaTeX options: ${page.path}`);
    const result = await evaluate(`(() => {
      const doc = new DOMParser().parseFromString(${JSON.stringify(page.html)}, 'text/html');
      const root = document.createElement('div');
      root.append(...doc.body.childNodes);
      document.body.append(root);
      const sources = [...root.querySelectorAll('[data-code-source]')];
      const originals = sources.map(e => e.textContent);
      const codeErrors = [];
      // Chroma may append one terminal newline; internal and leading whitespace must match.
      const normalize = text => text.replace(/\\n$/, '');
      sources.forEach((source, index) => {
        const block = source.closest('[data-code-block]');
        const highlighted = block?.querySelector('.highlight code[data-lang]') || block?.querySelector('.highlight code');
        if (!highlighted || normalize(highlighted.textContent) !== normalize(originals[index])) {
          codeErrors.push({index, error:'Copy source differs from highlighted code'});
        }
      });
      const options = ${options || 'null'};
      const callbacks = [];
      if (options) {
        // Unsupported commands otherwise become red text without a .katex-error node.
        options.throwOnError = true;
        options.errorCallback = (message, error) => callbacks.push(String(error));
        renderMathInElement(root, options);
      }
      sources.forEach((source, index) => {
        if (source.textContent !== originals[index] || source.querySelector('.katex,.katex-error')) {
          codeErrors.push({index, error:'Math rendering changed copy source'});
        }
      });
      const result = {
        count:root.querySelectorAll('.katex').length,
        errors:[...root.querySelectorAll('.katex-error')].map(e => ({tex:e.textContent, error:e.title})),
        callbacks, codeErrors, codeSources:sources.length,
      };
      root.remove();
      return result;
    })()`);
    total += result.count;
    codeSources += result.codeSources;
    if (result.errors.length || result.callbacks.length || result.codeErrors.length) failures.push({path:page.path, ...result});
  }
  for (const failure of failures) console.error(JSON.stringify(failure));
  console.log(`Checked ${total} formulas and ${codeSources} code sources in ${pages.length} HTML files; ${failures.length} failing pages`);
  if (failures.length) process.exitCode = 1;
} catch (error) {
  console.error(error);
  process.exitCode = rendererLoadExitCode(error, loadingRenderer);
} finally {
  socket?.close();
  if (chrome.pid && chrome.exitCode === null && chrome.signalCode === null) {
    await new Promise(resolve => {
      const timeout = setTimeout(() => { chrome.kill('SIGKILL'); resolve(); }, 3000);
      chrome.once('exit', () => { clearTimeout(timeout); resolve(); });
      chrome.kill();
    });
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
