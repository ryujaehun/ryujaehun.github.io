// Requires Chrome and Node 22+; uses the same CDN module as the built site.
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { rendererLoadExitCode } from './renderer-retry.mjs';
// Render every Mermaid block in the actual Hugo output, including older posts.
// Usage: node scripts/check-mermaid.mjs public
const directory = resolve(process.argv[2] || 'public');
const pages = [];
async function collect(folder) {
  for (const item of await readdir(folder, { withFileTypes: true })) {
    const path = join(folder, item.name);
    if (item.isDirectory()) await collect(path);
    else if (item.isFile() && item.name.endsWith('.html')) {
      const html = await readFile(path, 'utf8');
      if (/<pre\b[^>]*\bclass=(?:"[^"]*\bmermaid\b[^"]*"|'[^']*\bmermaid\b[^']*'|mermaid(?=[\s>]))/i.test(html)) {
        pages.push({ path: relative(directory, path), html });
      }
    }
  }
}
await collect(directory);
if (!pages.length) {
  console.log('Checked 0 Mermaid diagrams');
  process.exit(0);
}
// Use the renderer shipped by the build, so changes to the theme version apply here too.
const renderers = new Set(pages.map(({ html }) => html.match(/https:\/\/cdn\.jsdelivr\.net\/npm\/mermaid@[^"'\s<>]+\/mermaid\.esm\.min\.mjs/)?.[0]));
assert(renderers.size === 1 && !renderers.has(undefined), 'Expected one Mermaid renderer URL in the build');
const [renderer] = renderers;
const profile = await mkdtemp(join(tmpdir(), 'blog-mermaid-check-'));
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
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
  loadingRenderer = true;
  await evaluate(`(async () => {
    const {default: mermaid} = await import(${JSON.stringify(renderer)});
    mermaid.initialize({startOnLoad:false,securityLevel:'loose',theme:'default'});
    window.auditMermaid = mermaid;
  })()`);
  loadingRenderer = false;
  let total = 0;
  const failures = [];
  for (const page of pages) {
    const results = await evaluate(`(async () => {
      const doc = new DOMParser().parseFromString(${JSON.stringify(page.html)}, 'text/html');
      const results = [];
      for (const [index, block] of [...doc.querySelectorAll('pre.mermaid')].entries()) {
        const container = document.createElement('div');
        document.body.append(container);
        try {
          const {svg} = await window.auditMermaid.render('audit' + ${total} + '_' + index, block.textContent, container);
          results.push({index, ok:svg.includes('<svg')});
        } catch (error) {
          results.push({index, ok:false, error:String(error)});
        } finally {
          container.remove();
        }
      }
      return results;
    })()`);
    assert(results.length, `No Mermaid blocks extracted from ${page.path}`);
    total += results.length;
    failures.push(...results.filter(r => !r.ok).map(r => ({path:page.path, ...r})));
  }
  for (const failure of failures) console.error(`${failure.path} diagram ${failure.index + 1}: ${failure.error || 'No SVG rendered'}`);
  console.log(`Checked ${total} Mermaid diagrams in ${pages.length} HTML files; ${failures.length} failures`);
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
