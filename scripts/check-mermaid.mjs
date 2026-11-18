// Requires Chrome and Node 22+; checks the renderer shipped by the site.
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import assert from 'node:assert/strict';
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
const rendererPaths = new Set(pages.map(({ html }) => html.match(/<script\b[^>]*\bsrc=["']?(\/(?!\/)[^"'\s>]*\/mermaid@[\d.]+\/mermaid\.min\.js)(?=["'\s>])/i)?.[1]));
assert(rendererPaths.size === 1 && !rendererPaths.has(undefined), 'Expected one local Mermaid renderer in the build');
const [rendererPath] = rendererPaths;
const asset = rendererPath.match(/\/vendor\/mermaid@[\d.]+\/mermaid\.min\.js$/)?.[0];
assert(asset, `Unexpected Mermaid asset: ${rendererPath}`);
const bundle = await readFile(join(directory, asset.slice(1)));
const preview = createServer((request, response) => {
  if (request.url === rendererPath) {
    response.setHeader('Content-Type', 'text/javascript; charset=utf-8');
    response.end(bundle);
  } else if (request.url === '/mermaid-preview.html') {
    response.setHeader('Content-Type', 'text/html; charset=utf-8');
    response.end(pages[0].html);
  } else {
    response.writeHead(404).end();
  }
});
await new Promise((resolve, reject) => {
  preview.once('error', reject);
  preview.listen(0, '127.0.0.1', resolve);
});
const profile = await mkdtemp(join(tmpdir(), 'blog-mermaid-check-'));
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
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
  await send('Network.enable');
  await send('Network.setBlockedURLs', { urls: ['*://cdn.jsdelivr.net/*'] });
  const url = `http://127.0.0.1:${preview.address().port}/mermaid-preview.html`;
  await send('Page.navigate', { url });
  const waitFor = async (expression, message) => {
    for (let attempt = 0; attempt < 100; attempt++) {
      if (await evaluate(expression)) return;
      await pause(100);
    }
    assert.fail(message);
  };
  await waitFor(`location.href === ${JSON.stringify(url)} && !!document.querySelector('pre.mermaid svg')`, 'Mermaid must render on the page with the CDN blocked');
  const expectedTheme = await evaluate(`(() => {
    window.firstDiagram = document.querySelector('pre.mermaid svg');
    const dark = document.documentElement.classList.toggle('dark');
    window.dispatchEvent(new CustomEvent('themeChanged'));
    return dark ? 'dark' : 'default';
  })()`);
  await waitFor(`(() => {
    const svg = document.querySelector('pre.mermaid svg');
    return !!svg && svg !== window.firstDiagram && mermaid.mermaidAPI.getConfig().theme === ${JSON.stringify(expectedTheme)};
  })()`, 'Mermaid must render again after the theme changes');
  await evaluate(`mermaid.initialize({startOnLoad:false,securityLevel:'loose',theme:'default'})`);
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
          const {svg} = await mermaid.render('audit' + ${total} + '_' + index, block.textContent, container);
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
  process.exitCode = 1;
} finally {
  socket?.close();
  if (chrome.pid && chrome.exitCode === null && chrome.signalCode === null) {
    await new Promise(resolve => {
      const timeout = setTimeout(() => { chrome.kill('SIGKILL'); resolve(); }, 3000);
      chrome.once('exit', () => { clearTimeout(timeout); resolve(); });
      chrome.kill();
    });
  }
  await new Promise(resolve => preview.close(resolve));
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
