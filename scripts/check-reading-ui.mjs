// Dependency-free smoke checks against a running Hugo build. Requires Chrome and Node 22+.
// Usage: node scripts/check-reading-ui.mjs http://127.0.0.1:8765 [screenshot-directory]
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';

const base = process.argv[2] || 'http://127.0.0.1:8765';
const screenshots = process.argv[3];
const profile = await mkdtemp(join(tmpdir(), 'blog-reading-check-'));
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
  const index = await fetch(new URL('/index.json', base)).then(r => { assert(r.ok); return r.json(); });
  const recent = index.filter(p => p.section === 'posts').slice(0, 5).map(p => p.permalink);
  assert.equal(recent.length, 5);
  const series = index.find(p => p.categories.includes('code-series'))?.permalink;
  const paths = ['/', '/en/', ...recent, ...recent.map(p => `/en${p}`), ...(series ? [series, `/en${series}`] : [])];
  if (screenshots) await mkdir(screenshots, { recursive: true });
  for (const width of [390, 1440]) {
    await send('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: width < 600 });
    for (const path of paths) {
      const url = new URL(path, base).href;
      await send('Page.navigate', { url });
      let ready = false;
      for (let i = 0; i < 150; i++) {
        ready = await evaluate(`location.href === ${JSON.stringify(url)} && document.readyState === 'complete'`);
        if (ready) break;
        await pause(200);
      }
      assert(ready, `Page load timed out: ${path}`);
      await evaluate('document.fonts.ready.then(() => true)');
      const result = await evaluate(`(() => {
        const article = document.querySelector('main article.prose');
        const prose = article?.cloneNode(true);
        prose?.querySelectorAll('pre, code, script, style').forEach(e => e.remove());
        const home = document.querySelector('.home-intro');
        const firstCard = document.querySelector('main article.group');
        return {
          h1: document.querySelectorAll('main h1').length,
          overflow: document.documentElement.scrollWidth > innerWidth + 1,
          mathErrors: [...document.querySelectorAll('.katex-error')].map(e => e.title),
          renderedMath: document.querySelectorAll('.katex').length,
          expectsMath: !!article && /\\$[^$]+\\$/.test(prose.textContent),
          malformed: article ? [...article.querySelectorAll('*')].filter(e => e.namespaceURI === 'http://www.w3.org/1999/xhtml' && !/^[a-z][a-z0-9-]*$/.test(e.localName)).map(e => e.localName) : [],
          leftovers: !!article && /densegray|Draftsfor|operatorname_/.test(article.textContent),
          firstCardTop: home && firstCard ? firstCard.getBoundingClientRect().top : null,
          taxonomiesCollapsed: !article || !document.querySelector('.post-taxonomies')?.open,
          appendixCollapsed: !document.querySelector('.paper-tables')?.open,
        };
      })()`);
      assert.equal(result.h1, 1, `${path}: one main heading`);
      assert(!result.overflow, `${path} at ${width}px: horizontal page overflow`);
      assert.deepEqual(result.mathErrors, [], `${path}: KaTeX errors`);
      assert.deepEqual(result.malformed, [], `${path}: malformed HTML elements`);
      assert(!result.leftovers, `${path}: conversion artifacts`);
      assert(!result.expectsMath || result.renderedMath > 0, `${path}: math renderer did not run`);
      assert(result.taxonomiesCollapsed && result.appendixCollapsed, `${path}: details should start collapsed`);
      if (path === series || path === `/en${series}`) {
        assert(await evaluate("!!document.querySelector('#series-container a[aria-current=page]')"), `${path}: current series episode`);
        assert(await evaluate("document.querySelector('#series-link-btn')?.getAttribute('href') !== '#'"), `${path}: series navigation link`);
      }
      if (result.firstCardTop !== null) assert(result.firstCardTop < 844, `${path}: recent posts below first screen`);
      // Native details must work from the keyboard, including tables rendered while hidden.
      if (await evaluate("!!document.querySelector('.paper-tables')")) {
        await evaluate("document.querySelector('.paper-tables > summary').focus()");
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' });
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
        await pause(100);
        assert(await evaluate("document.querySelector('.paper-tables').open"), `${path}: keyboard expansion`);
        await evaluate("document.querySelector('.paper-tables').open = false; window.scrollTo(0, 0)");
      }
      await pause(100);
      if (screenshots && (path === '/' || path === recent[3])) {
        const capture = await send('Page.captureScreenshot', { format: 'png' });
        await writeFile(join(screenshots, `${path === '/' ? 'home' : 'post'}-${width}.png`), Buffer.from(capture.data, 'base64'));
      }
      console.log(`PASS ${width}px ${path} (${result.renderedMath} formulas${result.firstCardTop === null ? '' : `, first card y=${Math.round(result.firstCardTop)}`})`);
    }
  }
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
