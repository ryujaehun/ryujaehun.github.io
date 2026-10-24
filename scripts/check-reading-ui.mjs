// Dependency-free smoke checks against a running Hugo build. Requires Chrome and Node 22+.
// Usage: node scripts/check-reading-ui.mjs http://127.0.0.1:8765 [screenshot-directory]
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import { readingPaths } from './reading-paths.mjs';

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
  await send('Network.enable');
  // Production math must keep rendering even when the former CDN is unavailable.
  await send('Network.setBlockedURLs', { urls: ['*://cdn.jsdelivr.net/npm/katex@*'] });
  const index = await fetch(new URL('/index.json', base)).then(r => { assert(r.ok); return r.json(); });
  const englishIndex = await fetch(new URL('/en/index.json', base)).then(r => { assert(r.ok); return r.json(); });
  const { recent, series, paths } = readingPaths(index, englishIndex);
  assert.equal(recent.length, 5);
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
          articleWidth: article?.getBoundingClientRect().width,
          dockMode: document.getElementById('dock')?.dataset.dockMode,
          dockInitiallyHidden: !document.getElementById('dock') || getComputedStyle(document.getElementById('dock')).visibility === 'hidden',
          homeDock: !!home && !!document.getElementById('dock'),
          cardTagCounts: [...document.querySelectorAll('.post-card-tags')].map(e => e.children.length),
          overflow: document.documentElement.scrollWidth > innerWidth + 1,
          mathErrors: [...document.querySelectorAll('.katex-error')].map(e => e.title),
          renderedMath: document.querySelectorAll('.katex').length,
          mathAssets: [...document.querySelectorAll('script[src],link[rel=stylesheet]')]
            .map(e => e.src || e.href).filter(url => /katex@/.test(url)),
          mathFontsLoaded: [...document.fonts].filter(f => f.family.startsWith('KaTeX') && f.status === 'loaded').length,
          failedMathFonts: [...document.fonts].filter(f => f.family.startsWith('KaTeX') && f.status === 'error').map(f => f.family),
          description: document.querySelector('meta[name="description"]')?.content,
          brokenImageMarkup: !!prose && prose.textContent.includes('](/assets/images/') && prose.textContent.includes('!'),
          brokenLocalImages: article ? [...article.querySelectorAll('img')].filter(img =>
            new URL(img.src).origin === location.origin && img.complete && img.naturalWidth === 0
          ).map(img => img.getAttribute('src')) : [],
          expectsMath: !!article && /\\$[^$]+\\$/.test(prose.textContent),
          malformed: article ? [...article.querySelectorAll('*')].filter(e => e.namespaceURI === 'http://www.w3.org/1999/xhtml' && !/^[a-z][a-z0-9-]*$/.test(e.localName)).map(e => e.localName) : [],
          leftovers: !!article && /densegray|Draftsfor|operatorname_/.test(article.textContent),
          firstCardTop: home && firstCard ? firstCard.getBoundingClientRect().top : null,
          taxonomiesCollapsed: !article || !document.querySelector('.post-taxonomies')?.open,
          appendixCollapsed: !document.querySelector('.paper-tables')?.open,
        };
      })()`);
      assert(result.dockInitiallyHidden, `${path}: dock should not obscure initial content`);
      if (result.articleWidth) {
        assert(result.articleWidth <= 737, `${path}: readable default article width`);
        assert.equal(result.dockMode, 'scroll', `${path}: dock appears only while scrolling up`);
      }
      assert(!result.homeDock, `${path}: home should not have an overlapping dock`);
      assert(result.cardTagCounts.every(n => n <= 2), `${path}: at most two representative tags per card`);
      assert.equal(result.h1, 1, `${path}: one main heading`);
      assert(!result.overflow, `${path} at ${width}px: horizontal page overflow`);
      assert.deepEqual(result.mathErrors, [], `${path}: KaTeX errors`);
      assert(result.mathAssets.every(url => new URL(url).origin === new URL(base).origin), `${path}: KaTeX assets must be local`);
      assert.deepEqual(result.failedMathFonts, [], `${path}: failed KaTeX fonts`);
      if (result.renderedMath) assert(result.mathFontsLoaded > 0, `${path}: local math fonts did not load`);
      assert(!result.brokenImageMarkup, `${path}: image Markdown displayed as text`);
      assert.deepEqual(result.brokenLocalImages, [], `${path}: broken local images`);
      const record = [...index, ...englishIndex].find(p => p.permalink === path);
      if (record) assert.equal(record.summary, result.description, `${path}: search and page descriptions agree`);
      assert.deepEqual(result.malformed, [], `${path}: malformed HTML elements`);
      assert(!result.leftovers, `${path}: conversion artifacts`);
      assert(!result.expectsMath || result.renderedMath > 0, `${path}: math renderer did not run`);
      assert(result.taxonomiesCollapsed && result.appendixCollapsed, `${path}: details should start collapsed`);
      if (path === series || path === `/en${series}`) {
        assert(await evaluate("!!document.querySelector('#series-container a[aria-current=page]')"), `${path}: current series episode`);
        assert(await evaluate("document.querySelector('#series-link-btn')?.getAttribute('href') !== '#'"), `${path}: series navigation link`);
      }
      if (result.firstCardTop !== null) assert(result.firstCardTop < 844, `${path}: recent posts below first screen`);
      if (path === '/' || path === '/en/') {
        assert(await evaluate("!!document.querySelector('.home-search')"), `${path}: visible home search entry`);
        await evaluate("document.querySelector('.home-search').click()");
        assert(await evaluate("document.querySelector('#search-modal').getAttribute('aria-hidden') === 'false'"), `${path}: home search opens`);
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
        assert(await evaluate("document.querySelector('#search-modal').getAttribute('aria-hidden') === 'true'"), `${path}: home search closes with Escape`);
      }
      if (path === recent[0] || path === series) {
        await evaluate("window.scrollTo({ top: 650, behavior: 'instant' })");
        await pause(100);
        await evaluate("window.scrollTo({ top: 350, behavior: 'instant' })");
        await pause(350);
        assert(await evaluate("getComputedStyle(document.getElementById('dock')).visibility === 'visible'"), `${path}: tools available when scrolling up`);
        await evaluate("window.scrollTo({ top: 650, behavior: 'instant' })");
        await pause(100);
        assert(await evaluate("getComputedStyle(document.getElementById('dock')).visibility === 'hidden'"), `${path}: tools hide when reading down`);
        await evaluate("window.scrollTo({ top: 0, behavior: 'instant' })");
        await pause(350);
      }
      if (path === recent[0] && width === 1440) {
        const before = await evaluate("document.querySelector('article.prose').getBoundingClientRect().width");
        await evaluate("const range = document.getElementById('content-width-range'); range.value = String(Number(range.value) + 1); range.dispatchEvent(new Event('input', { bubbles: true }))");
        await pause(100);
        const after = await evaluate("document.querySelector('article.prose').getBoundingClientRect().width");
        assert(Math.abs(after - before - 16) < 1, `${path}: one slider step should resize text by one rem`);
        await evaluate("document.getElementById('content-width-reset').click()");
        await pause(100);
        assert.equal(await evaluate("document.querySelector('article.prose').getBoundingClientRect().width"), before, `${path}: reset restores reading width`);
      }
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
      if (screenshots && (path === '/' || path === recent[0] || path === recent[3])) {
        const capture = await send('Page.captureScreenshot', { format: 'png' });
        await writeFile(join(screenshots, `${path === '/' ? 'home' : path === recent[0] ? 'paper' : 'post'}-${width}.png`), Buffer.from(capture.data, 'base64'));
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
